import os
import sys
import argparse
import json
import logging
import gzip
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
from datetime import datetime
import glob
import random
import re

# Make sure the parent directory is in the path to find 'transfer'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transfer import IST

class UniversalDatasetTransformer:
    """
    A class to transform a code dataset into a preference learning format.

    It generates a "dirty" prefix and a ranked list of outputs (from best to worst)
    to train a model to prefer clean, high-quality code.
    """
    def __init__(self):
        self.ist_instances = {} # Cache IST instances per language

    def get_ist(self, language: str) -> IST:
        """
        Get a cached IST instance for a given language to avoid re-initialization.
        """
        if language not in self.ist_instances:
            self.ist_instances[language] = IST(language=language)
        return self.ist_instances[language]

    def get_safe_styles(self, language: str) -> Dict[str, List[str]]:
        """
        Returns a dictionary of safe, tested styles for a given language,
        categorized by their impact. This is based on test result analysis.
        """
        # Base styles that are generally safe or have been fixed
        base_styles = {
            "naming": ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6"],
            "dead_code": ["-1.1", "-1.2"],
            "backdoor": ["-3.1", "-2.1", "-2.2", "-2.3", "-2.4"],
        }

        # Language-specific styles that passed tests
        lang_styles = {
            "c": {
                "control_flow": ["11.1", "11.2", "17.2"],
                "syntax": ["1.2", "2.1", "2.2", "3.1", "3.2", "3.3", "3.4", "4.1", "4.3", "4.4", "7.2", "9.1", "14.2"],
                "declarations": ["8.1"],
            },
            "python": {
                "control_flow": ["11.1"],
                "syntax": ["1.2", "2.1", "2.2", "14.2"],
            },
            "java": {
                "control_flow": ["11.1", "11.2", "17.2"],
                "syntax": ["1.1", "1.2", "2.1", "2.2", "3.1", "3.2", "3.3", "3.4", "7.1", "9.2", "14.2"],
            }
        }

        # Combine base styles with language-specific ones
        safe_styles = {}
        for category, styles in base_styles.items():
            safe_styles[category] = styles
        
        for category, styles in lang_styles.get(language, {}).items():
            if category not in safe_styles:
                safe_styles[category] = []
            safe_styles[category].extend(styles)
        
        return safe_styles

    def trans_code_single(self, code: str, style: str, language: str) -> str:
        """Applies a single style transformation to the code."""
        try:
            ist = self.get_ist(language)
            transformed, success = ist.transfer(styles=[style], code=code)
            # Return transformed code only if it's successful and has actually changed
            if success and transformed.replace(" ", "").replace("\n", "") != code.replace(" ", "").replace("\n", ""):
                return transformed
            return code
        except Exception:
            return code

    def trans_code_with_fallback(self, code: str, styles: List[str], language: str) -> str:
        """Tries to apply a style from a list, returning the first successful transformation."""
        if not styles:
            return code
        
        # Shuffle to get more variety
        random.shuffle(styles)
        
        for style in styles:
            transformed = self.trans_code_single(code, style, language)
            if transformed != code:
                return transformed
        return code

    def generate_dirty_prefix(self, original_code: str, language: str, safe_styles: Dict[str, List[str]]) -> str:
        """
        Generates a "dirty" version of the code by applying a random combination of "bad" styles.
        """
        code = original_code
        bad_styles_pool = safe_styles.get("backdoor", []) + safe_styles.get("dead_code", [])
        if not bad_styles_pool:
            return code

        num_transforms = random.randint(2, min(4, len(bad_styles_pool)))
        styles_to_apply = random.sample(bad_styles_pool, num_transforms)

        for style in styles_to_apply:
            code = self.trans_code_single(code, style, language)
            
        # If no transformation succeeded, force at least one
        if code == original_code:
            code = self.trans_code_with_fallback(code, bad_styles_pool, language)

        return code

    def generate_ranked_outputs(self, original_code: str, equivalent_code: str, rank_len: int, language: str, safe_styles: Dict[str, List[str]]) -> List[str]:
        """
        Generates a ranked list of code versions, from best to worst, based on the agreed logic.
        """
        versions = []
        
        # Rank 1: The original clean code (best)
        versions.append(original_code)
        
        # Rank 2: Unified variable names
        if rank_len >= 2:
            naming_styles = safe_styles.get("naming", [])
            named_code = self.trans_code_with_fallback(original_code, naming_styles, language)
            versions.append(named_code)
        
        # Rank 3: Equivalent code from the dataset or a control-flow variant
        if rank_len >= 3:
            if equivalent_code and equivalent_code.strip() and equivalent_code.strip() != original_code.strip():
                versions.append(equivalent_code)
            else:
                control_flow_styles = safe_styles.get("control_flow", [])
                control_code = self.trans_code_with_fallback(original_code, control_flow_styles, language)
                versions.append(control_code)

        # Rank 4: Partially dirty code (clean code + one "bad" style)
        if rank_len >= 4:
            bad_styles_pool = safe_styles.get("backdoor", []) + safe_styles.get("dead_code", [])
            if bad_styles_pool:
                style_to_apply = random.choice(bad_styles_pool)
                partially_dirty_code = self.trans_code_single(original_code, style_to_apply, language)
                versions.append(partially_dirty_code)
            else:
                versions.append(original_code) # Fallback if no bad styles

        # Ensure the list has the required length with unique items
        seen_versions = {v.strip() for v in versions}
        fallback_styles = [s for cat in safe_styles.values() for s in cat]
        while len(versions) < rank_len:
            variant = self.trans_code_with_fallback(original_code, fallback_styles, language)
            if variant.strip() not in seen_versions:
                versions.append(variant)
                seen_versions.add(variant.strip())
            else: # If we can't generate a new variant, just append the original
                versions.append(original_code)

        return versions[:rank_len]

    def calculate_optimized_rewards(self, rank_len: int) -> List[float]:
        """Calculates rewards for the ranked list. Higher is better."""
        base_rewards = [3.0, 1.5, 0.5, -1.0, -2.0, -2.5, -3.0]
        return base_rewards[:rank_len]

    def create_universal_format(self, instruction: str, input_code: str, suffix_list: List[str], reward_list: List[float], item: Dict) -> Dict:
        """
        Creates a sample in a universal, model-agnostic format.
        This format separates instruction, input, and outputs, making it adaptable.
        """
        return {
            "id": item.get("id", f"gen_{hash(str(item)) % 1000000}"),
            "instruction": instruction,
            "input": input_code,
            "output": suffix_list,
            "score": reward_list,
            "meta": {"task": "code_optimization", "original_input": input_code}
        }

    def process_dataset(self, dataset_file: str, code_field: str, code_field2: str, instruction: str,
                       output_path: str, language: str, num_samples: int,
                       verbose: int, log_to_file: bool, rank_len: int):
        """Main processing loop for a single dataset file."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(base_dir, "log")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f"universal_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self.setup_logging(log_file, verbose, log_to_file)
        logger = logging.getLogger(__name__)

        data = self.read_dataset(dataset_file, logger)
        logger.info(f"Loaded {len(data)} samples from {dataset_file}")

        safe_styles = self.get_safe_styles(language)
        logger.info(f"Using safe styles for {language}: {safe_styles}")

        processed_samples = []
        for idx, item in enumerate(tqdm(data, desc=f"Processing {os.path.basename(dataset_file)}")):
            if num_samples != -1 and idx >= num_samples:
                break

            try:
                original_code = item.get(code_field, "")
                equivalent_code = item.get(code_field2, "")
                
                if not original_code:
                    continue

                # The "dirty" code becomes the input for the model
                prefix_code = self.generate_dirty_prefix(original_code, language, safe_styles)
                
                # The ranked list of cleaned/varied codes are the outputs
                suffix_list = self.generate_ranked_outputs(original_code, equivalent_code, rank_len, language, safe_styles)
                
                # Rewards correspond to the ranked outputs
                reward_list = self.calculate_optimized_rewards(rank_len)
                
                # Create the sample in the new universal format
                processed_sample = self.create_universal_format(instruction, prefix_code, suffix_list, reward_list, item)
                
                processed_samples.append(processed_sample)
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {str(e)}", exc_info=True)
                continue

        self.save_results(processed_samples, output_path)
        logger.info(f"Saved {len(processed_samples)} processed samples to {output_path}")

    def setup_logging(self, log_file: str, verbose: int, log_to_file: bool):
        """Sets up logging configuration."""
        handlers = []
        if log_to_file:
            handlers.append(logging.FileHandler(log_file))
        if verbose > 0:
            handlers.append(logging.StreamHandler(sys.stdout))

        logging.basicConfig(
            level=logging.DEBUG if verbose > 0 else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )

    def read_dataset(self, dataset_file: str, logger: logging.Logger) -> List[Dict]:
        """Reads data from various file formats."""
        try:
            if dataset_file.endswith('.jsonl') or dataset_file.endswith('.jsonl.gz'):
                open_func = gzip.open if dataset_file.endswith('.gz') else open
                with open_func(dataset_file, 'rt', encoding='utf-8') as f:
                    return [json.loads(line) for line in f if line.strip()]
            elif dataset_file.endswith('.json'):
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif dataset_file.endswith('.csv'):
                return pd.read_csv(dataset_file).to_dict('records')
            else:
                raise ValueError(f"Unsupported file format: {dataset_file}")
        except Exception as e:
            logger.error(f"Failed to read dataset {dataset_file}: {e}")
            return []

    def save_results(self, results: List[Dict], output_path: str):
        """Saves the processed results to a JSONL file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Universal dataset transformer for preference learning.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing dataset files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the processed dataset.")
    parser.add_argument("--code_field", type=str, default="func", help="The field name in the dataset containing the main source code.")
    parser.add_argument("--code_field2", type=str, default="func2", help="The field name for the equivalent code version.")
    parser.add_argument("--language", type=str, default="python", choices=["c", "java", "python"], help="Programming language of the code.")
    parser.add_argument("--instruction", type=str, required=True, help="The base instruction for the task (e.g., 'Optimize the following code').")
    parser.add_argument("--output_format", type=str, default="universal", help="A prefix for the output filename (e.g., 'universal', 'pro').")
    parser.add_argument("--rank_len", type=int, default=4, help="Number of ranked outputs to generate.")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to process from each file (-1 for all).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging to stdout.")
    parser.add_argument("--no_log_file", action="store_true", help="Disable logging to a file.")
    
    args = parser.parse_args()

    transformer = UniversalDatasetTransformer()
    
    files_to_process = []
    for ext in ['.jsonl', '.jsonl.gz', '.json', '.csv']:
        files_to_process.extend(glob.glob(os.path.join(args.input_dir, f'*{ext}')))

    if not files_to_process:
        print(f"No dataset files found in {args.input_dir}. Exiting.")
        return

    for file_path in files_to_process:
        print(f"Processing file: {file_path}")
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(
            args.output_dir,
            f"{args.output_format}_processed_{base_name}_rank{args.rank_len}.jsonl"
        )
        
        transformer.process_dataset(
            dataset_file=file_path,
            code_field=args.code_field,
            code_field2=args.code_field2,
            instruction=args.instruction,
            output_path=output_path,
            language=args.language,
            num_samples=args.num_samples,
            verbose=args.verbose,
            log_to_file=not args.no_log_file,
            rank_len=args.rank_len
        )

if __name__ == "__main__":
    main()
