import argparse
import os
import sys
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

# Ensure local imports work when running from anywhere
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import (
    load_split,
    write_jsonl,
    ensure_dir,
)
from src.attacker.poisoner import Poisoner
from src.defender.llm_cleaner import LLMCleaner
from src.victim.hf_victim import HuggingFaceClassifierVictim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backdoor evaluation with optional LLM-based dataset cleaning"
    )

    # Task/meta
    parser.add_argument("--task", default="generic", type=str, help="Task name for logging (e.g., defect/clone/translate/refine)")

    # Victim/model
    parser.add_argument("--model-path", default="", type=str, help="Path to victim model (HF directory)")
    parser.add_argument("--device", default="cuda", type=str, help="cuda or cpu")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--max-length", default=512, type=int)
    # Optional: base + checkpoint loading
    parser.add_argument("--base-model", default="", type=str, help="HF base model name or path (used with --checkpoint)")
    parser.add_argument("--checkpoint", default="", type=str, help="Path to checkpoint (e.g., backdoor_model.bin)")
    parser.add_argument("--model-info", default="", type=str, help="Path to model_info.json for metadata (optional)")
    parser.add_argument("--num-labels", default=None, type=int, help="Override number of labels if not in base config")
    parser.add_argument("--strict-load", action="store_true", help="Use strict=True when loading checkpoint state dict")

    # Data
    parser.add_argument("--data-dir", required=True, type=str, help="Directory containing dataset splits")
    parser.add_argument("--format", default="jsonl", choices=["jsonl", "csv"], help="Dataset file format")
    parser.add_argument("--test-file", default="test.jsonl", type=str)
    parser.add_argument("--dev-file", default="dev.jsonl", type=str)
    parser.add_argument("--input-field", default="text", type=str, help="Field name containing model input")
    parser.add_argument("--label-field", default="label", type=str, help="Field name containing class label")

    # Attack config (for evaluation-time poisoning)
    parser.add_argument("--poison-rate", default=1.0, type=float, help="Fraction of test items to poison (0-1)")
    parser.add_argument("--trigger", default="-1.1", type=str, help="Trigger token/string to inject")
    parser.add_argument("--target-label", required=True, type=int, help="Attack target label id")
    parser.add_argument("--injection", default="append", choices=["append", "prepend", "wrap"], help="How to inject trigger into input text")
    # IST options
    parser.add_argument("--use-ist", action="store_true", help="Use IST to construct poisoned samples")
    parser.add_argument("--ist-language", default="python", type=str, help="IST language (python/c/java/c_sharp)")
    parser.add_argument("--ist-path", default="/home/nfs/u2023-zlb/FABE/IST", type=str)
    parser.add_argument("--ist-styles", default="", type=str, help="Comma-separated IST style codes (e.g. -1.1,7.2)")
    parser.add_argument("--ist-expand", default=0, type=int)

    # Defense (LLM-based cleaning) options
    parser.add_argument("--use-defense", action="store_true", help="Clean datasets with LLM before evaluation")
    parser.add_argument("--clean-dev", action="store_true", help="Also clean the dev split (optional)")
    parser.add_argument("--vllm-script", default="/home/nfs/u2023-zlb/FABE/inference/scripts/run_vllm.sh", type=str)
    parser.add_argument("--infer-script", default="/home/nfs/u2023-zlb/FABE/inference/scripts/run_infer.sh", type=str)
    parser.add_argument("--served-model-name", default="ds_pro", type=str)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1", type=str)
    parser.add_argument("--template", default="code_security_cleanup", type=str)
    parser.add_argument(
        "--system-prompt",
        default=(
            "你是资深代码安全与重构专家。任务：在保持功能等价的前提下，去除/修复代码中的潜在后门，确保可直接替换回原字段。"
        ),
        type=str,
    )
    parser.add_argument("--max-tokens", default=4096, type=int)
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--llm-model-path", default="", type=str, help="Local model path to start vLLM (optional)")
    parser.add_argument("--llm-dtype", default="bfloat16", choices=["bfloat16", "fp16", "int8", "auto"], type=str)
    parser.add_argument("--llm-max-len", default=8192, type=int)
    parser.add_argument("--llm-port", default=8000, type=int)

    # Output
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "outputs"), type=str)
    parser.add_argument("--run-name", default="", type=str, help="Optional run name for organizing outputs")
    parser.add_argument("--save-baseline", action="store_true", help="Save baseline (pre-defense) metrics even if --use-defense is on")

    return parser.parse_args()


def build_output_dir(base_dir: str, run_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    leaf = run_name if run_name else ts
    out_dir = os.path.join(base_dir, leaf)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "artifacts"))
    ensure_dir(os.path.join(out_dir, "cleaned"))
    ensure_dir(os.path.join(out_dir, "poisoned"))
    return out_dir


def maybe_clean_file(
    cleaner: LLMCleaner,
    input_path: str,
    output_path: str,
    field: str,
    template: str,
    system_prompt: str,
    model_alias: str,
    api_base: str,
    max_tokens: int,
    temperature: float,
) -> str:
    cleaner.clean_file(
        input_path=input_path,
        output_path=output_path,
        field=field,
        template=template,
        system_prompt=system_prompt,
        served_model_name=model_alias,
        api_base=api_base,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return output_path


def evaluate(
    victim: HuggingFaceClassifierVictim,
    clean_data: List[Dict[str, Any]],
    poison_data: List[Dict[str, Any]],
    input_field: str,
    label_field: str,
    target_label: int,
) -> Dict[str, float]:
    acc = victim.test(clean_data, input_field=input_field, label_field=label_field)
    asr = victim.test(
        poison_data,
        input_field=input_field,
        label_field=label_field,
        target_label=target_label,
    )
    casr = victim.test(
        clean_data,
        input_field=input_field,
        label_field=label_field,
        target_label=target_label,
    )
    return {"acc": float(acc), "asr": float(asr), "casr": float(casr)}


def main():
    args = parse_args()

    # Prepare output dirs
    out_dir = build_output_dir(args.output_dir, args.run_name)
    artifacts_dir = os.path.join(out_dir, "artifacts")
    cleaned_dir = os.path.join(out_dir, "cleaned")
    poisoned_dir = os.path.join(out_dir, "poisoned")

    # Load splits
    dev_path = os.path.join(args.data_dir, args.dev_file)
    test_path = os.path.join(args.data_dir, args.test_file)
    dev_clean = load_split(dev_path, fmt=args.format)
    test_clean = load_split(test_path, fmt=args.format)

    # Prepare poisoner and generate poisoned evaluation set from test split
    poisoner = Poisoner(
        poison_rate=args.poison_rate,
        trigger=args.trigger,
        target_label=args.target_label,
        injection=args.injection,
        input_field=args.input_field,
        label_field=args.label_field,
        use_ist=args.use_ist,
        ist_language=args.ist_language,
        ist_path=args.ist_path,
        ist_styles=[s.strip() for s in args.ist_styles.split(",") if s.strip()] if args.ist_styles else None,
        ist_expand=args.ist_expand,
    )
    test_poisoned, test_clean_eval = poisoner.create_eval_sets(test_clean)

    # Load victim model (backdoored or clean as provided)
    victim = HuggingFaceClassifierVictim(
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        base_model=(args.base_model or None),
        checkpoint_path=(args.checkpoint or None),
        model_info_path=(args.model_info or None),
        num_labels=args.num_labels,
        strict_load=bool(args.strict_load),
    )

    # Baseline evaluation (pre-defense)
    baseline_metrics = evaluate(
        victim=victim,
        clean_data=test_clean_eval,
        poison_data=test_poisoned,
        input_field=args.input_field,
        label_field=args.label_field,
        target_label=args.target_label,
    )

    # Persist baseline metrics
    baseline_path = os.path.join(out_dir, "metrics.pre.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump({
            "task": args.task,
            "poison_rate": args.poison_rate,
            "trigger": args.trigger,
            "target_label": args.target_label,
            "counts": {
                "test_clean_eval": len(test_clean_eval),
                "test_poisoned": len(test_poisoned),
            },
            **baseline_metrics,
        }, f, indent=2, ensure_ascii=False)

    # Optionally run LLM-based cleaning before post-defense evaluation
    if args.use_defense:
        cleaner = LLMCleaner(
            run_vllm_script=args.vllm_script,
            run_infer_script=args.infer_script,
        )

        # Ensure vLLM is up or try to start if model path provided
        if not cleaner.is_server_alive(args.api_base):
            if args.llm_model_path:
                cleaner.start_vllm(
                    model_path=args.llm_model_path,
                    served_model_name=args.served_model_name,
                    port=args.llm_port,
                    max_len=args.llm_max_len,
                    dtype=args.llm_dtype,
                )
                # Wait a bit for server to come up
                time.sleep(8)
            else:
                print("[WARN] vLLM server not reachable and no --llm-model-path provided. Proceeding may fail.")

        # Persist intermediate files
        eval_clean_file = os.path.join(artifacts_dir, "test.clean.jsonl")
        eval_poison_file = os.path.join(artifacts_dir, "test.poison.jsonl")
        write_jsonl(eval_clean_file, test_clean_eval)
        write_jsonl(eval_poison_file, test_poisoned)

        cleaned_clean_file = os.path.join(cleaned_dir, "test.clean.cleaned.jsonl")
        cleaned_poison_file = os.path.join(cleaned_dir, "test.poison.cleaned.jsonl")

        maybe_clean_file(
            cleaner,
            input_path=eval_clean_file,
            output_path=cleaned_clean_file,
            field=args.input_field,
            template=args.template,
            system_prompt=args.system_prompt,
            model_alias=args.served_model_name,
            api_base=args.api_base,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        maybe_clean_file(
            cleaner,
            input_path=eval_poison_file,
            output_path=cleaned_poison_file,
            field=args.input_field,
            template=args.template,
            system_prompt=args.system_prompt,
            model_alias=args.served_model_name,
            api_base=args.api_base,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        # Reload cleaned datasets
        test_clean_eval = load_split(cleaned_clean_file, fmt="jsonl")
        test_poisoned = load_split(cleaned_poison_file, fmt="jsonl")

        # Post-defense evaluation on cleaned data
        defended_metrics = evaluate(
            victim=victim,
            clean_data=test_clean_eval,
            poison_data=test_poisoned,
            input_field=args.input_field,
            label_field=args.label_field,
            target_label=args.target_label,
        )

        # Persist defended metrics and deltas
        defended_path = os.path.join(out_dir, "metrics.post.json")
        with open(defended_path, "w", encoding="utf-8") as f:
            json.dump({
                "task": args.task,
                "poison_rate": args.poison_rate,
                "trigger": args.trigger,
                "target_label": args.target_label,
                "counts": {
                    "test_clean_eval": len(test_clean_eval),
                    "test_poisoned": len(test_poisoned),
                },
                **defended_metrics,
            }, f, indent=2, ensure_ascii=False)

        summary = {
            "task": args.task,
            "poison_rate": args.poison_rate,
            "trigger": args.trigger,
            "target_label": args.target_label,
            "baseline": baseline_metrics,
            "defended": defended_metrics,
            "delta": {
                "acc": float(defended_metrics["acc"]) - float(baseline_metrics["acc"]),
                "asr": float(defended_metrics["asr"]) - float(baseline_metrics["asr"]),
                "casr": float(defended_metrics["casr"]) - float(baseline_metrics["casr"]),
            },
        }
        metrics_path = os.path.join(out_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(json.dumps({"output_dir": out_dir, **summary}, indent=2, ensure_ascii=False))
    else:
        # No defense; keep baseline as final
        metrics_path = os.path.join(out_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "task": args.task,
                "poison_rate": args.poison_rate,
                "trigger": args.trigger,
                "target_label": args.target_label,
                "baseline": baseline_metrics,
            }, f, indent=2, ensure_ascii=False)

        print(json.dumps({"output_dir": out_dir, "baseline": baseline_metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


