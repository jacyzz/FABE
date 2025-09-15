#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List
from datasets import load_dataset
import torch
import torch.nn.functional as F
import transformers
from transformers import Trainer
from transformers import BitsAndBytesConfig
from transformers import AutoConfig
from transformers.utils import is_bitsandbytes_available
from peft import LoraConfig, get_peft_model, PeftModel
import itertools
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
import utils
from custom import TunaTrainer

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

PROMPT_EMPTY_DICT = {
    "prompt_input": ("{instruction}\n{input}\n"),
    "prompt_no_input": ("{instruction}\n"),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    peft: str = field(default="none", metadata={"help": "none|lora|qlora"})
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="auto", metadata={"help": "auto|qkv|mlp|all or comma-separated module names"})


@dataclass
class DataArguments:
    data_path: List[str] = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    chat_template: str = field(
        default="base",
        metadata={"help": "Chat template to use: auto|base|llama|mistral|qwen|deepseek|baichuan|internlm|yi|starcoder"},
    )
    system_prompt: str = field(
        default="",
        metadata={"help": "System prompt for defense instructions (e.g., remove backdoors)."},
    )
    no_system: bool = field(
        default=False,
        metadata={"help": "Disable system prompt injection even if provided."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mle_weight: float = field(default=1.0, metadata={"help": "Weight for MLE loss."})
    margin: float = field(default=0.1, metadata={"help": "Margin for margin loss."})
    no_discriminate: bool = field(
        default=False, metadata={"help": "Whether to discriminate gold and candidate"}
    )
    lenpen: float = field(
        default=1.0, metadata={"help": "Length penalty for generation."}
    )
    bf16: bool = field(default=False, metadata={"help": "Enable bfloat16 training."})


def _padding_fn(tensor_list, padding_value):
    # padding a list of tensors to the same length
    if not isinstance(tensor_list[0], torch.Tensor):
        tensor_list = [torch.tensor(t) for t in tensor_list]
    assert (
        len(set([t.shape[0] for t in tensor_list])) == 1
    ), "batch size should be the same"
    max_len = max([t.shape[1] for t in tensor_list])
    padded_tensor_list = []
    for t in tensor_list:
        padded_tensor_list.append(
            F.pad(t, (0, max_len - t.shape[1]), value=padding_value)
        )
    return torch.stack(padded_tensor_list, dim=0)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(
    sources: Sequence[str],
    scores: Sequence[float],
    instruction_prefix: str,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Tokenize a list of strings."""
    input_ids = tokenizer(
        sources,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    labels = input_ids.clone()
    labels.masked_fill_(labels == tokenizer.pad_token_id, IGNORE_INDEX)

    instructions_tokenized = tokenizer(
        instruction_prefix,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids[0]

    length = instructions_tokenized.ne(tokenizer.pad_token_id).sum().item()
    labels[:, :length] = IGNORE_INDEX

    return dict(
        input_ids=input_ids,
        labels=labels,
        scores=torch.tensor(scores),
    )


def preprocess(
    sources: Sequence[Sequence[str]],
    scores: Sequence[Sequence[float]],
    instruction_prefixes: Sequence[str],
    outputs: Sequence[Sequence[str]],
    ids: Sequence[int],
    tokenizer: transformers.PreTrainedTokenizer,
) -> List[Dict]:
    """sort the sources by scores and tokenize them."""
    num_generations = len(sources[0])
    assert (
        len(set(len(s) for s in sources)) == 1
    ), "All sources should have the same number of generations."
    # Sort the sources by scores, keep only the source string
    ss_sorted = [
        sorted(zip(source, score), key=lambda x: x[1], reverse=True)
        for source, score in zip(sources, scores)
    ]
    sources_sorted = [[s[0] for s in ss] for ss in ss_sorted]
    scores_sorted = [[s[1] for s in ss] for ss in ss_sorted]

    list_data_dict = [
        _tokenize_fn(so, sc, ins_pref, tokenizer)
        for so, sc, ins_pref in zip(sources_sorted, scores_sorted, instruction_prefixes)
    ]

    return list_data_dict


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, model_name_or_path: str, cfg: AutoConfig, data_args: DataArguments):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")

        instructions = [
            example["instruction"] for example in list_data_dict
        ]  # including instruction and input
        ids = [example["id"] for example in list_data_dict]
        outputs = [example["output"] for example in list_data_dict]
        scores = [example["score"] for example in list_data_dict]

        template = data_args.chat_template
        if template == "auto":
            template = _infer_model_family(model_name_or_path, cfg)

        sys_txt = "" if data_args.no_system else (data_args.system_prompt or "")
        instruction_prefixes = [
            _render_prefix(inst, sys_txt, template) for inst in instructions
        ]

        sources = [
            [f"{pref}{out}{tokenizer.eos_token}" for out in output]
            for pref, output in zip(instruction_prefixes, outputs)
        ]

        logging.warning("Tokenizing inputs... This may take some time...")
        self.data_dict_list = preprocess(
            sources, scores, instruction_prefixes, outputs, ids, tokenizer
        )

    def __len__(self):
        return len(self.data_dict_list)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.data_dict_list[i]["input_ids"],
            labels=self.data_dict_list[i]["labels"],
            scores=self.data_dict_list[i]["scores"],
        )


def _tokenize(
    sources: Sequence[Sequence[str]],
    scores: Sequence[Sequence[float]],
    instruction_prefixes: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Tokenize a list of strings."""
    input_ids = [
        tokenizer(
            s,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        for s in sources
    ]

    labels = copy.deepcopy(input_ids)
    labels = [l.masked_fill_(l == tokenizer.pad_token_id, IGNORE_INDEX) for l in labels]

    instructions_tokenized = tokenizer(
        instruction_prefixes,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    length = instructions_tokenized.ne(tokenizer.pad_token_id).sum(dim=-1).tolist()
    for i, l in enumerate(length):
        labels[i][:, :l] = IGNORE_INDEX

    return dict(
        input_ids=input_ids,
        labels=labels,
        scores=torch.tensor(scores),
    )


def _infer_model_family(name: str, config: AutoConfig) -> str:
    lowered = (config.model_type or "").lower()
    path_lower = (name or "").lower()
    if "deepseek" in path_lower:
        return "deepseek"
    if lowered in ["llama", "mistral", "gemma", "falcon", "yi"]:
        return lowered
    if "qwen" in path_lower or lowered.startswith("qwen"):
        return "qwen"
    if "baichuan" in path_lower:
        return "baichuan"
    if "internlm" in path_lower:
        return "internlm"
    if "starcoder" in path_lower or lowered in ["gpt_bigcode", "codegen"]:
        return "starcoder"
    return lowered or "base"


def _render_prefix(instruction: str, system_prompt: str, template: str) -> str:
    sys_txt = (system_prompt or "").strip()
    tpl = (template or "base").lower()
    if tpl == "base":
        if sys_txt:
            return f"{sys_txt}\n\n{instruction}\n"
        return f"{instruction}\n"
    if tpl in ["llama", "mistral"]:
        if sys_txt:
            return f"[INST] <<SYS>>\n{sys_txt}\n<</SYS>>\n{instruction} [/INST] "
        return f"[INST] {instruction} [/INST] "
    if tpl == "qwen":
        parts = []
        if sys_txt:
            parts.append("<|im_start|>system\n" + sys_txt + "<|im_end|>\n")
        parts.append("<|im_start|>user\n" + instruction + "<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)
    if tpl == "deepseek":
        if sys_txt:
            return f"System: {sys_txt}\nUser: {instruction}\nAssistant: "
        return f"User: {instruction}\nAssistant: "
    if tpl in ["baichuan", "internlm", "yi", "starcoder", "falcon", "gemma"]:
        if sys_txt:
            return f"{sys_txt}\n\n### Instruction:\n{instruction}\n\n### Response: "
        return f"### Instruction:\n{instruction}\n\n### Response: "
    return f"{instruction}\n"


def tokenize_function(examples, tokenizer, model_name_or_path: str, cfg: AutoConfig, data_args: DataArguments):
    instructions = examples["instruction"]
    ids = examples["id"]
    outputs = examples["output"]
    scores = examples["score"]
    num_generations = len(outputs[0])
    assert (
        len(set([len(out) for out in outputs])) == 1
    ), f"All instructions must have {num_generations} outputs."

    template = data_args.chat_template
    if template == "auto":
        template = _infer_model_family(model_name_or_path, cfg)

    sys_txt = "" if data_args.no_system else (data_args.system_prompt or "")
    instruction_prefixes = [
        _render_prefix(inst, sys_txt, template) for inst in instructions
    ]

    sources = [
        [f"{pref}{out}{tokenizer.eos_token}" for out in output]
        for pref, output in zip(instruction_prefixes, outputs)
    ]
    ss_sorted = [
        sorted(zip(source, score), key=lambda x: x[1], reverse=True)
        for source, score in zip(sources, scores)
    ]
    sources_sorted = [[s[0] for s in ss] for ss in ss_sorted]
    scores_sorted = [[s[1] for s in ss] for ss in ss_sorted]

    data_dict = _tokenize(sources_sorted, scores_sorted, instruction_prefixes, tokenizer)
    return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, scores = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "scores")
        )
        input_ids = _padding_fn(input_ids, padding_value=self.tokenizer.pad_token_id)
        labels = _padding_fn(labels, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, model_name_or_path: str, cfg: AutoConfig
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path, model_name_or_path=model_name_or_path, cfg=cfg, data_args=data_args
    )
    eval_dataset = None  # optional
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    load_kwargs = dict(cache_dir=training_args.cache_dir)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **{k: v for k, v in load_kwargs.items() if v is not None})

    quantization_config = None
    if model_args.peft.lower() == "qlora":
        if not is_bitsandbytes_available():
            raise RuntimeError("bitsandbytes not available but peft=qlora was requested.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        )
        load_kwargs.update(dict(
            device_map="auto",
            quantization_config=quantization_config,
        ))

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **{k: v for k, v in load_kwargs.items() if v is not None},
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
    # Define PAD Token = BOS Token
    tokenizer.pad_token = tokenizer.bos_token
    model.config.pad_token_id = model.config.bos_token_id

    # Apply PEFT if requested (LoRA/QLoRA)
    def _infer_lora_target_modules(cfg: AutoConfig, name_or_path: str) -> List[str]:
        lowered = (cfg.model_type or "").lower()
        if "auto" not in model_args.lora_target_modules:
            return [m.strip() for m in model_args.lora_target_modules.split(",") if m.strip()]
        # Heuristics per popular families
        if lowered in ["llama", "mistral", "falcon", "yi", "gemma"]:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if lowered in ["gpt_neox", "gptj", "opt"]:
            return ["q_proj", "k_proj", "v_proj", "out_proj"]
        if lowered in ["baichuan"]:
            return ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if lowered in ["qwen", "qwen2"]:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if lowered in ["internlm", "internlm2"]:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if lowered in ["codegen", "starcoder", "gpt_bigcode"]:
            return ["qkv_proj", "out_proj", "fc_in", "fc_out", "q_proj", "k_proj", "v_proj", "o_proj", "mlp.fc_in", "mlp.fc_out"]
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    if model_args.peft.lower() in ["lora", "qlora"]:
        target_modules = _infer_lora_target_modules(config, model_args.model_name_or_path)
        lora_cfg = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    raw_dataset = load_dataset("json", data_files=data_args.data_path, split="train")
    generation_dataset = raw_dataset.map(
        tokenize_function,
        fn_kwargs={
            "tokenizer": tokenizer,
            "model_name_or_path": model_args.model_name_or_path,
            "cfg": config,
            "data_args": data_args,
        },
        batched=True,
        batch_size=30,
        num_proc=32,
        remove_columns=raw_dataset.column_names,
        desc="Running tokenizer on dataset",
    )
    collator = DataCollatorForSupervisedDataset(tokenizer)
    # Tell Trainer not to attempt DataParallel
    model.is_parallelizable = True
    model.model_parallel = True
    print(f"training_args = {training_args}")
    trainer = TunaTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=generation_dataset,
        data_collator=collator,
    )
    model.config.use_cache = False

    trainer.train()
    trainer.save_state()
    # Save adapter weights if PEFT is enabled, otherwise full model
    if isinstance(model, PeftModel):
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()