import random
import numpy as np
import torch
import argparse
from transformers.trainer_utils import SchedulerType

def parse_args():
    parser = argparse.ArgumentParser(description="Preference Ranking Optimization For Human Alignment")
    parser.add_argument(
        "--task",
        type=str,
        default="hh",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
    )
    parser.add_argument(
        "--do_validation",
        action="store_true",
    )
    parser.add_argument(
        "--sft_weight",
        type=float,
        default=2,
    )
    parser.add_argument(
        "--index",
        type=str,
        default="100",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--training_stage_num",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--train_file_path", type=str, default=None,
    )
    parser.add_argument(
        "--validation_file_path", type=str, default=None,
    )
    parser.add_argument(
        "--validation_file_name", type=str, default=None,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=20,
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
    )
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--checkpointing_step",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="logs",
    )
    # Add LoRA related arguments
    parser.add_argument("--use_lora", action="store_true", help="Whether to use LoRA")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", nargs="+", type=str, default=["c_proj", "c_attn", "q_attn", "c_fc"],
                        help="List of module names to apply LoRA to")

    # Add quantization arguments  
    parser.add_argument("--use_4bit", action="store_true", help="Enable 4-bit quantization")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", help="4-bit quantization type")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16", help="Compute dtype for 4-bit quantization")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true", help="Use double quantization")
    
    # Add mixed precision training
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision training")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 mixed precision training")

    args = parser.parse_args()
    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

# 移除直接执行的代码
# args = parse_args()
# setup_seed(args.seed)

# 创建一个全局参数对象
args = None

def init_args():
    global args
    if args is None:
        args = parse_args()
        setup_seed(args.seed)
    return args

# 模型配置字典
MODEL_CONFIGS = {
    "llama": {
        "tokenizer_class": "LlamaTokenizer",
        "special_tokens": {"pad_token": "</s>", "bos_token": "<s>", "eos_token": "</s>"},
        "stop_sequences": ["Human:", "Assistant:", "\n\n"]
    },
    "deepseek-coder": {
        "tokenizer_class": "AutoTokenizer", 
        "special_tokens": {"pad_token": "", "bos_token": "", "eos_token": ""},
        "stop_sequences": ["\n\n", "\nclass", "\ndef", "\n#", "\n//"]
    },
    "starcoder": {
        "tokenizer_class": "AutoTokenizer",
        "special_tokens": {"pad_token": "<|endoftext|>", "bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>"},
        "stop_sequences": ["\n\n", "<|endoftext|>"]
    },
    "auto": {
        "tokenizer_class": "AutoTokenizer",
        "special_tokens": {"pad_token": "", "bos_token": "", "eos_token": ""},
        "stop_sequences": ["\n\n"]
    }
}

# 模型模板配置
MODEL_TEMPLATES = {
    "deepseek-coder": {
        "format": "chatml",
        "system_prefix": "<|im_start|>system\n",
        "system_suffix": "<|im_end|>\n",
        "user_prefix": "<|im_start|>user\n",
        "user_suffix": "<|im_end|>\n",
        "assistant_prefix": "<|im_start|>assistant\n",
        "assistant_suffix": "<|im_end|>\n"
    },
    "starcoder2": {
        "format": "simple",
        "user_prefix": "### Instruction:\n",
        "user_suffix": "\n",
        "assistant_prefix": "### Response:\n",
        "assistant_suffix": ""
    },
    "codellama": {
        "format": "llama2",
        "system_prefix": "<<SYS>>\n",
        "system_suffix": "\n<</SYS>>\n\n",
        "user_prefix": "[INST] ",
        "user_suffix": " [/INST]",
        "assistant_prefix": " ",
        "assistant_suffix": "</s>"
    },
    "default": {
        "format": "simple",
        "user_prefix": "Human: ",
        "user_suffix": "\nAssistant:",
        "assistant_prefix": " ",
        "assistant_suffix": ""
    }
}

# 防御任务系统提示配置
DEFENSE_SYSTEM_PROMPTS = {
    "default": "你是一个代码安全专家，负责检测和清除代码中的潜在后门和安全漏洞。请分析给定的代码，识别任何可疑模式，并提供安全的替代方案。",
    "backdoor_detection": "你是一个防御模型，专门检测和清除代码中的后门。对于每个代码片段，请仔细分析其安全性，如果发现后门，请提供修复后的安全版本。",
    "code_sanitization": "作为代码安全检查员，你的任务是清理代码中的安全漏洞和可疑模式，确保代码的安全性。"
}

# 提示策略配置
PROMPT_STRATEGY = {
    "use_system_prompt": True,
    "system_prompt_key": "default",
    "instruction_template": "请分析以下代码的安全性：\n{code}",
    "batch_inference": True
}
