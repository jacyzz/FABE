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
    # 验证相关参数已移除，仅保留训练
    parser.add_argument(
        "--sft_weight",
        type=float,
        default=2,
    )
    parser.add_argument(
        "--model_template",
        type=str,
        default="deepseek",
        help="The template for formatting the prompt. E.g., 'deepseek', 'default'."
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
        "--train_file_path", 
        type=str, 
        default=None, 
        help="Path to the training data. Can be: 1) A directory (loads all .json/.jsonl files), 2) A single .json/.jsonl file, 3) A glob pattern with * wildcard."
    )
    # 验证集路径参数已移除
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
    # 评估批大小参数已移除
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
    )
    # LR scheduler & warmup
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="Scheduler type: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup"
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps. If > 0, overrides warmup_ratio"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Warmup steps as a ratio of total training steps (used when num_warmup_steps == 0)"
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
        "--save_total_limit",
        type=int,
        default=2,
        help="Limit the total amount of checkpoints. Deletes the older checkpoints."
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
    parser.add_argument("--resume_adapter_path", type=str, default=None,
                        help="If set, resume LoRA training from this adapter checkpoint directory (step_* or epoch_*)")

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
