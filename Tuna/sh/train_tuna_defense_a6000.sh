#!/bin/bash
set -euo pipefail

# =============================================================================
# Tuna Defense Model Training Script - Aggressive A6000 Optimization
# =============================================================================

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1
conda activate lmfty

# =============================================================================
# Configuration - Aggressively Optimized for 2x A6000
# =============================================================================

# Model and Data Paths
MODEL_PATH=${MODEL_PATH:-"/home/nfs/share-yjy/dachuang2025/models/deepseek-coder-6.7b-instruct"}
DATA_PATH=${DATA_PATH:-"/home/nfs/share-yjy/dachuang2025/data/fabe_tuna/tuna_processed_val-00002-of-00003_optimized_ranking_rank4.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"/home/nfs/share-yjy/dachuang2025/defense_model/deepseek_codertuna_defense_lora_a6000"}

# Training Configuration - Aggressive A6000 Optimization
NUM_EPOCHS=${NUM_EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-1}  # Very conservative batch size
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-32}  # Large accumulation for effective batch size
LEARNING_RATE=${LEARNING_RATE:-1e-4}  # Slightly lower LR for stability
WARMUP_STEPS=${WARMUP_STEPS:-200}  # More warmup steps
SAVE_STEPS=${SAVE_STEPS:-500}  # Less frequent saving

# LoRA Configuration - Optimized for memory
PEFT_TYPE=${PEFT_TYPE:-"qlora"}  # Use QLoRA for better memory efficiency
LORA_R=${LORA_R:-8}  # Reduced rank for memory
LORA_ALPHA=${LORA_ALPHA:-16}  # Reduced alpha
LORA_DROPOUT=${LORA_DROPOUT:-0.1}  # Slightly higher dropout
LORA_TARGET=${LORA_TARGET:-"auto"}

# Template and System Prompt
CHAT_TEMPLATE=${CHAT_TEMPLATE:-"deepseek"}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-"You are a security-focused code assistant. Your task is to identify and neutralize any potential backdoors, vulnerabilities, or malicious code in the provided code. Always explain your findings and provide secure alternatives. Prioritize security over functionality."}
NO_SYSTEM=${NO_SYSTEM:-false}

# Hardware and Performance - Aggressive A6000 Optimization
NPROC=${NPROC:-2}
BF16=${BF16:-true}
FP16=${FP16:-false}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}
DATALOADER_WORKERS=${DATALOADER_WORKERS:-4}  # Reduced for memory

# Memory optimization for large models
MAX_GRAD_NORM=${MAX_GRAD_NORM:-0.5}  # More conservative gradient clipping
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}  # Higher weight decay for regularization
MAX_MEMORY=${MAX_MEMORY:-"40GB"}  # Reserve some memory for system

# Advanced optimizations
USE_FLASH_ATTENTION=${USE_FLASH_ATTENTION:-false}  # Set to true if flash-attn is available
USE_GRADIENT_ACCUMULATION=${USE_GRADIENT_ACCUMULATION:-true}
USE_GRADIENT_CHECKPOINTING=${USE_GRADIENT_CHECKPOINTING:-true}

# =============================================================================
# Validation and Setup
# =============================================================================

echo "=== Tuna Defense Model Training Configuration - Aggressive A6000 Optimization ==="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "PEFT: $PEFT_TYPE (r=$LORA_R, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT)"
echo "Template: $CHAT_TEMPLATE"
echo "System Prompt: $SYSTEM_PROMPT"
echo "Batch Size: $BATCH_SIZE per device (accum=$GRAD_ACCUM_STEPS, effective=$((BATCH_SIZE * GRAD_ACCUM_STEPS * NPROC)))"
echo "Learning Rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"
echo "GPUs: $NPROC"
echo "Memory Limit: $MAX_MEMORY per GPU"
echo "================================================"

# Check if data file exists
if [[ ! -f "$DATA_PATH" ]]; then
    echo "Error: Data file not found: $DATA_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Training Command - Aggressive Multi-GPU Optimization
# =============================================================================

cd /home/nfs/u2023-zlb/FABE/Tuna/src

echo "Starting Tuna training on $NPROC GPUs with aggressive optimization..."
echo "Logs will be saved to: $OUTPUT_DIR/training.log"

# Set environment variables for memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Use torchrun for multi-GPU training with aggressive optimization
torchrun --nproc_per_node=$NPROC \
    --master_port=29500 \
    train_tuna.py \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    \
    --peft "$PEFT_TYPE" \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules "$LORA_TARGET" \
    \
    --chat_template "$CHAT_TEMPLATE" \
    --system_prompt "$SYSTEM_PROMPT" \
    ${NO_SYSTEM:+--no_system} \
    \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type "cosine" \
    --warmup_steps $WARMUP_STEPS \
    \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit 2 \
    \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --bf16 $BF16 \
    --fp16 $FP16 \
    --dataloader_num_workers $DATALOADER_WORKERS \
    --max_grad_norm $MAX_GRAD_NORM \
    --weight_decay $WEIGHT_DECAY \
    \
    --logging_steps 20 \
    --report_to "tensorboard" \
    --remove_unused_columns true \
    \
    --mle_weight 1.0 \
    --margin 0.1 \
    --no_discriminate false \
    --lenpen 1.0 \
    \
    --ddp_backend "nccl" \
    --ddp_find_unused_parameters false \
    --dataloader_pin_memory false \
    --max_memory $MAX_MEMORY \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "Training log: $OUTPUT_DIR/training.log"

# =============================================================================
# Post-training Information
# =============================================================================

echo ""
echo "=== Post-training Information ==="
echo "To load the trained model:"
echo "from peft import PeftModel"
echo "from transformers import AutoModelForCausalLM, AutoTokenizer"
echo ""
echo "base_model = AutoModelForCausalLM.from_pretrained('$MODEL_PATH')"
echo "tokenizer = AutoTokenizer.from_pretrained('$MODEL_PATH')"
echo "model = PeftModel.from_pretrained(base_model, '$OUTPUT_DIR')"
echo ""
echo "To merge LoRA weights with base model (optional):"
echo "model = model.merge_and_unload()"
echo "model.save_pretrained('$OUTPUT_DIR/merged')"
echo "tokenizer.save_pretrained('$OUTPUT_DIR/merged')"
