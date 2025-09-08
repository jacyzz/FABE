#!/bin/bash
set -euo pipefail

# =============================================================================
# Tuna Defense Model Training Script - Debug Version
# =============================================================================

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1

# =============================================================================
# Configuration - Minimal for Debugging
# =============================================================================

# Model and Data Paths
MODEL_PATH=${MODEL_PATH:-"/home/nfs/share-yjy/dachuang2025/models/deepseek-coder-6.7b-instruct"}
DATA_PATH=${DATA_PATH:-"/home/nfs/share-yjy/dachuang2025/data/fabe_tuna/tuna_processed_val-00002-of-00003_optimized_ranking_rank4.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"/home/nfs/u2023-zlb/FABE/checkpoints/deepseek_codertuna_defense_lora_debug"}

# Training Configuration - Minimal
NUM_EPOCHS=${NUM_EPOCHS:-1}  # 只训练1个epoch用于调试
BATCH_SIZE=${BATCH_SIZE:-1}  # 最小批次大小
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}  # 减少梯度累积
LEARNING_RATE=${LEARNING_RATE:-1e-4}  # 降低学习率
WARMUP_STEPS=${WARMUP_STEPS:-10}  # 减少预热步数
SAVE_STEPS=${SAVE_STEPS:-100}  # 更频繁保存

# LoRA Configuration - Conservative
PEFT_TYPE=${PEFT_TYPE:-"lora"}
LORA_R=${LORA_R:-8}  # 减少rank
LORA_ALPHA=${LORA_ALPHA:-16}  # 减少alpha
LORA_DROPOUT=${LORA_DROPOUT:-0.1}
LORA_TARGET=${LORA_TARGET:-"q_proj,k_proj,v_proj,o_proj"}  # 明确指定目标模块

# Template and System Prompt
CHAT_TEMPLATE=${CHAT_TEMPLATE:-"deepseek"}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-"You are a security-focused code assistant."}
NO_SYSTEM=${NO_SYSTEM:-false}

# Hardware and Performance - Conservative
BF16=${BF16:-true}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}  # 暂时禁用梯度检查点
DATALOADER_WORKERS=${DATALOADER_WORKERS:-0}  # 减少worker数量

# Memory optimization
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}

# =============================================================================
# Validation and Setup
# =============================================================================

echo "=== Tuna Defense Model Training Configuration - Debug Version ==="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "PEFT: $PEFT_TYPE (r=$LORA_R, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT)"
echo "Target modules: $LORA_TARGET"
echo "Template: $CHAT_TEMPLATE"
echo "Batch Size: $BATCH_SIZE per device (accum=$GRAD_ACCUM_STEPS, effective=$((BATCH_SIZE * GRAD_ACCUM_STEPS * 2)))"
echo "Learning Rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"
echo "GPUs: 2"
echo "================================================"

# Check if data file exists
if [[ ! -f "$DATA_PATH" ]]; then
    echo "Error: Data file not found: $DATA_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Training Command - Debug Version
# =============================================================================

cd /home/nfs/u2023-zlb/FABE/Tuna/src

echo "Starting Tuna training in debug mode..."
echo "Logs will be saved to: $OUTPUT_DIR/training.log"

# Use DeepSpeed with minimal configuration
deepspeed --include localhost:0,1 train_tuna.py \
    \
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
    --warmup_steps $WARMUP_STEPS \
    --lr_scheduler_type "cosine" \
    \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit 2 \
    \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --bf16 $BF16 \
    --dataloader_num_workers $DATALOADER_WORKERS \
    --max_grad_norm $MAX_GRAD_NORM \
    --weight_decay $WEIGHT_DECAY \
    \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --remove_unused_columns false \
    \
    --mle_weight 1.0 \
    --margin 0.1 \
    --no_discriminate false \
    --lenpen 1.0 \
    \
    --log_level "info" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "Training log: $OUTPUT_DIR/training.log"
