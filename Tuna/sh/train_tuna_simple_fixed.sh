#!/bin/bash
set -euo pipefail

# =============================================================================
# Tuna Defense Model Training Script - Simple & Fixed Version
# =============================================================================

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1

# =============================================================================
# Configuration - Based on Successful train_3b.sh
# =============================================================================

# Model and Data Paths
MODEL_PATH=${MODEL_PATH:-"/home/nfs/share-yjy/dachuang2025/models/deepseek-coder-6.7b-instruct"}
DATA_PATH=${DATA_PATH:-"/home/nfs/share-yjy/dachuang2025/data/fabe_tuna/tuna_processed_val-00002-of-00003_optimized_ranking_rank4.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"/home/nfs/u2023-zlb/FABE/checkpoints/deepseek_codertuna_defense_lora_simple"}

# Training Configuration - Exactly like train_3b.sh
NUM_EPOCHS=${NUM_EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-2}  # Same as train_3b.sh
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}  # Same as train_3b.sh
LEARNING_RATE=${LEARNING_RATE:-2e-5}  # Same as train_3b.sh
WARMUP_RATIO=${WARMUP_RATIO:-0.03}  # Same as train_3b.sh
SAVE_STEPS=${SAVE_STEPS:-200}

# LoRA Configuration - Minimal
PEFT_TYPE=${PEFT_TYPE:-"lora"}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET=${LORA_TARGET:-"q_proj,k_proj,v_proj,o_proj"}

# Template and System Prompt
CHAT_TEMPLATE=${CHAT_TEMPLATE:-"deepseek"}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-"You are a security-focused code assistant."}
NO_SYSTEM=${NO_SYSTEM:-false}

# Hardware and Performance - Exactly like train_3b.sh
BF16=${BF16:-true}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}
DATALOADER_WORKERS=${DATALOADER_WORKERS:-4}

# Tuna specific parameters - Exactly like train_3b.sh
NO_DISCRIMINATE=${NO_DISCRIMINATE:-false}
LENPEN=${LENPEN:-1.0}
MLE_WEIGHT=${MLE_WEIGHT:-1.0}
MARGIN=${MARGIN:-0.1}

# =============================================================================
# Validation and Setup
# =============================================================================

echo "=== Tuna Defense Model Training Configuration - Simple & Fixed ==="
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
# Training Command - Using train_3b.sh Pattern
# =============================================================================

cd /home/nfs/u2023-zlb/FABE/Tuna/src

echo "Starting Tuna training using simple pattern..."
echo "Logs will be saved to: $OUTPUT_DIR/training.log"

# Use DeepSpeed pattern exactly like train_3b.sh
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
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type "cosine" \
    \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 3 \
    --logging_steps 10 \
    --report_to "tensorboard" \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
    --log_level "info" \
    \
    --bf16 ${BF16} \
    --remove_unused_columns false \
    \
    --no_discriminate ${NO_DISCRIMINATE} \
    --lenpen ${LENPEN} \
    --mle_weight ${MLE_WEIGHT} \
    --margin ${MARGIN} \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "Training log: $OUTPUT_DIR/training.log"
