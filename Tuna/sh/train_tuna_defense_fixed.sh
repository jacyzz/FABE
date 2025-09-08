#!/bin/bash
set -euo pipefail

# =============================================================================
# Tuna Defense Model Training Script - Fixed Version (Based on train_3b.sh)
# =============================================================================

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1

# =============================================================================
# Configuration - Based on Successful train_3b.sh Pattern
# =============================================================================

# Model and Data Paths
MODEL_PATH=${MODEL_PATH:-"/home/nfs/share-yjy/dachuang2025/models/deepseek-coder-6.7b-instruct"}
DATA_PATH=${DATA_PATH:-"/home/nfs/share-yjy/dachuang2025/data/fabe_tuna/tuna_processed_val-00002-of-00003_optimized_ranking_rank4.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"/home/nfs/u2023-zlb/FABE/checkpoints/deepseek_codertuna_defense_lora"}

# Training Configuration - Based on Successful Pattern
NUM_EPOCHS=${NUM_EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-2}  # Same as train_3b.sh
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}  # Same as train_3b.sh
LEARNING_RATE=${LEARNING_RATE:-2e-5}  # Same as train_3b.sh
WARMUP_RATIO=${WARMUP_RATIO:-0.03}  # Same as train_3b.sh
SAVE_STEPS=${SAVE_STEPS:-200}

# LoRA Configuration
PEFT_TYPE=${PEFT_TYPE:-"lora"}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET=${LORA_TARGET:-"auto"}

# Template and System Prompt
CHAT_TEMPLATE=${CHAT_TEMPLATE:-"deepseek"}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-"You are a security-focused code assistant. Your task is to identify and neutralize any potential backdoors, vulnerabilities, or malicious code in the provided code. Always explain your findings and provide secure alternatives. Prioritize security over functionality."}
NO_SYSTEM=${NO_SYSTEM:-false}

# Hardware and Performance - Based on Successful Pattern
BF16=${BF16:-true}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}
DATALOADER_WORKERS=${DATALOADER_WORKERS:-4}

# Memory optimization for large models
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}

# Tuna specific parameters - Same as train_3b.sh
NO_DISCRIMINATE=${NO_DISCRIMINATE:-false}
LENPEN=${LENPEN:-1.0}
MLE_WEIGHT=${MLE_WEIGHT:-1.0}
MARGIN=${MARGIN:-0.1}

# =============================================================================
# Validation and Setup
# =============================================================================

echo "=== Tuna Defense Model Training Configuration - Fixed Version ==="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "PEFT: $PEFT_TYPE (r=$LORA_R, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT)"
echo "Template: $CHAT_TEMPLATE"
echo "System Prompt: $SYSTEM_PROMPT"
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
# Training Command - Using DeepSpeed Pattern from train_3b.sh
# =============================================================================

cd /home/nfs/u2023-zlb/FABE/Tuna/src

echo "Starting Tuna training using DeepSpeed pattern..."
echo "Logs will be saved to: $OUTPUT_DIR/training.log"

# Use DeepSpeed pattern like train_3b.sh
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
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type "cosine" \
    \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit 3 \
    \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --bf16 $BF16 \
    --dataloader_num_workers $DATALOADER_WORKERS \
    --max_grad_norm $MAX_GRAD_NORM \
    --weight_decay $WEIGHT_DECAY \
    \
    --logging_steps 10 \
    --report_to "tensorboard" \
    --remove_unused_columns false \
    \
    --mle_weight $MLE_WEIGHT \
    --margin $MARGIN \
    --no_discriminate $NO_DISCRIMINATE \
    --lenpen $LENPEN \
    \
    --log_level "info" \
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
