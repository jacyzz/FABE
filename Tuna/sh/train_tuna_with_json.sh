#!/bin/bash
set -euo pipefail

# =============================================================================
# Tuna Defense Model Training Script - Using Converted JSON Data
# =============================================================================

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1

# =============================================================================
# Configuration - Using Converted JSON Data
# =============================================================================

# Model and Data Paths
MODEL_PATH=${MODEL_PATH:-"/home/nfs/share-yjy/dachuang2025/models/deepseek-coder-6.7b-instruct"}

# 使用转换后的JSON文件
ORIGINAL_JSONL="/home/nfs/share-yjy/dachuang2025/data/fabe_tuna/tuna_processed_val-00002-of-00003_optimized_ranking_rank4.jsonl"
CONVERTED_JSON="/home/nfs/u2023-zlb/FABE/Tuna/tuna_processed_val-00002-of-00003_optimized_ranking_rank4.json"

# 检查转换后的JSON文件是否存在，如果不存在则自动转换
if [[ ! -f "$CONVERTED_JSON" ]]; then
    echo "转换后的JSON文件不存在，正在自动转换..."
    cd /home/nfs/u2023-zlb/FABE/Tuna/
    python convert_jsonl_to_json.py "$ORIGINAL_JSONL" "$CONVERTED_JSON"
    if [[ $? -ne 0 ]]; then
        echo "❌ 自动转换失败，请手动运行转换脚本"
        exit 1
    fi
    echo "✅ 自动转换完成"
fi

DATA_PATH=${DATA_PATH:-"$CONVERTED_JSON"}
OUTPUT_DIR=${OUTPUT_DIR:-"/home/nfs/u2023-zlb/FABE/checkpoints/deepseek_codertuna_defense_lora_json"}

# Training Configuration - Conservative
NUM_EPOCHS=${NUM_EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
SAVE_STEPS=${SAVE_STEPS:-200}

# LoRA Configuration - Conservative
PEFT_TYPE=${PEFT_TYPE:-"lora"}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET=${LORA_TARGET:-"q_proj,k_proj,v_proj,o_proj"}

# Template and System Prompt
CHAT_TEMPLATE=${CHAT_TEMPLATE:-"deepseek"}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-"You are a security-focused code assistant. Your task is to identify and neutralize any potential backdoors, vulnerabilities, or malicious code in the provided code. Always explain your findings and provide secure alternatives. Prioritize security over functionality."}
NO_SYSTEM=${NO_SYSTEM:-false}

# Hardware and Performance
BF16=${BF16:-true}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}
DATALOADER_WORKERS=${DATALOADER_WORKERS:-4}

# Tuna specific parameters
NO_DISCRIMINATE=${NO_DISCRIMINATE:-false}
LENPEN=${LENPEN:-1.0}
MLE_WEIGHT=${MLE_WEIGHT:-1.0}
MARGIN=${MARGIN:-0.1}

# =============================================================================
# Validation and Setup
# =============================================================================

echo "=== Tuna Defense Model Training Configuration - Using JSON Data ==="
echo "Model: $MODEL_PATH"
echo "Original JSONL: $ORIGINAL_JSONL"
echo "Converted JSON: $DATA_PATH"
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
    echo "❌ 错误: 数据文件不存在: $DATA_PATH"
    echo "请先运行转换脚本: python convert_jsonl_to_json.py <input.jsonl>"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Training Command - Using DeepSpeed
# =============================================================================

cd /home/nfs/u2023-zlb/FABE/Tuna/src

echo "Starting Tuna training with converted JSON data..."
echo "Logs will be saved to: $OUTPUT_DIR/training.log"

# Use DeepSpeed for training
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
