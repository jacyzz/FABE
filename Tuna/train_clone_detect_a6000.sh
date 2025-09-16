#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# --- Model and Data ---
MODEL_PATH="/home/nfs/share-yjy/dachuang2025/models/deepseek-coder-6.7b-instruct"
# [修正] 修正通配符，确保只加载训练集数据
TRAIN_DATA_PATH="/home/nfs/share-yjy/dachuang2025/data/universe_data/universal_processed_train-*.jsonl"
OUTPUT_DIR="/home/nfs/u2023-zlb/FABE/checkpoints/tuna-deepseek-coder-6.7b-instruct"
LOG_DIR="logs/tuna-deepseek-coder-6.7b-instruct"

mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# --- Training Command for Single GPU ---
python /home/nfs/u2023-zlb/FABE/Tuna/src/train_tuna.py \
    --model_name_or_path $MODEL_PATH \
    --data_path "$TRAIN_DATA_PATH" \
    --output_dir $OUTPUT_DIR \
    --logging_dir $LOG_DIR \
    \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --model_max_length 2048 \
    --optim "adamw_torch" \
    \
    --peft lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    \
    --chat_template deepseek \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 3 \
    --report_to "tensorboard"