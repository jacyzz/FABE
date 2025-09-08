#!/bin/bash

# --- 專案路徑設定 ---
export PROJECT_ROOT="/home/nfs/u2023-zlb/FABE"
# --- 指定只使用第一張 GPU (索引為 0) ---
export CUDA_VISIBLE_DEVICES=1

# --- 模型與資料設定 ---
MODEL_PATH="/home/nfs/share-yjy/local_llm/BigCode/starcoder2-7b"
DATA_PATH="/home/nfs/share-yjy/dachuang2025/data/fabe_tuna/devign_fabe.json"
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/tuna_starcoder7b_deepspeed_single"

mkdir -p ${OUTPUT_DIR}

# --- 核心訓練超參數 (為單卡 A6000 優化) ---
# 降低單卡批次大小以確保能放入顯存
PER_DEVICE_TRAIN_BATCH_SIZE=2
# 增加梯度累積步數，以維持穩定的有效批次大小 (2 * 16 = 32)
GRADIENT_ACCUMULATION_STEPS=16
NUM_TRAIN_EPOCHS=3
LEARNING_RATE=2e-5
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.01

# --- DeepSpeed 設定 ---
# 指向為單卡優化的設定檔
DEEPSPEED_CONFIG_PATH="./configs/ds_config1.json"

# --- 啟動訓練 ---
cd "${PROJECT_ROOT}/Tuna/src"

echo "啟動單卡 DeepSpeed 訓練 (Stage 3)..."

# DeepSpeed 啟動器會自動偵測到只有一張可見的 GPU
deepspeed train_tuna.py \
    --deepspeed ${DEEPSPEED_CONFIG_PATH} \
    \
    --model_name_or_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_ratio ${WARMUP_RATIO} \
    --weight_decay ${WEIGHT_DECAY} \
    \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --log_level "info" \
    --bf16 True \
    --remove_unused_columns False 2>&1 | tee "${OUTPUT_DIR}/training.log"

echo "訓練腳本執行完畢。"