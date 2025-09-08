#!/bin/bash

# --- 环境变量与路径配置 ---
# 确保脚本能找到你的项目根目录
export PROJECT_ROOT="/home/nfs/u2023-zlb/FABE"
# 指定要使用的三张 GPU
export CUDA_VISIBLE_DEVICES=0,1

# --- 模型与数据配置 ---
MODEL_PATH="/home/nfs/share-yjy/local_llm/BigCode/starcoder2-7b" # 确认这是正确的 StarCoder-7B 模型路径
DATA_PATH="/home/nfs/share-yjy/dachuang2025/data/fabe_tuna/devign_fabe.json"
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/tuna_starcoder7b_deepspeed_optimized"

# 创建输出目录，避免因目录不存在而报错
mkdir -p ${OUTPUT_DIR}

# --- 核心训练超参数 (针对 3x A6000 优化) ---
NUM_TRAIN_EPOCHS=3
# 增大了单卡 Batch Size，40G 显存配合 DeepSpeed ZeRO-2 可以轻松应对
# 如果遇到 OOM (显存不足)，可以适当调低到 6 或 4
PER_DEVICE_TRAIN_BATCH_SIZE=2
# 保持不变，得到每个 GPU 上 32 (8*4) 的有效批次大小
# 总的全局有效批次大小为 32 * 3 = 96，这是一个非常健康的数值
GRADIENT_ACCUMULATION_STEPS=8
# 配合更大的 Batch Size，适当提高学习率是标准做法
LEARNING_RATE=2e-5
# 使用 warmup_ratio 更具灵活性，0.03 表示前 3% 的训练步数用于预热
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.01

# --- Tuna 特定超参数 ---
NO_DISCRIMINATE=False
LENPEN=1.0
MLE_WEIGHT=1.0
MARGIN=0.1

# --- DeepSpeed 配置 ---
# 指定 DeepSpeed 配置文件的路径
DEEPSPEED_CONFIG_PATH="./configs/ds_config.json"

# --- 日志与保存配置 ---
SAVE_STEPS=200
SAVE_TOTAL_LIMIT=3 # 建议多保存几个 checkpoint
LOGGING_STEPS=10

# --- 启动训练 ---
cd "${PROJECT_ROOT}/Tuna/src"

echo "开始使用 DeepSpeed 启动多卡训练..."

deepspeed --include localhost:0,1 train_tuna.py \
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
    --no_discriminate ${NO_DISCRIMINATE} \
    --lenpen ${LENPEN} \
    --mle_weight ${MLE_WEIGHT} \
    --margin ${MARGIN} \
    \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --logging_steps ${LOGGING_STEPS} \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --log_level "info" \
    \
    --bf16 True \
    --remove_unused_columns False 2>&1 | tee "${OUTPUT_DIR}/training_starcoder7b.log"

echo "训练完成。"