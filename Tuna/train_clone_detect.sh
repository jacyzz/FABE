#!/bin/bash

# =====================================================================================
#
# Tuna 项目训练脚本 (针对 clone-detect 数据集)
#
#   - 模型: DeepSeek-Coder 6.7B Instruct
#   - 数据: 使用 generate_clone_data.sh 生成的通用格式数据
#   - 核心逻辑:
#     1. 直接运行 train_tuna.py 脚本。
#     2. 指定模型路径、数据路径和输出目录。
#     3. 使用 Tuna 内置的 'deepseek' 聊天模板 (--chat_template deepseek)。
#     4. 启用 LoRA 进行高效微调。
#
# =====================================================================================

export CUDA_VISIBLE_DEVICES=2

# --- 训练参数配置 ---

# 基础模型路径
MODEL_PATH="/home/nfs/share-yjy/dachuang2025/models/deepseek-coder-6.7b-instruct"

# 训练数据路径 (包含多个 .jsonl 文件)
# 注意：这里使用了通配符 * 来匹配所有 train-*.jsonl 文件
TRAIN_DATA_PATH="/home/nfs/u2023-zlb/datasets/universal_clone_detect_format/train-*.jsonl"

# 模型输出目录
OUTPUT_DIR="/home/nfs/share-yjy/dachuang2025/defense_model/tuna-deepseek-clone-detect"

# 日志目录
LOG_DIR="logs/tuna-deepseek-clone-detect"

# --- 环境设置 ---

# 创建输出和日志目录
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

echo "开始 Tuna 训练..."
echo "模型: $MODEL_PATH"
echo "训练数据: $TRAIN_DATA_PATH"
echo "输出目录: $OUTPUT_DIR"

# --- 训练命令 ---

python /home/nfs/u2023-zlb/FABE/Tuna/src/train_tuna.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $TRAIN_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --logging_dir $LOG_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-5 \
    --model_max_length 4096 \
    --peft lora \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --chat_template deepseek \
    --bf16 \
    --tf32 True \
    --optim "adamw_torch" \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 250 \
    --eval_strategy "steps" \
    --eval_steps 250 \
    --save_total_limit 3 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --report_to "tensorboard"

# --- 训练状态检查 ---

if [ $? -eq 0 ]; then
    echo "Tuna 训练成功完成！"
    echo "模型保存在: $OUTPUT_DIR"
    echo "日志文件在: $LOG_DIR"
else
    echo "Tuna 训练失败，请检查错误信息。"
    exit 1
fi
