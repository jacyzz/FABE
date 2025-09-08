#!/bin/bash

# =====================================================================================
#
# PRO 项目训练脚本 (针对 clone-detect 数据集)
#
#   - 模型: DeepSeek-Coder 6.7B Instruct
#   - 数据: 使用 generate_clone_data.sh 生成的通用格式数据
#   - 核心逻辑:
#     1. 使用 torchrun 启动分布式训练。
#     2. 指定模型路径、数据路径和输出目录。
#     3. 动态应用 'deepseek' 提示模板 (--model_template deepseek)。
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

# 验证数据路径
VALIDATION_DATA_PATH="/home/nfs/u2023-zlb/datasets/universal_clone_detect_format/validation-*.jsonl"

# 模型输出目录
OUTPUT_DIR="/home/nfs/share-yjy/dachuang2025/defense_model/pro-deepseek-clone-detect"

# 日志目录
LOG_DIR="logs/pro-deepseek-clone-detect"

# --- 环境设置 ---

# 创建输出和日志目录
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

echo "开始 PRO 训练..."
echo "模型: $MODEL_PATH"
echo "训练数据: $TRAIN_DATA_PATH"
echo "输出目录: $OUTPUT_DIR"

# --- 训练命令 ---

torchrun \
    --nproc_per_node=1 \
    /home/nfs/u2023-zlb/FABE/PRO/train/main.py \
    --task coding \
    --do_train \
    --model_template deepseek \
    --model_name_or_path $MODEL_PATH \
    --train_file_path $TRAIN_DATA_PATH \
    --validation_file_path $VALIDATION_DATA_PATH \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 3e-5 \
    --block_size 2048 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 16 \
    --output_dir $OUTPUT_DIR \
    --log_path $LOG_DIR \
    --checkpointing_step 500 \
    --training_stage_num 4 \
    --sft_weight 1.5 \
    --use_lora \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --lora_target_modules q_proj v_proj \
    --bf16 \
    --seed 42

# --- 训练状态检查 ---

if [ $? -eq 0 ]; then
    echo "PRO 训练成功完成！"
    echo "模型保存在: $OUTPUT_DIR"
    echo "日志文件在: $LOG_DIR"
else
    echo "PRO 训练失败，请检查错误信息。"
    exit 1
fi
