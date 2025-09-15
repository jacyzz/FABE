#!/bin/bash

# =====================================================================================
#
# PRO 项目训练脚本 (针对 clone-detect 数据集)
#
#   - 模型: DeepSeek-Coder 6.7B Instruct
#   - 数据: 使用 generate_clone_data.sh 生成的通用格式数据
#   - 核心逻辑:
#     1. 使用单GPU训练（移除分布式）
#     2. 指定模型路径、数据路径和输出目录。
#     3. 动态应用 'deepseek' 提示模板 (--model_template deepseek)。
#     4. 启用 LoRA 进行高效微调。
#
# =====================================================================================

export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 训练参数配置 ---

# 基础模型路径
MODEL_PATH="/home/nfs/share-yjy/dachuang2025/models/deepseek-coder-6.7b-instruct"

# 训练数据路径 (只使用训练集，减少数据量)
TRAIN_DATA_PATH="/home/nfs/u2023-zlb/datasets/universal_clone_detect_format/universal_processed_train-00000-of-00006_rank4.jsonl"

# # 验证数据路径 (使用引号包围通配符，防止shell展开)
# VALIDATION_DATA_PATH="/home/nfs/u2023-zlb/datasets/universal_clone_detect_format/universal_processed_validation-*.jsonl"

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

# --- 训练命令 (单GPU，无分布式) ---

python /home/nfs/u2023-zlb/FABE/PRO/train/main.py \
    --task coding \
    --do_train \
    --model_template deepseek \
    --model_name_or_path $MODEL_PATH \
    --train_file_path $TRAIN_DATA_PATH \
    --per_device_train_batch_size 3 \
    --learning_rate 5e-4 \
    --block_size 1024 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 6 \
    --dataloader_num_workers 4 \
    --output_dir $OUTPUT_DIR \
    --log_path $LOG_DIR \
    --checkpointing_step 250 \
    --save_total_limit 2 \
    --training_stage_num 4 \
    --sft_weight 1.5 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj" "v_proj" \
    --use_4bit \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --bnb_4bit_use_double_quant \
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