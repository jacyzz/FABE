#!/bin/bash

# DeepSeek-Coder 训练脚本
# 使用单张A100 80G进行训练

export CUDA_VISIBLE_DEVICES=2

# 训练参数配置
MODEL_PATH="/home/nfs/share-yjy/dachuang2025/models/deepseek-coder-6.7b-instruct"
TRAIN_DATA_PATH="../data/universal_format/"
OUTPUT_DIR="/home/nfs/share-yjy/dachuang2025/defense_model/deepseek-coder-defense"
LOG_DIR="logs/deepseek-coder"

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# 训练命令
torchrun \
    --nproc_per_node=1 \
    main.py \
    --task coding \
    --do_train \
    --model_template deepseek \
    --model_name_or_path $MODEL_PATH \
    --train_file_path $TRAIN_DATA_PATH \
    --per_device_train_batch_size 1 \
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
    --seed 42 \


# 训练状态检查
if [ $? -eq 0 ]; then
    echo "训练成功完成！"
    echo "模型保存在: $OUTPUT_DIR"
    echo "日志文件在: $LOG_DIR"
else
    echo "训练失败，请检查错误信息"
    exit 1
fi
