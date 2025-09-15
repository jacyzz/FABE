source /home/nfs/share-yjy/miniconda3/bin/activate
conda activate lmfty

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 训练参数配置 ---

# 基础模型路径
MODEL_PATH="/home/nfs/share-yjy/dachuang2025/models/deepseek-coder-6.7b-instruct"

# 训练数据路径
TRAIN_DATA_PATH="/home/nfs/share-yjy/dachuang2025/data/universe_data/universal_processed_train-00000-of-00006_rank4.jsonl"

# 验证相关配置已移除

# 模型输出目录
OUTPUT_DIR="/home/nfs/share-yjy/dachuang2025/defense_model/pro-deepseek-clone-detect"

# 日志目录
LOG_DIR="logs/pro-deepseek-clone-detect"

# --- 环境设置 ---

# 创建输出和日志目录
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

echo "开始 PRO 多卡训练..."
echo "模型: $MODEL_PATH"
echo "训练数据: $TRAIN_DATA_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "使用GPU: $CUDA_VISIBLE_DEVICES"

# --- 多卡训练命令 ---

torchrun --nproc_per_node=4 \
    /home/nfs/u2023-zlb/FABE/PRO/train/main.py \
    --task coding \
    --do_train \
    --model_template deepseek \
    --model_name_or_path $MODEL_PATH \
    --train_file_path $TRAIN_DATA_PATH \
    --per_device_train_batch_size 3 \
    --learning_rate 5e-4 \
    --block_size 1536 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 8 \
    --output_dir $OUTPUT_DIR \
    --log_path $LOG_DIR \
    --checkpointing_step 250 \
    --save_total_limit 2 \
    --training_stage_num 3 \
    --sft_weight 1.5 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj" "v_proj"\
    --fp16 \
    --seed 42

# --- 训练状态检查 ---

if [ $? -eq 0 ]; then
    echo "PRO 多卡训练成功完成！"
    echo "模型保存在: $OUTPUT_DIR"
    echo "日志文件在: $LOG_DIR"
else
    echo "PRO 多卡训练失败，请检查错误信息。"
    exit 1
fi
