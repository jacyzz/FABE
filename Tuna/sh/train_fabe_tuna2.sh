#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
root_dir=/home/nfs/u2023-zlb/FABE
MODEL=/home/nfs/share-yjy/local_llm/BigCode/starcoder2-7b
datadir=/home/nfs/u2023-zlb/FABE/Tuna/data/devign_fabe.json
savedir=/home/nfs/share-yjy/dachuang2025/defense_model/tuna_starcoder7b_optimized

# --- 优化后的核心参数 ---
# 增加单设备批次大小，减少梯度累积步骤，以提升速度
PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=2
# 假设你将使用 LoRA，大幅提高学习率
LR=2e-4
# -------------------------

NUM_TRAIN_EPOCHS=3
SAVE_STEPS=200
# ... 其他参数保持不变 ...
WARMUP_STEPS=100
REMOVE_UNUSED_COLUMNS=False # 最好设置为True，除非你的脚本有特殊需求

cd ${root_dir}/Tuna/src

# 注意 --bf16 和 --optim 参数
torchrun --nproc_per_node=1 train_tuna.py \
    --model_name_or_path ${MODEL} \
    --data_path $datadir \
    --output_dir $savedir \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit 2 \
    --learning_rate $LR \
    --warmup_steps $WARMUP_STEPS \
    --logging_steps 10 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --remove_unused_columns True \
    --bf16 True \
    --dataloader_num_workers 4 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 2>&1 | tee training_starcoder7b.log
