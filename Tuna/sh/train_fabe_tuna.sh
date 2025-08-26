#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
root_dir=/home/nfs/u2023-zlb/FABE
# 修改为StarCoder-7B模型路径
MODEL=/home/nfs/dachuang/starcoder3b
#MODEL=/home/nfs/share-yjy/local_llm/BigCode/starcoder2-7b
# 使用转换后的数据集
datadir=/home/nfs/u2023-zlb/FABE/Tuna/data/devign_fabe.json
NO_DISCRIMINATE=False
LENPEN=1.0  
savedir=~/FABE/checkpoints/tuna_starcoder7b
NUM_TRAIN_EPOCHS=3
# 针对A100 80G优化的batch size
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
SAVE_STEPS=200
# 针对7B模型调整学习率
LR=5e-6
MLE_WEIGHT=1.0
MARGIN=0.1
BETA1=0.9
BETA2=0.998
WARMUP_STEPS=100
REMOVE_UNUSED_COLUMNS=False

cd ${root_dir}/Tuna/src

torchrun --nproc_per_node=1 train_tuna.py \
    --model_name_or_path ${MODEL} \
    --data_path $datadir \
    --no_discriminate $NO_DISCRIMINATE \
    --lenpen $LENPEN \
    --output_dir $savedir \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit 2 \
    --learning_rate $LR \
    --mle_weight $MLE_WEIGHT \
    --margin $MARGIN \
    --adam_beta1 $BETA1 \
    --adam_beta2 $BETA2 \
    --warmup_steps $WARMUP_STEPS \
    --logging_steps 10 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --log_level debug \
    --remove_unused_columns $REMOVE_UNUSED_COLUMNS \
    --fp16 True \
    --dataloader_num_workers 4 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 2>&1 | tee training_starcoder7b.log
