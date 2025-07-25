export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0
root_dir=/home/nfs/u2023-zlb/FABE/

#stage 23
id=$1
data_path=$2
ranking_len=$3
python main.py \
    --task hh \
    --train_file_path $root_dir/PRO/data/${data_path} \
    --validation_file_path $root_dir/PRO/data/hh_dev \
    --validation_file_name sampled_dev.json \
    --output_dir $root_dir/checkpoints/pro_starcoder3b/\
    --log_path $root_dir/PRO/ \
    --seed 42 \
    --temperature 1 \
    --sft_weight 0.05 \
    --num_train_epochs 2 \
    --index $id \
    --training_stage_num $ranking_len \
    --block_size 2048 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --model_name_or_path /home/nfs/dachuang/starcoder3b \
    --do_train \
    --do_validation > $root_dir/PRO/train_detail.log 2>&1