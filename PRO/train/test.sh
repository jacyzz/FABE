export CUDA_VISIBLE_DEVICES=2

python /home/nfs/u2023-zlb/FABE/PRO/train/main.py \
  --task coding \
  --do_train \
  --model_template deepseek_clean_sys \
  --model_name_or_path /home/nfs/share-yjy/dachuang2025/models/deepseek-coder-6.7b-instruct \
  --train_file_path /home/nfs/u2023-zlb/FABE/PRO/example/test.jsonl \
  --per_device_train_batch_size 2 \
  --learning_rate 3e-5 \
  --block_size 1536 \
  --num_train_epochs 1 \
  --gradient_accumulation_steps 8 \
  --output_dir /home/nfs/u2023-zlb/FABE/checkpoints/pro-clean-mini \
  --log_path /home/nfs/u2023-zlb/FABE/logs \
  --checkpointing_step 50 \
  --save_total_limit 2 \
  --training_stage_num 3 \
  --sft_weight 1.5 \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules "q_proj" "v_proj" \
  --bf16 \
  --seed 42