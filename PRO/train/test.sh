export CUDA_VISIBLE_DEVICES=2

python /home/nfs/share-yjy/dachuang2025/FABE/PRO/train/main.py \
  --task coding \
  --do_train \
  --model_template deepseek_clean_sys \
  --model_name_or_path /home/nfs/share-yjy/dachuang2025/models/deepseek-coder-6.7b-instruct \
  --train_file_path  "/home/nfs/share-yjy/dachuang2025/data/fabe/PRO/cd_train1.jsonl" \
  --per_device_train_batch_size 2 \
  --learning_rate 2e-4 \
  --block_size 1536 \
  --num_train_epochs 3 \
  --gradient_accumulation_steps 8 \
  --output_dir /home/nfs/share-yjy/dachuang2025/defense_model/pro-clean-6.7b \
  --log_path logs/ \
  --checkpointing_step 500 \
  --save_total_limit 3 \
  --training_stage_num 3 \
  --sft_weight 1.5 \
  --use_lora \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --lora_target_modules "q_proj" "v_proj" \
  --use_4bit \
  --bnb_4bit_quant_type nf4 \
  --bnb_4bit_compute_dtype bfloat16 \
  --bnb_4bit_use_double_quant \
  --bf16 \
  --seed 42