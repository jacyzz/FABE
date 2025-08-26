#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

# --- Best Practices: Define project paths clearly ---
# Using a central project root makes the script more portable.
PROJECT_ROOT="/home/nfs/u2023-zlb/FABE"

# --- Model & Data Configuration ---
MODEL_PATH="/home/nfs/share-yjy/local_llm/BigCode/starcoder2-7b"
DATA_PATH="${PROJECT_ROOT}/Tuna/data/devign_fabe.json"
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/starcoder2-7b-tuna-optimized"

# Create output directory if it doesn't exist to prevent errors
mkdir -p ${OUTPUT_DIR}

# --- Core Training Hyperparameters ---
# Increased batch size to better utilize 80GB VRAM.
# The effective batch size is now 64 (8 * 8), which is much more stable.
# You can experiment with increasing PER_DEVICE_TRAIN_BATCH_SIZE further to 12 or 16.
# Find the maximum that fits in memory, then adjust GRADIENT_ACCUMULATION_STEPS accordingly.
PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=8
NUM_TRAIN_EPOCHS=2
# A more standard and effective learning rate for fine-tuning 7B models.
LEARNING_RATE=2e-5
# Using a ratio for warmup is more flexible than a fixed number of steps.
WARMUP_RATIO=0.05

# --- Optimizer & Scheduler ---
ADAM_BETA1=0.9
# Using the standard, recommended value for Adam's beta2.
ADAM_BETA2=0.999
LR_SCHEDULER_TYPE="cosine"

# --- Tuna-Specific Hyperparameters (from original script) ---
NO_DISCRIMINATE=False
LENPEN=1.0
MLE_WEIGHT=1.0
MARGIN=0.1

# --- Logging & Saving ---
SAVE_STEPS=500
# It's good practice to save more than just the last checkpoint.
SAVE_TOTAL_LIMIT=2
LOGGING_STEPS=10

# --- Execution ---
cd "${PROJECT_ROOT}/Tuna/src"

# The `deepspeed` command can sometimes offer better performance and memory management.
# If you have it configured, you can use it. Otherwise, `python` is fine.
# For now, we'll stick with the original python command.

echo "Starting training..."
python train_tuna.py \
    --model_name_or_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --warmup_ratio ${WARMUP_RATIO} \
    --adam_beta1 ${ADAM_BETA1} \
    --adam_beta2 ${ADAM_BETA2} \
    \
    --no_discriminate ${NO_DISCRIMINATE} \
    --lenpen ${LENPEN} \
    --mle_weight ${MLE_WEIGHT} \
    --margin ${MARGIN} \
    \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --logging_steps ${LOGGING_STEPS} \
    --report_to "tensorboard" \
    --log_level "info" \
    \
    --bf16 True \
    --use_flash_attention_2 True \
    --gradient_checkpointing True \
    --remove_unused_columns ${REMOVE_UNUSED_COLUMNS} 2>&1 | tee "${OUTPUT_DIR}/training.log"

echo "Training finished."