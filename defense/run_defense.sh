#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
# One-click backdoor eval + LLM defense with IST poisoning
#
# Example:
#   bash run_defense.sh \
#     --task defect \
#     --model-path /path/to/backdoored-model \
#     --data-dir /path/to/dataset_dir \
#     --target-label 1 \
#     --poison-rate 0.1 \
#     --ist-styles "-1.1" \
#     --served-model-name ds_pro \
#     --api-base http://127.0.0.1:8000/v1 \
#     --run-name defect-p01-ist
#
# Notes:
# - IST is enabled by default (use style codes like -1.1, 7.2, etc.)
# - If IST is not used, the fallback in run.py is simple string injection (append/prepend/wrap)

TASK="defect"
MODEL_PATH=""
BASE_MODEL="/home/nfs/share-yjy/dachuang2025/models/codet5-base"         # e.g., Salesforce/codet5-base or a local HF dir
CHECKPOINT="/home/nfs/share-yjy/dachuang2025/backdoor_models/defect/CodeT5/-1.1_2percent/backdoor_model.bin"         # e.g., backdoor_model.bin
MODEL_INFO="/home/nfs/share-yjy/dachuang2025/backdoor_models/defect/CodeT5/-1.1_2percent/model_info.json"         # optional model_info.json
NUM_LABELS="2"         # override if needed
DATA_DIR="/home/nfs/share-yjy/dachuang2025/data/Defect_Detection/Devign/preprocessed"
FORMAT="jsonl"
DEV_FILE="valid.jsonl"
TEST_FILE="test.jsonl"
INPUT_FIELD="code"
LABEL_FIELD="label"

TARGET_LABEL="1"
POISON_RATE="0.1"
IST_STYLES="-1.1"
IST_LANGUAGE="java"
IST_PATH="/home/nfs/u2023-zlb/FABE/IST"
IST_EXPAND="0"
INJECTION="append"  # only used if IST fails

USE_DEFENSE=1
SERVED_MODEL_NAME="ds_pro"
API_BASE="http://127.0.0.1:8001/v1"
TEMPLATE="code_security_cleanup"
SYSTEM_PROMPT="你是资深代码安全与重构专家。任务：在保持功能等价的前提下，去除/修复代码中的潜在后门，确保可直接替换回原字段。"
MAX_TOKENS=4096
TEMPERATURE=0.1
LLM_MODEL_PATH=""     # optional; if provided, script will attempt to start vLLM automatically via run.py
LLM_PORT=8001
LLM_MAX_LEN=8192
LLM_DTYPE="bfloat16"

OUTPUT_DIR="$(pwd)/outputs"
RUN_NAME=""

DEVICE="cuda"
BATCH_SIZE=16
MAX_LENGTH=512

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --model-info) MODEL_INFO="$2"; shift 2 ;;
    --num-labels) NUM_LABELS="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --format) FORMAT="$2"; shift 2 ;;
    --dev-file) DEV_FILE="$2"; shift 2 ;;
    --test-file) TEST_FILE="$2"; shift 2 ;;
    --input-field) INPUT_FIELD="$2"; shift 2 ;;
    --label-field) LABEL_FIELD="$2"; shift 2 ;;

    --target-label) TARGET_LABEL="$2"; shift 2 ;;
    --poison-rate) POISON_RATE="$2"; shift 2 ;;
    --ist-styles) IST_STYLES="$2"; shift 2 ;;
    --ist-language) IST_LANGUAGE="$2"; shift 2 ;;
    --ist-path) IST_PATH="$2"; shift 2 ;;
    --ist-expand) IST_EXPAND="$2"; shift 2 ;;
    --injection) INJECTION="$2"; shift 2 ;;

    --no-defense) USE_DEFENSE=0; shift 1 ;;
    --served-model-name) SERVED_MODEL_NAME="$2"; shift 2 ;;
    --api-base) API_BASE="$2"; shift 2 ;;
    --template) TEMPLATE="$2"; shift 2 ;;
    --system-prompt) SYSTEM_PROMPT="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --llm-model-path) LLM_MODEL_PATH="$2"; shift 2 ;;
    --llm-port) LLM_PORT="$2"; shift 2 ;;
    --llm-max-len) LLM_MAX_LEN="$2"; shift 2 ;;
    --llm-dtype) LLM_DTYPE="$2"; shift 2 ;;

    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;

    --device) DEVICE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --max-length) MAX_LENGTH="$2"; shift 2 ;;

    *) echo "[ERR] Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Model source validation: either --model-path (HF dir) OR --checkpoint+--base-model
if [[ -z "${MODEL_PATH}" ]]; then
  if [[ -z "${CHECKPOINT}" || -z "${BASE_MODEL}" ]]; then
    echo "[ERR] Provide either --model-path (HF dir) OR --checkpoint and --base-model" >&2
    exit 1
  fi
fi
if [[ -z "${DATA_DIR}" ]]; then
  echo "[ERR] --data-dir is required" >&2
  exit 1
fi
if [[ -z "${TARGET_LABEL}" ]]; then
  echo "[ERR] --target-label is required" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

CMD=(python -u run.py \
  --task "${TASK}" \
  --model-path "${MODEL_PATH}" \
  --base-model "${BASE_MODEL}" \
  --checkpoint "${CHECKPOINT}" \
  --model-info "${MODEL_INFO}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --max-length "${MAX_LENGTH}" \
  --data-dir "${DATA_DIR}" \
  --format "${FORMAT}" \
  --dev-file "${DEV_FILE}" \
  --test-file "${TEST_FILE}" \
  --input-field "${INPUT_FIELD}" \
  --label-field "${LABEL_FIELD}" \
  --poison-rate "${POISON_RATE}" \
  --trigger "${IST_STYLES}" \
  --target-label "${TARGET_LABEL}" \
  --injection "${INJECTION}" \
  --use-ist \
  --ist-language "${IST_LANGUAGE}" \
  --ist-path "${IST_PATH}" \
  --ist-styles "${IST_STYLES}" \
  --ist-expand "${IST_EXPAND}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --api-base "${API_BASE}" \
  --template "${TEMPLATE}" \
  --system-prompt "${SYSTEM_PROMPT}" \
  --max-tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --llm-model-path "${LLM_MODEL_PATH}" \
  --llm-port "${LLM_PORT}" \
  --llm-max-len "${LLM_MAX_LEN}" \
  --llm-dtype "${LLM_DTYPE}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME}")

# Optional args appended safely (avoid set -e issues with command substitutions)
if [[ -n "${NUM_LABELS}" ]]; then
  CMD+=( --num-labels "${NUM_LABELS}" )
fi
if [[ ${USE_DEFENSE} -eq 1 ]]; then
  CMD+=( --use-defense )
fi

echo "[INFO] Running: ${CMD[*]}" >&2
"${CMD[@]}"


