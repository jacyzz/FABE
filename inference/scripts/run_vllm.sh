#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
set -euo pipefail

# 用法：
#   bash scripts/run_vllm.sh \
#     /home/nfs/share-yjy/dachuang2025/defense_model/dscoder-6.7b-pro-merged2 \
#     my-dscoder \
#     8000 \
#     8192 \
#     bf16
#
# 参数：
#   $1: 本地模型路径（HF目录或权重目录）
#   $2: 对外服务别名（served-model-name）
#   $3: 端口，默认 8000
#   $4: 上下文长度（max-model-len），默认 8192
#   $5: 精度（bf16/fp16/int8/auto），默认 bf16

MODEL_PATH=${1:-""}
ALIAS=${2:-"local-model"}
PORT=${3:-8000}
MAX_LEN=${4:-8192}
DTYPE=${5:-bf16}

if [[ -z "${MODEL_PATH}" ]]; then
  echo "[ERR] 需要提供本地模型路径作为第一个参数" >&2
  exit 1
fi

echo "[INFO] 启动 vLLM OpenAI 兼容服务:" >&2
echo "       model: ${MODEL_PATH}" >&2
echo "       alias: ${ALIAS}" >&2
echo "       port : ${PORT}" >&2
echo "       max-len: ${MAX_LEN}, dtype: ${DTYPE}" >&2

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${ALIAS}" \
  --port "${PORT}" \
  --max-model-len "${MAX_LEN}" \
  --dtype "${DTYPE}"


