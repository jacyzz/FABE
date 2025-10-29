#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
# Example usage:
#   bash run.sh \
#     --model-path /path/to/backdoored-model \
#     --data-dir /path/to/dataset \
#     --target-label 1 \
#     --poison-rate 1.0 \
#     --trigger "-1.1" \
#     --use-defense \
#     --llm-model-path /home/nfs/share-yjy/dachuang2025/defense_model/dscoder-6.7b-pro-merged2 \
#     --served-model-name ds_pro \
#     --api-base http://127.0.0.1:8000/v1

python -u run.py "$@"


