#!/usr/bin/env bash
set -euo pipefail

# 基本参数（按需修改）
INPUT="/home/nfs/share-yjy/dachuang2025/codefuse-evaluation/codefuseEval_202503/data/code_completion/IST_eval/humaneval_python.jsonl"
# 预测输出（predictions.jsonl，每行一个候选 {task_id, language, sample_id, completion}）
OUTPUT="/home/nfs/u2023-zlb/CausalCodeDefense/src/IST/data/code_completion/model_fix/python/predictions.jsonl"
FIELD="canonical_solution"
TEMPLATE="/home/nfs/u2023-zlb/FABE/inference/templates/code_security_cleanup.yaml"
MODEL="ds_pro"
API_BASE="http://127.0.0.1:8001/v1"

# 多样化束搜索与采样参数
N_SAMPLES=${N_SAMPLES:-4}
USE_BEAM=${USE_BEAM:-1}               # 1 开启多样化束搜索，0 仅 n-best 采样
NUM_BEAMS=${NUM_BEAMS:-8}
NUM_GROUPS=${NUM_GROUPS:-4}
DIVERSITY=${DIVERSITY:-0.5}
NO_REPEAT_NGRAM=${NO_REPEAT_NGRAM:-5}
LENGTH_PENALTY=${LENGTH_PENALTY:-1.0}
MAX_TOKENS=${MAX_TOKENS:-1024}
TEMP=${TEMP:-0.7}
TOP_P=${TOP_P:-0.95}

# 结果清洗（去 <think>/围栏/注释）
STRIP_THINK=${STRIP_THINK:-1}
STRIP_FENCES=${STRIP_FENCES:-1}
STRIP_COMMENTS=${STRIP_COMMENTS:-1}
CODE_LANG=${CODE_LANG:-python}

python -m diverse_infer \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --field "$FIELD" \
  --template "$TEMPLATE" \
  --model "$MODEL" \
  --api-base "$API_BASE" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMP" \
  --top-p "$TOP_P" \
  --n-samples "$N_SAMPLES" \
  $( [[ "$USE_BEAM" == "1" ]] && echo --use-beam-search ) \
  --num-beams "$NUM_BEAMS" \
  --num-beam-groups "$NUM_GROUPS" \
  --diversity-penalty "$DIVERSITY" \
  --no-repeat-ngram-size "$NO_REPEAT_NGRAM" \
  --length-penalty "$LENGTH_PENALTY" \
  $( [[ "$STRIP_THINK" == "1" ]] && echo --strip-think ) \
  $( [[ "$STRIP_FENCES" == "1" ]] && echo --strip-fences ) \
  $( [[ "$STRIP_COMMENTS" == "1" ]] && echo --strip-comments ) \
  --lang "$CODE_LANG"

echo "[DONE] diverse inference -> $OUTPUT"


