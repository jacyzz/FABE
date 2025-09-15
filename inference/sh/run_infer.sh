#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2
###############################################
# 用法与参数（全部）
# 必填：
#   -m MODEL           模型路径或ID（可逗号分隔多个）
# 常用：
#   -i INPUT           输入文件或文件夹（默认: examples/sample.jsonl）
#   -o OUTPUT          输出根目录（默认: outputs）
#   -d DEVICE          设备 cpu|cuda（默认: cpu）
#   -l LOCAL_ONLY      仅本地 1/0（默认: 1）
#   -s USE_CHAT        自动模板 1/0（默认: 1）
#   -n MAX_NEW         生成token上限（默认: 128）
#   -t TEMP            温度（默认: 0.0）
# 进阶：
#   -g GLOB            当 INPUT 为文件夹时的匹配（如 **/*.jsonl）
#   -I INPLACE         就地覆盖 1/0（默认: 0）
#   -F FIELD           目标字段（默认: code）
#   -B BACKUP          备份字段（默认: original_code）
#   -P PROVIDER        提供器 hf|ollama|openai|echo|modelscope（默认: hf）
#   -C CONCURRENCY     并发（默认: 1）
#   -S BATCH_SIZE      批处理大小（默认: 8）
#   -R RETRY           重试次数（默认: 0）
#   -T TOP_P           nucleus sampling（可选）
#   -D DO_SAMPLE       显式采样 1/0（默认: 空，按温度判定）
#   -U BASE_URL        远程服务 base_url（ollama/openai）
#   -K API_KEY         远程服务 api_key（openai）
#   -X INSTRUCTION     指令文本或模板路径（默认: examples/prompts/refactor_python.j2）
#   -Z STRIP           抽取代码块 1/0（默认: 1）
# 环境变量：
#   DRY_RUN=1          仅检查流程不真正生成
# 示例：
#   sh/run_infer.sh -m /path/to/model -i examples/auto_template_test.jsonl -o outputs -d cuda -l 1 -s 1 -n 128 -t 0.0
###############################################

# 基本路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 参数变量（默认值，可被命令行覆盖）
MODEL="/home/nfs/share-yjy/dachuang2025/models/deepseek-coder-6.7b-instruct"
INPUT="/home/nfs/u2023-zlb/FABE/inference/examples/-3.1_test.jsonl"
OUTPUT="${BASE_DIR}/outputs"
DEVICE="cuda"
LOCAL_ONLY="1"
USE_CHAT="1"
MAX_NEW=2048
TEMP=0.0
GLOB=""
INPLACE="0"
FIELD="code1"
BACKUP="original_code"
PROVIDER="modelscope"
CONCURRENCY="1"
BATCH_SIZE="4"
RETRY="0"
TOP_P=""
DO_SAMPLE=""
BASE_URL=""
API_KEY=""
INSTRUCTION="${BASE_DIR}/examples/prompts/ds.j2"
STRIP="1"

# 解析命令行参数
while getopts ":m:i:o:d:l:s:n:t:g:I:F:B:P:C:S:R:T:D:U:K:X:Z:h" opt; do
  case "$opt" in
    m) MODEL="$OPTARG" ;;
    i) INPUT="$OPTARG" ;;
    o) OUTPUT="$OPTARG" ;;
    d) DEVICE="$OPTARG" ;;
    l) LOCAL_ONLY="$OPTARG" ;;
    s) USE_CHAT="$OPTARG" ;;
    n) MAX_NEW="$OPTARG" ;;
    t) TEMP="$OPTARG" ;;
    g) GLOB="$OPTARG" ;;
    I) INPLACE="$OPTARG" ;;
    F) FIELD="$OPTARG" ;;
    B) BACKUP="$OPTARG" ;;
    P) PROVIDER="$OPTARG" ;;
    C) CONCURRENCY="$OPTARG" ;;
    S) BATCH_SIZE="$OPTARG" ;;
    R) RETRY="$OPTARG" ;;
    T) TOP_P="$OPTARG" ;;
    D) DO_SAMPLE="$OPTARG" ;;
    U) BASE_URL="$OPTARG" ;;
    K) API_KEY="$OPTARG" ;;
    X) INSTRUCTION="$OPTARG" ;;
    Z) STRIP="$OPTARG" ;;
    h)
      sed -n '1,60p' "$0" # 打印顶部说明
      exit 0 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 2;;
    \?) echo "Unknown option: -$OPTARG" >&2; exit 2;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "[error] 必须提供 -m MODEL" >&2
  exit 2
fi

mkdir -p "$OUTPUT"

# =============================================
# 执行命令（清晰直观）
# 支持多模型（逗号分隔）
# =============================================

IFS=',' read -r -a MODELS_ARR <<< "$MODEL"
for M in "${MODELS_ARR[@]}"; do
  echo "==> Running model: $M"
  python "$BASE_DIR/run_batch_infer.py" \
    --provider "$PROVIDER" \
    --input-path "$INPUT" \
    --output-path "$OUTPUT" \
    $( [[ -n "$GLOB" ]] && echo "--input-glob '$GLOB'" ) \
    --field "$FIELD" \
    --backup-field "$BACKUP" \
    --instruction "$INSTRUCTION" \
    --max-new-tokens "$MAX_NEW" \
    --temperature "$TEMP" \
    --device "$DEVICE" \
    --local-files-only "$LOCAL_ONLY" \
    $( [[ -n "$TOP_P" ]] && echo "--top-p '$TOP_P'" ) \
    $( [[ -n "$DO_SAMPLE" ]] && echo "--do-sample '$DO_SAMPLE'" ) \
    --concurrency "$CONCURRENCY" \
    --batch-size "$BATCH_SIZE" \
    --retry "$RETRY" \
    $( [[ -n "$BASE_URL" ]] && echo "--base-url '$BASE_URL'" ) \
    $( [[ -n "$API_KEY" ]] && echo "--api-key '$API_KEY'" ) \
    $( [[ "$USE_CHAT" == "1" ]] && echo "--use-chat-template" || echo "--no-use-chat-template" ) \
    $( [[ "$STRIP" == "1" ]] && echo "--strip-fences --auto-extract-code" || echo "--no-strip-fences --no-auto-extract-code" ) \
    $( [[ "$INPLACE" == "1" ]] && echo "--inplace" ) \
    $( [[ "${DRY_RUN:-0}" == "1" ]] && echo "--dry-run" ) \
    --model "$M"
done
