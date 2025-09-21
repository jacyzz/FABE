#!/bin/bash

# Runner for IST style_replace.py (single-field replacement)

INPUT_PATH="/home/nfs/share-yjy/dachuang2025/data/HumanEval/humaneval_python.jsonl"
OUTPUT_PATH="/home/nfs/share-yjy/dachuang2025/data/HumanEval/poisoned/humaneval_python.jsonl"
LANGUAGE="python"   # c | java | python
CODE_FIELD="canonical_solution"

# Optional
STYLES=""                 # e.g. "-1.1,11.3"
POISON_CANDIDATES=""      # e.g. "-1.1,-3.1,0.5,7.2"
POISON_MIN=1
POISON_MAX=2
AVOID_SIMILAR=true
STRICT_SYNTAX=true
LIMIT=0
SEED="42"
ID_FIELD=""
BACKUP_FIELD=""           # e.g. code_orig
POISON_TAG_FIELD="meta"
LOG_PATH=""

show_help() {
    echo "Usage: $0 -i INPUT -o OUTPUT -l LANG -c CODE_FIELD [options]"
    echo "Options:"
    echo "  -i PATH        input jsonl path"
    echo "  -o PATH        output jsonl path"
    echo "  -l LANG        language: c|java|python"
    echo "  -c FIELD       code field to replace (default: code)"
    echo "  --styles S     fixed styles, comma-separated"
    echo "  --pool S       custom poison pool, comma-separated"
    echo "  --min N        min styles when random (default: 2)"
    echo "  --max N        max styles when random (default: 3)"
    echo "  --avoid B      avoid similar groups: true|false (default: true)"
    echo "  --strict B     strict syntax check: true|false (default: true)"
    echo "  --limit N      limit rows (default: 0 = all)"
    echo "  --seed N       global random seed"
    echo "  --id FIELD     id field to derive per-row seed"
    echo "  --backup FIELD backup original code to this field"
    echo "  --tag FIELD    field to write poison_tag (default: meta)"
  echo "  --log PATH     write per-row JSONL logs"
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -i) INPUT_PATH="$2"; shift 2;;
    -o) OUTPUT_PATH="$2"; shift 2;;
    -l) LANGUAGE="$2"; shift 2;;
    -c) CODE_FIELD="$2"; shift 2;;
    --styles) STYLES="$2"; shift 2;;
    --pool) POISON_CANDIDATES="$2"; shift 2;;
    --min) POISON_MIN="$2"; shift 2;;
    --max) POISON_MAX="$2"; shift 2;;
    --avoid) AVOID_SIMILAR="$2"; shift 2;;
    --strict) STRICT_SYNTAX="$2"; shift 2;;
    --limit) LIMIT="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --id) ID_FIELD="$2"; shift 2;;
    --backup) BACKUP_FIELD="$2"; shift 2;;
    --tag) POISON_TAG_FIELD="$2"; shift 2;;
    --log) LOG_PATH="$2"; shift 2;;
    -h|--help) show_help; exit 0;;
    *) echo "Unknown option: $1"; show_help; exit 1;;
  esac
done

## Python 默认使用安全白名单（若未显式指定 styles 或 pool）
if [[ "$LANGUAGE" == "python" ]]; then
  SAFE_POOL_PY="-3.1,-1.1,-1.2,3.3,3.4,14.1,14.2"
  if [[ -z "$STYLES" && -z "$POISON_CANDIDATES" ]]; then
    POISON_CANDIDATES="$SAFE_POOL_PY"
    POISON_MIN=1
    POISON_MAX=1
  fi
fi

if [[ -z "$INPUT_PATH" || -z "$OUTPUT_PATH" ]]; then
  echo "Error: input/output required"
  show_help; exit 1
fi

cd ..

CMD="conda run -n IST python /home/nfs/u2023-zlb/FABE/IST/style_replace.py"
CMD="$CMD --input_path '$INPUT_PATH'"
CMD="$CMD --output_path '$OUTPUT_PATH'"
CMD="$CMD --language '$LANGUAGE'"
CMD="$CMD --code_field '$CODE_FIELD'"

if [[ -n "$STYLES" ]]; then CMD="$CMD --styles=\"$STYLES\""; fi
if [[ -n "$POISON_CANDIDATES" ]]; then CMD="$CMD --poison_candidates=\"$POISON_CANDIDATES\""; fi
CMD="$CMD --poison_min $POISON_MIN"
CMD="$CMD --poison_max $POISON_MAX"
CMD="$CMD --avoid_similar $AVOID_SIMILAR"
CMD="$CMD --strict_syntax $STRICT_SYNTAX"
CMD="$CMD --limit $LIMIT"
if [[ -n "$SEED" ]]; then CMD="$CMD --seed $SEED"; fi
if [[ -n "$ID_FIELD" ]]; then CMD="$CMD --id_field '$ID_FIELD'"; fi
if [[ -n "$BACKUP_FIELD" ]]; then CMD="$CMD --backup_field '$BACKUP_FIELD'"; fi
if [[ -n "$POISON_TAG_FIELD" ]]; then CMD="$CMD --poison_tag_field '$POISON_TAG_FIELD'"; fi
if [[ -n "$LOG_PATH" ]]; then CMD="$CMD --log_path '$LOG_PATH'"; fi

echo "=== IST Style Replace Runner ==="
echo "Input:    $INPUT_PATH"
echo "Output:   $OUTPUT_PATH"
echo "Language: $LANGUAGE"
echo "Code:     $CODE_FIELD"
echo "Styles:   ${STYLES:-[random]}"
echo "Pool:     ${POISON_CANDIDATES:-[default]}"
echo "Min/Max:  $POISON_MIN/$POISON_MAX"
echo "Avoid:    $AVOID_SIMILAR"
echo "Strict:   $STRICT_SYNTAX"
echo "Limit:    $LIMIT"
echo "Log:      ${LOG_PATH:-[none]}"
echo "Cmd:      $CMD"
echo "================================"

# 确保输出目录存在
mkdir -p "$(dirname "$OUTPUT_PATH")"

eval "$CMD"
RET=$?

if [[ $RET -eq 0 && -f "$OUTPUT_PATH" ]]; then
  LINES=$(wc -l < "$OUTPUT_PATH")
  echo "✅ Success: $LINES rows -> $OUTPUT_PATH"
else
  echo "❌ Failed"
  exit 1
fi


