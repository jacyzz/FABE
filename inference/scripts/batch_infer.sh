#!/usr/bin/env bash
set -euo pipefail

# 输入/输出目录（修改为你的目录）
INPUT_DIR="/home/nfs/share-yjy/dachuang2025/data/BigCloneBench/eval"
OUTPUT_DIR="/home/nfs/share-yjy/dachuang2025/data/BigCloneBench/fixed"

# 字段与推理配置
FIELD="func"
TEMPLATE="code_security_cleanup"
MODEL="ds_pro"                   # 与 vLLM 启动脚本里的 --served-model-name 保持一致
API_BASE="http://127.0.0.1:8000/v1"  # vLLM OpenAI 兼容地址
MAX_TOKENS=4096
TEMPERATURE=0.1

# 系统提示词
SYSTEM_PROMPT='你是资深代码安全与重构专家。任务：在保持功能等价的前提下，去除/修复代码中的潜在后门，确保可直接替换回原字段。'

# 准备输出目录
mkdir -p "$OUTPUT_DIR"

# 检查是否存在 .jsonl
if ! compgen -G "$INPUT_DIR/*.jsonl" >/dev/null; then
  echo "No .jsonl files found in: $INPUT_DIR" >&2
  exit 1
fi

# 逐个文件处理
for INPUT in "$INPUT_DIR"/*.jsonl; do
  BN="$(basename "$INPUT")"
  OUTPUT="$OUTPUT_DIR/$BN"
  echo "Processing: $BN -> $OUTPUT"

  python -m llm_infer.cli \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --field "$FIELD" \
    --template "$TEMPLATE" \
    --system-prompt "$SYSTEM_PROMPT" \
    --model "$MODEL" \
    --api-base "$API_BASE" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE"
done

echo "All done. Outputs in: $OUTPUT_DIR"
