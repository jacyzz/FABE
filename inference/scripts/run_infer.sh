#!/usr/bin/env bash
set -euo pipefail

# 简单变量配置
INPUT="/home/nfs/share-yjy/dachuang2025/data/BigCloneBench/poisoned/bigclonebench_-1.1_2percent_poisoned_data.jsonl"
OUTPUT="/home/nfs/share-yjy/dachuang2025/data/BigCloneBench/fixed/bigclonebench_-1.1_2percent_poisoned_data.jsonl"
FIELD="func"
TEMPLATE="code_security_cleanup"
MODEL="dscoder67b"                   # 与 vLLM 启动脚本里的 --served-model-name 保持一致
API_BASE="http://127.0.0.1:8000/v1"  # vLLM OpenAI 兼容地址
MAX_TOKENS=4096
TEMPERATURE=0.1

# 系统提示词（可直接修改为你需要的内容）
SYSTEM_PROMPT='你是资深代码安全与重构专家。任务：在保持功能等价的前提下，去除/修复代码中的潜在后门，确保可直接替换回原字段。'

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


