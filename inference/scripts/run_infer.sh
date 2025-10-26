#!/usr/bin/env bash
set -euo pipefail

# 简单变量配置
INPUT="/home/nfs/share-yjy/dachuang2025/codefuse-evaluation/codefuseEval_202503/data/code_completion/IST_eval/humaneval_python.jsonl"
OUTPUT="/home/nfs/u2023-zlb/CausalCodeDefense/src/IST/data/code_completion/model_fix/python/predictions.jsonl"
FIELD="canonical_solution"
TEMPLATE="code_security_cleanup"
MODEL="ds_pro"                   # 与 vLLM 启动脚本里的 --served-model-name 保持一致
API_BASE="http://127.0.0.1:8001/v1"  # vLLM OpenAI 兼容地址
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


