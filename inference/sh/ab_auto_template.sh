#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $(basename "$0") MODEL_PATH [DEVICE]" >&2
  exit 2
fi

MODEL="$1"
DEVICE="${2:-cpu}"

INPUT="${BASE_DIR}/examples/auto_template_test.jsonl"
OUT_ON="${BASE_DIR}/outputs/auto_template_on"
OUT_OFF="${BASE_DIR}/outputs/auto_template_off"

mkdir -p "$OUT_ON" "$OUT_OFF"

echo "[A] Auto template = ON"
"${SCRIPT_DIR}/run_infer.sh" -m "$MODEL" -i "$INPUT" -o "$OUT_ON" -d "$DEVICE" -l 1 -n 128 -t 0.0 -c 1 -r 0 -s 1

echo "[B] Auto template = OFF"
"${SCRIPT_DIR}/run_infer.sh" -m "$MODEL" -i "$INPUT" -o "$OUT_OFF" -d "$DEVICE" -l 1 -n 128 -t 0.0 -c 1 -r 0 -s 0

echo "Done. See outputs:"
echo "  ON : $OUT_ON"
echo "  OFF: $OUT_OFF"
