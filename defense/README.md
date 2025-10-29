# FABE Defense Evaluation

This evaluation harness mirrors BackdoorDefense flow and integrates LLM-based dataset cleaning (via vLLM + llm_infer).

## Features
- Load JSONL/CSV datasets with configurable input/label fields
- Generate poisoned evaluation set from test split (trigger injection + target label)
- Optional defense: clean datasets with your vLLM-served model using scripts under FABE/inference
- Evaluate a HuggingFace sequence classifier victim and compute ACC/ASR/CASR

## Quick Start

1) Start vLLM (optional, only if using defense cleaning):
```bash
bash /home/nfs/u2023-zlb/FABE/inference/scripts/run_vllm.sh \
  /home/nfs/share-yjy/dachuang2025/defense_model/dscoder-6.7b-pro-merged2 \
  ds_pro 8000 8192 bf16
```

2) Run evaluation:
```bash
bash run.sh \
  --model-path /path/to/backdoored-model \
  --data-dir /path/to/dataset \
  --format jsonl \
  --test-file test.jsonl \
  --dev-file dev.jsonl \
  --input-field canonical_solution \
  --label-field label \
  --target-label 1 \
  --poison-rate 1.0 \
  --trigger "-1.1" \
  --use-defense \
  --llm-model-path /home/nfs/share-yjy/dachuang2025/defense_model/dscoder-6.7b-pro-merged2 \
  --served-model-name ds_pro \
  --api-base http://127.0.0.1:8000/v1
```

Outputs are saved under `outputs/<run_id>/` including cleaned/poisoned artifacts and `metrics.json` with:
- acc: accuracy on clean test set
- asr: attack success rate on poisoned test set (predict target label)
- casr: clean attack success rate on clean test set (predict target label)

## Notes
- `attacker/poisoner.py` injects trigger into text and flips label to `--target-label` for poisoned items.
- Defense cleaning invokes your `run_infer.sh`; if the wrapper ignores env vars, direct `llm_infer.cli` fallback is used.
- IST resources can be copied to this project as needed and referenced by dataset files.
