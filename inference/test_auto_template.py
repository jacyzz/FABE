#!/usr/bin/env python3
"""
Quick A/B test: compare auto chat_template ON vs OFF for HF models.
Assumes the user sets MODEL_PATH in run_batch_infer.py or passes via env.
"""
from __future__ import annotations
import os
import sys

BASE_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from fabe_infer.config import AppConfig, ProviderConfig, IOConfig, PromptConfig, ExecConfig
from fabe_infer.runner import run_pipeline


def build_cfg(use_chat_template: bool, outdir: str) -> AppConfig:
    # Input dataset with weak/insecure code
    input_path = os.path.join(BASE_DIR, "examples", "auto_template_test.jsonl")
    instruction = os.path.join(BASE_DIR, "examples", "prompts", "refactor_python.j2")

    provider = ProviderConfig(
        name="hf",
        model=os.environ.get("MODEL_PATH", "/home/nfs/share-yjy/dachuang2025/models/deepseek-coder-6.7b-instruct"),
        params={
            "device": os.environ.get("DEVICE", "cpu"),
            "local_files_only": os.environ.get("LOCAL_FILES_ONLY", "1") not in ("0", "false", "False"),
            "max_new_tokens": int(os.environ.get("MAX_NEW_TOKENS", "128")),
            "temperature": float(os.environ.get("TEMPERATURE", "0.0")),
            "use_chat_template": use_chat_template,
            "system_prompt": os.environ.get(
                "SYSTEM_PROMPT",
                "You are a coding assistant. Output only the final code.",
            ),
        },
    )
    io = IOConfig(
        input_path=input_path,
        input_glob=None,
        output_path=os.path.join(BASE_DIR, "outputs", outdir),
        inplace=False,
        field="code",
        backup_field="original_code",
    )
    prompt = PromptConfig(
        instruction=instruction,
        template_vars={},
        strip_fences=True,
        auto_extract_code=True,
    )
    ex = ExecConfig(
        concurrency=int(os.environ.get("CONCURRENCY", "1")),
        retry=int(os.environ.get("RETRY", "0")),
        cache_dir=os.path.join(BASE_DIR, ".cache"),
        resume=True,
        dry_run=False,
    )
    return AppConfig(provider=provider, io=io, prompt=prompt, exec=ex)


def main():
    # A/B: auto ON
    cfg_auto = build_cfg(True, "auto_template_on")
    run_pipeline(cfg_auto)

    # A/B: auto OFF
    cfg_manual = build_cfg(False, "auto_template_off")
    run_pipeline(cfg_manual)


if __name__ == "__main__":
    main()
