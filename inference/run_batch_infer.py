#!/usr/bin/env python3
"""
Standalone batch inference runner (no CLI required).
Edit the config variables below and run: python run_batch_infer.py
"""
from __future__ import annotations
import os
import sys
from typing import Dict, Any, List
import argparse
import json


# Make local package importable without installing
BASE_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from fabe_infer.config import AppConfig, ProviderConfig, IOConfig, PromptConfig, ExecConfig
from fabe_infer.runner import run_pipeline

def str2bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch inference runner (arg-driven)")
    # I/O
    p.add_argument("--input-path", default=os.path.join(BASE_DIR, "examples", "sample.jsonl"), help="输入文件或文件夹")
    p.add_argument("--input-glob", default=None, help="当输入为文件夹时的glob模式，例如 **/*.jsonl")
    p.add_argument("--output-path", default=os.path.join(BASE_DIR, "outputs"), help="输出根目录（非inplace模式使用）")
    p.add_argument("--inplace", default=False, action="store_true", help="就地覆盖输入文件")
    p.add_argument("--field", default="code", help="需要被替换的字段名")
    p.add_argument("--backup-field", default="original_code", help="备份原内容的字段名（不存在时写入）")

    # Prompt
    p.add_argument("--instruction", default=os.path.join(BASE_DIR, "examples", "prompts", "refactor_python.j2"), help="纯文本指令或模板路径")
    p.add_argument("--template-vars-json", default="{}", help="Jinja2模板变量(JSON字符串)")
    p.add_argument("--strip-fences", dest="strip_fences", action="store_true", default=True, help="抽取首个代码块或去除空白")
    p.add_argument("--no-strip-fences", dest="strip_fences", action="store_false")
    p.add_argument("--auto-extract-code", dest="auto_extract_code", action="store_true", default=True, help="同strip_fences，保留向后兼容")
    p.add_argument("--no-auto-extract-code", dest="auto_extract_code", action="store_false")

    # Model (allow multiple via repeated --model)
    p.add_argument("--provider", default="hf", help="提供器：hf|ollama|openai|echo|modelscope")
    p.add_argument("--model", action="append", help="模型路径或ID，可重复传入以批量运行；若未提供则使用环境变量MODEL_PATH")
    p.add_argument("--device", default=os.environ.get("DEVICE", "cpu"))
    p.add_argument("--local-files-only", dest="local_files_only", default=os.environ.get("LOCAL_FILES_ONLY", "1"), help="是否仅本地文件(1/0)")
    p.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", "16")))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.0")))
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--do-sample", dest="do_sample", default=None, help="显式控制采样(1/0)，默认随temperature自动判定")
    p.add_argument("--use-chat-template", dest="use_chat_template", action="store_true", default=True, help="使用模型chat_template渲染")
    p.add_argument("--no-use-chat-template", dest="use_chat_template", action="store_false")
    p.add_argument("--system-prompt", default="You are a coding assistant. Output only the final code.")
    p.add_argument("--base-url", default=None, help="远程服务base_url（ollama/openai兼容）")
    p.add_argument("--api-key", default=None, help="远程服务api_key（openai兼容）")

    # Exec
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8, help="批处理大小")
    p.add_argument("--retry", type=int, default=1)
    p.add_argument("--cache-dir", default=os.path.join(BASE_DIR, ".cache"))
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--dry-run", dest="dry_run", action="store_true", default=False)
    p.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    return p.parse_args()

def build_app_config(spec: Dict[str, Any], args: argparse.Namespace) -> AppConfig:
    provider = ProviderConfig(
        name=spec.get("provider", "echo"),
        model=spec.get("model", "local"),
        base_url=spec.get("base_url"),
        api_key=spec.get("api_key"),
        params=spec.get("params", {}),
    )
    io = IOConfig(
        input_path=args.input_path,
        input_glob=args.input_glob,
        output_path=args.output_path,
        inplace=bool(args.inplace),
        field=args.field,
        backup_field=args.backup_field,
    )
    prompt = PromptConfig(
        instruction=args.instruction,
        template_vars=_parse_template_vars(args.template_vars_json),
        strip_fences=bool(args.strip_fences),
        auto_extract_code=bool(args.auto_extract_code),
    )
    ex = ExecConfig(
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        retry=args.retry,
        cache_dir=args.cache_dir,
        resume=bool(args.resume),
        dry_run=bool(args.dry_run),
    )
    return AppConfig(provider=provider, io=io, prompt=prompt, exec=ex)


def _parse_template_vars(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s or "{}")
    except Exception:
        return {}


def main():
    args = parse_args()

    # Build model specs list based on args
    models: List[str] = args.model or []
    if not models:
        env_model = os.environ.get("MODEL_PATH")
        if env_model:
            models = [env_model]
        else:
            # 兼容旧默认：示例模型路径（请尽量传参）
            models = ["/home/nfs/share-yjy/dachuang2025/models/starcoder3b"]

    specs: List[Dict[str, Any]] = []
    for m in models:
        params: Dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "device": args.device,
            "local_files_only": str2bool(args.local_files_only),
            "use_chat_template": bool(args.use_chat_template),
            "system_prompt": args.system_prompt,
        }
        if args.top_p is not None:
            params["top_p"] = args.top_p
        if args.do_sample is not None:
            params["do_sample"] = str2bool(args.do_sample)

        specs.append({
            "provider": args.provider,
            "model": m,
            "base_url": args.base_url,
            "api_key": args.api_key,
            "params": params,
        })

    for i, spec in enumerate(specs, 1):
        print(f"=== Running model {i}/{len(specs)}: {spec['provider']}::{spec['model']} ===")
        # 对本地目录做一个轻校验（若是远程ID则跳过）
        if spec.get("provider") in {"hf", "hf_local", "transformers", "local", "modelscope", "ms"}:
            m = str(spec.get("model", ""))
            if m.startswith("/") and not os.path.isdir(m):
                print(f"[error] 模型目录不存在: {m}")
                return
        cfg = build_app_config(spec, args)
        # write outputs to model-specific subfolder when not inplace
        if not cfg.io.inplace and cfg.io.output_path:
            model_dir_name = f"{cfg.provider.name}-{cfg.provider.model}".replace("/", "_").replace(":", "_")
            cfg.io.output_path = os.path.join(args.output_path, model_dir_name)
            os.makedirs(cfg.io.output_path, exist_ok=True)
        run_pipeline(cfg)


if __name__ == "__main__":
    main()
