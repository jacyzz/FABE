from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    import orjson as json
except Exception:  # pragma: no cover
    import json  # type: ignore

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class ProviderConfig:
    name: str = "echo"  # echo|ollama|openai
    model: str = "local"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IOConfig:
    input_path: str = ""
    input_glob: Optional[str] = None
    output_path: Optional[str] = None
    inplace: bool = False
    field: str = "code"
    backup_field: Optional[str] = None


@dataclass
class PromptConfig:
    instruction: str = ""
    template_vars: Dict[str, Any] = field(default_factory=dict)
    strip_fences: bool = False
    auto_extract_code: bool = False


@dataclass
class ExecConfig:
    concurrency: int = 4
    retry: int = 3
    cache_dir: str = ".cache"
    resume: bool = True
    dry_run: bool = False
    batch_size: int = 8  # 批处理大小，用于批量推理


@dataclass
class AppConfig:
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    io: IOConfig = field(default_factory=IOConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    exec: ExecConfig = field(default_factory=ExecConfig)


def _parse_template_vars(items: list[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for it in items:
        if "=" in it:
            k, v = it.split("=", 1)
            out[k] = v
    return out


def _read_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if yaml is None:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config_from_args(args) -> AppConfig:
    base_cfg = _read_yaml(getattr(args, "config", None))

    provider = ProviderConfig(
        name=getattr(args, "provider", base_cfg.get("provider", "echo")),
        model=getattr(args, "model", base_cfg.get("model", "local")),
        base_url=getattr(args, "base_url", base_cfg.get("base_url")),
        api_key=getattr(args, "api_key", base_cfg.get("api_key")),
        params=base_cfg.get("params", {}),
    )

    io = IOConfig(
        input_path=getattr(args, "input", base_cfg.get("input_path", "")),
        input_glob=getattr(args, "glob", base_cfg.get("input_glob")),
        output_path=getattr(args, "output", base_cfg.get("output_path")),
        inplace=bool(getattr(args, "inplace", base_cfg.get("inplace", False))),
        field=getattr(args, "field", base_cfg.get("field", "code")),
        backup_field=getattr(args, "backup_field", base_cfg.get("backup_field")),
    )

    prompt = PromptConfig(
        instruction=getattr(args, "instruction", base_cfg.get("instruction", "")),
        template_vars={**base_cfg.get("template_vars", {}), **_parse_template_vars(getattr(args, "template_var", []))},
        strip_fences=bool(getattr(args, "strip_fences", base_cfg.get("strip_fences", False))),
        auto_extract_code=bool(getattr(args, "auto_extract_code", base_cfg.get("auto_extract_code", False))),
    )

    ex = ExecConfig(
        concurrency=int(getattr(args, "concurrency", base_cfg.get("concurrency", 4))),
        retry=int(getattr(args, "retry", base_cfg.get("retry", 3))),
        cache_dir=getattr(args, "cache_dir", base_cfg.get("cache_dir", ".cache")),
        resume=bool(getattr(args, "resume", base_cfg.get("resume", True))),
        dry_run=bool(getattr(args, "dry_run", base_cfg.get("dry_run", False))),
        batch_size=int(getattr(args, "batch_size", base_cfg.get("batch_size", 8))),
    )

    return AppConfig(provider=provider, io=io, prompt=prompt, exec=ex)
