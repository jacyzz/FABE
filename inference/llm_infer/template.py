from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Environment, StrictUndefined


@dataclass
class PromptTemplate:
    name: str
    messages: List[Dict[str, str]]

    @staticmethod
    def load(name_or_path: str, templates_dir: Optional[str] = None) -> "PromptTemplate":
        p = Path(name_or_path)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return PromptTemplate(
                name=data.get("name") or p.stem,
                messages=data["messages"],
            )

        # try builtin
        search_dirs = []
        if templates_dir:
            search_dirs.append(Path(templates_dir))
        # project-level templates dir
        search_dirs.append(Path(__file__).resolve().parents[1] / "templates")
        for d in search_dirs:
            candidate = d / f"{name_or_path}.yaml"
            if candidate.exists():
                with candidate.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                return PromptTemplate(
                    name=data.get("name") or candidate.stem,
                    messages=data["messages"],
                )
        raise FileNotFoundError(f"Template '{name_or_path}' not found in {search_dirs} or as file path")


def _build_jinja_env() -> Environment:
    env = Environment(undefined=StrictUndefined, trim_blocks=False, lstrip_blocks=False, autoescape=False)
    env.filters["tojson"] = lambda x: json.dumps(x, ensure_ascii=False)
    return env


def render_messages(template: PromptTemplate, context: Dict[str, Any]) -> List[Dict[str, str]]:
    env = _build_jinja_env()
    # enrich context with defaults
    ctx = dict(context)
    ctx.setdefault("now", datetime.now().isoformat())
    ctx.setdefault("env", dict(os.environ))

    rendered: List[Dict[str, str]] = []
    for msg in template.messages:
        role = msg.get("role", "user")
        content_tmpl = env.from_string(msg.get("content", ""))
        content = content_tmpl.render(**ctx)
        rendered.append({"role": role, "content": content})
    return rendered


