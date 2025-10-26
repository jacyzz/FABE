from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml
from jinja2 import Environment, StrictUndefined
from openai import OpenAI


def build_env() -> Environment:
    env = Environment(undefined=StrictUndefined, trim_blocks=False, lstrip_blocks=False, autoescape=False)
    env.filters["tojson"] = lambda x: json.dumps(x, ensure_ascii=False)
    return env


def load_template_messages(template_path: str) -> List[Dict[str, str]]:
    with open(template_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["messages"]


def render_messages(messages: List[Dict[str, str]], context: Dict[str, Any]) -> List[Dict[str, str]]:
    env = build_env()
    rendered: List[Dict[str, str]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content_tmpl = env.from_string(msg.get("content", ""))
        content = content_tmpl.render(**context)
        rendered.append({"role": role, "content": content})
    return rendered


def strip_output(text: str, strip_think: bool, strip_fences: bool, strip_comments: bool, lang: str) -> str:
    s = text
    # 屏蔽服务端异常回填的 beam_search 调度日志
    if s.strip().lower().startswith("added request beam_search"):
        return ""
    if strip_think:
        import re
        s = re.sub(r"(?is)<think>.*?</think>", "", s)
    if strip_fences:
        s = s.replace("```", "").replace("~~~", "")
    if strip_comments:
        import re
        if lang.lower() in {"python", "py"}:
            s = re.sub(r"(?m)^\s*#.*$", "", s)
        # 通用去块注释与行注释（尽量保守）
        s = re.sub(r"(?s)/\*.*?\*/", "", s)
        s = re.sub(r"(?m)^[ \t]*//.*$", "", s)
    s = s.strip()
    # 清理指令模板残留
    for marker in ["### Instruction:", "### Response:", "<｜end▁of▁sentence｜>", "<｜begin▁of▁sentence｜>"]:
        if marker in s:
            try:
                # 取最后一段代码块（多段重复时）
                parts = [p.strip() for p in s.split(marker) if p.strip()]
                s = parts[-1] if parts else s
            except Exception:
                pass
    return s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diverse Beam Search or n-best sampling over OpenAI-compatible endpoint")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--field", required=True)
    p.add_argument("--template", required=True, help="YAML 模板路径")
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--user-prompt", default="")
    # model/api
    p.add_argument("--model", required=True)
    p.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"))
    p.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY") or "sk-local")
    # generation
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--presence-penalty", type=float, default=0.0)
    p.add_argument("--frequency-penalty", type=float, default=0.0)
    p.add_argument("--repetition-penalty", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    # n-best
    p.add_argument("--n-samples", type=int, default=4, help="返回候选个数（OpenAI n）")
    # diverse beam (若服务端支持，会生效；否则退化为 n-best 采样)
    p.add_argument("--use-beam-search", action="store_true")
    p.add_argument("--num-beams", type=int, default=8)
    p.add_argument("--num-beam-groups", type=int, default=4)
    p.add_argument("--diversity-penalty", type=float, default=0.5)
    p.add_argument("--no-repeat-ngram-size", type=int, default=0)
    p.add_argument("--length-penalty", type=float, default=1.0)
    # misc
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--lang", default="python")
    p.add_argument("--strip-think", action="store_true")
    p.add_argument("--strip-fences", action="store_true")
    p.add_argument("--strip-comments", action="store_true")
    return p.parse_args()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def main() -> None:
    args = parse_args()

    records = read_jsonl(args.input)
    if args.limit is not None:
        records = records[: args.limit]

    messages_tmpl = load_template_messages(args.template)

    client = OpenAI(api_key=args.api_key, base_url=args.api_base) if args.api_base else OpenAI(api_key=args.api_key)

    # 输出 predictions.jsonl：每个变体一行
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    f_out = open(args.output, "w", encoding="utf-8")
    for idx, rec in enumerate(records):
        code_input = str(rec.get(args.field, ""))
        ctx = {
            "system_prompt": args.system_prompt or "你是一个资深代码安全与重构专家，仅输出可直接替换的最终代码。",
            "user_prompt": args.user_prompt,
            "code_input": code_input,
            "record": rec,
        }
        messages = render_messages(messages_tmpl, ctx)

        params: Dict[str, Any] = {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "n": max(1, args.n_samples),
        }
        if args.seed is not None:
            params["seed"] = args.seed
        if args.presence_penalty:
            params["presence_penalty"] = args.presence_penalty
        if args.frequency_penalty:
            params["frequency_penalty"] = args.frequency_penalty
        # 将非标准/供应商特有参数放入 extra_body，避免 SDK 参数校验错误
        extra_body: Dict[str, Any] = {}
        if args.repetition_penalty is not None:
            extra_body["repetition_penalty"] = args.repetition_penalty

        # Diverse Beam（若后端支持则生效；OpenAI 兼容接口通常通过 extra_body 透传）
        if args.use_beam_search:
            extra_body["use_beam_search"] = True
            extra_body["best_of"] = max(args.n_samples, args.num_beams)
            extra_body["num_beams"] = args.num_beams
            extra_body["num_beam_groups"] = args.num_beam_groups
            extra_body["diversity_penalty"] = args.diversity_penalty
            if args.no_repeat_ngram_size:
                extra_body["no_repeat_ngram_size"] = args.no_repeat_ngram_size
            extra_body["length_penalty"] = args.length_penalty

        if extra_body:
            params["extra_body"] = extra_body

        resp = client.chat.completions.create(**params)
        variants: List[str] = []
        for ch in resp.choices:
            content = ch.message.content or ""
            content = strip_output(
                content,
                strip_think=args.strip_think,
                strip_fences=args.strip_fences,
                strip_comments=args.strip_comments,
                lang=args.lang,
            )
            if content:
                variants.append(content)

        # 去重（简单规范化去重）
        norm = lambda s: "\n".join(l.rstrip() for l in s.splitlines()).strip()
        seen = set()
        uniq: List[str] = []
        for v in variants:
            key = norm(v)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(v)

        # 写出 predictions.jsonl 所需行
        task_id = rec.get("task_id") or rec.get("id") or str(idx)
        lang = args.lang
        for sample_id, variant in enumerate(uniq):
            line = {
                "task_id": task_id,
                "language": lang,
                "sample_id": sample_id,
                "completion": variant,
            }
            f_out.write(json.dumps(line, ensure_ascii=False))
            f_out.write("\n")
    f_out.close()


if __name__ == "__main__":
    main()


