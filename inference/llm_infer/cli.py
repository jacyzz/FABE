from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm
from dotenv import load_dotenv

from .dataset_io import read_dataset, write_dataset
from .template import PromptTemplate, render_messages
from .client import ModelClient, ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM inference pipeline for code field processing")
    parser.add_argument("--input", required=True, help="输入数据集文件路径（jsonl/json/csv）")
    parser.add_argument("--output", required=True, help="输出数据集文件路径（jsonl/json/csv）")
    parser.add_argument("--field", required=True, help="需要处理并替换的字段名，例如 code 或 source_code")
    parser.add_argument("--template", required=True, help="模板名称或模板文件路径（YAML）")
    parser.add_argument("--user-prompt", default="", help="额外用户提示词，可为空")
    parser.add_argument("--system-prompt", default=None, help="自定义系统提示词，留空使用默认")
    parser.add_argument("--templates-dir", default=None, help="自定义模板目录，可选")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"), help="模型名称")
    parser.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL"), help="API Base，可选")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="API Key，可选，亦可用环境变量")
    parser.add_argument("--temperature", type=float, default=0.2, help="temperature")
    parser.add_argument("--max-tokens", type=int, default=2048, help="最大输出tokens")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，可选")
    parser.add_argument("--dry-run", action="store_true", help="仅渲染模板与消息，不调用模型")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前N条数据用于快速验证")
    # 长度控制
    parser.add_argument("--no-record-json", action="store_true", help="不在提示中包含整条record的JSON上下文")
    parser.add_argument(
        "--context-budget-chars", type=int, default=24000,
        help="上下文字符预算（近似控制总输入长度，防止超过模型上下文）。"
    )
    parser.add_argument(
        "--truncate-strategy", choices=["head", "tail", "middle"], default="middle",
        help="当输入超预算时截断代码的方式"
    )
    return parser.parse_args()


def _truncate_text(text: str, max_chars: int, strategy: str) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return ""
    if strategy == "head":
        return text[:max_chars]
    if strategy == "tail":
        return text[-max_chars:]
    # middle
    keep = max_chars
    half = keep // 2
    return text[:half] + text[-(keep - half):]


def build_messages(template_name: str, templates_dir: str | None, system_prompt: str, user_prompt: str, code_input: str, extra_context: Dict[str, Any] | None = None) -> List[Dict[str, str]]:
    template = PromptTemplate.load(template_name, templates_dir=templates_dir)
    context: Dict[str, Any] = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "code_input": code_input,
    }
    if extra_context:
        context.update(extra_context)
    return render_messages(template, context)


def main() -> None:
    # 先加载 .env（若存在）
    load_dotenv()
    args = parse_args()
    records = read_dataset(args.input)
    if args.limit is not None:
        records = records[: args.limit]

    client = None
    if not args.dry_run:
        cfg = ModelConfig(
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        client = ModelClient(cfg)

    output_records: List[Dict[str, Any]] = []

    system_prompt = (
        args.system_prompt
        if args.system_prompt is not None
        else "你是一个资深代码生成与重构助手。请根据模板与用户需求，输出可直接替换到字段的代码内容。"
    )

    for rec in tqdm(records, desc="processing"):
        code_input_raw = str(rec.get(args.field, ""))
        # 估算预算：系统+用户提示词占用后，剩余用于代码
        overhead = len(system_prompt) + len(args.user_prompt or "") + 800  # 模板文字估算
        allowed_for_code = max(200, args.context_budget_chars - overhead)
        code_input = _truncate_text(code_input_raw, allowed_for_code, args.truncate_strategy)

        record_for_ctx: Dict[str, Any] = {} if args.no_record_json else rec
        messages = build_messages(
            template_name=args.template,
            templates_dir=args.templates_dir,
            system_prompt=system_prompt,
            user_prompt=args.user_prompt,
            code_input=code_input,
            extra_context={"record": record_for_ctx},
        )

        if args.dry_run:
            # 保留原字段，同时增加一个渲染后的消息供检查
            new_rec = dict(rec)
            new_rec[args.field + "_messages_preview"] = messages
            output_records.append(new_rec)
            continue

        # 调用模型
        assert client is not None, "client should be initialized when not in dry-run"
        content, usage = client.complete(messages)
        new_rec = dict(rec)
        new_rec[args.field] = content
        if usage:
            new_rec["_usage"] = usage
        output_records.append(new_rec)

    write_dataset(args.output, output_records)


if __name__ == "__main__":
    main()


