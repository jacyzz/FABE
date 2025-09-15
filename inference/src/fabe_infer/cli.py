import argparse
import sys
from .config import load_config_from_args
from .runner import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fabe-infer", description="Batch LLM inference for dataset transformation")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run batch inference")
    run.add_argument("--config", type=str, default=None, help="YAML config file")
    run.add_argument("--input", type=str, help="Input file or directory")
    run.add_argument("--glob", type=str, default=None, help="Glob filter when input is a directory, e.g. *.jsonl")
    run.add_argument("--field", type=str, required=True, help="Target field to transform, e.g. code")
    run.add_argument("--instruction", type=str, required=True, help="Instruction text or path to template file")
    run.add_argument("--provider", type=str, default="echo", help="Provider: echo|ollama|openai|hf|modelscope")
    run.add_argument("--model", type=str, default="local", help="Model name or path")
    run.add_argument("--base-url", type=str, default=None, help="Base URL for HTTP providers (OpenAI-compatible)")
    run.add_argument("--api-key", type=str, default=None, help="API key for OpenAI-compatible provider")
    run.add_argument("--output", type=str, default=None, help="Output directory; if omitted, inplace unless dry-run")
    run.add_argument("--inplace", action="store_true", help="Modify files in place")
    run.add_argument("--backup-field", type=str, default=None, help="Backup original value into this field")
    run.add_argument("--concurrency", type=int, default=4, help="并发处理任务数")
    run.add_argument("--batch-size", type=int, default=8, help="批处理大小（仅支持批处理的提供者有效）")
    run.add_argument("--retry", type=int, default=3, help="失败重试次数")
    run.add_argument("--cache-dir", type=str, default=".cache", help="缓存目录")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--strip-fences", action="store_true")
    run.add_argument("--auto-extract-code", action="store_true")
    run.add_argument("--template-var", action="append", default=[], help="Template variable k=v, may be repeated")

    return p


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        cfg = load_config_from_args(args)
        run_pipeline(cfg)
        return 0
    else:
        parser.error("Unknown command")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
