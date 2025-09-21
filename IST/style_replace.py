#!/usr/bin/env python3
"""
Style Replace (Single-Field)
使用 IST 对指定代码字段进行风格转换/投毒，并用新代码直接替换原字段。

特性：
- 支持固定/指定风格序列（--styles）
- 支持从风格池随机选择 2-3 个不同类别风格（可配置）
- 仅写回单字段；同时写入 poison_tag 到指定字段（默认 meta）
- 语法严格校验，失败回退
"""

import os
import sys
import json
import random
import argparse
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from transfer import IST

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable


def parse_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):  # noqa: E712
        return True
    if v in ("0", "false", "no", "n", "off"):  # noqa: E712
        return False
    return default


class StyleReplacer:
    def __init__(self, language: str,
                 poison_candidates: Optional[List[str]] = None,
                 avoid_similar: bool = True,
                 strict_syntax: bool = True,
                 seed: Optional[int] = None):
        self.language = language
        self.ist = IST(language)
        self.avoid_similar = avoid_similar
        self.strict_syntax = strict_syntax
        self.global_random = random.Random(seed)

        # 默认候选池
        base_poison = ['-1.1', '-3.1', '0.5', '7.2', '8.1', '9.1', '11.3', '3.4', '4.4', '10.7']
        if language in ['c', 'java']:
            extra_loop = ['11.1', '11.2', '10.1', '10.3', '10.5', '4.1', '4.2', '4.3']
            default_pool = base_poison + extra_loop
        else:
            default_pool = base_poison

        if poison_candidates:
            self.poison_pool = [s.strip() for s in poison_candidates if str(s).strip()]
        else:
            self.poison_pool = default_pool

    def read_jsonl(self, path: str, limit: int = 0) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                rows.append(obj)
                if limit and len(rows) >= limit:
                    break
        return rows

    def write_jsonl(self, path: str, rows: List[Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

    def _check_ok(self, code_before: str, code_after: str) -> bool:
        if code_before.replace('\n', '').replace(' ', '') == code_after.replace('\n', '').replace(' ', ''):
            return False
        if self.strict_syntax and not self.ist.check_syntax(code_after):
            return False
        return True

    def _apply_styles_in_order(self, code: str, styles: List[str]) -> Tuple[str, List[str]]:
        current = code
        applied: List[str] = []
        for st in styles:
            try:
                new_code, ok = self.ist.transfer(styles=[st], code=current)
                if ok and self._check_ok(current, new_code):
                    current = new_code
                    applied.append(st)
            except Exception:
                continue
        return current, applied

    def _apply_styles_random(self, code: str, poison_min: int, poison_max: int, local_rng: random.Random) -> Tuple[str, List[str]]:
        pool = list(self.poison_pool)
        local_rng.shuffle(pool)
        target = local_rng.randint(max(1, poison_min), max(poison_min, poison_max))

        def group_key(style: str) -> str:
            return style.split('.')[0]

        current = code
        applied: List[str] = []
        used_groups = set()

        # 第一轮：尽量挑不同组
        for st in pool:
            if len(applied) >= target:
                break
            if self.avoid_similar and group_key(st) in used_groups:
                continue
            try:
                new_code, ok = self.ist.transfer(styles=[st], code=current)
                if ok and self._check_ok(current, new_code):
                    current = new_code
                    applied.append(st)
                    used_groups.add(group_key(st))
            except Exception:
                continue

        # 补齐：若仅成功 1 个且目标>=2，尝试补一个
        if len(applied) == 1 and target >= 2:
            for st in pool:
                if st in applied:
                    continue
                if self.avoid_similar and group_key(st) in used_groups:
                    continue
                try:
                    new_code, ok = self.ist.transfer(styles=[st], code=current)
                    if ok and self._check_ok(current, new_code):
                        current = new_code
                        applied.append(st)
                        used_groups.add(group_key(st))
                        break
                except Exception:
                    continue

        # 兜底：若完全失败，尝试从池中找一个能成功的
        if not applied:
            for st in self.poison_pool:
                try:
                    new_code, ok = self.ist.transfer(styles=[st], code=code)
                    if ok and self._check_ok(code, new_code):
                        return new_code, [st]
                except Exception:
                    continue
            return code, []

        return current, applied

    def process_row(self, obj: Dict[str, Any], *, code_field: str,
                    styles: Optional[List[str]], poison_min: int, poison_max: int,
                    id_field: Optional[str], backup_field: Optional[str]) -> Optional[Dict[str, Any]]:
        if code_field not in obj or not isinstance(obj[code_field], str) or not obj[code_field].strip():
            return None

        raw_code = obj[code_field]

        # 局部 RNG：基于全局种子 + id 派生
        seed_material = None
        if id_field and id_field in obj and isinstance(obj[id_field], (str, int)):
            seed_material = f"{obj[id_field]}"
        local_seed = None
        if seed_material is not None:
            local_seed = hash(seed_material) ^ self.global_random.getrandbits(32)
        local_rng = random.Random(local_seed)

        # 应用风格
        if styles and len(styles) > 0:
            new_code, poison_tag = self._apply_styles_in_order(raw_code, styles)
        else:
            new_code, poison_tag = self._apply_styles_random(raw_code, poison_min, poison_max, local_rng)

        # 若一个也没有成功，保留原代码
        if not poison_tag:
            new_code = raw_code

        # 写回
        if backup_field:
            obj[backup_field] = raw_code
        obj[code_field] = new_code

        return obj, poison_tag


def main():
    parser = argparse.ArgumentParser(description="IST Style Replace (single-field)")

    # 必填
    parser.add_argument("--input_path", required=True, help="输入 JSONL 路径")
    parser.add_argument("--output_path", required=True, help="输出 JSONL 路径")
    parser.add_argument("--language", required=True, choices=["c", "java", "python"], help="语言")
    parser.add_argument("--code_field", required=True, help="要替换的代码字段名")

    # 选填：风格与策略
    parser.add_argument("--styles", default="", help="固定/指定风格，逗号分隔，如 '-1.1,11.3'")
    parser.add_argument("--poison_candidates", default="", help="自定义风格池，逗号分隔")
    parser.add_argument("--poison_min", type=int, default=2, help="随机时最少风格数")
    parser.add_argument("--poison_max", type=int, default=3, help="随机时最多风格数")
    parser.add_argument("--avoid_similar", default="true", help="避免相近风格重复 true|false")
    parser.add_argument("--strict_syntax", default="true", help="严格语法校验 true|false")

    # 选填：流程与复现
    parser.add_argument("--limit", type=int, default=0, help="处理条数，0 表示全部")
    parser.add_argument("--seed", type=int, default=None, help="全局随机种子")
    parser.add_argument("--id_field", default="", help="样本 ID 字段名（用于派生局部种子）")
    parser.add_argument("--backup_field", default="", help="原代码备份字段名，不填则不备份")
    parser.add_argument("--poison_tag_field", default="meta", help="写入 poison_tag 的字段名，默认 meta")
    parser.add_argument("--log_path", default="", help="如提供，则输出每条的处理日志 JSONL")

    args = parser.parse_args()

    styles_list = [s.strip() for s in args.styles.split(',') if s.strip()] if args.styles else []
    poisons_list = [s.strip() for s in args.poison_candidates.split(',') if s.strip()] if args.poison_candidates else None
    avoid_similar = parse_bool(args.avoid_similar, True)
    strict_syntax = parse_bool(args.strict_syntax, True)
    id_field = args.id_field if args.id_field else None
    backup_field = args.backup_field if args.backup_field else None

    replacer = StyleReplacer(
        language=args.language,
        poison_candidates=poisons_list,
        avoid_similar=avoid_similar,
        strict_syntax=strict_syntax,
        seed=args.seed,
    )

    print("=== IST Style Replace ===")
    print(f"Input:    {args.input_path}")
    print(f"Output:   {args.output_path}")
    print(f"Language: {args.language}")
    print(f"CodeField:{args.code_field}")
    print(f"Styles:   {styles_list if styles_list else '[random from pool]'}")
    print(f"Pool:     {replacer.poison_pool}")
    print(f"Min/Max:  {args.poison_min}/{args.poison_max}")
    print(f"Strict:   {strict_syntax}")
    print(f"AvoidSim: {avoid_similar}")
    print("================================")

    rows = replacer.read_jsonl(args.input_path, limit=args.limit)
    out_rows: List[Dict[str, Any]] = []
    changed_cnt = 0
    tag_nonempty_cnt = 0
    syntax_ok_cnt = 0

    log_f = None
    if args.log_path:
        os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
        log_f = open(args.log_path, 'w', encoding='utf-8')

    for idx, obj in enumerate(tqdm(rows, desc="Replacing", unit="sample")):
        raw_code_val = obj.get(args.code_field, "") if isinstance(obj.get(args.code_field, ""), str) else ""
        orig_syntax_ok = replacer.ist.check_syntax(raw_code_val) if raw_code_val else False
        if orig_syntax_ok:
            syntax_ok_cnt += 1

        res = replacer.process_row(
            obj,
            code_field=args.code_field,
            styles=styles_list,
            poison_min=args.poison_min,
            poison_max=args.poison_max,
            id_field=id_field,
            backup_field=backup_field,
        )
        if res is None:
            if log_f is not None:
                log_f.write(json.dumps({
                    "index": idx,
                    "id": obj.get(id_field) if id_field else None,
                    "status": "skipped",
                    "reason": "missing_or_invalid_code_field",
                    "orig_syntax_ok": orig_syntax_ok,
                }, ensure_ascii=False) + "\n")
            continue
        new_obj, poison_tag = res
        # 写 poison_tag 到指定字段（默认 meta）
        new_obj[args.poison_tag_field] = poison_tag
        out_rows.append(new_obj)

        changed = isinstance(raw_code_val, str) and isinstance(new_obj.get(args.code_field, ""), str) and (
            raw_code_val.replace("\n", "").replace(" ", "") != new_obj.get(args.code_field, "").replace("\n", "").replace(" ", "")
        )
        if changed:
            changed_cnt += 1
        if poison_tag:
            tag_nonempty_cnt += 1

        if log_f is not None:
            log_f.write(json.dumps({
                "index": idx,
                "id": obj.get(id_field) if id_field else None,
                "status": "ok",
                "poison_tag": poison_tag,
                "changed": changed,
                "orig_syntax_ok": orig_syntax_ok,
            }, ensure_ascii=False) + "\n")

    replacer.write_jsonl(args.output_path, out_rows)
    if log_f is not None:
        log_f.close()

    total = len(rows)
    print(f"\n✅ Done. Wrote {len(out_rows)} rows to {args.output_path}")
    print(f"Summary: total={total}, syntax_ok_on_input={syntax_ok_cnt}, changed={changed_cnt}, poison_tag_nonempty={tag_nonempty_cnt}")


if __name__ == "__main__":
    main()


