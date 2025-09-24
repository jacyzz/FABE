import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


def _infer_format(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in [".jsonl", ".ndjson"]:
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    raise ValueError(f"Unsupported dataset format for file: {path}")


def read_dataset(path: str) -> List[Dict]:
    fmt = _infer_format(path)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if fmt == "jsonl":
        records: List[Dict] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    if fmt == "json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError("JSON dataset must be a list of objects")
    if fmt == "csv":
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)
    raise AssertionError("unreachable")


def write_dataset(path: str, records: Iterable[Dict]) -> None:
    fmt = _infer_format(path)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with p.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False))
                f.write("\n")
        return
    if fmt == "json":
        with p.open("w", encoding="utf-8") as f:
            json.dump(list(records), f, ensure_ascii=False, indent=2)
        return
    if fmt == "csv":
        records_list = list(records)
        if not records_list:
            with p.open("w", encoding="utf-8", newline="") as f:
                f.write("")
            return
        fieldnames = sorted({k for rec in records_list for k in rec.keys()})
        with p.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records_list:
                writer.writerow(rec)
        return
    raise AssertionError("unreachable")


