from __future__ import annotations
import os
import hashlib


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def content_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()


def cache_get(cache_dir: str, key: str) -> str | None:
    ensure_dir(cache_dir)
    fp = os.path.join(cache_dir, f"{key}.txt")
    if os.path.exists(fp):
        with open(fp, "r", encoding="utf-8") as f:
            return f.read()
    return None


def cache_put(cache_dir: str, key: str, value: str) -> None:
    ensure_dir(cache_dir)
    fp = os.path.join(cache_dir, f"{key}.txt")
    with open(fp, "w", encoding="utf-8") as f:
        f.write(value)
