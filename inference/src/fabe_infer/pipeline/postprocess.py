from __future__ import annotations
import re


FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)


def strip_fences(text: str) -> str:
    # If a fenced block exists, prefer the first fenced block
    m = FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # Otherwise return trimmed
    return text.strip()
