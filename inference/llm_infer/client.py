from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from openai import OpenAI


@dataclass
class ModelConfig:
    model: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.2
    max_tokens: Optional[int] = 2048
    timeout: Optional[float] = 120.0
    seed: Optional[int] = None


class ModelClient:
    def __init__(self, config: ModelConfig):
        # 兼容本地服务：若未提供API Key，设置占位符以满足SDK要求
        api_key = config.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY") or "sk-local"
        # 兼容多种环境变量名
        base_url = (
            config.api_base
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
        )
        # OpenAI v1 client
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=config.timeout)
        else:
            self.client = OpenAI(api_key=api_key, timeout=config.timeout)
        self.config = config

    def complete(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict]:
        params: Dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens is not None:
            params["max_tokens"] = self.config.max_tokens
        if self.config.seed is not None:
            params["seed"] = self.config.seed

        resp = self.client.chat.completions.create(**params)
        content = resp.choices[0].message.content or ""
        usage = {
            "prompt_tokens": getattr(resp, "usage", None) and resp.usage.prompt_tokens,
            "completion_tokens": getattr(resp, "usage", None) and resp.usage.completion_tokens,
            "total_tokens": getattr(resp, "usage", None) and resp.usage.total_tokens,
        }
        return content, usage


