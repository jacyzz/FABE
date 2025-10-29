import os
import subprocess
import time
import json
from typing import Optional
from urllib.parse import urlparse
import urllib.request


class LLMCleaner:
    def __init__(self, run_vllm_script: str, run_infer_script: str) -> None:
        self.run_vllm_script = run_vllm_script
        self.run_infer_script = run_infer_script

    def is_server_alive(self, api_base: str) -> bool:
        try:
            # OpenAI-compatible health check: try models endpoint
            parsed = urlparse(api_base)
            base = f"{parsed.scheme}://{parsed.netloc}"
            url = base + "/v1/models"
            with urllib.request.urlopen(url, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def start_vllm(
        self,
        model_path: str,
        served_model_name: str,
        port: int = 8000,
        max_len: int = 8192,
        dtype: str = "bfloat16",
        env: Optional[dict] = None,
    ) -> subprocess.Popen:
        cmd = [
            "bash",
            self.run_vllm_script,
            model_path,
            served_model_name,
            str(port),
            str(max_len),
            dtype,
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, **(env or {})},
        )
        return proc

    def clean_file(
        self,
        input_path: str,
        output_path: str,
        field: str,
        template: str,
        system_prompt: str,
        served_model_name: str,
        api_base: str,
        max_tokens: int,
        temperature: float,
        extra_env: Optional[dict] = None,
    ) -> None:
        input_abs = os.path.abspath(input_path)
        output_abs = os.path.abspath(output_path)

        # We avoid using the external shell script to ensure full control of params

        # Call python -m llm_infer.cli directly with absolute paths, inject PYTHONPATH to inference root
        env = {**os.environ, **(extra_env or {})}
        infer_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../inference"))
        env["PYTHONPATH"] = infer_root + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")

        cmd = [
            "python",
            "-m",
            "llm_infer.cli",
            "--input",
            input_abs,
            "--output",
            output_abs,
            "--field",
            field,
            "--template",
            template,
            "--system-prompt",
            system_prompt,
            "--model",
            served_model_name,
            "--api-base",
            api_base,
            "--max-tokens",
            str(max_tokens),
            "--temperature",
            str(temperature),
        ]
        subprocess.check_call(cmd, env=env)


