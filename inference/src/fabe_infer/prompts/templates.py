from __future__ import annotations
import os
from typing import Dict, Any, Tuple

from jinja2 import Template


def load_instruction(instruction: str, template_vars: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Return (rendered_instruction, extras). If instruction is a path, read it and render as Jinja2.
    extras can be used by providers that need system+user separation (not used here for simplicity).
    """
    if os.path.exists(instruction):
        with open(instruction, "r", encoding="utf-8") as f:
            template_text = f.read()
        tpl = Template(template_text)
        rendered = tpl.render(**template_vars)
        return rendered, {}
    else:
        # treat as plain instruction text; allow variable injection too
        tpl = Template(instruction)
        return tpl.render(**template_vars), {}
