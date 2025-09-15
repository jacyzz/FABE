# A dictionary mapping model template names to their specific prompt structures.
# This allows for flexible adaptation to different models.

# The placeholders {instruction} and {input} will be filled dynamically.
PROMPT_TEMPLATES = {
    "deepseek": (
        "You are an expert programmer and code reviewer. "
        "Please fulfill the following request.\n"
        "### Instruction:\n"
        "{instruction}\n"
        "### Input:\n"
        "```\n"
        "{input}\n"
        "```\n"
        "### Response:\n"
    ),
    # 系统提示驱动的清洗模板：将清洗要求写入系统提示，数据仅需提供 input（后门代码）。
    # instruction 可忽略或置空；模型被要求仅输出清洗后的代码。
    "deepseek_clean_sys": (
        "You are a senior code security engineer. "
        "Sanitize the given code by removing backdoors and malicious behaviors while preserving functionality. "
        "Return ONLY the sanitized code without any explanations or extra text.\n"
        "### Input:\n"
        "```\n"
        "{input}\n"
        "```\n"
        "### Response:\n"
    ),
    "default": (
        "Instruction: {instruction}\n"
        "Input: {input}\n"
        "Response:\n"
    ),
    # Add other templates like 'llama3', 'qwen', etc. here in the future.
}

# Chat 模板：仅提供系统提示文本，实际拼接依赖 tokenizer.apply_chat_template
CHAT_TEMPLATES = {
    # 原生 DeepSeekCoder 聊天式（系统提示固定为清洗任务）
    "deepseek_chat": (
        "You are a senior code security engineer. "
        "Sanitize the given code by removing backdoors and malicious behaviors while preserving functionality. "
        "Return ONLY the sanitized code without any explanations or extra text."
    ),
}

def get_template(template_name: str) -> str:
    """
    Retrieves a prompt template by name.
    
    Args:
        template_name (str): The name of the template (e.g., 'deepseek').

    Returns:
        str: The prompt template string.
        
    Raises:
        ValueError: If the template name is not found.
    """
    template = PROMPT_TEMPLATES.get(template_name)
    if template is None:
        raise ValueError(
            f"Template '{template_name}' not found. "
            f"Available templates: {list(PROMPT_TEMPLATES.keys())}"
        )
    return template

def is_chat_template(template_name: str) -> bool:
    return template_name in CHAT_TEMPLATES

def get_chat_system_prompt(template_name: str) -> str:
    if not is_chat_template(template_name):
        raise ValueError(
            f"Template '{template_name}' is not a chat template. "
            f"Available chat templates: {list(CHAT_TEMPLATES.keys())}"
        )
    return CHAT_TEMPLATES[template_name]
