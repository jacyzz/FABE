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
    "default": (
        "Instruction: {instruction}\n"
        "Input: {input}\n"
        "Response:\n"
    ),
    # Add other templates like 'llama3', 'qwen', etc. here in the future.
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
