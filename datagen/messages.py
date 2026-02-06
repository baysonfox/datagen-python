"""Message building helpers."""

from __future__ import annotations


def build_request_messages(
    system_prompt: str, user_prompt: str
) -> list[dict[str, str]]:
    """Builds request messages for chat completion.

    Args:
        system_prompt: Optional system prompt.
        user_prompt: User prompt content.

    Returns:
        Message list in OpenAI/OpenRouter chat format.
    """
    if system_prompt.strip():
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    return [{"role": "user", "content": user_prompt}]


def format_assistant_content(content: str, reasoning: str | None = None) -> str:
    """Formats assistant output with optional reasoning wrapper."""
    if reasoning is not None and reasoning.strip():
        return f"<think>{reasoning}</think>\n{content}"
    return content


def build_output_messages(
    system_prompt: str,
    user_prompt: str,
    assistant_content: str,
    store_system: bool,
) -> list[dict[str, str]]:
    """Builds output dataset messages according to store-system behavior."""
    has_system = bool(system_prompt.strip())
    if has_system and store_system:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content},
        ]
    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_content},
    ]
