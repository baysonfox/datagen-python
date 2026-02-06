"""Message building helpers."""

from __future__ import annotations

from typing import Any


def build_request_messages(
    system_prompt: str, user_prompt: str
) -> list[dict[str, Any]]:
    if system_prompt.strip():
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    return [{"role": "user", "content": user_prompt}]


def format_assistant_content(content: str, reasoning: str | None = None) -> str:
    if reasoning is not None and reasoning.strip():
        return f"<think>{reasoning}</think>\n{content}"
    return content


def build_output_messages(
    system_prompt: str,
    user_prompt: str,
    assistant_content: str,
    store_system: bool,
) -> list[dict[str, str]]:
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
