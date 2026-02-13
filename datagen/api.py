"""OpenAI-compatible API helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


@dataclass(frozen=True)
class Usage:
    """Token usage counts from a chat completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class FinishReasonLengthError(RuntimeError):
    """Raised when API returns finish_reason='length'."""


CHERRY_STUDIO_HTTP_REFERER = "https://cherry-ai.com"
CHERRY_STUDIO_X_TITLE = "Cherry Studio"
CHERRY_STUDIO_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "CherryStudio/1.7.10 Chrome/124.0.0.0 Safari/537.36"
)


def cherry_studio_headers() -> dict[str, str]:
    """Returns request headers aligned with Cherry Studio defaults."""
    return {
        "HTTP-Referer": CHERRY_STUDIO_HTTP_REFERER,
        "X-Title": CHERRY_STUDIO_X_TITLE,
        "User-Agent": CHERRY_STUDIO_USER_AGENT,
    }


def _to_dict_if_possible(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    return None


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            item_dict = _to_dict_if_possible(item)
            if item_dict is None:
                continue
            text = item_dict.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""


def _extract_reasoning_from_message(message: Any) -> str | None:
    direct = getattr(message, "reasoning_content", None)
    if isinstance(direct, str) and direct.strip():
        return direct

    direct = getattr(message, "reasoning", None)
    direct_text = _extract_text(direct)
    if direct_text.strip():
        return direct_text

    message_dict = _to_dict_if_possible(message)
    if message_dict is not None:
        dict_reasoning_content = message_dict.get("reasoning_content")
        if isinstance(dict_reasoning_content, str) and dict_reasoning_content.strip():
            return dict_reasoning_content

        dict_reasoning = _extract_text(message_dict.get("reasoning"))
        if dict_reasoning.strip():
            return dict_reasoning

        content_parts = message_dict.get("content")
        if isinstance(content_parts, list):
            reasoning_parts: list[str] = []
            for part in content_parts:
                part_dict = _to_dict_if_possible(part)
                if part_dict is None:
                    continue
                part_type = part_dict.get("type")
                if part_type not in {"reasoning", "thinking"}:
                    continue
                part_text = part_dict.get("text")
                if isinstance(part_text, str) and part_text.strip():
                    reasoning_parts.append(part_text)
            if reasoning_parts:
                return "".join(reasoning_parts)

    return None


def _serialize_response_for_error(response: Any) -> str:
    """Serializes full API response for error reporting."""
    model_dump_json = getattr(response, "model_dump_json", None)
    if callable(model_dump_json):
        try:
            return str(model_dump_json())
        except Exception:
            pass
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            return str(dumped)
        except Exception:
            pass
    return str(response)


def call_chat(
    api_base: str,
    api_key: str,
    model: str,
    messages: Sequence[dict[str, Any]],
    reasoning_effort: str | None = None,
    thinking: bool = False,
    timeout_ms: int | None = None,
    max_tokens: int | None = None,
) -> tuple[str, str | None, Usage | None]:
    """Calls an OpenAI-compatible chat/completions endpoint.

    Args:
        api_base: Base URL for the API.
        api_key: API key for authentication.
        model: Model name to use.
        messages: Chat messages to send.
        reasoning_effort: Optional reasoning effort level.
        thinking: Whether to enable extended thinking.
        timeout_ms: Optional request timeout in milliseconds.
        max_tokens: Optional max completion tokens.

    Returns:
        A tuple of (content, reasoning, usage).

    Raises:
        RuntimeError: If no assistant content is returned.
    """
    client = OpenAI(
        api_key=api_key,
        base_url=api_base.rstrip("/"),
        default_headers=cherry_studio_headers(),
    )
    try:
        extra_body: dict[str, Any] = {}
        if reasoning_effort is not None and reasoning_effort.strip():
            extra_body["reasoning"] = {"effort": reasoning_effort.strip()}
        if thinking:
            extra_body["thinking"] = {"type": "enabled"}
            extra_body["enable_thinking"] = True

        timeout_seconds = (timeout_ms / 1000.0) if timeout_ms is not None else None
        response = client.chat.completions.create(
            model=model,
            messages=cast(list[ChatCompletionMessageParam], list(messages)),
            timeout=timeout_seconds,
            max_tokens=max_tokens,
            extra_body=extra_body if extra_body else None,
        )

        content = ""
        reasoning = None
        if not response.choices:
            raise FinishReasonLengthError("no assistant message returned")

        first_choice = response.choices[0]
        if getattr(first_choice, "finish_reason", None) == "length":
            raise FinishReasonLengthError("finish_reason is length")

        message = getattr(first_choice, "message", None)
        if message is None:
            raise FinishReasonLengthError("no assistant message returned")

        message_role = getattr(message, "role", None)
        if message_role is not None and message_role != "assistant":
            raise FinishReasonLengthError("no assistant message returned")

        content = _extract_text(getattr(message, "content", None))
        reasoning = _extract_reasoning_from_message(message)
        if not content:
            raise FinishReasonLengthError("No assistant content returned from API")

        usage = None
        if response.usage is not None:
            usage = Usage(
                prompt_tokens=int(response.usage.prompt_tokens or 0),
                completion_tokens=int(response.usage.completion_tokens or 0),
                total_tokens=int(response.usage.total_tokens or 0),
            )

        return content, reasoning, usage
    finally:
        close_method = getattr(client, "close", None)
        if callable(close_method):
            close_method()
