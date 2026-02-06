"""Tests for message helpers."""

from __future__ import annotations

from datagen.messages import (
    build_output_messages,
    build_request_messages,
    format_assistant_content,
)


def test_build_request_messages_without_system() -> None:
    assert build_request_messages("", "hi") == [{"role": "user", "content": "hi"}]


def test_build_output_messages_with_store_system() -> None:
    assert build_output_messages("sys", "u", "a", True) == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]


def test_format_assistant_content_with_reasoning() -> None:
    assert (
        format_assistant_content("answer", "reason") == "<think>reason</think>\nanswer"
    )
