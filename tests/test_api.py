"""Tests for API helpers."""

from __future__ import annotations

from typing import Any

import datagen.api
from datagen.api import Usage, call_chat


def test_usage_dataclass_defaults() -> None:
    usage = Usage()
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0


def test_call_chat_extracts_reasoning_content_field(monkeypatch) -> None:
    class _Usage:
        prompt_tokens = 3
        completion_tokens = 7
        total_tokens = 10

    class _Message:
        content = "最终答案"
        reasoning_content = "这是推理过程"

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]
        usage = _Usage()

    class _Completions:
        def create(self, **kwargs) -> _Response:
            del kwargs
            return _Response()

    class _Chat:
        completions = _Completions()

    class _FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            del kwargs
            self.chat = _Chat()

    monkeypatch.setattr(datagen.api, "OpenAI", _FakeOpenAI)
    content, reasoning, usage = call_chat(
        "https://api.openai.com/v1",
        "k",
        "m",
        [{"role": "user", "content": "hi"}],
    )
    assert content == "最终答案"
    assert reasoning == "这是推理过程"
    assert usage is not None and usage.completion_tokens == 7


def test_call_chat_extracts_reasoning_from_content_parts(monkeypatch) -> None:
    class _Usage:
        prompt_tokens = 1
        completion_tokens = 2
        total_tokens = 3

    class _Message:
        content = "回答"

        def model_dump(self) -> dict[str, Any]:
            return {
                "content": [
                    {"type": "reasoning", "text": "先分析"},
                    {"type": "reasoning", "text": "再结论"},
                ]
            }

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]
        usage = _Usage()

    class _Completions:
        def create(self, **kwargs) -> _Response:
            del kwargs
            return _Response()

    class _Chat:
        completions = _Completions()

    class _FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            del kwargs
            self.chat = _Chat()

    monkeypatch.setattr(datagen.api, "OpenAI", _FakeOpenAI)
    content, reasoning, _ = call_chat(
        "https://api.openai.com/v1",
        "k",
        "m",
        [{"role": "user", "content": "hi"}],
    )
    assert content == "回答"
    assert reasoning == "先分析再结论"


def test_call_chat_closes_client(monkeypatch) -> None:
    class _Message:
        content = "ok"

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]
        usage = None

    class _Completions:
        def create(self, **kwargs) -> _Response:
            del kwargs
            return _Response()

    class _Chat:
        completions = _Completions()

    class _FakeOpenAI:
        closed_count = 0

        def __init__(self, **kwargs) -> None:
            del kwargs
            self.chat = _Chat()

        def close(self) -> None:
            _FakeOpenAI.closed_count += 1

    monkeypatch.setattr(datagen.api, "OpenAI", _FakeOpenAI)
    _ = call_chat(
        "https://api.openai.com/v1",
        "k",
        "m",
        [{"role": "user", "content": "hi"}],
    )
    assert _FakeOpenAI.closed_count == 1
