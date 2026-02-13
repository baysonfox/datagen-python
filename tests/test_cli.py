"""Tests for CLI argument parsing and behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

import datagen.cli
from datagen.cli import parse_args, run
from datagen.api import Usage


def test_parse_args_requires_model_and_prompts() -> None:
    with pytest.raises(ValueError, match="Usage:"):
        parse_args([])
    with pytest.raises(ValueError, match="Usage:"):
        parse_args(["--model", "x"])
    with pytest.raises(ValueError, match="Usage:"):
        parse_args(["--prompts", "p.txt"])


def test_parse_args_defaults_store_system_true() -> None:
    args = parse_args(["--model", "m", "--prompts", "p.txt"])
    assert args.store_system is True


def test_parse_args_defaults_concurrent_1() -> None:
    args = parse_args(["--model", "m", "--prompts", "p.txt"])
    assert args.concurrent == 1


def test_parse_args_concurrent() -> None:
    args = parse_args(["--model", "m", "--prompts", "p.txt", "--concurrent", "3"])
    assert args.concurrent == 3


def test_parse_args_reasoning_effort() -> None:
    args = parse_args(
        ["--model", "m", "--prompts", "p.txt", "--reasoningEffort", "high"]
    )
    assert args.reasoning_effort == "high"


def test_parse_args_providers_path() -> None:
    args = parse_args(
        ["--model", "m", "--prompts", "p.txt", "--providers", "pool.jsonl"]
    )
    assert args.providers_path == "pool.jsonl"


def test_parse_args_max_tokens() -> None:
    args = parse_args(["--model", "m", "--prompts", "p.txt", "--max-tokens", "256"])
    assert args.max_tokens == 256


def test_parse_args_verbose_defaults_false() -> None:
    args = parse_args(["--model", "m", "--prompts", "p.txt"])
    assert args.verbose is False


def test_parse_args_verbose_enabled() -> None:
    args = parse_args(["--model", "m", "--prompts", "p.txt", "--verbose"])
    assert args.verbose is True


def test_run_default_prints_request_line_not_verbose(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("hello\n", encoding="utf-8")

    def _fake_chat(
        api_base: str,
        api_key: str,
        model: str,
        messages,
        reasoning_effort=None,
        thinking=False,
        timeout_ms=None,
        max_tokens=None,
    ):
        del api_base, api_key, model, messages
        del reasoning_effort, thinking, timeout_ms, max_tokens
        return (
            "ok",
            None,
            Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )

    monkeypatch.setattr(datagen.cli, "call_chat", _fake_chat)

    out_path = tmp_path / "out.jsonl"
    code = run(
        [
            "--model",
            "m",
            "--prompts",
            str(prompts_path),
            "--out",
            str(out_path),
            "--api",
            "https://example.com/v1",
            "--apikey",
            "k",
            "--no-progress",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "REQ 1:" in captured.err
    assert "VERBOSE REQ 1:" not in captured.err


def test_run_verbose_prints_extra_diagnostics(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("hello\n", encoding="utf-8")

    def _fake_chat(
        api_base: str,
        api_key: str,
        model: str,
        messages,
        reasoning_effort=None,
        thinking=False,
        timeout_ms=None,
        max_tokens=None,
    ):
        del api_base, api_key, model, messages
        del reasoning_effort, thinking, timeout_ms, max_tokens
        return (
            "ok",
            None,
            Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )

    monkeypatch.setattr(datagen.cli, "call_chat", _fake_chat)

    out_path = tmp_path / "out.jsonl"
    code = run(
        [
            "--model",
            "m",
            "--prompts",
            str(prompts_path),
            "--out",
            str(out_path),
            "--api",
            "https://example.com/v1",
            "--apikey",
            "k",
            "--verbose",
            "--no-progress",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "REQ 1:" in captured.err
    assert "VERBOSE REQ 1:" in captured.err


def test_parse_args_verbose_flag() -> None:
    args = parse_args(["--model", "m", "--prompts", "p.txt", "--verbose"])
    assert args.verbose is True


def test_parse_args_verbose_default_false() -> None:
    args = parse_args(["--model", "m", "--prompts", "p.txt"])
    assert args.verbose is False


def test_parse_args_supports_config_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model: m",
                "prompts: p.txt",
                "out: o.jsonl",
                "api: https://api.openai.com/v1",
                "system: |",
                "  line1",
                "  line2",
                "store-system: false",
                "concurrent: 3",
                "reasoningEffort: high",
                "max-tokens: 300",
                "no-progress: true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    args = parse_args(["--config", str(config_path)])
    assert args.model == "m"
    assert args.prompts_path == "p.txt"
    assert args.out_path == "o.jsonl"
    assert args.api_base == "https://api.openai.com/v1"
    assert args.system_prompt == "line1\nline2\n"
    assert args.store_system is False
    assert args.concurrent == 3
    assert args.reasoning_effort == "high"
    assert args.max_tokens == 300
    assert args.progress is False


def test_parse_args_cli_overrides_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(["model: a", "prompts: p.txt", ""]), encoding="utf-8"
    )
    args = parse_args(["--config", str(config_path), "--model", "b"])
    assert args.model == "b"
    assert args.prompts_path == "p.txt"


def test_run_missing_api_key(tmp_path: Path, monkeypatch) -> None:
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("hello\n", encoding="utf-8")
    monkeypatch.delenv("API_KEY", raising=False)
    code = run(["--model", "m", "--prompts", str(prompt_file)])
    assert code == 1


def test_run_uses_round_robin_and_retries(tmp_path: Path, monkeypatch) -> None:
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("hello\n", encoding="utf-8")

    providers_path = tmp_path / "providers.jsonl"
    providers_path.write_text(
        "\n".join(
            [
                '{"base_url":"https://a.example/v1","apikey":"k1"}',
                '{"base_url":"https://b.example/v1","apikey":"k2"}',
                "",
            ]
        ),
        encoding="utf-8",
    )

    calls: list[tuple[str, str]] = []

    def _fake_chat(
        api_base: str,
        api_key: str,
        model: str,
        messages,
        reasoning_effort=None,
        thinking=False,
        timeout_ms=None,
        max_tokens=None,
    ):
        del model, messages, reasoning_effort, thinking, timeout_ms, max_tokens
        calls.append((api_base, api_key))
        if api_base == "https://a.example/v1":
            raise RuntimeError("temporary upstream failure")
        return (
            "ok",
            None,
            Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )

    monkeypatch.setattr(datagen.cli, "call_chat", _fake_chat)

    out_path = tmp_path / "out.jsonl"
    code = run(
        [
            "--model",
            "m",
            "--prompts",
            str(prompts_path),
            "--out",
            str(out_path),
            "--api",
            "https://example.com/v1",
            "--providers",
            str(providers_path),
            "--no-progress",
        ]
    )

    assert code == 0
    assert calls == [
        ("https://a.example/v1", "k1"),
        ("https://b.example/v1", "k2"),
    ]
    assert out_path.exists() is True


def test_run_provider_model_overrides_default(tmp_path: Path, monkeypatch) -> None:
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("first\n---END---\nsecond\n", encoding="utf-8")

    providers_path = tmp_path / "providers.jsonl"
    providers_path.write_text(
        "\n".join(
            [
                '{"base_url":"https://a.example/v1","apikey":"k1","model":"m-a"}',
                '{"base_url":"https://b.example/v1","apikey":"k2"}',
                "",
            ]
        ),
        encoding="utf-8",
    )

    used_models: list[tuple[str, str]] = []

    def _fake_chat(
        api_base: str,
        api_key: str,
        model: str,
        messages,
        reasoning_effort=None,
        thinking=False,
        timeout_ms=None,
        max_tokens=None,
    ):
        del api_key, messages, reasoning_effort, thinking, timeout_ms, max_tokens
        used_models.append((api_base, model))
        return (
            "ok",
            None,
            Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    monkeypatch.setattr(datagen.cli, "call_chat", _fake_chat)

    out_path = tmp_path / "out.jsonl"
    code = run(
        [
            "--model",
            "m-default",
            "--prompts",
            str(prompts_path),
            "--out",
            str(out_path),
            "--providers",
            str(providers_path),
            "--no-progress",
        ]
    )

    assert code == 0
    assert used_models == [
        ("https://a.example/v1", "m-a"),
        ("https://b.example/v1", "m-default"),
    ]


def test_extract_retry_after_seconds() -> None:
    assert datagen.cli._extract_retry_after_seconds("Retry-After: 45") == 45
    assert datagen.cli._extract_retry_after_seconds("please retry after 60s") == 60
    assert datagen.cli._extract_retry_after_seconds("no hint") is None


def test_is_http_429_error_supports_rate_limit_variants() -> None:
    assert datagen.cli._is_http_429_error("429 Too Many Requests") is True
    assert datagen.cli._is_http_429_error("rate limit reached for RPM") is True
    assert datagen.cli._is_http_429_error("rate_limit_exceeded_error") is True
    assert datagen.cli._is_http_429_error("connection timeout") is False


def test_wait_for_rate_limit_window_uses_shared_gate(monkeypatch) -> None:
    fake_now = {"value": 100.0}
    sleeps: list[int] = []

    def _fake_time() -> float:
        return fake_now["value"]

    def _fake_sleep(seconds: int) -> None:
        sleeps.append(int(seconds))
        fake_now["value"] += float(seconds)

    monkeypatch.setattr(datagen.cli.time, "time", _fake_time)
    monkeypatch.setattr(datagen.cli.time, "sleep", _fake_sleep)

    datagen.cli._RATE_LIMIT_UNTIL_BY_BASE.clear()
    datagen.cli._mark_rate_limited("https://api.example.com/v1", 45)
    datagen.cli._wait_for_rate_limit_window(
        "https://api.example.com/v1",
        request_index=0,
        verbose=False,
    )

    assert sleeps == [45]
    assert datagen.cli._RATE_LIMIT_UNTIL_BY_BASE == {}


def test_run_429_sleeps_then_retries(tmp_path: Path, monkeypatch) -> None:
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("hello\n", encoding="utf-8")

    providers_path = tmp_path / "providers.jsonl"
    providers_path.write_text(
        "\n".join(
            [
                '{"base_url":"https://a.example/v1","apikey":"k1"}',
                '{"base_url":"https://b.example/v1","apikey":"k2"}',
                "",
            ]
        ),
        encoding="utf-8",
    )

    calls: list[str] = []
    sleeps: list[int] = []

    def _fake_sleep(seconds: int) -> None:
        sleeps.append(int(seconds))

    def _fake_chat(
        api_base: str,
        api_key: str,
        model: str,
        messages,
        reasoning_effort=None,
        thinking=False,
        timeout_ms=None,
        max_tokens=None,
    ):
        del api_key, model, messages, reasoning_effort, thinking, timeout_ms
        del max_tokens
        calls.append(api_base)
        if api_base == "https://a.example/v1":
            raise RuntimeError("429 Too Many Requests; Retry-After: 45")
        return (
            "ok",
            None,
            Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    monkeypatch.setattr(datagen.cli.time, "sleep", _fake_sleep)
    monkeypatch.setattr(datagen.cli, "call_chat", _fake_chat)

    out_path = tmp_path / "out.jsonl"
    code = run(
        [
            "--model",
            "m",
            "--prompts",
            str(prompts_path),
            "--out",
            str(out_path),
            "--providers",
            str(providers_path),
            "--no-progress",
        ]
    )

    assert code == 0
    assert calls == ["https://a.example/v1", "https://b.example/v1"]
    assert sleeps == [45]
