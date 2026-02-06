"""Tests for CLI argument parsing and behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from datagen.cli import parse_args, run


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


def test_parse_args_openrouter_flags() -> None:
    args = parse_args(
        [
            "--model",
            "m",
            "--prompts",
            "p.txt",
            "--openrouter.provider",
            "openai, anthropic",
            "--openrouter.providerSort",
            "throughput",
        ]
    )
    assert args.openrouter_provider_order == ["openai", "anthropic"]
    assert args.openrouter_provider_sort == "throughput"


def test_parse_args_reasoning_effort() -> None:
    args = parse_args(
        ["--model", "m", "--prompts", "p.txt", "--reasoningEffort", "high"]
    )
    assert args.reasoning_effort == "high"


def test_parse_args_supports_config_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model: m",
                "prompts: p.txt",
                "out: o.jsonl",
                "api: https://openrouter.ai/api/v1",
                "system: |",
                "  line1",
                "  line2",
                "store-system: false",
                "concurrent: 3",
                "openrouter:",
                "  provider:",
                "    - openai",
                "    - anthropic",
                "  providerSort: throughput",
                "reasoningEffort: high",
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
    assert args.api_base == "https://openrouter.ai/api/v1"
    assert args.system_prompt == "line1\nline2\n"
    assert args.store_system is False
    assert args.concurrent == 3
    assert args.openrouter_provider_order == ["openai", "anthropic"]
    assert args.openrouter_provider_sort == "throughput"
    assert args.reasoning_effort == "high"
    assert args.progress is False


def test_parse_args_cli_overrides_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(["model: a", "prompts: p.txt", ""]), encoding="utf-8"
    )
    args = parse_args(["--config", str(config_path), "--model", "b"])
    assert args.model == "b"
    assert args.prompts_path == "p.txt"


def test_run_missing_api_key(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("hello\n", encoding="utf-8")
    code = run(["--model", "m", "--prompts", str(prompt_file)])
    assert code == 1
