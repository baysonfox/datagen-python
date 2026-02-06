"""Tests for config parsing."""

from __future__ import annotations

from pathlib import Path

from datagen.config import load_config_raw_args


def test_load_config_raw_args_yaml(tmp_path: Path) -> None:
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
    result = load_config_raw_args(str(config_path))
    assert result["model"] == "m"
    assert result["prompts"] == "p.txt"
    assert result["out"] == "o.jsonl"
    assert result["api"] == "https://openrouter.ai/api/v1"
    assert result["system"] == "line1\nline2\n"
    assert result["store-system"] is False
    assert result["concurrent"] == "3"
    assert result["openrouter.provider"] == "openai,anthropic"
    assert result["openrouter.providerSort"] == "throughput"
    assert result["reasoningEffort"] == "high"
    assert result["no-progress"] is True
