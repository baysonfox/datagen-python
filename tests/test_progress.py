"""Tests for progress helpers."""

from __future__ import annotations

from pathlib import Path

from datagen.progress import count_prompts


def test_count_prompts_delimiter_mode(tmp_path: Path) -> None:
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text(
        "\n".join(
            [
                "First line",
                "second line",
                "---END---",
                "",
                "Third prompt",
                "---END---",
                "last block",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert count_prompts(str(prompts_path)) == 3
