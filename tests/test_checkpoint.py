"""Tests for checkpoint module."""

from __future__ import annotations

import json
from pathlib import Path

from datagen.checkpoint import (
    append_checkpoint,
    checkpoint_path_for,
    finalize_checkpoint,
    load_checkpoint,
    remove_checkpoint,
)


def test_checkpoint_path_for() -> None:
    assert checkpoint_path_for("/tmp/out.jsonl") == "/tmp/out.jsonl.ckpt"


def test_load_checkpoint_empty(tmp_path: Path) -> None:
    ckpt = str(tmp_path / "missing.ckpt")
    assert load_checkpoint(ckpt) == {}


def test_append_and_load_checkpoint(tmp_path: Path) -> None:
    ckpt = str(tmp_path / "test.ckpt")
    append_checkpoint(ckpt, 0, '{"messages":[]}')
    append_checkpoint(ckpt, 2, '{"messages":["hi"]}')

    loaded = load_checkpoint(ckpt)
    assert loaded == {0: '{"messages":[]}', 2: '{"messages":["hi"]}'}


def test_load_checkpoint_skips_invalid_lines(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "broken.ckpt"
    ckpt_path.write_text(
        "not json\n"
        + json.dumps({"_idx": "bad", "_payload": "x"})
        + "\n"
        + json.dumps({"_idx": 1, "_payload": "good"})
        + "\n",
        encoding="utf-8",
    )
    loaded = load_checkpoint(str(ckpt_path))
    assert loaded == {1: "good"}


def test_finalize_checkpoint_writes_ordered_output(tmp_path: Path) -> None:
    ckpt = str(tmp_path / "test.ckpt")
    out = str(tmp_path / "final.jsonl")
    append_checkpoint(ckpt, 2, "line_2")
    append_checkpoint(ckpt, 0, "line_0")
    append_checkpoint(ckpt, 1, "line_1")

    results = load_checkpoint(ckpt)
    finalize_checkpoint(ckpt, out, results, total=3)

    lines = Path(out).read_text(encoding="utf-8").strip().split("\n")
    assert lines == ["line_0", "line_1", "line_2"]
    assert not Path(ckpt).exists()


def test_finalize_skips_missing_indices(tmp_path: Path) -> None:
    ckpt = str(tmp_path / "partial.ckpt")
    out = str(tmp_path / "partial.jsonl")
    append_checkpoint(ckpt, 0, "line_0")
    append_checkpoint(ckpt, 2, "line_2")

    results = load_checkpoint(ckpt)
    finalize_checkpoint(ckpt, out, results, total=3)

    lines = Path(out).read_text(encoding="utf-8").strip().split("\n")
    assert lines == ["line_0", "line_2"]


def test_remove_checkpoint(tmp_path: Path) -> None:
    ckpt = str(tmp_path / "to_delete.ckpt")
    Path(ckpt).write_text("data", encoding="utf-8")
    remove_checkpoint(ckpt)
    assert not Path(ckpt).exists()


def test_remove_checkpoint_nonexistent(tmp_path: Path) -> None:
    remove_checkpoint(str(tmp_path / "nope.ckpt"))
