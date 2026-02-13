"""Checkpoint persistence for resumable runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckpointState:
    """In-memory checkpoint state for completed and skipped prompts."""

    completed_results: dict[int, str]
    skipped_indices: set[int]


def checkpoint_path_for(out_path: str) -> str:
    return out_path + ".ckpt"


def load_checkpoint(ckpt_path: str) -> CheckpointState:
    path = Path(ckpt_path)
    if not path.exists():
        return CheckpointState(completed_results={}, skipped_indices=set())
    results: dict[int, str] = {}
    skipped_indices: set[int] = set()
    with path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            idx = record.get("_idx")
            if not isinstance(idx, int):
                continue
            status = record.get("_status")
            if status == "skipped":
                skipped_indices.add(idx)
                results.pop(idx, None)
                continue
            payload = record.get("_payload")
            if isinstance(payload, str):
                results[idx] = payload
                skipped_indices.discard(idx)
    return CheckpointState(completed_results=results, skipped_indices=skipped_indices)


def append_checkpoint(ckpt_path: str, idx: int, payload: str) -> None:
    record = json.dumps({"_idx": idx, "_payload": payload}, ensure_ascii=False)
    with Path(ckpt_path).open("a", encoding="utf-8") as stream:
        stream.write(record + "\n")


def append_checkpoint_skipped(ckpt_path: str, idx: int) -> None:
    """Appends one skipped prompt marker to checkpoint."""
    record = json.dumps({"_idx": idx, "_status": "skipped"}, ensure_ascii=False)
    with Path(ckpt_path).open("a", encoding="utf-8") as stream:
        stream.write(record + "\n")


def finalize_checkpoint(
    ckpt_path: str,
    out_path: str,
    results: dict[int, str],
    total: int,
) -> None:
    with Path(out_path).open("w", encoding="utf-8") as stream:
        for idx in range(total):
            line = results.get(idx)
            if line is not None:
                stream.write(line + "\n")
    path = Path(ckpt_path)
    if path.exists():
        path.unlink()


def remove_checkpoint(ckpt_path: str) -> None:
    path = Path(ckpt_path)
    if path.exists():
        path.unlink()
