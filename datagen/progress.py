"""Progress helpers for datagen."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

PROMPT_DELIMITER = "---END---"


def count_prompts(file_path: str) -> int:
    """Counts prompts in a prompt file using delimiter-based grouping."""
    count = 0
    has_content = False
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\n")
            if line.strip() == PROMPT_DELIMITER:
                if has_content:
                    count += 1
                    has_content = False
            elif line.strip():
                has_content = True
    if has_content:
        count += 1
    return count


def _format_duration(ms: float) -> str:
    total_seconds = int(ms // 1000)
    seconds = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


@dataclass(frozen=True)
class ProgressStats:
    """Progress counters displayed in progress bar."""

    ok: int
    err: int


class ProgressBar:
    """Terminal progress bar renderer based on rich.Progress."""

    def __init__(self, total: int, stream: TextIO | None = None) -> None:
        self._total = max(0, total)
        self._stream = stream if stream is not None else sys.stderr
        self._start = time.time()
        self._progress = Progress(
            TextColumn("{task.percentage:>3.0f}%"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("ok={task.fields[ok]} err={task.fields[err]}"),
            TextColumn("rps={task.fields[rps]}"),
            TimeElapsedColumn(),
            transient=False,
            expand=True,
        )
        self._task_id: TaskID | None = None
        self._started = False

    def _start_if_needed(self) -> None:
        if self._started:
            return
        self._progress.start()
        self._task_id = self._progress.add_task(
            "requests",
            total=self._total,
            ok=0,
            err=0,
            rps="0.00/s",
        )
        self._started = True

    def write_line(self, text: str) -> None:
        self._start_if_needed()
        self._progress.console.print(text)

    def render(self, current: int, stats: ProgressStats | None = None) -> None:
        self._start_if_needed()
        if self._task_id is None:
            return
        safe_current = max(0, current)
        safe_completed = min(safe_current, self._total)
        elapsed_seconds = max(0.001, time.time() - self._start)
        rps = f"{(safe_completed / elapsed_seconds):.2f}/s"
        ok = 0
        err = 0
        if stats is not None:
            ok = stats.ok
            err = stats.err
        self._progress.update(
            self._task_id,
            completed=safe_completed,
            ok=ok,
            err=err,
            rps=rps,
        )

    def finish(self, current: int, stats: ProgressStats | None = None) -> None:
        self.render(current, stats)
        if self._started:
            self._progress.stop()
