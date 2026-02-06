"""Progress helpers for datagen."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

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


def _trim_trailing_zeros(number: str) -> str:
    if "." not in number:
        return number
    trimmed = number.rstrip("0").rstrip(".")
    return trimmed if trimmed else "0"


def _format_usd(amount: float) -> str:
    if amount != amount:
        return "$0"
    absolute = abs(amount)
    if absolute == 0:
        return "$0"
    if absolute >= 10:
        decimals = 2
    elif absolute >= 1:
        decimals = 4
    elif absolute >= 0.01:
        decimals = 6
    elif absolute >= 0.0001:
        decimals = 8
    else:
        decimals = 10
    rounded = round(amount, decimals)
    if rounded == 0:
        minimum = 1 / (10**decimals)
        min_label = _trim_trailing_zeros(f"{minimum:.{decimals}f}")
        return f">-${min_label}" if amount < 0 else f"<${min_label}"
    return f"${_trim_trailing_zeros(f'{amount:.{decimals}f}')}"


@dataclass(frozen=True)
class ProgressStats:
    """Progress counters displayed in progress bar."""

    ok: int
    err: int
    spent_usd: float | None = None


class ProgressBar:
    """Terminal progress bar renderer."""

    def __init__(self, total: int, stream: TextIO | None = None) -> None:
        self._total = max(0, total)
        self._stream = stream if stream is not None else sys.stderr
        self._start = time.time()
        self._last_line = ""

    def _clear_line(self) -> None:
        self._stream.write("\x1b[2K\r")

    def write_line(self, text: str) -> None:
        self._clear_line()
        self._stream.write(text + "\n")
        self._stream.flush()
        self._last_line = ""

    def render(self, current: int, stats: ProgressStats | None = None) -> None:
        safe_current = max(0, current)
        denominator = self._total if self._total > 0 else 1
        pct = min(1.0, safe_current / denominator)
        percent_label = f"{int(pct * 100):3d}%"

        columns = 80
        suffix_base = f" {safe_current}/{self._total}"
        suffix_stats = ""
        suffix_money = ""
        if stats is not None:
            suffix_stats = f" ok={stats.ok} err={stats.err}"
            if stats.spent_usd is not None:
                suffix_money = f" spent={_format_usd(stats.spent_usd)}"
        duration = _format_duration((time.time() - self._start) * 1000)
        suffix = f"{suffix_base}{suffix_stats}{suffix_money} {duration}"

        bar_width = max(10, min(40, columns - len(suffix) - 10))
        filled = round(bar_width * pct)
        bar = "█" * filled + "░" * max(0, bar_width - filled)
        line = f"{percent_label} [{bar}]{suffix}"
        if line == self._last_line:
            return

        self._clear_line()
        self._stream.write(line)
        self._stream.flush()
        self._last_line = line

    def finish(self, current: int, stats: ProgressStats | None = None) -> None:
        self.render(current, stats)
        self._stream.write("\n")
        self._stream.flush()
