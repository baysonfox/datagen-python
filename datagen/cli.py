"""CLI implementation for datagen."""

from __future__ import annotations

import json
import math
import os
import queue
import re
import signal
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datagen import __version__
from datagen.api import FinishReasonLengthError, call_chat
from datagen.checkpoint import (
    append_checkpoint,
    append_checkpoint_skipped,
    checkpoint_path_for,
    finalize_checkpoint,
    load_checkpoint,
)
from datagen.config import load_config_raw_args
from datagen.messages import (
    build_output_messages,
    build_request_messages,
    format_assistant_content,
)
from datagen.progress import PROMPT_DELIMITER, ProgressBar, ProgressStats


@dataclass(frozen=True)
class Args:
    """Parsed CLI args."""

    model: str
    prompts_path: str
    out_path: str
    api_base: str
    api_key: str | None
    system_prompt: str
    store_system: bool
    progress: bool
    concurrent: int
    reasoning_effort: str | None
    thinking: bool
    timeout: int | None
    max_tokens: int | None
    providers_path: str | None
    verbose: bool


@dataclass(frozen=True)
class Endpoint:
    """API endpoint configuration for one provider."""

    api_base: str
    api_key: str
    model: str | None = None


class ProviderSlotPool:
    """Manages fixed provider slots for request scheduling."""

    def __init__(self, endpoint_pool: list[Endpoint], concurrent: int) -> None:
        if not endpoint_pool:
            raise ValueError("endpoint_pool must not be empty")
        self._endpoint_pool = endpoint_pool
        self._available_slots: queue.Queue[int] = queue.Queue()
        self._slot_to_endpoint_index: list[int] = []

        provider_count = len(endpoint_pool)
        base_slots = concurrent // provider_count
        remainder_slots = concurrent % provider_count
        for endpoint_index in range(provider_count):
            provider_slots = base_slots + (1 if endpoint_index < remainder_slots else 0)
            for _ in range(provider_slots):
                slot_index = len(self._slot_to_endpoint_index)
                self._slot_to_endpoint_index.append(endpoint_index)
                self._available_slots.put(slot_index)

        if not self._slot_to_endpoint_index:
            slot_index = 0
            self._slot_to_endpoint_index.append(0)
            self._available_slots.put(slot_index)

    def acquire(self) -> tuple[int, Endpoint]:
        """Acquires one free slot and returns (slot_index, endpoint)."""
        slot_index = self._available_slots.get(block=True)
        endpoint_index = self._slot_to_endpoint_index[slot_index]
        return slot_index, self._endpoint_pool[endpoint_index]

    def release(self, slot_index: int) -> None:
        """Releases one slot back to pool."""
        self._available_slots.put(slot_index)

    @property
    def slot_count(self) -> int:
        """Returns total slot count."""
        return len(self._slot_to_endpoint_index)


def _usage_line() -> str:
    return "Usage: datagen --model <model> --prompts <file> [options]"


def _help_text() -> str:
    lines = [
        f"datagen {__version__}",
        "",
        _usage_line(),
        "",
        "Options:",
        "  --help                          Show this help message and exit.",
        "  --version                       Print the CLI version and exit.",
        "  --config <file>                 Load options from a YAML/JSON config file.",
        "  --model <name>                  Model name to use for completions.",
        "  --prompts <file>                Path to a prompts file (multi-line supported).",
        "  --out <file>                    Output JSONL file (default: dataset.jsonl).",
        "  --api <baseUrl>                 API base URL (default: https://api.openai.com/v1).",
        "  --apikey <key>                  API key (overrides API_KEY env var).",
        "  --system <text>                 Optional system prompt to include.",
        "  --store-system true|false       Whether to emit the system prompt in dataset.",
        "  --concurrent <num>              Number of parallel requests (default: 1).",
        "  --reasoningEffort <level>       Reasoning effort.",
        "  --thinking                      Enable extended thinking.",
        "  --timeout <ms>                  Request timeout in milliseconds.",
        "  --max-tokens <num>              Max completion tokens per request.",
        "  --providers <file>              JSONL file with base_url/apikey/model.",
        "  --verbose                       Print per-request details.",
        "  --no-progress                   Disable progress bar.",
        "",
    ]
    return "\n".join(lines)


def _parse_raw_args(argv: list[str]) -> dict[str, str | bool]:
    args: dict[str, str | bool] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            index += 1
            continue
        if "=" in token:
            key, value = token[2:].split("=", 1)
            if key:
                args[key] = value
            index += 1
            continue
        key = token[2:]
        next_token = argv[index + 1] if index + 1 < len(argv) else None
        if next_token is None or next_token.startswith("--"):
            args[key] = True
            index += 1
        else:
            args[key] = next_token
            index += 2
    return args


def _parse_args_from_raw(raw: dict[str, str | bool]) -> Args:
    model = str(raw.get("model", "") or "")
    prompts_path = str(raw.get("prompts", "") or "")
    out_path = str(raw.get("out", "dataset.jsonl") or "dataset.jsonl")
    api_base = str(raw.get("api", "https://api.openai.com/v1") or "")
    system_prompt = str(raw.get("system", "") or "")

    store_raw = raw.get("store-system")
    store_system = True
    if store_raw is not None:
        store_system = str(store_raw).lower() != "false"

    progress_raw = raw.get("progress")
    progress = True if progress_raw is None else str(progress_raw).lower() != "false"
    if raw.get("no-progress") is not None:
        progress = False

    concurrent_raw = raw.get("concurrent")
    concurrent = 1
    if concurrent_raw is not None:
        try:
            value = int(float(str(concurrent_raw)))
            concurrent = value if value > 0 else 1
        except ValueError:
            concurrent = 1

    reasoning_effort_raw = raw.get("reasoningEffort")
    reasoning_effort = (
        reasoning_effort_raw.strip()
        if isinstance(reasoning_effort_raw, str) and reasoning_effort_raw.strip()
        else None
    )

    thinking = raw.get("thinking") is not None

    timeout_raw = raw.get("timeout")
    timeout: int | None = None
    if timeout_raw is not None:
        try:
            parsed = int(float(str(timeout_raw)))
            if parsed > 0:
                timeout = parsed
        except ValueError:
            timeout = None

    max_tokens_raw = raw.get("max-tokens")
    max_tokens: int | None = None
    if max_tokens_raw is not None:
        try:
            parsed = int(float(str(max_tokens_raw)))
            if parsed > 0:
                max_tokens = parsed
        except ValueError:
            max_tokens = None

    apikey_raw = raw.get("apikey")
    api_key = (
        apikey_raw.strip()
        if isinstance(apikey_raw, str) and apikey_raw.strip()
        else None
    )

    providers_raw = raw.get("providers")
    providers_path = (
        providers_raw.strip()
        if isinstance(providers_raw, str) and providers_raw.strip()
        else None
    )

    verbose = raw.get("verbose") is not None

    if not model or not prompts_path:
        raise ValueError(
            _usage_line()
            + " [--out dataset.jsonl] [--api https://api.openai.com/v1]"
            + ' [--system "..."] [--store-system true|false]'
            + " [--concurrent 1]"
            + " [--reasoningEffort low|medium|high] [--timeout <ms>]"
            + " [--max-tokens <num>]"
            + " [--providers endpoints.jsonl]"
            + " [--verbose]"
            + " [--no-progress]"
        )

    return Args(
        model=model,
        prompts_path=prompts_path,
        out_path=out_path,
        api_base=api_base,
        api_key=api_key,
        system_prompt=system_prompt,
        store_system=store_system,
        progress=progress,
        concurrent=concurrent,
        reasoning_effort=reasoning_effort,
        thinking=thinking,
        timeout=timeout,
        max_tokens=max_tokens,
        providers_path=providers_path,
        verbose=verbose,
    )


def parse_args(argv: list[str]) -> Args:
    """Parses CLI and config arguments with CLI precedence."""
    cli_raw = _parse_raw_args(argv)
    config_raw_value = cli_raw.get("config")
    if config_raw_value is not None and not isinstance(config_raw_value, str):
        raise ValueError(f"{_usage_line()} --config <file>")

    config_path = None
    if isinstance(config_raw_value, str) and config_raw_value.strip():
        config_path = str(Path(config_raw_value).resolve())

    merged: dict[str, str | bool]
    if config_path is not None:
        merged = {**load_config_raw_args(config_path), **cli_raw}
    else:
        merged = cli_raw
    return _parse_args_from_raw(merged)


def _ensure_readable_file(file_path: str) -> None:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {file_path}")
    with path.open("r", encoding="utf-8"):
        return


def _read_prompts(file_path: str) -> list[tuple[str, int]]:
    prompts: list[tuple[str, int]] = []
    current_lines: list[str] = []
    line_num = 0
    with Path(file_path).open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line_num += 1
            line = raw_line.rstrip("\n")
            if line.strip() == PROMPT_DELIMITER:
                prompt = "\n".join(current_lines).strip()
                if prompt:
                    prompts.append((prompt, line_num))
                current_lines = []
            else:
                current_lines.append(line)
    tail = "\n".join(current_lines).strip()
    if tail:
        prompts.append((tail, line_num))
    return prompts


def _load_endpoints_from_jsonl(file_path: str) -> list[Endpoint]:
    endpoints: list[Endpoint] = []
    with Path(file_path).open("r", encoding="utf-8") as stream:
        for index, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at providers line {index}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Providers line {index} must be a JSON object.")

            base_url = payload.get("base_url")
            if not isinstance(base_url, str) or not base_url.strip():
                raise ValueError(
                    f"Providers line {index} missing non-empty 'base_url'."
                )

            api_key = payload.get("apikey")
            if not isinstance(api_key, str) or not api_key.strip():
                raise ValueError(f"Providers line {index} missing non-empty 'apikey'.")

            model_raw = payload.get("model")
            model = (
                model_raw.strip()
                if isinstance(model_raw, str) and model_raw.strip()
                else None
            )

            endpoints.append(
                Endpoint(
                    api_base=base_url.strip(),
                    api_key=api_key.strip(),
                    model=model,
                )
            )

    if not endpoints:
        raise ValueError("Providers file has no valid endpoints.")
    return endpoints


def _preview_text(text: str, max_chars: int = 50) -> str:
    """Builds a single-line preview for logs."""
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars]


def _format_request_result_line(
    index: int,
    prompt_preview: str,
    result_preview: str,
    output_tokens: int,
) -> str:
    """Formats per-request summary line."""
    return (
        f"REQ {index + 1}: "
        f'Q="{prompt_preview}" '
        f'R="{result_preview}" '
        f"output_tokens={output_tokens}"
    )


def _format_verbose_line(index: int, message: str) -> str:
    """Formats one verbose diagnostic line."""
    return f"VERBOSE REQ {index + 1}: {message}"


_RATE_LIMIT_UNTIL_BY_BASE: dict[str, float] = {}
_RATE_LIMIT_LOCK = threading.Lock()


def _normalized_base_url(api_base: str) -> str:
    """Normalizes endpoint URL for rate-limit map keys."""
    return api_base.rstrip("/")


def _mark_rate_limited(api_base: str, wait_seconds: int) -> None:
    """Marks one endpoint as rate-limited until now + wait_seconds."""
    until = time.time() + max(0, wait_seconds)
    base_key = _normalized_base_url(api_base)
    with _RATE_LIMIT_LOCK:
        existing = _RATE_LIMIT_UNTIL_BY_BASE.get(base_key, 0.0)
        if until > existing:
            _RATE_LIMIT_UNTIL_BY_BASE[base_key] = until


def _wait_for_rate_limit_window(
    api_base: str, request_index: int, verbose: bool
) -> None:
    """Waits if endpoint is currently in a shared rate-limit cooldown."""
    base_key = _normalized_base_url(api_base)
    while True:
        now = time.time()
        with _RATE_LIMIT_LOCK:
            until = _RATE_LIMIT_UNTIL_BY_BASE.get(base_key)
            if until is None or until <= now:
                if until is not None:
                    del _RATE_LIMIT_UNTIL_BY_BASE[base_key]
                return
            sleep_seconds = max(1, int(math.ceil(until - now)))
        if verbose:
            sys.stderr.write(
                _format_verbose_line(
                    request_index,
                    f"cooldown base={api_base} waiting={sleep_seconds}s",
                )
                + "\n"
            )
        time.sleep(sleep_seconds)


def _extract_retry_after_seconds(error_text: str) -> int | None:
    """Extracts retry-after seconds from an error string."""
    patterns = [
        r"retry[- ]?after[^0-9]*(\d+)",
        r"after[^0-9]*(\d+)\s*s",
    ]
    lowered = error_text.lower()
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match is None:
            continue
        try:
            seconds = int(match.group(1))
        except ValueError:
            continue
        if seconds > 0:
            return seconds
    return None


def _is_http_429_error(error_text: str) -> bool:
    """Checks whether an exception message indicates HTTP 429."""
    lowered = error_text.lower()
    return (
        "429" in lowered
        or "too many requests" in lowered
        or "rate limit reached" in lowered
        or "rate_limit_exceeded" in lowered
    )


def _wait_seconds_for_429(error_text: str) -> int:
    """Calculates cooldown seconds for HTTP 429."""
    retry_after = _extract_retry_after_seconds(error_text)
    return retry_after if retry_after is not None else 60


def _process_one_prompt(
    idx: int,
    prompt: str,
    line_num: int,
    args: Args,
    api_key: str,
    slot_pool: ProviderSlotPool,
) -> tuple[int, str | None, str | None, int, str, str, int]:
    prompt_preview = _preview_text(prompt)
    try:
        messages = build_request_messages(args.system_prompt, prompt)
        max_attempts = max(2, slot_pool.slot_count)

        content = ""
        reasoning = None
        usage = None
        last_error = "Unknown error"
        for attempt_index in range(max_attempts):
            slot_index, endpoint = slot_pool.acquire()
            request_model = endpoint.model if endpoint.model is not None else args.model
            try:
                _wait_for_rate_limit_window(endpoint.api_base, idx, args.verbose)
                if args.verbose:
                    sys.stderr.write(
                        _format_verbose_line(
                            idx,
                            (
                                f"attempt={attempt_index + 1}/{max_attempts} "
                                f"slot={slot_index} "
                                f"base={endpoint.api_base} model={request_model}"
                            ),
                        )
                        + "\n"
                    )
                content, reasoning, usage = call_chat(
                    endpoint.api_base,
                    endpoint.api_key,
                    request_model,
                    messages,
                    reasoning_effort=args.reasoning_effort,
                    thinking=args.thinking,
                    timeout_ms=args.timeout,
                    max_tokens=args.max_tokens,
                )
                if args.verbose:
                    completion_tokens = (
                        usage.completion_tokens if usage is not None else 0
                    )
                    sys.stderr.write(
                        _format_verbose_line(
                            idx,
                            (
                                f"slot={slot_index} "
                                f"success base={endpoint.api_base} "
                                f"completion_tokens={completion_tokens}"
                            ),
                        )
                        + "\n"
                    )
                break
            except FinishReasonLengthError:
                return (
                    idx,
                    None,
                    "SKIP_FINISH_REASON_LENGTH",
                    line_num,
                    prompt_preview,
                    "finish_reason=length",
                    0,
                )
            except Exception as exc:
                exc_text = str(exc)
                if _is_http_429_error(exc_text):
                    wait_seconds = _wait_seconds_for_429(exc_text)
                    _mark_rate_limited(endpoint.api_base, wait_seconds)
                    sys.stderr.write(
                        "WARN: HTTP 429 from "
                        f"{endpoint.api_base}, sleeping {wait_seconds}s before retry.\n"
                    )
                    time.sleep(wait_seconds)
                last_error = (
                    f"attempt {attempt_index + 1}/{max_attempts} "
                    f"slot={slot_index} "
                    f"base={endpoint.api_base}: {exc_text}"
                )
                if args.verbose:
                    sys.stderr.write(
                        _format_verbose_line(
                            idx,
                            (
                                f"error attempt={attempt_index + 1}/{max_attempts} "
                                f"slot={slot_index} "
                                f"base={endpoint.api_base} detail={exc_text}"
                            ),
                        )
                        + "\n"
                    )
            finally:
                slot_pool.release(slot_index)
        else:
            raise RuntimeError(last_error)

        assistant_content = format_assistant_content(content, reasoning)
        output_messages = build_output_messages(
            args.system_prompt,
            prompt,
            assistant_content,
            args.store_system,
        )
        line = json.dumps({"messages": output_messages}, ensure_ascii=False)
        output_tokens = usage.completion_tokens if usage is not None else 0
        return (
            idx,
            line,
            None,
            line_num,
            prompt_preview,
            _preview_text(assistant_content),
            output_tokens,
        )
    except Exception as exc:
        return (
            idx,
            None,
            str(exc),
            line_num,
            prompt_preview,
            _preview_text(str(exc)),
            0,
        )


def run(argv: list[str]) -> int:
    if "--help" in argv or "-h" in argv:
        print(_help_text())
        return 0
    if "--version" in argv or "-v" in argv:
        print(f"datagen {__version__}")
        return 0

    try:
        args = parse_args(argv)
    except Exception as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    endpoint_pool: list[Endpoint]
    if args.providers_path is not None:
        abs_providers = str(Path(args.providers_path).resolve())
        try:
            _ensure_readable_file(abs_providers)
            endpoint_pool = _load_endpoints_from_jsonl(abs_providers)
        except Exception as exc:
            sys.stderr.write(f"{exc}\n")
            return 1
        api_key = endpoint_pool[0].api_key
    else:
        api_key = args.api_key or os.environ.get("API_KEY")
        if not api_key:
            sys.stderr.write(
                "Missing API key. Provide --apikey or set API_KEY env var.\n"
            )
            return 1
        endpoint_pool = [Endpoint(api_base=args.api_base, api_key=api_key)]

    abs_prompts = str(Path(args.prompts_path).resolve())
    abs_out = str(Path(args.out_path).resolve())
    try:
        _ensure_readable_file(abs_prompts)
    except Exception as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    prompts = _read_prompts(abs_prompts)

    ckpt_path = checkpoint_path_for(abs_out)
    checkpoint_state = load_checkpoint(ckpt_path)
    completed_results = checkpoint_state.completed_results
    skipped_indices = checkpoint_state.skipped_indices
    if completed_results:
        sys.stderr.write(
            f"Resuming: {len(completed_results)}/{len(prompts)} prompts already done.\n"
        )
    if skipped_indices:
        sys.stderr.write(f"Resuming: {len(skipped_indices)} prompts already skipped.\n")

    pending: list[tuple[int, str, int]] = []
    for idx, (prompt, line_num) in enumerate(prompts):
        if idx not in completed_results and idx not in skipped_indices:
            pending.append((idx, prompt, line_num))

    total = len(prompts)
    use_progress = args.progress and sys.stderr.isatty()
    bar = ProgressBar(total, stream=sys.stderr) if use_progress and total > 0 else None
    slot_pool = ProviderSlotPool(endpoint_pool, args.concurrent)

    ok_count = len(completed_results)
    err_count = 0
    skipped_count = len(skipped_indices)

    shutdown_event = threading.Event()

    original_sigint = signal.getsignal(signal.SIGINT)

    def _handle_sigint(signum: int, frame: Any) -> None:
        if shutdown_event.is_set():
            sys.stderr.write("\nForce quit.\n")
            raise KeyboardInterrupt
        shutdown_event.set()
        sys.stderr.write(
            "\nInterrupted. Waiting for in-flight requests… "
            "Press Ctrl+C again to force quit.\n"
        )

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrent)) as executor:
            futures: dict[
                Future[tuple[int, str | None, str | None, int, str, str, int]],
                int,
            ] = {}
            for idx, prompt, line_num in pending:
                if shutdown_event.is_set():
                    break
                future = executor.submit(
                    _process_one_prompt,
                    idx,
                    prompt,
                    line_num,
                    args,
                    api_key,
                    slot_pool,
                )
                futures[future] = idx

            for future in as_completed(futures):
                (
                    result_idx,
                    line,
                    error,
                    result_line_num,
                    prompt_preview,
                    result_preview,
                    output_tokens,
                ) = future.result()
                if error is None and line is not None:
                    completed_results[result_idx] = line
                    append_checkpoint(ckpt_path, result_idx, line)
                    ok_count += 1
                elif error == "SKIP_FINISH_REASON_LENGTH":
                    skipped_count += 1
                    skipped_indices.add(result_idx)
                    append_checkpoint_skipped(ckpt_path, result_idx)
                    sys.stderr.write(
                        f"SKIP line {result_line_num}: finish_reason=length\n"
                    )
                else:
                    err_count += 1
                    sys.stderr.write(f"ERR line {result_line_num}: {error}\n")

                request_line = _format_request_result_line(
                    result_idx,
                    prompt_preview,
                    result_preview,
                    output_tokens,
                )
                if bar is not None:
                    bar.write_line(request_line)
                else:
                    sys.stderr.write(request_line + "\n")

                if bar is not None:
                    bar.render(
                        ok_count + err_count + skipped_count,
                        ProgressStats(
                            ok=ok_count,
                            err=err_count,
                            skipped=skipped_count,
                        ),
                    )
    finally:
        signal.signal(signal.SIGINT, original_sigint)

    if bar is not None:
        bar.finish(
            ok_count + err_count + skipped_count,
            ProgressStats(
                ok=ok_count,
                err=err_count,
                skipped=skipped_count,
            ),
        )

    if shutdown_event.is_set():
        sys.stderr.write(
            f"Checkpoint saved: {ok_count}/{total} done. "
            f"Re-run same command to resume.\n"
        )
        return 130

    finalize_checkpoint(ckpt_path, abs_out, completed_results, total)
    return 0


def entrypoint() -> None:
    """Console script entrypoint."""
    raise SystemExit(run(sys.argv[1:]))
