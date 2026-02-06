"""CLI implementation for datagen."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datagen import __version__
from datagen.config import load_config_raw_args
from datagen.messages import (
    build_output_messages,
    build_request_messages,
    format_assistant_content,
)
from datagen.openrouter import (
    OpenRouterModelPricing,
    calculate_openrouter_spend_usd,
    call_openai_compatible_chat,
    get_openrouter_model_pricing,
    is_openrouter_api_base,
)
from datagen.progress import PROMPT_DELIMITER, ProgressBar, ProgressStats, count_prompts


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
    openrouter_provider_order: list[str] | None
    openrouter_provider_sort: str | None
    reasoning_effort: str | None
    timeout: int | None


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
        "  --api <baseUrl>                 API base URL (default: https://openrouter.ai/api/v1).",
        "  --apikey <key>                  API key (overrides API_KEY env var).",
        "  --system <text>                 Optional system prompt to include.",
        "  --store-system true|false       Whether to emit the system prompt in dataset.",
        "  --concurrent <num>              Number of parallel requests (default: 1).",
        "  --openrouter.provider <slugs>   OpenRouter provider slugs (comma-separated).",
        "  --openrouter.providerSort <x>   Provider sorting (price|throughput|latency).",
        "  --reasoningEffort <level>       Reasoning effort.",
        "  --timeout <ms>                  Request timeout in milliseconds.",
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
    api_base = str(raw.get("api", "https://openrouter.ai/api/v1") or "")
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

    provider_raw = raw.get("openrouter.provider")
    provider_order = None
    if isinstance(provider_raw, str) and provider_raw.strip():
        provider_order = [
            item.strip() for item in provider_raw.split(",") if item.strip()
        ]

    provider_sort_raw = raw.get("openrouter.providerSort")
    provider_sort = (
        provider_sort_raw.strip()
        if isinstance(provider_sort_raw, str) and provider_sort_raw.strip()
        else None
    )

    reasoning_effort_raw = raw.get("reasoningEffort")
    reasoning_effort = (
        reasoning_effort_raw.strip()
        if isinstance(reasoning_effort_raw, str) and reasoning_effort_raw.strip()
        else None
    )

    timeout_raw = raw.get("timeout")
    timeout: int | None = None
    if timeout_raw is not None:
        try:
            parsed = int(float(str(timeout_raw)))
            if parsed > 0:
                timeout = parsed
        except ValueError:
            timeout = None

    apikey_raw = raw.get("apikey")
    api_key = (
        apikey_raw.strip()
        if isinstance(apikey_raw, str) and apikey_raw.strip()
        else None
    )

    if not model or not prompts_path:
        raise ValueError(
            _usage_line()
            + " [--out dataset.jsonl] [--api https://openrouter.ai/api/v1]"
            + ' [--system "..."] [--store-system true|false]'
            + " [--concurrent 1] [--openrouter.provider openai,anthropic]"
            + " [--openrouter.providerSort price|throughput|latency]"
            + " [--reasoningEffort low|medium|high] [--timeout <ms>]"
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
        openrouter_provider_order=provider_order,
        openrouter_provider_sort=provider_sort,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
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


def _make_provider_pref(args: Args) -> dict[str, Any] | None:
    if not is_openrouter_api_base(args.api_base):
        return None
    if args.openrouter_provider_order is None and args.openrouter_provider_sort is None:
        return None
    result: dict[str, Any] = {}
    if args.openrouter_provider_order is not None:
        result["order"] = args.openrouter_provider_order
    if args.openrouter_provider_sort is not None:
        result["sort"] = args.openrouter_provider_sort
    return result


def _get_pricing_if_needed(args: Args, api_key: str) -> OpenRouterModelPricing | None:
    if not is_openrouter_api_base(args.api_base):
        return None
    try:
        return get_openrouter_model_pricing(args.api_base, api_key, args.model)
    except Exception as exc:
        sys.stderr.write(f"WARN: Failed to fetch OpenRouter models/pricing: {exc}\n")
        return None


def _process_one_prompt(
    prompt: str,
    line_num: int,
    args: Args,
    api_key: str,
    provider_pref: dict[str, Any] | None,
    pricing: OpenRouterModelPricing | None,
) -> tuple[str | None, float, str | None, int]:
    try:
        messages = build_request_messages(args.system_prompt, prompt)
        content, reasoning, usage = call_openai_compatible_chat(
            args.api_base,
            api_key,
            args.model,
            messages,
            provider=provider_pref,
            reasoning_effort=args.reasoning_effort,
            timeout_ms=args.timeout,
        )
        assistant_content = format_assistant_content(content, reasoning)
        output_messages = build_output_messages(
            args.system_prompt,
            prompt,
            assistant_content,
            args.store_system,
        )
        line = json.dumps({"messages": output_messages}, ensure_ascii=False)
        spent_usd = 0.0
        if usage is not None and pricing is not None:
            if (
                pricing.known_prompt
                and pricing.known_completion
                and pricing.known_request
            ):
                spent_usd = calculate_openrouter_spend_usd(pricing, usage)
        return line, spent_usd, None, line_num
    except Exception as exc:
        return None, 0.0, str(exc), line_num


def run(argv: list[str]) -> int:
    """Runs datagen CLI with provided argv."""
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

    api_key = args.api_key or os.environ.get("API_KEY")
    if not api_key:
        sys.stderr.write("Missing API key. Provide --apikey or set API_KEY env var.\n")
        return 1

    abs_prompts = str(Path(args.prompts_path).resolve())
    abs_out = str(Path(args.out_path).resolve())
    try:
        _ensure_readable_file(abs_prompts)
    except Exception as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    prompts = _read_prompts(abs_prompts)
    provider_pref = _make_provider_pref(args)
    pricing = _get_pricing_if_needed(args, api_key)

    use_progress = args.progress and sys.stderr.isatty()
    total = count_prompts(abs_prompts) if use_progress else len(prompts)
    bar = ProgressBar(total, stream=sys.stderr) if use_progress and total > 0 else None

    ok_count = 0
    err_count = 0
    spent_usd = 0.0

    with Path(abs_out).open("w", encoding="utf-8") as out_stream:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrent)) as executor:
            futures = [
                executor.submit(
                    _process_one_prompt,
                    prompt,
                    line_num,
                    args,
                    api_key,
                    provider_pref,
                    pricing,
                )
                for prompt, line_num in prompts
            ]
            for future in as_completed(futures):
                line, delta_spend, error, line_num = future.result()
                if error is None and line is not None:
                    out_stream.write(line + "\n")
                    ok_count += 1
                    spent_usd += delta_spend
                else:
                    err_count += 1
                    sys.stderr.write(f"ERR line {line_num}: {error}\n")
                if bar is not None:
                    bar.render(
                        ok_count + err_count,
                        ProgressStats(ok=ok_count, err=err_count, spent_usd=spent_usd),
                    )

    if bar is not None:
        bar.finish(
            ok_count + err_count,
            ProgressStats(ok=ok_count, err=err_count, spent_usd=spent_usd),
        )
    return 0


def entrypoint() -> None:
    """Console script entrypoint."""
    raise SystemExit(run(sys.argv[1:]))
