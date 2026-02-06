"""Configuration parsing for datagen."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

_CONFIG_KEY_ALIASES: dict[str, str] = {
    "promptsPath": "prompts",
    "outPath": "out",
    "apiBase": "api",
    "storeSystem": "store-system",
    "noProgress": "no-progress",
    "openrouterProviderOrder": "openrouter.provider",
    "openrouterProviderSort": "openrouter.providerSort",
}


def _flatten_config(value: Any, prefix: str, out: dict[str, Any]) -> None:
    if value is None:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_config(child, next_prefix, out)
        return
    out[prefix] = value


def _normalize_config_key(key: str) -> str:
    trimmed = key.strip()
    without_prefix = trimmed[2:] if trimmed.startswith("--") else trimmed
    return _CONFIG_KEY_ALIASES.get(without_prefix, without_prefix)


def _to_cli_raw_value(value: Any) -> str | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)):
        return str(value)
    if value is None:
        return ""
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, bool):
                parts.append("true" if item else "false")
            elif isinstance(item, (str, int, float)):
                parts.append(str(item))
            else:
                raise ValueError("Unsupported array value in config.")
        return ",".join(parts)
    raise ValueError("Unsupported config value.")


def load_config_raw_args(config_path: str) -> dict[str, str | bool]:
    """Loads YAML/JSON config as raw CLI-like key-value pairs.

    Args:
        config_path: Path to YAML or JSON file.

    Returns:
        A dict where keys are normalized flag names and values are raw values.

    Raises:
        FileNotFoundError: Config file does not exist.
        ValueError: Config format is invalid or unsupported.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    text = path.read_text(encoding="utf-8")
    trimmed = text.lstrip("\ufeff \t\r\n")

    parsed: Any
    if trimmed.startswith("{") or trimmed.startswith("["):
        parsed = json.loads(text)
    else:
        parsed = yaml.safe_load(text)

    if not isinstance(parsed, dict):
        raise ValueError("Config must be a YAML/JSON object at the root.")

    flat: dict[str, Any] = {}
    _flatten_config(parsed, "", flat)

    out: dict[str, str | bool] = {}
    for key, value in flat.items():
        if not key.strip():
            continue
        out[_normalize_config_key(key)] = _to_cli_raw_value(value)
    return out
