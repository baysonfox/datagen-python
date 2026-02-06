"""OpenRouter helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from openai import OpenAI
import requests


@dataclass(frozen=True)
class OpenRouterUsage:
    """OpenRouter usage token counts."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class OpenRouterModelPricing:
    """Pricing information for an OpenRouter model."""

    prompt_per_token_usd: float
    completion_per_token_usd: float
    request_usd: float
    model_id: str
    canonical_slug: str | None
    known_prompt: bool
    known_completion: bool
    known_request: bool
    raw_prompt: str | None
    raw_completion: str | None
    raw_request: str | None


_MODELS_CACHE: dict[str, list[dict[str, Any]]] = {}


def _safe_parse_number(value: Any) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return 0.0
        try:
            return float(stripped)
        except ValueError:
            return 0.0
    return 0.0


def _is_finite_number_string(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped:
        return False
    try:
        float(stripped)
    except ValueError:
        return False
    return True


def is_openrouter_api_base(api_base: str) -> bool:
    """Checks whether a base URL is OpenRouter."""
    try:
        parsed = urlparse(api_base)
        host = parsed.netloc
        return host == "openrouter.ai" or host.endswith(".openrouter.ai")
    except ValueError:
        return "openrouter.ai" in api_base


def fetch_openrouter_models(
    api_base: str,
    api_key: str,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Fetches model catalog from OpenRouter with memoization."""
    cache_key = api_base.rstrip("/")
    cached = _MODELS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    url = f"{cache_key}/models"
    response = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"OpenRouter models error {response.status_code}: {response.text}"
        )

    payload = response.json()
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        models = []
    _MODELS_CACHE[cache_key] = models
    return models


def get_openrouter_model_pricing(
    api_base: str,
    api_key: str,
    model_id_or_slug: str,
    timeout: float = 30.0,
) -> OpenRouterModelPricing | None:
    """Gets pricing metadata for a model ID or canonical slug."""
    models = fetch_openrouter_models(api_base, api_key, timeout=timeout)

    exact: dict[str, Any] | None = None
    slug_matches: list[dict[str, Any]] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        if model.get("id") == model_id_or_slug:
            exact = model
            break
    if exact is None:
        for model in models:
            if not isinstance(model, dict):
                continue
            if model.get("canonical_slug") == model_id_or_slug:
                slug_matches.append(model)

    chosen = exact
    if chosen is None and slug_matches:
        chosen = next(
            (
                model
                for model in slug_matches
                if model.get("id") == model.get("canonical_slug")
            ),
            slug_matches[0],
        )
    if chosen is None:
        return None

    pricing = chosen.get("pricing")
    if not isinstance(pricing, dict):
        return None

    raw_prompt = pricing.get("prompt")
    raw_completion = pricing.get("completion")
    raw_request = pricing.get("request")

    return OpenRouterModelPricing(
        prompt_per_token_usd=_safe_parse_number(raw_prompt),
        completion_per_token_usd=_safe_parse_number(raw_completion),
        request_usd=_safe_parse_number(raw_request),
        model_id=str(chosen.get("id", "")),
        canonical_slug=(
            str(chosen.get("canonical_slug"))
            if chosen.get("canonical_slug") is not None
            else None
        ),
        known_prompt=_is_finite_number_string(raw_prompt),
        known_completion=_is_finite_number_string(raw_completion),
        known_request=_is_finite_number_string(raw_request),
        raw_prompt=str(raw_prompt) if isinstance(raw_prompt, str) else None,
        raw_completion=(
            str(raw_completion) if isinstance(raw_completion, str) else None
        ),
        raw_request=str(raw_request) if isinstance(raw_request, str) else None,
    )


def calculate_openrouter_spend_usd(
    pricing: OpenRouterModelPricing,
    usage: OpenRouterUsage | None,
) -> float:
    """Calculates per-request spend in USD."""
    prompt_tokens = usage.prompt_tokens if usage is not None else 0
    completion_tokens = usage.completion_tokens if usage is not None else 0
    return (
        pricing.request_usd
        + prompt_tokens * pricing.prompt_per_token_usd
        + completion_tokens * pricing.completion_per_token_usd
    )


def call_openai_compatible_chat(
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    provider: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    timeout_ms: int | None = None,
) -> tuple[str, str | None, OpenRouterUsage | None]:
    """Calls an OpenAI-compatible chat/completions endpoint."""
    client = OpenAI(api_key=api_key, base_url=api_base.rstrip("/"))

    extra_body: dict[str, Any] = {}
    if provider is not None:
        extra_body["provider"] = provider
    if reasoning_effort is not None and reasoning_effort.strip():
        extra_body["reasoning"] = {"effort": reasoning_effort.strip()}

    timeout_seconds = (timeout_ms / 1000.0) if timeout_ms is not None else None
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        timeout=timeout_seconds,
        extra_body=extra_body if extra_body else None,
    )

    content = ""
    if response.choices:
        message = response.choices[0].message
        if message.content is not None:
            content = message.content
    if not content:
        raise RuntimeError("No assistant content returned from API.")

    usage = None
    if response.usage is not None:
        usage = OpenRouterUsage(
            prompt_tokens=int(response.usage.prompt_tokens or 0),
            completion_tokens=int(response.usage.completion_tokens or 0),
            total_tokens=int(response.usage.total_tokens or 0),
        )

    return content, None, usage
