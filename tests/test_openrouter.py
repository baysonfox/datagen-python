"""Tests for OpenRouter helpers."""

from __future__ import annotations

from datagen.openrouter import (
    OpenRouterModelPricing,
    OpenRouterUsage,
    calculate_openrouter_spend_usd,
    is_openrouter_api_base,
)


def test_is_openrouter_api_base() -> None:
    assert is_openrouter_api_base("https://openrouter.ai/api/v1") is True
    assert is_openrouter_api_base("https://x.openrouter.ai/api/v1") is True
    assert is_openrouter_api_base("https://example.com/api/v1") is False


def test_calculate_openrouter_spend_usd() -> None:
    pricing = OpenRouterModelPricing(
        prompt_per_token_usd=0.001,
        completion_per_token_usd=0.002,
        request_usd=0.01,
        model_id="m",
        canonical_slug=None,
        known_prompt=True,
        known_completion=True,
        known_request=True,
        raw_prompt="0.001",
        raw_completion="0.002",
        raw_request="0.01",
    )
    usage = OpenRouterUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert calculate_openrouter_spend_usd(pricing, usage) == 0.03


def test_openrouter_usage_dataclass_defaults() -> None:
    usage = OpenRouterUsage()
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0
