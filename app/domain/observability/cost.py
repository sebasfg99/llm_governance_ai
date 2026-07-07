# ============================================================
# app/domain/observability/cost.py — Per-call cost estimation
#
# WHY THIS FILE EXISTS:
# The metrics endpoint, the response payload, and the audit
# trace all need a per-call USD cost estimate. The OpenAI /
# Gemini SDKs only return token counts; this module owns the
# pricing table and converts tokens into dollars.
#
# IMPORTANT:
# Prices are public list prices in USD per 1M tokens, captured
# at the time of writing. The numbers should be reviewed
# quarterly — pricing tiers change. Tracking the version below
# helps audit when a number was last refreshed.
#
# DESIGN:
# We store prices "per 1M tokens" because that's how OpenAI and
# Google publish them. Conversion to actual USD multiplies by
# (tokens / 1_000_000).
# ============================================================

from __future__ import annotations

from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)

PRICING_VERSION = "2026.07.refresh"


@dataclass(frozen=True)
class ModelPrice:
    """Price points for one model in USD per 1M tokens."""
    input_per_1m:  float
    output_per_1m: float
    notes:         str = ""


# ============================================================
# Pricing table (USD per 1 million tokens)
#
# Keep keys lowercase for easy lookup. Defaults applied when a
# specific model is not found.
# ============================================================
MODEL_PRICING: dict[str, ModelPrice] = {
    # OpenAI chat models
    "gpt-4o-mini":           ModelPrice(0.15, 0.60, "Default chat model"),
    "gpt-4o":                ModelPrice(2.50, 10.00, "High-end OpenAI chat"),
    "gpt-4-turbo":           ModelPrice(10.00, 30.00, "Legacy turbo"),
    "gpt-3.5-turbo":         ModelPrice(0.50, 1.50, "Legacy 3.5"),
    "o1-mini":               ModelPrice(3.00, 12.00, "Reasoning compact"),
    "o1":                    ModelPrice(15.00, 60.00, "Reasoning"),
    # Google Gemini
    "gemini-1.5-flash":      ModelPrice(0.075, 0.30, "Legacy flash"),
    "gemini-1.5-pro":        ModelPrice(1.25, 5.00, "Legacy pro"),
    "gemini-2.0-flash":      ModelPrice(0.10, 0.40, "Legacy 2.0 flash"),
    # Output price includes thinking tokens when thinking is enabled.
    "gemini-2.5-flash":      ModelPrice(0.30, 2.50, "Default sidecar model"),
    "gemini-2.5-pro":        ModelPrice(1.25, 10.00, "Edge high-reasoning tier"),
    "gemini-3-flash-preview": ModelPrice(0.50, 3.00, "Edge default (preview; estimated pricing)"),
    # Embeddings (input only)
    "text-embedding-3-small": ModelPrice(0.02, 0.0, "1536-d embeddings"),
    "text-embedding-3-large": ModelPrice(0.13, 0.0, "3072-d embeddings"),
}


# Fallback used when a model name is not in the table. We pick
# a conservative mid-range estimate so the cost is never zero
# (which would mask billing surprises).
_FALLBACK_PRICE = ModelPrice(input_per_1m=1.00, output_per_1m=3.00, notes="fallback")


def estimate_cost(
    *,
    provider: str | None,
    model: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> float:
    """
    Compute the USD cost of one LLM call.

    Args:
        provider:          "openai" | "gemini" | other. Currently used
                           only for logging — pricing is keyed by model.
        model:             Model name (e.g. "gpt-4o-mini").
        prompt_tokens:     Input token count, may be None.
        completion_tokens: Output token count, may be None.

    Returns:
        USD cost rounded to 6 decimals. Always >= 0.

    NOTE:
        When token counts are missing the function returns 0.0
        rather than guessing — better to under-report than mislead.
    """
    # Coerce to int defensively: MagicMock / None / weird types should
    # never crash the pipeline. Anything we can't read becomes 0.
    pt = _coerce_int(prompt_tokens)
    ct = _coerce_int(completion_tokens)
    if pt == 0 and ct == 0:
        return 0.0

    key = (model or "").lower().strip()
    price = MODEL_PRICING.get(key)
    if price is None:
        logger.debug(
            f"No pricing entry for model='{model}' provider='{provider}'. "
            f"Falling back to {_FALLBACK_PRICE}."
        )
        price = _FALLBACK_PRICE

    cost = (
        pt * price.input_per_1m / 1_000_000
        + ct * price.output_per_1m / 1_000_000
    )
    return round(max(cost, 0.0), 6)


def _coerce_int(value: object) -> int:
    """Best-effort int coercion. Returns 0 for None and unparseable values."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
