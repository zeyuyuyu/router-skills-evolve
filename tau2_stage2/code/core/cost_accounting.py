"""Per-record cost formulas (PROJECT_CHARTER.md §3.2)."""
from __future__ import annotations

from core.configs.loader import Pricing
from core.schemas.artifacts import Tier


def per_record_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    cached_read_tokens: int,
    cached_write_tokens: int,
    pricing: Pricing,
) -> float:
    m = pricing.models[model_id]
    return (
        input_tokens * m.input_usd_per_m / 1e6
        + output_tokens * m.output_usd_per_m / 1e6
        + cached_read_tokens * m.cache_read_usd_per_m / 1e6
        + cached_write_tokens * m.cache_write_usd_per_m / 1e6
    )


def fixed_tier_cost(
    tier: Tier,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_read_tokens: int = 0,
    cached_write_tokens: int = 0,
    pricing: Pricing,
) -> float:
    """Cost under the fixed-tier accounting assumption (Exp-1 baseline metric).

    All four token types are priced at the tier's rate, mirroring
    `per_record_cost` so the actual-vs-fixed comparison is apples-to-apples.
    Unspecified token kinds default to 0.
    """
    t = pricing.fixed_tier_prices[tier]
    return (
        input_tokens * t.input_usd_per_m / 1e6
        + output_tokens * t.output_usd_per_m / 1e6
        + cached_read_tokens * t.cache_read_usd_per_m / 1e6
        + cached_write_tokens * t.cache_write_usd_per_m / 1e6
    )
