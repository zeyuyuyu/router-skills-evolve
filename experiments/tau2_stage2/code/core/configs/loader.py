"""YAML config loading with pydantic validation."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat

from core.schemas.artifacts import Tier


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelPricing(_Frozen):
    """Per-model rates as published by the provider's models endpoint.
    Drives `actual_usd` accounting. Cache fields are 0 when the API doesn't
    list them — we don't fabricate rates."""
    input_usd_per_m: NonNegativeFloat
    output_usd_per_m: NonNegativeFloat
    cache_read_usd_per_m: NonNegativeFloat
    cache_write_usd_per_m: NonNegativeFloat


class TierPricing(_Frozen):
    """Tier-uniform rates set by the pillar owner. Drives the Exp-1
    fixed-tier comparison metric (`fixed_tier_usd`). Same shape as
    ModelPricing but a distinct semantic — the rate every model in a
    tier is charged at under the fixed-tier accounting assumption."""
    input_usd_per_m: NonNegativeFloat
    output_usd_per_m: NonNegativeFloat
    cache_read_usd_per_m: NonNegativeFloat
    cache_write_usd_per_m: NonNegativeFloat


class Pricing(_Frozen):
    models: dict[str, ModelPricing]
    fixed_tier_prices: dict[Tier, TierPricing]
    # Declared so extra="forbid" accepts the yaml's provider_overrides:
    # block, but load_pricing strips it before model_validate, so the
    # field on returned instances is always {}. Don't read it from the
    # returned object — it's a load-time input, not part of Pricing's
    # public surface.
    # Shape: {provider_name: {model_id: ModelPricing-shaped dict}}
    provider_overrides: dict[str, dict[str, ModelPricing]] = Field(default_factory=dict)


class StudentPricing(_Frozen):
    central: ModelPricing | dict
    sensitivity_sweep_usd_per_m: list[float]
    market_to_market: dict[str, dict]


class SimulatorConfig(_Frozen):
    model: str


class BenchmarkRepo(_Frozen):
    url: str
    tag: str
    sha: str | None = None
    upstream: str | None = None


class TierPool(_Frozen):
    tiers: dict[Tier, list[str]]
    search_tiers: list[Tier]
    baseline_model: str
    baseline_tier: Tier = "mid-high"
    analyzer_model: str
    simulator: SimulatorConfig
    benchmark_repo: BenchmarkRepo
    # Per-provider model id + provider-pin overrides. Empty for setups
    # that only use the default provider (CommonStack today). Shape:
    # {provider_name: {canonical_id: {"id": str, "provider": str}}}
    provider_overrides: dict[str, dict[str, dict[str, str]]] = Field(default_factory=dict)


def _load(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def load_pricing(path: Path, *, provider: str = "commonstack") -> Pricing:
    """Load pricing.yaml, optionally merging provider-specific overrides.

    For provider="commonstack" (default), returns the base `models:` block
    unmerged. For other providers, overlays `provider_overrides[provider]`
    on top of `models:` (override wins per model_id; OR-only entries are
    added).

    The returned Pricing object always has `provider_overrides == {}` —
    overrides are consumed by this function and stripped before
    validation, so downstream code never sees the raw block (which would
    risk a double-merge if mistakenly applied again).
    """
    raw = _load(path)
    overrides = raw.get("provider_overrides", {}).get(provider, {})
    # Strip provider_overrides from the validated payload regardless of
    # which branch we take — the field is purely an input to this loader,
    # not state that survives onto the Pricing object.
    raw_without_overrides = {k: v for k, v in raw.items() if k != "provider_overrides"}
    if provider != "commonstack" and overrides:
        base_models = raw_without_overrides.get("models", {})
        merged_models = {**base_models, **overrides}
        merged_raw = {**raw_without_overrides, "models": merged_models}
        return Pricing.model_validate(merged_raw)
    return Pricing.model_validate(raw_without_overrides)


def load_student_pricing(path: Path) -> StudentPricing:
    return StudentPricing.model_validate(_load(path))


def load_tier_pool(path: Path) -> TierPool:
    return TierPool.model_validate(_load(path))
