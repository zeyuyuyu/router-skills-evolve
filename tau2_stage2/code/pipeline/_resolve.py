"""Provider-aware model id resolution.

Extracted from `pipeline.runner` so phase modules can call `_resolve_model`
without re-importing the runner (which already imports them — would be a
circular import).
"""
from __future__ import annotations


def _resolve_model(
    canonical_id: str,
    provider: str,
    provider_overrides: dict[str, dict[str, dict[str, str]]],
) -> tuple[str, dict]:
    """Map a canonical (CommonStack) model id to its (api_id, extra_body)
    under the chosen provider.

    For provider="commonstack": returns (canonical_id, {}). No transform.
    For other providers: looks up canonical_id in
    provider_overrides[provider] and returns
    (entry["id"], {"provider": {"order": [entry["provider"]],
                                "allow_fallbacks": False}}).
    Raises ValueError if the override is missing — fail fast rather than
    silently auto-route.

    The provider_overrides dict has shape
    {provider_name: {canonical_id: {"id": str, "provider": str}}}
    — typically passed as `tier_pool.provider_overrides` or
    `ctx.provider_overrides`.
    """
    if provider == "commonstack":
        return canonical_id, {}
    overrides = provider_overrides.get(provider, {})
    entry = overrides.get(canonical_id)
    if entry is None:
        raise ValueError(
            f"--provider {provider}: no override for {canonical_id!r} "
            f"in tier_pool. Add an entry under provider_overrides.{provider} "
            f"in core/configs/tier_pools/<adapter>.yaml."
        )
    return entry["id"], {
        "provider": {
            "order": [entry["provider"]],
            "allow_fallbacks": False,
        }
    }
