"""Make litellm.model_cost tolerant of OpenRouter's snapshot id format.

OpenRouter returns response.model with shape `openai/gpt-5.2-20251211`
(provider prefix + bare-model + date suffix) for date-versioned snapshots.
litellm's price DB only has the bare-model form (`gpt-5.2`). On miss,
tau2's get_response_cost catches the exception and silently returns 0,
which manifests as user_cost_usd / simulator_usd being 0 in saved
artifacts under --provider openrouter.

This module installs a thin wrapper around `litellm.model_cost` that
falls back to the bare-model form on miss, so lookups for OR snapshot
ids resolve to the same prices the bare names already have. CommonStack
runs are unaffected (their response.model is the bare form to begin with).
"""
from __future__ import annotations

import re

_SNAPSHOT_DATE_RE = re.compile(r"-\d{8}$")
_PROVIDER_PREFIX = "openai/"
_INSTALLED = False


class _FuzzyModelCost(dict):
    """dict subclass for litellm.model_cost that does fuzzy fallback on miss.

    Order of fallback (each step tries the resulting key in self):
    1. exact key
    2. strip leading "openai/" prefix
    3. strip trailing "-YYYYMMDD" date suffix
    4. strip both
    """

    def _resolve_key(self, key):
        if dict.__contains__(self, key):
            return key
        # 2. strip prefix
        no_prefix = key
        if key.startswith(_PROVIDER_PREFIX):
            no_prefix = key[len(_PROVIDER_PREFIX):]
            if dict.__contains__(self, no_prefix):
                return no_prefix
        # 3. strip date suffix
        no_date = _SNAPSHOT_DATE_RE.sub("", key)
        if no_date != key and dict.__contains__(self, no_date):
            return no_date
        # 4. both
        if no_prefix != key:
            no_both = _SNAPSHOT_DATE_RE.sub("", no_prefix)
            if no_both != no_prefix and dict.__contains__(self, no_both):
                return no_both
        return None

    def __getitem__(self, key):
        resolved = self._resolve_key(key)
        if resolved is None:
            raise KeyError(key)
        return dict.__getitem__(self, resolved)

    def __contains__(self, key):
        return self._resolve_key(key) is not None

    def get(self, key, default=None):
        resolved = self._resolve_key(key)
        if resolved is None:
            return default
        return dict.__getitem__(self, resolved)


def install_fuzzy_model_cost() -> None:
    """Replace litellm.model_cost with a fuzzy-lookup wrapper. Idempotent.

    Safe to call multiple times — re-calling on an already-wrapped
    model_cost returns the same instance. CommonStack lookups are
    unaffected (the bare-model path is the first match).
    """
    global _INSTALLED
    import litellm

    if _INSTALLED and isinstance(litellm.model_cost, _FuzzyModelCost):
        return
    # Preserve original entries
    original = dict(litellm.model_cost)
    litellm.model_cost = _FuzzyModelCost(original)
    _INSTALLED = True


def register_pricing_with_litellm(pricing) -> None:
    """Register every pricing.yaml model into litellm.model_cost so tau2's
    get_response_cost can price our gateway-specific models.

    litellm's price DB doesn't know our active models (qwen3.5-397b-a17b,
    z-ai/glm-4.5-air, minimax-m2.7, etc.) under any name. Without this,
    tau2's get_response_cost catches the lookup failure and silently
    returns 0 (logging an "ERROR | model isn't mapped yet" line per call).

    Combined with install_fuzzy_model_cost(), this also handles
    OpenRouter's date-versioned snapshot ids — the fuzzy wrapper strips
    the suffix to find the bare-name entry we register here.

    Skips any model id that litellm already knows (via the fuzzy
    wrapper's resolution) — don't clobber upstream-authoritative prices
    (e.g. our placeholder pricing for openai/gpt-5.2 must NOT replace
    litellm's real $1.75/$14 per million entry).

    Caller is expected to install_fuzzy_model_cost() first; otherwise
    the "already knows" check is exact-match-only and may register
    duplicates under multiple keys.
    """
    import litellm
    for model_id, mp in pricing.models.items():
        if model_id in litellm.model_cost:
            # Fuzzy wrapper resolved this id (exactly or via prefix/suffix
            # stripping). Don't clobber the upstream entry.
            continue
        entry = {
            "input_cost_per_token": float(mp.input_usd_per_m) / 1_000_000.0,
            "output_cost_per_token": float(mp.output_usd_per_m) / 1_000_000.0,
        }
        if mp.cache_read_usd_per_m:
            entry["cache_read_input_token_cost"] = (
                float(mp.cache_read_usd_per_m) / 1_000_000.0
            )
        litellm.model_cost[model_id] = entry
