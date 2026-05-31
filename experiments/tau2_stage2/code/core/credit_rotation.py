"""Credit-exhaustion detection for CommonStack-key rotation.

When the runner sees an exception from a (task, seed) run that matches
one of these patterns, it advances to the next API key in the fallback
list and retries the same (task, seed). Anything that doesn't match
propagates as-is.

Patterns are matched case-insensitively against `str(exc)`. Designed to
be tolerant — false positives only cause an unneeded key swap; false
negatives leave behavior identical to a single-key setup.
"""
from __future__ import annotations

_PATTERNS = (
    "insufficient_balance",
    "insufficient balance",
    "insufficient credit",
    "insufficient credits",
    "payment required",
    " 402 ",
    "402:",
    "credit limit",
    "out of credits",
    "balance is too low",
    "额度不足",          # zh: insufficient balance
    "余额不足",          # zh: balance insufficient
    # CommonStack also enforces a per-key spend cap which surfaces as HTTP
    # 429 with body "Access key quota exceeded: <used>/<limit> used" and
    # "Limit Exceed" (sic). Treating this as credit-exhaustion makes the
    # runner rotate to the next key just like a 402 — the alternative is
    # the patched per-pair catch silently marking every pair as failed
    # until --limit is exhausted, which actually happened on 2026-04-29
    # and lost ~300 pairs of work before recovery.
    "access key quota exceeded",
    "limit exceed",
    "quota exceeded",
)


def is_credit_exhaustion(exc: BaseException) -> bool:
    msg = f" {str(exc).lower()} "      # pad so " 402 " matcher works at edges
    return any(p.lower() in msg for p in _PATTERNS)
