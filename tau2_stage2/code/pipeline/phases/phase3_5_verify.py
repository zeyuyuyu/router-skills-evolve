"""Phase 3.5 — verify each Phase-3 lock by probing strictly cheaper tiers.

Phase 3's tier-escalation search has a known order bias: once a candidate
passes at tier T, it breaks before exploring T-1 / T-2. Sibling project
observed ~60% lock corrections on BFCL multi-turn when re-running below the
lock. For Exp-2 to be a real oracle, we must catch these.

Same `run_task_with_substitution` machinery as Phase 3; different search
strategy: for each lock at tier T, try every candidate at every tier
strictly < T (ordered low → mid-1), and replace the lock if any probe
beats it on `actual_usd`.
"""
from __future__ import annotations

from typing import Any

from core.cost_accounting import fixed_tier_cost, per_record_cost
from core.schemas.artifacts import (
    Attempt,
    ExplorationLog,
    LLMSpec,
    LockResult,
)
from pipeline._resolve import _resolve_model
from pipeline.phases.phase3_search import Phase3Context


def run_phase3_5_for_task(
    adapter: Any,
    task: Any,
    baseline_messages: list[Any],
    phase3_log: ExplorationLog,
    ctx: Phase3Context,
) -> ExplorationLog:
    extra_attempts: list[Attempt] = []
    updated_locks: list[LockResult] = []

    for lock in phase3_log.locked_results:
        lock_tier_idx = ctx.search_tiers.index(lock.tier)
        if lock_tier_idx == 0:
            # Already at the cheapest tier — nothing strictly below to probe.
            updated_locks.append(lock)
            continue

        step_idx = lock.step - 1
        best_model_id = lock.model_id
        best_actual = lock.actual_usd
        best_fixed = lock.fixed_tier_usd
        best_tier = lock.tier
        best_source = lock.source

        for tier_idx in range(0, lock_tier_idx):
            tier = ctx.search_tiers[tier_idx]
            for model_id in ctx.tier_pool.get(tier, []):
                api_id, extra_body = _resolve_model(
                    model_id, ctx.provider, ctx.provider_overrides
                )
                base_args = {**ctx.commonstack_args, "custom_llm_provider": "openai"}
                if extra_body:
                    base_args = {**base_args, "extra_body": extra_body}
                sub_spec = LLMSpec(
                    model=f"openai/{api_id}",
                    args=base_args,
                )
                result = adapter.run_task_with_substitution(
                    task,
                    ctx.baseline_run_cfg,
                    baseline_messages=baseline_messages,
                    step_idx=step_idx,
                    sub=sub_spec,
                    domain=ctx.domain,
                )

                probe_input = 0
                probe_output = 0
                probe_cache_r = 0
                probe_cache_w = 0
                probe_response: str | None = None
                if 0 <= step_idx < len(result.steps):
                    s = result.steps[step_idx]
                    probe_input = s.input_tokens
                    probe_output = s.output_tokens
                    probe_cache_r = s.cached_read_tokens
                    probe_cache_w = s.cached_write_tokens
                    # Pull the substituted step's response text. tau2 step responses
                    # come back as dicts {"content": "...", "tool_calls": [...]} —
                    # flatten to a string so we can store it; tool calls are
                    # preserved in raw form upstream.
                    if isinstance(s.response, dict):
                        probe_response = s.response.get("content") or ""
                    else:
                        probe_response = str(s.response) if s.response is not None else None

                actual = per_record_cost(
                    model_id=model_id,
                    input_tokens=probe_input,
                    output_tokens=probe_output,
                    cached_read_tokens=probe_cache_r,
                    cached_write_tokens=probe_cache_w,
                    pricing=ctx.pricing,
                )
                fixed = fixed_tier_cost(
                    tier=tier,
                    input_tokens=probe_input,
                    output_tokens=probe_output,
                    cached_read_tokens=probe_cache_r,
                    cached_write_tokens=probe_cache_w,
                    pricing=ctx.pricing,
                )
                extra_attempts.append(
                    Attempt(
                        step=lock.step,
                        model_id=model_id,
                        tier=tier,
                        passed=result.passed,
                        actual_usd=actual,
                        fixed_tier_usd=fixed,
                        input_tokens=probe_input,
                        output_tokens=probe_output,
                        cached_read_tokens=probe_cache_r,
                        cached_write_tokens=probe_cache_w,
                        response=probe_response,
                        termination_reason=getattr(result, "termination_reason", None),
                        probe_agent_cost_usd=float(getattr(result, "agent_cost_usd", 0.0) or 0.0),
                        probe_user_cost_usd=float(getattr(result, "user_cost_usd", 0.0) or 0.0),
                    )
                )

                if result.passed and actual < best_actual:
                    best_model_id = model_id
                    best_actual = actual
                    best_fixed = fixed
                    best_tier = tier
                    best_source = "phase3_5"

        updated_locks.append(
            LockResult(
                step=lock.step,
                model_id=best_model_id,
                tier=best_tier,
                actual_usd=best_actual,
                fixed_tier_usd=best_fixed,
                source=best_source,
            )
        )

    return ExplorationLog(
        attempts=list(phase3_log.attempts) + extra_attempts,
        locked_results=updated_locks,
    )
