"""Phase 3 — per-step downgrade search, first-passing-wins.

For each step in the baseline trajectory, probe cheaper models tier by tier
until one passes. Within a tier, candidates are tried in YAML-listed order
and the FIRST passing candidate is locked immediately — remaining
candidates in that tier (and all higher tiers) are skipped. This trades
optimal in-tier cost for fewer probe API calls; the per-task savings on
long trajectories more than offset the occasional non-cheapest lock.

Search order per step: escalate through `search_tiers` from low to high
until something passes. If no tier below `high` passes, the step is
preserved at the baseline tier via the keep-implicit sweep at the end of
the search.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.configs.loader import Pricing
from core.cost_accounting import fixed_tier_cost, per_record_cost
from core.schemas.artifacts import (
    AnalysisResult,
    Attempt,
    BaselineResult,
    ExplorationLog,
    LLMSpec,
    LockResult,
    Phase3Checkpoint,
    RunTaskConfig,
    Tier,
)
from pipeline._resolve import _resolve_model


@dataclass(frozen=True)
class Phase3Context:
    """Per-run knobs for Phase 3 searches."""
    tier_pool: dict[Tier, list[str]]
    search_tiers: list[Tier]
    pricing: Pricing
    commonstack_args: dict[str, Any]
    baseline_run_cfg: RunTaskConfig
    domain: str
    # Model id that backs the Phase-0 baseline trajectory. Used to emit
    # keep_implicit LockResults for steps that no cheaper tier could replace,
    # so the lock map stays a complete per-step assignment.
    baseline_model: str = ""
    # Tier label for the baseline. keep_implicit locks get this tier so their
    # fixed_tier_usd uses the right entry in pricing.fixed_tier_prices.
    # Defaults to "high" for backwards-compat with older configs that didn't
    # set an explicit baseline tier.
    baseline_tier: Tier = "high"
    # Optional override for the adapter's evaluation type during probes.
    # `None` = use the adapter's default (task-native grader, NL-assertions
    # included). Set to `EvaluationType.ACTIONS` in the runner so the
    # search loop grades on deterministic action-trace match, removing
    # the NL judge from the inner loop. Typed `Any` to avoid importing
    # tau2's EvaluationType at module-top (tau2 has heavy import cost).
    evaluation_type: Any = None
    # Active provider for sub_spec model resolution. Defaults to
    # "commonstack" so existing tests / fixtures that don't set this keep
    # working (commonstack is a no-op pass-through inside _resolve_model).
    provider: str = "commonstack"
    # Per-provider canonical→api id overrides. Mirrors the
    # `provider_overrides` block in the tier-pool yaml. Empty dict default
    # is safe under provider="commonstack" since that branch never reads it.
    provider_overrides: dict[str, dict[str, dict[str, str]]] = field(
        default_factory=dict
    )


def run_phase3_for_task(
    adapter: Any,
    task: Any,
    baseline_messages: list[Any],
    analysis: AnalysisResult,
    ctx: Phase3Context,
) -> ExplorationLog:
    attempts: list[Attempt] = []
    locks: list[LockResult] = []

    for step_analysis in analysis.steps:
        if step_analysis.replaceability != "replaceable":
            continue
        step_idx = step_analysis.step - 1  # 1-based → 0-based

        if step_analysis.start_tier not in ctx.search_tiers:
            # Shouldn't happen with a well-formed analyzer output, but be safe.
            continue
        start_tier_idx = ctx.search_tiers.index(step_analysis.start_tier)
        start_search_idx = max(0, start_tier_idx - 2)  # spec §4 Phase-3

        locked = False
        for tier_idx in range(start_search_idx, len(ctx.search_tiers)):
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
                probe_response: str | None = None
                if 0 <= step_idx < len(result.steps):
                    s = result.steps[step_idx]
                    probe_input = s.input_tokens
                    probe_output = s.output_tokens
                    probe_cache_r = s.cached_read_tokens
                    probe_cache_w = s.cached_write_tokens
                    if isinstance(s.response, dict):
                        probe_response = s.response.get("content") or ""
                    else:
                        probe_response = str(s.response) if s.response is not None else None
                else:
                    probe_cache_r = 0
                    probe_cache_w = 0

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
                attempts.append(
                    Attempt(
                        step=step_idx + 1,
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
                if result.passed:
                    locks.append(
                        LockResult(
                            step=step_idx + 1,
                            model_id=model_id,
                            tier=tier,
                            actual_usd=actual,
                            fixed_tier_usd=fixed,
                            source="phase3",
                        )
                    )
                    locked = True
                    break  # first-passing-wins: skip remaining candidates in tier
            if locked:
                break  # also stop escalating to higher tiers

        if not locked:
            # Keep baseline — no LockResult emitted for this step.
            pass

    return ExplorationLog(attempts=attempts, locked_results=locks)


def run_phase3_probe_all(
    adapter: Any,
    task: Any,
    baseline: BaselineResult,
    baseline_messages: list[Any],
    ctx: Phase3Context,
    *,
    checkpoint_path: Path | None = None,
    resume_from: Phase3Checkpoint | None = None,
) -> ExplorationLog:
    """Phase 3 without the analyzer — probe every agent step directly.

    For each step in baseline.steps, escalate tiers low → mid → mid-high and
    lock the cheapest-actual-cost passing candidate. No `replaceability`
    filter; no `start_tier` hint; every step always starts at low. Steps for
    which no tier yields a passing probe remain at baseline (no LockResult).

    If `checkpoint_path` is given, a Phase3Checkpoint is written after each
    step's lock decision so a crash mid-task only loses the probes for the
    in-flight step. If `resume_from` is given, attempts/locks/current_messages
    are restored from it and the loop skips ahead to the first unprocessed
    step (`resume_from.last_completed_step + 1`).
    """
    if resume_from is not None:
        attempts: list[Attempt] = list(resume_from.attempts)
        locks: list[LockResult] = list(resume_from.locks)
        current_messages: list[Any] = list(resume_from.current_messages)
        start_step_idx = resume_from.last_completed_step + 1
    else:
        attempts = []
        locks = []
        # current_messages evolves: each lock's winning probe overwrites it
        # with the post-lock trajectory. Subsequent steps probe against that
        # prefix.
        current_messages = list(baseline_messages)
        start_step_idx = 0

    def _save_checkpoint(last_completed: int) -> None:
        if checkpoint_path is None:
            return
        ckpt = Phase3Checkpoint(
            last_completed_step=last_completed,
            attempts=list(attempts),
            locks=list(locks),
            current_messages=list(current_messages),
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(ckpt.model_dump_json(indent=2))

    for step_idx in range(start_step_idx, len(baseline.steps)):
        baseline_step = baseline.steps[step_idx]
        # Canned non-LLM messages (e.g. τ²'s hard-coded retail opening "Hi!
        # How can I help you today?") report 0 tokens on both sides because
        # no LLM call backs them. Skip — nothing to substitute.
        if baseline_step.input_tokens == 0 and baseline_step.output_tokens == 0:
            _save_checkpoint(step_idx)
            continue
        locked = False
        for tier in ctx.search_tiers:
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
                try:
                    result = adapter.run_task_with_substitution(
                        task,
                        ctx.baseline_run_cfg,
                        baseline_messages=current_messages,
                        step_idx=step_idx,
                        sub=sub_spec,
                        domain=ctx.domain,
                        evaluation_type=ctx.evaluation_type,
                    )
                except ValueError as exc:
                    # tau2's orchestrator raises ValueError when a substituted
                    # cheap model returns an empty AssistantMessage (no
                    # content, no tool_calls — happens when the model hits
                    # max_tokens or stops mid-response). Treat this candidate
                    # as a failed probe and continue to the next one rather
                    # than abandoning the entire (task, seed) pair. Without
                    # this catch, airline tasks 10/11/12 alone would cost
                    # ~50% of the shard's data.
                    print(
                        f"[phase3] candidate '{model_id}' raised "
                        f"{type(exc).__name__} on step {step_idx + 1}; "
                        f"skipping candidate. cause: {str(exc)[:200]}",
                        flush=True,
                    )
                    continue

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
                attempts.append(
                    Attempt(
                        step=step_idx + 1,
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
                if result.passed:
                    locks.append(
                        LockResult(
                            step=step_idx + 1,
                            model_id=model_id,
                            tier=tier,
                            actual_usd=actual,
                            fixed_tier_usd=fixed,
                            source="phase3",
                        )
                    )
                    # Promote the winning probe's post-run trajectory:
                    # subsequent steps probe against this evolving prefix,
                    # so each lock is tested in the cumulative-cheap context.
                    winning_messages = list(getattr(result, "messages", []) or [])
                    if winning_messages:
                        current_messages = winning_messages
                    locked = True
                    break  # first-passing-wins: skip remaining candidates in tier
            if locked:
                break  # also stop escalating to higher tiers

        if not locked:
            # No tier passed — fall through to the keep_implicit sweep below.
            pass
        _save_checkpoint(step_idx)

    # Keep-implicit sweep: every step that wasn't locked by the probing loop
    # (zero-cost skipped OR probed but nothing passed) gets an explicit
    # LockResult at the baseline tier. This makes the lock map a complete
    # per-step assignment so Phase 4 / Exp-2 accounting don't need an
    # "absence means baseline" rule.
    locked_steps = {lr.step for lr in locks}
    for step_idx in range(len(baseline.steps)):
        step_num = step_idx + 1
        if step_num in locked_steps:
            continue
        baseline_step = baseline.steps[step_idx]
        locks.append(
            LockResult(
                step=step_num,
                model_id=ctx.baseline_model,
                tier=ctx.baseline_tier,
                actual_usd=baseline_step.actual_usd,
                fixed_tier_usd=fixed_tier_cost(
                    tier=ctx.baseline_tier,
                    input_tokens=baseline_step.input_tokens,
                    output_tokens=baseline_step.output_tokens,
                    cached_read_tokens=baseline_step.cached_read_tokens,
                    cached_write_tokens=baseline_step.cached_write_tokens,
                    pricing=ctx.pricing,
                ),
                source="keep_implicit",
            )
        )
    locks.sort(key=lambda lr: lr.step)

    return ExplorationLog(
        attempts=attempts,
        locked_results=locks,
        locked_trajectory=current_messages,
    )
