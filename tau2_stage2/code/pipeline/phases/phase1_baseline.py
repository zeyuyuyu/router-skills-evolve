"""Phase 1 — single-SOTA baseline repricing.

The baseline trajectory is run via `adapter.run_task(task, run_cfg)` — owned
by the runner, not this module — and returns a `TaskRunResult` whose steps
carry raw token counts. We reprice those steps against our pinned
`pricing.yaml` so `actual_usd` reflects our cost track (list prices at
measurement time) rather than whatever LiteLLM's internal pricing returned.
"""
from __future__ import annotations

from core.configs.loader import Pricing
from core.cost_accounting import fixed_tier_cost, per_record_cost
from core.schemas.artifacts import (
    BaselineArtifact,
    BaselineResult,
    StepData,
    TaskRunResult,
    Tier,
)


def reprice_baseline(
    run_result: TaskRunResult,
    *,
    task_id: str,
    domain: str,
    baseline_pricing_id: str,
    baseline_tier: Tier,
    pricing: Pricing,
) -> BaselineResult:
    """Build a BaselineResult by repricing each StepData through our pricing.yaml.

    `run_result.agent.model` is the LiteLLM-routed model string; it may not
    match the CommonStack catalog id (`baseline_pricing_id`) used as the key
    in pricing.yaml. Passing both keeps the decoupling explicit.
    """
    repriced_steps: list[StepData] = []
    total_actual_usd = 0.0
    total_fixed_tier_usd = 0.0
    for step in run_result.steps:
        actual = per_record_cost(
            model_id=baseline_pricing_id,
            input_tokens=step.input_tokens,
            output_tokens=step.output_tokens,
            cached_read_tokens=step.cached_read_tokens,
            cached_write_tokens=step.cached_write_tokens,
            pricing=pricing,
        )
        fixed = fixed_tier_cost(
            tier=baseline_tier,
            input_tokens=step.input_tokens,
            output_tokens=step.output_tokens,
            cached_read_tokens=step.cached_read_tokens,
            cached_write_tokens=step.cached_write_tokens,
            pricing=pricing,
        )
        repriced_steps.append(step.model_copy(update={"actual_usd": actual}))
        total_actual_usd += actual
        total_fixed_tier_usd += fixed

    error = None
    if not run_result.passed and run_result.termination_reason not in (
        "AGENT_STOP",
        "USER_STOP",
    ):
        error = run_result.termination_reason

    return BaselineResult(
        task_id=task_id,
        domain=domain,
        passed=run_result.passed,
        pass_at_k=1.0 if run_result.passed else 0.0,  # post-hoc pass^k aggregates over seeds
        actual_usd=total_actual_usd,
        simulator_usd=run_result.user_cost_usd,
        steps=repriced_steps,
        system_prompt=run_result.system_prompt,
        tools=list(run_result.tools),
        error=error,
    )


def reprice_baseline_artifact(
    artifact: BaselineArtifact,
    *,
    pricing: Pricing,
    baseline_tier: Tier = "high",
) -> BaselineResult:
    """Same as `reprice_baseline` but takes a persisted BaselineArtifact.

    Used by the resumable runner: Phase 0 wrote `phase0_baseline/<id>.json`,
    Phase 1 reads it back and produces a `BaselineResult` written to
    `phase1_repriced/<id>.json`. Pure: no API calls, no I/O.
    """
    repriced_steps: list[StepData] = []
    total_actual = 0.0
    for step in artifact.steps:
        actual = per_record_cost(
            model_id=artifact.baseline_model,
            input_tokens=step.input_tokens,
            output_tokens=step.output_tokens,
            cached_read_tokens=step.cached_read_tokens,
            cached_write_tokens=step.cached_write_tokens,
            pricing=pricing,
        )
        repriced_steps.append(step.model_copy(update={"actual_usd": actual}))
        total_actual += actual

    return BaselineResult(
        task_id=artifact.task_id,
        domain=artifact.domain,
        passed=artifact.passed,
        pass_at_k=1.0 if artifact.passed else 0.0,
        actual_usd=total_actual,
        simulator_usd=artifact.user_cost_usd,
        steps=repriced_steps,
        system_prompt=artifact.system_prompt or "",
        tools=list(artifact.tools),
        error=artifact.error,
    )


# Back-compat shim: earlier code / tests called run_phase1_task(adapter, task, ...)
# which combined adapter.run_task + repricing. Keep the function for callers
# that aren't the runner (they typically want an end-to-end helper).
def run_phase1_task(
    adapter,
    task: dict,
    run_cfg,
    *,
    baseline_pricing_id: str,
    baseline_tier: Tier,
    pricing: Pricing,
    domain: str,
) -> BaselineResult:
    tr = adapter.run_task(task, run_cfg, domain=domain)
    return reprice_baseline(
        tr,
        task_id=str(task["id"]),
        domain=domain,
        baseline_pricing_id=baseline_pricing_id,
        baseline_tier=baseline_tier,
        pricing=pricing,
    )
