"""Phase 4 — assemble CaseJSON + emit real-JSONL supervision records.

Two input modes:

1. Probe-all (analysis=None). Iterates exploration.locked_results where
   source ∈ {phase3, phase3_5} (keep_implicit locks do not yield supervision
   records — the student has nothing to learn from a step that stayed at
   Opus). Pulls each step's prefix + optimal_output from
   exploration.locked_trajectory so the training data reflects the
   cumulative-cheap conversation the student will actually see at
   deployment.

2. Analyzer-driven (analysis=AnalysisResult). Legacy: iterate the
   analyzer's replaceable steps, skip non-replaceable, skip unlocked.
   optimal_output falls back to baseline_output (placeholder) when no
   locked_trajectory is available.

JSONL format: one compact `SupervisionRecord.model_dump_json()` per line.
File opens in append mode so the runner can checkpoint across tasks.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from core.configs.loader import Pricing
from core.cost_accounting import fixed_tier_cost
from core.schemas.artifacts import (
    AnalysisResult,
    BaselineResult,
    CostBreakdown,
    ExplorationLog,
    SupervisionRecord,
    Tier,
)
from core.slice_filter import classify_slice


def run_phase4(
    baseline: BaselineResult,
    analysis: AnalysisResult | None,
    exploration: ExplorationLog,
    adapter_name: str,
    subset: str,
    supervision_path: Path,
    *,
    emitted_marker: Path | None = None,
    baseline_model: str = "anthropic/claude-opus-4-6",
    baseline_tier: Tier | None = None,
    pricing: Pricing | None = None,
    response_format: dict | None = None,
    confirm_passed: bool = True,
    seed: int | None = None,
) -> list[SupervisionRecord]:
    """Append one SupervisionRecord per locked (replaceable) step.

    See module docstring for the two input modes.

    `emitted_marker` is the per-task "already processed" flag file. Path is
    task-local (e.g. `.../tasks/<id>/emitted.ok`). When omitted we fall back
    to the legacy layout `<supervision_path.parent>/.emitted/<task_id>.ok`
    so older tests don't break.
    """
    if emitted_marker is None:
        marker_dir = supervision_path.parent / ".emitted"
        emitted_marker = marker_dir / f"{baseline.task_id}.ok"
    if emitted_marker.exists():
        return []

    if not confirm_passed:
        emitted_marker.parent.mkdir(parents=True, exist_ok=True)
        emitted_marker.touch()
        return []

    lock_by_step = {lock.step: lock for lock in exploration.locked_results}
    records: list[SupervisionRecord] = []
    total_steps = len(baseline.steps)
    today = date.today().isoformat()
    system_message = (
        [{"role": "system", "content": baseline.system_prompt}]
        if baseline.system_prompt
        else []
    )
    task_tools = list(baseline.tools)

    # Mode selection + iteration list.
    if analysis is not None:
        # Legacy analyzer-driven mode.
        iter_steps = [
            s.step for s in analysis.steps if s.replaceability == "replaceable"
        ]
        reason_by_step: dict[int, str] = {s.step: s.reason for s in analysis.steps}
    else:
        # Probe-all mode: iterate locks that came from actual probing.
        iter_steps = [
            lr.step
            for lr in exploration.locked_results
            if lr.source in ("phase3", "phase3_5")
        ]
        reason_by_step = {}

    # Index into the locked trajectory by 1-based assistant-turn position.
    # Each agent-authored assistant message is one "step"; τ²'s dual-control
    # telecom flow tags user-side tool calls with requestor="user" on the
    # assistant message, so skip those.
    locked_prefix_by_step: dict[int, list[dict[str, Any]]] = {}
    locked_response_by_step: dict[int, dict[str, Any]] = {}
    if exploration.locked_trajectory:
        prefix_so_far: list[dict[str, Any]] = []
        assistant_seen = 0
        for msg in exploration.locked_trajectory:
            if msg.get("role") == "assistant" and msg.get("requestor", "assistant") == "assistant":
                assistant_seen += 1
                locked_prefix_by_step[assistant_seen] = list(prefix_so_far)
                locked_response_by_step[assistant_seen] = {
                    "content": msg.get("content") or "",
                    "tool_calls": msg.get("tool_calls") or [],
                }
            prefix_so_far.append(msg)

    for step_num in sorted(set(iter_steps)):
        lock = lock_by_step.get(step_num)
        if lock is None:
            continue
        idx = step_num - 1
        if not (0 <= idx < total_steps):
            continue
        baseline_step = baseline.steps[idx]
        # Compute baseline.fixed_tier_usd: tier-uniform price applied to this
        # step's tokens. Only available when both pricing + baseline_tier
        # were threaded through from the runner (or assemble.py); falls back
        # to 0 for legacy callers / tests that don't pass them.
        if pricing is not None and baseline_tier is not None:
            baseline_fixed = fixed_tier_cost(
                tier=baseline_tier,
                input_tokens=baseline_step.input_tokens,
                output_tokens=baseline_step.output_tokens,
                cached_read_tokens=baseline_step.cached_read_tokens,
                cached_write_tokens=baseline_step.cached_write_tokens,
                pricing=pricing,
            )
        else:
            baseline_fixed = 0.0
        # Prefix: locked trajectory if we have it (probe-all), else baseline
        # step's own input_messages (legacy / no trajectory available).
        if step_num in locked_prefix_by_step:
            step_prefix = locked_prefix_by_step[step_num]
        else:
            step_prefix = list(baseline_step.input_messages)
        # optimal_output: locked trajectory's assistant turn at this step.
        # Falls back to baseline's response (placeholder) when the trajectory
        # isn't available — e.g. legacy analyzer mode with empty locked_trajectory.
        optimal_output = locked_response_by_step.get(step_num, baseline_step.response)
        slice_type = classify_slice(
            {
                "input_messages": step_prefix,
                "response": baseline_step.response,
                "input_meta": {"response_format": response_format},
            }
        )
        seed_segment = f"_seed_{seed}" if seed is not None else ""
        rec = SupervisionRecord(
            id=f"{adapter_name}_{subset}_{baseline.task_id}{seed_segment}_step_{step_num}",
            benchmark=adapter_name,
            adapter=adapter_name,
            subset=subset,
            task_id=baseline.task_id,
            step_index=step_num,
            total_steps=total_steps,
            input={
                "messages": system_message + step_prefix,
                "functions": task_tools or baseline_step.functions,
                "response_format": response_format,
            },
            slice_type=slice_type,
            baseline_model=baseline_model,
            optimal_model=lock.model_id,
            optimal_tier=lock.tier,
            baseline_output=baseline_step.response,
            optimal_output=optimal_output,
            costs={
                "baseline": CostBreakdown(
                    input_tokens=baseline_step.input_tokens,
                    output_tokens=baseline_step.output_tokens,
                    cached_read_tokens=baseline_step.cached_read_tokens,
                    cached_write_tokens=baseline_step.cached_write_tokens,
                    actual_usd=baseline_step.actual_usd,
                    fixed_tier_usd=baseline_fixed,
                ),
                "optimal": CostBreakdown(
                    input_tokens=baseline_step.input_tokens,
                    output_tokens=baseline_step.output_tokens,
                    cached_read_tokens=0,
                    cached_write_tokens=0,
                    actual_usd=lock.actual_usd,
                    fixed_tier_usd=lock.fixed_tier_usd,
                ),
            },
            reason=reason_by_step.get(
                step_num, f"Locked via probe-all at tier {lock.tier}"
            ),
            collected_at=today,
        )
        records.append(rec)

    if records:
        supervision_path.parent.mkdir(parents=True, exist_ok=True)
        with supervision_path.open("a") as f:
            for r in records:
                f.write(r.model_dump_json() + "\n")

    # Always write the marker on successful processing — even if zero records
    # were emitted (e.g. all steps were `keep`-labeled or unlocked). The marker
    # means "we processed this task," not "we wrote records."
    emitted_marker.parent.mkdir(parents=True, exist_ok=True)
    emitted_marker.touch()
    return records
