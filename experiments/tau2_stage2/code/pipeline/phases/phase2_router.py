"""Phase 2 — Sonnet-4.5 step labeler via CommonStack. Routing axis ONLY.

`slice_type` is NOT labeled here (PROJECT_CHARTER.md §3.12 / spec §5) — it is
computed deterministically in Phase 4 from the step's morphology. The labeler
only emits `replaceability` + `start_tier` + `reason`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from core.configs.loader import Pricing
from core.cost_accounting import per_record_cost
from core.schemas.artifacts import AnalysisResult, BaselineResult, LabelerMeta, StepAnalysis

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "phase2_step_labeler.txt"

_LABEL_STEPS_TOOL = {
    "type": "function",
    "function": {
        "name": "label_steps",
        "description": "Emit per-step labels.",
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "integer", "minimum": 1},
                            "replaceability": {"enum": ["replaceable", "keep"]},
                            "start_tier": {"enum": ["low", "mid", "mid-high"]},
                            "reason": {"type": "string"},
                        },
                        "required": ["step", "replaceability", "start_tier", "reason"],
                    },
                },
            },
            "required": ["steps"],
        },
    },
}


async def run_phase2(
    baseline: BaselineResult,
    client,
    analyzer_model: str,
    pricing: Pricing | None = None,
) -> AnalysisResult:
    prompt = _PROMPT_PATH.read_text().replace("{total_steps}", str(len(baseline.steps)))
    trajectory_serialized = _serialize_trajectory(baseline)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": trajectory_serialized},
    ]
    res = await client.call(
        model=analyzer_model,
        messages=messages,
        tools=[_LABEL_STEPS_TOOL],
        max_tokens=4000,
    )
    if not res.success or not res.tool_calls:
        raise RuntimeError(f"Phase 2 labeler failed: {res.error}")
    raw = res.tool_calls[0]["function"]["arguments"]
    data = json.loads(raw) if isinstance(raw, str) else raw
    steps = [
        StepAnalysis(
            step=int(d["step"]),
            replaceability=d["replaceability"],
            start_tier=d["start_tier"],
            reason=d["reason"],
        )
        for d in data["steps"]
    ]
    cost_usd = (
        per_record_cost(
            model_id=analyzer_model,
            input_tokens=res.input_tokens,
            output_tokens=res.output_tokens,
            cached_read_tokens=res.cached_read_tokens,
            cached_write_tokens=res.cached_write_tokens,
            pricing=pricing,
        )
        if pricing is not None
        else 0.0
    )
    # Hash the user-message string actually sent to the labeler (the
    # serialized trajectory). NOT the system message — that's the static
    # labeler prompt template; the trajectory is what determines what was
    # being labeled.
    prompt_hash = hashlib.sha256(trajectory_serialized.encode("utf-8")).hexdigest()
    raw_args = raw if isinstance(raw, str) else json.dumps(raw)
    meta = LabelerMeta(
        model_id=analyzer_model,
        prompt_hash=prompt_hash,
        raw_arguments=raw_args,
        input_tokens=res.input_tokens,
        output_tokens=res.output_tokens,
    )
    return AnalysisResult(
        analyzer_model=analyzer_model,
        cost_usd=cost_usd,
        steps=steps,
        meta=meta,
    )


def _serialize_trajectory(b: BaselineResult) -> str:
    """Compact trajectory serialization for the labeler.

    Caps per-step response content to ~2 KB to keep the total under
    Sonnet-4.5's context even on long trajectories. If the total still
    exceeds ~15k input tokens at call time, T2.2 in the spec authorizes
    chunked labeling — not yet wired here; add when a long-trajectory
    task surfaces in preflight.
    """
    lines = [f"Task {b.task_id} ({b.domain}); steps={len(b.steps)}"]
    for i, s in enumerate(b.steps, start=1):
        lines.append(f"-- step {i} --")
        lines.append(f"response: {json.dumps(s.response)[:2000]}")
    return "\n".join(lines)
