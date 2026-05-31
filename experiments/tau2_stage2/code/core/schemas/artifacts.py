"""Pydantic schemas for every artifact produced by the pipeline.

Tier labels use the hyphenated English convention (PROJECT_CHARTER.md §3.4).
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt

Tier = Literal["low", "mid", "mid-high", "high"]
Replaceability = Literal["replaceable", "keep"]
SliceType = Literal[
    "strict_json",
    "single_tool_call",
    "tool_result_to_json",
    "clarify",
    "analytical",
    "multi_tool",
    "other",
]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class StepData(_Frozen):
    """One model invocation within a multi-turn trajectory."""
    input_messages: list[dict[str, Any]]
    functions: list[dict[str, Any]]
    response: dict[str, Any]
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    cached_read_tokens: NonNegativeInt
    cached_write_tokens: NonNegativeInt
    actual_usd: NonNegativeFloat
    latency_ms: NonNegativeInt
    # Provider-reported snapshot id from the response body (e.g.
    # "gpt-5.2" under CommonStack or "openai/gpt-5.2-20251211" under
    # OpenRouter). None if the underlying response didn't expose one.
    model_snapshot: str | None = None


class TrajectoryData(_Frozen):
    passed: bool
    actual_usd: NonNegativeFloat
    fixed_tier_usd: NonNegativeFloat
    steps: list[StepData]
    error: str | None = None


class BaselineResult(_Frozen):
    task_id: str
    domain: str
    passed: bool
    pass_at_k: float
    actual_usd: NonNegativeFloat
    simulator_usd: NonNegativeFloat
    steps: list[StepData]
    # Task-level agent context (constant across steps within a task). The
    # trajectory τ² returns is the conversational trace only — the agent's
    # domain policy (system prompt) and tool schemas are held on the agent
    # object itself. We capture them so phase 4 can emit self-contained
    # supervision records: student training needs the full prompt the
    # baseline agent saw (policy + tools + prior turns).
    system_prompt: str = ""
    tools: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class StepAnalysis(_Frozen):
    step: int                       # 1-based (§3.6)
    replaceability: Replaceability
    start_tier: Tier
    reason: str


class LabelerMeta(_Frozen):
    model_id: str
    prompt_hash: str
    raw_arguments: str
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0


class AnalysisResult(_Frozen):
    analyzer_model: str
    cost_usd: NonNegativeFloat
    steps: list[StepAnalysis]
    meta: LabelerMeta | None = None


class Attempt(_Frozen):
    step: int                       # 1-based
    model_id: str
    tier: Tier
    passed: bool
    actual_usd: NonNegativeFloat
    fixed_tier_usd: NonNegativeFloat
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    cached_read_tokens: NonNegativeInt = 0
    cached_write_tokens: NonNegativeInt = 0
    response: str | None = None
    termination_reason: str | None = None
    # Total upstream-reported cost for the entire probe trajectory — the
    # substituted step itself, the baseline-model rerun on the tail, and the
    # simulator turns. Different from `actual_usd` (only the substituted
    # step's per_record_cost). Used for Phase-3 budget accounting; sourced
    # from tau2 / LiteLLM's own cost field, so subject to upstream pricing.
    probe_agent_cost_usd: NonNegativeFloat = 0.0
    probe_user_cost_usd: NonNegativeFloat = 0.0


class LockResult(_Frozen):
    step: int                       # 1-based
    model_id: str
    tier: Tier
    actual_usd: NonNegativeFloat
    fixed_tier_usd: NonNegativeFloat
    # "phase3" / "phase3_5" = step probed and a cheap model locked.
    # "keep_implicit"      = step was either zero-cost (no LLM call backed it)
    #                        or probed with no passing cheap candidate; the
    #                        baseline (high-tier) model stands. Emitted so the
    #                        lock map is a complete per-step assignment.
    source: Literal["phase3", "phase3_5", "keep_implicit"]


class ExplorationLog(_Frozen):
    attempts: list[Attempt]
    locked_results: list[LockResult]
    # Flat message list (plain dicts) from the trajectory after ALL probe-all
    # locks were applied — i.e. the "current_messages" state at the end of
    # run_phase3_probe_all. Used by Phase 4 to thread the locked prefix into
    # supervision records. Empty when legacy Phase-3 (analyzer-driven) was
    # used instead of probe-all.
    locked_trajectory: list[dict[str, Any]] = Field(default_factory=list)


class Phase3Checkpoint(_Frozen):
    """Mid-flight Phase-3 progress for one task.

    Written after each step's lock decision so a crash inside Phase 3 doesn't
    lose the probes already paid for. Once Phase 3 finishes the runner emits
    the final ExplorationLog as `phase3_attempts.json` and deletes this file.
    `last_completed_step` is 0-based; -1 means no step has been processed yet.
    """
    last_completed_step: int
    attempts: list[Attempt]
    locks: list[LockResult]
    current_messages: list[dict[str, Any]]


class CostBreakdown(_Frozen):
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    cached_read_tokens: NonNegativeInt
    cached_write_tokens: NonNegativeInt
    actual_usd: NonNegativeFloat
    fixed_tier_usd: NonNegativeFloat


class SupervisionRecord(_Frozen):
    id: str
    benchmark: str
    adapter: str
    subset: str                     # e.g. "retail"
    task_id: str
    step_index: int = Field(ge=1)   # 1-based
    total_steps: int = Field(ge=1)
    input: dict[str, Any]           # {messages, functions, response_format}
    slice_type: SliceType
    baseline_model: str
    optimal_model: str
    optimal_tier: Tier
    baseline_output: dict[str, Any]
    optimal_output: dict[str, Any]
    costs: dict[Literal["baseline", "optimal"], CostBreakdown]
    reason: str
    collected_at: str               # YYYY-MM-DD
    pillar_extras: dict[str, Any] = Field(default_factory=dict)


class CaseJSON(_Frozen):
    task_id: str
    benchmark: str
    adapter: str
    subset: str
    baseline: BaselineResult
    analysis: AnalysisResult
    exploration: ExplorationLog
    locked_swaps: list[LockResult]
    actual_usd: NonNegativeFloat
    fixed_tier_usd: NonNegativeFloat


class LLMSpec(BaseModel):
    """A minimum LLM identity + auth bundle for plugging into a benchmark runner.

    `model` is the LiteLLM-style id (e.g. "openai/claude-opus-4-6" when the
    provider is OpenAI-compatible). `args` is forwarded verbatim as litellm
    keyword arguments — `api_base`, `api_key`, `custom_llm_provider`, etc.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    args: dict[str, Any] = Field(default_factory=dict)


class RunTaskConfig(BaseModel):
    """Per-task run knobs. Adapter-agnostic."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    agent: LLMSpec
    user: LLMSpec
    seed: int
    max_steps: int = 100
    max_errors: int = 10


class TaskRunResult(BaseModel):
    """Outcome of running one benchmark task end-to-end.

    `steps` are the agent-side steps in trajectory order (user-sim turns are
    excluded since they are not part of our routing surface). `raw_simulation`
    holds the adapter-specific trajectory object so callers can do
    benchmark-specific post-processing without re-running.
    """
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    passed: bool
    reward: float
    termination_reason: str
    steps: list[StepData]
    agent_cost_usd: NonNegativeFloat = 0.0
    user_cost_usd: NonNegativeFloat = 0.0
    system_prompt: str = ""
    tools: list[dict[str, Any]] = Field(default_factory=list)
    # Flat, JSON-safe post-run trajectory (system + user + assistant + tool
    # turns as plain dicts). Populated by adapters so that Phase-3 sequential
    # locking can thread the post-lock trajectory into the next probe without
    # needing to pick apart `raw_simulation`. Empty list when unavailable.
    messages: list[dict[str, Any]] = Field(default_factory=list)
    raw_simulation: Any = None


class BaselineArtifact(BaseModel):
    """Self-contained per-task baseline record.

    Carries everything Phase 1+ needs (steps for repricing) and Phase 3
    needs (raw_messages for substitution). Persisted to
    `phase0_baseline/<task_id>.json`. `extra="forbid"` to surface schema drift.
    """

    model_config = ConfigDict(extra="forbid", frozen=False)

    # Identity
    task_id: str
    domain: str
    benchmark: str = "tau2_bench"
    adapter: str = "tau2_bench"

    # Run config (echoed for reproducibility)
    seed: int
    baseline_model: str
    simulator_model: str
    max_steps: int
    max_errors: int

    # Outcome
    passed: bool
    reward: float
    termination_reason: str
    error: str | None = None

    # Trajectory — Phase 3 substitution feeds this back into
    # tau2's `task.initial_state.message_history`.
    raw_messages: list[dict]
    system_prompt: str | None
    tools: list[dict]

    # Per-step accounting (mirrors what `_extract_agent_steps` produces).
    steps: list[StepData]

    # Cost (raw — Phase 1 reprices via pricing.yaml).
    agent_cost_usd: float
    user_cost_usd: float
    simulator_usd: float = 0.0

    # Provenance
    collected_at: str
    pipeline_version: str = "v2.resumable"
