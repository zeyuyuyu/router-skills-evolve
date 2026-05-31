"""τ²-bench adapter — load / run / grade wrappers over the v1.0.0 runner.

Pinned to tau2-bench v1.0.0 (SHA 17e07b1d), sourced from a personal fork
at tonyyunyang/tau2-bench-evol-llm so reproducibility does not depend on
sierra-research keeping the tag in place. The plan's original v0.2.1 tag
does not exist upstream.

Design: we call τ²'s own orchestrator + evaluator as a library rather than
reimplementing the agentic loop. LLM routing goes through LiteLLM via
`llm_args` — CommonStack is OpenAI-compatible so the same api_base+api_key
pair drives both the agent and the user simulator.

Dual-control (telecom): user-side tool calls land on `ToolCall.requestor ==
"user"`; the step extractor here drops them so downstream cost / slice
filtering only sees agent-side steps.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.schemas.artifacts import (
    LLMSpec,
    RunTaskConfig,
    StepData,
    TaskRunResult,
    Tier,
)

if TYPE_CHECKING:
    from core.schemas.artifacts import BaselineArtifact


class Tau2BenchAdapter:
    benchmark_name = "tau2_bench"

    def __init__(self, vendor_root: Path, domain: str | None = None) -> None:
        """
        vendor_root: path to the cloned tau2-bench repo (vendor/tau2-bench).
        domain: default domain for run_task; can be overridden per-call by
            providing a domain-bearing task or overriding via keyword.
        """
        self._root = Path(vendor_root)
        self._domains_root = self._root / "data" / "tau2" / "domains"
        self._default_domain = domain
        self._nl_judge_routed = False

    def _route_nl_judge_through(self, cs_args: dict[str, Any]) -> None:
        """Monkey-patch τ²'s NL-assertion judge to hit our active provider
        instead of raw OpenAI.

        τ² hardcodes `gpt-4.1-2025-04-14` for NL-assertion grading and reads
        the OpenAI key from OPENAI_API_KEY. We override to `openai/gpt-5.2`
        and pipe through `cs_args` (the same {api_base, api_key,
        custom_llm_provider} the agent + user simulator use). Switching to
        gpt-5.2 also makes the judge work under --provider openrouter:
        OpenRouter does not host an `openai/gpt-4.1` slug, but it does host
        `openai/gpt-5.2` (which is also the pinned user simulator). gpt-5.2
        is at least as capable a judge as gpt-4.1 on retail NL-assertion
        grading, so the change is safe under --provider commonstack too.

        Must patch the evaluator module's own names — `evaluator_nl_assertions`
        does `from tau2.config import DEFAULT_LLM_NL_ASSERTIONS`, which copies
        the value at its own import time; mutating `tau2.config` alone does
        nothing. Idempotent.
        """
        if self._nl_judge_routed:
            return
        from tau2 import config as tau2_config
        from tau2.evaluator import evaluator_nl_assertions as nl_mod
        # LiteLLM strips the leading `openai/` as the provider prefix; the
        # api_base expects `openai/gpt-5.2` on the wire, so we double-prefix
        # here (same convention as the user simulator's model string in
        # runner.py).
        model = "openai/openai/gpt-5.2"
        # The caller passes the AGENT's args (api_base + api_key +
        # custom_llm_provider + agent's extra_body). Under --provider
        # openrouter, the agent's extra_body pins routing to "anthropic" —
        # which would (correctly!) refuse to serve openai/gpt-5.2. Strip the
        # caller's extra_body and pin the judge directly to OpenAI; under
        # --provider commonstack, OR-style extra_body is ignored by CS so
        # this pin is harmless.
        base_args = {k: v for k, v in cs_args.items() if k != "extra_body"}
        args = {
            "temperature": tau2_config.DEFAULT_LLM_NL_ASSERTIONS_TEMPERATURE,
            **base_args,
            "extra_body": {
                "provider": {"order": ["openai"], "allow_fallbacks": False}
            },
        }
        tau2_config.DEFAULT_LLM_NL_ASSERTIONS = model
        tau2_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS = args
        nl_mod.DEFAULT_LLM_NL_ASSERTIONS = model
        nl_mod.DEFAULT_LLM_NL_ASSERTIONS_ARGS = args
        self._nl_judge_routed = True

    # ------------------------------------------------------------------ loaders

    def load_tasks(self, subset: str) -> list[dict[str, Any]]:
        """Return the raw tasks.json entries for a domain (retail/airline/telecom)."""
        path = self._domains_root / subset / "tasks.json"
        with path.open() as f:
            data = json.load(f)
        return list(data)

    def load_ground_truth(self, subset: str) -> dict[str, Any]:
        """Return per-task GT keyed by task id.

        We only expose `evaluation_criteria.actions` (gold action trace).
        τ²'s DB-state check is trajectory-vs-env-diff, not a stored target
        state — so there is no `expected_db_state` field despite the plan's
        earlier assumption.
        """
        tasks = self.load_tasks(subset)
        out: dict[str, Any] = {}
        for t in tasks:
            crit = t.get("evaluation_criteria") or {}
            out[t["id"]] = {
                "actions": crit.get("actions"),
                "reward_basis": crit.get("reward_basis"),
            }
        return out

    # -------------------------------------------------------------------- run

    def run_task(
        self,
        task: Any,
        config: RunTaskConfig,
        *,
        domain: str | None = None,
    ) -> TaskRunResult:
        """Run one task end-to-end via τ²'s half-duplex orchestrator.

        `task` may be either a raw dict (from load_tasks) or a τ² Task object.
        `domain` falls back to the adapter's default_domain when not given.
        """
        # Lazy imports — τ² has heavy init side effects (registry, LiteLLM,
        # loguru config) we don't want to pay unless run_task is actually used.
        from tau2.data_model.tasks import Task
        from tau2.data_model.simulation import TextRunConfig
        from tau2.runner import build_text_orchestrator
        from tau2.runner.simulation import run_simulation

        if isinstance(task, dict):
            task_obj = Task.model_validate(task)
        else:
            task_obj = task

        effective_domain = domain or self._default_domain
        if effective_domain is None:
            raise ValueError(
                "domain is required: pass via run_task(..., domain=...) or "
                "construct Tau2BenchAdapter(vendor_root=..., domain=...)."
            )

        text_cfg_kwargs: dict[str, Any] = {
            "domain": effective_domain,
            "agent": "llm_agent",
            "llm_agent": config.agent.model,
            "llm_args_agent": dict(config.agent.args),
            "user": "user_simulator",
            "llm_user": config.user.model,
            "llm_args_user": dict(config.user.args),
            "seed": config.seed,
            "max_steps": config.max_steps,
            "max_errors": config.max_errors,
            "num_trials": 1,
        }
        text_cfg = TextRunConfig(**text_cfg_kwargs)

        self._route_nl_judge_through(dict(config.agent.args))
        orch = build_text_orchestrator(text_cfg, task_obj, seed=config.seed)
        system_prompt, tools = _capture_agent_context(orch)
        sim = run_simulation(
            orch, evaluation_type=_evaluation_type_for(task_obj)
        )

        reward = float(sim.reward_info.reward) if sim.reward_info else 0.0
        return TaskRunResult(
            passed=reward > 0.0,
            reward=reward,
            termination_reason=str(sim.termination_reason),
            steps=self._extract_agent_steps(sim.messages or []),
            agent_cost_usd=float(getattr(sim, "agent_cost", 0.0) or 0.0),
            user_cost_usd=float(getattr(sim, "user_cost", 0.0) or 0.0),
            system_prompt=system_prompt,
            tools=tools,
            raw_simulation=sim,
        )

    # --------------------------------------------------- single-step substitution

    def run_task_with_substitution(
        self,
        task: Any,
        config: RunTaskConfig,
        *,
        baseline_messages: list[Any],
        step_idx: int,
        sub: LLMSpec,
        domain: str | None = None,
        evaluation_type: Any = None,
    ) -> TaskRunResult:
        """Replay a baseline trajectory but swap the agent model for step `step_idx` only.

        `step_idx` is 0-based and refers to the `step_idx`-th AssistantMessage in
        the baseline trajectory (not τ²'s internal `step_count`, which alternates
        across agent / user / env).

        `baseline_messages` is the ordered message list from a prior successful
        baseline run (e.g. `sim.messages`). We feed the prefix up to — but not
        including — the target AssistantMessage into
        `task.initial_state.message_history`, which makes the agent / user / env
        resume from that state without re-running the prefix. The substitute
        model then generates exactly one agent message; from there on the
        baseline model is restored and the trajectory continues live.
        """
        from copy import deepcopy

        from tau2.data_model.message import AssistantMessage
        from tau2.data_model.simulation import TextRunConfig
        from tau2.data_model.tasks import InitialState, Task
        from tau2.evaluator.evaluator import evaluate_simulation
        from tau2.orchestrator.orchestrator import Role
        from tau2.runner import build_text_orchestrator

        effective_domain = domain or self._default_domain
        if effective_domain is None:
            raise ValueError("domain is required")

        task_obj = task if not isinstance(task, dict) else Task.model_validate(task)

        # Re-hydrate persisted baseline dicts into Pydantic Messages so τ²'s
        # isinstance-based orchestrator / initial_state logic works. Without
        # this the substitution silently replays the baseline (see
        # _parse_messages docstring).
        parsed_messages = _parse_messages(baseline_messages)
        prefix = _prefix_up_to_agent_msg(parsed_messages, step_idx)
        resume_task = deepcopy(task_obj)
        if resume_task.initial_state is None:
            resume_task.initial_state = InitialState(
                initialization_data=None,
                initialization_actions=None,
                message_history=prefix,
            )
        else:
            resume_task.initial_state.message_history = prefix

        text_cfg_kwargs: dict[str, Any] = {
            "domain": effective_domain,
            "agent": "llm_agent",
            "llm_agent": config.agent.model,
            "llm_args_agent": dict(config.agent.args),
            "user": "user_simulator",
            "llm_user": config.user.model,
            "llm_args_user": dict(config.user.args),
            "seed": config.seed,
            "max_steps": config.max_steps,
            "max_errors": config.max_errors,
            "num_trials": 1,
        }
        text_cfg = TextRunConfig(**text_cfg_kwargs)

        self._route_nl_judge_through(dict(config.agent.args))
        orch = build_text_orchestrator(text_cfg, resume_task, seed=config.seed)
        system_prompt, tools = _capture_agent_context(orch)
        baseline_llm = orch.agent.llm
        baseline_args = dict(orch.agent.llm_args)
        # τ²'s public run() sets these before initialize(); _finalize() dies if
        # _run_start_perf is still None.
        import time as _time
        from tau2.utils.utils import get_now as _get_now
        orch._run_start_time = _get_now()
        orch._run_start_perf = _time.perf_counter()
        orch.initialize()

        substituted: Any = False
        # Substitution gating: τ²'s orchestrator initializes `self.trajectory =
        # message_history` (orchestrator.py:625 at SHA 17e07b1d), so after
        # `orch.initialize()` the trajectory ALREADY contains every prefix
        # message — including the K assistant messages in the prefix (where
        # K = len(_agent_prefix(prefix))). Therefore
        # `_count_assistant_messages(orch.trajectory) == step_idx` is the
        # condition that fires exactly when the NEXT agent step is the target
        # substitution step. Earlier code wrote
        # `step_idx - len(_agent_prefix(prefix))` on the RHS, which expected
        # trajectory to track only NEW assistants — that assumption never
        # held against tau2 1.0.0, so the swap silently failed for any
        # step_idx > 0 and the baseline silently ran every probe. Verified via
        # iter-4 smoke test on 2026-05-23: with the buggy RHS, Method B at
        # step_idx=1 logged zero student LLM calls; with this fix the student
        # is called exactly once at the substitution point.
        while not orch.done:
            if (
                substituted is False
                and orch.to_role == Role.AGENT
                and _count_assistant_messages(orch.trajectory) == step_idx
            ):
                orch.agent.llm = sub.model
                orch.agent.llm_args = {**baseline_args, **dict(sub.args)}
                substituted = "pending"
            elif substituted == "pending" and orch.to_role != Role.AGENT:
                orch.agent.llm = baseline_llm
                orch.agent.llm_args = baseline_args
                substituted = True
            orch.step()
            orch._check_termination()  # private-by-convention; called by public run()

        sim = orch._finalize()  # private; only path to cost/duration/cleanup
        # evaluation_type override lets Phase 3 grade with ACTIONS only (no
        # NL-judge) so the inner search loop is deterministic. Falls back to
        # the task-native grader (ALL_WITH_NL_ASSERTIONS for retail, etc.)
        # when the caller doesn't specify — preserves Phase 0 behavior.
        eval_type = evaluation_type or _evaluation_type_for(task_obj)
        sim.reward_info = evaluate_simulation(
            simulation=sim,
            task=task_obj,
            evaluation_type=eval_type,
            solo_mode=getattr(orch, "solo_mode", False),
            domain=effective_domain,
            mode=_communication_mode(),
        )

        reward = float(sim.reward_info.reward) if sim.reward_info else 0.0
        # Serialize τ²'s Pydantic messages into plain dicts so Phase 3 can
        # thread them as the evolving prefix without dragging τ² types
        # through downstream code.
        flat_messages: list[dict[str, Any]] = []
        for m in (sim.messages or []):
            if hasattr(m, "model_dump"):
                flat_messages.append(m.model_dump(mode="json"))
            elif isinstance(m, dict):
                flat_messages.append(m)
            else:
                flat_messages.append(dict(m))
        return TaskRunResult(
            passed=reward > 0.0,
            reward=reward,
            termination_reason=str(sim.termination_reason),
            steps=self._extract_agent_steps(sim.messages or []),
            agent_cost_usd=float(getattr(sim, "agent_cost", 0.0) or 0.0),
            user_cost_usd=float(getattr(sim, "user_cost", 0.0) or 0.0),
            system_prompt=system_prompt,
            tools=tools,
            messages=flat_messages,
            raw_simulation=sim,
        )

    # ---------------------------------------------------------- step extraction

    @staticmethod
    def _extract_agent_steps(messages: list[Any]) -> list[StepData]:
        """Walk a trajectory and emit a StepData per agent-emitted message.

        Token totals come from `msg.usage`; cache counters live in
        `msg.raw_data["usage"]` per the Explore notes. `actual_usd` is left 0
        — downstream `core.cost_accounting` recomputes it with the pricing yaml.
        """
        steps: list[StepData] = []
        for i, msg in enumerate(messages):
            role = getattr(msg, "role", None)
            if role != "assistant":
                continue
            # Only keep assistant-authored calls (exclude any user-originated
            # ones propagated through dual-control machinery).
            requestor = getattr(msg, "requestor", "assistant")
            if requestor != "assistant":
                continue

            usage = dict(getattr(msg, "usage", None) or {})
            raw_usage = dict((getattr(msg, "raw_data", None) or {}).get("usage") or {})
            prompt_details = dict(raw_usage.get("prompt_tokens_details") or {})
            cached_read = int(
                prompt_details.get("cached_tokens")
                or raw_usage.get("cache_read_input_tokens")
                or 0
            )
            cached_write = int(
                raw_usage.get("cache_creation_input_tokens") or 0
            )

            input_messages = [_serialize_msg(m) for m in messages[:i]]
            functions = _serialize_functions(getattr(msg, "functions", None))
            response = {
                "content": getattr(msg, "content", "") or "",
                "tool_calls": _serialize_tool_calls(
                    getattr(msg, "tool_calls", None) or []
                ),
            }
            latency_s = float(getattr(msg, "generation_time_seconds", 0.0) or 0.0)

            # Extract the provider's snapshot id from raw_data if present.
            # AssistantMessage.raw_data is the full litellm response dict; it
            # has a "model" key when the provider reports it (CS: bare name;
            # OR: date-versioned slug). Safe-fallback to None on any miss.
            raw = getattr(msg, "raw_data", None) or {}
            snapshot = raw.get("model") if isinstance(raw, dict) else None

            steps.append(
                StepData(
                    input_messages=input_messages,
                    functions=functions,
                    response=response,
                    input_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    output_tokens=int(usage.get("completion_tokens", 0) or 0),
                    cached_read_tokens=cached_read,
                    cached_write_tokens=cached_write,
                    actual_usd=0.0,
                    latency_ms=int(latency_s * 1000),
                    model_snapshot=snapshot,
                )
            )
        return steps


def _serialize_msg(m: Any) -> dict[str, Any]:
    """Convert a τ² Message into a plain dict our downstream code can consume."""
    if hasattr(m, "model_dump"):
        return m.model_dump()
    if isinstance(m, dict):
        return m
    return {"role": getattr(m, "role", None), "content": getattr(m, "content", None)}


def _serialize_tool_calls(tcs: list[Any]) -> list[dict[str, Any]]:
    out = []
    for tc in tcs:
        if hasattr(tc, "model_dump"):
            out.append(tc.model_dump())
        elif isinstance(tc, dict):
            out.append(tc)
        else:
            out.append(
                {
                    "id": getattr(tc, "id", None),
                    "name": getattr(tc, "name", None),
                    "arguments": getattr(tc, "arguments", None),
                }
            )
    return out


def _serialize_functions(fns: Any) -> list[dict[str, Any]]:
    if not fns:
        return []
    if isinstance(fns, list):
        return [_serialize_msg(f) for f in fns]
    return []


def _parse_messages(msgs: list[Any]) -> list[Any]:
    """Re-hydrate a list[dict] (e.g. from BaselineArtifact.raw_messages) into
    τ²'s Pydantic Message objects.

    The pipeline persists baseline trajectories as plain dicts (JSON-safe).
    `InitialState.message_history` is typed `list[Message]`, and several
    downstream helpers here (`_prefix_up_to_agent_msg`,
    `_count_assistant_messages`, `_agent_prefix`) dispatch on isinstance.
    Without this step, a probed substitution silently replays the baseline:
    the prefix never truncates, the orchestrator finds all assistant turns
    already present, and the cheap model never gets called.

    Idempotent: already-Pydantic entries pass through unchanged.
    """
    from tau2.data_model.message import (
        AssistantMessage,
        MultiToolMessage,
        SystemMessage,
        ToolMessage,
        UserMessage,
    )

    parsed: list[Any] = []
    for m in msgs:
        if not isinstance(m, dict):
            parsed.append(m)
            continue
        role = m.get("role")
        if role == "system":
            parsed.append(SystemMessage.model_validate(m))
        elif role == "assistant":
            parsed.append(AssistantMessage.model_validate(m))
        elif role == "user":
            parsed.append(UserMessage.model_validate(m))
        elif role == "tool":
            # τ² ToolMessage vs MultiToolMessage is distinguished by the
            # presence of `messages` (list of nested tool outputs). Fall
            # back to ToolMessage for the common case.
            if "messages" in m:
                parsed.append(MultiToolMessage.model_validate(m))
            else:
                parsed.append(ToolMessage.model_validate(m))
        else:
            raise ValueError(f"Unknown message role in baseline trajectory: {role!r}")
    return parsed


def _prefix_up_to_agent_msg(messages: list[Any], step_idx: int) -> list[Any]:
    """Return messages up to but NOT including the step_idx-th AssistantMessage.

    step_idx is 0-based. If the trajectory has fewer than step_idx + 1
    AssistantMessages, the full list is returned. Dict entries with
    role="assistant" are also counted, so callers can pass either
    deserialized dicts or Pydantic Message objects.
    """
    from tau2.data_model.message import AssistantMessage

    out: list[Any] = []
    seen = 0
    for m in messages:
        is_assistant = (
            isinstance(m, AssistantMessage)
            or (isinstance(m, dict) and m.get("role") == "assistant")
        )
        if is_assistant:
            if seen == step_idx:
                break
            seen += 1
        out.append(m)
    return out


def _count_assistant_messages(messages: list[Any]) -> int:
    from tau2.data_model.message import AssistantMessage

    return sum(
        1
        for m in messages
        if isinstance(m, AssistantMessage)
        or (isinstance(m, dict) and m.get("role") == "assistant")
    )


def _agent_prefix(messages: list[Any]) -> list[Any]:
    """AssistantMessages within a prefix list. Used for agent-turn counting math."""
    from tau2.data_model.message import AssistantMessage

    return [
        m
        for m in messages
        if isinstance(m, AssistantMessage)
        or (isinstance(m, dict) and m.get("role") == "assistant")
    ]


def _communication_mode():
    from tau2.orchestrator.modes import CommunicationMode

    return CommunicationMode.HALF_DUPLEX


def _capture_agent_context(orch: Any) -> tuple[str, list[dict[str, Any]]]:
    """Pull the agent's system prompt + OpenAI-schema tool list off the orch.

    τ² constructs these at `build_text_orchestrator` time and holds them on
    `orch.agent` — the conversational trajectory (`sim.messages`) does NOT
    include the system message, because τ² re-injects it fresh at every
    `generate()` call via `state.system_messages + state.messages`. If we
    want self-contained supervision records (so the student sees the same
    prompt the baseline saw), we have to grab these out-of-band.
    """
    agent = getattr(orch, "agent", None)
    system_prompt = str(getattr(agent, "system_prompt", "") or "")
    tool_objs = list(getattr(agent, "tools", None) or [])
    tools: list[dict[str, Any]] = []
    for t in tool_objs:
        schema = getattr(t, "openai_schema", None)
        if schema is None:
            continue
        tools.append(schema)
    return system_prompt, tools


def _evaluation_type_for(task_obj: Any):
    """Pick the minimum evaluation type that satisfies the task's reward_basis.

    Retail tasks almost always include NL_ASSERTION in reward_basis, which
    requires EvaluationType.ALL_WITH_NL_ASSERTIONS (else the evaluator raises
    'NL assertions are part of the reward basis, but they are not being
    evaluated'). Airline/telecom don't use NL, so EvaluationType.ALL skips
    the extra NL-judge call and the associated cost.
    """
    from tau2.evaluator.evaluator import EvaluationType

    crit = getattr(task_obj, "evaluation_criteria", None)
    basis = [str(b) for b in (getattr(crit, "reward_basis", None) or [])]
    needs_nl = any("NL_ASSERTION" in b for b in basis)
    return (
        EvaluationType.ALL_WITH_NL_ASSERTIONS if needs_nl else EvaluationType.ALL
    )


def serialize_baseline(
    tr: TaskRunResult,
    *,
    task_id: str,
    domain: str,
    seed: int,
    baseline_model: str,
    simulator_model: str,
    max_steps: int,
    max_errors: int,
) -> "BaselineArtifact":
    """Convert a TaskRunResult into a serializable BaselineArtifact.

    Coerces raw_simulation.messages into plain dicts so the artifact
    survives JSON round-trip (tau2 Pydantic Message subtypes don't
    survive a `model_dump_json()` from BaselineArtifact otherwise).
    """
    from datetime import datetime as _dt

    from core.schemas.artifacts import BaselineArtifact

    raw_messages: list[dict] = []
    msgs = getattr(tr.raw_simulation, "messages", None) or []
    for m in msgs:
        if hasattr(m, "model_dump"):
            raw_messages.append(m.model_dump(mode="json"))
        elif isinstance(m, dict):
            raw_messages.append(m)
        else:
            raw_messages.append(dict(m))  # last-resort coercion

    return BaselineArtifact(
        task_id=task_id,
        domain=domain,
        seed=seed,
        baseline_model=baseline_model,
        simulator_model=simulator_model,
        max_steps=max_steps,
        max_errors=max_errors,
        passed=tr.passed,
        reward=tr.reward,
        termination_reason=tr.termination_reason,
        error=None,
        raw_messages=raw_messages,
        system_prompt=tr.system_prompt,
        tools=list(tr.tools) if tr.tools else [],
        steps=list(tr.steps),
        agent_cost_usd=tr.agent_cost_usd,
        user_cost_usd=tr.user_cost_usd,
        collected_at=_dt.now().isoformat(timespec="seconds"),
    )


def deserialize_baseline_messages(artifact: "BaselineArtifact") -> list[dict]:
    """Return the saved raw_messages list as plain dicts.

    `tau2.data_model.tasks.InitialState.message_history` accepts
    list[dict] directly — no need to revive into Pydantic Message
    subtypes for substitution to work.
    """
    return list(artifact.raw_messages)
