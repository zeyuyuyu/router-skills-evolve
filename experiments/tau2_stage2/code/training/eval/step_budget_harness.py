"""Step-budget eval harness — Method B (per-step substitution with baseline tail)
+ Method A (E2E supplement). Runs ONE target (= one vLLM-served student) end-to-end.

Methodology:
  For each (task, seed) where the original strong baseline passed, with
  B = #agent-steps-in-baseline:
    Method B (primary): run B rollouts, one per step_idx ∈ {0..B-1}. In each
      rollout, the first step_idx agent messages are taken verbatim from the
      baseline trajectory; the student substitutes at step_idx; the baseline
      model resumes for steps step_idx+1..end. Per-rollout pass iff final
      reward >= 0.5. Aggregate to student_rep_rate = (#passing rollouts) / B.
    Method A (supplement): one extra rollout in which the student handles the
      ENTIRE task from scratch with max_steps = B. Goal-oriented: passed iff
      reward >= 0.5 within B steps. Used for the "autonomy" headline.

Resume semantics:
  Every rollout writes its own JSON file atomically (tmp + rename). On
  restart the harness scans output_dir/raw_eval/ and skips rollouts whose
  files exist AND end with {"_marker": "COMPLETE"}. Lossless resume at the
  per-rollout level.

Credit exhaustion:
  CommonStack returns 402 (or specific error bodies) when funds are out.
  The harness detects these and EXITS 42 cleanly without writing a "FAIL"
  marker for the in-flight rollout. Operator tops up, reruns the same
  command; harness resumes from the same rollout.

Logging:
  Every litellm call (sim / judge / baseline-tail) is recorded to
  output_dir/llm_calls/<rollout_id>.jsonl via litellm.success_callback +
  failure_callback. Full request + response + timing + cost-per-call.

USAGE:
  python -m training.eval.step_budget_harness \
      --target 05_qwen3_5_4b_273 \
      --checkpoint /path/to/checkpoint \
      --bundle-root $BUNDLE_ROOT \
      --baselines-root $BUNDLE_ROOT/eval_baselines \
      --eval-tasks $BUNDLE_ROOT/data_processed/stage2_v1/eval_tasks.jsonl \
      --output-dir $BUNDLE_ROOT/step_budget_outputs/05_qwen3_5_4b_273 \
      --port 8000 --gpu 0 --tp 1 \
      --seed-policy primary
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

# Exit codes:
EXIT_OK = 0
EXIT_VLLM_FAIL = 1
EXIT_ARGS_FAIL = 2
EXIT_NO_BASELINES = 3
EXIT_CREDIT_EXHAUSTED = 42

# vLLM serves under this name regardless of which checkpoint is loaded; see
# stage2_ship/code/training/eval/vllm_serve.sh:61.
SERVED_MODEL_NAME = "evol-llm-student"
DEFAULT_STUDENT_MAX_TOKENS = 2048

# Pricing per million tokens — kept in code (NOT pricing.yaml) because the
# eval is deliberately decoupled from the data-collection pricing config.
# Prompt-token cache reads on CommonStack: cache hit reduces cost by 10x for
# gpt-5.2 (0.175/M vs 1.75/M) — we approximate using NON-cached pricing as
# the upper bound. Verified against /v1/models response on 2026-05-22.
_PRICING = {
    "openai/gpt-5.2": {"prompt": 1.75, "completion": 14.00, "cached_prompt": 0.175},
    # Tau2's NL judge talks to "openai/openai/gpt-5.2" on the wire (litellm
    # provider prefix convention); price the same as openai/gpt-5.2.
    "openai/openai/gpt-5.2": {"prompt": 1.75, "completion": 14.00, "cached_prompt": 0.175},
    "qwen/qwen3.5-397b-a17b": {"prompt": 0.60, "completion": 3.60, "cached_prompt": None},
    "anthropic/claude-opus-4-7": {"prompt": 5.00, "completion": 25.00, "cached_prompt": 0.50},
}


# --------------------------------------------------------------------------- #
# CHOSEN SET                                                                  #
# --------------------------------------------------------------------------- #

def _baseline_path(baselines_root: Path, domain: str, task_id: str, seed: int) -> Path:
    return baselines_root / domain / "tasks" / task_id / f"seed_{seed}" / "phase0_baseline.json"


def _count_agent_steps(raw_messages: list[dict]) -> int:
    """Count assistant messages where requestor == "assistant" (excludes
    user-originated tool calls from the telecom dual-control adapter)."""
    n = 0
    for m in raw_messages:
        if m.get("role") != "assistant":
            continue
        if m.get("requestor", "assistant") != "assistant":
            continue
        n += 1
    return n


def compute_chosen_set(
    *, eval_tasks_jsonl: Path, baselines_root: Path, seed_policy: str,
) -> list[dict]:
    """Return [{domain, task_id, seed, B, baseline_model, baseline_path, ...}].

    seed_policy:
      "primary"   — prefer seed 300 if its baseline passed, else 301.
      "secondary" — pick the OTHER seed (the one not chosen as primary), iff
                    its baseline also passed. Tasks without two passing
                    seeds yield no rows in secondary.
    """
    if seed_policy not in ("primary", "secondary"):
        raise ValueError(f"seed_policy must be 'primary' or 'secondary', got {seed_policy!r}")

    chosen: list[dict] = []
    for line in eval_tasks_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        dom, tid = t["domain"], t["task_id"]
        seeds_avail = t.get("available_seeds", [])

        # Per-seed baseline status
        seed_info: dict[int, dict] = {}
        for seed in seeds_avail:
            p = _baseline_path(baselines_root, dom, tid, seed)
            if not p.exists():
                continue
            try:
                b = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if not b.get("passed"):
                continue
            B = _count_agent_steps(b.get("raw_messages") or [])
            if B <= 0:
                continue
            seed_info[seed] = {
                "B": B,
                "baseline_model": b.get("baseline_model"),
                "baseline_path": str(p),
                "tau2_task_id": b.get("task_id") or tid,
                "baseline_reward": float(b.get("reward") or 0.0),
                "baseline_termination_reason": b.get("termination_reason"),
                "simulator_model": b.get("simulator_model"),
            }

        # Selection logic per seed_policy.
        primary_seed: int | None = None
        if 300 in seed_info:
            primary_seed = 300
        elif 301 in seed_info:
            primary_seed = 301
        if primary_seed is None:
            continue  # no passing baseline at any seed — task is excluded

        if seed_policy == "primary":
            sel = primary_seed
        else:  # secondary
            other = 301 if primary_seed == 300 else 300
            if other not in seed_info:
                continue  # no second-seed passing baseline available
            sel = other

        info = seed_info[sel]
        chosen.append({
            "domain": dom,
            "task_id": tid,
            "seed": sel,
            "B": info["B"],
            "baseline_model": info["baseline_model"],
            "baseline_path": info["baseline_path"],
            "tau2_task_id": info["tau2_task_id"],
            "baseline_reward": info["baseline_reward"],
            "baseline_termination_reason": info["baseline_termination_reason"],
            "simulator_model": info["simulator_model"],
            "max_steps_cap_field": int(t.get("max_steps", 100)),
            "max_errors": int(t.get("max_errors", 10)),
            "bucket": t.get("bucket"),
            "rep_rate_pinned": float(t.get("rep_rate_pinned") or 0.0),
            "expected_locked_steps_summed": int(t.get("expected_locked_steps") or 0),
            "reward_basis": t.get("reward_basis", "ALL_WITH_NL_ASSERTIONS"),
            "tau2_repo_sha": t.get("tau2_repo_sha"),
        })
    # Stable ordering: domain ASC, task_id natural ordering, seed ASC.
    chosen.sort(key=lambda r: (r["domain"], r["task_id"], r["seed"]))
    return chosen


# --------------------------------------------------------------------------- #
# LITELLM CALL CAPTURE                                                        #
# --------------------------------------------------------------------------- #

class CallRecorder:
    """Captures every litellm call to a JSONL file, scoped per rollout.

    Implementation note: tau2's `tau2.utils.llm_utils` module sets
    `litellm.success_callback = []` at import time (verified against pinned
    SHA 17e07b1d, lines 67-69), which would WIPE any callback we register via
    `litellm.success_callback`. To survive that, we register via the NEWER
    `litellm.callbacks` list (the CustomLogger API), which tau2 does NOT
    touch. The two callback systems coexist in litellm.

    Usage:
        rec = CallRecorder(output_dir / "llm_calls", default_rollout_id="...")
        rec.install()         # registers a CustomLogger in litellm.callbacks
        rec.set_rollout("<rollout_id>")
        # ... run rollout ...
        rec.set_rollout(None) # back to default (likely sets up next rollout)

    The recorder is process-wide. We rely on the harness running rollouts
    sequentially within one process; parallelism happens at the BASH level
    (multiple harness processes, each on its own GPU). No cross-process state.
    """

    def __init__(self, calls_dir: Path, default_rollout_id: str = "unscoped") -> None:
        self.calls_dir = calls_dir
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        self._current_rollout_id = default_rollout_id
        self._call_counters: dict[str, int] = defaultdict(int)
        # Per-rollout accumulated cost so the harness can ask "what did this
        # rollout cost so far?" The keys are rollout_ids; values are floats.
        self._cost_per_rollout: dict[str, float] = defaultdict(float)
        self._tokens_per_rollout: dict[str, dict[str, int]] = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "cached_prompt": 0}
        )
        self._logger_instance = None  # set in install()

    def set_rollout(self, rollout_id: str | None) -> None:
        self._current_rollout_id = rollout_id or "unscoped"

    def reset_cost(self, rollout_id: str) -> None:
        self._cost_per_rollout[rollout_id] = 0.0
        self._tokens_per_rollout[rollout_id] = {"prompt": 0, "completion": 0, "cached_prompt": 0}
        self._call_counters[rollout_id] = 0

    def cost_so_far(self, rollout_id: str) -> float:
        return self._cost_per_rollout.get(rollout_id, 0.0)

    def tokens_so_far(self, rollout_id: str) -> dict[str, int]:
        return dict(self._tokens_per_rollout.get(
            rollout_id, {"prompt": 0, "completion": 0, "cached_prompt": 0}))

    def install(self) -> None:
        """Register a CustomLogger in litellm.callbacks. Idempotent.

        We deliberately do NOT use litellm.success_callback / failure_callback
        because tau2.utils.llm_utils resets those lists at module import time.
        litellm.callbacks (CustomLogger-based) is untouched by tau2.
        """
        import litellm
        from litellm.integrations.custom_logger import CustomLogger

        outer = self

        class _Logger(CustomLogger):
            def log_success_event(self, kwargs, response_obj, start_time, end_time):
                outer._record(kwargs, response_obj, start_time, end_time, ok=True, exc=None)

            def log_failure_event(self, kwargs, response_obj, start_time, end_time):
                exc = None
                if response_obj is not None and not hasattr(response_obj, "usage"):
                    exc = repr(response_obj)
                outer._record(kwargs, response_obj, start_time, end_time, ok=False, exc=exc)

            async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
                outer._record(kwargs, response_obj, start_time, end_time, ok=True, exc=None)

            async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
                exc = None
                if response_obj is not None and not hasattr(response_obj, "usage"):
                    exc = repr(response_obj)
                outer._record(kwargs, response_obj, start_time, end_time, ok=False, exc=exc)

        # Avoid double-install on idempotent re-calls.
        if self._logger_instance is None:
            self._logger_instance = _Logger()
            existing = list(litellm.callbacks or [])
            if not any(isinstance(c, _Logger) for c in existing):
                litellm.callbacks = existing + [self._logger_instance]

    def _record(self, kwargs, response_obj, start_time, end_time, *, ok: bool, exc: str | None) -> None:
        rid = self._current_rollout_id
        self._call_counters[rid] += 1
        call_idx = self._call_counters[rid]
        # Compute timings
        try:
            lat_ms = int((end_time - start_time).total_seconds() * 1000)
        except Exception:
            lat_ms = -1

        # Extract usage if available
        usage = {}
        model_snapshot = None
        try:
            usage_obj = getattr(response_obj, "usage", None)
            if usage_obj is not None:
                # litellm normalizes usage to OpenAI-style dict-like
                usage = {
                    "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
                }
                # Cached-token details (OpenAI exposes these in
                # usage.prompt_tokens_details.cached_tokens)
                ptd = getattr(usage_obj, "prompt_tokens_details", None)
                if ptd is not None:
                    cached = getattr(ptd, "cached_tokens", None)
                    if cached is not None:
                        usage["cached_prompt_tokens"] = int(cached)
            model_snapshot = getattr(response_obj, "model", None)
        except Exception:
            pass

        # Estimate cost from tokens × pricing
        model_id = kwargs.get("model") or ""
        cost = _estimate_call_cost(model_id, usage)
        self._cost_per_rollout[rid] += cost
        self._tokens_per_rollout[rid]["prompt"] += usage.get("prompt_tokens", 0)
        self._tokens_per_rollout[rid]["completion"] += usage.get("completion_tokens", 0)
        self._tokens_per_rollout[rid]["cached_prompt"] += usage.get("cached_prompt_tokens", 0)

        rec = {
            "rollout_id": rid,
            "call_idx": call_idx,
            "ok": ok,
            "model_requested": model_id,
            "model_returned": model_snapshot,
            "latency_ms": lat_ms,
            "start_time": str(start_time),
            "end_time": str(end_time),
            "usage": usage,
            "cost_usd_estimate": cost,
            "cumulative_cost_usd_rollout": self._cost_per_rollout[rid],
            "exception": exc,
        }
        # Don't dump full messages/tools into the call log by default —
        # they'd duplicate the raw_eval/<rid>.json simulation dump. We DO
        # keep some lightweight fields for cross-reference.
        rec["request_summary"] = {
            "n_messages": len(kwargs.get("messages") or []),
            "has_tools": bool(kwargs.get("tools")),
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens") or kwargs.get("max_completion_tokens"),
        }
        # Optional full-payload dump for the most-thorough analysis use case.
        # Toggled via env var to avoid huge files in the common case.
        if os.environ.get("STEP_BUDGET_LOG_FULL_PAYLOADS") == "1":
            try:
                rec["full_request"] = {
                    k: _json_safe(v) for k, v in kwargs.items()
                    if k not in ("api_key",)  # don't write secrets
                }
                rec["full_response"] = _json_safe(response_obj)
            except Exception as e:  # pragma: no cover — defensive
                rec["full_payload_error"] = repr(e)

        out_path = self.calls_dir / f"{rid}.jsonl"
        with out_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


def _estimate_call_cost(model: str, usage: dict) -> float:
    """Compute USD cost from usage + _PRICING. Returns 0.0 for unknown models."""
    p = _PRICING.get(model) or _PRICING.get(model.split("/")[-1])
    if not p:
        return 0.0
    prompt = usage.get("prompt_tokens", 0) or 0
    cached = usage.get("cached_prompt_tokens", 0) or 0
    completion = usage.get("completion_tokens", 0) or 0
    nonCached = max(0, prompt - cached)
    cost = nonCached * p["prompt"] / 1e6
    cost += completion * p["completion"] / 1e6
    if p.get("cached_prompt") is not None:
        cost += cached * p["cached_prompt"] / 1e6
    else:
        cost += cached * p["prompt"] / 1e6
    return cost


def _json_safe(o: Any) -> Any:
    """Coerce arbitrary objects into JSON-serializable form."""
    try:
        json.dumps(o)
        return o
    except TypeError:
        pass
    if hasattr(o, "model_dump"):
        try:
            return o.model_dump(mode="json")
        except Exception:
            pass
    if hasattr(o, "__dict__"):
        try:
            return {k: _json_safe(v) for k, v in vars(o).items() if not k.startswith("_")}
        except Exception:
            pass
    if isinstance(o, (list, tuple)):
        return [_json_safe(x) for x in o]
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    return str(o)


def _student_llm_args(port: int) -> dict:
    max_tokens = int(os.environ.get("STUDENT_MAX_TOKENS", str(DEFAULT_STUDENT_MAX_TOKENS)))
    return {
        "api_base": f"http://localhost:{port}/v1",
        "api_key": "EMPTY",
        "max_tokens": max_tokens,
    }


# --------------------------------------------------------------------------- #
# vLLM lifecycle                                                              #
# --------------------------------------------------------------------------- #

def spawn_vllm(checkpoint: Path, port: int, gpu: int, tp: int,
               vllm_serve_sh: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["TP_SIZE"] = str(tp)
    log_root = Path(env.get("BUNDLE_ROOT", checkpoint.parent)) / "vllm_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    label = checkpoint.parent.name if checkpoint.name.startswith("checkpoint") else checkpoint.name
    label = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)
    log_path = log_root / f"{label}.port{port}.gpu{gpu}.log"
    print(f"[harness] vLLM logs: {log_path}", flush=True)
    log_fh = log_path.open("ab", buffering=0)
    log_fh.write(f"\n\n===== vLLM start {datetime.datetime.now(datetime.timezone.utc).isoformat()} port={port} gpu={gpu} tp={tp} checkpoint={checkpoint} =====\n".encode())
    proc = subprocess.Popen(
        ["bash", str(vllm_serve_sh), str(checkpoint), str(port), str(gpu)],
        env=env,
        stdout=log_fh, stderr=subprocess.STDOUT,
    )
    log_fh.close()
    return proc


def wait_for_vllm(port: int, timeout_s: int = 900, launcher: subprocess.Popen | None = None) -> bool:
    """Poll /v1/models until ready. Mirrors harness.py:wait_for_vllm."""
    import urllib.request
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        if launcher is not None:
            rc = launcher.poll()
            if rc is not None and rc != 0:
                return False
        time.sleep(5)
    return False


def stop_vllm(checkpoint: Path) -> None:
    pid_file = checkpoint / "vllm_serve.pid"
    if not pid_file.exists():
        return
    text = pid_file.read_text().split()
    for tok in text:
        if tok.startswith("PID="):
            try:
                pid = int(tok.removeprefix("PID="))
                os.kill(pid, signal.SIGTERM)
                time.sleep(2)
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass
    with contextlib.suppress(OSError):
        pid_file.unlink()


# --------------------------------------------------------------------------- #
# CREDIT EXHAUSTION DETECTION                                                 #
# --------------------------------------------------------------------------- #

_CREDIT_KEYWORDS = (
    "402", "insufficient", "balance", "credit", "out of funds",
    "quota exceeded", "payment required", "billing",
)


def _is_credit_exhausted(exc: BaseException) -> bool:
    """Detect "credits exhausted" errors emitted by CommonStack via litellm.

    CommonStack's /v1/chat/completions returns HTTP 402 with a JSON body
    that includes phrases like "insufficient balance" when the org credit
    is depleted. litellm wraps this into BadRequestError / AuthenticationError
    depending on the exact error class. We match conservatively on the
    string form to handle multiple framings.
    """
    msg = (str(exc) or "").lower()
    if any(kw in msg for kw in _CREDIT_KEYWORDS):
        return True
    # Some litellm exceptions expose a `.status_code` attr.
    sc = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if sc in (402,):
        return True
    return False


# --------------------------------------------------------------------------- #
# ROLLOUT EXECUTION                                                           #
# --------------------------------------------------------------------------- #

def _drain_litellm_callbacks(max_wait_s: float = 2.0, idle_polls: int = 3, poll_s: float = 0.05) -> None:
    """Wait for litellm's success/failure callbacks to fully drain.

    litellm dispatches CustomLogger callbacks (incl. our CallRecorder) onto a
    100-worker ThreadPoolExecutor at
    `litellm.litellm_core_utils.litellm_logging.executor`. The dispatch is
    fire-and-forget — `litellm.completion(...)` returns BEFORE the callback
    runs. If we snapshot the recorder's counters immediately, the rollout's
    LAST call is systematically missing.

    We drain by polling the executor's work queue (must be empty for at least
    `idle_polls` consecutive polls), bounded by `max_wait_s`. Returns silently
    if the executor module isn't where we expect it (forward-compat: the fix
    becomes a no-op, the bug returns but doesn't break anything else).
    """
    try:
        from litellm.litellm_core_utils.litellm_logging import executor
    except Exception:
        time.sleep(0.2)  # best-effort fallback
        return
    deadline = time.time() + max_wait_s
    consecutive_idle = 0
    while time.time() < deadline:
        qsize = getattr(executor, "_work_queue", None)
        qsize = qsize.qsize() if qsize is not None else 0
        if qsize == 0:
            consecutive_idle += 1
            if consecutive_idle >= idle_polls:
                return
        else:
            consecutive_idle = 0
        time.sleep(poll_s)


class CreditExhausted(Exception):
    """Raised when a CommonStack call fails due to credit exhaustion."""


def _rollout_filename(method: str, dom: str, task_id: str, seed: int, step_idx: int | None) -> str:
    if method == "B":
        assert step_idx is not None
        return f"{dom}__{task_id}__seed{seed}__methodB_step{step_idx}.json"
    return f"{dom}__{task_id}__seed{seed}__methodA.json"


def _rollout_id(method: str, dom: str, task_id: str, seed: int, step_idx: int | None) -> str:
    return _rollout_filename(method, dom, task_id, seed, step_idx).removesuffix(".json")


def _is_complete(p: Path) -> bool:
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        # Robust quick check: tail-check the file for "_marker": "COMPLETE"
        data = json.loads(p.read_text())
        return data.get("_marker") == "COMPLETE"
    except Exception:
        return False


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, default=str))
    os.replace(tmp, path)


def _evaluation_type_for(reward_basis: str) -> Any:
    from tau2.evaluator.evaluator import EvaluationType
    if "NL_ASSERTION" in (reward_basis or ""):
        return EvaluationType.ALL_WITH_NL_ASSERTIONS
    return EvaluationType.ALL


# Action-check / depth diagnostic ---------------------------------------------

def _gold_actions_from_task(task_obj: Any) -> list[dict]:
    """Extract assistant-side gold actions from a tau2 Task — accepting either
    a dict (from adapter.load_tasks) or a Pydantic Task instance.

    Per tau2.data_model.tasks.Action, each entry has:
      - name: tool name
      - arguments: dict of args
      - requestor: "assistant" or "user"  (only "assistant" counts for us)
      - compare_args: optional list[str] — when set, ONLY these arg keys are
        required to match. Other arg differences are ignored.
    """
    crit = None
    if isinstance(task_obj, dict):
        crit = task_obj.get("evaluation_criteria")
    else:
        crit = getattr(task_obj, "evaluation_criteria", None)
    if crit is None:
        return []

    raw_actions = (crit.get("actions") if isinstance(crit, dict)
                   else getattr(crit, "actions", None)) or []
    out: list[dict] = []
    for a in raw_actions:
        if isinstance(a, dict):
            requestor = a.get("requestor", "assistant")
            if requestor != "assistant":
                continue
            out.append({
                "name": a.get("name"),
                "arguments": a.get("arguments") or {},
                "compare_args": a.get("compare_args"),
            })
        else:
            requestor = getattr(a, "requestor", "assistant")
            if requestor != "assistant":
                continue
            args = getattr(a, "arguments", None) or {}
            if hasattr(args, "model_dump"):
                args = args.model_dump()
            out.append({
                "name": getattr(a, "name", None),
                "arguments": args,
                "compare_args": getattr(a, "compare_args", None),
            })
    return out


def _depth_in_gold_actions(sim_messages: list, task_obj: Any) -> dict:
    """Two complementary per-step diagnostics on the student's tool-call trajectory:

    Returns a dict with:
      gold_actions_total:           |task.evaluation_criteria.actions| (assistant-side)
      gold_actions_executed:        # gold actions that appeared ANYWHERE in the
                                    trajectory (matches tau2's ACTION grader
                                    semantics — order doesn't matter, extra
                                    non-gold tool calls don't penalize).
      consecutive_aligned_depth:    # leading tool calls (in trajectory order)
                                    that matched some still-unmatched gold action
                                    before the first non-matching call. Stricter
                                    drift detector.

    "Lenient" gold_actions_executed is the headline diagnostic for failed
    rollouts: "the student got 3 of 5 gold actions done before failing" — this
    is what improves monotonically with training.

    "Strict" consecutive_aligned_depth tells us whether the student stayed on
    the baseline's exact path or wandered (useful when the student passes —
    higher depth = closer to the gold trajectory).
    """
    gold_full = _gold_actions_from_task(task_obj)
    total = len(gold_full)
    if total == 0:
        return {
            "gold_actions_total": 0,
            "gold_actions_executed": 0,
            "consecutive_aligned_depth": 0,
        }

    # Lenient pass: how many gold actions appeared anywhere?
    lenient_remaining = list(gold_full)
    lenient_count = 0
    # Strict pass: leading-prefix consecutive matches
    strict_remaining = list(gold_full)
    strict_count = 0
    strict_broken = False

    for m in sim_messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role != "assistant":
            continue
        msg_requestor = (m.get("requestor", "assistant") if isinstance(m, dict)
                         else getattr(m, "requestor", "assistant"))
        if msg_requestor != "assistant":
            continue
        tcs = m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)
        if not tcs:
            continue
        for tc in tcs:
            tc_name, tc_args = _extract_tool_call(tc)
            # Lenient pass
            for i, gr in enumerate(lenient_remaining):
                if gr["name"] == tc_name and _args_match(
                    gr.get("arguments") or {}, tc_args,
                    compare_keys=gr.get("compare_args"),
                ):
                    del lenient_remaining[i]
                    lenient_count += 1
                    break
            # Strict pass (skipped once a non-match has occurred)
            if not strict_broken:
                matched = None
                for i, gr in enumerate(strict_remaining):
                    if gr["name"] == tc_name and _args_match(
                        gr.get("arguments") or {}, tc_args,
                        compare_keys=gr.get("compare_args"),
                    ):
                        matched = i
                        break
                if matched is None:
                    strict_broken = True
                else:
                    del strict_remaining[matched]
                    strict_count += 1
    return {
        "gold_actions_total": total,
        "gold_actions_executed": lenient_count,
        "consecutive_aligned_depth": strict_count,
    }


def _extract_tool_call(tc: Any) -> tuple[str | None, Any]:
    """Pull (name, arguments) from a tool_call in either OpenAI-tool-call or
    tau2-internal-tool-call shape (with .function or top-level fields)."""
    if isinstance(tc, dict):
        fn = tc.get("function") or {}
        name = fn.get("name") or tc.get("name")
        args = fn.get("arguments") if "arguments" in fn else tc.get("arguments")
    else:
        fn = getattr(tc, "function", None)
        if fn is not None:
            name = getattr(fn, "name", None)
            args = getattr(fn, "arguments", None)
        else:
            name = getattr(tc, "name", None)
            args = getattr(tc, "arguments", None)
    return name, args


def _args_match(gold_args: Any, student_args: Any, *,
                compare_keys: list[str] | None = None) -> bool:
    """Compare action arguments.

    - Accepts dict, JSON string, or None for either side.
    - When `compare_keys` is set (from Action.compare_args), only those keys
      need to match. Student may pass extra args (e.g., a "note" field) without
      penalty.
    """
    def _norm(x):
        if isinstance(x, str):
            try:
                return json.loads(x)
            except json.JSONDecodeError:
                return {"__str__": x}
        if x is None:
            return {}
        if hasattr(x, "model_dump"):
            return x.model_dump()
        return dict(x) if not isinstance(x, dict) else x

    g = _norm(gold_args)
    s = _norm(student_args)
    if compare_keys:
        for k in compare_keys:
            if k not in s:
                return False
            if g.get(k) != s.get(k):
                return False
        return True
    return g == s


def _first_diverged_step(baseline_msgs: list[dict], student_sim_messages: list) -> int | None:
    """At which AGENT step (0-indexed) does the student first emit a different
    tool name than baseline did at the same position? Returns None if every
    agent step the student took matched the baseline."""
    def _msg_role(m):
        return m.get("role") if isinstance(m, dict) else getattr(m, "role", None)

    def _msg_tool_name(m):
        tcs = m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)
        if not tcs:
            return None
        tc = tcs[0]
        if isinstance(tc, dict):
            return (tc.get("function") or {}).get("name") or tc.get("name")
        fn = getattr(tc, "function", None)
        return (getattr(fn, "name", None) if fn is not None
                else getattr(tc, "name", None))

    def _msg_requestor(m):
        if isinstance(m, dict):
            return m.get("requestor", "assistant")
        return getattr(m, "requestor", "assistant")

    baseline_agent_msgs = [m for m in baseline_msgs
                           if _msg_role(m) == "assistant"
                           and _msg_requestor(m) == "assistant"]
    student_agent_msgs = [m for m in student_sim_messages
                          if _msg_role(m) == "assistant"
                          and _msg_requestor(m) == "assistant"]
    for i, (bm, sm) in enumerate(zip(baseline_agent_msgs, student_agent_msgs)):
        if _msg_tool_name(bm) != _msg_tool_name(sm):
            return i
    return None


# Method A: full E2E rollout --------------------------------------------------

def run_method_a(
    *, entry: dict, port: int, output_dir: Path, adapter, recorder: CallRecorder,
    cs_url: str, cs_key: str, vendor_root: Path,
) -> dict:
    """One E2E rollout: student handles all B steps from scratch."""
    from core.schemas.artifacts import LLMSpec, RunTaskConfig

    out_file = output_dir / "raw_eval" / _rollout_filename(
        "A", entry["domain"], entry["task_id"], entry["seed"], None)
    if _is_complete(out_file):
        return {"status": "skipped_already_done", "path": str(out_file)}

    rid = _rollout_id("A", entry["domain"], entry["task_id"], entry["seed"], None)
    recorder.reset_cost(rid)
    recorder.set_rollout(rid)

    task_dict = _load_tau2_task(adapter, entry["domain"], entry.get("tau2_task_id", entry["task_id"]))
    # Match pipeline/runner.py:461-494 routing convention for CommonStack:
    # cs_args includes custom_llm_provider="openai", and CS-routed model
    # strings are double-prefixed (e.g. "openai/openai/gpt-5.2") so litellm
    # strips ONE "openai/" provider prefix and ships the wire model that CS
    # actually hosts (which has its own "openai/" prefix). See
    # adapter.py:72-76 for the same convention on the NL judge.
    cs_args = {"api_base": cs_url, "api_key": cs_key, "custom_llm_provider": "openai"}
    config = RunTaskConfig(
        agent=LLMSpec(
            model=f"openai/{SERVED_MODEL_NAME}",
            args=_student_llm_args(port),
        ),
        user=LLMSpec(
            model="openai/openai/gpt-5.2",
            args=cs_args,
        ),
        seed=entry["seed"],
        max_steps=entry["B"],
        max_errors=entry["max_errors"],
    )

    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()
    try:
        result = adapter.run_task(task_dict, config, domain=entry["domain"])
    except Exception as e:
        if _is_credit_exhausted(e):
            raise CreditExhausted(str(e)) from e
        # Non-credit failure: drain callbacks then log + propagate as task-failed.
        _drain_litellm_callbacks()
        return _write_failed_rollout(
            out_file=out_file, rid=rid, entry=entry, method="A",
            exc=e, started_at=started_at, elapsed=time.time() - t0,
            recorder=recorder,
        )

    elapsed = time.time() - t0
    # LiteLLM dispatches success/failure callbacks via a ThreadPoolExecutor.
    # Drain pending callbacks so the recorder's counters reflect every call
    # made during the rollout BEFORE we snapshot them into the payload.
    _drain_litellm_callbacks()
    payload = _build_rollout_payload(
        rid=rid, method="A", step_idx=None, entry=entry,
        result=result, started_at=started_at, elapsed=elapsed,
        recorder=recorder, baseline_msgs=None, task_dict=task_dict,
    )
    _atomic_write_json(out_file, payload)
    return {"status": "ok", "passed": payload["passed"], "path": str(out_file)}


# Method B: per-step substitution rollouts -----------------------------------

def run_method_b_for_step(
    *, entry: dict, step_idx: int, port: int, output_dir: Path,
    adapter, recorder: CallRecorder, cs_url: str, cs_key: str,
    baseline_msgs: list[dict], vendor_root: Path,
) -> dict:
    """One substitution rollout: student does step_idx, baseline does the rest."""
    from core.schemas.artifacts import LLMSpec, RunTaskConfig

    out_file = output_dir / "raw_eval" / _rollout_filename(
        "B", entry["domain"], entry["task_id"], entry["seed"], step_idx)
    if _is_complete(out_file):
        return {"status": "skipped_already_done", "path": str(out_file)}

    rid = _rollout_id("B", entry["domain"], entry["task_id"], entry["seed"], step_idx)
    recorder.reset_cost(rid)
    recorder.set_rollout(rid)

    task_dict = _load_tau2_task(adapter, entry["domain"], entry.get("tau2_task_id", entry["task_id"]))
    # Match pipeline/runner.py routing convention for CommonStack — see notes
    # in run_method_a above.
    cs_args = {"api_base": cs_url, "api_key": cs_key, "custom_llm_provider": "openai"}
    config = RunTaskConfig(
        # The "agent" in this config is the BASELINE that handles the tail.
        # The harness then substitutes the student at step_idx via the
        # adapter's run_task_with_substitution kwargs (sub=LLMSpec(student)).
        agent=LLMSpec(
            model=f"openai/{entry['baseline_model']}",
            args=cs_args,
        ),
        user=LLMSpec(
            model="openai/openai/gpt-5.2",
            args=cs_args,
        ),
        seed=entry["seed"],
        # Drift buffer above B. Baseline finished in B agent steps; the
        # substitution at step_idx can knock the strong baseline-tail onto a
        # slightly different recovery path that needs several extra turns. +10
        # gives that headroom while still capping at a defined budget so a weak
        # student wandering forever still terminates. The tail uses the strong
        # baseline, so a generous buffer does NOT let a weak student "win" by
        # taking lots of extra steps — it just prevents max_steps from killing
        # the rollout for unrelated reasons. Method A is intentionally NOT
        # buffered (it's the autonomy benchmark: student finishes in B or not).
        max_steps=entry["B"] + 10,
        max_errors=entry["max_errors"],
    )
    sub = LLMSpec(
        model=f"openai/{SERVED_MODEL_NAME}",
        args=_student_llm_args(port),
    )

    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()
    try:
        result = adapter.run_task_with_substitution(
            task_dict, config,
            baseline_messages=baseline_msgs,
            step_idx=step_idx,
            sub=sub,
            domain=entry["domain"],
        )
    except Exception as e:
        if _is_credit_exhausted(e):
            raise CreditExhausted(str(e)) from e
        _drain_litellm_callbacks()
        return _write_failed_rollout(
            out_file=out_file, rid=rid, entry=entry, method="B",
            exc=e, started_at=started_at, elapsed=time.time() - t0,
            recorder=recorder, step_idx=step_idx,
        )

    elapsed = time.time() - t0
    _drain_litellm_callbacks()
    payload = _build_rollout_payload(
        rid=rid, method="B", step_idx=step_idx, entry=entry,
        result=result, started_at=started_at, elapsed=elapsed,
        recorder=recorder, baseline_msgs=baseline_msgs, task_dict=task_dict,
    )
    _atomic_write_json(out_file, payload)
    return {"status": "ok", "passed": payload["passed"], "path": str(out_file)}


# --------------------------------------------------------------------------- #
# Rollout payload + failure helpers                                           #
# --------------------------------------------------------------------------- #

def _build_rollout_payload(
    *, rid: str, method: str, step_idx: int | None, entry: dict,
    result: Any, started_at: str, elapsed: float, recorder: CallRecorder,
    baseline_msgs: list[dict] | None, task_dict: dict,
) -> dict:
    """Assemble the final rollout JSON, including the full tau2 sim dump.

    `task_dict` is the raw task entry from adapter.load_tasks(domain) — used
    to access evaluation_criteria.actions for per-step diagnostics, since
    tau2's SimulationRun does NOT carry the Task reference (verified against
    tau2.data_model.simulation.SimulationRun fields at SHA 17e07b1d).
    """
    raw_sim = getattr(result, "raw_simulation", None)
    sim_messages = getattr(raw_sim, "messages", None) or []
    reward_info = getattr(raw_sim, "reward_info", None)
    reward = float(getattr(result, "reward", 0.0) or 0.0)
    # Use reward >= 0.5 threshold — matches the existing end-to-end harness's
    # convention (stage2_ship/code/training/eval/harness.py:175). The adapter
    # internally uses passed = reward > 0.0 (any reward → passed), but that's
    # too lenient: tau2 returns partial rewards (e.g. 0.33) when only the
    # ACTION component passed but COMMUNICATE/NL_ASSERTIONS failed. For
    # eval-grade pass/fail, we need ≥0.5.
    passed = reward >= 0.5

    # Per-step diagnostics — compute from task_dict (which we have at call site)
    diagnostics: dict[str, Any] = {}
    try:
        depth_info = _depth_in_gold_actions(sim_messages, task_dict)
        diagnostics.update(depth_info)
        # Keep legacy `depth_in_gold_actions` key (= lenient gold_actions_executed)
        # so summarize_step_budget.py reads the headline diagnostic without
        # having to know about the new struct.
        diagnostics["depth_in_gold_actions"] = depth_info["gold_actions_executed"]
    except Exception as e:
        diagnostics["depth_in_gold_actions_error"] = repr(e)
        diagnostics["depth_in_gold_actions"] = None

    if baseline_msgs is not None:
        try:
            diagnostics["first_diverged_step"] = _first_diverged_step(baseline_msgs, sim_messages)
        except Exception as e:
            diagnostics["first_diverged_step_error"] = repr(e)

    # Termination diagnostics
    term = str(getattr(result, "termination_reason", "") or "")
    diagnostics["termination_reason"] = term
    if "USER_STOP" in term and reward < 0.5:
        # Last agent step (assistant-side only) before USER_STOP fired.
        agent_step = 0
        for m in sim_messages:
            mrole = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            if mrole != "assistant":
                continue
            mreq = (m.get("requestor", "assistant") if isinstance(m, dict)
                    else getattr(m, "requestor", "assistant"))
            if mreq != "assistant":
                continue
            agent_step += 1
        diagnostics["step_of_user_simulator_giveup"] = agent_step

    # Reward-info breakdown
    reward_info_dump: Any = None
    if reward_info is not None:
        if hasattr(reward_info, "model_dump"):
            try:
                reward_info_dump = reward_info.model_dump(mode="json")
            except Exception:
                reward_info_dump = str(reward_info)
        else:
            reward_info_dump = str(reward_info)

    # Token + cost totals from recorder
    tokens = recorder.tokens_so_far(rid)
    cost_total = recorder.cost_so_far(rid)

    # Full tau2 simulation dump
    sim_dump: Any = None
    if raw_sim is not None:
        sim_dump = _json_safe(raw_sim)

    return {
        "_marker": "COMPLETE",
        "rollout_id": rid,
        "method": method,
        "step_idx": step_idx,
        "target_task": {
            "domain": entry["domain"],
            "task_id": entry["task_id"],
            "seed": entry["seed"],
            "budget_B": entry["B"],
            "max_errors": entry["max_errors"],
            "rep_rate_pinned": entry["rep_rate_pinned"],
            "reward_basis": entry["reward_basis"],
            "tau2_repo_sha": entry["tau2_repo_sha"],
        },
        "baseline_metadata": {
            "model": entry["baseline_model"],
            "passed": True,                       # only chosen if baseline passed
            "reward": entry["baseline_reward"],
            "termination_reason": entry["baseline_termination_reason"],
            "simulator_model_recorded": entry.get("simulator_model"),
            "path": entry["baseline_path"],
        },
        "student_outcome": {
            "passed": passed,
            "reward": reward,
            "termination_reason": term,
            # Match B (baseline) counting: assistant-side only. Excludes
            # dual-control user-side tool calls (telecom).
            "agent_steps_in_rollout": sum(
                1 for m in sim_messages
                if ((m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "assistant")
                and ((m.get("requestor", "assistant") if isinstance(m, dict)
                      else getattr(m, "requestor", "assistant")) == "assistant")
            ),
            "wall_seconds": elapsed,
        },
        "diagnostics": diagnostics,
        "reward_info": reward_info_dump,
        "cost_usd": {
            "total_estimate": cost_total,
            "n_llm_calls": recorder._call_counters.get(rid, 0),
        },
        "tokens": tokens,
        "started_at": started_at,
        "passed": passed,
        "tau2_simulation": sim_dump,
    }


def _write_failed_rollout(
    *, out_file: Path, rid: str, entry: dict, method: str, exc: Exception,
    started_at: str, elapsed: float, recorder: CallRecorder, step_idx: int | None = None,
) -> dict:
    """Record a task failure (non-credit) so we don't retry, and continue."""
    payload = {
        "_marker": "COMPLETE",
        "rollout_id": rid,
        "method": method,
        "step_idx": step_idx,
        "target_task": {
            "domain": entry["domain"],
            "task_id": entry["task_id"],
            "seed": entry["seed"],
            "budget_B": entry["B"],
        },
        "student_outcome": {
            "passed": False,
            "reward": 0.0,
            "error_type": type(exc).__name__,
            "error_msg": str(exc)[:1000],
            "wall_seconds": elapsed,
        },
        "passed": False,
        "cost_usd": {
            "total_estimate": recorder.cost_so_far(rid),
            "n_llm_calls": recorder._call_counters.get(rid, 0),
        },
        "started_at": started_at,
        "exception_traceback": "".join(traceback.format_exception_only(type(exc), exc)),
    }
    _atomic_write_json(out_file, payload)
    return {"status": "error_recorded", "passed": False, "path": str(out_file)}


# --------------------------------------------------------------------------- #
# Tau2 task loading                                                           #
# --------------------------------------------------------------------------- #

def _load_tau2_task(adapter: Any, domain: str, task_id: str) -> dict:
    """Find the tau2 task entry by id within a domain's tasks.json."""
    tasks = adapter.load_tasks(domain)
    for t in tasks:
        if str(t.get("id")) == str(task_id):
            return t
    raise ValueError(f"Task {task_id} not found in domain {domain}")


# --------------------------------------------------------------------------- #
# Progress / Stop-Reason files                                                #
# --------------------------------------------------------------------------- #

def _write_progress(output_dir: Path, **kv) -> None:
    p = output_dir / "progress.json"
    payload = {}
    if p.exists():
        try:
            payload = json.loads(p.read_text())
        except Exception:
            payload = {}
    payload.update(kv)
    payload["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _atomic_write_json(p, payload)


def _write_stop_reason(output_dir: Path, reason: str, **kv) -> None:
    payload = {"reason": reason, "stopped_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), **kv}
    _atomic_write_json(output_dir / "STOP_REASON.json", payload)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Step-budget eval harness for one target.")
    ap.add_argument("--target", required=True,
                    help="Short name (e.g. 05_qwen3_5_4b_273, base_Qwen3.5-4B). Used in output paths.")
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="Local directory vLLM should serve (trained ckpt or pre-downloaded base model).")
    ap.add_argument("--bundle-root", required=True, type=Path)
    ap.add_argument("--baselines-root", required=True, type=Path,
                    help="Directory containing eval_baselines/<dom>/tasks/<tid>/seed_<s>/phase0_baseline.json")
    ap.add_argument("--eval-tasks", required=True, type=Path,
                    help="Path to data_processed/stage2_v1/eval_tasks.jsonl")
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Per-target output dir, e.g. step_budget_outputs/<target>/")
    ap.add_argument("--port", required=True, type=int)
    ap.add_argument("--gpu", required=True, type=int)
    ap.add_argument("--tp", type=int, default=1, help="Tensor-parallel size for vLLM.")
    ap.add_argument("--seed-policy", choices=("primary", "secondary"), default="primary")
    ap.add_argument("--method", choices=("both", "A", "B"), default="both",
                    help="Which methods to run. 'both' = Method B + Method A supplement.")
    ap.add_argument("--vllm-serve-sh", type=Path, default=None,
                    help="Path to vllm_serve.sh. Defaults to $BUNDLE_ROOT/code/training/eval/vllm_serve.sh")
    ap.add_argument("--no-vllm", action="store_true",
                    help="DEBUG: assume vLLM is already running on the given port.")
    args = ap.parse_args(argv)

    # --- preflight env vars
    cs_url = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    cs_key = os.environ.get("OPENAI_API_KEY")
    if not cs_url or not cs_key:
        print("ERROR: OPENAI_API_BASE / OPENAI_BASE_URL and OPENAI_API_KEY must be set "
              "(point at CommonStack).", file=sys.stderr)
        return EXIT_ARGS_FAIL

    output_dir: Path = args.output_dir
    (output_dir / "raw_eval").mkdir(parents=True, exist_ok=True)
    (output_dir / "llm_calls").mkdir(parents=True, exist_ok=True)

    vllm_serve_sh = args.vllm_serve_sh or (args.bundle_root / "code" / "training" / "eval" / "vllm_serve.sh")

    # Stale STOP_REASON.json from a prior interrupted run — archive it so the
    # next harness call doesn't trip over it. Archived as a sibling file.
    stop_marker = output_dir / "STOP_REASON.json"
    if stop_marker.exists():
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
        archive = output_dir / f"STOP_REASON.{ts}.json.archived"
        os.replace(stop_marker, archive)
        print(f"[harness] archived stale STOP_REASON → {archive.name}")

    # --- Apply judge patch
    from training.eval._judge_patch import apply_judge_patch
    apply_judge_patch(cs_url, cs_key)

    # --- Install litellm capture
    recorder = CallRecorder(output_dir / "llm_calls")
    recorder.install()

    # --- Build adapter (with judge patch ALREADY applied → block adapter's
    #     own monkey-patch by flagging it as done)
    from adapters.tau2_bench.adapter import Tau2BenchAdapter
    vendor_root = args.bundle_root / "code" / "vendor" / "tau2-bench"
    adapter = Tau2BenchAdapter(vendor_root=vendor_root)
    adapter._nl_judge_routed = True  # prevent adapter from re-routing the judge

    # --- Build the chosen set
    try:
        chosen = compute_chosen_set(
            eval_tasks_jsonl=args.eval_tasks,
            baselines_root=args.baselines_root,
            seed_policy=args.seed_policy,
        )
    except Exception as e:
        print(f"ERROR: failed to build chosen set: {e}", file=sys.stderr)
        traceback.print_exc()
        return EXIT_ARGS_FAIL
    if not chosen:
        print("ERROR: chosen set is empty (no passing baselines at the requested seed_policy)",
              file=sys.stderr)
        return EXIT_NO_BASELINES

    n_method_b = sum(c["B"] for c in chosen) if args.method in ("both", "B") else 0
    n_method_a = len(chosen) if args.method in ("both", "A") else 0
    print(f"[harness] target={args.target} chosen={len(chosen)} tasks, "
          f"Method B rollouts={n_method_b}, Method A rollouts={n_method_a}")

    _write_progress(output_dir,
                    target=args.target,
                    seed_policy=args.seed_policy,
                    n_chosen=len(chosen),
                    method_b_total=n_method_b,
                    method_a_total=n_method_a,
                    started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())

    # --- Spin up vLLM
    vllm_proc = None
    try:
        if not args.no_vllm:
            print(f"[harness] launching vLLM on port {args.port}, gpu {args.gpu}, TP={args.tp} ...")
            vllm_proc = spawn_vllm(args.checkpoint, args.port, args.gpu, args.tp, vllm_serve_sh)
            if not wait_for_vllm(args.port, launcher=vllm_proc):
                print(f"ERROR: vLLM failed to come up on port {args.port}", file=sys.stderr)
                _write_stop_reason(output_dir, "vllm_failed_to_start")
                return EXIT_VLLM_FAIL
            print(f"[harness] vLLM ready on port {args.port}")

        # --- Loop over chosen tasks
        n_done = 0
        n_pass = 0
        n_skipped = 0
        cumulative_cost = 0.0
        try:
            for entry in chosen:
                # Load baseline messages ONCE per task (used by every step_idx)
                baseline_msgs = None
                if args.method in ("both", "B"):
                    baseline_msgs = json.loads(Path(entry["baseline_path"]).read_text()).get("raw_messages") or []

                # Method B: B rollouts (one per step_idx)
                if args.method in ("both", "B"):
                    for step_idx in range(entry["B"]):
                        res = run_method_b_for_step(
                            entry=entry, step_idx=step_idx, port=args.port,
                            output_dir=output_dir, adapter=adapter, recorder=recorder,
                            cs_url=cs_url, cs_key=cs_key,
                            baseline_msgs=baseline_msgs,
                            vendor_root=vendor_root,
                        )
                        if res["status"] == "skipped_already_done":
                            n_skipped += 1
                        else:
                            n_done += 1
                            if res.get("passed"):
                                n_pass += 1
                        cumulative_cost += recorder.cost_so_far(
                            _rollout_id("B", entry["domain"], entry["task_id"], entry["seed"], step_idx)
                        )
                        _write_progress(output_dir,
                                        last_completed_rollout=_rollout_id(
                                            "B", entry["domain"], entry["task_id"],
                                            entry["seed"], step_idx),
                                        n_rollouts_done=n_done,
                                        n_rollouts_skipped=n_skipped,
                                        n_rollouts_passed=n_pass,
                                        cumulative_cost_usd=cumulative_cost)

                # Method A: 1 E2E rollout
                if args.method in ("both", "A"):
                    res = run_method_a(
                        entry=entry, port=args.port, output_dir=output_dir,
                        adapter=adapter, recorder=recorder,
                        cs_url=cs_url, cs_key=cs_key, vendor_root=vendor_root,
                    )
                    if res["status"] == "skipped_already_done":
                        n_skipped += 1
                    else:
                        n_done += 1
                        if res.get("passed"):
                            n_pass += 1
                    cumulative_cost += recorder.cost_so_far(
                        _rollout_id("A", entry["domain"], entry["task_id"], entry["seed"], None)
                    )
                    _write_progress(output_dir,
                                    last_completed_rollout=_rollout_id(
                                        "A", entry["domain"], entry["task_id"], entry["seed"], None),
                                    n_rollouts_done=n_done,
                                    n_rollouts_skipped=n_skipped,
                                    n_rollouts_passed=n_pass,
                                    cumulative_cost_usd=cumulative_cost)

        except CreditExhausted as e:
            print(f"\n[harness] CREDIT EXHAUSTED — stopping cleanly: {e}", file=sys.stderr)
            _write_stop_reason(output_dir, "credit_exhausted",
                               last_error=str(e)[:500],
                               n_rollouts_done=n_done,
                               cumulative_cost_usd=cumulative_cost)
            return EXIT_CREDIT_EXHAUSTED

        print(f"[harness] target={args.target} done: "
              f"{n_done} new rollouts, {n_skipped} skipped (already done), "
              f"{n_pass} passed; est cost ${cumulative_cost:.2f}")
        _write_progress(output_dir,
                        finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        final_n_done=n_done,
                        final_n_skipped=n_skipped,
                        final_n_passed=n_pass,
                        final_cumulative_cost_usd=cumulative_cost)
        return EXIT_OK

    finally:
        if vllm_proc is not None and not args.no_vllm:
            stop_vllm(args.checkpoint)


if __name__ == "__main__":
    sys.exit(main())
