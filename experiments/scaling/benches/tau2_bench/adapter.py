"""tau2-bench adapter.

Wraps colleague's `experiments/tau2_stage2/code/adapters/tau2_bench/Tau2BenchAdapter`
so this scaling pipeline can drive tau2 tasks. Produces trace rows in the
schema the main-branch scripts (run_evolve / train_learnable_router /
run_e2e_ablation) already consume.

Prerequisites — run BEFORE invoking this adapter:
    cd experiments/tau2_stage2
    bash code/training/orchestration/setup_env_server.sh    # clones tau2-bench@17e07b1d under code/vendor/
    conda activate tau2-stage2

If `--mock` is passed to collect_traces, the adapter generates synthetic
deterministic trace data WITHOUT invoking tau2 (useful for smoke testing the
rest of the pipeline without GPUs / OpenAI key).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
TAU2_BUNDLE = REPO_ROOT / "experiments" / "tau2_stage2"


def _extract_signature(prompt: str) -> str:
    """Cheap signature: domain tag + first sentence + length bucket."""
    head = (prompt or "").strip().split("\n")[0][:80]
    bucket = len(prompt or "") // 200
    return f"{head}::len_bucket={bucket}"


def _completion_from_steps(steps: list) -> str:
    """Render an SFT completion from TaskRunResult.steps.

    Each StepData.response is {"content": str, "tool_calls": [...]}. We join the
    assistant content per step and render any tool calls as compact lines, so
    the small model learns the full agentic trajectory (content + tool use), not
    just a final answer. (review 2026-06-05: extract from StepData.response —
    TaskRunResult.messages is empty on live tau2.)
    """
    parts: list[str] = []
    for s in steps:
        resp = getattr(s, "response", None)
        if resp is None and isinstance(s, dict):
            resp = s.get("response")
        if not isinstance(resp, dict):
            continue
        content = resp.get("content") or ""
        if content:
            parts.append(str(content))
        for tc in resp.get("tool_calls") or []:
            if isinstance(tc, dict):
                fn = (tc.get("function") or {}) if isinstance(tc.get("function"), dict) else {}
                name = fn.get("name") or tc.get("name") or "tool"
                arguments = fn.get("arguments") or tc.get("arguments") or ""
                parts.append(f"<tool_call>{name}({arguments})</tool_call>")
    return "\n".join(p for p in parts if p)


class Adapter:
    """tau2-bench adapter."""

    def __init__(self) -> None:
        self.mock = os.environ.get("SCALING_MOCK", "0") == "1"
        self._tau2_adapter = None
        self.domain = os.environ.get("TAU2_DOMAIN", "retail")  # airline | retail | telecom
        domains = os.environ.get("TAU2_DOMAINS", "").strip()
        self.domains = [d.strip() for d in domains.split(",") if d.strip()] or [self.domain]
        if not os.environ.get("TAU2_API_BASE") and os.environ.get("OPENAI_API_BASE"):
            os.environ["TAU2_API_BASE"] = os.environ["OPENAI_API_BASE"]
        if not os.environ.get("TAU2_API_KEY") and os.environ.get("OPENAI_API_KEY"):
            os.environ["TAU2_API_KEY"] = os.environ["OPENAI_API_KEY"]
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        try:
            import litellm  # type: ignore
            litellm.drop_params = True
        except Exception:
            pass
        # User-simulator + run knobs for live tau2 (RunTaskConfig). The agent
        # model is the one under test (passed to _run_one); the user simulator
        # and litellm args (api_base/api_key/custom_llm_provider for a
        # CommonStack-style gateway) come from env so live runs are operator-
        # configurable without code edits.
        self.user_model = os.environ.get("TAU2_USER_MODEL", "openai/openai/gpt-5.2")
        self.seed = int(os.environ.get("TAU2_SEED", "0"))
        self.max_steps = int(os.environ.get("TAU2_MAX_STEPS", "100"))
        self.max_errors = int(os.environ.get("TAU2_MAX_ERRORS", "10"))
        self._domains_root: Path | None = None

    def _is_local_student_model(self, model: str) -> bool:
        served = os.environ.get("TAU2_LOCAL_SERVED_MODEL", "evol-llm-student")
        aliases = {
            served,
            f"openai/{served}",
            os.environ.get("TAU2_LOCAL_MODEL", ""),
        }
        return model in aliases

    def _llm_args(self, role: str, model: str | None = None) -> dict:
        """litellm args for agent/user from env (api_base/api_key/provider)."""
        if role == "agent" and model and self._is_local_student_model(model):
            base = os.environ.get("TAU2_LOCAL_API_BASE")
            key = os.environ.get("TAU2_LOCAL_API_KEY", "EMPTY")
            args: dict[str, Any] = {}
            if base:
                args["api_base"] = base
                args["api_key"] = key
                args["custom_llm_provider"] = os.environ.get("TAU2_LOCAL_LLM_PROVIDER", "openai")
                return args

        base = os.environ.get(f"TAU2_{role.upper()}_API_BASE") or os.environ.get("TAU2_API_BASE")
        key = os.environ.get(f"TAU2_{role.upper()}_API_KEY") or os.environ.get("TAU2_API_KEY")
        args: dict[str, Any] = {}
        if base:
            args["api_base"] = base
        if key:
            args["api_key"] = key
        if base or key:
            args["custom_llm_provider"] = os.environ.get("TAU2_LLM_PROVIDER", "openai")
        return args

    # ------------------------------------------------------------------ load
    def load_tasks(self, n: int, split: str = "train") -> list[dict]:
        if self.mock:
            rng = random.Random(42)
            tasks = []
            for domain in self.domains:
                for i in range(n):
                    tid = f"mock-{domain}-{split}-{i:04d}"
                    tasks.append({
                        "task_id": tid,
                        "original_task_id": f"mock-{split}-{i:04d}",
                        "prompt": f"[{domain}] mock task #{i}: please help the user with a {rng.choice(['refund','booking','transfer','cancellation','address change'])} request.",
                        "domain": domain,
                    })
            return tasks

        if self._tau2_adapter is None:
            self._lazy_import_tau2()
        out = []
        for domain in self.domains:
            raw = self._tau2_adapter.load_tasks(domain)
            raw = self._filter_split(raw, split, domain=domain)
            out.extend(self._normalize_task(t, domain) for t in raw[:n])
        return out

    # ----------------------------------------------------------- run pair
    def run_task_pair(
        self,
        task: dict,
        small_model: str,
        large_model: str,
        cycle: int,
        force_both: bool = False,
    ) -> dict:
        """Run small (and large) on a task.

        force_both=False (deployment / cost-control default): run small, and
        only run large if small fails. `large_*` is then a SKIP placeholder
        when small succeeded (large_skipped=True) — do NOT treat it as a real
        large outcome.

        force_both=True (closed-loop trace collection, review 2026-05-21): run
        BOTH models unconditionally so `large_success`/`large_cost`/
        `large_completion` are always REAL. Needed when a downstream policy may
        route to large on a task where small also succeeded — otherwise the
        policy outcome would be billed against a fake skip placeholder.
        """
        sig = _extract_signature(task.get("prompt", ""))
        if self.mock:
            return self._mock_run(task, small_model, large_model, sig, cycle,
                                  force_both=force_both)

        small_res = self._run_one(task, small_model)
        large_skipped = False
        if small_res["success"] and not force_both:
            large_res = {"success": True, "cost": 0.0, "skipped": True, "completion": ""}
            large_skipped = True
            decision = "probe:small→small_OK"
            final_model, final_success, final_cost = small_model, True, small_res["cost"]
        else:
            large_res = self._run_one(task, large_model)
            if small_res["success"]:
                # force_both: small already OK, large run for oracle completeness
                decision = "oracle:small_OK+large_run"
                final_model, final_success, final_cost = small_model, True, small_res["cost"]
            else:
                decision = f"probe:small_fail→large_{'OK' if large_res['success'] else 'fail'}"
                final_model, final_success, final_cost = (
                    large_model,
                    large_res["success"],
                    small_res["cost"] + large_res["cost"],
                )

        return {
            "task_id": task["task_id"],
            "original_task_id": task.get("original_task_id", task["task_id"]),
            "domain": task.get("domain", self.domain),
            "signature": sig,
            "decision": decision,
            "attempts": 1 if (small_res["success"] and not force_both) else 2,
            "attempts_count": 1 if (small_res["success"] and not force_both) else 2,
            "final_success": final_success,
            "final_model": final_model,
            "total_cost": final_cost,
            "round": cycle,
            # extras (kept for ablation / DPO):
            "small_success": small_res["success"],
            "large_success": large_res.get("success", False),
            "small_cost": small_res["cost"],
            "large_cost": large_res.get("cost", 0.0),
            "large_skipped": large_skipped,  # True => large_* is a placeholder, not real
            "prompt": task.get("prompt", ""),
            # completion text is required by traces_to_sft.py for LLM SFT
            # (review 2026-05-21: per-cycle traces must feed LLM training).
            "small_completion": small_res.get("completion", ""),
            "large_completion": large_res.get("completion", ""),
        }

    # ----------------------------------------------------------- internals
    def _lazy_import_tau2(self) -> None:
        """Import colleague's Tau2BenchAdapter from tau2_stage2 bundle."""
        if not TAU2_BUNDLE.exists():
            raise RuntimeError(
                f"tau2_stage2 bundle not found at {TAU2_BUNDLE}.\n"
                f"Either:\n"
                f"  1) Merge codex/tau2-stage2-training-eval branch first:\n"
                f"       git merge origin/codex/tau2-stage2-training-eval --no-edit\n"
                f"  2) Or set SCALING_MOCK=1 to use the mock adapter."
            )
        code_dir = TAU2_BUNDLE / "code"
        sys.path.insert(0, str(code_dir))
        try:
            from adapters.tau2_bench.adapter import Tau2BenchAdapter  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                f"Could not import Tau2BenchAdapter. Did you run "
                f"experiments/tau2_stage2/code/training/orchestration/setup_env_server.sh? "
                f"Original error: {e}"
            )
        self._patch_tau2_litellm()
        vendor_root = code_dir / "vendor" / "tau2-bench"
        self._domains_root = vendor_root / "data" / "tau2" / "domains"
        self._tau2_adapter = Tau2BenchAdapter(vendor_root=vendor_root, domain=self.domain)

    def _patch_tau2_litellm(self) -> None:
        try:
            import tau2.utils.llm_utils as llm_utils  # type: ignore
        except Exception:
            return
        if getattr(llm_utils, "_scaling_completion_patched", False):
            return
        original = llm_utils.completion
        original_cost = llm_utils.get_response_cost

        def completion_drop_unsupported(*args, **kwargs):
            kwargs.setdefault("drop_params", True)
            kwargs.pop("seed", None)
            return original(*args, **kwargs)

        def get_response_cost_or_zero(response):
            try:
                return original_cost(response)
            except Exception:
                return 0.0

        llm_utils.completion = completion_drop_unsupported
        llm_utils.get_response_cost = get_response_cost_or_zero
        llm_utils._scaling_completion_patched = True

    def _filter_split(
        self,
        tasks: list[dict[str, Any]],
        split: str,
        *,
        domain: str | None = None,
    ) -> list[dict[str, Any]]:
        """Filter tau2 domain tasks by split_tasks.json when available."""
        if self._domains_root is None:
            return tasks
        split_path = self._domains_root / (domain or self.domain) / "split_tasks.json"
        if not split_path.exists():
            return tasks
        with split_path.open() as f:
            split_map = json.load(f)
        split_key = "test" if split == "eval" else "train"
        wanted = split_map.get(split_key)
        if not wanted:
            return tasks
        wanted_ids = {str(x) for x in wanted}
        filtered = [
            t for t in tasks
            if str(t.get("id") or t.get("task_id")) in wanted_ids
        ]
        return filtered or tasks

    def _task_prompt(self, t: dict[str, Any]) -> str:
        """Extract the user-visible task prompt without leaking gold criteria."""
        for key in ("prompt", "user_message", "scenario"):
            value = t.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        scenario = t.get("user_scenario")
        if isinstance(scenario, dict):
            instructions = scenario.get("instructions")
            if isinstance(instructions, dict):
                parts = [
                    instructions.get("reason_for_call"),
                    instructions.get("known_info"),
                    instructions.get("unknown_info"),
                    instructions.get("task_instructions"),
                ]
                prompt = "\n".join(str(p).strip() for p in parts if p)
                if prompt.strip():
                    return prompt.strip()

        description = t.get("description")
        if isinstance(description, dict):
            parts = [
                description.get("purpose"),
                description.get("relevant_policies"),
                description.get("notes"),
            ]
            prompt = "\n".join(str(p).strip() for p in parts if p)
            if prompt.strip():
                return prompt.strip()
        elif isinstance(description, str) and description.strip():
            return description.strip()

        return ""

    def _normalize_task(self, t: dict[str, Any], domain: str | None = None) -> dict:
        domain = domain or self.domain
        original_task_id = str(
            t.get("task_id") or t.get("id") or hashlib.md5(str(t).encode()).hexdigest()[:12]
        )
        return {
            "task_id": f"{domain}:{original_task_id}",
            "original_task_id": original_task_id,
            "prompt": self._task_prompt(t),
            "domain": domain,
            "_raw": t,
        }

    def _run_one(self, task: dict, model: str) -> dict:
        """Single tau2 task run with a given agent model.

        Uses the colleague's real run_task signature:
            run_task(task, config: RunTaskConfig, *, domain=None) -> TaskRunResult
        (review 2026-05-21: the previous `run_task(task, student_model=...)`
        call was wrong and would fail on the live adapter.)
        """
        try:
            from core.schemas.artifacts import LLMSpec, RunTaskConfig  # type: ignore
        except ImportError as e:  # pragma: no cover - live-only path
            return {"success": False, "cost": 0.0, "completion": "",
                    "error": f"could not import RunTaskConfig/LLMSpec: {e}"}

        config = RunTaskConfig(
            agent=LLMSpec(model=model, args=self._llm_args("agent", model=model)),
            user=LLMSpec(model=self.user_model, args=self._llm_args("user")),
            seed=self.seed,
            max_steps=self.max_steps,
            max_errors=self.max_errors,
        )
        try:
            res = self._tau2_adapter.run_task(
                task["_raw"], config, domain=task.get("domain", self.domain)
            )
            # TaskRunResult: passed / agent_cost_usd / user_cost_usd / steps / messages
            cost = float(getattr(res, "agent_cost_usd", 0.0) or 0.0) + \
                float(getattr(res, "user_cost_usd", 0.0) or 0.0)
            # Completion for SFT: extract from steps[].response (StepData.response
            # is {"content": str, "tool_calls": [...]}). `messages` is the flat
            # JSON trajectory and is "Empty list when unavailable" on live tau2,
            # so steps[] is the reliable source (review 2026-06-05). messages is
            # only a fallback.
            completion = _completion_from_steps(getattr(res, "steps", None) or [])
            if not completion:
                msgs = getattr(res, "messages", None) or []
                if isinstance(msgs, list):
                    completion = "\n".join(
                        m.get("content", "") for m in msgs
                        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("content")
                    )
            return {
                "success": bool(getattr(res, "passed", False)),
                "cost": cost,
                "completion": completion,
                "raw": None,  # TaskRunResult holds a heavy raw_simulation; drop it
            }
        except Exception as e:  # noqa: BLE001 — surface as failed task, log
            print(f"[tau2_adapter] model={model} task_id={task.get('task_id')} failed: {e}", file=sys.stderr)
            return {"success": False, "cost": 0.0, "completion": "", "error": str(e)}

    def _mock_run(self, task: dict, small_model: str, large_model: str, sig: str,
                  cycle: int, force_both: bool = False) -> dict:
        """Deterministic mock for smoke tests."""
        h = int(hashlib.md5(f"{task['task_id']}|{cycle}".encode()).hexdigest(), 16)
        small_ok = (h % 10) < 6        # small succeeds ~60%
        large_ok = (h % 10) < 9        # large succeeds ~90%
        prompt = task.get("prompt", "")
        large_comp = f"[mock large completion for {task['task_id']}]" if large_ok else ""

        if small_ok and not force_both:
            # cost-control path: large skipped, large_* is a placeholder
            return {
                "task_id": task["task_id"],
                "original_task_id": task.get("original_task_id", task["task_id"]),
                "domain": task.get("domain", self.domain),
                "signature": sig,
                "decision": "probe:small→small_OK", "attempts": 1, "attempts_count": 1,
                "final_success": True, "final_model": small_model, "total_cost": 0.001,
                "round": cycle, "small_success": True, "large_success": False,
                "small_cost": 0.001, "large_cost": 0.0, "large_skipped": True,
                "prompt": prompt,
                "small_completion": f"[mock small completion for {task['task_id']}]",
                "large_completion": "",
            }
        if small_ok and force_both:
            # oracle path: small OK but large run anyway -> real large outcome
            return {
                "task_id": task["task_id"],
                "original_task_id": task.get("original_task_id", task["task_id"]),
                "domain": task.get("domain", self.domain),
                "signature": sig,
                "decision": "oracle:small_OK+large_run", "attempts": 2, "attempts_count": 2,
                "final_success": True, "final_model": small_model, "total_cost": 0.001,
                "round": cycle, "small_success": True, "large_success": large_ok,
                "small_cost": 0.001, "large_cost": 0.01, "large_skipped": False,
                "prompt": prompt,
                "small_completion": f"[mock small completion for {task['task_id']}]",
                "large_completion": large_comp,
            }
        return {
            "task_id": task["task_id"],
            "original_task_id": task.get("original_task_id", task["task_id"]),
            "domain": task.get("domain", self.domain),
            "signature": sig,
            "decision": f"probe:small_fail→large_{'OK' if large_ok else 'fail'}",
            "attempts": 2, "attempts_count": 2,
            "final_success": large_ok, "final_model": large_model,
            "total_cost": 0.011, "round": cycle,
            "small_success": False, "large_success": large_ok,
            "small_cost": 0.001, "large_cost": 0.01, "large_skipped": False,
            "prompt": prompt,
            "small_completion": "",
            "large_completion": large_comp,
        }
