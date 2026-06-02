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


class Adapter:
    """tau2-bench adapter."""

    def __init__(self) -> None:
        self.mock = os.environ.get("SCALING_MOCK", "0") == "1"
        self._tau2_adapter = None
        self.domain = os.environ.get("TAU2_DOMAIN", "retail")  # airline | retail | telecom
        # User-simulator + run knobs for live tau2 (RunTaskConfig). The agent
        # model is the one under test (passed to _run_one); the user simulator
        # and litellm args (api_base/api_key/custom_llm_provider for a
        # CommonStack-style gateway) come from env so live runs are operator-
        # configurable without code edits.
        self.user_model = os.environ.get("TAU2_USER_MODEL", "openai/openai/gpt-5.2")
        self.seed = int(os.environ.get("TAU2_SEED", "0"))
        self.max_steps = int(os.environ.get("TAU2_MAX_STEPS", "100"))
        self.max_errors = int(os.environ.get("TAU2_MAX_ERRORS", "10"))

    def _llm_args(self, role: str) -> dict:
        """litellm args for agent/user from env (api_base/api_key/provider)."""
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
            for i in range(n):
                tid = f"mock-{self.domain}-{split}-{i:04d}"
                tasks.append({
                    "task_id": tid,
                    "prompt": f"[{self.domain}] mock task #{i}: please help the user with a {rng.choice(['refund','booking','transfer','cancellation','address change'])} request.",
                    "domain": self.domain,
                })
            return tasks

        if self._tau2_adapter is None:
            self._lazy_import_tau2()
        subset = "eval" if split == "eval" else "train"
        raw = self._tau2_adapter.load_tasks(subset)
        return [self._normalize_task(t) for t in raw[:n]]

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
        vendor_root = code_dir / "vendor" / "tau2-bench"
        self._tau2_adapter = Tau2BenchAdapter(vendor_root=vendor_root, domain=self.domain)

    def _normalize_task(self, t: dict[str, Any]) -> dict:
        return {
            "task_id": str(t.get("task_id") or t.get("id") or hashlib.md5(str(t).encode()).hexdigest()[:12]),
            "prompt": t.get("prompt") or t.get("user_message") or t.get("scenario", ""),
            "domain": self.domain,
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
            agent=LLMSpec(model=model, args=self._llm_args("agent")),
            user=LLMSpec(model=self.user_model, args=self._llm_args("user")),
            seed=self.seed,
            max_steps=self.max_steps,
            max_errors=self.max_errors,
        )
        try:
            res = self._tau2_adapter.run_task(task["_raw"], config, domain=self.domain)
            # TaskRunResult: passed / agent_cost_usd / user_cost_usd / messages / steps
            cost = float(getattr(res, "agent_cost_usd", 0.0) or 0.0) + \
                float(getattr(res, "user_cost_usd", 0.0) or 0.0)
            completion = ""
            msgs = getattr(res, "messages", None) or []
            if isinstance(msgs, list):
                parts = [m.get("content", "") for m in msgs
                         if isinstance(m, dict) and m.get("role") == "assistant" and m.get("content")]
                completion = "\n".join(parts)
            if not completion:
                # fall back to the structured agent steps
                steps = getattr(res, "steps", None) or []
                parts = []
                for s in steps:
                    txt = getattr(s, "content", None) or (s.get("content") if isinstance(s, dict) else None)
                    if txt:
                        parts.append(str(txt))
                completion = "\n".join(parts)
            return {
                "success": bool(getattr(res, "passed", False)),
                "cost": cost,
                "completion": completion,
                "raw": None,  # TaskRunResult holds a heavy raw_simulation; drop it
            }
        except Exception as e:  # noqa: BLE001 — surface as failed task, log
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
                "task_id": task["task_id"], "signature": sig,
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
                "task_id": task["task_id"], "signature": sig,
                "decision": "oracle:small_OK+large_run", "attempts": 2, "attempts_count": 2,
                "final_success": True, "final_model": small_model, "total_cost": 0.001,
                "round": cycle, "small_success": True, "large_success": large_ok,
                "small_cost": 0.001, "large_cost": 0.01, "large_skipped": False,
                "prompt": prompt,
                "small_completion": f"[mock small completion for {task['task_id']}]",
                "large_completion": large_comp,
            }
        return {
            "task_id": task["task_id"], "signature": sig,
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
