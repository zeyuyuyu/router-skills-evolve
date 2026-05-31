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
    ) -> dict:
        sig = _extract_signature(task.get("prompt", ""))
        if self.mock:
            return self._mock_run(task, small_model, large_model, sig, cycle)

        small_res = self._run_one(task, small_model)
        # only run large if small fails or signature unseen (cost-control)
        if small_res["success"]:
            large_res = {"success": True, "cost": 0.0, "skipped": True}
            decision = f"probe:small→small_OK"
            final_model, final_success, final_cost = small_model, True, small_res["cost"]
        else:
            large_res = self._run_one(task, large_model)
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
            "attempts": 1 if small_res["success"] else 2,
            "attempts_count": 1 if small_res["success"] else 2,
            "final_success": final_success,
            "final_model": final_model,
            "total_cost": final_cost,
            "round": cycle,
            # extras (kept for ablation / DPO):
            "small_success": small_res["success"],
            "large_success": large_res.get("success", False),
            "small_cost": small_res["cost"],
            "large_cost": large_res.get("cost", 0.0),
            "prompt": task.get("prompt", ""),
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
        """Single tau2 task run with a given model.

        TODO(colleague): wire `Tau2BenchAdapter.run_task` properly. Current
        implementation invokes it via the run_task signature documented in
        the colleague's adapter.py. If signature differs, patch here.
        """
        try:
            res = self._tau2_adapter.run_task(task["_raw"], student_model=model)
            return {
                "success": bool(res.get("evaluation", {}).get("passed", False)),
                "cost": float(res.get("cost_total", 0.0)),
                "raw": res,
            }
        except Exception as e:  # noqa: BLE001 — surface as failed task, log
            return {"success": False, "cost": 0.0, "error": str(e)}

    def _mock_run(self, task: dict, small_model: str, large_model: str, sig: str, cycle: int) -> dict:
        """Deterministic mock for smoke tests."""
        h = int(hashlib.md5(f"{task['task_id']}|{cycle}".encode()).hexdigest(), 16)
        small_ok = (h %10) < 6        # small succeeds ~60%!
        large_ok = (h % 10) < 9        # large succeeds ~90%
        if small_ok:
            return {
                "task_id": task["task_id"], "signature": sig,
                "decision": "probe:small→small_OK", "attempts": 1, "attempts_count": 1,
                "final_success": True, "final_model": small_model, "total_cost": 0.001,
                "round": cycle, "small_success": True, "large_success": False,
                "small_cost": 0.001, "large_cost": 0.0, "prompt": task.get("prompt", ""),
            }
        return {
            "task_id": task["task_id"], "signature": sig,
            "decision": f"probe:small_fail→large_{'OK' if large_ok else 'fail'}",
            "attempts": 2, "attempts_count": 2,
            "final_success": large_ok, "final_model": large_model,
            "total_cost": 0.011, "round": cycle,
            "small_success": False, "large_success": large_ok,
            "small_cost": 0.001, "large_cost": 0.01, "prompt": task.get("prompt", ""),
        }
