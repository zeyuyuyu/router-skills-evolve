"""Bench adapters: tau2_bench (default), swe_bench (stub)."""
from importlib import import_module
from typing import Protocol


class BenchAdapter(Protocol):
    """Contract every bench adapter must satisfy.

    Trace rows produced by `run_task_pair` MUST match the schema that
    `src/skills.py`, `experiments/train_learnable_router.py`, and
    `experiments/run_e2e_ablation.py` already consume:

        {
          "task_id":       str,
          "signature":     str,
          "decision":      str,          # e.g. "probe:small→small_OK"
          "attempts":      int,
          "attempts_count":int,
          "final_success": bool,
          "final_model":   str,
          "total_cost":    float,
          "round":         int,
          # optional extras (passed through unchanged):
          "small_success": bool,
          "large_success": bool,
          "small_cost":    float,
          "large_cost":    float,
          "prompt":        str,
        }
    """

    def load_tasks(self, n: int, split: str = "train") -> list[dict]: ...
    def run_task_pair(
        self,
        task: dict,
        small_model: str,
        large_model: str,
        cycle: int,
    ) -> dict: ...


def load_adapter(name: str) -> BenchAdapter:
    """Dynamically import bench by name."""
    mod = import_module(f"src.pipeline.benches.{name}.adapter")
    return mod.Adapter()
