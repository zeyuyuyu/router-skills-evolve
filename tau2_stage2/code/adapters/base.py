"""The adapter Protocol every benchmark must satisfy.

Adapters translate between a benchmark's native API (task schema, env,
grader) and our shared pipeline (supervision records, cost accounting,
routing search). The Protocol is deliberately task-scoped, not turn-scoped —
tau2-bench v1.0.0 ships its own multi-turn orchestrator and we should call
it as a library rather than reimplementing the loop.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from core.schemas.artifacts import RunTaskConfig, TaskRunResult


@runtime_checkable
class BenchmarkAdapter(Protocol):
    benchmark_name: str

    def load_tasks(self, subset: str) -> list[dict[str, Any]]: ...

    def load_ground_truth(self, subset: str) -> dict[str, Any]: ...

    def run_task(self, task: Any, config: RunTaskConfig) -> TaskRunResult: ...
