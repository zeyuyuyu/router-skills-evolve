#!/usr/bin/env python3
"""Phase 1: bench-agnostic trace collection.

For each task in the chosen bench, run BOTH a small and a large model and
write one trace row per task. Output schema is the same one consumed by
`src/skills.py`, `experiments/train_learnable_router.py`, and
`experiments/run_e2e_ablation.py`.

Usage:
    python experiments/scaling/collect_traces.py \
        --bench tau2_bench \
        --n-tasks 30 \
        --small-model deepseek/deepseek-v3.2 \
        --large-model openai/gpt-5.4-2026-03-05 \
        --out results/scaling_xxx/cycle_0/traces.jsonl

For smoke testing without GPUs/API keys:
    SCALING_MOCK=1 python experiments/scaling/collect_traces.py --bench tau2_bench --n-tasks 30 --out /tmp/traces.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Make `experiments.scaling.benches` importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scaling.benches import load_adapter  # noqa: E402


class TaskTimeoutError(TimeoutError):
    """Raised when one benchmark task exceeds the configured wall-time cap."""


@contextmanager
def _task_timeout(seconds: int | None):
    if not seconds or seconds <= 0:
        yield
        return

    def _raise_timeout(_signum: int, _frame: object) -> None:
        raise TaskTimeoutError(f"task exceeded timeout of {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _load_existing_traces(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    """Read valid existing trace rows keyed by task_id.

    A killed process can leave a partial final line. Ignore invalid rows rather
    than failing the resume; the task will be run again if its task_id was not
    recovered from a complete JSON row.
    """
    rows: dict[str, dict[str, Any]] = {}
    invalid = 0
    if not path.exists():
        return rows, invalid
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            task_id = str(row.get("task_id") or "")
            if task_id:
                rows[task_id] = row
    return rows, invalid


def _parse_task_ids(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def _signature_for_task(task: dict[str, Any]) -> str:
    prompt = task.get("prompt", "")
    head = (prompt or "").strip().split("\n")[0][:80]
    bucket = len(prompt or "") // 200
    return f"{head}::len_bucket={bucket}" if prompt else ""


def _failure_trace(task: dict[str, Any], cycle: int, decision: str, error: str) -> dict[str, Any]:
    return {
        "task_id": str(task.get("task_id", "")),
        "signature": _signature_for_task(task),
        "decision": decision,
        "attempts": 0,
        "attempts_count": 0,
        "final_success": False,
        "final_model": "",
        "total_cost": 0.0,
        "round": cycle,
        "small_success": False,
        "large_success": False,
        "small_cost": 0.0,
        "large_cost": 0.0,
        "prompt": task.get("prompt", ""),
        "error": error,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", required=True, choices=["tau2_bench", "swe_bench"])
    ap.add_argument("--n-tasks", type=int, required=True)
    ap.add_argument("--small-model", required=True)
    ap.add_argument("--large-model", required=True)
    ap.add_argument("--out", required=True, help="output traces.jsonl path")
    ap.add_argument("--cycle", type=int, default=0)
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--mock", action="store_true",
                    help="generate synthetic deterministic traces (no API/GPU). Equivalent to SCALING_MOCK=1.")
    ap.add_argument("--resume-existing", action="store_true",
                    help="append to an existing output file and skip task_ids that already have valid trace rows.")
    ap.add_argument("--max-cost-usd", type=float, default=None,
                    help="stop before starting another task once accumulated trace cost reaches this USD cap.")
    ap.add_argument("--task-timeout-s", type=int, default=None,
                    help="wall-time cap for a single task pair; writes an error row and continues on timeout.")
    ap.add_argument("--max-zero-cost-failures", type=int, default=None,
                    help="stop after this many consecutive zero-cost failed trace rows, usually an API/runtime issue.")
    ap.add_argument("--skip-task-ids", default=None,
                    help="comma-separated task_id list to write as skipped failure rows and continue.")
    args = ap.parse_args()

    if args.max_cost_usd is None and os.environ.get("SCALING_MAX_COST_USD"):
        args.max_cost_usd = float(os.environ["SCALING_MAX_COST_USD"])
    if args.task_timeout_s is None and os.environ.get("SCALING_TASK_TIMEOUT_S"):
        args.task_timeout_s = int(os.environ["SCALING_TASK_TIMEOUT_S"])
    if args.max_zero_cost_failures is None and os.environ.get("SCALING_MAX_ZERO_COST_FAILURES"):
        args.max_zero_cost_failures = int(os.environ["SCALING_MAX_ZERO_COST_FAILURES"])
    if args.skip_task_ids is None and os.environ.get("SCALING_SKIP_TASK_IDS"):
        args.skip_task_ids = os.environ["SCALING_SKIP_TASK_IDS"]

    skip_task_ids = _parse_task_ids(args.skip_task_ids)

    if args.mock:
        os.environ["SCALING_MOCK"] = "1"

    adapter = load_adapter(args.bench)
    print(f"[collect_traces] bench={args.bench} cycle={args.cycle} split={args.split} "
          f"mock={os.environ.get('SCALING_MOCK', '0') == '1'}", file=sys.stderr)

    tasks = adapter.load_tasks(args.n_tasks, split=args.split)
    print(f"[collect_traces] loaded {len(tasks)} tasks", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict[str, Any]] = {}
    if args.resume_existing:
        existing, invalid = _load_existing_traces(out_path)
        if existing or invalid:
            print(f"[collect_traces] resume_existing: loaded {len(existing)} valid rows "
                  f"from {out_path} (ignored {invalid} invalid rows)", file=sys.stderr)
    if skip_task_ids:
        print(f"[collect_traces] skip_task_ids={','.join(sorted(skip_task_ids))}", file=sys.stderr)

    total_cost = sum(float(row.get("total_cost") or 0.0) for row in existing.values())
    n_success = sum(1 for row in existing.values() if row.get("final_success"))
    zero_cost_failures = 0
    n_done = 0
    t0 = time.time()
    mode = "a" if args.resume_existing else "w"
    with out_path.open(mode) as fh:
        for i, task in enumerate(tasks, 1):
            if args.max_cost_usd is not None and total_cost >= args.max_cost_usd:
                print(f"[collect_traces] COST CAP reached before task {i}: "
                      f"total_cost=${total_cost:.6f} >= cap=${args.max_cost_usd:.6f}; stopping",
                      file=sys.stderr)
                break

            task_id = str(task.get("task_id", f"unknown-{i}"))
            if task_id in existing:
                n_done += 1
                if i %10 == 0 or i == len(tasks):
                    elapsed = time.time() - t0
                    print(f"[collect_traces] {i}/{len(tasks)}  "
                          f"success={n_success}/{n_done}  skipped_existing={n_done}  "
                          f"elapsed={elapsed:.1f}s",
                          file=sys.stderr)
                continue
            if task_id in skip_task_ids:
                trace = _failure_trace(
                    task,
                    args.cycle,
                    decision="skipped:operator_requested",
                    error="operator requested skip via --skip-task-ids",
                )
                fh.write(json.dumps(trace, ensure_ascii=False) + "\n")
                fh.flush()
                existing[task_id] = trace
                n_done += 1
                print(f"[collect_traces] task {task_id} SKIPPED by operator request", file=sys.stderr)
                continue
            try:
                with _task_timeout(args.task_timeout_s):
                    trace = adapter.run_task_pair(
                        task,
                        small_model=args.small_model,
                        large_model=args.large_model,
                        cycle=args.cycle,
                    )
                fh.write(json.dumps(trace, ensure_ascii=False) + "\n")
                fh.flush()
                n_done += 1
                trace_cost = float(trace.get("total_cost") or 0.0)
                total_cost += trace_cost
                if trace.get("final_success"):
                    n_success += 1
                    zero_cost_failures = 0
                elif trace_cost <= 0:
                    zero_cost_failures += 1
                else:
                    zero_cost_failures = 0
                if i %10 == 0 or i == len(tasks):
                    elapsed = time.time() - t0
                    print(f"[collect_traces] {i}/{len(tasks)}  "
                          f"success={n_success}/{n_done}  "
                          f"cost=${total_cost:.6f}  elapsed={elapsed:.1f}s",
                          file=sys.stderr)
                if args.max_cost_usd is not None and total_cost >= args.max_cost_usd:
                    print(f"[collect_traces] COST CAP reached after task {i}: "
                          f"total_cost=${total_cost:.6f} >= cap=${args.max_cost_usd:.6f}; stopping",
                          file=sys.stderr)
                    break
                if (
                    args.max_zero_cost_failures is not None
                    and zero_cost_failures >= args.max_zero_cost_failures
                ):
                    print(f"[collect_traces] ZERO-COST FAILURE CAP reached after task {i}: "
                          f"{zero_cost_failures} consecutive failed rows with zero cost; stopping",
                          file=sys.stderr)
                    break
            except Exception as e:  # noqa: BLE001
                print(f"[collect_traces] task {task_id} FAILED: {e}", file=sys.stderr)
                # write a failure row so downstream can see it
                fh.write(json.dumps(_failure_trace(
                    task,
                    args.cycle,
                    decision="error",
                    error=str(e),
                ), ensure_ascii=False) + "\n")
                fh.flush()
                n_done += 1
                zero_cost_failures += 1
                if (
                    args.max_zero_cost_failures is not None
                    and zero_cost_failures >= args.max_zero_cost_failures
                ):
                    print(f"[collect_traces] ZERO-COST FAILURE CAP reached after task {i}: "
                          f"{zero_cost_failures} consecutive exceptions; stopping",
                          file=sys.stderr)
                    break

    elapsed = time.time() - t0
    print(f"[collect_traces] DONE  out={out_path}  "
          f"final_success={n_success}/{n_done} ({100*n_success/max(1,n_done):.1f}%)  "
          f"total_cost=${total_cost:.6f}  "
          f"elapsed={elapsed:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
