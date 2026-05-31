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
import sys
import time
from pathlib import Path
from typing import Any

# Make `experiments.scaling.benches` importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scaling.benches import load_adapter  # noqa: E402


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
    args = ap.parse_args()

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

    n_success = sum(1 for row in existing.values() if row.get("final_success"))
    n_done = 0
    t0 = time.time()
    mode = "a" if args.resume_existing else "w"
    with out_path.open(mode) as fh:
        for i, task in enumerate(tasks, 1):
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
            try:
                trace = adapter.run_task_pair(
                    task,
                    small_model=args.small_model,
                    large_model=args.large_model,
                    cycle=args.cycle,
                )
                fh.write(json.dumps(trace, ensure_ascii=False) + "\n")
                fh.flush()
                n_done += 1
                if trace.get("final_success"):
                    n_success += 1
                if i %10 == 0 or i == len(tasks):
                    elapsed = time.time() - t0
                    print(f"[collect_traces] {i}/{len(tasks)}  "
                          f"success={n_success}/{n_done}  elapsed={elapsed:.1f}s",
                          file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f"[collect_traces] task {task_id} FAILED: {e}", file=sys.stderr)
                # write a failure row so downstream can see it
                fh.write(json.dumps({
                    "task_id": task_id,
                    "signature": "",
                    "decision": "error",
                    "attempts": 0,
                    "attempts_count": 0,
                    "final_success": False,
                    "final_model": "",
                    "total_cost": 0.0,
                    "round": args.cycle,
                    "error": str(e),
                }) + "\n")
                fh.flush()
                n_done += 1

    elapsed = time.time() - t0
    print(f"[collect_traces] DONE  out={out_path}  "
          f"final_success={n_success}/{n_done} ({100*n_success/max(1,n_done):.1f}%)  "
          f"elapsed={elapsed:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
