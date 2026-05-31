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

# Make `experiments.scaling.benches` importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scaling.benches import load_adapter  # noqa: E402


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

    n_success = 0
    t0 = time.time()
    with out_path.open("w") as fh:
        for i, task in enumerate(tasks, 1):
            try:
                trace = adapter.run_task_pair(
                    task,
                    small_model=args.small_model,
                    large_model=args.large_model,
                    cycle=args.cycle,
                )
                fh.write(json.dumps(trace, ensure_ascii=False) + "\n")
                fh.flush()
                if trace.get("final_success"):
                    n_success += 1
                if i %10 == 0 or i == len(tasks):
                    elapsed = time.time() - t0
                    print(f"[collect_traces] {i}/{len(tasks)}  "
                          f"success={n_success}/{i}  elapsed={elapsed:.1f}s",
                          file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f"[collect_traces] task {task.get('task_id')} FAILED: {e}", file=sys.stderr)
                # write a failure row so downstream can see it
                fh.write(json.dumps({
                    "task_id": task.get("task_id", f"unknown-{i}"),
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

    elapsed = time.time() - t0
    print(f"[collect_traces] DONE  out={out_path}  "
          f"final_success={n_success}/{len(tasks)} ({100*n_success/max(1,len(tasks)):.1f}%)  "
          f"elapsed={elapsed:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
