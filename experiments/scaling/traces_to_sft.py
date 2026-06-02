#!/usr/bin/env python3
"""Phase 3 helper: bench-agnostic trace -> SFT prompt/completion converter.

Why this exists
---------------
`experiments/extract_training_data.py` is HumanEval-era code: it loads
`data/HumanEval.jsonl`, looks tasks up by `task_id`, and *re-generates* code
with the large model via `src.solve_task`. On tau2 / SWE traces that lookup
misses and nothing useful comes out — it is NOT a trajectory -> prompt/completion
converter. (Review finding, 2026-05-21.)

This converter reads the scaling trace schema directly and emits SFT pairs from
the trajectories already collected in Phase 1 — no extra model calls, no
HumanEval coupling, works for any bench whose adapter follows the trace schema
in `experiments/scaling/benches/__init__.py`.

Selection rule (the "hard tasks worth distilling")
--------------------------------------------------
Keep a row iff the small model FAILED and the large model SUCCEEDED — i.e. the
tasks the router currently has to escalate. The large model's completion on
those tasks is the SFT target that hardens the small model. This mirrors the
intent of the old extract_training_data.py but sources the completion straight
from the trace instead of re-querying.

Requires the trace to carry the large model's completion text. Adapters write
this as `large_completion` (and `small_completion`); see the adapter patch in
the same review. Rows without a usable completion are skipped and counted.

Output: JSONL, one object per line:
    {"task_id", "signature", "prompt", "completion", "source_model"}
in Alpaca-ish shape (also mirrored as instruction/output for compatibility with
train_small_model.py).

Usage:
    python experiments/scaling/traces_to_sft.py \
        --traces results/scaling_xxx/cycle_1/traces.jsonl \
        --output results/scaling_xxx/cycle_1/training_data.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _completion_for_large(trace: dict) -> str | None:
    """Best-effort extraction of the large model's completion text from a trace."""
    # Preferred explicit field written by patched adapters.
    for key in ("large_completion", "completion", "output"):
        v = trace.get(key)
        if isinstance(v, str) and v.strip():
            return v
    # Fall back to a raw result blob if the adapter passed one through.
    raw = trace.get("large_raw") or trace.get("raw")
    if isinstance(raw, dict):
        for key in ("completion", "output", "final_answer", "answer", "text"):
            v = raw.get(key)
            if isinstance(v, str) and v.strip():
                return v
    return None


def _is_hard_task(trace: dict) -> bool:
    """small failed AND large succeeded — the tasks worth distilling."""
    small_ok = trace.get("small_success")
    large_ok = trace.get("large_success")
    if isinstance(small_ok, bool) and isinstance(large_ok, bool):
        return (not small_ok) and large_ok
    # Fall back to the decision string if booleans are absent.
    decision = (trace.get("decision") or "").lower()
    return ("small_fail" in decision) and ("large_ok" in decision or "large_OK" in decision.lower())


def convert(traces_path: Path, out_path: Path) -> dict:
    n_total = 0
    n_hard = 0
    n_written = 0
    n_no_completion = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with traces_path.open() as fh, out_path.open("w") as out:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                trace = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1
            if not _is_hard_task(trace):
                continue
            n_hard += 1
            prompt = trace.get("prompt") or ""
            completion = _completion_for_large(trace)
            if not prompt or not completion:
                n_no_completion += 1
                continue
            row = {
                "task_id": trace.get("task_id", ""),
                "signature": trace.get("signature", ""),
                "prompt": prompt,
                "completion": completion,
                "source_model": trace.get("final_model") or trace.get("large_model") or "",
                # train_small_model.py compatibility:
                "instruction": prompt,
                "input": "",
                "output": completion,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    return {
        "traces_total": n_total,
        "hard_tasks": n_hard,
        "written": n_written,
        "hard_without_completion": n_no_completion,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    traces_path = Path(args.traces)
    if not traces_path.exists():
        print(f"[traces_to_sft] ERROR: traces not found at {traces_path}", file=sys.stderr)
        return 2

    stats = convert(traces_path, Path(args.output))
    print(f"[traces_to_sft] {stats['traces_total']} traces -> "
          f"{stats['hard_tasks']} hard tasks -> {stats['written']} SFT pairs "
          f"(skipped {stats['hard_without_completion']} hard tasks lacking a usable completion)  "
          f"out={args.output}", file=sys.stderr)
    if stats["written"] == 0:
        print("[traces_to_sft] WARN: 0 SFT pairs written. Either no hard tasks this "
              "cycle, or the adapter did not record large_completion. Check the "
              "adapter writes a completion field (see PIPELINE_AUDIT.md).",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
