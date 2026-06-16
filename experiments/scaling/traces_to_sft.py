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
import os
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
    """small failed (all turns) AND large succeeded — the tasks worth distilling."""
    small_ok = trace.get("small_success")
    large_ok = trace.get("large_success")
    if isinstance(small_ok, bool) and isinstance(large_ok, bool):
        return (not small_ok) and large_ok
    # Fall back to the decision string if booleans are absent.
    decision = (trace.get("decision") or "").lower()
    return ("small_fail" in decision) and ("large_ok" in decision or "large_OK" in decision.lower())


def _extract_repair_samples(trace: dict, procedure: str = "") -> list[dict]:
    """Extract self-repair SFT samples from a multi-turn trace.

    A repair sample is emitted when the small model FAILED on turn 1 but
    SUCCEEDED on a later turn — the repair conversation teaches the model to
    fix its own errors. The full turn history becomes a multi-turn chat, stored
    both as `messages` (for chat-aware trainers) and flattened into
    `instruction`/`output` (for backward compat with train_small_model.py).
    """
    small_turns = trace.get("small_turns", [])
    if len(small_turns) < 2:
        return []
    # First turn must have failed; find the first successful repair turn
    if small_turns[0].get("ok"):
        return []
    success_turn = next((t for t in small_turns[1:] if t.get("ok")), None)
    if success_turn is None:
        return []

    problem = trace.get("prompt", "")
    if not problem:
        return []

    # Build the conversation up to (and including) the successful repair
    first_content = f"{procedure}\n\n---\n\n{problem}" if procedure else problem
    messages = [{"role": "user", "content": first_content}]
    for t in small_turns:
        if t.get("ok"):
            break
        messages.append({"role": "assistant", "content": t.get("code", "")})
        messages.append({
            "role": "user",
            "content": (
                f"Your code failed with the following error:\n\n"
                f"{t.get('error', '')}\n\n"
                "Analyze the error and return a corrected version of the complete function."
            ),
        })
    target_code = success_turn.get("code", "")
    if not target_code:
        return []
    messages.append({"role": "assistant", "content": target_code})

    # Flatten to a single instruction for train_small_model.py compat:
    # concatenate all user turns separated by the failed assistant attempts.
    flat_parts = []
    for msg in messages[:-1]:  # exclude final assistant (= output)
        if msg["role"] == "user":
            flat_parts.append(msg["content"])
        else:
            flat_parts.append(f"[Previous attempt (failed)]:\n{msg['content']}")
    flat_instruction = "\n\n".join(flat_parts)

    return [{
        "task_id": trace.get("task_id", ""),
        "signature": trace.get("signature", ""),
        "sft_type": "self_repair",
        "repair_turns": success_turn.get("turn", 2),
        "prompt": flat_instruction,
        "completion": target_code,
        "source_model": "small",
        "has_procedure": bool(procedure),
        "messages": messages,
        # train_small_model.py compat
        "instruction": flat_instruction,
        "input": "",
        "output": target_code,
    }]


def _load_procedures(skillbook_path):
    """Return a callable prompt->procedure (or one that always returns '')."""
    if not skillbook_path:
        return lambda _p: ""
    p = Path(skillbook_path)
    if not p.exists():
        return lambda _p: ""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.skills import SkillBook
        sb = SkillBook()
        sb.load(p)
        return sb.get_procedure
    except Exception as e:  # noqa: BLE001
        print(f"[traces_to_sft] WARN could not load skillbook procedures ({e})", file=sys.stderr)
        return lambda _p: ""


def convert(traces_path: Path, out_path: Path, skillbook_path=None,
            include_success: bool = False) -> dict:
    """Build SFT pairs from traces.

    Path A (always): self-repair — small fixed its own error across turns.
    Path B (always): teacher-forcing — small failed all turns, large succeeded.
    Path C (include_success): success-clone — small ALREADY solved it; clone its
        own correct solution. Expands the set beyond just hard tasks so SFT also
        reinforces what the model gets right (more volume, less forgetting). The
        target is the small model's own first correct completion (on-policy), so
        this is behaviour cloning, not teacher distillation.
    """
    n_total = 0
    n_hard = 0
    n_written = 0
    n_no_completion = 0
    n_with_procedure = 0
    n_repair = 0
    n_success = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    get_procedure = _load_procedures(skillbook_path)

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
            prompt = trace.get("prompt") or ""
            procedure = get_procedure(prompt) if prompt else ""
            sft_prompt = f"{procedure}\n\n---\n\n{prompt}" if procedure else prompt

            # ── Path A: self-repair samples (small fixed itself after turn 1) ──
            for repair_row in _extract_repair_samples(trace, procedure=procedure):
                out.write(json.dumps(repair_row, ensure_ascii=False) + "\n")
                n_repair += 1
                n_written += 1

            if not _is_hard_task(trace):
                # ── Path C: success-clone (opt-in) — small solved it itself ──
                if include_success and trace.get("small_success") and prompt:
                    small_code = trace.get("small_completion") or ""
                    # Skip if the win came purely from a repair turn (Path A already
                    # captured that conversation) and there is no single-shot code.
                    if small_code:
                        if procedure:
                            n_with_procedure += 1
                        row = {
                            "task_id": trace.get("task_id", ""),
                            "original_task_id": trace.get("original_task_id", trace.get("task_id", "")),
                            "domain": trace.get("domain", ""),
                            "signature": trace.get("signature", ""),
                            "sft_type": "success",
                            "prompt": sft_prompt,
                            "completion": small_code,
                            "source_model": "small",
                            "has_procedure": bool(procedure),
                            "instruction": sft_prompt,
                            "input": "",
                            "output": small_code,
                        }
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        n_success += 1
                        n_written += 1
                continue

            # ── Path B: teacher-forcing (small failed all turns, large succeeded) ──
            n_hard += 1
            completion = _completion_for_large(trace)
            if not prompt or not completion:
                n_no_completion += 1
                continue
            if procedure:
                n_with_procedure += 1
            row = {
                "task_id": trace.get("task_id", ""),
                "original_task_id": trace.get("original_task_id", trace.get("task_id", "")),
                "domain": trace.get("domain", ""),
                "signature": trace.get("signature", ""),
                "sft_type": "teacher",
                "prompt": sft_prompt,
                "completion": completion,
                "source_model": trace.get("final_model") or trace.get("large_model") or "",
                "has_procedure": bool(procedure),
                "instruction": sft_prompt,
                "input": "",
                "output": completion,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    return {
        "traces_total": n_total,
        "hard_tasks": n_hard,
        "written": n_written,
        "teacher_pairs": n_hard - n_no_completion,
        "repair_pairs": n_repair,
        "success_pairs": n_success,
        "hard_without_completion": n_no_completion,
        "with_procedure": n_with_procedure,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--skillbook", default=None,
                    help="optional skillbook.json; prepend the matched cluster's "
                         "distilled procedure to each SFT prompt")
    ap.add_argument("--include-success", action="store_true",
                    default=os.environ.get("SFT_INCLUDE_SUCCESS", "0") == "1",
                    help="also emit behaviour-clone pairs for tasks the small "
                         "model already solved (Path C) — not just hard tasks. "
                         "Env: SFT_INCLUDE_SUCCESS=1")
    args = ap.parse_args()

    traces_path = Path(args.traces)
    if not traces_path.exists():
        print(f"[traces_to_sft] ERROR: traces not found at {traces_path}", file=sys.stderr)
        return 2

    stats = convert(traces_path, Path(args.output), skillbook_path=args.skillbook,
                    include_success=args.include_success)
    print(f"[traces_to_sft] {stats['traces_total']} traces -> "
          f"{stats['written']} SFT pairs total  "
          f"(teacher={stats.get('teacher_pairs', 0)}  "
          f"repair={stats.get('repair_pairs', 0)}  "
          f"success={stats.get('success_pairs', 0)}  "
          f"with_procedure={stats.get('with_procedure', 0)}  "
          f"skipped={stats['hard_without_completion']})  "
          f"out={args.output}", file=sys.stderr)
    if stats["written"] == 0:
        print("[traces_to_sft] WARN: 0 SFT pairs written. Either no hard tasks this "
              "cycle, or the adapter did not record large_completion. Check the "
              "adapter writes a completion field (see PIPELINE_AUDIT.md).",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
