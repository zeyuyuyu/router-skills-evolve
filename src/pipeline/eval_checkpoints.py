#!/usr/bin/env python3
"""Held-out pass@1 attribution waterfall: how much did each component add?

Runs the SAME generation path (the HumanEval adapter) on the held-out eval
split and reports a 4-stage waterfall, isolating each component's contribution:

    base   : base model, RAW prompt (no procedure)        ← starting point
    +skills: base model, prompt WITH distilled procedure   → Δ skills
    +sft   : SFT adapter, prompt with procedure            → Δ sft
    +grpo  : GRPO/DAPO adapter, prompt with procedure       → Δ grpo

Each Δ vs the previous stage attributes the gain to that one component. The
default run reproduces this waterfall; pass --no-skills-stage to compare model
weights only (all stages with procedure).

Usage:
    python src/pipeline/eval_checkpoints.py \
        --base Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --sft  results/.../cycle_0/llm_adapter \
        --rl   results/.../cycle_0/grpo_adapter \
        --skillbook results/.../cycle_0/skillbook.json \
        [--n 82] [--repair-turns 1]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _eval_one(label: str, model_id: str, tasks: list[dict], procedure_fn,
              repair_turns: int, use_procedure: bool) -> dict:
    """Run a single stage over the eval tasks; return pass@1 stats.

    use_procedure=False forces the raw prompt even if a skillbook is loaded
    (used for the 'base' baseline so the skills Δ can be isolated).
    """
    os.environ["HE_MAX_REPAIR_TURNS"] = str(repair_turns)
    # Backend is resolved from model_id + HE_VLLM_MAP (set by caller). vllm/api
    # backends are HTTP → fire tasks concurrently (vLLM continuous-batches them);
    # the in-process HF backend stays sequential (one shared GPU model).
    from src.pipeline.benches.humaneval.adapter import Adapter, _resolve_backend

    adapter = Adapter()
    backend = _resolve_backend(model_id)
    workers = (int(os.environ.get("EVAL_WORKERS", "8"))
               if backend in ("vllm", "api") else 1)

    def _run(t):
        procedure = (procedure_fn(t["prompt"]) if (procedure_fn and use_procedure) else "")
        ok, _code, turns = adapter._gen_and_test(model_id, t, procedure=procedure)
        return bool(ok), bool(turns and turns[0].get("ok"))

    flags = []
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        print(f"  [{label}] backend={backend} parallel workers={workers}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, r in enumerate(ex.map(_run, tasks), 1):
                flags.append(r)
                if i % 20 == 0:
                    print(f"  [{label}] {i}/{len(tasks)}  pass={sum(f[0] for f in flags)}", flush=True)
    else:
        for i, t in enumerate(tasks, 1):
            flags.append(_run(t))
            if i % 20 == 0:
                print(f"  [{label}] {i}/{len(tasks)}  pass={sum(f[0] for f in flags)}", flush=True)
    n_pass = sum(1 for ok, _ in flags if ok)
    n_pass_turn1 = sum(1 for _, t1 in flags if t1)
    return {
        "label": label,
        "model": model_id,
        "use_procedure": use_procedure,
        "n": len(tasks),
        "pass@1": round(n_pass / len(tasks), 4),
        "pass@1_turn1": round(n_pass_turn1 / len(tasks), 4),
        "n_pass": n_pass,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--sft", default=None, help="SFT adapter dir (llm_adapter)")
    ap.add_argument("--rl", default=None, help="GRPO/DAPO adapter dir (grpo_adapter)")
    ap.add_argument("--skillbook", default=None,
                    help="prepend matched procedure (as deployed). Omit for raw prompt.")
    ap.add_argument("--bench-data", default=str(REPO_ROOT / "data" / "HumanEval.jsonl"))
    ap.add_argument("--n", type=int, default=82)
    ap.add_argument("--repair-turns", type=int, default=1,
                    help="1 = pure pass@1 (single shot); >1 = allow ReAct repair")
    ap.add_argument("--no-skills-stage", action="store_true",
                    help="skip the base+skills row; compare model weights only "
                         "(all stages use the procedure)")
    ap.add_argument("--fewshot-k", type=int, default=int(os.environ.get("SKILL_FEWSHOT_K", "0")),
                    help="skill = procedure + K nearest-neighbour worked examples "
                         "(0 = procedure only). Few-shot is part of the skill.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Eval split = odd indices (matches adapter.load_tasks split='eval').
    all_tasks = []
    with open(args.bench_data) as f:
        for line in f:
            line = line.strip()
            if line:
                all_tasks.append(json.loads(line))
    tasks = all_tasks[1::2][: args.n]
    print(f"[eval] {len(tasks)} held-out tasks  repair_turns={args.repair_turns}", flush=True)

    procedure_fn = None
    if args.skillbook and Path(args.skillbook).exists():
        from src.skills import SkillBook
        sb = SkillBook()
        sb.load(Path(args.skillbook))
        if args.fewshot_k > 0:
            procedure_fn = lambda p: sb.get_context(p, fewshot_k=args.fewshot_k)
            print(f"[eval] skill = procedure + {args.fewshot_k} few-shot exemplars "
                  f"({len(sb.skills)} clusters)", flush=True)
        else:
            procedure_fn = sb.get_procedure
            print(f"[eval] using skillbook procedures ({len(sb.skills)} clusters)", flush=True)

    # Build the waterfall stages: (label, model_id, use_procedure).
    have_skills = procedure_fn is not None and not args.no_skills_stage
    stages = []
    stages.append(("base", args.base, False))            # raw prompt baseline
    if have_skills:
        stages.append(("+skills", args.base, True))      # base + procedure
    if args.sft:
        stages.append(("+sft", args.sft, have_skills or bool(procedure_fn)))
    if args.rl:
        stages.append(("+grpo", args.rl, have_skills or bool(procedure_fn)))

    results = []
    for label, model_id, use_proc in stages:
        tag = f"{label} (proc={'on' if use_proc else 'off'})"
        print(f"\n[eval] === {tag}: {model_id} ===", flush=True)
        r = _eval_one(label, model_id, tasks, procedure_fn, args.repair_turns, use_proc)
        results.append(r)

    # Waterfall report: each Δ is vs the PREVIOUS stage = that component's gain.
    print("\n" + "=" * 72)
    print(f"{'stage':10} {'pass@1':>8} {'Δ vs prev':>11} {'component':>14}")
    print("-" * 72)
    component_for = {"+skills": "skills", "+sft": "SFT", "+grpo": "GRPO/DAPO"}
    prev = None
    for r in results:
        if prev is None:
            print(f"{r['label']:10} {r['pass@1']:>8.4f} {'—':>11} {'(baseline)':>14}")
        else:
            d = r["pass@1"] - prev
            comp = component_for.get(r["label"], "")
            print(f"{r['label']:10} {r['pass@1']:>8.4f} {d:>+11.4f} {comp:>14}")
        prev = r["pass@1"]
    if len(results) >= 2:
        total = results[-1]["pass@1"] - results[0]["pass@1"]
        print("-" * 72)
        print(f"{'TOTAL':10} {results[-1]['pass@1']:>8.4f} {total:>+11.4f} {'all stages':>14}")
    print("=" * 72)

    if args.out:
        json.dump({"stages": results,
                   "n_eval": len(tasks),
                   "repair_turns": args.repair_turns},
                  open(args.out, "w"), indent=2)
        print(f"[eval] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
