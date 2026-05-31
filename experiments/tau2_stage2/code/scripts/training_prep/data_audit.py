"""Training-prep data audit for sweep-B (data-size sweep) planning.

Extends profile_lengths.py with:
  - Tools-aware chat-template token counts (system + tools + messages).
  - Prompt vs loss-target token separation.
  - Stratification by (domain, phase, step-depth bucket).
  - Per-run aggregates: naive sum, sequence-packed bound, redundancy ratio.
  - Subsample-band simulation for both row-uniform and run-stratified plans.
  - Inference-rollout estimate from per-domain training distribution.

Usage (run from bundle root):
    .venv/bin/python -m scripts.training_prep.data_audit \\
        --root data_processed/stage2_v1 \\
        --tokenizer Qwen/Qwen2.5-1.5B-Instruct \\
        --out data_processed/stage2_v1/audit/audit_for_training.json
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def stat_block(vals: list) -> dict:
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    n = len(s)

    def pct(q):
        idx = min(n - 1, int(n * q))
        return s[idx]

    return {
        "n": n,
        "mean": round(sum(s) / n, 1),
        "p50": pct(0.50),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": s[-1],
    }


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def step_bucket(frac: float) -> str:
    if frac < 0.25:
        return "0-25"
    if frac < 0.50:
        return "25-50"
    if frac < 0.75:
        return "50-75"
    return "75-100"


def build_token_counter(tok, sys_texts: dict, tools_per_domain: dict):
    """Returns a function(row) -> (prompt_toks, target_toks, total_toks)."""

    def count(row: dict) -> tuple[int, int, int]:
        domain = row["_p"]["domain"]
        sys_text = sys_texts[domain]
        tools = tools_per_domain.get(domain) or None
        msgs_full = [{"role": "system", "content": sys_text}] + row["messages"]
        target_idx = row["_target_index"] + 1  # +1 for prepended system msg
        try:
            full_text = tok.apply_chat_template(
                msgs_full, tools=tools, tokenize=False, add_generation_prompt=False,
            )
            full_toks = len(tok.encode(full_text))
        except Exception:
            full_text = sys_text + "\n".join(m.get("content") or "" for m in msgs_full)
            full_toks = len(tok.encode(full_text))
        try:
            prompt_text = tok.apply_chat_template(
                msgs_full[:target_idx], tools=tools,
                tokenize=False, add_generation_prompt=True,
            )
            prompt_toks = len(tok.encode(prompt_text))
        except Exception:
            prompt_toks = max(0, full_toks - 200)
        target_toks = max(0, full_toks - prompt_toks)
        return prompt_toks, target_toks, full_toks

    return count


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="stage2 dir, e.g. data_processed/stage2_v1")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap on train rows (for smoke testing)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {args.tokenizer}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Domain assets
    da_dir = args.root / "domain_assets"
    sys_texts = {p.stem.replace("_system", ""): p.read_text() for p in da_dir.glob("*_system.txt")}
    tools_per_domain = {
        p.stem.replace("_tools", ""): json.loads(p.read_text())
        for p in da_dir.glob("*_tools.json")
    }
    print(f"Loaded domains: {sorted(sys_texts.keys())}", file=sys.stderr)

    train = load_jsonl(args.root / "train.jsonl")
    if args.limit:
        train = train[: args.limit]
    print(f"Loaded {len(train)} train rows", file=sys.stderr)

    counter = build_token_counter(tok, sys_texts, tools_per_domain)
    rows_data: list[dict] = []
    for i, row in enumerate(train):
        if i and i % 500 == 0:
            print(f"  tokenized {i}/{len(train)}", file=sys.stderr)
        prompt_t, target_t, total_t = counter(row)
        p = row["_p"]
        rows_data.append({
            "row_id": p["row_id"],
            "run_dir": p["run_dir"],
            "task_id": p["task_id"],
            "task_uniqueness_key": p["task_uniqueness_key"],
            "domain": p["domain"],
            "phase": p["phase"],
            "step_1based": p["step_1based"],
            "step_depth_frac": p["step_depth_frac"],
            "epoch_alias": p["epoch_alias"],
            "prompt_toks": prompt_t,
            "target_toks": target_t,
            "total_toks": total_t,
        })

    out: dict = {
        "tokenizer_id": args.tokenizer,
        "n_rows_train": len(rows_data),
    }

    # Stratified by (domain, phase, depth_bucket)
    by_stratum_total: dict = defaultdict(list)
    by_stratum_target: dict = defaultdict(list)
    for r in rows_data:
        key = f"{r['domain']}/{r['phase']}/depth_{step_bucket(r['step_depth_frac'])}"
        by_stratum_total[key].append(r["total_toks"])
        by_stratum_target[key].append(r["target_toks"])
    out["per_stratum_total_toks"] = {k: stat_block(v) for k, v in sorted(by_stratum_total.items())}
    out["per_stratum_target_toks"] = {k: stat_block(v) for k, v in sorted(by_stratum_target.items())}

    # Per-domain
    by_domain_total: dict = defaultdict(list)
    by_domain_target: dict = defaultdict(list)
    for r in rows_data:
        by_domain_total[r["domain"]].append(r["total_toks"])
        by_domain_target[r["domain"]].append(r["target_toks"])
    out["per_domain_total_toks"] = {k: stat_block(v) for k, v in sorted(by_domain_total.items())}
    out["per_domain_target_toks"] = {k: stat_block(v) for k, v in sorted(by_domain_target.items())}
    out["per_domain_row_counts"] = {k: len(v) for k, v in sorted(by_domain_total.items())}

    # Per-phase
    by_phase: dict = defaultdict(list)
    for r in rows_data:
        by_phase[r["phase"]].append(r["total_toks"])
    out["per_phase_total_toks"] = {k: stat_block(v) for k, v in sorted(by_phase.items())}
    out["per_phase_row_counts"] = {k: len(v) for k, v in sorted(by_phase.items())}

    # Per-run aggregates
    by_run: dict = defaultdict(list)
    for r in rows_data:
        by_run[r["run_dir"]].append(r)
    run_summaries: dict = {}
    for run_dir, rs in by_run.items():
        rs_sorted = sorted(rs, key=lambda x: x["step_1based"])
        sum_total = sum(r["total_toks"] for r in rs_sorted)
        max_total = max(r["total_toks"] for r in rs_sorted)
        sum_target = sum(r["target_toks"] for r in rs_sorted)
        run_summaries[run_dir] = {
            "domain": rs_sorted[0]["domain"],
            "n_steps": len(rs_sorted),
            "sum_total_toks": sum_total,
            "max_total_toks": max_total,
            "sum_target_toks": sum_target,
            "redundancy_ratio": round(sum_total / max(1, max_total), 2),
        }
    out["per_run"] = {
        "n_runs": len(run_summaries),
        "n_steps_dist": stat_block([s["n_steps"] for s in run_summaries.values()]),
        "sum_total_toks_dist": stat_block([s["sum_total_toks"] for s in run_summaries.values()]),
        "max_total_toks_dist": stat_block([s["max_total_toks"] for s in run_summaries.values()]),
        "sum_target_toks_dist": stat_block([s["sum_target_toks"] for s in run_summaries.values()]),
        "redundancy_ratio_dist": stat_block([s["redundancy_ratio"] for s in run_summaries.values()]),
    }

    # Global totals
    naive_total = sum(r["total_toks"] for r in rows_data)
    target_total = sum(r["target_toks"] for r in rows_data)
    packed_total = sum(s["max_total_toks"] for s in run_summaries.values())
    out["totals"] = {
        "naive_train_tokens_per_epoch": naive_total,
        "loss_target_tokens_per_epoch": target_total,
        "sequence_packed_tokens_per_epoch": packed_total,
        "redundancy_factor_overall": round(naive_total / max(1, packed_total), 2),
    }

    # Subsample-band simulation
    rng = random.Random(args.seed)

    def stratified_runs(target_n: int) -> list[str]:
        runs_by_domain: dict = defaultdict(list)
        for run_dir, rs in by_run.items():
            runs_by_domain[rs[0]["domain"]].append(run_dir)
        total = sum(len(v) for v in runs_by_domain.values())
        sampled = []
        for domain, runs in runs_by_domain.items():
            n_take = max(1, round(target_n * len(runs) / total))
            shuffled = list(runs)
            rng.shuffle(shuffled)
            sampled.extend(shuffled[: min(n_take, len(runs))])
        return sampled[:target_n]

    sweep_plan: dict = {}
    for n in [1000, 2000, 4000, len(rows_data)]:
        sample = rng.sample(rows_data, min(n, len(rows_data)))
        dc = defaultdict(int)
        for r in sample:
            dc[r["domain"]] += 1
        sweep_plan[f"rows_{n}"] = {
            "mode": "row-uniform",
            "n_rows": len(sample),
            "naive_total_toks": sum(r["total_toks"] for r in sample),
            "loss_target_toks": sum(r["target_toks"] for r in sample),
            "domain_counts": dict(sorted(dc.items())),
        }
    n_runs_total = len(by_run)
    for n in [50, 100, 200, n_runs_total]:
        runs_taken = stratified_runs(n)
        run_set = set(runs_taken)
        sample = [r for r in rows_data if r["run_dir"] in run_set]
        drc = defaultdict(int)
        for run_dir in runs_taken:
            drc[by_run[run_dir][0]["domain"]] += 1
        dc = defaultdict(int)
        for r in sample:
            dc[r["domain"]] += 1
        sweep_plan[f"runs_{n}"] = {
            "mode": "run-stratified-by-domain",
            "n_runs": len(runs_taken),
            "n_rows": len(sample),
            "naive_total_toks": sum(r["total_toks"] for r in sample),
            "sequence_packed_toks": sum(run_summaries[rd]["max_total_toks"] for rd in runs_taken),
            "loss_target_toks": sum(r["target_toks"] for r in sample),
            "domain_run_counts": dict(sorted(drc.items())),
            "domain_row_counts": dict(sorted(dc.items())),
        }
    out["sweep_band_simulation"] = sweep_plan

    # Inference-rollout estimate per domain
    eval_path = args.root / "eval_tasks.jsonl"
    eval_estimate: dict = {}
    if eval_path.exists():
        eval_tasks = load_jsonl(eval_path)
        for task in eval_tasks:
            d = task["domain"]
            if d in eval_estimate:
                continue
            train_dom = sorted(by_domain_total.get(d, []))
            if not train_dom:
                continue
            n = len(train_dom)
            eval_estimate[d] = {
                "p95_train_total_toks": train_dom[min(n - 1, int(n * 0.95))],
                "p99_train_total_toks": train_dom[min(n - 1, int(n * 0.99))],
                "max_train_total_toks": train_dom[-1],
                "n_eval_tasks": sum(1 for t in eval_tasks if t["domain"] == d),
            }
    out["inference_rollout_estimate"] = eval_estimate

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True))

    print(f"\nWrote {args.out}", file=sys.stderr)
    print(f"Train rows         : {out['n_rows_train']:>10,}", file=sys.stderr)
    print(f"Naive total tokens : {naive_total:>10,}", file=sys.stderr)
    print(f"Loss-target tokens : {target_total:>10,}", file=sys.stderr)
    print(f"Packed total tokens: {packed_total:>10,}", file=sys.stderr)
    print(f"Redundancy factor  : {out['totals']['redundancy_factor_overall']:>10.2f}x", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
