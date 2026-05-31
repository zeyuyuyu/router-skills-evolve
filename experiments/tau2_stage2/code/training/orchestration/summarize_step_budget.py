"""Aggregate per-target rollout JSONs into SUMMARY_step_budget.csv.

Reads from $BUNDLE_ROOT/step_budget_outputs/<target>/raw_eval/*.json and
joins with each target's training_log.json (when present, for n_train_rows)
and base_models/manifest.json (for the untrained anchors). Output:

    $BUNDLE_ROOT/step_budget_outputs/SUMMARY_step_budget.csv

Columns:
    target_name, target_kind ("trained"|"base"), model_family (2B/4B/9B/35B-A3B),
    n_train_rows, seed_policy, n_chosen, n_rollouts_method_b, n_rollouts_method_a,
    student_rep_rate (Method B headline),
    student_rep_rate_airline / _retail / _telecom,
    rep_rate_pinned_chosen_mean    (data-collection ceiling on the chosen set),
    closure_ratio  (= student_rep_rate / rep_rate_pinned_chosen_mean),
    pass_rate_e2e  (Method A headline),
    pass_rate_e2e_<domain>,
    mean_depth_in_gold_actions_failed_b,    (per-step diagnostic on failed B rollouts)
    mean_steps_used_e2e,
    total_cost_usd, n_llm_calls

Tolerates incomplete data: a target with no rollouts emits NaN/0 for stats.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

# Map target name → model family
_FAMILY_PATTERNS = [
    (re.compile(r".*qwen3_5_2b.*|.*Qwen3\.5-2B"), "Qwen3.5-2B"),
    (re.compile(r".*qwen3_5_4b.*|.*Qwen3\.5-4B"), "Qwen3.5-4B"),
    (re.compile(r".*qwen3_5_9b.*|.*Qwen3\.5-9B"), "Qwen3.5-9B"),
    (re.compile(r".*qwen3_6_35b.*|.*Qwen3\.6-35B-A3B"), "Qwen3.6-35B-A3B"),
]


def detect_family(target_name: str) -> str:
    for rx, fam in _FAMILY_PATTERNS:
        if rx.match(target_name):
            return fam
    return "unknown"


def detect_kind(target_name: str) -> str:
    return "base" if target_name.startswith("base_") else "trained"


def find_n_train_rows(bundle_root: Path, target_name: str) -> int | None:
    """Return n_train_rows for a trained target. None for base or missing log."""
    if target_name.startswith("base_"):
        return 0
    log = bundle_root / "train_outputs" / target_name / "training_log.json"
    if not log.exists():
        return None
    try:
        d = json.loads(log.read_text())
    except Exception:
        return None
    return d.get("n_train_rows") or d.get("n_rows") or d.get("dataset_size")


def _load_rollout_files(target_dir: Path) -> list[dict]:
    rollouts = []
    raw_dir = target_dir / "raw_eval"
    if not raw_dir.exists():
        return rollouts
    for p in sorted(raw_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
            if d.get("_marker") == "COMPLETE":
                rollouts.append(d)
        except Exception:
            continue
    return rollouts


def _per_domain_rate(rollouts: list[dict]) -> dict[str, float]:
    by_dom: dict[str, list[bool]] = defaultdict(list)
    for r in rollouts:
        dom = (r.get("target_task") or {}).get("domain", "unknown")
        by_dom[dom].append(bool(r.get("passed")))
    return {dom: (sum(v) / len(v) if v else 0.0) for dom, v in by_dom.items()}


def aggregate_target(target_dir: Path) -> dict:
    rollouts = _load_rollout_files(target_dir)
    method_b = [r for r in rollouts if r.get("method") == "B"]
    method_a = [r for r in rollouts if r.get("method") == "A"]

    # Method B headlines: # passing / # total rollouts.
    rep_rate_b = (sum(1 for r in method_b if r.get("passed")) / max(1, len(method_b))) if method_b else 0.0
    pd_b = _per_domain_rate(method_b)

    # Per-task rep rate (#passing / B) for closure-ratio computation.
    # Group method_b rollouts by (domain, task_id, seed).
    per_task_passing: dict[tuple, list[bool]] = defaultdict(list)
    per_task_rep_pinned: dict[tuple, float] = {}
    for r in method_b:
        tt = r.get("target_task") or {}
        key = (tt.get("domain"), tt.get("task_id"), tt.get("seed"))
        per_task_passing[key].append(bool(r.get("passed")))
        if tt.get("rep_rate_pinned"):
            per_task_rep_pinned[key] = float(tt["rep_rate_pinned"])
    per_task_rates = {k: (sum(v) / len(v)) for k, v in per_task_passing.items()}
    closure_ratios = []
    for k, rate in per_task_rates.items():
        pinned = per_task_rep_pinned.get(k, 0.0)
        if pinned > 0:
            closure_ratios.append(rate / pinned)

    # Method A headlines
    pass_rate_a = (sum(1 for r in method_a if r.get("passed")) / max(1, len(method_a))) if method_a else 0.0
    pd_a = _per_domain_rate(method_a)

    # Diagnostics on failed Method B rollouts
    failed_b = [r for r in method_b if not r.get("passed")]
    depths_in_gold = [
        (r.get("diagnostics") or {}).get("depth_in_gold_actions")
        for r in failed_b
    ]
    depths_in_gold = [int(x) for x in depths_in_gold if isinstance(x, int)]
    mean_depth_failed = mean(depths_in_gold) if depths_in_gold else 0.0

    # Steps used in Method A
    a_steps = [(r.get("student_outcome") or {}).get("agent_steps_in_rollout") or 0 for r in method_a]
    mean_a_steps = mean(a_steps) if a_steps else 0.0

    # Cost + token totals
    total_cost = 0.0
    total_calls = 0
    for r in rollouts:
        c = r.get("cost_usd") or {}
        total_cost += float(c.get("total_estimate") or 0.0)
        total_calls += int(c.get("n_llm_calls") or 0)

    n_chosen_b = len({(r["target_task"]["domain"], r["target_task"]["task_id"], r["target_task"]["seed"])
                       for r in method_b})
    n_chosen_a = len(method_a)
    pinned_mean = mean(per_task_rep_pinned.values()) if per_task_rep_pinned else 0.0
    closure_ratio_overall = mean(closure_ratios) if closure_ratios else 0.0

    return {
        "n_rollouts_method_b": len(method_b),
        "n_rollouts_method_a": len(method_a),
        "n_chosen_b_unique": n_chosen_b,
        "n_chosen_a_unique": n_chosen_a,
        "student_rep_rate": rep_rate_b,
        "student_rep_rate_airline": pd_b.get("airline", 0.0),
        "student_rep_rate_retail":  pd_b.get("retail", 0.0),
        "student_rep_rate_telecom": pd_b.get("telecom", 0.0),
        "rep_rate_pinned_chosen_mean": pinned_mean,
        "closure_ratio_per_task_mean": closure_ratio_overall,
        "pass_rate_e2e": pass_rate_a,
        "pass_rate_e2e_airline": pd_a.get("airline", 0.0),
        "pass_rate_e2e_retail":  pd_a.get("retail", 0.0),
        "pass_rate_e2e_telecom": pd_a.get("telecom", 0.0),
        "mean_depth_in_gold_actions_failed_b": mean_depth_failed,
        "mean_steps_used_e2e": mean_a_steps,
        "total_cost_usd": total_cost,
        "n_llm_calls": total_calls,
    }


def discover_targets(bundle_root: Path) -> list[str]:
    root = bundle_root / "step_budget_outputs"
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir() and (p / "raw_eval").exists()])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle-root", type=Path, default=Path(os.environ.get("BUNDLE_ROOT", ".")))
    args = ap.parse_args(argv)
    bundle_root = args.bundle_root.resolve()
    outputs_root = bundle_root / "step_budget_outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)

    targets = discover_targets(bundle_root)
    if not targets:
        print("No targets with rollouts found under step_budget_outputs/.", file=sys.stderr)
        return 1
    print(f"[summarize] aggregating {len(targets)} targets")

    rows = []
    for t in targets:
        target_dir = outputs_root / t
        agg = aggregate_target(target_dir)
        # Read progress.json for seed_policy + total counts
        prog_path = target_dir / "progress.json"
        seed_policy = ""
        if prog_path.exists():
            try:
                prog = json.loads(prog_path.read_text())
                seed_policy = prog.get("seed_policy") or ""
            except Exception:
                pass
        row = {
            "target_name": t,
            "target_kind": detect_kind(t),
            "model_family": detect_family(t),
            "n_train_rows": find_n_train_rows(bundle_root, t) or 0,
            "seed_policy_last_run": seed_policy,
            **agg,
        }
        rows.append(row)

    # Write CSV
    csv_path = outputs_root / "SUMMARY_step_budget.csv"
    cols = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[summarize] wrote {csv_path} ({len(rows)} rows)")

    # Also dump a human-readable JSON view
    json_path = outputs_root / "SUMMARY_step_budget.json"
    json_path.write_text(json.dumps(rows, indent=2))
    print(f"[summarize] wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
