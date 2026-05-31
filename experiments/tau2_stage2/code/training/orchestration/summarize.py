"""Aggregate per-run results into SUMMARY.csv (Spec §7.4)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _flatten_eval_into_row(row: dict, ev: dict, suffix: str) -> None:
    """Pull the harness summary into row columns under a consistent suffix.

    Top-level keys land at `pass_rate{suffix}` etc.; the harness's
    `per_domain` map is expanded into one column per (metric, domain),
    e.g. `pass_rate_airline{suffix}`. Without this expansion, summarize
    would silently drop everything tau2 returned per-domain — which is
    exactly the per-domain breakdown plotting.py needs.
    """
    row[f"pass_rate{suffix}"] = ev.get("pass_rate")
    row[f"replacement_rate_k_mean{suffix}"] = ev.get("replacement_rate_k_mean")
    row[f"total_task_cost_usd_mean{suffix}"] = ev.get("total_task_cost_usd_mean")
    row[f"n_rollouts{suffix}"] = ev.get("n_rollouts")
    for domain, dstats in (ev.get("per_domain") or {}).items():
        row[f"pass_rate_{domain}{suffix}"] = dstats.get("pass_rate")
        row[f"total_task_cost_usd_mean_{domain}{suffix}"] = dstats.get("total_task_cost_usd_mean")
        row[f"n_rollouts_{domain}{suffix}"] = dstats.get("n_rollouts")


def collect_run_results(train_outputs_root: Path) -> list[dict]:
    """Scan train_outputs/<run_id>/ for results JSONs."""
    rows: list[dict] = []
    for run_dir in sorted(train_outputs_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_") or run_dir.name == "plots":
            continue
        log_path = run_dir / "training_log.json"
        eval_path = run_dir / "eval_results.json"
        eval_held_path = run_dir / "eval_results_heldout.json"
        eval_seed301_path = run_dir / "eval_results_seed301.json"
        if not log_path.exists():
            continue

        log = json.loads(log_path.read_text())
        row: dict = {
            "run_id": log.get("run_id", run_dir.name),
            "model_name": log.get("model_name", "?"),
            "n_train_rows": log.get("n_train_rows", 0),
            "n_runs_kept": log.get("n_runs_kept", 0),
            "train_loss": log.get("metrics", {}).get("train_loss"),
            "eval_loss": log.get("metrics", {}).get("eval_loss"),
        }
        if eval_path.exists():
            ev = json.loads(eval_path.read_text())
            _flatten_eval_into_row(row, ev, suffix="")
            # Legacy column names kept for backward-compat with consumers
            # written before the per-domain expansion.
            row["pass_rate"] = ev.get("pass_rate")
            row["replacement_rate_k"] = ev.get("replacement_rate_k_mean")
            row["total_task_cost_usd"] = ev.get("total_task_cost_usd_mean")
        if eval_held_path.exists():
            evh = json.loads(eval_held_path.read_text())
            _flatten_eval_into_row(row, evh, suffix="_heldout")
            row["pass_rate_heldout"] = evh.get("pass_rate")
            row["total_task_cost_usd_heldout"] = evh.get("total_task_cost_usd_mean")
        if eval_seed301_path.exists():
            ev301 = json.loads(eval_seed301_path.read_text())
            _flatten_eval_into_row(row, ev301, suffix="_seed301")
            row["pass_rate_seed301"] = ev301.get("pass_rate")
        rows.append(row)
    return rows


def write_summary_csv(rows: list[dict], out_path: Path) -> None:
    """Write a CSV with the union of all keys as columns."""
    if not rows:
        out_path.write_text("")
        return
    cols = sorted({k for r in rows for k in r.keys()})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate per-run training/eval results into SUMMARY.csv.")
    ap.add_argument("--root", required=True, type=Path, help="train_outputs root")
    args = ap.parse_args(argv)
    rows = collect_run_results(args.root)
    out_path = args.root / "SUMMARY.csv"
    write_summary_csv(rows, out_path)
    print(f"Wrote {out_path} with {len(rows)} rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
