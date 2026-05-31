"""Scaling-curve plots from SUMMARY_step_budget.csv.

Three plots, written to $BUNDLE_ROOT/step_budget_outputs/plots/:

  1. rep_rate_scaling.png       — student_rep_rate vs n_train_rows, one series
                                  per model family (2B/4B/9B/35B-A3B). Base
                                  models anchor each series at n_train_rows=0.
                                  Horizontal dashed line = rep_rate_pinned
                                  ceiling (data-collection's chosen replacers).

  2. closure_ratio_scaling.png — closure_ratio (= rep_rate / rep_rate_pinned)
                                  vs n_train_rows. y=1.0 is the ceiling.

  3. pass_rate_e2e_scaling.png  — Method A pass_rate vs n_train_rows. The
                                  "did the student do the WHOLE task" view.

If `--per-domain` is passed, also writes per-domain subplots.

Usage:
    python -m training.orchestration.plotting_step_budget --bundle-root $BUNDLE_ROOT
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Matplotlib only — no seaborn dep.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_FAMILY_ORDER = ["Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.6-35B-A3B"]
_FAMILY_COLORS = {
    "Qwen3.5-2B":       "#1f77b4",  # blue
    "Qwen3.5-4B":       "#2ca02c",  # green
    "Qwen3.5-9B":       "#ff7f0e",  # orange
    "Qwen3.6-35B-A3B":  "#d62728",  # red
}


def _plot_scaling(df: pd.DataFrame, *, y_col: str, title: str, ylabel: str,
                  out_path: Path, ceiling_value: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for fam in _FAMILY_ORDER:
        sub = df[df["model_family"] == fam].sort_values("n_train_rows")
        if sub.empty:
            continue
        color = _FAMILY_COLORS.get(fam, "gray")
        # base point as a star
        base_pts = sub[sub["target_kind"] == "base"]
        trained_pts = sub[sub["target_kind"] == "trained"]
        if not trained_pts.empty:
            ax.plot(trained_pts["n_train_rows"], trained_pts[y_col],
                    marker="o", linestyle="-", color=color, label=fam)
        if not base_pts.empty:
            ax.scatter(base_pts["n_train_rows"], base_pts[y_col],
                       marker="*", s=200, color=color,
                       edgecolors="black", linewidths=1.0,
                       label=f"{fam} (base)" if trained_pts.empty else None)
    if ceiling_value is not None:
        ax.axhline(ceiling_value, color="#7f7f7f", linestyle="--", linewidth=1,
                   label=f"rep_rate_pinned ceiling = {ceiling_value:.3f}")
    ax.set_xlabel("n_train_rows  (0 = untrained base)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_per_domain(df: pd.DataFrame, *, y_col_template: str, title: str,
                     ylabel: str, out_path: Path) -> None:
    domains = ["airline", "retail", "telecom"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, dom in zip(axes, domains):
        y_col = y_col_template.format(dom=dom)
        for fam in _FAMILY_ORDER:
            sub = df[df["model_family"] == fam].sort_values("n_train_rows")
            if sub.empty or y_col not in sub.columns:
                continue
            color = _FAMILY_COLORS.get(fam, "gray")
            base_pts = sub[sub["target_kind"] == "base"]
            trained_pts = sub[sub["target_kind"] == "trained"]
            if not trained_pts.empty:
                ax.plot(trained_pts["n_train_rows"], trained_pts[y_col],
                        marker="o", linestyle="-", color=color, label=fam)
            if not base_pts.empty:
                ax.scatter(base_pts["n_train_rows"], base_pts[y_col],
                           marker="*", s=160, color=color,
                           edgecolors="black", linewidths=1.0)
        ax.set_title(dom)
        ax.set_xlabel("n_train_rows")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.02, 1.05)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle-root", type=Path, default=Path(os.environ.get("BUNDLE_ROOT", ".")))
    ap.add_argument("--per-domain", action="store_true", help="Also write per-domain subplots.")
    args = ap.parse_args(argv)
    bundle_root = args.bundle_root.resolve()
    summary_csv = bundle_root / "step_budget_outputs" / "SUMMARY_step_budget.csv"
    plots_dir = bundle_root / "step_budget_outputs" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not summary_csv.exists():
        print(f"ERROR: {summary_csv} not found. Run summarize_step_budget.py first.", file=sys.stderr)
        return 1
    df = pd.read_csv(summary_csv)
    if df.empty:
        print("ERROR: SUMMARY is empty.", file=sys.stderr)
        return 1
    print(f"[plotting] loaded {len(df)} target rows")

    # Headline plots
    pinned_mean = df["rep_rate_pinned_chosen_mean"].astype(float).max()
    pinned_ref = pinned_mean if pinned_mean > 0 else None

    _plot_scaling(
        df, y_col="student_rep_rate",
        title="Student step-replacement rate vs training data (Method B)",
        ylabel="student_rep_rate",
        out_path=plots_dir / "rep_rate_scaling.png",
        ceiling_value=pinned_ref,
    )
    _plot_scaling(
        df, y_col="closure_ratio_per_task_mean",
        title="Closure ratio vs training data (per-task student_rep_rate / rep_rate_pinned)",
        ylabel="closure_ratio (1.0 = matches data-collection ceiling)",
        out_path=plots_dir / "closure_ratio_scaling.png",
        ceiling_value=1.0,
    )
    _plot_scaling(
        df, y_col="pass_rate_e2e",
        title="End-to-end pass rate vs training data (Method A supplement)",
        ylabel="pass_rate_e2e",
        out_path=plots_dir / "pass_rate_e2e_scaling.png",
    )

    if args.per_domain:
        _plot_per_domain(
            df, y_col_template="student_rep_rate_{dom}",
            title="Step-replacement rate vs training data (per domain)",
            ylabel="student_rep_rate",
            out_path=plots_dir / "rep_rate_per_domain.png",
        )
        _plot_per_domain(
            df, y_col_template="pass_rate_e2e_{dom}",
            title="End-to-end pass rate vs training data (per domain)",
            ylabel="pass_rate_e2e",
            out_path=plots_dir / "pass_rate_e2e_per_domain.png",
        )

    print(f"[plotting] wrote plots under {plots_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
