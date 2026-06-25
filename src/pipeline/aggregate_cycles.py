#!/usr/bin/env python3
"""Phase 6: aggregate cycle summaries into a final markdown table + iteration curve.

Reads:
    results/$EXPERIMENT_NAME/cycle_0/e2e_ablation_summary.json
    results/$EXPERIMENT_NAME/cycle_1/e2e_ablation_summary.json
    ...

Writes:
    results/$EXPERIMENT_NAME/final_ablation_table.md
    results/$EXPERIMENT_NAME/curve.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Single global skill → no signature routing, so there is no separate
# "base" (always-small, no-procedure) arm. The "skills" arm is always-small
# WITH the distilled procedure and doubles as the small-only baseline.
VARIANT_ORDER = ["large", "skills", "router", "full"]
VARIANT_LABEL = {
    "large":  "Always-large",
    "skills": "Small + Skills (always-small + procedure)",
    "router": "+ Router training",
    "full":   "Full (+ LLM training)",
}


def _safe_get(d: dict, *path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_cycles(exp_dir: Path, n_cycles: int) -> list[dict]:
    cycles = []
    for c in range(n_cycles):
        p = exp_dir / f"cycle_{c}" / "e2e_ablation_summary.json"
        if not p.exists():
            print(f"[aggregate] WARN missing {p}", file=sys.stderr)
            continue
        with p.open() as fh:
            cycles.append(json.load(fh))
    return cycles


def load_heldout(exp_dir: Path) -> dict | None:
    """Held-out TEST split four-arm result (from phase6_heldout_eval), if present."""
    p = exp_dir / "heldout_eval" / "e2e_ablation_summary.json"
    if not p.exists():
        return None
    with p.open() as fh:
        return json.load(fh)


def _four_arm_table(summary: dict) -> list[str]:
    rows = ["| System variant | Routing Acc | Large F1 | Fallback | Cost vs Always-Large | Task Pass |",
            "|---|---:|---:|---:|---:|---:|"]
    for v in VARIANT_ORDER:
        m = _safe_get(summary, "variants", v, default={})
        rows.append(
            f"| {VARIANT_LABEL[v]} "
            f"| {m.get('routing_acc', 0):.2%} "
            f"| {m.get('large_f1', 0):.2%} "
            f"| {m.get('fallback', 0):.2%} "
            f"| {m.get('cost_vs_large', 0):.2%} "
            f"| {m.get('task_pass', 0):.2%} |"
        )
    return rows


def emit_markdown(cycles: list[dict], out: Path, heldout: dict | None = None) -> None:
    if not cycles:
        out.write_text("# No cycle data found.\n")
        return

    lines = ["# Final ablation table", "",
             f"Cycles: **{len(cycles)}**  |  "
             f"Bench: **{cycles[0].get('bench', 'unknown')}**  |  "
             f"Model: **{cycles[0].get('model_config', 'unknown')}**  |  "
             f"Schedule: **{cycles[0].get('schedule', 'SLR')}**",
             "",
             "Results are split into the **training split** (used to evolve "
             "skills/router/adapter; routing is evaluated in-sample so it is "
             "optimistic) and the **held-out test split** (the honest "
             "generalization estimate).",
             ""]

    # ── TRAINING split ──────────────────────────────────────────────────────
    lines += [f"## Training split — final cycle (cycle {len(cycles)-1})", ""]
    lines += _four_arm_table(cycles[-1])

    lines += ["", "## Training split — per-cycle progression (Full variant)", "",
              "| Cycle | Routing Acc | Fallback | Task Pass | Cost vs Large |",
              "|---:|---:|---:|---:|---:|"]
    for i, c in enumerate(cycles):
        m = _safe_get(c, "variants", "full", default={})
        lines.append(
            f"| {i} "
            f"| {m.get('routing_acc', 0):.2%} "
            f"| {m.get('fallback', 0):.2%} "
            f"| {m.get('task_pass', 0):.2%} "
            f"| {m.get('cost_vs_large', 0):.2%} |"
        )

    # ── Held-out TEST split ─────────────────────────────────────────────────
    lines += ["", "## Held-out TEST split — final cycle (router trained on train, evaluated here)", ""]
    if heldout is not None:
        lines += _four_arm_table(heldout)
    else:
        lines += ["_No held-out eval found (run with `RUN_HELDOUT_EVAL=1`)._"]

    lines += ["", "## Baseline reference (HumanEval × 1.5B, main branch 2026-05-09)", "",
              "| System | Routing Acc | Task Pass |",
              "|---|---:|---:|",
              "| Base | 68.28%| 47%|",
              "| + Skills | 69.46%| 47%|",
              "| + Router | **93.04%**  | 47%|",
              "| Full | **93.04%**  | **49%**  |",
              ""]
    out.write_text("\n".join(lines))


def emit_plot(cycles: list[dict], out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[aggregate] matplotlib not installed, skipping plot", file=sys.stderr)
        return
    if not cycles:
        return
    x = list(range(len(cycles)))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for v in VARIANT_ORDER:
        acc = [_safe_get(c, "variants", v, "routing_acc", default=0) for c in cycles]
        passes = [_safe_get(c, "variants", v, "task_pass", default=0) for c in cycles]
        ax1.plot(x, acc, marker="o", label=VARIANT_LABEL[v])
        ax2.plot(x, passes, marker="o", label=VARIANT_LABEL[v])
    ax1.set_ylabel("Routing accuracy")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="lower right")
    ax2.set_ylabel("Task pass rate")
    ax2.set_xlabel("Cycle")
    ax2.grid(True, alpha=0.3)
    fig.suptitle(f"Multi-cycle iteration  |  schedule={cycles[0].get('schedule', 'SLR')}")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-dir", required=True)
    ap.add_argument("--n-cycles", type=int, required=True)
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--output-png", required=True)
    args = ap.parse_args()

    exp_dir = Path(args.experiment_dir)
    cycles = load_cycles(exp_dir, args.n_cycles)
    heldout = load_heldout(exp_dir)

    md_path = Path(args.output_md)
    png_path = Path(args.output_png)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    emit_markdown(cycles, md_path, heldout)
    emit_plot(cycles, png_path)
    print(f"[aggregate] wrote {md_path}", file=sys.stderr)
    print(f"[aggregate] wrote {png_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
