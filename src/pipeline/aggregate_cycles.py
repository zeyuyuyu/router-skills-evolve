#!/usr/bin/env python3
"""Phase 6: aggregate per-cycle deployed eval + final ablation into a paper-ready
markdown table + iteration curve, SPLIT BY DATASET (HumanEval vs MBPP).

Reads trace files directly (not just summaries) so it can split by task_id:
  * per-cycle DEPLOYED cascade  ← results/$EXP/cycle_{c}/heldout/traces.jsonl
        pass = policy_final_success; cost = route=large→1.0,
        route=small→0.1 (+1.0 if it fell back to large, i.e. not large_skipped).
  * final four-arm (force-both)  ← results/$EXP/heldout_eval/traces.jsonl
        always-large = mean(large_success); always-small+skills = mean(small_success);
        pre-router (no fallback) = per-task routed model's success, route-only cost.
        The cascade column is taken from the LAST cycle's deployed traces (the
        force-both file runs large on every task, so its fallback cost is not a
        real deployment cost).

Writes:  results/$EXP/final_ablation_table.md , results/$EXP/curve.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COST_SMALL, COST_LARGE = 0.1, 1.0


def _dataset(task_id: str) -> str:
    return "HumanEval" if str(task_id).startswith("HumanEval") else "MBPP"


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _by_dataset(rows: list[dict]) -> dict[str, list[dict]]:
    d: dict[str, list[dict]] = {"HumanEval": [], "MBPP": []}
    for r in rows:
        d[_dataset(r.get("task_id", ""))].append(r)
    return d


def cascade_stats(rows: list[dict]) -> dict:
    """Deployed cascade: router decides; small failure falls back to large."""
    n = len(rows)
    if not n:
        return {"pass": None, "cost": None, "n": 0, "route_large": 0, "fallback": 0}
    succ = sum(1 for r in rows if r.get("policy_final_success"))
    cost = 0.0
    route_large = fallback = 0
    for r in rows:
        if r.get("policy_route") == "large":
            cost += COST_LARGE
            route_large += 1
        else:
            cost += COST_SMALL
            if not r.get("large_skipped", True):
                cost += COST_LARGE
                fallback += 1
    return {"pass": succ / n, "cost": cost / n, "n": n,
            "route_large": route_large, "fallback": fallback}


def fourarm_stats(rows: list[dict]) -> dict:
    """From the force-both ablation traces: always-large / always-small+skills /
    pre-router (no fallback). Cascade is supplied separately (deployed)."""
    n = len(rows)
    if not n:
        return {}
    large = sum(1 for r in rows if r.get("large_success")) / n
    small = sum(1 for r in rows if r.get("small_success")) / n
    pr = prc = 0.0
    for r in rows:
        if r.get("policy_route") == "large":
            pr += 1 if r.get("large_success") else 0
            prc += COST_LARGE
        else:
            pr += 1 if r.get("small_success") else 0
            prc += COST_SMALL
    return {"large_pass": large, "small_pass": small,
            "prerouter_pass": pr / n, "prerouter_cost": prc / n, "n": n}


def _pct(x):
    return f"{x:.1%}" if isinstance(x, (int, float)) else "—"


def emit_markdown(exp_dir: Path, n_cycles: int, out: Path) -> None:
    cyc_traces = [_load(exp_dir / f"cycle_{c}" / "heldout" / "traces.jsonl")
                  for c in range(n_cycles)]
    final = _load(exp_dir / "heldout_eval" / "traces.jsonl")
    last = next((c for c in range(n_cycles - 1, -1, -1) if cyc_traces[c]), None)

    lines = ["# Final ablation table — HumanEval + MBPP (held-out test), split by dataset",
             "",
             "Per-cycle = **deployed end-to-end system** (learned router + that cycle's "
             "SFT adapter; the large model runs only on the router's large picks and as "
             "a fallback when the routed-small model fails). Cost is normalized to "
             "always-large (small:large = 1:10).", ""]

    for dname, ntag in (("HumanEval", "82"), ("MBPP", "500")):
        lines += [f"## {dname} (held-out test ≈ {ntag} tasks)", "",
                  "### Per-cycle deployed cascade",
                  "| Cycle | Task pass | Cost vs large | Routed→large | Fallback |",
                  "|---:|---:|---:|---:|---:|"]
        for c in range(n_cycles):
            s = cascade_stats(_by_dataset(cyc_traces[c])[dname]) if cyc_traces[c] else {}
            lines.append(
                f"| {c} | {_pct(s.get('pass'))} | {_pct(s.get('cost'))} "
                f"| {s.get('route_large','—')} | {s.get('fallback','—')} |")

        # four-arm at the final cycle: 3 arms from force-both, cascade from deployed
        fa = fourarm_stats(_by_dataset(final)[dname]) if final else {}
        casc = cascade_stats(_by_dataset(cyc_traces[last])[dname]) if last is not None else {}
        lines += ["", "### Final four-arm (held-out test)",
                  "| System arm | Task pass | Cost vs large |",
                  "|---|---:|---:|",
                  f"| Always-large (GPT-5.5) | {_pct(fa.get('large_pass'))} | 100.0% |",
                  f"| Always-small + Skills | {_pct(fa.get('small_pass'))} | 10.0% |",
                  f"| Pre-router (no fallback) | {_pct(fa.get('prerouter_pass'))} | {_pct(fa.get('prerouter_cost'))} |",
                  f"| **Cascade (router + fallback)** | **{_pct(casc.get('pass'))}** | **{_pct(casc.get('cost'))}** |",
                  ""]

    out.write_text("\n".join(lines) + "\n")


def emit_plot(exp_dir: Path, n_cycles: int, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[aggregate] matplotlib not installed, skipping plot", file=sys.stderr)
        return
    cyc = [_load(exp_dir / f"cycle_{c}" / "heldout" / "traces.jsonl") for c in range(n_cycles)]
    x = list(range(n_cycles))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for dname, mk in (("HumanEval", "o"), ("MBPP", "s")):
        p = [cascade_stats(_by_dataset(cyc[c])[dname]).get("pass") if cyc[c] else None for c in x]
        co = [cascade_stats(_by_dataset(cyc[c])[dname]).get("cost") if cyc[c] else None for c in x]
        ax1.plot(x, p, marker=mk, label=f"{dname} cascade pass")
        ax2.plot(x, co, marker=mk, label=f"{dname} cost")
    ax1.set_ylabel("Cascade task pass"); ax1.grid(True, alpha=0.3); ax1.legend(fontsize=9)
    ax2.set_ylabel("Cost vs always-large"); ax2.set_xlabel("Cycle")
    ax2.grid(True, alpha=0.3); ax2.legend(fontsize=9)
    fig.suptitle("MERA deployed cascade by dataset (held-out test)")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-dir", required=True)
    ap.add_argument("--n-cycles", type=int, required=True)
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--output-png", required=True)
    args = ap.parse_args()
    exp_dir = Path(args.experiment_dir)
    md = Path(args.output_md); md.parent.mkdir(parents=True, exist_ok=True)
    emit_markdown(exp_dir, args.n_cycles, md)
    emit_plot(exp_dir, args.n_cycles, Path(args.output_png))
    print(f"[aggregate] wrote {md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
