"""Plot per-run + aggregate training/eval results (Spec §7.4)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


# Active params per model (used as x-axis on capacity curve).
ACTIVE_PARAMS_B = {
    "Qwen/Qwen3.5-2B": 2.0,
    "Qwen/Qwen3.5-4B": 4.0,
    "Qwen/Qwen3.5-9B": 9.0,
    "Qwen/Qwen3.6-35B-A3B": 3.0,        # MoE: 3B active per token, despite 35B total
}

# Reference points from spec (glm-4.5-air baseline).
GLM_AIR_BASELINE = {
    "pass_rate_assumed": 0.50,        # placeholder; user edits if external benchmark known
    "total_task_cost_usd": 0.0212,
}


def load_summary(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def _safe_float(s):
    try:
        return float(s) if s not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def plot_capacity_curve(rows: list[dict], out_path: Path) -> None:
    """Pass-rate vs log(active params) at full data (273 runs)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    pts = []
    for r in rows:
        if int(r.get("n_runs_kept") or 0) < 273:
            continue
        ap = ACTIVE_PARAMS_B.get(r["model_name"])
        if ap is None:
            continue
        pr = _safe_float(r.get("pass_rate"))
        cost = _safe_float(r.get("total_task_cost_usd"))
        if pr is None:
            continue
        pts.append((ap, pr, cost, r["model_name"]))
    pts.sort()
    if pts:
        xs = [p[0] for p in pts]
        ax1.plot(xs, [p[1] for p in pts], "o-", color="tab:blue", label="pass-rate")
        if any(p[2] is not None for p in pts):
            ax2.plot(xs, [p[2] for p in pts], "s--", color="tab:red", label="task cost ($)")
    ax1.axhline(GLM_AIR_BASELINE["pass_rate_assumed"], color="gray", linestyle=":", alpha=0.5,
                label="glm-4.5-air baseline")
    ax2.axhline(GLM_AIR_BASELINE["total_task_cost_usd"], color="darkred", linestyle=":", alpha=0.5)
    ax1.set_xscale("log")
    ax1.set_xlabel("active params (B)")
    ax1.set_ylabel("pass-rate", color="tab:blue")
    ax2.set_ylabel("total task cost ($)", color="tab:red")
    ax1.set_title("Capacity curve at full data (273 runs)\n[Qwen3.6-35B-A3B point: architecture-confound caveat]")
    ax1.grid(True, alpha=0.3)
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95), fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_data_scaling(rows: list[dict], out_path: Path) -> None:
    """Pass-rate vs n_runs at fixed model (4B, with 9B-50/9B-273 overlay)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    by_model: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        if r.get("model_name") not in {"Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"}:
            continue
        nrun = int(r.get("n_runs_kept") or 0)
        pr = _safe_float(r.get("pass_rate"))
        if pr is None or nrun == 0:
            continue
        by_model.setdefault(r["model_name"], []).append((nrun, pr))
    for model, pts in sorted(by_model.items()):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                label=model.replace("Qwen/", ""))
    ax.set_xscale("log")
    ax.set_xlabel("n_train_runs")
    ax.set_ylabel("pass-rate")
    ax.set_title("Data scaling at 4B (9B overlay validates shape transfer)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_cost_pareto(rows: list[dict], out_path: Path) -> None:
    """Total task cost vs pass-rate scatter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for r in rows:
        cost = _safe_float(r.get("total_task_cost_usd"))
        pr = _safe_float(r.get("pass_rate"))
        if cost is None or pr is None:
            continue
        label = f"{r['run_id']} ({r.get('n_runs_kept')}runs)"
        ax.scatter(cost, pr, s=80)
        ax.annotate(label, (cost, pr), fontsize=7, alpha=0.7,
                    xytext=(5, 5), textcoords="offset points")
    ax.axvline(GLM_AIR_BASELINE["total_task_cost_usd"], color="red", linestyle="--",
               label=f"glm-4.5-air cost (${GLM_AIR_BASELINE['total_task_cost_usd']})")
    ax.set_xlabel("total task cost ($)")
    ax.set_ylabel("pass-rate")
    ax.set_title("Cost-pass-rate Pareto")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_lr_check(rows: list[dict], out_path: Path) -> None:
    """Val loss vs LR for the 3 4B-273 LR variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = []
    lr_map = {
        "05_qwen3_5_4b_273": 2e-5,
        "09_qwen3_5_4b_273_lr1e5": 1e-5,
        "10_qwen3_5_4b_273_lr3e5": 3e-5,
    }
    for r in rows:
        lr = lr_map.get(r["run_id"])
        vl = _safe_float(r.get("eval_loss"))
        if lr is None or vl is None:
            continue
        pts.append((lr, vl, r["run_id"]))
    pts.sort()
    fig, ax = plt.subplots(figsize=(7, 4))
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-")
        for x, y, lbl in pts:
            ax.annotate(lbl.split("_")[-1], (x, y), fontsize=8,
                        xytext=(5, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("eval loss")
    ax.set_title("LR sanity check at 4B-273")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_per_run_loss_curve(run_dir: Path) -> None:
    """Read training_log.json and write loss_curves.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log_path = run_dir / "training_log.json"
    if not log_path.exists():
        return
    log = json.loads(log_path.read_text())
    history = log.get("history") or log.get("metrics", {}).get("history") or []
    if not history:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    train_steps = [h["step"] for h in history if "loss" in h]
    train_loss = [h["loss"] for h in history if "loss" in h]
    eval_steps = [h["step"] for h in history if "eval_loss" in h]
    eval_loss = [h["eval_loss"] for h in history if "eval_loss" in h]
    if train_loss:
        ax.plot(train_steps, train_loss, label="train")
    if eval_loss:
        ax.plot(eval_steps, eval_loss, label="val", marker="o")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(log.get("run_id", "?"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = run_dir / "plots" / "loss_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args(argv)
    summary_csv = args.root / "SUMMARY.csv"
    if not summary_csv.exists():
        print(f"No SUMMARY.csv at {summary_csv}; run summarize.py first.", file=sys.stderr)
        return 2
    rows = load_summary(summary_csv)

    plots_dir = args.root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_capacity_curve(rows, plots_dir / "capacity_curve.png")
    plot_data_scaling(rows, plots_dir / "data_scaling.png")
    plot_cost_pareto(rows, plots_dir / "cost_pareto.png")
    plot_lr_check(rows, plots_dir / "lr_check.png")

    for d in args.root.iterdir():
        if d.is_dir() and not d.name.startswith("_") and d.name != "plots":
            plot_per_run_loss_curve(d)

    print(f"Plots written to {plots_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
