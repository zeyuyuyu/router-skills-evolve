"""Shared training-curve plotting for SFT (Phase 3a) and RL (Phase 3b).

Reads a HuggingFace/TRL `trainer.state.log_history` (list of per-log dicts) and
saves a stacked PNG — one panel per metric vs step — so training can be eyeballed
without a live dashboard. Designed to NEVER raise: any failure prints a warning
and returns, so it can be called unconditionally at the end of training.
"""
from __future__ import annotations

from pathlib import Path

# Metrics we know how to plot, in display order. Only those present in the
# log_history are drawn; everything else is ignored.
_METRIC_ORDER = [
    "loss",
    "reward",
    "mean_token_accuracy",
    "kl",
    "entropy",
    "grad_norm",
    "learning_rate",
    "completions/mean_length",
]


def plot_training_curves(log_history: list[dict], out_path, title: str = "",
                         reward_std_key: str = "reward_std") -> str | None:
    """Plot each numeric metric in log_history vs step; save to out_path (PNG).

    Returns the path written, or None if nothing could be plotted.
    For RL, if `reward` and `reward_std` are both present, a ±std band is shaded.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({e}); skipping curve.", flush=True)
        return None

    try:
        # Collect series keyed by metric → list of (step, value)
        def _num(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        series: dict[str, list[tuple[float, float]]] = {}
        std_series: list[tuple[float, float]] = []
        for i, entry in enumerate(log_history or []):
            step = entry.get("step", entry.get("global_step", i))
            for k, v in entry.items():
                if k in _METRIC_ORDER:
                    val = _num(v)
                    if val is not None:
                        series.setdefault(k, []).append((float(step), val))
            sv = _num(entry.get(reward_std_key))
            if sv is not None:
                std_series.append((float(entry.get("step", i)), sv))

        # Keep only metrics with ≥1 finite point and non-constant-NaN.
        present = [m for m in _METRIC_ORDER if series.get(m)]
        if not present:
            print("[plot] no plottable metrics in log_history; skipping.", flush=True)
            return None

        n = len(present)
        fig, axes = plt.subplots(n, 1, figsize=(8, 2.2 * n), squeeze=False)
        axes = axes[:, 0]
        for ax, metric in zip(axes, present):
            xs = [s for s, _ in series[metric]]
            ys = [y for _, y in series[metric]]
            ax.plot(xs, ys, marker="o", ms=3, lw=1.3, color="tab:blue")
            ax.set_ylabel(metric, fontsize=9)
            ax.grid(alpha=0.3)
            # reward ±std band
            if metric == "reward" and std_series:
                sd = dict(std_series)
                lo = [y - sd.get(s, 0) for s, y in series[metric]]
                hi = [y + sd.get(s, 0) for s, y in series[metric]]
                ax.fill_between(xs, lo, hi, alpha=0.15, color="tab:blue",
                                label="±std")
                ax.legend(fontsize=8)
            # annotate first/last for quick delta reading
            if len(ys) >= 2:
                ax.set_title(f"{ys[0]:.4g} → {ys[-1]:.4g}  (Δ={ys[-1]-ys[0]:+.4g})",
                             fontsize=8, loc="right")
        axes[-1].set_xlabel("step")
        fig.suptitle(title or "training curves", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.98))

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"[plot] saved training curve → {out_path}", flush=True)
        return str(out_path)
    except Exception as e:  # noqa: BLE001
        print(f"[plot] WARN could not plot curves: {e}", flush=True)
        return None
