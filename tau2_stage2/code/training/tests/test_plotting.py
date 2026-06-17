"""Tests for plotting module — matplotlib-free path uses load_summary + _safe_float."""
import csv
from pathlib import Path

import pytest

from training.orchestration.plotting import _safe_float, load_summary


def test_safe_float_normal():
    assert _safe_float("3.14") == 3.14
    assert _safe_float("0") == 0.0


def test_safe_float_none_like():
    assert _safe_float(None) is None
    assert _safe_float("") is None
    assert _safe_float("None") is None


def test_safe_float_invalid():
    assert _safe_float("not a number") is None


def test_load_summary(tmp_path: Path):
    csv_path = tmp_path / "SUMMARY.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "model_name", "n_runs_kept", "pass_rate"])
        writer.writeheader()
        writer.writerow({"run_id": "01_x", "model_name": "Qwen/Qwen3.5-2B", "n_runs_kept": "273", "pass_rate": "0.42"})
        writer.writerow({"run_id": "02_y", "model_name": "Qwen/Qwen3.5-4B", "n_runs_kept": "50", "pass_rate": "0.30"})
    rows = load_summary(csv_path)
    assert len(rows) == 2
    assert rows[0]["run_id"] == "01_x"
    assert rows[1]["pass_rate"] == "0.30"


def test_plot_capacity_curve_renders(tmp_path: Path):
    """If matplotlib is available, verify rendering produces a non-empty PNG."""
    matplotlib = pytest.importorskip("matplotlib")  # skips cleanly if missing
    from training.orchestration.plotting import plot_capacity_curve

    rows = [
        {"run_id": "01", "model_name": "Qwen/Qwen3.5-2B", "n_runs_kept": "273",
         "pass_rate": "0.4", "total_task_cost_usd": "0.005", "eval_loss": "1.5"},
        {"run_id": "07", "model_name": "Qwen/Qwen3.5-9B", "n_runs_kept": "273",
         "pass_rate": "0.55", "total_task_cost_usd": "0.012", "eval_loss": "1.2"},
    ]
    out_path = tmp_path / "plots" / "capacity_curve.png"
    plot_capacity_curve(rows, out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 1000
