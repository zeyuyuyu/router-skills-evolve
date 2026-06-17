import json
from pathlib import Path

from training.orchestration.summarize import collect_run_results, write_summary_csv


def test_collect_run_results(tmp_path: Path):
    rid_dir = tmp_path / "01_test"
    rid_dir.mkdir()
    (rid_dir / "training_log.json").write_text(json.dumps({
        "run_id": "01_test",
        "metrics": {"train_loss": 1.5, "eval_loss": 1.7, "epoch": 5.0},
        "model_name": "Qwen/Qwen3.5-2B",
        "n_train_rows": 6413, "n_runs_kept": 273,
    }))
    (rid_dir / "eval_results.json").write_text(json.dumps({
        "n_rollouts": 35, "pass_rate": 0.42,
        "replacement_rate_k_mean": 0.65,
        "total_task_cost_usd_mean": 0.005,
    }))
    rows = collect_run_results(tmp_path)
    assert len(rows) == 1
    assert rows[0]["run_id"] == "01_test"
    assert rows[0]["pass_rate"] == 0.42
    assert rows[0]["model_name"] == "Qwen/Qwen3.5-2B"


def test_write_csv(tmp_path: Path):
    rid_dir = tmp_path / "x"
    rid_dir.mkdir()
    (rid_dir / "training_log.json").write_text(json.dumps({
        "run_id": "x", "metrics": {}, "model_name": "?", "n_train_rows": 0, "n_runs_kept": 0,
    }))
    rows = collect_run_results(tmp_path)
    out = tmp_path / "SUMMARY.csv"
    write_summary_csv(rows, out)
    text = out.read_text()
    assert "run_id" in text
    assert "x" in text


def test_skips_underscore_dirs(tmp_path: Path):
    """Should skip _data_cache, plots, etc."""
    for name in ["_data_cache", "plots", "_smoke", "01_real"]:
        d = tmp_path / name
        d.mkdir()
    (tmp_path / "01_real" / "training_log.json").write_text(json.dumps({
        "run_id": "01_real", "metrics": {}, "model_name": "?",
        "n_train_rows": 0, "n_runs_kept": 273,
    }))
    rows = collect_run_results(tmp_path)
    ids = {r["run_id"] for r in rows}
    assert ids == {"01_real"}


def test_collects_eval_held_and_seed301(tmp_path: Path):
    rd = tmp_path / "01_a"
    rd.mkdir()
    (rd / "training_log.json").write_text(json.dumps({
        "run_id": "01_a", "metrics": {"eval_loss": 1.0},
        "model_name": "Qwen/Qwen3.5-2B", "n_train_rows": 100, "n_runs_kept": 50,
    }))
    (rd / "eval_results.json").write_text(json.dumps({"pass_rate": 0.5, "total_task_cost_usd_mean": 0.01}))
    (rd / "eval_results_heldout.json").write_text(json.dumps({"pass_rate": 0.4, "total_task_cost_usd_mean": 0.011}))
    (rd / "eval_results_seed301.json").write_text(json.dumps({"pass_rate": 0.55}))
    rows = collect_run_results(tmp_path)
    r = rows[0]
    assert r["pass_rate"] == 0.5
    assert r["pass_rate_heldout"] == 0.4
    assert r["pass_rate_seed301"] == 0.55


def test_per_domain_expansion(tmp_path: Path):
    """The harness emits per_domain.{airline,retail,telecom}; summarize must
    expand it into per-(metric, domain) columns so plotting can break it
    down. Also confirms heldout per-domain expansion is suffixed correctly."""
    rd = tmp_path / "02_pd"
    rd.mkdir()
    (rd / "training_log.json").write_text(json.dumps({
        "run_id": "02_pd", "metrics": {}, "model_name": "Qwen/Qwen3.5-2B",
        "n_train_rows": 0, "n_runs_kept": 0,
    }))
    (rd / "eval_results.json").write_text(json.dumps({
        "n_rollouts": 30, "pass_rate": 0.5, "replacement_rate_k_mean": 1.0,
        "total_task_cost_usd_mean": 0.012,
        "per_domain": {
            "airline": {"n_rollouts": 9, "pass_rate": 0.7, "total_task_cost_usd_mean": 0.010},
            "retail":  {"n_rollouts": 13, "pass_rate": 0.4, "total_task_cost_usd_mean": 0.012},
            "telecom": {"n_rollouts": 8, "pass_rate": 0.3, "total_task_cost_usd_mean": 0.014},
        },
    }))
    (rd / "eval_results_heldout.json").write_text(json.dumps({
        "n_rollouts": 15, "pass_rate": 0.3,
        "per_domain": {
            "airline": {"n_rollouts": 5, "pass_rate": 0.6, "total_task_cost_usd_mean": 0.011},
        },
    }))
    rows = collect_run_results(tmp_path)
    r = rows[0]
    # Main eval: per-domain columns present.
    assert r["pass_rate_airline"] == 0.7
    assert r["pass_rate_retail"] == 0.4
    assert r["pass_rate_telecom"] == 0.3
    assert r["total_task_cost_usd_mean_airline"] == 0.010
    assert r["n_rollouts_telecom"] == 8
    # Heldout per-domain columns: suffixed and don't collide with main.
    assert r["pass_rate_airline_heldout"] == 0.6
    assert r["pass_rate_airline"] != r["pass_rate_airline_heldout"]
    # Legacy columns still populated.
    assert r["pass_rate"] == 0.5
    assert r["pass_rate_heldout"] == 0.3
