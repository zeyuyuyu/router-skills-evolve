"""Tests for harness logic (no vLLM / OpenAI / tau2 subprocess actually invoked)."""
import json
from pathlib import Path

from training.eval.harness import (
    flatten_simulations_to_rollouts,
    load_eval_tasks_grouped,
    summarize_rollouts,
    summarize_with_per_domain,
)


def test_summarize_rollouts_empty():
    out = summarize_rollouts([])
    assert out["n_rollouts"] == 0
    assert out["pass_rate"] == 0.0
    assert out["replacement_rate_k_mean"] == 0.0
    assert out["total_task_cost_usd_mean"] == 0.0


def test_summarize_rollouts_basic():
    rows = [
        {"passed": True,  "replacement_rate_k": 0.8, "total_task_cost_usd": 0.05},
        {"passed": False, "replacement_rate_k": 0.6, "total_task_cost_usd": 0.04},
        {"passed": True,  "replacement_rate_k": 0.9, "total_task_cost_usd": 0.06},
    ]
    out = summarize_rollouts(rows)
    assert out["n_rollouts"] == 3
    assert abs(out["pass_rate"] - 2/3) < 1e-9
    assert abs(out["replacement_rate_k_mean"] - (0.8 + 0.6 + 0.9)/3) < 1e-9
    assert abs(out["total_task_cost_usd_mean"] - (0.05 + 0.04 + 0.06)/3) < 1e-9


def test_summarize_rollouts_missing_keys():
    """Rollouts with missing keys → defaults (0.0, False)."""
    rows = [{"passed": True}, {}, {"passed": False, "replacement_rate_k": 0.5}]
    out = summarize_rollouts(rows)
    assert out["n_rollouts"] == 3
    assert abs(out["pass_rate"] - 1/3) < 1e-9
    assert abs(out["replacement_rate_k_mean"] - 0.5/3) < 1e-9
    assert out["total_task_cost_usd_mean"] == 0.0


def test_load_eval_tasks_grouped(tmp_path: Path) -> None:
    """Tasks with same (domain, max_steps, max_errors) collapse into one group."""
    src = tmp_path / "eval.jsonl"
    src.write_text(
        "\n".join(
            json.dumps({"task_id": str(i), "domain": d, "max_steps": s, "max_errors": e})
            for i, (d, s, e) in enumerate([
                ("airline", 100, 10),
                ("airline", 100, 10),
                ("airline",  25,  1),
                ("retail",  100, 10),
                ("telecom", 400, 30),
            ])
        ) + "\n"
    )
    groups = load_eval_tasks_grouped(src)
    assert set(groups.keys()) == {
        ("airline", 100, 10),
        ("airline",  25,  1),
        ("retail",  100, 10),
        ("telecom", 400, 30),
    }
    assert len(groups[("airline", 100, 10)]) == 2
    # Order within a group must be preserved (deterministic eval invocation).
    assert groups[("airline", 100, 10)] == ["0", "1"]


def test_flatten_simulations_to_rollouts_basic() -> None:
    """tau2 Results dict → per-rollout records with passed/cost/replacement_rate_k."""
    results = {
        "simulations": [
            {
                "task_id": "0",
                "seed": 300,
                "agent_cost": 0.0,
                "user_cost": 0.013,
                "reward_info": {"reward": 1.0},
                "termination_reason": "USER_STOP",
            },
            {
                "task_id": "1",
                "seed": 300,
                "agent_cost": 0.0,
                "user_cost": 0.022,
                "reward_info": {"reward": 0.0},
                "termination_reason": "MAX_STEPS",
            },
        ]
    }
    rollouts = flatten_simulations_to_rollouts(results)
    assert len(rollouts) == 2
    assert rollouts[0]["passed"] is True
    assert rollouts[0]["replacement_rate_k"] == 1.0
    assert abs(rollouts[0]["total_task_cost_usd"] - 0.013) < 1e-9
    assert rollouts[1]["passed"] is False
    assert rollouts[1]["agent_cost_usd"] == 0.0
    assert rollouts[1]["user_cost_usd"] == 0.022


def test_flatten_simulations_handles_missing_reward_info() -> None:
    """A SimulationRun with reward_info=None → reward 0.0 → not passed."""
    results = {
        "simulations": [
            {"task_id": "x", "agent_cost": None, "user_cost": None, "reward_info": None},
        ]
    }
    rollouts = flatten_simulations_to_rollouts(results)
    assert rollouts[0]["passed"] is False
    assert rollouts[0]["total_task_cost_usd"] == 0.0


def test_summarize_with_per_domain() -> None:
    """Per-domain breakdown rolls up correctly."""
    rollouts = [
        {"task_id": "0", "passed": True,  "replacement_rate_k": 1.0, "total_task_cost_usd": 0.01},
        {"task_id": "1", "passed": False, "replacement_rate_k": 1.0, "total_task_cost_usd": 0.02},
        {"task_id": "2", "passed": True,  "replacement_rate_k": 1.0, "total_task_cost_usd": 0.03},
    ]
    task_id_to_domain = {"0": "airline", "1": "airline", "2": "retail"}
    out = summarize_with_per_domain(rollouts, task_id_to_domain)
    assert out["n_rollouts"] == 3
    assert abs(out["pass_rate"] - 2/3) < 1e-9
    assert out["per_domain"]["airline"]["n_rollouts"] == 2
    assert abs(out["per_domain"]["airline"]["pass_rate"] - 0.5) < 1e-9
    assert out["per_domain"]["retail"]["n_rollouts"] == 1
    assert out["per_domain"]["retail"]["pass_rate"] == 1.0
