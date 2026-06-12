#!/usr/bin/env python3
"""
Paper-pipeline queue patch — 2026-06-12 v2 (EXP-066, EXP-067).
Queued by the weekly AAAI paper pipeline in response to the tau2 held-out
evaluation finding: router hurts on held-out (routing acc 80%→66%, task
pass 80%→71%).

Run after all previous patches have been applied.
Apply order when A800 (or H200) connectivity is available:

    python3 auto_research/pending_queue_update_2026_06_12_v2.py

EXP-066 requires only CPU (tau2 router regularization sweep).
EXP-067 requires 8× H200 GPUs (tau2 agentic GRPO).
"""
import json, os, tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_06_12_003_tau2_router_regularization",
        "priority": 9,
        "kind": "forgetting_eval",
        "rationale": (
            "CRITICAL soundness fix. The tau2 fullsplit held-out evaluation (2026-06-10) "
            "revealed that the TF-IDF+LR router, trained on 178 in-distribution training "
            "examples, drops from 80% routing accuracy (always-small baseline) to 66% on "
            "the 100-task held-out split. This causes task pass to degrade from 80% to 71% "
            "while cost increases from 10% to 46% of always-large — the router actively "
            "hurts on held-out. This is the most damaging result in the current paper draft "
            "(Section 6, Table 7, paper version v2 2026-06-12). Two candidate root causes: "
            "(A) overfitting to the in-distribution TF-IDF vocabulary (tau2 prompts are "
            "multi-turn structured tool calls; held-out tasks may use novel phrasing); "
            "(B) insufficient regularization on the logistic regression classifier. This "
            "experiment tests (B) by sweeping the L2 regularization parameter C on the "
            "same training/held-out split. If C=0.01 recovers held-out routing accuracy "
            "to ≥75% (from 66%), the fix is trivial and the paper's soundness claim is "
            "restored. If no regularization value helps, the root cause is (A) — vocabulary "
            "mismatch — and we need a different feature extractor (e.g., sentence embeddings "
            "or the BERT router from the code routing track). This experiment also tests "
            "max_features truncation (500 vs 5000 default) and Platt scaling calibration "
            "as secondary mitigations. No GPU required; ~30 min on CPU."
        ),
        "spec": {
            "bench": "tau2_bench",
            "experiment_type": "router_regularization_sweep",
            "train_traces": "results/cwy_35b_fullsplit_20260610_083624/cycle_3/traces.jsonl",
            "heldout_eval": "results/cwy_35b_fullsplit_20260610_083624/heldout_eval/",
            "router_configs": [
                {"C": 1.0, "max_features": 5000, "calibration": "none", "label": "baseline"},
                {"C": 0.1, "max_features": 5000, "calibration": "none", "label": "C=0.1"},
                {"C": 0.01, "max_features": 5000, "calibration": "none", "label": "C=0.01"},
                {"C": 0.001, "max_features": 5000, "calibration": "none", "label": "C=0.001"},
                {"C": 0.1, "max_features": 500, "calibration": "none", "label": "C=0.1_small_vocab"},
                {"C": 0.1, "max_features": 5000, "calibration": "platt", "label": "C=0.1_calibrated"},
            ],
            "metrics": ["routing_accuracy", "task_pass", "cost_vs_large", "precision_large", "recall_large"],
            "eval_on": "held_out_100",
            "report_path": "results/tau2_router_reg_sweep_20260612.json",
            "gpu": "none",
        },
        "gpu": "none",
    },
    {
        "id": "exp_2026_06_12_004_tau2_agentic_grpo",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "First agentic RL experiment. The tau2 SFT track (Section 5.4, paper v2) "
            "established that SFT on 273 trajectory rows yields 0% autonomous task "
            "completion (Method A) despite 12.78% step-imitation (Method B) for the 35B "
            "model. The 0% Method A result is consistent with the SFT distribution-mismatch "
            "hypothesis: SFT trains on expert trajectories, but at inference the model must "
            "recover from its own errors, which are absent from the training distribution. "
            "GRPO with binary task-completion reward addresses this directly: the model "
            "generates complete trajectories, receives a 0/1 reward based on tau2 evaluator "
            "outcome, and learns from execution feedback. This is the direct analogue of "
            "the MBPP GRPO track (Section 5.1–5.3) but applied to multi-turn agentic tasks. "
            "The key research question is: does execution-feedback RL improve Method A from "
            "0% to >0% for Qwen3.6-35B-A3B on the tau2 178-task train split, with evaluation "
            "on the 100-task held-out split? Success criterion: Method A ≥ 5% on held-out. "
            "Even a small improvement (>0%) would validate the RL-for-agentic thesis and "
            "provide the paper's key contribution in the agentic track. Requires 8× H200. "
            "Estimated wall-clock: 4–6 hours (multi-turn rollout collection is the bottleneck: "
            "~14 turns/task × 178 tasks × 4 rollouts = ~10k LLM calls)."
        ),
        "spec": {
            "bench": "tau2_bench",
            "experiment_type": "grpo_agentic",
            "base_model": "Qwen/Qwen3.6-35B-A3B",
            "base_model_checkpoint": "experiments/tau2_stage2/code/training/train_outputs/08_qwen3_6_35b_a3b_273/checkpoint-best",
            "train_split": "results/cwy_35b_fullsplit_20260610_083624/cycle_3/traces.jsonl",
            "eval_split": "results/cwy_35b_fullsplit_20260610_083624/heldout_eval/",
            "train_task_limit": 178,
            "rollouts_per_task": 4,
            "reward": "binary_tau2_task_completion",
            "lora_r": 16,
            "lr": 5e-6,
            "epochs": 1,
            "max_turns": 20,
            "eval_method": "method_A",
            "eval_limit": 100,
            "success_criterion": "method_a_pass_rate > 0.05",
            "gpu": "8xH200",
            "estimated_hours": 5,
        },
        "gpu": "8xH200",
    },
]


def main():
    if not STATE_PATH.exists():
        print(f"ERROR: {STATE_PATH} not found. Are you on the A800 or H200?")
        return 1
    with open(STATE_PATH) as f:
        state = json.load(f)
    queue = state.setdefault("queue", [])
    existing_ids = {e["id"] for e in queue}
    existing_ids |= {e.get("id", "") for e in state.get("history", [])}
    added, skipped = [], []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] in existing_ids:
            skipped.append(exp["id"])
        else:
            queue.append(exp)
            added.append(exp["id"])
    tmp_fd, tmp_path = tempfile.mkstemp(dir=STATE_PATH.parent, suffix=".tmp", prefix="state_")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, STATE_PATH)
    except Exception:
        os.unlink(tmp_path)
        raise
    print(f"Added {len(added)} experiments:")
    for eid in added:
        print(f"  + {eid}")
    if skipped:
        print(f"Skipped {len(skipped)} (already present):")
        for eid in skipped:
            print(f"  - {eid}")
    print(f"Queue now has {len(queue)} pending experiments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
