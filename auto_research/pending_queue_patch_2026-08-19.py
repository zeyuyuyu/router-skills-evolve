#!/usr/bin/env python3
"""Apply the 2026-08-19 daily ideas to auto_research/state.json on the A800 server.

Run on the A800:
    python3 /data0/home/zeyuwang/auto_research/pending_queue_patch_2026-08-19.py

The SSH connection from the cloud scheduler was blocked by egress policy on 2026-08-19,
so this patch is applied manually.
"""
import json, os, tempfile, shutil

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_08_19_001_grpo_continual_replay",
        "priority": 7,
        "kind": "grpo_continual",
        "gpu": "auto",
        "rationale": (
            "arXiv 2607.04364 shows GRPO still causes catastrophic forgetting in continual "
            "settings. Our cycle-3 data shows 92.68% train routing acc vs 60.98% test — a gap "
            "that suggests the small model forgets unseen task distribution. Continuing from "
            "cycle-3 GRPO adapter with a 20% replay buffer of successful earlier traces tests "
            "whether replay mitigates test-split degradation."
        ),
        "spec": {
            "base_model": "05_qwen3_5_4b_273",
            "from_checkpoint": "results/e2e_4cyc_gpt55/cycle_3/grpo_adapter",
            "bench": "humaneval",
            "n_cycles": 2,
            "grpo_temperature": 0.9,
            "grpo_batch_size": 1,
            "replay_fraction": 0.2,
            "n_generations": 8,
            "algo": "grpo",
        },
    },
    {
        "id": "exp_2026_08_19_002_grpo_curriculum_hard_first",
        "priority": 8,
        "kind": "grpo_curriculum_continual",
        "gpu": "auto",
        "rationale": (
            "arXiv 2605.00433 (RECRL) shows adaptive curriculum sampling — harder tasks first — "
            "improves RL data utilization. Cycle-3 shows 75.6% task_pass on train but 62.2% on "
            "test, meaning easy tasks dominate and produce zero-advantage GRPO groups. Sorting "
            "HumanEval by per-task pass@1 (hardest first) should push training on the model's "
            "failure tail."
        ),
        "spec": {
            "base_model": "05_qwen3_5_4b_273",
            "from_checkpoint": "results/e2e_4cyc_gpt55/cycle_3/grpo_adapter",
            "bench": "humaneval",
            "curriculum": "hard_first",
            "difficulty_metric": "cycle3_pass_rate",
            "n_cycles": 2,
            "grpo_temperature": 0.9,
            "grpo_batch_size": 1,
            "n_generations": 8,
        },
    },
    {
        "id": "exp_2026_08_19_003_forgetting_eval_cycle3_mbpp_xbench",
        "priority": 6,
        "kind": "forgetting_eval",
        "gpu": "auto",
        "rationale": (
            "Before designing further GRPO runs we need to pinpoint where forgetting occurs: "
            "base -> SFT -> GRPO checkpoint, evaluated on both HumanEval (in-domain) and MBPP "
            "(out-of-domain). Aligns with arXiv 2607.04364 methodology for measuring forgetting "
            "curves. Runtime ~1h (eval-only, no training)."
        ),
        "spec": {
            "base_model": "05_qwen3_5_4b_273",
            "checkpoint_sequence": [
                "base",
                "results/e2e_4cyc_gpt55/cycle_3/llm_adapter/checkpoint-best",
                "results/e2e_4cyc_gpt55/cycle_3/grpo_adapter",
            ],
            "eval_benches": ["humaneval", "mbpp"],
            "n_eval": 100,
            "temperature": 0.0,
        },
    },
    {
        "id": "exp_2026_08_19_004_grpo_staircase_3seeds_c3_costtarget25",
        "priority": 7,
        "kind": "grpo_multi_seed_staircase",
        "gpu": "auto",
        "rationale": (
            "Cycle-3 cost vs large is 27.56%. Test if pushing router threshold from 0.5 to 0.4 "
            "with multi-seed staircase GRPO (seed 42: hardest deciles 8-10 only; seed 123: "
            "deciles 5-10; seed 777: all tasks) reduces cost below 25% while keeping task_pass "
            ">=90%. Motivated by arXiv 2605.00433 showing curriculum scope affects GRPO "
            "convergence stability."
        ),
        "spec": {
            "base_model": "05_qwen3_5_4b_273",
            "from_checkpoint": "results/e2e_4cyc_gpt55/cycle_3/grpo_adapter",
            "bench": "humaneval",
            "seeds": [42, 123, 777],
            "staircase_difficulty_deciles": [[8, 10], [5, 10], [1, 10]],
            "router_threshold": 0.4,
            "cost_target": 0.25,
            "grpo_temperature": 0.9,
            "grpo_batch_size": 1,
            "n_generations": 8,
        },
    },
    {
        "id": "exp_2026_08_19_005_joint_35b_humaneval_2seed",
        "priority": 9,
        "kind": "joint_cycle_multiseed",
        "gpu": "auto",
        "rationale": (
            "The 35B model (Qwen3.6-35B-A3B) was only tested on tau2_bench; cwy_35b_joint shows "
            "router regression (93.2%->89.2%) and task pass degradation over cycles. Running on "
            "humaneval gives cleaner pytest reward signal and tests whether the 35B model + "
            "router achieves <20% cost vs large (target: 72B equiv budget). arXiv 2508.12491 "
            "(Cost-Aware Contrastive Routing) motivates tighter cost objectives. Two seeds give "
            "variance estimate. SKIP_GRPO=1 keeps runtime under 4h."
        ),
        "spec": {
            "base_model": "08_qwen3_6_35b_a3b_273",
            "bench": "humaneval",
            "n_cycles": 2,
            "seeds": [42, 99],
            "schedule": "SLR",
            "skip_grpo": True,
            "scaling_force_both": True,
            "sft_include_success": True,
        },
    },
]

def main():
    with open(STATE_PATH) as f:
        state = json.load(f)

    existing_ids = {e["id"] for e in state.get("queue", [])} | {
        e["id"] for e in state.get("history", [])
    }

    added = []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] in existing_ids:
            print(f"SKIP (duplicate): {exp['id']}")
            continue
        state.setdefault("queue", []).append(exp)
        added.append(exp["id"])
        print(f"QUEUED: {exp['id']} (priority={exp['priority']}, kind={exp['kind']})")

    # Atomic write
    dir_ = os.path.dirname(STATE_PATH)
    with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, suffix=".tmp") as tf:
        json.dump(state, tf, indent=2)
        tmp_path = tf.name
    shutil.move(tmp_path, STATE_PATH)

    print(f"\nDONE: queued {len(added)} new experiments")
    if not added:
        print("(all were already present)")


if __name__ == "__main__":
    main()
