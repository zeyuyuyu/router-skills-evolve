#!/usr/bin/env python3
"""
Pending queue update — generated 2026-05-15 by autonomous research agent.
Apply on A800 when connectivity is restored:
    python3 auto_research/pending_queue_update.py
"""

import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_05_15_001_curriculum_grpo_15b",
        "priority": 8,
        "kind": "grpo_curriculum_continual",
        "rationale": (
            "RECRL (arxiv:2605.00433, Yin et al. 2026) shows that sorting code tasks by "
            "model-specific solve difficulty and training easy-to-hard improves "
            "Qwen2.5-Coder-3B from 60.07->66.50 pass@1. Our best case is 1.5B GRPO flat "
            "47->49 (+2pts). Applying requirement-aware curriculum (sort 200 MBPP train "
            "tasks by base-model solve rate ascending, train easy first then hard) to the "
            "1.5B model should push the gain to 4-6pts. Uses same hyperparams as best case "
            "(lr=5e-6, LoRA r=16, qwen-chat) but with curriculum=easy_to_hard ordering."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/train_aug_excluding_eval20.jsonl",
            "eval_data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/test_eval_all.jsonl",
            "train_task_limit": 200,
            "curriculum": "easy_to_hard",
            "difficulty_metric": "base_model_solve_rate",
            "epochs": 1,
            "rollouts_per_prompt": 4,
            "lr": 5e-6,
            "lora_r": 16,
            "prompt_style": "qwen-chat",
            "reward": "binary",
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_05_15_002_kl_grpo_15b_300tasks",
        "priority": 9,
        "kind": "grpo_continual",
        "rationale": (
            "History shows 200-task GRPO on 1.5B gives +2pts but 400-task GRPO degrades "
            "(-1pt). The degradation likely reflects reward hacking / catastrophic forgetting "
            "without a reference constraint. GRPO analysis papers (arxiv:2503.06639, Guo et "
            "al. 2025) show that the KL penalty to a frozen reference model prevents "
            "distribution collapse in longer training runs. We try 300 tasks (between the "
            "successful 200 and failing 400) with kl_coeff=0.02 against a frozen reference "
            "copy, expecting to extend the gain safely. Same LoRA r=16, lr=5e-6 otherwise."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/train_aug_excluding_eval20.jsonl",
            "eval_data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/test_eval_all.jsonl",
            "train_task_limit": 300,
            "epochs": 1,
            "rollouts_per_prompt": 4,
            "lr": 5e-6,
            "lora_r": 16,
            "prompt_style": "qwen-chat",
            "reward": "binary",
            "kl_coeff": 0.02,
            "use_reference_model": True,
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_05_15_003_multiseed_verify_15b_200x4",
        "priority": 7,
        "kind": "grpo_multi_seed_staircase",
        "rationale": (
            "The best positive case (1.5B GRPO 200x4, 47->49/100) is from a single run. "
            "With 100-example eval, binomial std is ~5pts, so +2pts could be noise. "
            "A multi-seed experiment across 3 seeds verifies whether the gain is a reliable "
            "signal or lucky variance — critical before making paper claims. Extends the "
            "observation in results/best_llm_training_case_20260509.json which recommends "
            "verifying with fixed splits. Seeds [42, 123, 456] should run in ~3h total on "
            "one A800 GPU using the exact same hyperparams."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/train_aug_excluding_eval20.jsonl",
            "eval_data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/test_eval_all.jsonl",
            "train_task_limit": 200,
            "epochs": 1,
            "rollouts_per_prompt": 4,
            "lr": 5e-6,
            "lora_r": 16,
            "prompt_style": "qwen-chat",
            "reward": "binary",
            "seeds": [42, 123, 456],
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_05_15_004_forgetting_eval_15b_grpo_adapter",
        "priority": 6,
        "kind": "forgetting_eval",
        "rationale": (
            "The 1.5B GRPO adapter has been evaluated on MBPP only. Before scaling, we need "
            "to know whether it catastrophically forgets HumanEval or general instruction "
            "following. The HANDOFF doc (section 6, open question 1) flags this as essential "
            "before paper claims. This is a cheap eval-only run: load the GRPO adapter at "
            "/data0/home/zeyuwang/router-skills-evolve-runs/rl_15b/qwen25_coder_15b_grpo_200x4 "
            "and evaluate on (a) HumanEval 164 problems, (b) MBPP OOD subset not seen in "
            "training (tasks 200-250). Expected wall-clock: ~30 minutes."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "adapter": "/data0/home/zeyuwang/router-skills-evolve-runs/rl_15b/qwen25_coder_15b_grpo_200x4",
            "eval_sets": [
                {
                    "name": "HumanEval",
                    "data": "data/HumanEval.jsonl",
                    "limit": 164,
                    "prompt_style": "qwen-chat",
                },
                {
                    "name": "MBPP_ood",
                    "data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/test_eval_all.jsonl",
                    "limit": 50,
                    "offset": 100,
                    "prompt_style": "qwen-chat",
                },
            ],
            "max_new_tokens": 384,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_05_15_005_joint_cycle_grpo_adapter_learned_router",
        "priority": 5,
        "kind": "joint_cycle_multiseed",
        "rationale": (
            "E2E ablation shows the learned router achieves 93.04% routing accuracy. "
            "LLM GRPO adapter (1.5B, 47->49 MBPP) has only been evaluated in isolation. "
            "Combining both in a joint cycle should show compounding gains: the router "
            "correctly routes hard tasks to the large model while the trained small model "
            "handles the easy tier better than base. Cross-attention routing work "
            "(arxiv:2509.09782) shows joint training of router+LLM yields 6.6% quality "
            "improvement. We run 2 seeds as a smoke test for the joint pipeline end-to-end."
        ),
        "spec": {
            "small_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "small_adapter": "/data0/home/zeyuwang/router-skills-evolve-runs/rl_15b/qwen25_coder_15b_grpo_200x4",
            "router_model": "/data0/home/zeyuwang/router-skills-evolve-runs/learned-router-mixed-pretrained",
            "router_threshold": 0.57,
            "eval_data": "data/HumanEval.jsonl",
            "seeds": [42, 123],
            "prompt_style": "qwen-chat",
            "eval_limit": 100,
        },
        "gpu": "auto",
    },
]


def main():
    if not STATE_PATH.exists():
        print(f"ERROR: {STATE_PATH} not found. Are you on the A800?")
        return 1

    with open(STATE_PATH) as f:
        state = json.load(f)

    queue = state.setdefault("queue", [])
    existing_ids = {e["id"] for e in queue}
    existing_ids |= {e.get("id", "") for e in state.get("history", [])}

    added = []
    skipped = []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] in existing_ids:
            skipped.append(exp["id"])
            continue
        queue.append(exp)
        added.append(exp["id"])

    # Atomic write
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=STATE_PATH.parent, suffix=".tmp", prefix="state_"
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, STATE_PATH)
    except Exception:
        os.unlink(tmp_path)
        raise

    print(f"Added {len(added)} experiments to queue:")
    for eid in added:
        print(f"  + {eid}")
    if skipped:
        print(f"Skipped {len(skipped)} (already in queue/history):")
        for eid in skipped:
            print(f"  - {eid}")
    print(f"Queue now has {len(queue)} pending experiments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
