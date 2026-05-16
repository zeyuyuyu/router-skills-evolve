#!/usr/bin/env python3
"""
Pending queue update — generated 2026-05-16 by autonomous research agent.
Accumulates experiments from 2026-05-15 (5 entries) and 2026-05-16 (4 entries).
Apply on A800 when connectivity is restored:
    python3 auto_research/pending_queue_update.py
"""

import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

NEW_EXPERIMENTS = [
    # ── 2026-05-15 batch (5 experiments) ───────────────────────────────────
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
    # ── 2026-05-16 batch (4 experiments) ───────────────────────────────────
    {
        "id": "exp_2026_05_16_001_pgrpo_partial_reward_15b",
        "priority": 9,
        "kind": "grpo_continual",
        "rationale": (
            "Posterior-GRPO (arxiv:2508.05170, Aug 2025) shows that conditioning a "
            "partial reward on execution success — reward = (passing_asserts / "
            "total_asserts) if code runs without exception, else 0 — prevents reward "
            "hacking while providing denser gradient signal than binary pass/fail. "
            "Our current best case (1.5B, 200x4, binary reward) gives only +2pts. The "
            "binary signal gives zero gradient on 53/100 eval tasks that consistently "
            "fail all tests (model never finds a passing rollout). P-GRPO's partial "
            "signal provides learning signal on these hard tasks. Same architecture: "
            "1.5B, 200 tasks, 4 rollouts, lr=5e-6, LoRA r=16, qwen-chat prompt."
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
            "reward": "partial_test_count",
            "reward_condition": "execution_success",
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_05_16_002_curriculum_grpo_3b",
        "priority": 8,
        "kind": "grpo_curriculum_continual",
        "rationale": (
            "E2H Reasoner (arxiv:2506.06632, 2026) specifically demonstrates that 1.5B–3B "
            "models which fail or degrade under vanilla RL recover and surpass baseline "
            "when tasks are sorted easy-to-hard. Our 3B model (Qwen2.5-Coder-3B-Instruct) "
            "degraded from 62->60/100 (-2pts) with flat GRPO on 200 MBPP tasks. The paper "
            "attributes this to gradient collapse on hard tasks seen before the model can "
            "handle them. An easy-to-hard curriculum (sort 200 MBPP tasks by base-3B solve "
            "rate ascending) directly targets this failure mode. This is complementary to "
            "exp_2026_05_15_001 which tests curriculum on 1.5B; here we test on 3B where "
            "the degradation was larger and the benefit should be more pronounced."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-3B-Instruct",
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
        "id": "exp_2026_05_16_003_grpo_15b_8rollouts_200tasks",
        "priority": 7,
        "kind": "grpo_continual",
        "rationale": (
            "GRPO theory (arxiv:2503.06639, Guo et al.) shows that advantage normalization "
            "quality improves with group size G: with G=4 rollouts the per-task advantage "
            "estimate has high variance, especially for tasks with mixed outcomes "
            "(2 pass / 2 fail). Doubling to G=8 tightens the estimate at the cost of ~2x "
            "peak VRAM (still safe: 1.5B model with LoRA r=16 + 8 rollouts fits in 80GB). "
            "All other params identical to the best case (200 tasks, 1 epoch, lr=5e-6, "
            "LoRA r=16, qwen-chat, binary reward). If 8 rollouts gives >+4pts, this "
            "becomes the new standard rollout count; if it matches +2pts, 4 rollouts is "
            "sufficient and cheaper."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/train_aug_excluding_eval20.jsonl",
            "eval_data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/test_eval_all.jsonl",
            "train_task_limit": 200,
            "epochs": 1,
            "rollouts_per_prompt": 8,
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
        "id": "exp_2026_05_16_004_staircase_grpo_15b_100_100",
        "priority": 6,
        "kind": "grpo_multi_seed_staircase",
        "rationale": (
            "Self-Distillation Enables Continual Learning (arxiv:2601.19897, Jan 2026) "
            "demonstrates that incremental task introduction — training on a small chunk, "
            "checkpointing, then continuing on the next chunk — prevents catastrophic "
            "forgetting better than training on all tasks at once. Our history shows: "
            "200-task GRPO gives +2pts but 400-task degrades (-1pt). A staircase schedule "
            "— train on tasks 1-100, save checkpoint; load checkpoint and train on tasks "
            "101-200 — tests whether the degradation between 200 and 400 tasks is a "
            "batch-size effect or a fundamental instability. If the 2-stage staircase "
            "matches or beats flat-200 (+2pts), we can extend to 4 stages x 100 tasks "
            "to safely utilize 400 tasks. Each stage runs < 45 min; total < 2h."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/train_aug_excluding_eval20.jsonl",
            "eval_data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/test_eval_all.jsonl",
            "staircase_stages": [
                {"task_offset": 0, "task_limit": 100},
                {"task_offset": 100, "task_limit": 100},
            ],
            "epochs_per_stage": 1,
            "rollouts_per_prompt": 4,
            "lr": 5e-6,
            "lora_r": 16,
            "prompt_style": "qwen-chat",
            "reward": "binary",
            "eval_limit": 100,
            "max_new_tokens": 192,
            "seeds": [42],
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
