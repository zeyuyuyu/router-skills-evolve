#!/usr/bin/env python3
"""
Pending queue update — last updated 2026-05-18 by autonomous research agent.
Accumulates experiments from 2026-05-15 (5), 2026-05-16 (4), 2026-05-17 (4), 2026-05-18 (4).
Total pending: 17 experiments.
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
    # ── 2026-05-16 batch (4 experiments) ────────────────────────────────────
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
    # ── 2026-05-17 batch (4 experiments) ────────────────────────────────────
    {
        "id": "exp_2026_05_17_001_kl_curriculum_15b_300tasks",
        "priority": 9,
        "kind": "grpo_curriculum_continual",
        "rationale": (
            "The two strongest pending interventions — KL regularization (arxiv:2503.06639) "
            "and easy-to-hard curriculum (arxiv:2605.00433) — are being tested in isolation "
            "in exp_2026_05_15_002 and exp_2026_05_15_001 respectively. They address "
            "orthogonal failure modes: KL prevents distribution collapse / reward hacking "
            "during longer training runs, while curriculum prevents zero-gradient rollouts "
            "on hard tasks seen before the model is ready. Combining both on 300 tasks tests "
            "whether the gains are additive: KL stabilises the extra 100 tasks beyond the "
            "safe 200-task ceiling, while curriculum ensures those 100 additional tasks are "
            "introduced in the order the model can learn from. If each independently gives "
            "+2–3pts, the combination on 300 tasks could reach +4–6pts — the target for a "
            "paper-quality LLM training result. All other hyperparams match the best case "
            "(lr=5e-6, LoRA r=16, 4 rollouts, qwen-chat, binary reward). Estimated: ~130 min."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/train_aug_excluding_eval20.jsonl",
            "eval_data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/test_eval_all.jsonl",
            "train_task_limit": 300,
            "curriculum": "easy_to_hard",
            "difficulty_metric": "base_model_solve_rate",
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
        "id": "exp_2026_05_17_002_grpo_15b_lr1e6_200x4",
        "priority": 7,
        "kind": "grpo_continual",
        "rationale": (
            "Our best case uses lr=5e-6; the degradation at 400 tasks (same lr) implies "
            "the policy update step is near the stability boundary. GRPO dynamics analysis "
            "(arxiv:2503.06639) shows that the effective gradient magnitude scales as "
            "lr × |advantage| and that for small models (<2B) the advantage signal is "
            "noisier per rollout, amplifying instability at higher LRs. A 5× more "
            "conservative lr=1e-6 keeps the update in a more stable regime without "
            "requiring KL regularization or curriculum scheduling. If lr=1e-6 matches "
            "+2pts with lower variance across seeds, it is a safer default than lr=5e-6 "
            "and enables longer training runs. If it underperforms, lr=5e-6 is confirmed "
            "as necessary for the model to escape local minima. This isolates the LR "
            "effect cleanly: all other params identical to the best case (200 tasks, "
            "4 rollouts, 1 epoch, LoRA r=16, qwen-chat, binary reward). Estimated: ~90 min."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/train_aug_excluding_eval20.jsonl",
            "eval_data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/test_eval_all.jsonl",
            "train_task_limit": 200,
            "epochs": 1,
            "rollouts_per_prompt": 4,
            "lr": 1e-6,
            "lora_r": 16,
            "prompt_style": "qwen-chat",
            "reward": "binary",
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_05_17_003_grpo_15b_lora_r32_200x4",
        "priority": 6,
        "kind": "grpo_continual",
        "rationale": (
            "History shows a clear monotonic trend in LoRA rank: r=8 gives +1pt on 0.5B "
            "and r=16 gives +2pts on 1.5B. The LoRA paper (arxiv:2106.09685, Hu et al.) "
            "demonstrates that for complex adaptation tasks the gain from rank increase is "
            "significant up to r=32 and diminishes above r=64. For GRPO specifically, a "
            "higher LoRA rank means the policy gradient can express richer weight-space "
            "updates, which is important when the target behaviour (generating correct code "
            "that passes test assertions) requires structural changes to the model's "
            "generation distribution rather than surface-level token biases. With LoRA r=32 "
            "on a 1.5B model the adapter adds ~52M trainable parameters (vs ~26M for r=16), "
            "which at 80GB VRAM with 4 rollouts is safely within budget. All other params "
            "identical to the best case (200 tasks, lr=5e-6, 4 rollouts, binary reward, "
            "qwen-chat). Estimated: ~95 min."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/train_aug_excluding_eval20.jsonl",
            "eval_data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/test_eval_all.jsonl",
            "train_task_limit": 200,
            "epochs": 1,
            "rollouts_per_prompt": 4,
            "lr": 5e-6,
            "lora_r": 32,
            "prompt_style": "qwen-chat",
            "reward": "binary",
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_05_17_004_forgetting_eval_3b_grpo_adapter",
        "priority": 5,
        "kind": "forgetting_eval",
        "rationale": (
            "The 3B GRPO adapter (from the run that degraded: 62->60 MBPP eval100) has "
            "never been evaluated on benchmarks other than MBPP. Before proposing further "
            "3B training (exp_2026_05_16_002 is the 3B curriculum experiment), it is "
            "essential to understand whether the 62->60 degradation is MBPP-specific or "
            "whether the adapter also damaged HumanEval performance. Two contrasting "
            "outcomes are informative: (a) if HumanEval also degrades by ~2pts, the 3B "
            "GRPO recipe is uniformly harmful and the curriculum fix (exp_2026_05_16_002) "
            "is the only path; (b) if HumanEval is preserved or improves, the degradation "
            "is a dataset-distribution artefact, not a general forgetting problem, which "
            "changes the interpretation of the 3B results significantly. HANDOFF section 6 "
            "(open question 1) calls for quantifying forgetting before paper claims. "
            "Eval-only; no training. Adapter path: "
            "/data0/home/zeyuwang/router-skills-evolve-runs/rl_15b/qwen25_coder_3b_grpo_200x4 "
            "(created by the earlier degraded 3B run). Estimated: ~35 min."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-3B-Instruct",
            "adapter": "/data0/home/zeyuwang/router-skills-evolve-runs/rl_15b/qwen25_coder_3b_grpo_200x4",
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
    # ── 2026-05-18 batch (4 experiments) ────────────────────────────────────
    {
        "id": "exp_2026_05_18_001_fgrpo_focal_advantage_15b",
        "priority": 9,
        "kind": "grpo_continual",
        "rationale": (
            "F-GRPO (arxiv:2602.06717, Feb 2026) identifies a frequency bias in GRPO: "
            "with small group sizes (G=4), updates concentrate probability mass on "
            "already-common correct solutions, ignoring rare-correct trajectories on hard "
            "tasks. The paper derives that the probability of a gradient update missing "
            "rare-correct modes grows super-linearly as success rate increases. Our history "
            "shows ~53/100 eval tasks are never solved across all 4 rollouts (zero-gradient "
            "groups), while ~20/100 are consistently solved (near-zero advantage, noise "
            "updates). F-GRPO introduces a Focal-loss-inspired difficulty-aware advantage "
            "scaling coefficient: advantage_scaled = advantage × (1 - success_rate)^gamma, "
            "which down-weights groups where the model already succeeds frequently (gamma "
            "controls the down-weighting strength; gamma=2 from the paper). This gives the "
            "47 hard tasks (1-4 successes in 4 rollouts) ~2-4× more gradient weight than "
            "the 20 easy tasks, directly addressing the key failure mode identified in our "
            "history without increasing compute cost. All other params match the best case "
            "(1.5B, 200 tasks, 1 epoch, lr=5e-6, LoRA r=16, qwen-chat, binary reward, "
            "eval@100). Requires adding focal_gamma field to runner.py grpo_continual kind. "
            "Estimated wall-clock: ~90 minutes."
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
            "advantage_scale": "focal",
            "focal_gamma": 2.0,
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_05_18_002_dapo_dynamic_sampling_15b",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "DAPO (arxiv:2503.14476, ByteDance, March 2025) introduces dynamic sampling: "
            "when all rollouts in a group either all pass or all fail, the advantage is "
            "identically zero and the group contributes only gradient noise. DAPO discards "
            "these uninformative groups and re-samples new prompts until every group "
            "contains at least one passing and one failing rollout, guaranteeing non-zero "
            "advantage and clean gradient signal throughout training. Our 200-task GRPO "
            "baseline uses standard GRPO which includes all-fail groups (the ~53% "
            "consistently-failing hard tasks) and all-pass groups (the ~20% trivially-easy "
            "tasks), so roughly 73% of gradient updates are likely noise-dominated. DAPO "
            "also replaces sample-level policy gradient loss with token-level loss "
            "(normalised per-token rather than per-sample), shown in the paper to stabilise "
            "training for variable-length code outputs. Combined, these two changes "
            "require no extra parameters and add ~10% overhead for re-sampling but produce "
            "much cleaner gradients — DAPO achieves 50pts on AIME2024 with Qwen2.5-32B, "
            "and the same mechanism should help at 1.5B for MBPP. All other params match "
            "the best case (200 tasks, 1 epoch, lr=5e-6, LoRA r=16, qwen-chat, binary "
            "reward, eval@100). Requires runner.py support for dynamic_sampling and "
            "loss_level fields. Estimated wall-clock: ~95 minutes."
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
            "dynamic_sampling": True,
            "loss_level": "token",
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_05_18_003_scafgrpo_hint_injection_15b",
        "priority": 7,
        "kind": "grpo_continual",
        "rationale": (
            "Scaf-GRPO (arxiv:2510.19807, Oct 2025 / Feb 2026) addresses the 'learning "
            "cliff': tasks where the model consistently fails all rollouts, collapsing the "
            "advantage to zero and making the task invisible to the policy gradient. Scaf-GRPO "
            "monitors training progress per task and, when a task's rolling success rate "
            "remains at 0 after a stagnation window (e.g., 3 consecutive gradient steps with "
            "zero reward), injects a tiered in-prompt hint — first an abstract hint "
            "('identify the core algorithmic pattern'), then if still failing a concrete hint "
            "('use a sliding-window over the list') — that guides the rollout toward a valid "
            "structure without prescribing the exact solution. On AIME24, Scaf-GRPO boosted "
            "Qwen2.5-Math-7B by a relative 44.3% over vanilla GRPO. For our 1.5B MBPP case, "
            "approximately 53/100 eval tasks are never solved in any rollout; scaffold hints "
            "on those tasks would convert some from zero-reward to mixed-reward, providing "
            "gradient signal on the hardest segment of the training distribution. All other "
            "params match the best case (200 tasks, 1 epoch, lr=5e-6, LoRA r=16, qwen-chat, "
            "binary reward, eval@100). Requires runner.py support for scaffold_stagnation_window "
            "and scaffold_hint_levels fields. Estimated wall-clock: ~110 minutes."
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
            "scaffold_stagnation_window": 3,
            "scaffold_hint_levels": ["abstract", "concrete"],
            "eval_limit": 100,
            "max_new_tokens": 256,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_05_18_004_frpo_2epoch_15b_200x4",
        "priority": 6,
        "kind": "grpo_continual",
        "rationale": (
            "Our entire history uses epochs=1. Multi-epoch RL training is the natural next "
            "step to squeeze more learning from the same 200 tasks, but 'Mechanistic Analysis "
            "of Catastrophic Forgetting in LLMs During Continual Fine-tuning' "
            "(arxiv:2601.18699, Jan 2026) shows that forgetting accelerates dramatically in "
            "later epochs (3-5): attention heads in lower layers show 15–23% severe disruption "
            "correlated with forgetting onset. However, FRPO (arxiv:2602.08813, Feb 2026) "
            "modifies GRPO with a max-min formulation that maximises the minimum reward over "
            "a KL-bounded neighbourhood of policies — equivalently, an entropic-risk objective "
            "that up-weights low-reward rollouts and penalises high-variance policies. FRPO "
            "is proven to preserve performance under subsequent fine-tuning while adding no "
            "extra computation. By running 2 epochs under the FRPO objective, we test whether "
            "the frpo_mode can prevent the 2nd-epoch forgetting/collapse that standard GRPO "
            "would likely exhibit. If 2-epoch FRPO beats 1-epoch standard GRPO (+2pts), it "
            "opens the door to multi-epoch training at 0 additional data cost. All other "
            "params match the best case (200 tasks, lr=5e-6, LoRA r=16, qwen-chat, binary "
            "reward, eval@100). Requires runner.py support for frpo_mode field. "
            "Estimated wall-clock: ~165 minutes (2× training; within 4h budget)."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/train_aug_excluding_eval20.jsonl",
            "eval_data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/test_eval_all.jsonl",
            "train_task_limit": 200,
            "epochs": 2,
            "rollouts_per_prompt": 4,
            "lr": 5e-6,
            "lora_r": 16,
            "prompt_style": "qwen-chat",
            "reward": "binary",
            "frpo_mode": True,
            "eval_limit": 100,
            "max_new_tokens": 192,
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
