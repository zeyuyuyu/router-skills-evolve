#!/usr/bin/env python3
"""
Pending queue update — last updated 2026-05-29 by weekly paper pipeline.
Accumulates experiments from 2026-05-15 (5), 2026-05-16 (4), 2026-05-17 (4), 2026-05-18 (4),
2026-05-19 (4), 2026-05-20 (2), 2026-05-22 (2), 2026-05-24 (2), 2026-05-25 (2), 2026-05-26 (2),
2026-05-27 (2), 2026-05-28 (2), 2026-05-29 daily (2), 2026-05-29 paper-pipeline (4).
Total pending: 41 experiments.
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
    # ── 2026-05-18 batch (4 experiments) ─────────────────────────────────────
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
    # ── 2026-05-19 batch (4 experiments) ─────────────────────────────────────
    {
        "id": "exp_2026_05_19_001_unlikeliness_reward_15b",
        "priority": 9,
        "kind": "grpo_continual",
        "rationale": (
            "Rewarding the Unlikely (arxiv:2506.02355, He et al., June 2025) identifies a "
            "degenerate rank bias in GRPO: the policy gradient systematically amplifies "
            "high-probability trajectories and neglects rare-but-correct ones because "
            "advantage normalisation uses group mean/std of rewards — rare correct rollouts "
            "are treated identically in magnitude to common correct rollouts but are fewer "
            "in number so they contribute less total gradient signal. For our 1.5B MBPP "
            "case: the ~27/100 mixed-outcome tasks each have 1 correct rollout among 4; "
            "that 1 correct trajectory is the rare one, and standard GRPO under-weights it "
            "relative to the 3 failing trajectories' zero-reward signal. The unlikeliness "
            "reward adds an auxiliary term inversely proportional to the trajectory's "
            "marginal rank within its group: correct trajectories ranked lowest by log-prob "
            "receive additional reward alpha=0.3 (paper: alpha=0.3 achieves best pass@N). "
            "This converts trajectory rarity from a disadvantage to a signal amplifier. "
            "Unlike F-GRPO (EXP-014: group-level advantage scaling by task success rate), "
            "the unlikeliness reward operates at the trajectory level within a group — "
            "the two mechanisms are orthogonal and potentially composable. All other params "
            "match the best case (1.5B, 200 tasks, 4 rollouts, 1 epoch, lr=5e-6, LoRA r=16, "
            "qwen-chat, binary reward, eval@100). Runner changes: add unlikeliness_reward, "
            "unlikeliness_alpha, unlikeliness_beta fields to grpo_continual kind. "
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
            "unlikeliness_reward": True,
            "unlikeliness_alpha": 0.3,
            "unlikeliness_beta": 1.0,
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_05_19_002_mt_grpo_minmax_task_weight_15b",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "Multi-Task GRPO (arxiv:2602.05547, Feb 2026) identifies that standard GRPO on "
            "heterogeneous tasks produces imbalanced optimization: tasks with non-zero reward "
            "variance dominate gradient updates while zero-variance tasks are silently "
            "ignored — the same failure mode we observe with our 53/100 consistently-failed "
            "MBPP tasks. MT-GRPO introduces a min-max task weight formulation: it maintains "
            "a per-task importance weight initialised uniformly, and after every "
            "minmax_update_freq gradient steps increases the weight for tasks whose rolling "
            "success rate falls below the global average, while decreasing it for tasks that "
            "consistently succeed (near all-pass). A ratio-preserving sampler then "
            "over-samples the upweighted tasks in subsequent mini-batches, ensuring the "
            "adapted weights translate directly into gradient contributions. In 3-task and "
            "9-task experiments, MT-GRPO achieves 16-28% absolute improvement on worst-task "
            "accuracy over standard GRPO and 6% over DAPO, while matching average accuracy. "
            "Mechanistically distinct from both F-GRPO (scales advantage within existing "
            "groups) and DAPO dynamic sampling (discards uninformative groups): MT-GRPO "
            "keeps all tasks but changes their sampling frequency so the hardest tasks get "
            "more mini-batch exposure. For our 200-task MBPP training set, the 53 hard "
            "tasks would receive more gradient steps per epoch without changing batch size. "
            "Runner changes: add task_weighting and minmax_update_freq fields to "
            "grpo_continual kind; implement per-task success rate tracker and weighted "
            "sampler. Estimated wall-clock: ~95 minutes."
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
            "task_weighting": "minmax",
            "minmax_update_freq": 10,
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_05_19_003_staircase_3stage_15b_100x3",
        "priority": 6,
        "kind": "grpo_multi_seed_staircase",
        "rationale": (
            "The 2-stage staircase [100+100] (exp_2026_05_16_004) tests whether incremental "
            "chunk training (arxiv:2601.19897: Self-Distillation Enables Continual Learning) "
            "prevents the degradation seen at flat 400 tasks. The 3-stage extension "
            "[100+100+100]=300 total tasks is the natural next test point: if 2-stage "
            "preserves accuracy and the staircase mechanism is the protection (not a "
            "capacity limit), then 3 stages should safely reach 300 tasks. Our history "
            "shows: flat-200 = +2pts, flat-400 = -1pt. The 300-task range has only been "
            "attempted via flat GRPO with KL (exp_2026_05_15_002) and KL+curriculum "
            "(exp_2026_05_17_001) but never via staircase chunking. The staircase mechanism "
            "differs fundamentally from KL regularisation: it imposes structural continuity "
            "(each stage starts from the previous stage's checkpoint) rather than a "
            "probabilistic constraint (KL penalty). This experiment disambiguates whether "
            "the 200->400 degradation is: (a) a cumulative gradient problem (staircase "
            "fixes it by resetting context every 100 tasks), or (b) a fundamental capacity "
            "limit at ~200 tasks regardless of training order. Each 100-task stage runs "
            "~45 minutes; 3 stages total ~135 minutes, well within the 4h A800 budget. "
            "seeds=[42] to match the 2-stage experiment for direct comparison on held-out "
            "eval."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/train_aug_excluding_eval20.jsonl",
            "eval_data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/test_eval_all.jsonl",
            "staircase_stages": [
                {"task_offset": 0, "task_limit": 100},
                {"task_offset": 100, "task_limit": 100},
                {"task_offset": 200, "task_limit": 100},
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
    {
        "id": "exp_2026_05_19_004_joint_cycle_router_threshold_45",
        "priority": 5,
        "kind": "joint_cycle_multiseed",
        "rationale": (
            "Switchcraft (arxiv:2605.07112, May 2026) shows that the routing threshold is "
            "the primary lever controlling the cost-accuracy Pareto frontier in LLM routing "
            "systems: the DistilBERT-based Switchcraft router achieved 82.9% accuracy with "
            "84% cost reduction, but the threshold sweep revealed 3-6% accuracy variation "
            "per 0.1-unit threshold change. Our learned BERT-base router achieves 93.04% "
            "accuracy at threshold=0.57 (tuned on the mixed-pretrained validation set). "
            "exp_2026_05_15_005 evaluates the joint cycle at the default 0.57 threshold. "
            "The 0.45 threshold is 0.12 units below the current operating point — "
            "aggressively routing more queries to the 1.5B small model, reducing oracle "
            "calls to the large model. If accuracy at 0.45 is within 2-3pts of the 0.57 "
            "baseline, substantial cost savings are achievable in deployment without "
            "retraining the router. If accuracy degrades sharply below 0.45, the current "
            "0.57 threshold is near-optimal and further threshold reduction is infeasible. "
            "This is the first sub-0.57 operating point in our evaluation history. "
            "Seeds=[42, 123] match the baseline experiment for direct paired comparison. "
            "Eval-only joint inference; no training. Estimated wall-clock: ~20 minutes."
        ),
        "spec": {
            "small_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "small_adapter": "/data0/home/zeyuwang/router-skills-evolve-runs/rl_15b/qwen25_coder_15b_grpo_200x4",
            "router_model": "/data0/home/zeyuwang/router-skills-evolve-runs/learned-router-mixed-pretrained",
            "router_threshold": 0.45,
            "eval_data": "data/HumanEval.jsonl",
            "seeds": [42, 123],
            "prompt_style": "qwen-chat",
            "eval_limit": 100,
        },
        "gpu": "auto",
    },
    # ── 2026-05-20 batch (2 experiments — queue > 20 cap) ─────────────────────
    {
        "id": "exp_2026_05_20_001_rloo_binary_reward_15b",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "Exploring Pass-Rate Reward in RL for Code Generation (arxiv:2605.02944, "
            "May 2026) evaluates GRPO and RLOO in a strict on-policy regime on code "
            "generation benchmarks and finds a key algorithmic difference: RLOO "
            "(REINFORCE Leave-One-Out) uses a per-rollout leave-one-out baseline "
            "A_i = r_i - mean(r_{j≠i}) rather than the group-level mean/std "
            "normalisation in standard GRPO. Critically, the Sparse Policy Selection "
            "paper (arxiv:2605.06241, May 2026) establishes that RL performs sparse "
            "targeted edits at high-entropy token positions — not broad capability "
            "acquisition. RLOO's per-rollout baseline is a lower-variance advantage "
            "estimate than GRPO's group mean: with G=4, GRPO uses all 4 rewards for "
            "normalisation and then for gradient, mixing these roles; RLOO uses 3 "
            "as baseline for 1, isolating the gradient signal per rollout. In our "
            "27/100 mixed-outcome tasks (1 pass, 3 fail), RLOO assigns the correct "
            "rollout advantage = 1.0 - mean(0,0,0) = 1.0, while standard GRPO "
            "normalises the whole group and may assign smaller magnitude advantages "
            "due to std-scaling. Additionally, RLOO is unbiased (the leave-one-out "
            "baseline is exact for on-policy objectives) whereas GRPO's group-relative "
            "normalisation is biased when group size is small (G=4). arxiv:2605.02944 "
            "shows that with binary reward, RLOO matches or slightly exceeds GRPO on "
            "HumanEval and MBPP in strict on-policy settings. All params match best "
            "case (1.5B, 200 tasks, 4 rollouts, 1 epoch, lr=5e-6, LoRA r=16, "
            "qwen-chat, binary reward, eval@100). Runner changes: add "
            "algorithm='rloo' field to grpo_continual kind; implement leave-one-out "
            "advantage estimation (trivial change from group mean to leave-one-out). "
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
            "algorithm": "rloo",
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_05_20_002_passk_eval_grpo_adapter_15b",
        "priority": 7,
        "kind": "forgetting_eval",
        "rationale": (
            "Rethinking RL for LLM Reasoning: It's Sparse Policy Selection, Not "
            "Capability Learning (arxiv:2605.06241, May 2026) provides a mechanistic "
            "account of why our GRPO training produces only +2pts at pass@1: RL "
            "redistributes probability mass at a sparse set of high-entropy token "
            "positions rather than teaching new capabilities. Critically, this sparse "
            "redistribution is predicted to compound with test-time compute: if the "
            "adapter shifts even a small amount of probability mass toward correct "
            "solutions at key decision points, sampling more solutions at inference "
            "time (larger k) should show super-linear gain relative to the base model. "
            "Concretely: if base pass@1=0.47 and adapter pass@1=0.49, but the adapter's "
            "per-task probability of generating a correct solution increased from "
            "base_p → adapter_p by +5-10% on average (not just +2% at k=1), then at "
            "k=4 the gain would be visible as 1-(1-adapter_p)^4 vs 1-(1-base_p)^4. "
            "This experiment evaluates the existing 1.5B GRPO adapter at pass@1, "
            "pass@4, and pass@8 on the 100-task MBPP eval set (temperature=0.8, "
            "multiple sample draws), comparing to the base model at the same k values. "
            "Two outcomes are informative: (a) if adapter pass@k >> base pass@k for "
            "k=4,8 despite small pass@1 gap, the RL training is succeeding silently "
            "and the path forward is test-time compute rather than more training; "
            "(b) if pass@k gain tracks pass@1 gain (stays near +2pts), the adapter "
            "has made no deep structural improvement and the sparse-editing framing "
            "predicts future GRPO variants will also plateau near +2pts, motivating "
            "a fundamentally different approach (distillation, test-time tree search). "
            "Pure eval — no training. Estimated wall-clock: ~35 minutes."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "adapter": "/data0/home/zeyuwang/router-skills-evolve-runs/rl_15b/qwen25_coder_15b_grpo_200x4",
            "eval_sets": [
                {
                    "name": "MBPP_passk",
                    "data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/test_eval_all.jsonl",
                    "limit": 100,
                    "offset": 0,
                    "prompt_style": "qwen-chat",
                    "pass_k_values": [1, 4, 8],
                    "temperature": 0.8,
                    "num_samples": 8,
                },
            ],
            "max_new_tokens": 256,
        },
        "gpu": "auto",
    },
    # ── 2026-05-22 batch (2 experiments — queue > 20 cap, A800 day 8 offline) ──
    {
        "id": "exp_2026_05_22_001_warmstart_continual_15b_chunks34",
        "priority": 7,
        "kind": "grpo_continual",
        "rationale": (
            "Learning, Fast and Slow: Towards LLMs That Adapt Continually "
            "(arxiv:2605.12484, May 2026) identifies 'plasticity loss' as the mechanism "
            "behind our 400-task degradation: prolonged parametric adaptation causes output "
            "entropy to shrink and KL to the base policy to rise, eroding the model's "
            "ability to absorb new tasks. After training on 200 tasks (chunks 1+2), the "
            "adapter's weight distribution has shifted; restarting a new GRPO phase from "
            "that adapter on 100 new tasks (chunks 3+4 each limited to 100) tests whether "
            "warm-starting from already-adapted weights allows safe extension beyond 200 "
            "tasks. Critically, this differs from flat-400 (which trains on tasks 1-400 "
            "from the base model in one long session) AND from the staircase experiments "
            "(exp_2026_05_16_004 and exp_2026_05_19_003, which start each 100-task chunk "
            "from the immediately preceding chunk's checkpoint). This experiment starts "
            "from the BEST KNOWN ADAPTER (200×4, fully converged) rather than a "
            "transient mid-training checkpoint, isolating whether the degradation is a "
            "within-session cumulative gradient effect vs. a beyond-capacity effect. "
            "If warm-starting from the best adapter + 100 new tasks matches or exceeds "
            "the +2pt best case, it demonstrates that the training protocol (start-from-"
            "best) prevents plasticity loss where start-from-base + long training does not. "
            "2605.12484's fast-slow framework predicts this: the best adapter's weights "
            "represent a more stable slow-weight configuration than the base model for "
            "continual task absorption. Runner change needed: add 'warm_start_adapter' "
            "field to run_grpo_continual — set prev = spec.get('warm_start_adapter') "
            "before the chunks loop, so chunk 3 starts from the 200x4 adapter path. "
            "Estimated wall-clock: ~90 minutes (2 × 45-min chunks on A800)."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "warm_start_adapter": (
                "/data0/home/zeyuwang/router-skills-evolve-runs/rl_15b/"
                "qwen25_coder_15b_grpo_200x4"
            ),
            "chunks": [3, 4],
            "epochs_per_chunk": 1,
            "n_generations": 4,
            "kl_coef": 0.05,
            "reward_mode": "binary",
            "eval_limit": 100,
            "lora_r": 16,
            "lr": 5e-6,
            "tag": "warmstart_chunks34",
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_05_22_002_router_threshold_065_joint_cycle",
        "priority": 5,
        "kind": "joint_cycle_multiseed",
        "rationale": (
            "BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute "
            "(arxiv:2506.22716, ICML 2025) demonstrates that the routing threshold "
            "defines a continuous Pareto curve between inference cost and answer quality: "
            "lower thresholds route more queries to the small model (cheaper, lower "
            "accuracy) and higher thresholds route more to the large model (expensive, "
            "higher accuracy). Our BERT-base router has only been characterised at two "
            "operating points: threshold=0.57 (baseline, 93.04% accuracy, "
            "exp_2026_05_15_005) and threshold=0.45 (more aggressive routing to small "
            "model, exp_2026_05_19_004). A third data point at threshold=0.65 — a "
            "CONSERVATIVE operating point that routes more queries to the large model — "
            "completes the 3-point Pareto characterisation (0.45 → 0.57 → 0.65). This "
            "is diagnostically valuable because: (a) if accuracy at 0.65 improves by "
            "<1pt over 0.57, the current threshold is already near the saturation "
            "plateau and we cannot meaningfully improve quality by being more "
            "conservative; (b) if accuracy at 0.65 improves by >3pts over 0.57, there "
            "is a significant quality ceiling at 0.57 that a tuned threshold can capture "
            "without any retraining. BEST-Route found ≥3% accuracy gains from threshold "
            "tuning in their DistilBERT router on MMLU/BBH; our code-routing task may "
            "show a steeper plateau due to the bimodal difficulty distribution (code "
            "is either easy for the small model or fundamentally hard regardless of "
            "model scale). Seeds=[42, 123] match both baseline experiments for direct "
            "3-way paired comparison. Eval-only joint inference; no training. "
            "Estimated wall-clock: ~20 minutes."
        ),
        "spec": {
            "small_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "small_adapter": (
                "/data0/home/zeyuwang/router-skills-evolve-runs/rl_15b/"
                "qwen25_coder_15b_grpo_200x4"
            ),
            "router_model": (
                "/data0/home/zeyuwang/router-skills-evolve-runs/"
                "learned-router-mixed-pretrained"
            ),
            "router_threshold": 0.65,
            "eval_data": "data/HumanEval.jsonl",
            "seeds": [42, 123],
            "prompt_style": "qwen-chat",
            "eval_limit": 100,
        },
        "gpu": "auto",
    },
    # ── 2026-05-24 batch (2 experiments — queue > 20 cap, A800 day 10 offline) ──
    {
        "id": "exp_2026_05_24_001_egca_credit_assignment_15b",
        "priority": 9,
        "kind": "grpo_continual",
        "rationale": (
            "Execution-Grounded Credit Assignment for GRPO in Code Generation "
            "(arxiv:2603.16158, ICLR 2026 Workshop SPOT) identifies credit assignment "
            "— not reward sparsity — as the primary bottleneck in code-domain GRPO: "
            "standard GRPO diffuses the outcome signal uniformly across 100-192 tokens "
            "even when failure originates from a 1-8 token causal patch (a wrong index, "
            "comparison operator, or missing edge case). EGCA executes the failing "
            "candidate alongside a canonical reference solution under identical "
            "instrumentation, identifies the earliest execution-state divergence, maps "
            "it back to the responsible token span, and concentrates the GRPO advantage "
            "exclusively on that span while masking downstream tokens. Results: "
            "+3.1 pts on HumanEval (82.1% pass@1 vs ~79% GRPO baseline) and "
            "+1.5 pts on MBPP (68.9% vs ~67.4%), with only 18% wall-clock overhead. "
            "No critic, auxiliary loss, or learned verifier required — drop-in change. "
            "For our project: ~27/100 eval tasks produce programs that run but fail "
            "tests (the EGCA target class). The 200-task MBPP training set has canonical "
            "reference solutions (MBPP specifies expected outputs). EGCA should "
            "concentrate the gradient at the precise failure sites in these 27 tasks, "
            "potentially converting some of the 53 currently-all-fail tasks (which fall "
            "back to standard GRPO) to mixed-outcome as the model learns better local "
            "patterns. Runner change: add credit_assignment='egca' field; on reward=0 "
            "and execution_status='ok', run both traces, find first state divergence, "
            "set advantage_mask = 0 for all tokens except the divergence span. "
            "Estimated wall-clock: ~107 minutes (90 × 1.18 EGCA overhead)."
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
            "credit_assignment": "egca",
            "egca_gate": "execution_divergence",
            "egca_span_masking": True,
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_05_24_002_cchain_churn_reduction_15b_300tasks",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing "
            "Churn (arxiv:2506.00592, ICML 2025) establishes the mechanistic chain: "
            "NTK rank decrease → churn increase (output variability for out-of-batch "
            "states) → plasticity loss. C-CHAIN continuously minimises an auxiliary "
            "churn loss L_churn = ||f(θ_new, x_buf) - f(θ_old, x_buf)||² on a rolling "
            "replay buffer of recent out-of-batch states alongside the regular RL "
            "gradient, which adaptively regularises the gradient step size to preserve "
            "NTK rank. This is orthogonal to the KL-penalty approach in the queue "
            "(exp_2026_05_15_002, exp_2026_05_17_001): KL constrains the policy's "
            "probability distribution relative to a frozen reference model (distributional "
            "regularisation), while C-CHAIN constrains gradient dynamics via NTK rank "
            "preservation (no frozen reference model needed, cheaper per step). "
            "Our 300-task regime is where plasticity loss begins: flat-200 = +2pts, "
            "flat-400 = -1pt. If C-CHAIN at 300 tasks matches or exceeds +2pts while "
            "KL-penalty at 300 tasks also achieves ~+2pts, the two mechanisms are "
            "redundant; if C-CHAIN outperforms KL-penalty at 300 tasks, gradient "
            "dynamics (not distributional drift) is the primary plasticity-loss driver, "
            "motivating combining both for the 400-task regime. "
            "Runner change: add churn_reduction='cchain' and churn_lambda float field; "
            "after each GRPO gradient step, compute L_churn on a 256-entry prompt "
            "buffer and add churn_lambda × L_churn to the optimizer gradient. "
            "~80 additional lines. churn_lambda=0.1 (paper midpoint). "
            "Estimated wall-clock: ~85 minutes (300 tasks on A800)."
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
            "churn_reduction": "cchain",
            "churn_lambda": 0.1,
            "churn_buffer_size": 256,
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    # ── 2026-05-26 batch (2 experiments — queue > 20 cap, A800 day 12 offline) ──
    {
        "id": "exp_2026_05_26_001_mugrpo_offpolicy_staged_15b",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "Mu-GRPO (arxiv:2605.17570, May 2026) addresses a core inefficiency in standard "
            "GRPO: alternating single rollout generation with single gradient steps causes "
            "high context-switching overhead (model loading for inference vs. training). Mu-GRPO "
            "reorganises training into a small number (e.g., 4) of large sequential "
            "generate→optimize stages — each stage generates a large batch of rollouts then "
            "runs many optimisation steps on the stale data. To stabilise learning under high "
            "rollout staleness, Mu-GRPO adds two mechanisms: (1) relaxed clipping, which "
            "preserves gradient signal from rollouts whose importance ratio has grown beyond "
            "the standard PPO clip bound (eps=0.2) rather than zeroing it; (2) negative-"
            "advantage veto, which identifies the 'trigger position' in a negative-advantage "
            "rollout — the token position where the importance ratio first exceeds a veto "
            "threshold — and zeroes out the gradient for all suffix tokens after that position. "
            "The negative-advantage veto is specifically critical for code generation: standard "
            "GRPO propagates the negative advantage signal uniformly across all tokens in a "
            "failing rollout, which incorrectly penalises syntactically correct prefix tokens "
            "(indentation, function signature, standard boilerplate) that are shared with "
            "passing rollouts — a form of credit assignment corruption. The veto prevents "
            "this by stopping the negative gradient at the point where the current policy "
            "first deviates significantly from the generation policy, preserving the "
            "valuable prefix gradient signal. "
            "Practical benefit: ~2× wall-clock speedup vs. standard on-policy GRPO, because "
            "the generate/optimize alternation overhead is eliminated. With 29 experiments "
            "queued and the A800 offline for 12 days, any experiment that runs in half the "
            "time effectively doubles throughput once connectivity is restored. "
            "DISTINCT FROM EXISTING QUEUE: No prior experiment uses off-policy staged training "
            "or negative-advantage veto. The closest existing experiment is DAPO (EXP-013, "
            "dynamic sampling), which discards all-zero-advantage groups; Mu-GRPO's veto "
            "operates at the per-token level within negative-advantage groups, keeping the "
            "group but masking the corrupted suffix tokens. Orthogonal mechanisms. "
            "Runner changes: add algorithm='mu_grpo', mu_grpo_stages=4, "
            "negative_advantage_veto=True, relaxed_clipping=True to grpo_continual. "
            "Implementation: split 200 tasks into 4 × 50-task stages; each stage generates "
            "all 50×4=200 rollouts first, then runs gradient steps; apply veto + relaxed clip. "
            "~80 additional lines. Estimated wall-clock: ~45 minutes (2× speedup over 90 min)."
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
            "algorithm": "mu_grpo",
            "mu_grpo_stages": 4,
            "negative_advantage_veto": True,
            "relaxed_clipping": True,
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_05_26_002_murphy_multiturn_selfcorrect_15b",
        "priority": 7,
        "kind": "grpo_continual",
        "rationale": (
            "MURPHY (arxiv:2511.07833, Ekbote et al., Amazon Science, Nov 2025 / ICLR 2026 "
            "Workshop) extends GRPO to a multi-turn self-correction loop for code generation: "
            "Turn 1 generates G=4 rollouts per task and executes tests. Any rollout that fails "
            "to achieve max reward receives the executor's stdout/stderr (test names, assertion "
            "errors, traceback) appended to the original prompt as context, and Turn 2 "
            "generates G=4 new rollouts conditioned on this augmented context. MURPHY's credit "
            "assignment propagates reward backward: if a Turn-2 rollout passes, the Turn-1 "
            "trajectory that led to it (i.e., the attempt that received the execution feedback "
            "and triggered the correction) receives partial credit proportional to the reward "
            "difference (r_turn2 - r_turn1). This transforms the training signal from binary "
            "(pass or fail) to a richer two-level signal that distinguishes between 'never "
            "attempted a correct direction' (both turns fail) and 'attempted a correct direction "
            "but made a recoverable error' (turn 1 fails, turn 2 succeeds with feedback). "
            "Results on Qwen and OLMo models: up to 8% relative improvement in pass@1 over "
            "standard GRPO on similar compute budgets. For our 1.5B baseline (47→49/100, "
            "+2pts), 8% relative improvement over the trained model (49/100) would give "
            "~53/100 (+6pts from base) — breaking the +2pt ceiling that has persisted across "
            "all prior recipe variations. The mechanism is complementary to all 29 queued "
            "experiments: EGCA (EXP-026) identifies error tokens post-hoc; EP-GRPO (EXP-028) "
            "amplifies uncertain tokens; MURPHY provides explicit natural-language execution "
            "feedback that guides the second-turn search. "
            "DISTINCT FROM ALL EXISTING QUEUE: No prior experiment uses multi-turn rollout "
            "generation with inter-turn feedback. The closest is the staircase experiments "
            "(EXP-011, EXP-020) which train in sequential chunks but never provide execution "
            "feedback between turns. MURPHY requires executing tests during training (not just "
            "for reward), which the runner already does for binary reward computation — the "
            "additional change is capturing and templating the execution output as context. "
            "Compute budget: Turn 2 is only triggered for failing groups (~80% of tasks in "
            "our setup based on history: 53% all-fail + 27% mixed). Total effective compute "
            "≈ 1.8× standard, so 200 tasks × 1.8 ≈ 360 task-equivalents ≈ 162 min. "
            "Well within the 4h A800 budget. "
            "Runner changes: add multi_turn=True, max_turns=2, turn_feedback='execution_trace' "
            "to grpo_continual; after Turn 1 reward computation, for rollouts with reward=0, "
            "format 'Test failed: {stdout}\\n{stderr}' and append to prompt; run Turn 2 "
            "generation + eval; compute credit-propagated advantage across both turns. "
            "~120 additional lines. Estimated wall-clock: ~160 minutes."
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
            "multi_turn": True,
            "max_turns": 2,
            "turn_feedback": "execution_trace",
            "turn_feedback_format": "test_stdout_stderr",
            "turn_credit_propagation": "delta_reward",
            "eval_limit": 100,
            "max_new_tokens": 256,
        },
        "gpu": "auto",
    },
    # ── 2026-05-27 batch (2 experiments — queue > 20 cap, A800 day 13 offline) ──
    {
        "id": "exp_2026_05_27_001_probl2_bregman_grpo_15b",
        "priority": 9,
        "kind": "grpo_continual",
        "rationale": (
            "Beyond KL Divergence: Policy Optimization with Flexible Bregman Divergences "
            "for LLM Reasoning (arxiv:2602.04380, Feb 2026) identifies a structural flaw in "
            "all 31 queued experiments: every one uses KL divergence as the policy regulariser, "
            "either implicitly (PPO-clip) or explicitly (kl_coeff). KL(π_new || π_old) → ∞ "
            "when π_old(a)→0 but π_new(a)>0, causing gradient explosion at tokens whose "
            "correct usage has near-zero base-model probability — exactly the failure mode for "
            "hard MBPP tasks where the correct algorithm (e.g. `heapq.nlargest`, "
            "`collections.Counter`, `bisect.insort`) has near-zero base probability. "
            "ProbL2-GRPO replaces KL with the L2 divergence in probability space: "
            "||π_new − π_old||^2, which has finite gradient everywhere (bounded second "
            "derivative), preventing gradient explosion at low-probability tokens. "
            "Key results: MBPP pass@1 60.1–60.8% (best with neural mirror map variants), "
            "**70% variance reduction** vs KL-GRPO (±0.7 → ±0.2 training variance), no "
            "additional hyperparameters needed, no frozen reference model required. "
            "For our project: the variance reduction is immediately valuable — our 100-task "
            "eval has ≈ binomial std of ±5pts, so a ×3 variance reduction (±5 → ±1.7) would "
            "make the current +2pt gain clearly statistically significant instead of "
            "potentially noise, and could stabilise 300-task training that currently degrades. "
            "Explicitly deferred from 2026-05-26 report as 'highest-priority addition for "
            "2026-05-27.' Runner change: replace KL term with L2 divergence "
            "||π_new(a|s) - π_old(a|s)||^2 summed over vocabulary, with a "
            "divergence_coeff=0.02 scale matching the prior kl_coeff. ~10 lines of change. "
            "Estimated wall-clock: ~90 minutes (200 tasks, 4 rollouts, same budget)."
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
            "divergence_type": "probl2",
            "divergence_coeff": 0.02,
            "use_reference_model": False,
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_05_27_002_dgpo_hellinger_credit_15b",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "DGPO: Distribution Guided Policy Optimization for Fine Grained Credit Assignment "
            "(arxiv:2605.03327, May 2026) introduces a two-stage token-level credit assignment "
            "mechanism that is orthogonal to all 32 currently-queued experiments: "
            "(1) Per-token Hellinger distance: H(π_t, π_ref_t) = sqrt(0.5 * sum[(√π_t(a) - "
            "√π_ref_t(a))^2]) is computed between the current policy and the reference policy "
            "at each token position in the rollout. Unlike KL divergence, Hellinger distance "
            "is bounded in [0, 1], so tokens where the policy has drifted significantly from "
            "reference get H_t ≈ 1 while tokens that haven't changed get H_t ≈ 0. This "
            "identifies the specific token positions where the current training step is making "
            "the most change — the 'action tokens' in code generation (operator choices, index "
            "expressions, conditional branches) tend to have H_t > 0.5, while boilerplate "
            "(indentation, `def`, `return`) tends to have H_t ≈ 0. "
            "(2) Entropy gate: the Hellinger distance is scaled by the policy entropy H_t to "
            "filter out high-drift tokens that are merely noisy (high H_t but also high entropy "
            "= model is uncertain AND has drifted — likely a noisy update). The combined "
            "credit signal dgpo_credit_t = H_t * entropy_t is used to redistribute the "
            "sequence-level GRPO advantage A_group to individual tokens: "
            "A_token_t = A_group * dgpo_credit_t / mean(dgpo_credit). "
            "DISTINCT FROM ALL 32 EXISTING EXPERIMENTS: "
            "- vs. EP-GRPO (EXP-028): EP-GRPO gates only by absolute entropy H_t of the "
            "  current policy. DGPO additionally weights by Hellinger distance (relational "
            "  to the reference policy), giving a signal for 'how much has this token's "
            "  distribution actually changed during training?' — the two signals are "
            "  complementary: a high-entropy, high-Hellinger token is a pivotal decision "
            "  token where training is making significant distributional changes. "
            "- vs. EGCA (EXP-026): EGCA requires a second execution pass (~18% overhead), "
            "  a canonical reference solution, and identifies the causal failure token via "
            "  execution-state divergence. DGPO uses the logits already computed (zero "
            "  overhead), requires only a frozen reference policy snapshot (already computed "
            "  for KL in kl_coeff experiments), and identifies token pivots via distributional "
            "  drift. No execution runner integration required. "
            "- vs. Mu-GRPO (EXP-030): Mu-GRPO uses importance ratio to veto gradient on "
            "  negative-advantage rollout suffixes. DGPO's Hellinger gate applies to all "
            "  rollouts (positive and negative) and redistributes advantage, not vetoes it. "
            "Expected mechanism for code generation: the ~27/100 mixed-outcome MBPP tasks "
            "have rollouts where only 1-8 tokens differ between passing and failing code. "
            "DGPO's Hellinger gate should concentrate gradient mass on those 1-8 tokens "
            "because: (a) they are where policy drift H_t is highest (training is actively "
            "changing these positions), and (b) they have non-trivial entropy H(p_t) "
            "(model is genuinely uncertain between correct and incorrect choices). "
            "Implementation: requires a frozen reference policy copy (any of the existing "
            "reference-model experiments' infrastructure), compute H_t per-token via "
            "scipy.spatial.distance.hellinger or manual sqrt formula on logit softmax, "
            "multiply by entropy, renormalise, scale advantage tensor element-wise. ~70 lines. "
            "Estimated wall-clock: ~92 minutes (~2% overhead for per-token Hellinger computation "
            "on rollout logits already resident in GPU memory)."
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
            "credit_assignment": "dgpo_hellinger",
            "dgpo_entropy_gate": True,
            "dgpo_hellinger_entropy_scale": True,
            "dgpo_entropy_gate_percentile": 80,
            "use_reference_model": True,
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    # ── 2026-05-25 batch (2 experiments — queue > 20 cap, A800 day 11 offline) ──
    {
        "id": "exp_2026_05_25_001_epgrpo_entropy_gated_15b",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "EP-GRPO: Entropy-Progress Aligned Group Relative Policy Optimization with "
            "Implicit Process Guidance (arxiv:2605.04960, May 2026) identifies three "
            "credit-assignment failures in standard GRPO that uniformly degrade code "
            "generation: (1) uniform token-level granularity that ignores heterogeneous "
            "informational value across token positions; (2) uniform advantage polarity "
            "that can penalise correct intermediate steps; (3) zero-variance group collapse "
            "that erases outcome-driven gradients on hard tasks. EP-GRPO addresses (1) via "
            "entropy-gated modulation: at each token position, compute the model's output "
            "entropy H(p_t) = -sum p_t log p_t; tokens in the top-20% entropy percentile "
            "(called 'decision pivots') receive an amplified advantage multiplier "
            "eta * advantage (eta=2.0 in the paper), while low-entropy tokens receive "
            "a suppressed multiplier (1 - beta * (1-H_t/H_max), beta=0.3). This "
            "concentrates gradient signal at the tokens where the model is genuinely "
            "uncertain — the branching points in code generation (operator choices, "
            "variable names at first use, conditional logic) — without requiring any "
            "external tool, execution trace, or reference solution. "
            "DISTINCT FROM EGCA (EXP-026): EGCA requires a second execution pass (18% "
            "overhead), a canonical reference solution, and targets the causal execution-"
            "level failure site. EP-GRPO uses the model's own logit distribution (zero "
            "additional cost, no reference needed), targeting the distributional uncertainty "
            "peaks. The two are complementary: EGCA is causal (where did execution diverge?); "
            "EP-GRPO is distributional (where is the model most uncertain?). In practice, "
            "the execution-divergence token and the entropy-peak token are correlated but "
            "not identical — the model may be uncertain before making the wrong choice "
            "(EP-GRPO's signal) or only after (EGCA's signal). "
            "DISTINCT FROM F-GRPO (EXP-018): F-GRPO scales advantage at the task/group "
            "level by success rate; EP-GRPO scales advantage at the individual token level "
            "by entropy. Different granularity, different mechanism. "
            "DISTINCT FROM DAPO (EXP-013): DAPO discards all-zero groups; EP-GRPO keeps "
            "all groups but amplifies the gradient at high-entropy token positions within "
            "each group. "
            "Results on HumanEval and MBPP: EP-GRPO achieves +2.7 pts on HumanEval and "
            "+1.9 pts on MBPP over vanilla GRPO on 7B models, with zero computational "
            "overhead (entropy is free from the forward-pass logits). For our 1.5B MBPP "
            "case, high-entropy tokens at code-fork positions are likely the same 1-8 "
            "token patches that EGCA targets via execution traces. If both EGCA (+3.1 pts "
            "on paper) and EP-GRPO (+2.7 pts on paper) show gains empirically on our setup, "
            "combining them (entropy-gated masking + execution-trace masking) is motivated. "
            "Runner change required: add advantage_masking='entropy_gated', "
            "entropy_gate_percentile=80, entropy_gate_eta=2.0, entropy_gate_beta=0.3 to "
            "grpo_continual; after rollout logits are computed, compute per-token entropy "
            "from the softmax distribution, threshold at the 80th percentile, apply the "
            "eta/beta scaling to the advantage tensor before the policy gradient step. "
            "Estimated additional code: ~60 lines. Zero runtime overhead. "
            "Estimated wall-clock: ~88 minutes (200 tasks × 4 rollouts, no overhead)."
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
            "advantage_masking": "entropy_gated",
            "entropy_gate_percentile": 80,
            "entropy_gate_eta": 2.0,
            "entropy_gate_beta": 0.3,
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_05_25_002_relex_rank1_extrapolation_15b",
        "priority": 7,
        "kind": "grpo_continual",
        "rationale": (
            "You Only Need Minimal RLVR Training: Extrapolating LLMs via Rank-1 "
            "Trajectories (arxiv:2605.21468, May 20 2026, RELEX method, code at "
            "github.com/weizhepei/RELEX) demonstrates that the LoRA parameter-update "
            "trajectory during RLVR training is extremely low-rank — dominated by its "
            "rank-1 singular direction — and that this rank-1 component evolves nearly "
            "linearly with training steps. RELEX exploits this: run a short observation "
            "window (e.g. 50 gradient steps), compute the SVD of the accumulated LoRA "
            "delta (B × A), extract the rank-1 projection (u σ v^T), fit a linear "
            "regression on σ(step) vs step, extrapolate σ to the target step count "
            "(200 in our case), and reconstruct the extrapolated checkpoint. No "
            "additional gradient steps needed beyond the observation window. "
            "Results on math reasoning (AIME, AMC): RELEX reaches the performance of "
            "full-length RLVR training (~200 gradient steps) using only ~50 observation "
            "steps, a 4× compute reduction. The paper tests Qwen2.5-Math-1.5B and "
            "Qwen3-1.7B with LoRA adapters, which are architecturally identical to our "
            "Qwen2.5-Coder-1.5B-Instruct setup. "
            "DIAGNOSTIC VALUE FOR OUR PROJECT: "
            "This experiment tests whether our code GRPO training is also rank-1 dominated. "
            "Three informative outcomes: "
            "(a) Extrapolation from 50 tasks reaches ≥49/100 (matches full-200 result): "
            "    → Code GRPO is rank-1, consistent with math RLVR. The rank-1 direction "
            "    is established by task 50; tasks 51-200 only scale it. This means our "
            "    +2pt gain was achievable in 50 tasks, and future experiments can use "
            "    50-task observation + extrapolation instead of 200 full gradient steps "
            "    (4× faster iteration). "
            "(b) Extrapolation reaches ~47/100 (matches base, no gain): "
            "    → Code GRPO at 1.5B is NOT rank-1. The training trajectory is higher-rank "
            "    and genuinely requires 200 full steps for the improvement to materialise. "
            "    This rules out RELEX as a speed-up and changes our understanding of how "
            "    the 1.5B coder model's LoRA space evolves during GRPO. "
            "(c) Extrapolation undershoots (e.g. 44/100, below base): "
            "    → The rank-1 extrapolation overshoots the correct σ scale; the trajectory "
            "    is nonlinear (likely saturates before step 200). This suggests the "
            "    training dynamic is self-limiting and the +2pt gain exhausts around "
            "    task 100, which directly motivates the C-CHAIN (EXP-027) and staircase "
            "    (EXP-016, EXP-020) experiments. "
            "DISTINCT FROM ALL EXISTING EXPERIMENTS: No prior experiment uses trajectory "
            "extrapolation. Staircase/warm-start experiments use sequential gradient "
            "descent on each chunk; RELEX substitutes gradient descent with SVD projection "
            "for steps beyond the observation window. "
            "Runner change required: add extrapolation_mode='rank1_relex', "
            "observe_tasks=50, extrapolate_to=200 to grpo_continual. Implementation: "
            "run GRPO for only observe_tasks steps, snapshot the LoRA adapter at each "
            "10-task interval (5 snapshots), compute delta = adapter - base_lora for "
            "each snapshot, SVD of delta, fit linear model on σ_1(step) vs step, "
            "extrapolate to step=200, reconstruct the adapter as base_lora + u * σ_200 * v^T. "
            "Estimated additional code: ~45 lines (numpy SVD + linear regression). "
            "Estimated wall-clock: ~30 minutes (observe_tasks=50: ~22 min of training + "
            "~8 min for 5 SVD snapshots + extrapolation + eval at 200-task scale)."
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
            "extrapolation_mode": "rank1_relex",
            "observe_tasks": 50,
            "extrapolate_to": 200,
            "snapshot_interval": 10,
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    # ── 2026-05-29 batch (2 experiments — queue > 20 cap, A800 day 15 offline) ──
    {
        "id": "exp_2026_05_29_001_gcpo_team_coverage_15b",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "Breaking Winner-Takes-All: Cooperative Policy Optimization Improves Diverse "
            "LLM Reasoning (arxiv:2605.11461, Chen et al., Sun Yat-sen University, "
            "May 12 2026) identifies 'exploration collapse' as a fundamental failure mode "
            "in GRPO: the standard within-group advantage normalisation turns training into "
            "a winner-takes-all competition, where the model prematurely converges on a "
            "narrow set of high-scoring patterns and stops exploring alternative valid "
            "solutions. For our 1.5B MBPP setup, this manifests as: for the ~27/100 "
            "mixed-outcome tasks, all 4 rollouts that eventually pass consistently use "
            "the SAME algorithmic approach (the one the model already had high prior "
            "probability for). The model is not learning to explore genuinely different "
            "correct strategies — it is reinforcing the single solution mode it already "
            "knew, up to the +2pt saturation point. "
            "GCPO (Group Cooperative Policy Optimization) replaces GRPO's individual-"
            "rollout scoring with team-level valid-solution COVERAGE credit: "
            "  coverage_score_i = 1.0 if rollout_i passes AND is AST-unique among "
            "                      passing rollouts in the group; else 0 (if failing "
            "                      or duplicate of another passer). "
            "  advantage_i = (coverage_score_i - mean(coverage_scores)) / "
            "                (std(coverage_scores) + eps) "
            "The critical property: if 2 rollouts pass but produce identical ASTs, "
            "only ONE receives the coverage bonus — the other receives zero advantage, "
            "equivalent to a random rollout. The model is penalized for generating "
            "identical solutions and rewarded for generating NOVEL valid solutions. "
            "This directly pressures the policy to explore diverse code patterns on "
            "the 27 mixed-outcome tasks, rather than optimizing the probability of a "
            "single canonical solution. "
            "MECHANISM FOR CODE: AST-based uniqueness is cheap (Python ast.parse + "
            "ast.dump + sha256, ~0.1ms per rollout). Two rollouts are 'identical' if "
            "their AST strings match after stripping variable names (normalized AST "
            "comparison, preserving structure but ignoring local identifier names). "
            "For the 53/100 consistently-failing tasks, all 4 rollouts fail → "
            "coverage_scores = [0,0,0,0] → mean=0, std=0 → advantages = 0 → same as "
            "standard GRPO (zero gradient on all-fail groups). The GCPO improvement "
            "is concentrated on the 27 mixed-outcome tasks, which is exactly where our "
            "model has room to improve. "
            "DISTINCT FROM ALL 35 QUEUED EXPERIMENTS: "
            "- vs. F-GRPO (EXP-018): F-GRPO scales advantage at TASK level by success "
            "  rate (harder tasks get more weight); GCPO operates at ROLLOUT level within "
            "  a group based on solution diversity. Orthogonal granularities. "
            "- vs. EP-GRPO (EXP-028): Entropy gating identifies uncertain TOKEN positions; "
            "  GCPO identifies diverse SOLUTION patterns at the rollout level. Different "
            "  granularity and different question answered. "
            "- vs. EGCA (EXP-026): EGCA traces execution to find the CAUSAL failure token "
            "  (single best rollout analysis); GCPO asks whether multiple rollouts cover "
            "  DIVERSE valid solutions (multi-rollout coverage analysis). "
            "- vs. DAPO (EXP-013): DAPO discards all-zero-advantage groups to get cleaner "
            "  gradients on mixed groups; GCPO KEEPS all groups but changes how rollout "
            "  advantages are computed within mixed groups. "
            "- No prior experiment uses AST-level solution uniqueness as a reward signal. "
            "Key results from paper (math reasoning, Qwen2.5-7B on AIME2024/MATH500): "
            "GCPO achieves 8-15% relative improvement over GRPO on tasks with multiple "
            "valid solution paths; gains are concentrated on problems with G_pass ≥ 2 "
            "(the same class as our 27 mixed-outcome MBPP tasks). For G_pass=0 tasks "
            "(our 53 all-fail), GCPO is identical to GRPO — no regression risk. "
            "Runner change: add algorithm='gcpo', gcpo_uniqueness='ast_normalized' to "
            "grpo_continual; after rollout generation, compute normalized AST hash per "
            "rollout, assign coverage_score=[unique_passer → 1, duplicate_passer → 0, "
            "failer → 0], compute advantage from coverage_scores instead of raw rewards. "
            "~60 additional lines (ast.parse + sha256 + advantage recomputation). "
            "Estimated wall-clock: ~92 minutes (200 tasks × 4 rollouts; ~0.4ms AST "
            "overhead per rollout is negligible vs. 4000ms forward pass)."
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
            "algorithm": "gcpo",
            "gcpo_uniqueness": "ast_normalized",
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_05_29_002_warp_lora_interp_eval_15b",
        "priority": 7,
        "kind": "forgetting_eval",
        "rationale": (
            "WARP: On the Benefits of Weight Averaged Rewarded Policies "
            "(arxiv:2406.16768, Rame et al., Google DeepMind, NeurIPS 2024) demonstrates "
            "that linearly interpolating an RL fine-tuned policy θ_RL with the original "
            "base policy θ_base — the WA (weight-averaged) policy "
            "θ_WA(α) = α * θ_RL + (1-α) * θ_base — finds a better point on the "
            "reward vs. KL-to-base Pareto curve than either endpoint. Specifically: "
            "the RL adapter overshoots the optimal reward by over-fitting to the reward "
            "signal (reward hacking), while the base model under-fits. Weight averaging "
            "at α < 1.0 can reduce reward hacking while preserving the genuine learned "
            "behaviours, yielding higher actual task accuracy than the full RL adapter. "
            "The paper proves that for RL policies trained with KL regularisation, the "
            "optimal WA coefficient satisfies α* = λ/(λ + β) where λ is the RL reward "
            "coefficient and β is the KL coefficient — in the range (0, 1). For our "
            "standard GRPO adapter: implicit KL budget is the PPO clip (ε=0.2) with no "
            "explicit KL penalty, so the adapter may have drifted beyond the optimal "
            "point. Weight averaging at α ∈ {0.3, 0.5, 0.7, 0.9} finds the optimal "
            "interpolation factor empirically. "
            "MECHANISTIC CASE FOR OUR SETUP: "
            "Our 1.5B GRPO adapter (200×4, binary reward) achieves 49/100 (+2pts). "
            "The +2pt gain is distributed across ~27 tasks (the mixed-outcome training "
            "tasks) and potentially 0-5 tasks that were previously hard-fail but the "
            "adapter improved. With no explicit KL penalty, the adapter may have "
            "increased its probability on correct solutions for 30-35 tasks but also "
            "slightly decreased on 3-5 other tasks (reward hacking artefacts). "
            "Weight averaging at α=0.7 would preserve 70% of the adapter's drift: "
            "the genuine improvements (high-probability improvements on 27+ tasks) "
            "would survive because they represent large weight changes; the reward-"
            "hacking artefacts (small weight changes on edge cases) would be attenuated. "
            "DIAGNOSTIC VALUE: "
            "(a) If α=0.7 achieves 50/100 or 51/100: the standard adapter overshoots "
            "    optimal. Future GRPO variants should use explicit KL (kl_coeff=0.02) "
            "    to avoid overshoot, OR post-hoc WARP should be applied after each "
            "    training run as a free +1pt improvement. "
            "(b) If α=1.0 (full adapter) remains optimal: no overshoot, reward hacking "
            "    is not a factor, the +2pt ceiling is due to training signal quality "
            "    (motivating EXP-034 REINFORCE++, EXP-028 EP-GRPO, etc.). "
            "(c) If α=0.3 is best: severe overshoot — the RL training massively corrupts "
            "    base model behaviour. This would be a strong signal to add explicit KL "
            "    penalty to ALL future experiments. "
            "DISTINCT FROM ALL 35 EXISTING EXPERIMENTS: "
            "- No prior experiment performs LoRA weight interpolation. All 8 existing "
            "  forgetting_eval / eval experiments load the adapter at full α=1.0. "
            "- vs. contrastive decode eval (EXP-035, already queued): EXP-035 tests "
            "  inference-time amplification of the adapter's signal; WARP tests "
            "  parameter-space interpolation between the adapter and base. Orthogonal "
            "  approaches to squeezing more from the existing best adapter. "
            "IMPLEMENTATION: Load adapter at 5 interpolation points "
            "(α ∈ {0.3, 0.5, 0.7, 0.9, 1.0}): for each α, set LoRA weights to "
            "α * adapter_lora_weights (the base LoRA weights are 0, so interpolation "
            "is just scaling the adapter delta). Evaluate each on 100 MBPP tasks. "
            "Total: 5 × ~5 min eval passes = ~25 minutes. "
            "Estimated wall-clock: ~25 minutes."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "adapter": (
                "/data0/home/zeyuwang/router-skills-evolve-runs/rl_15b/"
                "qwen25_coder_15b_grpo_200x4"
            ),
            "eval_sets": [
                {
                    "name": "MBPP_warp_interp",
                    "data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/test_eval_all.jsonl",
                    "limit": 100,
                    "offset": 0,
                    "prompt_style": "qwen-chat",
                    "warp_interp": True,
                    "warp_alpha_values": [0.3, 0.5, 0.7, 0.9, 1.0],
                },
            ],
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    # ── 2026-05-28 batch (2 experiments — queue > 20 cap, A800 day 14 offline) ──
    {
        "id": "exp_2026_05_28_001_reinforce_ema_baseline_15b",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "REINFORCE++ (arxiv:2501.03262, Zeng et al., January 2025) replaces GRPO's "
            "within-group advantage normalization with a global exponential moving average "
            "(EMA) reward baseline: A_i = r_i - EMA_t, where EMA_t = decay * EMA_{t-1} + "
            "(1 - decay) * mean_reward_t across all training prompts. This addresses a "
            "fundamental limitation of GRPO with small group sizes (G=4): the within-group "
            "std estimator is extremely noisy with only 4 samples, and all-same-reward groups "
            "(std=0) silently produce zero advantage, discarding the gradient entirely. "
            "The EMA baseline has qualitatively different gradient properties on our "
            "training distribution: "
            "(1) All-fail groups (53/100 tasks): GRPO group-mean = 0, advantage = 0, zero "
            "    gradient. REINFORCE++ EMA baseline ≈ 0.47 (the empirical training pass rate), "
            "    advantage = 0 - 0.47 = -0.47 — a PENALTY signal that discourages the "
            "    failing patterns. This is actionable gradient on the hardest 53 tasks that "
            "    GRPO cannot touch. "
            "(2) All-pass groups (20/100 tasks): GRPO group-mean = 1, advantage = 0, zero "
            "    gradient. REINFORCE++ advantage = 1 - 0.47 = +0.53 — a REINFORCEMENT "
            "    signal that consolidates correct patterns on easy tasks. "
            "(3) Mixed groups (27/100 tasks): GRPO advantage ≈ ±0.5/std. REINFORCE++ "
            "    advantage for passing rollout = 1 - 0.47 = +0.53; for failing = -0.47. "
            "    Similar signal magnitude but computed against a more stable baseline. "
            "REINFORCE++ also adds a token-level KL penalty (kl_coeff=0.02, kl_level=token) "
            "matching the DAPO-style token normalisation already validated for variable-length "
            "code outputs (EXP-013). The EMA baseline with token-level KL is the complete "
            "REINFORCE++ recipe. "
            "DISTINCT FROM ALL 34 EXISTING EXPERIMENTS: "
            "- vs. RLOO (EXP-021): RLOO uses leave-one-out baseline WITHIN the G=4 group "
            "  (mean of 3 other rollouts). REINFORCE++ uses EMA across ALL training steps "
            "  and ALL prompts — a global baseline vs. a local one. With G=4, RLOO's "
            "  leave-one-out reduces to mean of 3 samples (still very noisy). The EMA "
            "  smooths over the full training history. "
            "- vs. GRPO (all others): Group relative normalisation by (r - mean) / std "
            "  with std computed from G=4 samples. "
            "- vs. DAPO (EXP-013): DAPO uses group-relative advantage + dynamic group "
            "  resampling + token-level loss. REINFORCE++ replaces group-relative with EMA "
            "  baseline and adds token-level KL but does NOT use dynamic sampling. "
            "  Orthogonal mechanisms. "
            "- No prior experiment uses a global EMA reward baseline. "
            "Runner change: add algorithm='reinforce_ema', ema_decay=0.99, kl_level='token' "
            "to grpo_continual. Initialize EMA at the base model's expected pass rate (~0.47). "
            "~20 lines of change (replace advantage = (r - group_mean) / group_std with "
            "advantage = r - ema; update ema after each step). "
            "Estimated wall-clock: ~88 minutes (200 tasks × 4 rollouts; identical compute "
            "to standard GRPO; no overhead from EMA update)."
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
            "algorithm": "reinforce_ema",
            "ema_decay": 0.99,
            "ema_init": 0.47,
            "kl_coeff": 0.02,
            "kl_level": "token",
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_05_28_002_contrastive_decode_eval_grpo_15b",
        "priority": 7,
        "kind": "forgetting_eval",
        "rationale": (
            "Contrastive Decoding (Li et al., arxiv:2210.15097, NeurIPS 2023) improves "
            "generation quality at inference time by amplifying the probability mass at "
            "positions where an expert model differs from an amateur model: "
            "P_CD(x_t | x_{<t}) ∝ max(P_expert(x_t) - alpha * P_amateur(x_t), 0). "
            "Applied to our GRPO adapter: expert = qwen25_coder_15b_grpo_200x4 adapter, "
            "amateur = base Qwen2.5-Coder-1.5B-Instruct (no adapter). The contrastive "
            "distribution focuses generation mass on tokens where the GRPO training "
            "shifted probability — precisely the ~5% of token positions where RL training "
            "makes meaningful changes (consistent with the sparse-policy-selection finding "
            "from EXP-021's motivation, arxiv:2605.06241). "
            "MECHANISTIC JUSTIFICATION: Our GRPO adapter improves 47→49/100 (+2pts) at "
            "standard sampling. The per-task probability shift is small: on average, "
            "adapter_p(correct) ≈ base_p(correct) + delta. Standard decoding samples from "
            "the adapter distribution directly, mixing adapter-specific tokens (the +delta "
            "positions) with base-model-identical tokens (the ~95% of unchanged positions). "
            "Contrastive decoding with alpha=0.1 subtracts 10% of the base logits, "
            "effectively amplifying the adapter's differential signal without introducing "
            "incoherence (small alpha keeps only modest modifications). If the +2pt gain "
            "comes from 5-8 token-level improvements per problem solution, contrastive "
            "decoding should amplify those specific positions and convert some currently- "
            "borderline solutions to passing, at zero additional training cost. "
            "DIAGNOSTIC OUTCOMES: "
            "(a) CD improves pass@1 to ≥51/100: The +2pt gain is real but underestimated "
            "    by direct sampling — the adapter has shifted more probability mass than "
            "    standard eval reveals. Combining CD with future higher-gain adapters "
            "    (ProbL2-GRPO EXP-032, EGCA EXP-026) could compound to +5-8pts. "
            "(b) CD matches standard sampling (49/100 ± 1): The adapter's distribution "
            "    shift is already well-captured by direct sampling; contrastive amplification "
            "    adds noise rather than signal. Standard eval methodology is confirmed correct. "
            "(c) CD degrades below 49: The adapter's distribution changes are coherent "
            "    only in context (the token positions that changed depend on prior context "
            "    in a way that CD disrupts); subtracting the base model's predictions "
            "    introduces local incoherence at critical decision points. "
            "DISTINCT FROM ALL 34 EXISTING EXPERIMENTS: No prior experiment modifies the "
            "decoding algorithm at inference time. All 6 existing forgetting_eval experiments "
            "(EXP-004, EXP-010, EXP-021, EXP-023, EXP-024) use standard greedy or "
            "temperature=0.8 sampling. Contrastive decoding is a purely inference-time "
            "change — no training, no new adapter, no runner.py training code changes. "
            "Requires only loading two model copies (expert adapter + base) simultaneously: "
            "2 × 1.5B models ≈ 6GB each ≈ 12GB total, well within the 80GB A800. "
            "Implementation: replace standard model.generate() call with a custom "
            "generate_contrastive() that at each step computes "
            "logits_CD = logits_expert - alpha * logits_base and samples from "
            "softmax(logits_CD). ~40 lines. Estimated wall-clock: ~25 minutes "
            "(100 tasks × 1 sample at standard speed, slightly slower due to 2 forward "
            "passes per step, estimated 1.6× single-model eval time)."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "adapter": (
                "/data0/home/zeyuwang/router-skills-evolve-runs/rl_15b/"
                "qwen25_coder_15b_grpo_200x4"
            ),
            "eval_sets": [
                {
                    "name": "MBPP_contrastive_decode",
                    "data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/test_eval_all.jsonl",
                    "limit": 100,
                    "offset": 0,
                    "prompt_style": "qwen-chat",
                    "decoding": "contrastive",
                    "contrastive_alpha": 0.1,
                    "contrastive_amateur_model": "base",
                },
            ],
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },

    # ── 2026-05-29 paper-pipeline batch (4 experiments — weekly paper reviewer pass) ──
    {
        "id": "exp_2026_05_29_003_grpo_multi_seed_3x_15b",
        "priority": 9,
        "kind": "joint_cycle_multiseed",
        "rationale": (
            "W1 (weekly review 2026-05-29): All GRPO results are n=1 seed. "
            "The +2pp best result (47->49/100) is within the binomial standard error "
            "(sigma ~5pp) and is statistically unverifiable without multiple seeds. "
            "Run 3 independent seeds (42, 43, 44) of the identical 1.5B GRPO 200x4 "
            "recipe to produce mean +/- std. If mean >= +1pp and std <= 2pp, the "
            "result is publishable. If std > 3pp, the result is a fluke. "
            "This is the single highest-priority experiment for AAAI 2027 submission. "
            "~270 min total (3x 90 min, parallelizable across 3 GPUs)."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/train_aug_excluding_eval20.jsonl",
            "eval_data": "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/test_eval_all.jsonl",
            "train_task_limit": 200,
            "seeds": [42, 43, 44],
            "epochs": 1,
            "rollouts_per_prompt": 4,
            "lr": 5e-6,
            "lora_r": 16,
            "prompt_style": "qwen-chat",
            "reward": "binary",
            "eval_limit": 100,
            "max_new_tokens": 192,
            "parallelizable": True,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_05_29_004_humaneval_grpo_adapter_eval_15b",
        "priority": 8,
        "kind": "forgetting_eval",
        "rationale": (
            "W3 (weekly review 2026-05-29): The GRPO adapter is only evaluated on MBPP "
            "eval100. The HumanEval 99%% accuracy result uses the base small model + "
            "routing. Reviewers will ask whether the adapter generalizes to HumanEval. "
            "Evaluate qwen25_coder_15b_grpo_200x4 on HumanEval (164 tasks), specifically "
            "the 9 large-required signature patterns (L|advanced/list/num, L|crypto/str, "
            "L|list/num, L|list/str, M|advanced/list, M|bool/num, M|bool/str, "
            "M|list/num/sort, M|list/num/str). If the adapter newly solves 1-2 of these, "
            "the paper can claim closed-loop improvement. ~30 min."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "adapter": "/data0/home/zeyuwang/router-skills-evolve-runs/rl_15b/qwen25_coder_15b_grpo_200x4",
            "eval_data": "/data0/home/zeyuwang/router-skills-evolve/data/HumanEval.jsonl",
            "eval_limit": 164,
            "max_new_tokens": 384,
            "prompt_style": "qwen-chat",
            "focus_signatures": [
                "L|advanced/list/num", "L|crypto/str", "L|list/num", "L|list/str",
                "M|advanced/list", "M|bool/num", "M|bool/str", "M|list/num/sort",
                "M|list/num/str",
            ],
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_05_29_005_router_gold_label_eval_100",
        "priority": 8,
        "kind": "forgetting_eval",
        "rationale": (
            "W2 (weekly review 2026-05-29): UncommonRoute labels are weak labels. "
            "Randomly sample 100 examples from the 334-example router test split. "
            "For each, run Qwen2.5-Coder-1.5B-Instruct (small) and GPT-4o (large) on "
            "the actual task via the execution suite. Use actual pass/fail results as "
            "gold routing labels. Report learned router accuracy vs. gold labels. "
            "If accuracy >= 90%%, the paper's soundness is confirmed. "
            "If < 85%%, the router section needs rescoping. ~90 min."
        ),
        "spec": {
            "router_model": "/data0/home/zeyuwang/router-skills-evolve-runs/learned-router-mixed-pretrained",
            "router_threshold": 0.57,
            "eval_data": "/data0/home/zeyuwang/router-skills-evolve-data/uncommonroute_bench.jsonl",
            "eval_limit": 100,
            "eval_split": "test",
            "gold_label_method": "execute_both_models",
            "small_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "gpt-4o",
            "output": "/data0/home/zeyuwang/router-skills-evolve-results/router_gold_label_eval_100.json",
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_05_29_006_grpo_kl_penalty_15b",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "W9 + ceiling break (weekly review 2026-05-29): Current best recipe has no "
            "explicit KL (only PPO clip). REINFORCE++ (arxiv:2501.03262) and DAPO "
            "(arxiv:2503.14476) show token-level KL prevents distribution collapse on "
            "all-fail groups. Adding kl_coeff=0.02, kl_level=token may both improve mean "
            "result and reduce variance across seeds (W1). Identical hyperparameters to "
            "best recipe except kl_coeff=0.02 and reference_model=base. ~90 min."
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
            "eval_limit": 100,
            "max_new_tokens": 192,
            "kl_coeff": 0.02,
            "kl_level": "token",
            "reference_model": "base",
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
