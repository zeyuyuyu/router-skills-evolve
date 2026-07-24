#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-24 (EXP-130, EXP-131).

A800 connectivity: offline since 2026-05-14 (day ~71). Apply when restored.

Prior patches to apply before this one (daily-patch chain):
    python3 auto_research/pending_queue_update.py
    python3 auto_research/pending_queue_update_2026_06_06.py            # EXP-054, EXP-055
    python3 auto_research/pending_queue_update_2026_06_07.py            # EXP-056, EXP-057
    python3 auto_research/pending_queue_update_2026_06_08.py            # EXP-058, EXP-059
    python3 auto_research/pending_queue_update_2026_06_10.py            # EXP-060, EXP-061
    python3 auto_research/pending_queue_update_2026_06_11.py            # EXP-062, EXP-063
    python3 auto_research/pending_queue_update_2026_06_12.py            # EXP-064, EXP-065
    python3 auto_research/pending_queue_update_2026_06_12_v2.py         # EXP-066, EXP-067
    python3 auto_research/pending_queue_update_2026_06_14.py            # EXP-068, EXP-069
    python3 auto_research/pending_queue_update_2026_06_15.py            # EXP-070, EXP-071
    python3 auto_research/pending_queue_update_2026_06_16.py            # EXP-072, EXP-073
    python3 auto_research/pending_queue_update_2026_06_17.py            # EXP-074, EXP-075
    python3 auto_research/pending_queue_update_2026_06_18.py            # EXP-076, EXP-077
    python3 auto_research/pending_queue_update_2026_06_19.py            # EXP-078, EXP-079
    python3 auto_research/pending_queue_update_2026_06_19_paper.py      # EXP-080, EXP-081
    python3 auto_research/pending_queue_update_2026_06_21.py            # EXP-082, EXP-083
    python3 auto_research/pending_queue_update_2026_06_22.py            # EXP-084, EXP-085
    python3 auto_research/pending_queue_update_2026_06_23.py            # EXP-086, EXP-087
    python3 auto_research/pending_queue_update_2026_06_25.py            # EXP-088, EXP-089
    python3 auto_research/pending_queue_update_2026_06_26.py            # EXP-090, EXP-091
    python3 auto_research/pending_queue_update_2026_06_27.py            # EXP-092, EXP-093
    python3 auto_research/pending_queue_update_2026_06_28.py            # EXP-094, EXP-095
    python3 auto_research/pending_queue_update_2026_06_29.py            # EXP-096, EXP-097
    python3 auto_research/pending_queue_update_2026_06_30.py            # EXP-098, EXP-099
    python3 auto_research/pending_queue_update_2026_07_01.py            # EXP-100, EXP-101
    python3 auto_research/pending_queue_update_2026_07_02.py            # EXP-102, EXP-103
    python3 auto_research/pending_queue_update_2026_07_03.py            # EXP-104, EXP-105
    python3 auto_research/pending_queue_update_2026_07_03_paper.py      # EXP-106, EXP-107
    python3 auto_research/pending_queue_update_2026_07_05.py            # EXP-108, EXP-109
    python3 auto_research/pending_queue_update_2026_07_10_paper.py      # (paper-pipeline EXPs)
    python3 auto_research/pending_queue_update_2026_07_11.py            # EXP-110, EXP-111
    python3 auto_research/pending_queue_update_2026_07_12.py            # EXP-112, EXP-113
    python3 auto_research/pending_queue_update_2026_07_13.py            # EXP-114, EXP-115
    python3 auto_research/pending_queue_update_2026_07_14.py            # EXP-116, EXP-117
    python3 auto_research/pending_queue_update_2026_07_16.py            # EXP-118, EXP-119
    python3 auto_research/pending_queue_update_2026_07_17.py            # EXP-120, EXP-121
    python3 auto_research/pending_queue_update_2026_07_18.py            # EXP-122, EXP-123
    python3 auto_research/pending_queue_update_2026_07_19.py            # EXP-124, EXP-125
    python3 auto_research/pending_queue_update_2026_07_21.py            # EXP-126, EXP-127
    python3 auto_research/pending_queue_update_2026_07_23.py            # EXP-128, EXP-129

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_24.py            # EXP-130, EXP-131

Queue was ~139 pending on 2026-07-23 (+2 added: EXP-128, EXP-129).
Queue cap applied: >20 → max 2 new experiments.

Data motivating today's proposals
----------------------------------
results/e2e_4cyc_gpt55/cycle_3/e2e_ablation_summary.json (unchanged — A800 offline day 71):
    HumanEval 4-cycle:
      large (always-large):        task_pass=96.34%, cost_vs_large=100%
      skills (always-small+proc):  task_pass=75.61%, cost_vs_large=10%
      router (logistic, cycle 3):  task_pass=92.68%, routing_acc=92.68%,
                                   cost_vs_large=27.56%, fallback=6.10%
      full (router+GRPO):          task_pass=92.68%  ← identical to router
    Root problems:
      (A) ACR=52.4% — 43/82 GRPO groups have zero within-group reward variance
          → zero gradient → Full = Router.
      (B) Non-collapsed groups (47.6%) carry ALL GRPO signal; credit quality matters.
      (C) Skills gap: 75.61% vs large 96.34% — 20.7pp gap.
      (D) [NEW hypothesis] GRPO may forget SFT gains during Phase 3b training,
          causing the GRPO adapter to regress toward base model quality → Full = Router.

AAAI 2027 deadline: 2026-08-15 (22 days from today 2026-07-24).
A800 offline since 2026-05-14 (day 71).

arxiv:2607.04364 — "RL Forgets! Towards Continual Policy Optimization" (July 4, 2026)

    BACKGROUND:
    Standard GRPO / PPO post-training assumes that the policy does not catastrophically
    forget previously acquired capabilities during RL training. This assumption holds
    in single-task settings (one training distribution, one model lifetime) but breaks
    in continual post-training where the model is repeatedly updated across cycles.
    The paper studies catastrophic forgetting in RL post-training of LLMs:
    - Sequential continual RL training (learn task A → learn task B → ...) causes the
      model to lose capabilities it learned on task A while gaining on task B.
    - Critically, even single-phase GRPO training can cause forgetting of SFT gains:
      the GRPO gradient may push the model away from the SFT optimum on sub-tasks
      that are not directly rewarded, causing regression in solved-but-not-sampled tasks.

    PROPOSED METHOD — CPO (Continual Policy Optimization):
    CPO adds two components to the GRPO training objective:
    1. Prior-task behavioral KL regularization:
         KL(π_θ || π_ref) term penalizes divergence from the reference policy
         (= the SFT checkpoint, θ_SFT) on a small set of rehearsal prompts.
         Unlike standard GRPO's KL (which compares to the current rollout policy),
         this KL uses the SFT checkpoint as the anchor across training steps.
    2. Parameter-movement regularization:
         L_reg = λ * ||θ - θ_SFT||² penalizes large Euclidean drift from the SFT
         checkpoint in parameter space. This is an L2-SP (L2 from Starting Point)
         regularizer — equivalent to a Gaussian prior centered on θ_SFT.
    Combined loss: L_CPO = L_GRPO + λ_KL * KL(π_θ || π_ref) + λ_EWC * ||θ - θ_SFT||²

    KEY RESULTS:
    - CPO reduces forgetting by 13.7% and improves pretrained capability by 7.0%
      on Qwen3-VL-8B in the MRCL (Multimodal Reasoning Continual Learning) benchmark.
    - Replay-free: CPO does not require storing previous task data.
    - CPO consistently outperforms standard GRPO and other continual RL baselines
      (EWC, PackNet, AGEM, experience replay) in multi-task forgetting metrics.

    RELEVANCE TO OUR PIPELINE (NEW HYPOTHESIS D):
    In our 4-cycle HumanEval pipeline:
    - Phase 3a (SFT): fine-tunes the model on SCALING_FORCE_BOTH traces → improves
      small model from base quality to "SFT quality" (estimated ~75-80% pass@1).
    - Phase 3b (GRPO): further fine-tunes from the SFT checkpoint on rollout rewards.
    - If GRPO causes the model to forget SFT gains — specifically, if GRPO gradient
      pushes the model away from good SFT solutions for easy tasks (which the GRPO
      rollout never samples because the small model already passes them, making their
      GRPO advantage = 0) — the GRPO adapter ends cycle n with quality ≤ SFT quality.
    - This would explain Full = Router: the GRPO adapter is no better than the SFT
      checkpoint, and the router makes routing decisions → router accuracy determines
      the full arm's pass rate, not GRPO quality.

    MECHANISM IN OUR CONTEXT:
    Our GRPO training (Phase 3b) uses:
    - K=8 rollouts per task, T=82 tasks → 656 rollouts per gradient step (batch).
    - Tasks where the small model already passes (p̂ ≈ 1.0): all 8 rollouts succeed →
      group reward variance = 0 → GRPO advantage = 0 → zero gradient for those tasks.
    - But the GRPO gradient for hard tasks (advantage ≠ 0) may have components that
      push weights in directions that inadvertently hurt easy-task performance.
    - L2-SP (||θ - θ_SFT||²) keeps the GRPO adapter anchored near the SFT checkpoint,
      limiting regression on easy tasks while still allowing improvement on hard tasks.

    DISTINCTION FROM QUEUE:
    - EXP-125 (CPO-KL): adds KL regularization to Phase 3a SFT loss — different phase.
    - EXP-120 (Dr.GRPO): individual advantage normalization per token — different mechanism.
    - EXP-122 (AVSPO): advantage variance stabilization — targets group variance, not forgetting.
    - EXP-126 (GEPO): entropy-conditioned asymmetric attenuation — group-level, not parameter space.
    CPO-PMP is the ONLY queued experiment that:
    (1) adds parameter-space regularization to Phase 3b GRPO training (not SFT),
    (2) tests the Hypothesis D "GRPO forgets SFT gains" explanation for Full = Router.

    IMPLEMENTATION:
    In grpo_train_simple.py, after loading the SFT checkpoint as GRPO starting point:
    1. Store θ_SFT as a frozen reference: sft_params = {n: p.detach().clone() for n,p in model.named_parameters()}
    2. In the GRPO training loop's loss computation, add:
         l2_reg = sum(((p - sft_params[n])**2).sum() for n, p in model.named_parameters())
         loss = grpo_loss + CPO_PMP_LAMBDA * l2_reg
    3. CPO_PMP_LAMBDA controls the regularization strength:
         - Too small (< 1e-5): no effect; GRPO drifts freely from SFT.
         - Too large (> 1e-2): GRPO can't improve; stuck near SFT quality.
         - Target range: 1e-4 to 1e-3 (one order of magnitude sweep).
    4. Also implement the behavioral KL anchor: add KL(π_θ(rollouts) || π_SFT(rollouts))
       to the loss for the same rollout batch (reuse rollout log-probs from SFT model).
    Add env vars: CPO_PMP_ENABLED (default 0), CPO_PMP_LAMBDA (default 5e-4),
    CPO_PMP_KL_COEF (default 0.1 — behavioral KL anchor coefficient).
    Total: ~15 lines in grpo_train_simple.py.
    Implementation reference: https://arxiv.org/abs/2607.04364.

arxiv:2606.18810 — "Learning from Own Solutions: Self-Conditioned Credit Assignment
    for Reinforcement Learning with Verifiable Rewards"
    (Yingyu Shan et al., July 2026)

    BACKGROUND — GRPO CREDIT ASSIGNMENT IN NON-ACR GROUPS:
    Our ACR analysis: 43/82 groups (52.4%) have zero within-group reward variance
    (all-fail or all-pass → zero advantage → zero gradient). The remaining 47.6%
    of groups have non-zero variance — these groups carry ALL the GRPO signal.
    In these non-ACR groups, GRPO assigns credit via group-relative normalization:
    A_i = (r_i - mean(r_group)) / std(r_group)
    This assigns the same credit to all tokens of a successful rollout, regardless
    of which tokens were actually responsible for the solution's success vs. failure.

    PROPOSED METHOD — SCCA (Self-Conditioned Credit Assignment):
    For tasks with verifiable rewards (code: pass/fail), the model's own successful
    solutions provide a ground-truth signal: the tokens that diverged from the
    successful solution path are exactly the tokens that caused failures.
    SCCA uses self-conditioned advantage: for each failed rollout, find the model's
    own successful rollout(s) and identify the earliest divergence point (by comparing
    token sequences). Assign higher credit to tokens near the divergence and lower
    credit to tokens far from it:
        A_token(i, t) = A_group(i) × decay(dist_to_divergence(rollout_i, best_rollout_k))
    Where decay() is a monotone increasing function of distance to the nearest
    successful rollout's divergence point (e.g., exponential or linear).
    For all-pass or all-fail groups (ACR groups), SCCA falls back to standard GRPO
    advantage (no divergence to compute → no self-conditioned credit available).

    KEY RESULTS (from paper):
    - SCCA improves GRPO by +3.1 pass@1 on HumanEval and +2.7 pass@1 on MBPP
      for a 7B coder model.
    - Stable training (no entropy collapse or loss spike with SCCA).
    - SCCA's token-level credit assignment concentrates gradient on the tokens that
      actually caused success/failure, reducing noise from unrelated tokens.

    RELEVANCE TO OUR PIPELINE:
    For our non-ACR groups (47.6% of groups, ALL GRPO signal):
    - Standard GRPO assigns uniform advantage across all tokens of a rollout.
    - SCCA would concentrate advantage on the tokens near the divergence between
      failed and successful rollouts — exactly the tokens where the model's policy
      made a wrong decision.
    - HumanEval tasks produce solutions where divergences are often syntactically
      identifiable (wrong algorithm choice, off-by-one error, wrong return statement).
    - Expected improvement: better signal in non-ACR groups → GRPO adapter quality
      improves faster → Full arm > Router arm.

    DISTINCTION FROM QUEUE:
    - EXP-120 (Dr.GRPO): individual normalization per group/token, but no self-comparison
      between rollouts. SCCA explicitly uses the model's own successful rollouts as anchors.
    - EXP-122 (AVSPO): variance stabilization; no divergence-based credit.
    - EXP-123 (EDGE-GRPO): group-level entropy; no token-level credit conditioning.
    - EXP-124 (FADE): focal advantage via batch entropy; no self-conditioned comparison.
    SCCA is the ONLY queued experiment that uses the model's own successful solutions
    as explicit credit anchors — a fundamentally different mechanism from all normalization-
    or entropy-based GRPO modifications.

    IMPLEMENTATION:
    In grpo_train_simple.py, in the advantage computation step:
    1. For each group g (rollouts for task t), identify successful rollouts S_g = {i: r_i=1}.
    2. If S_g is empty or full → standard GRPO advantage (ACR group, no self-conditioning).
    3. For each failed rollout f (r_f=0), find the closest successful rollout s* ∈ S_g
       (closest = minimum edit distance or first divergence position in token sequence).
    4. For each token position t_pos in rollout f, compute:
         dist_t = |t_pos - divergence_pos(f, s*)| (token distance to divergence point)
         scca_weight(f, t_pos) = max(MIN_SCCA_WEIGHT, 1.0 - decay_rate * dist_t / seq_len)
    5. Modify token-level loss:
         loss_token(f, t_pos) = standard_loss(f, t_pos) × scca_weight(f, t_pos)
    6. For successful rollouts, SCCA weight = 1.0 (no modification — success gets
       full credit).
    Add env vars: SCCA_ENABLED (default 0), SCCA_DECAY_RATE (default 2.0),
    SCCA_MIN_WEIGHT (default 0.1), SCCA_DIVERGENCE_METHOD ('token_id' or 'edit_dist').
    Total: ~25 lines in grpo_train_simple.py.
    Implementation reference: https://arxiv.org/abs/2606.18810.
"""

import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = "data/humaneval_eval.jsonl"

NEW_EXPERIMENTS = [
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-130: CPO-PMP — Parameter-Movement Penalized GRPO
    #          Tests Hypothesis D: "GRPO forgets SFT gains → Full = Router"
    #          (arXiv:2607.04364, July 4, 2026)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_24_001_cpo_pmp_grpo_forgetting_humaneval",
        "priority": 7,
        "gpu": "auto",
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2607.04364 — 'RL Forgets! Towards Continual Policy Optimization' "
            "(July 4, 2026). Proposes CPO: parameter-movement regularization to prevent "
            "catastrophic forgetting of SFT gains during GRPO training. Motivated by "
            "Hypothesis D: our Full arm equals Router arm (full=router, all 4 cycles) "
            "not only because GRPO groups have zero variance (ACR=52.4%, addressed by "
            "EXP-120 through EXP-127) but also because GRPO may be forgetting the Phase "
            "3a SFT gains, regressing the model toward base quality. In our pipeline, "
            "GRPO gradient for hard tasks (non-ACR groups, 47.6%) may push weights in "
            "directions that inadvertently hurt easy-task performance — exactly the tasks "
            "that GRPO cannot improve (advantage=0 when all rollouts pass). L2-SP "
            "regularization (λ × ||θ - θ_SFT||²) keeps the GRPO adapter anchored near "
            "the SFT checkpoint, limiting regression on easy tasks while still allowing "
            "improvement on hard-task groups. Paper result: 13.7% forgetting reduction "
            "and +7.0% pretrained capability on Qwen3-VL-8B (MRCL benchmark). For our "
            "HumanEval pipeline, the SFT checkpoint is the 'prior policy' anchor; GRPO "
            "Phase 3b is the 'new task' that must not forget it. CPO-PMP implementation "
            "requires ~15 lines in grpo_train_simple.py: store θ_SFT at Phase 3b start, "
            "add l2_reg = Σ ||θ - θ_SFT||² to GRPO loss, with hyperparameter "
            "CPO_PMP_LAMBDA in [1e-4, 1e-3]. Also add behavioral KL anchor: "
            "KL(π_θ || π_SFT) on the same rollout batch. Distinct from EXP-125 "
            "(CPO-KL adds KL to Phase 3a SFT, not Phase 3b GRPO), EXP-120–127 (all "
            "address advantage computation, not parameter-space regularization), and all "
            "prior queue items. This is the ONLY experiment testing Hypothesis D "
            "directly. If confirmed: explains Full=Router and points to CPO-PMP as the "
            "fix. Expected: Full arm 92.68% → ≥94%; parameter drift ||θ_GRPO - θ_SFT||² "
            "reduced; easy-task pass rate maintained while hard-task pass rate improves."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 4,
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "cpo_pmp_enabled": True,
            "cpo_pmp_lambda": 5e-4,
            "cpo_pmp_kl_coef": 0.1,
            "cpo_pmp_lambda_sweep": [1e-4, 5e-4, 1e-3],
            "grpo_temperature": 1.0,
            "n_generations": 8,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "analysis": [
                "full_vs_router_pass_at_1_per_cycle_cpo_vs_vanilla",
                "parameter_drift_theta_grpo_minus_theta_sft_per_cycle",
                "easy_task_pass_rate_before_after_grpo_cpo_vs_vanilla",
                "hard_task_pass_rate_before_after_grpo_cpo_vs_vanilla",
                "grpo_loss_curve_with_without_l2_reg",
                "acr_fraction_cpo_vs_vanilla",
                "kl_divergence_policy_from_sft_per_step",
            ],
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (baseline): full=router=92.68%",
                "EXP-125 (CPO-KL): KL regularization in Phase 3a SFT (not Phase 3b GRPO)",
                "arxiv:2607.04364 (CPO): -13.7% forgetting, +7.0% capability on Qwen3-VL-8B",
            ],
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
            ],
            "implementation_note": (
                "In grpo_train_simple.py, at GRPO training loop entry point: "
                "1. Store SFT reference: "
                "   sft_params = {n: p.detach().clone().to('cpu') for n,p in model.named_parameters() if p.requires_grad} "
                "   [store on CPU to avoid doubling GPU VRAM; move to GPU per batch]. "
                "2. In each GRPO training step, after computing grpo_loss: "
                "   l2_reg = sum((p - sft_params[n].to(p.device)).pow(2).sum() "
                "              for n, p in model.named_parameters() if p.requires_grad) "
                "   kl_reg = GRPO_KL_COEFF * kl_div_from_sft(rollout_log_probs, sft_log_probs) "
                "   loss = grpo_loss + CPO_PMP_LAMBDA * l2_reg + CPO_PMP_KL_COEF * kl_reg "
                "3. To compute kl_reg: before unloading SFT checkpoint, store a frozen "
                "   SFT model reference (or SFT log_probs for the current rollout batch). "
                "   On GPU memory concern: use SFT log_probs computed once per rollout "
                "   batch (they are already computed for standard GRPO KL term). "
                "4. Log ||θ - θ_SFT||² and KL(π_θ||π_SFT) per step for diagnostic analysis. "
                "5. Run with CPO_PMP_LAMBDA_SWEEP=[1e-4, 5e-4, 1e-3] — use 5e-4 as default, "
                "   log all three in a single run using a sweep loop over GRPO cycles. "
                "Add env vars: CPO_PMP_ENABLED (default 0), CPO_PMP_LAMBDA (default 5e-4), "
                "CPO_PMP_KL_COEF (default 0.1). "
                "Total: ~15 lines. VRAM overhead: negligible (SFT params stored on CPU). "
                "Time overhead: ~5% per GRPO step (one extra sum of parameter norms). "
                "Implementation reference: https://arxiv.org/abs/2607.04364."
            ),
            "arxiv_ref": "2607.04364",
            "estimated_gpu_hours": 3.5,
        },
    },
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-131: SCCA — Self-Conditioned Credit Assignment for GRPO
    #          Uses model's own successful solutions as credit anchors
    #          (arXiv:2606.18810, June 2026)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_24_002_scca_self_conditioned_credit_grpo_humaneval",
        "priority": 7,
        "gpu": "auto",
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2606.18810 — 'Learning from Own Solutions: Self-Conditioned Credit "
            "Assignment for Reinforcement Learning with Verifiable Rewards' (Shan et al., "
            "June 2026). Addresses Problem B: the 47.6% of non-ACR GRPO groups carry ALL "
            "gradient signal, but standard GRPO assigns uniform advantage to every token "
            "of a rollout, regardless of which specific tokens were responsible for "
            "success vs. failure. SCCA uses the model's own successful rollouts as "
            "reference: for each failed rollout, finds the earliest token-level divergence "
            "from the closest successful rollout (within the same group), and assigns "
            "higher credit to tokens near that divergence point. The advantage is "
            "re-weighted: tokens far from the divergence (where success and failure agree) "
            "get down-weighted; tokens at and near the divergence (where the policy made "
            "a different choice leading to failure) get full credit. For ACR groups "
            "(all-pass or all-fail), SCCA falls back to standard GRPO — it only enriches "
            "the signal for non-ACR groups. Paper results: +3.1 pass@1 HumanEval, +2.7 "
            "pass@1 MBPP on a 7B coder, stable training dynamics. Our HumanEval tasks "
            "have syntactically clear divergence points (wrong loop bound, off-by-one, "
            "wrong return type), making token-level divergence identification reliable. "
            "SCCA is distinctly different from all EXP-120–129: EXP-120 (Dr.GRPO) "
            "normalizes advantage per-token without cross-rollout comparison; EXP-122 "
            "(AVSPO) stabilizes group variance without token-level credit; EXP-123 "
            "(EDGE-GRPO), EXP-124 (FADE), EXP-126 (GEPO) all operate at group/batch "
            "entropy level. SCCA is the ONLY experiment using the model's own successful "
            "rollouts as explicit token-level credit anchors. Implementation: ~25 lines "
            "in grpo_train_simple.py — for each non-ACR group, identify successful "
            "rollouts, compute token-sequence divergence position, and apply "
            "decay-weighted credit to failed rollout tokens near the divergence. "
            "Expected: Full arm 92.68% → ≥94%; improved gradient signal in non-ACR "
            "groups; faster GRPO convergence across cycles."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 4,
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "scca_enabled": True,
            "scca_decay_rate": 2.0,
            "scca_min_weight": 0.1,
            "scca_divergence_method": "token_id",
            "grpo_temperature": 1.0,
            "n_generations": 8,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "analysis": [
                "full_vs_router_pass_at_1_per_cycle_scca_vs_vanilla",
                "non_acr_group_gradient_magnitude_scca_vs_vanilla",
                "divergence_position_distribution_per_task",
                "token_credit_weight_distribution_scca",
                "grpo_loss_convergence_scca_vs_vanilla",
                "acr_fraction_scca_vs_vanilla",
            ],
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (baseline): full=router=92.68%",
                "EXP-120 (Dr.GRPO): per-token normalization without cross-rollout comparison",
                "EXP-122 (AVSPO): group variance stabilization, no token-level credit",
                "arxiv:2606.18810 (SCCA): +3.1 pass@1 HumanEval, +2.7 pass@1 MBPP (7B coder)",
            ],
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
            ],
            "implementation_note": (
                "In grpo_train_simple.py, in the advantage computation / loss step: "
                "1. For each group g (K rollouts for task t): "
                "   successful = [i for i in range(K) if rollout_rewards[g][i] == 1.0] "
                "   failed     = [i for i in range(K) if rollout_rewards[g][i] == 0.0] "
                "2. If len(successful) == 0 or len(failed) == 0: "
                "   scca_weights[g] = ones(K, seq_len)  # ACR group, fall back to GRPO "
                "3. Else (non-ACR group): "
                "   For each failed rollout f: "
                "     best_s = argmin over s in successful of first_divergence_pos(f, s) "
                "     # first_divergence_pos: first token position where token IDs differ "
                "     div_pos = first_divergence_pos(rollout_tokens[g][f], rollout_tokens[g][best_s]) "
                "     For each token position t_pos in [0, seq_len): "
                "       dist = abs(t_pos - div_pos) "
                "       scca_weights[g][f][t_pos] = max(SCCA_MIN_WEIGHT, 1.0 - SCCA_DECAY_RATE * dist / seq_len) "
                "   For each successful rollout s: "
                "     scca_weights[g][s] = ones(seq_len)  # full credit for successes "
                "4. Apply: loss = (grpo_loss_per_token * scca_weights).mean() "
                "Add env vars: SCCA_ENABLED (default 0), SCCA_DECAY_RATE (default 2.0), "
                "SCCA_MIN_WEIGHT (default 0.1), SCCA_DIVERGENCE_METHOD ('token_id'). "
                "Computational overhead: O(K² × seq_len) per group to compute pairwise "
                "divergence positions — negligible (K=8, seq_len~256, 82 groups). "
                "No extra GPU memory (token IDs already in rollout buffer). "
                "Implementation reference: https://arxiv.org/abs/2606.18810."
            ),
            "arxiv_ref": "2606.18810",
            "estimated_gpu_hours": 3.0,
        },
    },
]


def atomic_save(path: Path, obj) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


def main():
    with open(STATE_PATH) as f:
        state = json.load(f)

    queue = state.get("queue", [])
    history = state.get("history", [])
    existing_ids = {e["id"] for e in queue} | {e["id"] for e in history}

    added = []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] in existing_ids:
            print(f"  SKIP (already queued): {exp['id']}")
            continue
        queue.append(exp)
        existing_ids.add(exp["id"])
        added.append(exp["id"])
        print(f"  ADDED: {exp['id']}")

    state["queue"] = queue
    atomic_save(STATE_PATH, state)
    print(f"\nDone. Added {len(added)} experiments. Total queue: {len(queue)} pending.")


if __name__ == "__main__":
    main()
