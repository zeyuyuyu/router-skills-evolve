#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-28 (EXP-136, EXP-137).

A800 connectivity: offline since 2026-05-14 (day ~75). Apply when restored.

Prior patches to apply before this one (daily-patch chain):
    python3 auto_research/pending_queue_update.py
    ... (see pending_queue_update_2026_07_24.py header for full chain) ...
    python3 auto_research/pending_queue_update_2026_07_24.py            # EXP-130, EXP-131
    python3 auto_research/pending_queue_update_2026_07_24_paper.py      # EXP-132, EXP-133
    python3 auto_research/pending_queue_update_2026_07_27.py            # EXP-134, EXP-135

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_28.py            # EXP-136, EXP-137

Queue was ~145 pending on 2026-07-27 (+2 added: EXP-134, EXP-135).
Queue cap applied: >20 → max 2 new experiments.

AAAI 2027 deadline: 2026-08-15 (18 days from today 2026-07-28).
A800 offline since 2026-05-14 (day 75).
GPU window CLOSES 2026-08-01 (4 days) — new results must land by Aug 1 to reach paper.

Data motivating today's proposals
----------------------------------
results/e2e_4cyc_gpt55/cycle_3/e2e_ablation_summary.json (unchanged — A800 offline day 75):
    HumanEval 4-cycle:
      large (always-large):        task_pass=96.34%, cost_vs_large=100%
      skills (always-small+proc):  task_pass=75.61%, cost_vs_large=10%
      router (logistic, cycle 3):  task_pass=92.68%, routing_acc=92.68%,
                                   cost_vs_large=27.56%, fallback=6.10%
      full (router+GRPO):          task_pass=92.68%  ← identical to router (all 4 cycles)
    Root problems (cumulative understanding):
      (A) ACR=52.4% — 43/82 GRPO groups have zero within-group reward variance
          → zero (or worse: amplified-noise) gradient → Full = Router.
      (B) Non-collapsed groups (47.6%) carry ALL GRPO signal; credit quality matters.
      (C) Skills gap: 75.61% vs large 96.34% — 20.7pp gap.
      (D) GRPO may forget SFT gains (EXP-130 CPO-PMP, EXP-135 no-std-norm address this).
      (E) Binary-reward setting may cause group-mean centering to provide poor gradient
          signal even in non-ACR groups (new hypothesis from arxiv:2605.07689 below).

New papers motivating today's proposals:

arxiv:2602.08813 — "Robust Policy Optimization to Prevent Catastrophic Forgetting"
    (Sabbaghi, Pappas, Javanmard, Hassani; UPenn / USC; February 2026)

    BACKGROUND:
    Post-training LLMs via multi-stage RL (RLHF then downstream fine-tuning) introduces
    catastrophic forgetting: downstream updates compromise capabilities learned in earlier
    stages. Standard GRPO/PPO optimizes reward at a single policy point (the current
    θ), without considering whether that policy point is robust to further updates.
    Result: after GRPO training, even small downstream perturbations (e.g., next-cycle
    SFT) drive the policy off its reward plateau — it was not optimised for robustness
    near its current location in policy space.

    PROPOSED METHOD — FRPO (Fine-tuning Robust Policy Optimization):
    FRPO modifies the GRPO objective with a max-min (minimax) formulation:
        max_{θ} min_{q: KL(q‖π_θ) ≤ ε} E_q[R(x, y)]
    i.e., instead of maximizing reward at the current policy θ, maximize the
    WORST-CASE reward achievable by any policy within a KL ball of radius ε around θ.
    This forces θ to sit in a region of policy space where reward is stable under
    small perturbations — a "flat reward landscape" around the training fixed point.

    KEY FEATURES:
    - No extra computation: the max-min objective has an efficient closed-form
      subgradient that modifies only the advantage weights in GRPO:
          A_FRPO_i = A_GRPO_i × exp(α × A_GRPO_i)
      where α = ε (the KL ball radius hyperparameter). This is a single
      element-wise multiplication on the advantages — zero overhead vs standard GRPO.
    - No extra memory: no frozen reference model storage, no explicit rollout buffer
      replay. FRPO modifies only the advantage computation step.
    - Reduction: at ε→0, FRPO ≡ standard GRPO. As ε grows, FRPO amplifies high-
      advantage rollouts and down-weights near-zero-advantage rollouts (exponential
      reweighting).
    - Results (UPenn evaluation on math-focused RL): FRPO preserves accuracy under
      subsequent fine-tuning, substantially reducing safety degradation across
      multiple base models and fine-tuning regimes (SFT and RL).

    RELEVANCE TO OUR PIPELINE — NEW HYPOTHESIS E:
    In our 4-cycle continual pipeline, GRPO Phase 3b is not the last training step:
    the GRPO adapter becomes the starting point for cycle n+1's SFT (Phase 3a).
    If GRPO training produces a policy at a sharp reward peak (high reward only at
    exactly the training θ, not nearby), then:
    - Phase (n+1) SFT immediately perturbs θ away from the GRPO fixed point.
    - The perturbed θ is no longer on the reward plateau → GRPO quality regresses
      in cycle n+1.
    - This compounds across cycles: each cycle's GRPO is undone by the next SFT.
    FRPO forces GRPO to converge to a FLAT region: even after next-cycle SFT perturbs θ,
    the policy stays near the reward plateau. Expected: Full arm degrades less across
    cycles; cycle-to-cycle GRPO quality more stable; Full > Router gap grows per cycle.

    MECHANISM DIFFERENCE FROM CPO-PMP (EXP-130):
    - CPO-PMP: parameter-space regularization (L2 from SFT point). Prevents parameter
      drift FROM the SFT checkpoint. Anchors the GRPO adapter near the SFT fixed point.
    - FRPO: policy-space distributional robustness. Does not anchor near SFT; instead
      forces GRPO to converge to a flat reward landscape in policy space.
    - These are orthogonal: CPO-PMP prevents SFT→GRPO forgetting; FRPO prevents
      GRPO→SFT(next cycle) forgetting. Both can be active simultaneously.
    - FRPO requires only A_i ← A_i × exp(α × A_i) in advantage computation.

    DISTINCTION FROM QUEUE:
    - EXP-130 (CPO-PMP): L2-SP parameter regularization — different space (parameter
      vs. policy), different phase (SFT forgetting vs. cycle-over-cycle robustness).
    - EXP-135 (no-std-norm): removes std normalization — modifies advantage SCALING,
      not advantage WEIGHTING. FRPO reweights advantages exponentially; no-std-norm
      removes the scaling denominator. Independent mechanisms.
    - EXP-126 (GEPO): entropy-conditioned attenuation — scales down advantages for
      high-entropy groups; FRPO scales UP high-advantage rollouts (opposite direction).
    FRPO is the ONLY queued experiment using distributional robustness (max-min
    policy optimization) as the anti-forgetting mechanism.

    IMPLEMENTATION:
    In grpo_train_simple.py, after computing standard GRPO advantages:
        # FRPO reweighting (add ~3 lines):
        if FRPO_ENABLED:
            frpo_weights = torch.exp(FRPO_EPSILON * advantages).detach()
            advantages = advantages * frpo_weights
    FRPO_EPSILON controls the KL ball radius:
    - Small (0.01): minimal reweighting, nearly identical to GRPO.
    - Default (0.1): moderate amplification of high-advantage rollouts.
    - Large (0.5): strong amplification; risk of advantage explosion if not clipped.
    Add env vars: FRPO_ENABLED (default 0), FRPO_EPSILON (default 0.1),
    FRPO_CLIP_ADV (default 5.0 — clips |frpo_reweighted_advantage| to prevent explosion).
    Total: ~5 lines. Zero memory overhead. ~1% compute overhead.
    Implementation reference: https://arxiv.org/abs/2602.08813

arxiv:2605.07689 — "Gradient Starvation in Binary-Reward GRPO: Why Group-Mean
    Centering Fails and Why the Simplest Fix Works"
    (Anonymous submission, May 2026)

    BACKGROUND — BINARY REWARD PECULIARITY:
    Standard GRPO uses group-mean centering as the baseline:
        A_i = (r_i - mean(r_group)) / std(r_group)    [or without std, see EXP-135]
    This works well when rewards are continuous (e.g., scalar quality scores), because
    the mean is a meaningful center of mass. For BINARY rewards (r_i ∈ {0, 1}):
    - mean(r_group) = k/K, where k = number of successes in the group of K rollouts.
    - The "mean" is NOT a meaningful reward baseline: it conflates task difficulty
      (p̂ = k/K reflects the task pass rate) with the credit signal.
    - A task with p̂ = 0.9 (easy): k = 7 or 8; A_success ≈ (1 - 7/8) = 0.125 tiny
      gradient for the few successes, large negative gradient for the rare failures.
    - A task with p̂ = 0.1 (hard): k = 1; A_failure ≈ (0 - 1/8) = −0.125 large
      negative gradient for the many failures, tiny positive for the rare success.
    In both cases, gradient magnitude is small when the task is at the extremes of
    difficulty. The within-group mean acts as an adaptive normalizer that systematically
    REDUCES gradient for easy tasks (over-represented in mixed groups at high p̂) and
    for hard tasks (under-represented successes at low p̂).

    ROOT CAUSE ANALYSIS:
    For binary rewards, the optimal Bellman baseline should be the population-level
    expected reward E[r] over the full task distribution (not within the group). The
    within-group mean is an extremely noisy estimate of E[r] (based on only K=8 samples).
    Its variance is Var(mean) = p(1-p)/K — highest for intermediate tasks (the "sweet
    spot" tasks SC-SDPO targets), causing gradient estimates to be noisiest precisely
    where the signal matters most.

    PROPOSED FIX — Running-Mean Baseline (RMB):
    Replace within-group mean with a running exponential average of recent group rewards:
        baseline_t = β × baseline_{t-1} + (1 - β) × mean(r_current_batch)
        A_i = (r_i - baseline_t) / (std_running + ε)
    where std_running is the running std of r across recent batches.
    RMB eliminates the within-group sampling noise: baseline_t is based on N_batches × K
    samples (e.g., 10 batches × 82 groups × 8 rollouts = 6,560 samples), not just 8.
    For binary rewards:
    - baseline_t ≈ E[r] = global average pass rate.
    - A_success = 1 - E[r]: higher reward for tasks where 1.0 is surprising (hard tasks).
    - A_failure = 0 - E[r]: penalizes failure more for tasks that should be easy (E[r] high).
    This naturally concentrates gradient on intermediate-difficulty tasks — equivalent to
    SC-SDPO's √(p(1-p)) weighting but applied at the GRPO advantage level, not SFT loss.

    KEY RESULTS (from paper):
    - RMB-GRPO: +2.8 pass@1 on HumanEval vs standard GRPO for a 1.5B code model.
    - +1.9 pass@1 on MBPP.
    - Variance of gradient estimates reduced by 3.1× in non-ACR groups.
    - Stable training dynamics with no entropy collapse.

    RELEVANCE TO OUR PIPELINE — NEW HYPOTHESIS E:
    In our non-ACR groups (47.6% of 82 groups carry all GRPO signal), the within-group
    mean is estimated from K=8 binary rollouts. For a task with p̂=0.5 (the most
    informative), within-group mean variance = 0.5 × 0.5 / 8 = 0.03 (std 0.18).
    This is not negligible: the baseline can vary from 0 to 1 across groups of the
    same task in different batches, creating noisy advantage estimates.
    RMB with β=0.9 and N_recent=10 batches: baseline estimated from 6,560 samples →
    variance = p(1-p) / 6560 ≈ 0.00004 (std 0.006) — 30× more stable than within-group.

    DISTINCTION FROM QUEUE:
    - EXP-135 (no-std-norm): removes std division, keeps within-group mean. Problem B
      (noisy group-mean baseline) is still present in EXP-135 for non-ACR groups.
    - EXP-130 (CPO-PMP): parameter-space regularization, does not modify advantage.
    - EXP-122 (AVSPO): injects virtual samples to prevent ACR — different mechanism
      (virtual samples perturb the group composition; RMB changes the baseline estimator).
    - EXP-126 (GEPO): entropy-controlled attenuation, not a baseline replacement.
    RMB-GRPO is the ONLY queued experiment replacing within-group mean with a population-
    level running baseline for binary rewards.

    IMPLEMENTATION:
    In grpo_train_simple.py:
    1. Initialize running mean/std before the GRPO training loop:
           rmb_baseline = torch.tensor(0.5)   # initial prior for binary rewards
           rmb_std = torch.tensor(0.5)         # initial prior
    2. Before computing advantages each step:
           current_mean = rewards.mean().detach()
           current_std  = rewards.std().detach().clamp(min=1e-8)
           rmb_baseline = RMB_BETA * rmb_baseline + (1 - RMB_BETA) * current_mean
           rmb_std      = RMB_BETA * rmb_std      + (1 - RMB_BETA) * current_std
    3. Compute advantage:
           advantages = (rewards - rmb_baseline) / (rmb_std + 1e-8)
    Add env vars: RMB_GRPO_ENABLED (default 0), RMB_BETA (default 0.9),
    RMB_WARMUP_STEPS (default 5 — use within-group mean for first 5 steps before
    the running mean has enough history).
    Total: ~10 lines. Zero memory overhead. <1% compute overhead.
    Implementation reference: https://arxiv.org/abs/2605.07689
"""

import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = "data/humaneval_eval.jsonl"

NEW_EXPERIMENTS = [
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-136: FRPO — Distributional Robust GRPO for Cycle-Over-Cycle Robustness
    #          Tests Hypothesis E: GRPO converges to a sharp reward peak, undone
    #          by the next cycle's SFT perturbation.
    #          (arXiv:2602.08813, Feb 2026)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_28_001_frpo_distributional_robust_grpo_humaneval",
        "priority": 8,
        "gpu": "auto",
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2602.08813 — 'Robust Policy Optimization to Prevent Catastrophic "
            "Forgetting' (Sabbaghi et al., UPenn/USC, Feb 2026). Proposes FRPO: a "
            "max-min policy optimization that forces GRPO to converge to a flat reward "
            "region in policy space, robust to small downstream perturbations (including "
            "next-cycle SFT). Implementation: 3-line addition to GRPO advantage computation: "
            "A_i ← A_i × exp(ε × A_i), where ε (FRPO_EPSILON=0.1) is the KL ball radius. "
            "Zero memory overhead; <1% compute overhead. Motivated by Hypothesis E: our "
            "continual pipeline applies SFT then GRPO then SFT again (cycle n+1), so the "
            "GRPO fixed point must be stable to downstream SFT perturbation. Standard GRPO "
            "converges to a sharp reward peak (locally optimal but not robust) — the "
            "next-cycle SFT immediately drifts θ off the peak, undoing GRPO gains. FRPO "
            "finds a flatter peak where reward stays high even after SFT perturbation, "
            "explaining why Full arm should improve over cycles rather than resetting to "
            "Router quality each cycle. Paper results (math-focused RL, Qwen3): substantially "
            "reduces safety degradation under subsequent fine-tuning; preserves downstream "
            "task performance. Distinct from EXP-130 (CPO-PMP): CPO-PMP prevents GRPO from "
            "forgetting the SFT checkpoint (SFT→GRPO direction); FRPO prevents the next SFT "
            "from undoing the GRPO gains (GRPO→SFT direction). Orthogonal mechanisms. "
            "Distinct from EXP-135 (no-std-norm): FRPO reweights advantage exponentially; "
            "EXP-135 removes the std-scaling denominator. FRPO is the ONLY experiment using "
            "distributional robustness (max-min over KL ball) as the anti-forgetting mechanism. "
            "4-cycle run, ~3.5h GPU, fits within Aug 1 GPU window if A800 restores by July 29."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 4,
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "frpo_enabled": True,
            "frpo_epsilon": 0.1,
            "frpo_clip_adv": 5.0,
            "grpo_temperature": 1.0,
            "n_generations": 8,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "analysis": [
                "full_vs_router_pass_at_1_per_cycle_frpo_vs_vanilla",
                "advantage_reweighting_frpo_vs_standard_per_group",
                "cycle_over_cycle_grpo_quality_stability_frpo_vs_vanilla",
                "acr_fraction_frpo_vs_vanilla",
                "grpo_loss_curve_frpo_vs_vanilla",
                "policy_space_reward_landscape_flatness_frpo_vs_vanilla",
                "kl_divergence_grpo_to_next_sft_frpo_vs_vanilla",
            ],
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (baseline): full=router=92.68%",
                "EXP-130 (CPO-PMP): SFT→GRPO direction anti-forgetting (parameter space)",
                "EXP-135 (no-std-norm): removes std-scaling, does not reweight advantages",
                "arxiv:2602.08813 (FRPO): preserves task perf under subsequent fine-tuning",
            ],
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
            ],
            "implementation_note": (
                "In grpo_train_simple.py, after computing standard GRPO advantages "
                "(after any other advantage modifications like no-std-norm): "
                "  if FRPO_ENABLED: "
                "    frpo_weights = torch.exp(FRPO_EPSILON * advantages.detach()) "
                "    frpo_weights = frpo_weights.clamp(max=FRPO_CLIP_ADV_WEIGHT) "
                "    advantages = advantages * frpo_weights "
                "where FRPO_CLIP_ADV_WEIGHT = exp(FRPO_EPSILON * FRPO_CLIP_ADV). "
                "Note: frpo_weights must be detached (no gradient through the weight). "
                "This multiplies high-advantage rollouts by exp(ε × A_i) >> 1 "
                "(amplifies correct-direction updates) and down-weights near-zero "
                "advantage rollouts by exp(ε × ~0) ≈ 1 (minimal effect on near-ACR groups). "
                "FRPO_CLIP_ADV=5.0 clips the advantage before exponentiation to prevent "
                "weight explosion: max_weight = exp(0.1 × 5.0) = exp(0.5) ≈ 1.65. "
                "Epsilon sweep: run with FRPO_EPSILON ∈ {0.05, 0.1, 0.2} in a single "
                "4-cycle run by varying epsilon across cycles (cycle 0: 0.05, ...). "
                "Add env vars: FRPO_ENABLED (default 0), FRPO_EPSILON (default 0.1), "
                "FRPO_CLIP_ADV (default 5.0). "
                "Implementation reference: https://arxiv.org/abs/2602.08813"
            ),
            "arxiv_ref": "2602.08813",
            "estimated_gpu_hours": 3.5,
            "aaai_priority": "HIGH — orthogonal to EXP-130/135; tests new Hypothesis E",
        },
    },
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-137: RMB-GRPO — Running-Mean Baseline for Binary-Reward GRPO
    #          Replaces within-group mean with population-level running baseline
    #          to reduce gradient noise in non-ACR groups (binary rewards).
    #          (arXiv:2605.07689, May 2026)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_28_002_rmb_grpo_running_mean_baseline_binary_humaneval",
        "priority": 7,
        "gpu": "auto",
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2605.07689 — 'Gradient Starvation in Binary-Reward GRPO: Why "
            "Group-Mean Centering Fails and Why the Simplest Fix Works' (May 2026). "
            "Standard GRPO uses within-group mean as the reward baseline: "
            "A_i = (r_i - mean(r_group)) / std. For binary rewards (code: pass/fail), "
            "mean(r_group) = k/K is a noisy estimate of the task pass rate from only "
            "K=8 samples — variance = p(1-p)/8 ≈ 0.03 for intermediate tasks, causing "
            "advantage estimates to be unstable across gradient steps. The paper proposes "
            "Running-Mean Baseline (RMB): replace within-group mean with a running EMA "
            "of reward across recent batches (β=0.9), which smooths the baseline over "
            "~6,560 samples vs 8 — reducing baseline variance by ~30×. Paper results on "
            "1.5B code model: +2.8 pass@1 HumanEval, +1.9 MBPP, 3.1× gradient variance "
            "reduction in non-ACR groups. Our non-ACR groups (47.6% = all GRPO signal) "
            "use K=8 binary rollouts; RMB directly reduces noise in these groups. "
            "Distinct from EXP-135 (no-std-norm): EXP-135 eliminates the std denominator "
            "to silence exactly-zero groups; RMB replaces the numerator baseline for all "
            "groups, complementary interventions. Distinct from EXP-122 (AVSPO): AVSPO "
            "adds virtual samples to increase within-group variance; RMB changes the "
            "baseline estimator without modifying the reward signal. Distinct from "
            "EXP-128 (SC-SDPO): SC-SDPO applies √(p(1-p)) weighting to SFT loss (Phase 3a); "
            "RMB applies running-mean baseline to GRPO advantages (Phase 3b) — orthogonal. "
            "RMB is the ONLY queued experiment replacing within-group mean with a "
            "population-level running baseline for binary rewards. ~10 lines in "
            "grpo_train_simple.py. Expected: Full arm 92.68% → ≥94%; gradient variance "
            "reduction in non-ACR groups; stable GRPO convergence across 4 cycles."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 4,
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "rmb_grpo_enabled": True,
            "rmb_beta": 0.9,
            "rmb_warmup_steps": 5,
            "grpo_temperature": 1.0,
            "n_generations": 8,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "analysis": [
                "full_vs_router_pass_at_1_per_cycle_rmb_vs_vanilla",
                "baseline_value_rmb_vs_within_group_per_step",
                "advantage_variance_per_group_rmb_vs_vanilla",
                "gradient_variance_rmb_vs_vanilla_non_acr_groups",
                "acr_fraction_rmb_vs_vanilla",
                "grpo_loss_curve_rmb_vs_vanilla",
            ],
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (baseline): full=router=92.68%",
                "EXP-122 (AVSPO): virtual sample injection — different mechanism",
                "EXP-128 (SC-SDPO): SFT-level pass-rate weighting — Phase 3a, not 3b",
                "EXP-135 (no-std-norm): removes std denominator, keeps within-group mean",
                "arxiv:2605.07689 (RMB): +2.8 HumanEval, +1.9 MBPP, 3.1× variance reduction",
            ],
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
            ],
            "implementation_note": (
                "In grpo_train_simple.py, before the GRPO training loop, initialize: "
                "  rmb_mean = torch.tensor(0.5, device=device)  # prior for binary rewards "
                "  rmb_std  = torch.tensor(0.5, device=device)  # prior "
                "  rmb_step = 0 "
                "Inside each GRPO gradient step, before computing advantages: "
                "  batch_mean = rewards.mean().detach() "
                "  batch_std  = rewards.std().detach().clamp(min=1e-8) "
                "  rmb_step  += 1 "
                "  if rmb_step <= RMB_WARMUP_STEPS: "
                "    baseline = rewards.mean(dim=-1, keepdim=True)   # per-group mean "
                "    std_norm = rewards.std(dim=-1, keepdim=True).clamp(min=1e-8) "
                "  else: "
                "    rmb_mean = RMB_BETA * rmb_mean + (1 - RMB_BETA) * batch_mean "
                "    rmb_std  = RMB_BETA * rmb_std  + (1 - RMB_BETA) * batch_std "
                "    baseline = rmb_mean  # broadcast to [n_groups, K] "
                "    std_norm = rmb_std "
                "  advantages = (rewards - baseline) / (std_norm + 1e-8) "
                "Note: rmb_mean/rmb_std are global scalars (not per-group), reflecting "
                "the population-level expected pass rate rather than within-group rate. "
                "Add env vars: RMB_GRPO_ENABLED (default 0), RMB_BETA (default 0.9), "
                "RMB_WARMUP_STEPS (default 5). "
                "Total: ~10 lines. Zero extra memory, <1% compute overhead. "
                "Implementation reference: https://arxiv.org/abs/2605.07689"
            ),
            "arxiv_ref": "2605.07689",
            "estimated_gpu_hours": 3.5,
            "aaai_priority": "MEDIUM — +2.8 pass@1 expected; complements EXP-135 no-std-norm",
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
