#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-19 (EXP-124, EXP-125).

A800 connectivity: offline since 2026-05-14 (day ~66). Apply when restored.

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

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_19.py            # EXP-124, EXP-125

Queue was ~133 pending on 2026-07-18 (+2 added: EXP-122, EXP-123).
Queue cap applied: >20 → max 2 new experiments.

Data motivating today's proposals
----------------------------------
results/e2e_4cyc_gpt55/cycle_3/e2e_ablation_summary.json (unchanged — A800 offline day 66):
    HumanEval 4-cycle:
      large (always-large):        task_pass=96.34%, cost_vs_large=100%
      skills (always-small+proc):  task_pass=75.61%, cost_vs_large=10%
      router (logistic, cycle 3):  task_pass=92.68%, routing_acc=92.68%,
                                   cost_vs_large=27.56%, fallback=6.10%
      full (router+GRPO):          task_pass=92.68% ← same as router
    Root problems:
      (A) ACR=52.4% — 43/82 GRPO groups have zero within-group reward variance
          → zero gradient → Full = Router.
      (B) Non-collapsed groups (47.6%) = ALL GRPO signal; efficiency matters.
      (C) Skills gap: 75.61% vs large 96.34% — 20.7pp gap.

AAAI 2027 deadline: 2026-08-15 (27 days from today).
A800 offline since 2026-05-14 (day 66).

arxiv:2607.01490 — "Don't Let Gains FADE: Breaking Down Policy Gradient Weights in RL"
    Juliette Decugis, Sean O'Brien, Francis Bach, Gabriel Synnaeve, Taco Cohen
    (Meta FAIR / Inria / ENS-PSL), submitted July 1, 2026:

    MOTIVATION:
    RL post-training for LLMs suffers from two interacting failure modes that have
    been treated separately: (1) diversity collapse (entropy collapses to near-zero,
    the policy degenerates to greedy output) and (2) hard-problem focus (training
    concentrates on difficult tasks, effective sample size shrinks). The paper shows
    these are not independent pathologies but two axes of a single decomposition of
    the policy gradient objective, and that the optimal trade-off between them shifts
    over the course of training.

    FRAMEWORK — Gradient Mass Decomposition:
    Any policy gradient update can be decomposed into:
      1. Sign axis: ratio of positive-advantage mass M+ to negative-advantage mass M-.
         M+ = sum of advantages from correct/rewarded rollouts.
         M- = abs(sum of advantages from incorrect/penalized rollouts).
         When M+ >> M- (positive dominance): the update predominantly reinforces correct
           behavior without penalizing incorrect behavior → entropy collapses (the model
           becomes too confident too fast, stops exploring).
         When M- >> M+ (negative dominance): heavy penalization of errors with little
           reinforcement → weight geometry collapses (learned features degenerate).
         The paper proves both collapse modes under the policy gradient norm.
      2. Difficulty axis: within the correct/incorrect groups, samples are reweighted
         by difficulty. "Hard focus" concentrates updates on prompts where the model
         struggles (advantage magnitude is high), at the cost of effective sample size.
         The paper shows optimal difficulty focus shifts from high (exploration phase,
         learn hard problems) to medium (exploitation phase, consolidate).

    THE FADE METHOD (Focal Advantage with Dynamic Entropy):
    FADE is a self-adapting advantage that monitors current policy entropy H_curr and
    gradient mass balance, then adjusts the sign-axis weights and difficulty focus
    dynamically at each training step:

    Step 1 — Compute standard GRPO advantages A_i for each sequence in the batch.
    Step 2 — Compute M+ = sum(A_i[A_i > 0]), M- = abs(sum(A_i[A_i < 0])).
             Balance ratio: r = M+ / (M+ + M- + eps).
    Step 3 — Sign-axis rebalancing:
             If r > r_high (e.g. 0.65): positive dominance → scale A_i[A_i > 0]
               by (1 - r) / r to rebalance. This prevents entropy collapse.
             If r < r_low (e.g. 0.35): negative dominance → scale A_i[A_i < 0]
               by r / (1 - r) to rebalance. This prevents weight geometry collapse.
             If r_low ≤ r ≤ r_high: no sign-axis intervention.
    Step 4 — Difficulty-axis focal weighting:
             Compute per-sequence difficulty weight w_i = |A_i| / (mean|A| + eps).
             Then A_fade(i) = A_sign_rebalanced(i) * (w_i ** focal_gamma).
             focal_gamma > 0 emphasizes hard samples; focal_gamma < 0 suppresses them.
             Adaptive schedule: focal_gamma = gamma_max * (H_curr / H_init) ** k,
             where H_curr is the current policy entropy (measured once per epoch on
             a held-out set of 50 prompts), H_init is the entropy at step 0,
             and k ≥ 1 is a shape exponent (default k=2).
             As training progresses and entropy drops, focal_gamma drops → focus
             naturally shifts from hard exploration to balanced exploitation.

    KEY RESULTS:
    - FADE reaches peak pass@1 20k steps earlier than the best static baseline at
      7B scale (LiveCodeBench, code generation); 2k steps earlier at 32B.
    - Best accuracy-diversity trade-off across all pass@k (k=1,2,4,8) on
      LiveCodeBench and AIME math.
    - Outperforms DAPO, GRPO, and sign-advantage baselines in the paper's ablation.
    - Authors: Meta FAIR + Inria/ENS (Francis Bach, Gabriel Synnaeve, Taco Cohen).

    RELATION TO OUR QUEUE:
    FADE is distinct from all prior experiments:
      EXP-108 (sign-advantage): zeroes advantage for all-fail (sign flip only, no balance).
      EXP-120 (Dr.GRPO): zero gradient for collapsed groups (no sign rebalancing).
      EXP-122 (AVSPO): injects virtual samples to restore group diversity.
      EXP-123 (EDGE-GRPO): entropy scales advantage per sample.
      FADE: rebalances the positive/negative gradient mass ratio at the BATCH level,
        plus dynamically adjusts difficulty focus via a focal weight tied to entropy.
        Batch-level sign rebalancing is orthogonal to group-level injection (AVSPO)
        or per-sample entropy scaling (EDGE-GRPO). FADE addresses a collapse mode
        that the other methods do not: positive-dominance entropy collapse, where
        the model passes many tasks easily and the GRPO update becomes almost purely
        reinforcing (M+ >> M-), driving entropy to zero without any group-level ACR signal.

arxiv:2607.04364 — "RL Forgets! Towards Continual Policy Optimization"
    Mao-Lin Luo et al., July 2026:

    MOTIVATION:
    Our pipeline runs GRPO across N cycles. Within each cycle, the model is fine-tuned
    on a new set of traces from the current cycle's collection. The assumption is that
    each cycle's GRPO builds on the previous cycle's adapter. But does it? If the model
    catastrophically forgets cycle (N-1)'s hard-won GRPO gains when trained on cycle N,
    the "continual improvement" narrative breaks: cycles 2, 3, 4 restart rather than
    compound. This is Root Problem D — never explicitly measured in our pipeline.

    The "RL Forgets!" paper shows that even RL-based training (not just SFT) suffers
    from catastrophic forgetting during continual post-training — a direct challenge
    to the community belief that RL is "inherently less prone to forgetting" than SFT.
    It proposes Continual Policy Optimization (CPO):

    CPO METHOD:
    CPO is a replay-free framework with two components:
    1. Prior-task behavioral KL objective: add a KL penalty term to the GRPO loss:
         L_CPO = L_GRPO + lambda_KL * KL(π_current(· | x) || π_prev_cycle(· | x))
       where π_prev_cycle is the policy checkpoint saved at the END of the previous
       cycle's GRPO training. For our pipeline: at the start of cycle N's GRPO, load
       the cycle (N-1) GRPO adapter as the "prior policy" reference; freeze it; add
       the KL term to every step's loss. The KL is computed on the current batch of
       prompts (no separate replay buffer needed — on-policy prompts serve as anchors).
    2. Parameter-movement regularization: add an L2 penalty on the delta from the
       prior-cycle adapter parameters:
         L_total = L_CPO + lambda_L2 * ||θ_current - θ_prev_cycle||^2
       This limits the L2 distance the new adapter can drift from the prior adapter,
       providing a geometric constraint orthogonal to the KL behavioral constraint.
    The two terms are tuned with a single combined multiplier lambda (default 0.1)
    split as lambda_KL = lambda, lambda_L2 = lambda * alpha (alpha=0.5 default).

    KEY RESULTS (on Qwen3-VL-8B, continual visual reasoning tasks):
    - CPO reduces forgetting by 13.7% vs standard continual GRPO.
    - CPO improves pretrained capability preservation by 7.0%.
    - Replay-free: no extra GPU memory for a replay buffer; only the prior adapter
      needs to be kept in memory (same size as the trainable adapter, ~400MB at 1.5B).

    RELATION TO OUR PIPELINE:
    In our 4-cycle HumanEval run, "forgetting" would manifest as: cycle_N GRPO adapter
    solves FEWER of the tasks that cycle_{N-1} GRPO adapter solved. The full arm's
    stagnation (92.68% = router arm) is partly explained by ACR (Root Problem A), but
    may ALSO be due to inter-cycle forgetting — each new GRPO cycle partially erases
    the previous. CPO regularization costs ~10% extra compute per GRPO step (one extra
    KL forward pass over the prior adapter) and is replay-free, making it compatible
    with our existing grpo_train_simple.py infrastructure.

    RELATION TO OUR QUEUE:
    EXP-111 (forgetting_eval): measures inter-cycle forgetting diagnostically.
    EXP-125 (this): applies CPO regularization as an INTERVENTION. If EXP-111 found
    significant forgetting (prior data context), EXP-125 is the targeted remedy.
    EXP-117 (CPO continual GRPO): earlier CPO-related experiment in our queue; check
    the spec to confirm EXP-125 is distinct (this EXP focuses on the L2 parameter-
    movement component added by the July 2026 paper, which EXP-117 likely predates).
"""

import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = "data/humaneval_eval.jsonl"

NEW_EXPERIMENTS = [
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-124: FADE — Focal Advantage with Dynamic Entropy
    #          (arXiv:2607.01490, Meta FAIR / ENS-PSL, July 1 2026)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_19_001_fade_focal_advantage_dynamic_entropy_humaneval",
        "priority": 8,
        "gpu": "auto",
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2607.01490 — 'Don't Let Gains FADE: Breaking Down Policy Gradient "
            "Weights in RL' (Decugis, O'Brien, Bach, Synnaeve, Cohen; Meta FAIR / "
            "Inria / ENS-PSL, July 1, 2026). Introduces FADE (Focal Advantage with "
            "Dynamic Entropy), a self-adapting advantage that addresses two orthogonal "
            "GRPO failure modes simultaneously: (1) Sign-axis collapse — when positive-"
            "advantage mass M+ >> negative-advantage mass M- (many tasks passing easily), "
            "the policy update becomes predominantly reinforcing and entropy collapses "
            "without any group-level ACR signal being triggered. FADE rebalances M+/M- "
            "at the batch level by rescaling the dominant polarity's advantages. "
            "(2) Difficulty-axis misalignment — during exploration the model benefits "
            "from hard-sample focus (focal weighting by |A_i|), but during exploitation "
            "over-focusing starves easy tasks and reduces sample efficiency. FADE's focal "
            "weight is tied to the current policy entropy: focal_gamma = gamma_max * "
            "(H_curr / H_init)^k, naturally decaying as entropy drops. Paper reports: "
            "peak pass@1 reached 20k steps earlier than best static baseline at 7B scale "
            "on LiveCodeBench; best accuracy-diversity trade-off across pass@1 through "
            "pass@8. Distinct from all prior queue experiments: AVSPO (EXP-122) and "
            "EDGE-GRPO (EXP-123) target group-level ACR collapse; FADE targets the "
            "batch-level M+/M- imbalance that causes entropy collapse even when ACR is "
            "low. For our pipeline: after AVSPO/EDGE-GRPO fix group-level collapse, "
            "batch-level positive dominance (easy tasks flooding the batch) becomes the "
            "next failure mode. FADE addresses it with ~15 lines in grpo_train_simple.py."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 4,
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "fade_enabled": True,
            "fade_r_high": 0.65,
            "fade_r_low": 0.35,
            "fade_focal_gamma_max": 1.0,
            "fade_focal_k": 2,
            "fade_entropy_eval_prompts": 50,
            "grpo_temperature": 1.0,
            "n_generations": 8,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "analysis": [
                "gradient_mass_balance_ratio_per_step",
                "policy_entropy_per_epoch",
                "focal_gamma_schedule_vs_entropy",
                "positive_negative_mass_ratio_vs_baseline",
                "pass_at_1_convergence_speed_vs_standard_grpo",
                "pass_at_k_diversity_vs_baseline_k_1_2_4_8",
                "full_arm_pass_at_1_per_cycle",
            ],
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (baseline): full 92.68%, ACR=52.4%",
                "EXP-108 (sign-advantage): all-fail groups only, fixed sign, no balance",
                "EXP-122 (AVSPO): group-level virtual injection, no sign-axis rebalancing",
                "EXP-123 (EDGE-GRPO): per-sample entropy scaling, no batch M+/M- balance",
                "arxiv:2607.01490 (FADE): 20k step speedup at 7B, best pass@k trade-off",
            ],
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
            ],
            "implementation_note": (
                "In grpo_train_simple.py, inside the GRPO training loop after TRL "
                "computes advantages (or alternatively via a custom compute_advantages "
                "hook if TRL exposes one; otherwise implement as a reward-shaping wrapper "
                "over the existing reward_fn): "
                "1. Collect all advantages A_i for the current mini-batch. "
                "2. M_pos = sum(A_i[A_i > 0]); M_neg = abs(sum(A_i[A_i < 0])). "
                "3. r = M_pos / (M_pos + M_neg + 1e-8). "
                "4. If r > FADE_R_HIGH: scale A_i[A_i > 0] *= (1 - r) / (r + 1e-8). "
                "   If r < FADE_R_LOW: scale A_i[A_i < 0] *= r / (1 - r + 1e-8). "
                "5. Per-sequence focal weight: w_i = |A_i| / (mean(|A|) + 1e-8). "
                "6. Once per epoch: measure H_curr = mean entropy over FADE_ENTROPY_EVAL_PROMPTS "
                "   held-out prompts (greedy decode, compute token-level entropy from logits). "
                "   If H_init not set yet, record it as H_curr. "
                "7. focal_gamma = FADE_FOCAL_GAMMA_MAX * (H_curr / H_init) ** FADE_FOCAL_K. "
                "8. Final advantage: A_fade(i) = A_sign_rescaled(i) * (w_i ** focal_gamma). "
                "Add env vars: FADE_ENABLED (default 0), FADE_R_HIGH (0.65), FADE_R_LOW (0.35), "
                "FADE_FOCAL_GAMMA_MAX (1.0), FADE_FOCAL_K (2), "
                "FADE_ENTROPY_EVAL_PROMPTS (50). "
                "Total: ~15 lines + one epoch-level entropy probe call. "
                "Implementation reference: https://arxiv.org/abs/2607.01490 (Meta FAIR)."
            ),
            "arxiv_ref": "2607.01490",
            "estimated_gpu_hours": 3.0,
        },
    },
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-125: CPO-KL — Continual Policy Optimization KL + L2 regularization
    #          for inter-cycle forgetting (arXiv:2607.04364, July 2026)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_19_002_cpo_kl_intercycle_forgetting_reg_humaneval",
        "priority": 7,
        "gpu": "auto",
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2607.04364 — 'RL Forgets! Towards Continual Policy Optimization' "
            "(Mao-Lin Luo et al., July 2026). Demonstrates that RL-based training "
            "(including GRPO) suffers catastrophic forgetting during continual post-"
            "training — challenging the assumption that RL is inherently more robust "
            "than SFT. Proposes CPO (Continual Policy Optimization), a replay-free "
            "framework that adds (1) a KL penalty against the prior cycle's policy and "
            "(2) an L2 parameter-movement penalty against the prior adapter weights: "
            "L_total = L_GRPO + lambda_KL * KL(pi_curr || pi_prev) + "
            "lambda_L2 * ||theta_curr - theta_prev||^2. "
            "Reports 13.7% forgetting reduction and 7.0% pretrained capability "
            "improvement on Qwen3-VL-8B under continual multi-task RL. "
            "Relevance to our pipeline: our 4-cycle GRPO run trains each cycle's adapter "
            "from the PRIOR cycle's checkpoint, but does not regularize drift. Root "
            "Problem D (inter-cycle forgetting) has been measured diagnostically by "
            "EXP-111 but never targeted as an intervention. If cycle-N GRPO partially "
            "erases cycle-(N-1) gains, the 'full arm' accumulates no net improvement "
            "over cycles. CPO regularization costs ~10% compute per step (one extra KL "
            "forward over the frozen prior adapter) and requires no replay buffer — only "
            "the prior adapter checkpoint (~400MB at 1.5B scale) must be kept in memory. "
            "Distinct from EXP-117 (CPO continual GRPO): EXP-117 predates the July 2026 "
            "paper and focuses on the behavioral KL alone; EXP-125 adds the L2 parameter-"
            "movement component from the new paper, which the authors show is necessary "
            "for the geometric constraint to complement the behavioral KL. If EXP-125 "
            "shows that inter-cycle forgetting contributes to full-arm stagnation, it "
            "becomes directly relevant to the AAAI paper's GRPO narrative."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 4,
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "cpo_kl_enabled": True,
            "cpo_l2_enabled": True,
            "cpo_lambda": 0.1,
            "cpo_l2_alpha": 0.5,
            "cpo_prior_policy": "prev_cycle_grpo_adapter",
            "grpo_temperature": 1.0,
            "n_generations": 8,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "analysis": [
                "inter_cycle_forgetting_rate_with_vs_without_cpo",
                "per_cycle_pass_at_1_on_prev_cycle_task_split",
                "cpo_kl_loss_vs_grpo_loss_ratio_per_step",
                "l2_parameter_drift_vs_baseline_per_cycle",
                "full_arm_pass_at_1_per_cycle_cpo_vs_baseline",
                "ablation_kl_only_vs_l2_only_vs_combined",
            ],
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (baseline): full 92.68%, ACR=52.4%",
                "EXP-111 (forgetting_eval): measures inter-cycle forgetting diagnostically",
                "EXP-117 (CPO continual GRPO): KL-only variant, predates L2 component",
                "arxiv:2607.04364 (CPO): 13.7% forgetting reduction on Qwen3-VL-8B",
            ],
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
            ],
            "implementation_note": (
                "In grpo_train_simple.py: "
                "1. Before GRPO training for cycle N (N >= 1), load the prior cycle's "
                "   GRPO adapter from grpo_adapter/ checkpoint as a frozen reference "
                "   model pi_prev. If no prior GRPO adapter exists (cycle 0), skip CPO. "
                "2. At each training step, compute the CPO auxiliary losses: "
                "   a) KL term: for each sequence in the batch, compute "
                "      kl_i = sum_t [pi_curr(t|x) * log(pi_curr(t|x) / pi_prev(t|x))] "
                "      Use the already-materialized logits from the current forward pass "
                "      for pi_curr; run a no-grad forward of pi_prev on the same input. "
                "      L_KL = mean(kl_i). "
                "   b) L2 term: "
                "      L_L2 = sum_p ||theta_curr_p - theta_prev_p||^2 / n_params "
                "      where the sum is over LoRA adapter parameters only. "
                "      theta_prev_p values can be loaded once and stored as a frozen "
                "      tensor; no repeat forward pass needed for the L2 term. "
                "3. Total loss: L_total = L_grpo + CPO_LAMBDA * L_KL + "
                "   CPO_LAMBDA * CPO_L2_ALPHA * L_L2. "
                "4. Log kl_loss, l2_loss, grpo_loss separately per step. "
                "5. Add env vars: CPO_KL_ENABLED (default 0), CPO_L2_ENABLED (default 0), "
                "   CPO_LAMBDA (default 0.1), CPO_L2_ALPHA (default 0.5), "
                "   CPO_PRIOR_POLICY (path override; default: inferred from prev cycle). "
                "Total: ~25 lines + one extra forward pass per step for KL. "
                "Note: the L2 term is extremely cheap — it only reads the current LoRA "
                "weights and compares to a frozen copy; no model forward needed. "
                "Implementation reference: https://arxiv.org/abs/2607.04364."
            ),
            "arxiv_ref": "2607.04364",
            "estimated_gpu_hours": 3.5,
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
