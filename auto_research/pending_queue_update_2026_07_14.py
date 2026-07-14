#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-14 (EXP-116, EXP-117).

A800 connectivity: offline since 2026-05-14 (day ~61). Apply when restored.

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

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_14.py            # EXP-116, EXP-117

Queue was ~123 pending on 2026-07-13 (+2 added: EXP-114, EXP-115).
Queue cap applied: >20 → max 2 new experiments.

Data motivating today's proposals
----------------------------------
results/e2e_4cyc_gpt55/cycle_3/e2e_ablation_summary.json (unchanged — A800 offline):
    HumanEval 4-cycle:
      large (always-large):        task_pass=96.34%, cost_vs_large=100%
      skills (always-small+proc):  task_pass=75.61%, cost_vs_large=10%
      router (logistic, cycle 3):  task_pass=92.68%, routing_acc=92.68%,
                                   cost_vs_large=27.56%, fallback=6.10%
      full (router+GRPO):          task_pass=92.68% ← SAME as router
    Root problems:
      (A) ACR=52.4% — 43/82 GRPO groups have zero within-group reward variance
          → zero gradient → Full = Router.
      (B) Non-collapsed groups (47.6%) = ALL GRPO learning signal; each must be
          used as efficiently as possible.
      (C) Skills gap: 75.61% vs large 96.34% — 20.7pp gap.

AAAI 2027 deadline: 2026-08-15 (32 days from today).
A800 offline since 2026-05-14 (day 61).

arxiv:2607.07976 — When Implausible Tokens Get Reinforced: Tail-Aware Credit
    Calibration for LLM Reinforcement Learning (July 9-11, 2026):

    PROBLEM THEY SOLVE:
    In standard GRPO (and any uniform advantage broadcast), ALL tokens in a
    winning rollout receive the same positive advantage A. But some tokens in
    a correct solution are "tail tokens" — they have very low probability under
    the policy (p_θ(t) << 1) and are contextually anomalous (e.g., an unusual
    variable name that happens to be syntactically correct, a coincidentally
    chosen loop bound). These tokens receive the same positive credit as the
    genuinely decision-critical tokens (the algorithmic choice of data structure,
    the boundary condition that decides pass vs fail).

    This "Positive-Credit Contamination" (PCC) effect causes the model to
    reinforce low-probability contextually-wrong tokens, which can destabilize
    training and degrade generation quality. PCC is distinct from:
      - Advantage collapse (zero-variance groups): PCC occurs WITHIN non-zero-
        variance groups, in the winning rollouts.
      - Token-level credit assignment (EGCA): EGCA finds the EARLIEST divergence
        point from an execution trace and zeros later tokens; TACO suppresses
        positive credit proportional to HOW IMPLAUSIBLE the token is, leaving the
        credit smooth (no binary mask). TACO requires no execution traces.

    TACO MECHANISM:
    For each token t at position i in winning rollout r:
      1. Estimate tail-risk: τ_i = softmax(-log p_θ(t_i)) clipped to [0, 1].
         (High τ_i = low p_θ = implausible token.)
      2. Compute scaled credit: A_i = A * (1 - α * τ_i)  where α = 0.3.
         (Implausible tokens receive 30% less credit than the base advantage.)
      3. For losing rollouts: negative advantage is applied uniformly (no
         suppression — we WANT to push away from all tokens in a losing rollout,
         regardless of implausibility).
    Total loss: L_TACO = -mean_i [ A_i * log p_θ(t_i) ].
    Reduces to standard GRPO when α=0 (no suppression).

    RESULTS:
    Evaluated across 3 LLMs (Qwen2.5-7B, Llama-3.1-8B, Mistral-7B) and
    8 benchmarks including GSM8K, MATH500, HumanEval, and MBPP.
    TACO consistently outperforms vanilla GRPO: +1.3–2.8pp on math,
    +0.8–1.5pp on code (HumanEval, MBPP). Training stability also improves
    (lower gradient variance, fewer loss spikes).

    RELEVANCE TO OUR NON-COLLAPSED GROUPS (47.6%):
    Our 40 non-collapsed GRPO groups (ACR=47.6%) are WHERE ALL OUR LEARNING
    HAPPENS — these are the only groups that produce non-zero gradient and
    push Full arm above the Router baseline. If each of these 40 groups suffers
    from PCC, the gradient direction is partially corrupted by credit on
    implausible tokens. TACO cleans up the credit signal in these groups,
    making each non-collapsed step more informative.

    ADAPTATION TO CODE (distinct from math):
    Code generation has specific tail-token patterns:
      - Low-probability variable names that are semantically equivalent
        (e.g., `result` vs the unusual `temp_accumulator`) — stylistic tail
        tokens that should receive no extra positive credit.
      - Syntactically unusual but correct constructs (e.g., `[i for i in range(n)]`
        at a position where the policy typically uses `list(range(n))`) — structural
        tail tokens that ARE semantically important.
    TACO's token-level softening with α=0.3 will automatically distinguish these:
    unusual variable names have low p_θ(t) for the SPECIFIC token, while unusual
    constructs have low p_θ(t) for the specific token BUT high importance via
    execution impact. TACO only softens at the token level; it does not zero out
    structurally important tokens.
    The result: gradient concentrates on the algorithmic and structural decisions
    rather than the stylistic anomalies.

    DISTINCT FROM EXISTING EXPERIMENTS:
    EXP-094 (AVSPO): injects virtual samples into COLLAPSED (zero-variance)
      groups. Does not affect non-collapsed groups. TACO and AVSPO are
      complementary: AVSPO expands gradient coverage to collapsed groups,
      TACO improves gradient quality in non-collapsed groups.
    EXP-108 (sign advantage, A=2r-1): changes the advantage FORMULA for
      all-fail groups (A=−1). Does not modify per-token credit in winning rollouts.
    EXP-EGCA (execution-grounded credit): uses execution divergence traces to
      binary-mask tokens after the first divergence. Requires running execution;
      no soft weighting; zeros out rather than suppresses.
    EXP-114 (RLCSD): contrastive on-policy self-distillation; provides dense
      gradient independent of rollout pass/fail via KL terms. Targets all-fail
      groups. No per-token implausibility weighting.
    EXP-116 (this): soft per-token credit calibration in winning rollouts of
      non-collapsed groups. No execution traces. Complements all the above.

    PREDICTED OUTCOME:
    Full arm pass@1 improves from 92.68% to 93.5–94.5% (+0.8–1.8pp).
    Skills arm pass@1 improves from 75.61% to 76.5–78% (better GRPO quality
    → warmed-up student for next cycle SFT).
    ACR stays at ~52.4% (TACO does not affect collapsed groups).
    Gradient variance in non-collapsed groups decreases by ~20% (training
    stability improvement per paper).

    COMPOSABILITY:
    TACO + AVSPO: AVSPO injects virtual samples for collapsed groups; TACO
      cleans credit in non-collapsed groups. These target disjoint group subsets.
    TACO + sign advantage: sign advantage changes A for all-fail groups; TACO
      softens credit in winning rollouts of non-collapsed groups. Non-overlapping.
    TACO + RLCSD: RLCSD provides KL-based dense signal; TACO suppresses
      implausible-token credit in standard GRPO path. Composable via addition
      of loss terms.
    A fully-composed run (AVSPO + sign advantage + RLCSD + TACO) is the natural
    EXP-118 if EXP-108/EXP-114/EXP-094/EXP-116 all show positive results.

arxiv:2607.04364 — RL Forgets! Towards Continual Policy Optimization (July 2026):

    PROBLEM THEY SOLVE:
    Prior work (arxiv:2507.05386, July 2025) showed that RL fine-tuning (RFT/GRPO)
    "naturally" mitigates forgetting compared to SFT, citing the on-policy anchor.
    The new paper (2607.04364) challenges this on a harder benchmark: MRCL
    (Multimodal Reasoning Continual Learning), which includes 7 diverse tasks and
    measures both backward transfer (forgetting of old tasks) and forward transfer
    (improvement on new tasks). Finding: RL fine-tuning DOES suffer from catastrophic
    forgetting under realistic multi-task diversity — the "natural mitigation" claim
    holds only within a narrow task distribution. When tasks shift significantly
    (different reasoning types, different domain distributions), GRPO updates on new
    tasks do interfere with parameters encoding old task knowledge.

    CPO (CONTINUAL POLICY OPTIMIZATION) MECHANISM:
    CPO is a replay-FREE continual RL framework. Instead of storing and replaying
    past rollouts, it regularizes the parameter update to preserve BEHAVIORAL
    SIMILARITY to the prior policy on tasks it previously learned:
      L_CPO = L_GRPO + β * KL(π_old(·|x_prior) || π_θ(·|x_prior))
    where x_prior is sampled from a small frozen MEMORY INDEX (a compressed,
    non-rollout representation of prior tasks — analogous to Kernel Herding
    for exemplar selection). The KL term penalizes deviating from the prior
    policy's OUTPUT DISTRIBUTION on prior task prompts, not its parameters
    (as EWC would). This makes CPO:
      - Replay-free: no rollouts from old tasks are stored or re-run.
      - Parameter-agnostic: no Fisher information matrix computation.
      - Compatible with LoRA (the KL operates on outputs, not on LoRA weights).
    On Qwen3-VL-8B: CPO reduces forgetting by 13.7% and improves pretrained
    capabilities by 7.0% relative to vanilla GRPO sequential training.

    RELEVANCE TO OUR MULTI-CYCLE PIPELINE:
    Our 4-cycle GRPO pipeline is EXACTLY the sequential continual RL setting
    CPO was designed for. Each cycle trains on new traces from the previous
    cycle's collection, potentially interfering with GRPO's learning from prior
    cycles. The forgetting diagnostic (EXP-115) will quantify how much cross-cycle
    forgetting occurs in our pipeline. If EXP-115 shows forgetting_rate > 2%,
    CPO is the prescribed remedy: add the behavioral KL penalty to GRPO in each
    cycle's training, with x_prior = the previous cycle's task prompts.

    DISTINCT FROM EWC (Elastic Weight Consolidation):
    EWC computes the Fisher information matrix (per-parameter importance) from
    prior task gradients and regularizes parameter changes proportionally.
    For LoRA adapters (~4M params), Fisher computation is feasible, but the
    approach is parameter-level — it doesn't know which behaviors to preserve,
    only which PARAMETERS matter. CPO is behavior-level (KL on outputs), which
    is more aligned with our objective: preserve task-pass rates on previously-
    learned HumanEval problems, even if the parameters that implement this shift.
    CPO's KL term also degrades gracefully as new tasks improve performance
    (the "prior policy" is updated at cycle boundaries), while EWC's Fisher
    matrix is static within a cycle.

    DISTINCT FROM EXP-111 (RFT vs SFT cycle retention, forgetting_eval):
    EXP-111 MEASURES forgetting by comparing RFT vs SFT methods.
    EXP-115 (yesterday) DIAGNOSES forgetting within the default GRPO pipeline.
    EXP-117 (this) MITIGATES forgetting using CPO during GRPO training.
    The three form a logical sequence: measure → diagnose → fix.

    IMPLEMENTATION (LoRA-compatible):
    1. At cycle k start, extract cycle k-1's task prompts (from traces.jsonl).
    2. Cache π_{k-1}'s output log-probs on those prompts (forward pass, no grad).
    3. During cycle k GRPO training: at each batch step, sample B_prior
       prompts from the memory index; compute KL(π_{k-1}(·|x_prior) ||
       π_θ(·|x_prior)) via forward passes through the CURRENT LoRA adapter.
    4. Add β * KL to L_GRPO (β=0.01 initially; sweep β ∈ {0.001, 0.01, 0.1}).
    5. Memory index size: 256 prompts (4× batch size; fits in 3 GB GPU RAM).
    6. Compute overhead: +B_prior/B_grpo ≈ +25% forward passes per step.
    Implementation: ~60 lines in grpo_train_simple.py.
    Est. time: ~3.5h per 4-cycle run (vs. 2.5h baseline for full pipeline).

    CONDITIONAL ON EXP-115:
    If EXP-115 (forgetting diagnostic) shows forgetting_rate < 2%, CPO is
    not needed and this experiment can be skipped. If forgetting_rate ≥ 2%,
    CPO is the minimum-overhead remedy (replay-free, LoRA-compatible) and
    should be run before AAAI deadline to add §5.6 "Continual GRPO with CPO"
    to the paper.

    PREDICTED OUTCOME (if forgetting_rate ≥ 2%):
    Full arm pass@1 improves from 92.68% to 93.5–94.5% (+0.8–1.8pp) by
    recovering tasks that were previously forgotten between cycles.
    Cycle-3 skills arm pass@1 improves from 75.61% to 77–78% (retention
    of improvements from earlier cycles).
    CPO achieves this without replay buffer or Fisher matrix overhead.
"""
import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = (
    "/data0/home/zeyuwang/router-skills-evolve-data/humaneval/HumanEval.jsonl.gz"
)
_CYCLE_DIR = "/data0/home/zeyuwang/router-skills-evolve-data/e2e_4cyc_gpt55"

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_07_14_001_taco_tail_aware_credit_calibration_grpo_humaneval",
        "priority": 7,
        "kind": "grpo_continual",
        "gpu": "auto",
        "rationale": (
            "arxiv:2607.07976 — When Implausible Tokens Get Reinforced: Tail-Aware "
            "Credit Calibration for LLM Reinforcement Learning (July 9-11, 2026). "
            ""
            "THE CORE PROBLEM THIS SOLVES: "
            "In standard GRPO, ALL tokens in a winning rollout receive the same positive "
            "advantage A. Some tokens in a correct solution are 'tail tokens' — low "
            "p_θ(t) tokens that are contextually anomalous (unusual variable name, "
            "coincidental loop bound) but syntactically valid. These receive the same "
            "positive credit as decision-critical tokens (data structure choice, boundary "
            "conditions). This 'Positive-Credit Contamination' (PCC) corrupts gradient "
            "direction in the winning rollouts of non-collapsed groups. "
            ""
            "RELEVANCE TO OUR NON-COLLAPSED GROUPS (47.6% = 39/82 groups): "
            "ACR=52.4% means 43/82 groups are collapsed (zero gradient). The remaining "
            "39 non-collapsed groups are WHERE ALL LEARNING HAPPENS. If each of these "
            "suffers from PCC, our gradient is partially misdirected. TACO fixes the "
            "credit signal in exactly these groups, making each non-collapsed update "
            "more informative. "
            ""
            "TACO MECHANISM (drop-in modification to advantage broadcast): "
            "For token t at position i in winning rollout r: "
            "  τ_i = softmax(-log p_θ(t_i)), clipped to [0, 1]  (tail-risk score) "
            "  A_i = A * (1 - α * τ_i)  with α=0.3 (suppressed credit for tail tokens) "
            "For losing rollouts: advantage broadcast unchanged (push away from all tokens). "
            "L_TACO = -mean_i[ A_i * log p_θ(t_i) ]. "
            "Reduces to standard GRPO when α=0. No execution traces required. "
            "~+10% compute (one extra forward pass to compute p_θ logits for τ_i, "
            "which are already computed during the loss — O(0) overhead in practice). "
            ""
            "ADAPTATION TO CODE: "
            "Code has two tail-token patterns: "
            "  (a) Stylistic tail tokens: unusual variable names, spacing choices — "
            "      low p_θ, low semantic importance. TACO suppresses these. GOOD. "
            "  (b) Structural tail tokens: unusual constructs (list comprehension vs "
            "      loop) that ARE algorithmically important but low p_θ initially. "
            "      TACO only partially suppresses these (α=0.3, not zeroed). SAFE. "
            "Residual credit at 70% (α=0.3) preserves gradient for structurally "
            "important unusual tokens while reducing contamination from stylistic ones. "
            ""
            "PAPER RESULTS: "
            "+1.3–2.8pp on math (GSM8K, MATH500); +0.8–1.5pp on code (HumanEval, MBPP). "
            "Lower gradient variance, fewer loss spikes across 3 LLMs × 8 benchmarks. "
            "Results on Qwen2.5-7B, Llama-3.1-8B, Mistral-7B. "
            "We use Qwen2.5-Coder-1.5B-Instruct (not in their eval) — smaller model "
            "may benefit proportionally more (stronger PCC effect at lower capacity). "
            ""
            "DISTINCT FROM ALL EXISTING EXPERIMENTS: "
            "EXP-094 (AVSPO): injects virtual samples into COLLAPSED groups. "
            "  TACO targets NON-COLLAPSED groups — complementary, disjoint targets. "
            "EXP-108 (sign advantage): advantage formula change for all-fail groups. "
            "  TACO modifies per-token credit in WINNING rollouts — orthogonal. "
            "EXP-EGCA (exec-grounded credit): binary mask after execution divergence. "
            "  Requires execution; hard zeros vs. TACO's soft suppression. "
            "EXP-114 (RLCSD): KL-based dense signal for all groups via contrastive hints. "
            "  Different mechanism; TACO and RLCSD composable (loss additivity). "
            "EXP-116 (this): soft per-token implausibility-based credit suppression, "
            "  no execution, targets winning rollouts in non-collapsed groups only. "
            ""
            "PREDICTED OUTCOME: "
            "  Full arm pass@1: 92.68% → 93.5–94.5% (+0.8–1.8pp, breaking Full=Router). "
            "  Skills arm pass@1: 75.61% → 76.5–78% (better GRPO update quality per step). "
            "  ACR stays at ~52.4% (TACO does not affect collapsed groups). "
            "  Gradient variance in non-collapsed groups: ↓~20%. "
            ""
            "COMPOSABILITY (natural EXP-118): "
            "TACO + AVSPO: disjoint targets (collapsed vs non-collapsed groups). "
            "TACO + sign advantage: disjoint (all-fail advantage vs winning-token credit). "
            "TACO + RLCSD: additive loss terms (no interaction). "
            "If EXP-094 (AVSPO), EXP-108 (sign), EXP-114 (RLCSD), EXP-116 (TACO) all "
            "positive, full composition (EXP-118) is the natural ceiling experiment. "
        ),
        "spec": {
            "pipeline": "humaneval_taco_tail_aware_credit_calibration",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 1,
            "grpo_algo": "taco_dapo",
            "taco_alpha": 0.3,
            "taco_tail_risk_fn": "softmax_neg_log_prob",
            "taco_clip_tau_min": 0.0,
            "taco_clip_tau_max": 1.0,
            "taco_suppress_losing_rollouts": False,
            "n_generations": 8,
            "grpo_temperature": 1.0,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (DAPO): full 92.68%, skills 75.61%",
                "EXP-094 (AVSPO): targets collapsed groups; TACO targets non-collapsed",
                "EXP-108 (sign advantage): advantage formula change, not per-token credit",
                "EXP-EGCA: execution-trace binary mask vs TACO soft weighting",
                "arxiv:2607.07976 (TACO): +0.8-1.5pp code on Qwen2.5-7B/Llama-8B",
            ],
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
            ],
            "implementation_note": (
                "In grpo_train_simple.py, in the advantage broadcast step: "
                "after computing per-group advantage A for each rollout, before "
                "computing the policy loss, for WINNING rollouts (r > mean_reward): "
                "  tau_i = clamp(softmax(-log_probs[i]), 0, 1)  # tail-risk "
                "  A_i = A * (1 - 0.3 * tau_i)                 # per-token credit "
                "loss_i = -A_i * log_probs[i]                   # TACO loss "
                "For LOSING rollouts: apply standard uniform A (no suppression). "
                "log_probs are already computed during the forward pass → O(0) overhead. "
                "~15 lines. No extra forward passes. No execution traces. "
            ),
            "arxiv_ref": "2607.07976",
            "estimated_gpu_hours": 2.5,
        },
    },
    {
        "id": "exp_2026_07_14_002_cpo_continual_policy_optim_multicycle_grpo_humaneval",
        "priority": 6,
        "kind": "grpo_continual",
        "gpu": "auto",
        "rationale": (
            "arxiv:2607.04364 — RL Forgets! Towards Continual Policy Optimization "
            "(July 2026). "
            ""
            "BACKGROUND — CHALLENGING PRIOR WORK: "
            "arxiv:2507.05386 (July 2025, cited in our paper as supporting GRPO): "
            "showed RFT 'naturally' mitigates forgetting vs SFT, attributed to the "
            "on-policy anchor. EXP-115 (yesterday) queues a diagnostic to check "
            "this for our pipeline. "
            "arxiv:2607.04364 (July 2026) challenges that finding on a harder benchmark: "
            "MRCL (7 diverse task types). Key finding: RL fine-tuning DOES suffer "
            "catastrophic forgetting when tasks diverge significantly between cycles — "
            "the 'natural mitigation' holds only within narrow task distributions. "
            "Our multi-cycle HumanEval pipeline trains on traces that shift per cycle "
            "(harder tasks surface as the policy improves), making this a non-trivial "
            "concern particularly for cycles 2–4 where task difficulty distribution "
            "shifts most from cycle 0. "
            ""
            "CPO MECHANISM (replay-free behavioral KL regularization): "
            "CPO adds a behavioral KL term to GRPO that keeps the current policy "
            "close to its prior-cycle output distribution on previously-seen prompts: "
            "  L_CPO = L_GRPO + β * KL(π_{k-1}(·|x_prior) || π_θ(·|x_prior)) "
            "where x_prior is sampled from a MEMORY INDEX (small frozen prompt set "
            "from cycle k-1, selected via kernel herding). "
            "KEY PROPERTIES: "
            "  Replay-free: no old rollouts stored. Memory index = prompts only. "
            "  Parameter-agnostic: no Fisher information matrix. "
            "  LoRA-compatible: KL operates on outputs, not weights. "
            "  Graceful updating: memory index advances at cycle boundaries. "
            "On Qwen3-VL-8B (7 tasks): −13.7% forgetting, +7.0% pretrained capability. "
            ""
            "RELEVANCE TO OUR 4-CYCLE PIPELINE: "
            "Our pipeline runs 4 GRPO cycles, each on new traces from the prior cycle. "
            "Without CPO: cycle k GRPO updates on the hardest unresolved tasks; "
            "  those updates may push parameters away from easy-task representations "
            "  learned in cycle k-1. "
            "With CPO: the behavioral KL term anchors cycle k's policy to cycle k-1's "
            "  output distribution on the cycle k-1 task prompts, preventing regression "
            "  on previously-learned tasks while still improving on new ones. "
            ""
            "RELATION TO EXISTING EXPERIMENTS: "
            "EXP-111 (RFT vs SFT cycle retention): MEASURES forgetting — which method "
            "  causes less forgetting? Prescriptive comparison. "
            "EXP-115 (forgetting diagnostic): DIAGNOSES forgetting within our default "
            "  pipeline — how much occurs, which tasks are affected? "
            "EXP-117 (this): MITIGATES forgetting using CPO — adds behavioral KL "
            "  regularization to prevent regression between cycles. "
            "EXP-111 → EXP-115 → EXP-117 form a logical measure-diagnose-fix chain. "
            ""
            "CONDITIONAL EXECUTION RECOMMENDATION: "
            "  If EXP-115 shows forgetting_rate < 2%: skip EXP-117 (not needed). "
            "  If EXP-115 shows forgetting_rate ≥ 2%: run EXP-117 before AAAI deadline. "
            "  If EXP-115 cannot run before July 26 (6 days before A800 'minviable "
            "  restore' of July 20): run EXP-117 unconditionally as a safety measure. "
            "  The compute cost is low (3.5h) and the benefit is guaranteed if forgetting "
            "  is present; if forgetting is absent, CPO is a no-op (β=0.01 is small). "
            ""
            "DISTINCT FROM EWC: "
            "EWC regularizes at the PARAMETER level (Fisher information × ||θ - θ*||²). "
            "CPO regularizes at the BEHAVIOR level (KL between output distributions). "
            "For LoRA adapters, EWC needs Fisher for LoRA weights only — computationally "
            "feasible but still parameter-level. CPO's output-level KL is more aligned "
            "with our actual objective (preserve task pass rates, not preserve weights). "
            "If the model learns to represent task solutions in different weight "
            "configurations across cycles (a common LoRA adaptation pattern), CPO "
            "gracefully handles this while EWC would over-constrain. "
            ""
            "IMPLEMENTATION: "
            "1. At cycle k start: collect memory index M_k = 256 prompts from "
            "   cycle k-1 traces.jsonl (kernel herding: maximize prompt diversity). "
            "2. Cache π_{k-1} log-probs on M_k (forward pass, no grad, ~30s on A800). "
            "3. During cycle k GRPO: every step, sample B_prior=16 prompts from M_k, "
            "   compute KL(π_{k-1}(·|x) || π_θ(·|x)) via π_{k-1} log-probs (cached) "
            "   and π_θ log-probs (current model forward pass). "
            "4. L_total = L_GRPO + β * KL. Sweep β ∈ {0.001, 0.01, 0.1} across seeds. "
            "5. Output: per-cycle pass@1 matrix (same format as EXP-115 diagnostic), "
            "   forgetting_rate_cpo, compare to baseline EXP-115 forgetting_rate_vanilla. "
            "Memory index overhead: 256 × 256 tokens × fp32 log-probs = ~67 MB GPU RAM. "
            "Extra forward pass overhead: B_prior/B_grpo = 16/4 = +4 fwd passes per step. "
            "Est. wall-clock: 3.5h per 4-cycle run (vs. 2.5h baseline). "
            ""
            "PREDICTED OUTCOME (assuming forgetting_rate ≥ 2% per EXP-115): "
            "  Cycle-3 full arm pass@1: 92.68% → 93.5–95% (CPO recovers forgotten tasks). "
            "  Skills arm pass@1: 75.61% → 77–79% (retained cycle-1/2 improvements). "
            "  CPO adds §5.6 to AAAI paper: 'Continual GRPO with CPO reduces cross-cycle "
            "  forgetting by X%, improving full arm by Y pp'. "
            "  Novel finding: multi-cycle code GRPO suffers from cross-cycle forgetting; "
            "  CPO is a practical, replay-free remedy compatible with LoRA. "
        ),
        "spec": {
            "pipeline": "humaneval_cpo_continual_policy_optim",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 4,
            "grpo_algo": "cpo_dapo",
            "cpo_beta_sweep": [0.001, 0.01, 0.1],
            "cpo_beta_default": 0.01,
            "cpo_memory_index_size": 256,
            "cpo_memory_selection": "kernel_herding",
            "cpo_prior_batch_size": 16,
            "cpo_kl_direction": "fwd_kl_prior_old",
            "n_generations": 8,
            "grpo_temperature": 1.0,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "analysis": [
                "per_task_pass_fail_matrix_cpo_vs_vanilla",
                "forgetting_rate_cpo",
                "forgetting_rate_reduction_vs_baseline",
                "beta_sensitivity_full_arm_pass_at_1",
            ],
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (baseline): full 92.68%",
                "EXP-115 (forgetting diagnostic): quantifies baseline forgetting",
                "EXP-111 (RFT vs SFT method comparison): prescriptive baseline",
                "arxiv:2607.04364 (CPO): -13.7% forgetting on Qwen3-VL-8B (7 tasks)",
            ],
            "conditional_execution": (
                "Run if EXP-115 shows forgetting_rate >= 2%. "
                "Run unconditionally if EXP-115 cannot complete before 2026-07-26. "
                "Skip if EXP-115 shows forgetting_rate < 2%."
            ),
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
            ],
            "implementation_note": (
                "In grpo_train_simple.py: "
                "Before cycle k training: forward-pass π_{k-1} checkpoint on M_k prompts "
                "and cache log_probs_prior (saved to disk). "
                "In each training step: sample B_prior=16 prompts from M_k; forward-pass "
                "current π_θ on those prompts; compute KL = sum(exp(lp_prior) * (lp_prior - lp_theta)). "
                "Add beta * KL.mean() to L_GRPO. ~60 lines including memory index management. "
                "Uses same LoRA adapter infrastructure as existing GRPO loop. "
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
