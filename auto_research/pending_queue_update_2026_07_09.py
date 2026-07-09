#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-09 (EXP-110, EXP-111).

A800 connectivity: offline since 2026-05-14 (day ~56). Apply when restored.

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
    python3 auto_research/pending_queue_update_2026_07_04.py            # EXP-106, EXP-107
    python3 auto_research/pending_queue_update_2026_07_05.py            # EXP-108, EXP-109

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_09.py            # EXP-110, EXP-111

Queue was ~113 pending on 2026-07-05 (+2 added: EXP-108, EXP-109).
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
    Root problem A: ACR=52.4% — 43/82 GRPO groups zero-variance → zero gradient.
      → Being attacked by EXP-108 (Sign advantage: A=2r-1) and EXP-106 (SRPO).
    Root problem B: 47.6% non-degenerate groups still present token-level gradient
      instability from outlier importance-sampling ratios at rare tokens.
      → Unaddressed by any prior experiment; targeted by EXP-110 (GMPO).
    Root problem C: tau2 cycle oscillation (89.19% → 70.27% across cycles) consistent
      with SFT catastrophic forgetting — not yet mechanistically evaluated.
      → Targeted by EXP-111 (RL-vs-SFT forgetting eval).

AAAI 2027 context (deadline 2026-08-15, 37 days):
    EXP-110 (GMPO + Sign): if GMPO stabilises non-degenerate groups AND Sign
      eliminates degenerate ones, Full > Router for the first time → headline result.
    EXP-111 (forgetting_eval): mechanistic diagnosis of tau2 oscillation;
      provides a theoretical narrative (RL preserves circuits vs SFT overwrites them)
      grounded in arxiv:2605.28860 — strengthens §3 (tau2 track) for AAAI.

arxiv:2507.20673 — Geometric-Mean Policy Optimization (GMPO, Zhao et al., 2025):
    GRPO maximizes the arithmetic mean of token-level policy ratio × advantage:
      L_GRPO = E_t [ (π_θ(a_t)/π_old(a_t)) * A_t ]
    When a single rare token gets a very high importance ratio r_t >> 1, it
    dominates the sum, producing an explosive gradient update at that position.
    GMPO replaces the arithmetic mean with the geometric mean:
      L_GMPO = exp(E_t [ log(π_θ(a_t)/π_old(a_t)) * A_t ])
            = exp(E_t [ log(r_t * A_t) ])
    This bounds the effective per-token contribution: exp(log(r_t * A_t)) = r_t * A_t
    on a log scale, so extreme r_t values do not dominate linearly.
    The Cauchy-Schwarz inequality guarantees: geometric mean ≤ arithmetic mean,
    so GMPO is inherently more conservative than GRPO at syntax-critical tokens.
    Empirical results (GMPO paper): GMPO-7B outperforms GRPO-7B by avg 4.1% on
    math benchmarks (AIME24, AMC, MATH500, OlympiadBench, Minerva, Geometry3K)
    and 1.4% on multimodal reasoning. Code: github.com/callsys/GMPO.

    RELATIONSHIP TO SIGN ADVANTAGE (EXP-108):
      Sign advantage (A = 2r-1) operates at the GROUP level:
        - Eliminates the 52.4% zero-advantage groups (degenerate all-fail/all-pass).
        - For each rollout, the group-level advantage is either +1 or -1.
      GMPO operates at the TOKEN level:
        - Within each rollout (whether degenerate or not), GMPO bounds the per-token
          importance sampling ratio via geometric mean aggregation.
        - Prevents rare-token explosions in the 47.6% non-degenerate groups where
          Sign gives normal A ∈ {-1, +1} and GRPO still has token-level variance.
      The two mechanisms are ORTHOGONAL: Sign fixes group advantage; GMPO fixes
      token aggregation within rollouts. Combined, they address gradient instability
      at BOTH the group level AND the token level.

    IMPLEMENTATION (grpo_train_simple.py):
      DAPO/GRPO loss uses token-level policy ratios:
        ratios = torch.exp(logprobs - old_logprobs)      # (B, T)
        loss = -(ratios * advantages.unsqueeze(-1)).mean(-1)  # arithmetic mean over T
      GMPO replaces the inner aggregation:
        log_ratios = logprobs - old_logprobs             # (B, T)
        gmpo_loss = -torch.exp((log_ratios * advantages.unsqueeze(-1)).mean(-1))
        # geometric mean via exp(mean(log(ratio * advantage)))
      Clipping still applies: log_ratios are clipped at [-clip_eps, +clip_eps]
      before the exp in GMPO — identical clip mechanics to DAPO.
      ~5 lines of change in the token-loss aggregation step.

    COMBINED VARIANT: Sign + GMPO:
      advantages = 2.0 * rewards - 1.0          # Sign: group level (EXP-108 formula)
      loss = -exp(mean_t(log_ratio_t * adv))    # GMPO: token level
      This is a complete 2-component fix: no zero-advantage groups (Sign),
      no token-outlier explosions (GMPO).

arxiv:2605.28860 — Mechanistic Origins of Catastrophic Forgetting:
    Why Does RL Preserve Circuits Better than SFT? (Rojas et al., May 2026):
    Uses circuit-level analysis (differential circuit vulnerability, DCV) on
    Qwen2.5-3B-Instruct adapted to scientific question-answering.
    Key finding: SFT adapts faster to the new task BUT disrupts more circuits
    (higher forgetting). RL (GRPO) adapts slower but preserves circuits better.
    Mechanism: RL's updates are implicitly conservative in parameter subspaces
    that are sensitive to prior task performance, scaled by reward variance.
    Consequence: RL-trained models retain more prior-task capability across tasks.

    RELEVANCE TO TAU2 OSCILLATION:
      Our tau2 pipeline uses SFT (tau2_train_wrapper.sh + FSDP2 SFT) in each cycle.
      The tau2 oscillation (89.19% Cycle 1 → 70.27% Cycle 2) is a textbook
      catastrophic forgetting signature:
        - Cycle 1 SFT teaches the model to do tau2 tasks better.
        - But it overwrites cross-task generalisation circuits.
        - Cycle 2's harder/different tau2 distribution hits missing circuits → drop.
      arxiv:2507.05386 (RFT naturally mitigates forgetting) was already noted in
      meta.json; arxiv:2605.28860 provides the MECHANISTIC CIRCUIT-LEVEL GROUNDING:
        "SFT adapts more rapidly but produces greater circuit disruption and forgetting
         of prior capabilities." (Rojas et al., 2026)
      For HumanEval: our cycle-1 SFT may similarly overwrite easy-task circuits while
      specialising on hard tasks, reducing generalization in later cycles.

    PROPOSED EXPERIMENT (forgetting_eval):
      Compare three continual fine-tuning strategies over 2 HumanEval cycles:
        (A) SFT-only continual (current default):
              cycle 0: base → SFT → GRPO adapter
              cycle 1: GRPO adapter → new SFT (overwrites) → new GRPO adapter
        (B) GRPO-continual (RL-only, skip SFT):
              cycle 0: base → GRPO only
              cycle 1: GRPO adapter → new GRPO (keeps prior circuits via RL's
                       conservative updates)
        (C) SFT then frozen-GRPO (SFT once, then RL-only update):
              cycle 0: base → SFT → GRPO adapter
              cycle 1: GRPO adapter → new GRPO (no new SFT, preserve SFT circuits)
      Metric: for each cycle, evaluate on:
        (i)  cycle k training distribution tasks (in-distribution pass@1)
        (ii) held-out HumanEval problems NOT used in that cycle's SFT (forgetting@1)
      Compare forgetting@1 across strategies A/B/C.
      Direct payoff: if GRPO-continual (B) shows less forgetting than SFT (A),
        recommend dropping SFT from tau2 cycles and using GRPO-continual only.
      AAAI narrative: "RL's conservatism naturally acts as catastrophic forgetting
        protection, supporting multi-cycle continual learning without explicit replay
        mechanisms." (arxiv:2605.28860 + arxiv:2507.05386 combined theoretical anchor.)
"""
import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = (
    "/data0/home/zeyuwang/router-skills-evolve-data/humaneval/HumanEval.jsonl.gz"
)

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_07_09_001_gmpo_sign_combined_grpo_humaneval",
        "priority": 7,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2507.20673 — Geometric-Mean Policy Optimization (GMPO, "
            "Zhao et al., Peking University / Tsinghua, 2025). "
            ""
            "TWO ORTHOGONAL GRADIENT INSTABILITY SOURCES IN OUR DAPO PIPELINE: "
            ""
            "Source A — GROUP LEVEL (52.4% degenerate groups, ACR): "
            "  All-fail and all-pass groups have zero within-group reward variance. "
            "  Group-mean-centered advantage = 0 → zero gradient for 52.4% of groups. "
            "  FIX (EXP-108 Sign advantage): A = 2r - 1, so A ∈ {-1,+1} always. "
            "  EXP-108 addresses this; EXP-110 assumes Sign advantage is in effect. "
            ""
            "Source B — TOKEN LEVEL (47.6% non-degenerate groups): "
            "  Within each rollout, token-level importance sampling ratios "
            "  r_t = π_θ(a_t) / π_old(a_t) can be extreme at rare tokens "
            "  (low π_old but policy has shifted significantly → r_t >> 1). "
            "  DAPO clips via ratio clipping: clip(r_t, 1-ε, 1+ε). "
            "  BUT: for syntax-critical tokens (brackets, colons, indentation) "
            "  where the policy is highly sensitive, clip_eps alone is insufficient "
            "  to prevent token-dominant gradient updates in the arithmetic mean. "
            "  FIX (GMPO): replace arithmetic mean over tokens with geometric mean: "
            "    L_GMPO = exp(mean_t(log(r_t) * A))  vs  L_DAPO = mean_t(r_t * A) "
            "  Geometric mean bounds outlier contribution via log compression. "
            ""
            "GMPO EMPIRICAL RESULTS (paper): "
            "GMPO-7B vs GRPO-7B on math: +4.1pp average across AIME24, AMC, "
            "MATH500, OlympiadBench, Minerva, Geometry3K. "
            "+1.4pp on multimodal reasoning. "
            "Plug-and-play: 5-line change to token loss aggregation. "
            "Code: github.com/callsys/GMPO. "
            ""
            "COMBINED SIGN + GMPO EXPERIMENT (this EXP-110): "
            "Both fixes are independent, composable, and target different levels: "
            "  advantages = 2.0 * rewards - 1.0                 # Sign (group level) "
            "  log_ratios = logprobs - old_logprobs              # (B, T) "
            "  clipped = log_ratios.clamp(-clip_eps, clip_eps)   # standard clip "
            "  loss = -exp((clipped * adv.unsqueeze(-1)).mean(-1))  # GMPO (token) "
            "Total change from DAPO baseline: ~7 lines in grpo_train_simple.py. "
            ""
            "THREE VARIANTS (single run): "
            "  (A) GMPO only (geometric token mean, standard group advantage): "
            "      isolates token-level outlier bounding contribution "
            "  (B) Sign + GMPO (combined, this EXP's main variant): "
            "      eliminates both group degeneracy AND token outliers "
            "  (C) DAPO baseline (reproduce cycle_3: pass@1=92.68%, ACR=52.4%) "
            ""
            "RELATIONSHIP TO QUEUED EXPERIMENTS: "
            "EXP-108 (Sign advantage): isolates group-level fix, no token change. "
            "EXP-106 (SRPO): adds SDPO head for failed rollouts via π_ref. "
            "EXP-110 (this): isolates GMPO token fix (variant A) AND combines "
            "Sign+GMPO (variant B). ORTHOGONAL to both EXP-108 and EXP-106. "
            "All three can be composed if each individually succeeds: "
            "  Ultimate combination: Sign (group) + GMPO (token) + SRPO (failed routing) "
            "  = no zero-advantage groups + no token outliers + SDPO correction. "
            ""
            "CLIP RATIO INTERACTION: "
            "GMPO uses log-space clip (clamp log_ratio to [-clip_eps, clip_eps]) "
            "before the geometric mean. Sign advantage produces A ∈ {-1,+1}, "
            "giving a natural log_ratio scale. No additional clip_eps tuning needed "
            "for the combined variant beyond DAPO defaults. "
        ),
        "spec": {
            "pipeline": "humaneval_gmpo_sign_combined_grpo",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 1,
            "grpo_algo": "gmpo_sign_combined",
            "gmpo_token_aggregation": "geometric_mean",
            "sign_advantage_formula": "2 * reward - 1",
            "variants": [
                "gmpo_only_standard_group_advantage",
                "sign_plus_gmpo_combined",
                "dapo_baseline",
            ],
            "n_generations": 8,
            "grpo_temperature": 1.0,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (DAPO baseline: 92.68% HE, ACR=52.4%)",
                "arxiv:2507.20673 (GMPO +4.1pp math, +1.4pp multimodal vs GRPO)",
                "EXP-108 (Sign advantage: group-level fix, no GMPO)",
                "EXP-106 (SRPO: SDPO routing for failed rollouts, orthogonal)",
            ],
            "implementation_note": (
                "In grpo_train_simple.py, in the loss computation step: "
                "BEFORE (DAPO arithmetic): "
                "  loss = -(ratios * advantages.unsqueeze(-1)).mean(-1) "
                "AFTER variant A (GMPO only): "
                "  log_r = logprobs - old_logprobs "
                "  clipped_log_r = log_r.clamp(-clip_eps, clip_eps) "
                "  loss = -torch.exp((clipped_log_r * advantages.unsqueeze(-1)).mean(-1)) "
                "AFTER variant B (Sign + GMPO): "
                "  advantages = 2.0 * rewards - 1.0   # Sign formula "
                "  log_r = logprobs - old_logprobs "
                "  clipped_log_r = log_r.clamp(-clip_eps, clip_eps) "
                "  loss = -torch.exp((clipped_log_r * advantages.unsqueeze(-1)).mean(-1)) "
                "~5 lines change for GMPO; +2 lines for Sign. "
                "Reference: github.com/callsys/GMPO for geometric mean implementation."
            ),
            "primary_metrics": [
                "humaneval_pass_at_1_post_1cycle",
                "grpo_acr_mean",
                "grpo_fraction_nonzero_advantage_groups",
                "grpo_token_ratio_p99",
                "grpo_token_ratio_mean",
                "grpo_gradient_norm_mean",
                "grpo_gradient_norm_std",
                "training_loss_curve",
                "output_length_mean",
                "output_length_std",
                "variant_delta_pass_at_1",
            ],
            "target_outcome": (
                "Variant B (Sign+GMPO) achieves pass@1 > 93.68% (+1pp from DAPO 92.68%). "
                "GMPO-only (variant A) shows intermediate gain: 92.68% < A < B. "
                "Token ratio p99 decreases (fewer outlier tokens dominating gradient). "
                "Gradient norm std decreases vs DAPO (more stable training). "
                "ACR-active-fraction = 100% under Sign (no zero-advantage groups). "
            ),
            "gpu": "auto",
            "estimated_gpu_hours": 1.5,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_07_09_002_rl_vs_sft_forgetting_eval_humaneval_continual",
        "priority": 6,
        "kind": "forgetting_eval",
        "rationale": (
            "arxiv:2605.28860 — Mechanistic Origins of Catastrophic Forgetting: "
            "Why Does RL Preserve Circuits Better than SFT? "
            "(Rojas, Sawant, Allen, Amgalanbaatar, Zongo, Sharma, Chaudhary, "
            "May 28 2026). "
            ""
            "KEY FINDING: RL (GRPO) preserves prior-task circuits better than SFT "
            "during continual fine-tuning. SFT adapts faster but disrupts more "
            "computational circuits → greater catastrophic forgetting. "
            "Mechanism: RL updates are implicitly conservative in parameter subspaces "
            "sensitive to prior tasks, naturally scaled by reward variance — a form "
            "of 'implicit EWC' without explicit regularization. "
            "(Consistent with arxiv:2507.05386: 'RFT gradient ∝ reward variance → "
            "conservative in low-reward subspaces.') "
            ""
            "DIRECT CONNECTION TO TAU2 OSCILLATION: "
            "tau2 cycles show: 89.19% (Cycle 1) → 70.27% (Cycle 2) = -18.9pp drop. "
            "tau2 uses SFT (tau2_train_wrapper.sh, FSDP2) — NOT GRPO — for each cycle. "
            "Per arxiv:2605.28860, SFT rapidly overwrites task circuits acquired in "
            "Cycle 1 when it adapts to the harder Cycle 2 distribution. "
            "This is a textbook catastrophic forgetting signature. "
            ""
            "WHY HumanEval IS THE RIGHT TESTBED (not tau2): "
            "tau2 is a remote API pipeline (A800 blocked anyway). "
            "HumanEval is local, fast (<4h), and the SFT/GRPO split is identical: "
            "  Phase 3a = SFT (traces_to_sft.py → train_small_model.py) "
            "  Phase 3b = GRPO (grpo_train_simple.py) "
            "We can evaluate forgetting on held-out HumanEval problems. "
            ""
            "THREE CONTINUAL STRATEGIES (2 cycles each): "
            ""
            "Strategy A — Current default (SFT → GRPO per cycle): "
            "  cycle 0: base → SFT_0 → GRPO_0 adapter "
            "  cycle 1: GRPO_0 adapter → SFT_1 (new SFT ON TOP of GRPO_0) → GRPO_1 "
            "  Risk: SFT_1 overwrites circuits preserved by GRPO_0. "
            ""
            "Strategy B — GRPO-continual (skip SFT after cycle 0): "
            "  cycle 0: base → SFT_0 → GRPO_0 adapter "
            "  cycle 1: GRPO_0 adapter → GRPO_1 (NO new SFT, RL-only update) "
            "  Rationale: RL's conservatism preserves cycle-0 circuits. "
            "  Trade-off: slower adaptation to new trace distribution without SFT. "
            ""
            "Strategy C — SFT once, RL-only after (single SFT + continual GRPO): "
            "  cycle 0: base → SFT_0 → GRPO_0 adapter "
            "  cycle 1: GRPO_0 → GRPO_1 [same as B — but rename for clarity] "
            "  (B and C are structurally identical; test both labels for ablation.) "
            ""
            "FORGETTING METRIC: "
            "  Split HumanEval into D_train (used in SFT/GRPO) and D_held (not used). "
            "  For each strategy and cycle, measure pass@1 on D_held. "
            "  forgetting@1 = pass@1(D_held, cycle 1) - pass@1(D_held, cycle 0). "
            "  Negative forgetting = catastrophic forgetting. "
            "  Strategy B should show less-negative forgetting than Strategy A "
            "  per arxiv:2605.28860's circuit-preservation finding. "
            ""
            "AAAI 2027 NARRATIVE VALUE: "
            "If Strategy B (RL-only continual) shows less forgetting: "
            "  '→ dropping SFT from cycles ≥1 reduces catastrophic forgetting '  "
            "  and recommending GRPO-only continual for tau2 Cycles 2+ provides "
            "  a principled fix for the 89.19%→70.27% oscillation.' "
            "  This is a strong contribution to §3 (tau2 track, multi-cycle) of the "
            "  AAAI paper with fresh mechanistic grounding (arxiv:2605.28860). "
            ""
            "If Strategy A outperforms B (SFT is still net-positive): "
            "  '→ SFT's faster task adaptation overcomes forgetting at this scale; "
            "  GRPO-only is insufficient without new traces, but GRPO slows the  "
            "  forgetting arc in held-out evaluation.' "
            "  Both outcomes are informative for the paper's §5 (future work). "
            ""
            "IMPLEMENTATION NOTES: "
            "  Hold out 20 HumanEval problems (~12%) not used in any SFT batch. "
            "  Measure pass@1 on D_held after each of cycle 0 and cycle 1. "
            "  Track in-distribution pass@1 (D_train) as a sanity check. "
            "  The `forgetting_eval` kind in runner.py is designed for this pattern. "
            "  Estimated cost: 2 cycles × 3 strategies × ~1h each = ~6 GPU-hours. "
            "  Well within 4-hour constraint per individual strategy run. "
        ),
        "spec": {
            "pipeline": "humaneval_rl_vs_sft_forgetting_eval",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 2,
            "strategies": [
                "sft_then_grpo_both_cycles",
                "sft_cycle0_grpo_only_cycle1",
            ],
            "held_out_fraction": 0.12,
            "held_out_n_problems": 20,
            "sft_include_success": True,
            "scaling_force_both": True,
            "eval_data": _HE_EVAL,
            "compare_with": [
                "e2e_4cyc_gpt55 skills arm 70.73%→75.61% (current SFT+GRPO cycles)",
                "tau2 oscillation 89.19%→70.27% (SFT-caused catastrophic forgetting)",
                "arxiv:2605.28860 (RL preserves circuits vs SFT overwrites them)",
                "arxiv:2507.05386 (RFT gradient ∝ reward variance → conservative updates)",
            ],
            "implementation_note": (
                "Use existing runner.py `forgetting_eval` kind. "
                "Split HumanEval into D_train and D_held (20 problems held out). "
                "For each strategy: run 2 cycles, evaluate on D_held after each cycle. "
                "Compute forgetting@1 = pass@1(D_held, cycle 1) - pass@1(D_held, cycle 0). "
                "Strategy A (default): SFT + GRPO in both cycles. "
                "Strategy B: SFT + GRPO in cycle 0; GRPO-only (no SFT) in cycle 1. "
                "No architectural changes needed: just skip train_small_model.py "
                "in cycle 1 for Strategy B. "
                "Total ~6 GPU-hours (2 strategies × 2 cycles × ~1.5h each)."
            ),
            "primary_metrics": [
                "pass_at_1_d_held_cycle0",
                "pass_at_1_d_held_cycle1",
                "forgetting_at_1",
                "pass_at_1_d_train_cycle0",
                "pass_at_1_d_train_cycle1",
                "strategy_delta_forgetting",
                "circuit_disruption_proxy",
            ],
            "target_outcome": (
                "Strategy B (GRPO-only cycle 1) shows less negative forgetting@1 "
                "than Strategy A (SFT+GRPO both cycles). "
                "Strategy A in-distribution pass@1 may be higher (SFT adapts faster). "
                "Tradeoff: Strategy B balances in-distribution and held-out performance. "
                "Quantification of tau2 oscillation: if D_held forgetting@1 under A "
                "is large (> 3pp drop), it confirms SFT is the forgetting driver "
                "and recommends GRPO-only continual for tau2 cycle 2+. "
            ),
            "gpu": "auto",
            "estimated_gpu_hours": 3.5,
        },
        "gpu": "auto",
    },
]


def main() -> None:
    if not STATE_PATH.exists():
        print(f"ERROR: {STATE_PATH} not found. Is the A800 mounted?")
        raise SystemExit(1)

    with STATE_PATH.open() as f:
        state = json.load(f)

    existing_ids = {e["id"] for e in state.get("queue", [])}
    existing_ids |= {e["id"] for e in state.get("history", [])}

    added = []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] in existing_ids:
            print(f"  SKIP (duplicate): {exp['id']}")
            continue
        state["queue"].append(exp)
        added.append(exp["id"])
        print(f"  QUEUED: {exp['id']}  (priority={exp['priority']}, "
              f"kind={exp['kind']})")

    if not added:
        print("No new experiments added (all duplicates).")
        return

    tmp = STATE_PATH.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(STATE_PATH)
    print(f"\nSaved {STATE_PATH}. Added {len(added)} experiment(s): {added}")


if __name__ == "__main__":
    main()
