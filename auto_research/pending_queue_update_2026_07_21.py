#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-21 (EXP-126, EXP-127).

A800 connectivity: offline since 2026-05-14 (day ~68). Apply when restored.

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

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_21.py            # EXP-126, EXP-127

Queue was ~135 pending on 2026-07-19 (+2 added: EXP-124, EXP-125).
Queue cap applied: >20 → max 2 new experiments.

Data motivating today's proposals
----------------------------------
results/e2e_4cyc_gpt55/cycle_3/e2e_ablation_summary.json (unchanged — A800 offline day 68):
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

AAAI 2027 deadline: 2026-08-15 (25 days from today).
A800 offline since 2026-05-14 (day ~68).

arxiv:2607.16850 — "Group Entropy-Controlled Policy Optimization" (GEPO)
    Guangran Cheng et al., submitted July 18, 2026:

    PROBLEM ADDRESSED:
    RL post-training on heterogeneous task mixtures (e.g. a mix of easy and hard
    coding problems) induces distinct entropy regimes across prompt groups under the
    same policy. GRPO-style normalized advantages compute within-group z-scores, but
    these z-scores are NOT statistically comparable across groups that differ in entropy:
    a group where the model has near-deterministic outputs (low entropy, easy tasks)
    will have advantage magnitudes inflated relative to a high-entropy group (hard tasks
    where outputs vary wildly). Uniform advantage normalization therefore creates an
    entropy-dependent bias that allows easy-task gradients to dominate the update.

    THE GEPO METHOD:
    GEPO adds a group-level entropy estimate H_g (computed from the K rollouts of group g
    using the group's token distribution) and applies entropy-conditioned asymmetric
    advantage attenuation:

      For low-entropy groups (easy tasks, H_g < H_low_threshold):
        Attenuate POSITIVE advantages: A_i *= (H_g / H_low_threshold)
        (Reduces the over-exploitation of already-mastered tasks; the model is
         already confident here — reinforcing further adds diminishing value and
         crowds out gradient budget from harder groups.)

      For high-entropy groups (hard tasks, H_g > H_high_threshold):
        Attenuate NEGATIVE advantages: |A_i| *= (H_high_threshold / H_g)
        (Preserves exploration stability when all rollouts fail; heavy penalization
         of high-entropy failure groups destabilizes training and collapses the
         entropy further, worsening group-level ACR in future steps.)

      For mid-entropy groups (H_low <= H_g <= H_high): no attenuation.

    The thresholds H_low and H_high are estimated as the 25th and 75th percentile of
    group entropies across the current mini-batch, updated each step (no hyperparameter
    tuning needed — percentile-based thresholds are self-calibrating).

    KEY RESULTS (from the paper's abstract and search snippet):
    - Outperforms GRPO on heterogeneous task alignment benchmarks.
    - Attenuates positive advantages in low-entropy groups → reduces over-exploitation.
    - Attenuates negative advantages in high-entropy groups → preserves exploration.
    - Lightweight: adds only the group entropy estimate (computable from existing rollout
      logits at negligible cost) and two percentile thresholds.

    RELATION TO OUR PIPELINE:
    Our HumanEval task mix is highly heterogeneous:
    - Easy tasks (solved by small model in cycle 0): 62/82 labels are 0 (small sufficient)
      → within these groups, all K rollouts succeed → uniform R=1 → zero advantage (ACR)
      → no gradient.
    - Hard tasks (labels 1, routed to large): 20/82 → small model fails these → diverse
      rewards across rollouts → GRPO gets gradient only here.

    The ACR=52.4% (43 collapsed groups) could be reduced by GEPO's entropy-attenuation:
    - Easy-task groups have low entropy AND zero advantage — GEPO would reduce their
      positive advantage weight (already zero, so no harm), but more importantly, the
      group-entropy monitoring provides visibility into which groups are contributing
      gradient vs. which are inert.
    - Hard-task groups with partial success have high entropy — GEPO prevents over-penalizing
      the failed rollouts, preserving the gradient signal from the non-collapsed groups.

    RELATION TO QUEUE:
    Distinct from all prior experiments:
    - EXP-120 (Dr.GRPO): zeros gradient for collapsed groups (threshold on std), no entropy signal.
    - EXP-122 (AVSPO): injects virtual samples to artificially inflate group diversity.
    - EXP-123 (EDGE-GRPO): per-sample entropy scaling of individual advantages (not group-level).
    - EXP-124 (FADE): batch-level M+/M- sign rebalancing (not group-entropy-conditioned).
    - GEPO: group-entropy-conditioned asymmetric attenuation — the only approach that
      explicitly differentiates easy vs. hard groups via their entropy signature and applies
      asymmetric treatment (positive vs. negative advantage scaling depending on group entropy).

arxiv:2606.21090 — "Self-Improvement Can Self-Regress: The Rise-and-Collapse Failure
    Mode of LLM Self-Training" (June 2026):

    MOTIVATION:
    Our 4-cycle run shows full arm = router arm (92.68%), implying GRPO adds zero net value.
    The common explanation is Root Problem A (ACR = 52.4% collapsed groups). But another
    explanation — not yet measured in our pipeline — is that GRPO training DOES improve pass@1
    within each cycle, but the improvement collapses before or during checkpoint saving.

    The "Self-Improvement Can Self-Regress" paper documents this rise-and-collapse pattern in
    code LLMs: in REINFORCE post-training for code generation, pass@1 peaks within tens of
    gradient steps and then falls back, sometimes to near zero. Crucially:
    - KL constraints (standard reference model regularization) do NOT prevent it.
    - EWC (elastic weight consolidation) does NOT prevent it.
    - The collapse is not cross-task catastrophic forgetting but WITHIN-TASK policy
      over-optimization on a fixed distribution — the model "overfits" to the GRPO reward.
    - The mechanism: the model concentrates probability mass on its top-1 solution,
      reducing the diversity of rollouts, which reduces the group-relative advantage
      variance, which reduces the GRPO gradient, which triggers the classic GRPO collapse
      (ACR) — a self-reinforcing feedback loop.

    RELATION TO OUR PIPELINE:
    In our grpo_train_simple.py, we currently save only the FINAL checkpoint after
    GRPO_EPOCHS epochs and select it as the next cycle's starting point. If the true
    peak is at epoch 1 or 2 and the model then collapses by epoch 5 (the default), we
    are ALWAYS loading a collapsed model — which explains the full arm stagnation at
    92.68% = router arm.

    Measuring this would require: evaluate pass@1 on the HumanEval eval set AFTER EACH
    GRPO EPOCH and log the per-epoch curve. This is a forgetting_eval kind experiment
    since it monitors within-cycle degradation rather than cross-cycle forgetting.

    Fix implication: if rise-and-collapse is confirmed, the remedy is early stopping
    on GRPO: save the best-epoch checkpoint (by eval pass@1) rather than the final one.
    This is a 5-line change to grpo_train_simple.py and would directly fix the full-arm
    stagnation WITHOUT requiring any advantage modification (orthogonal to AVSPO/FADE/GEPO).

    RELATION TO QUEUE:
    EXP-111 (forgetting_eval): measures CROSS-CYCLE forgetting (cycle N erases cycle N-1).
    EXP-127 (this): measures WITHIN-CYCLE collapse (pass@1 peaks then degrades within a
      single GRPO training run). These are orthogonal measurements; EXP-127 may diagnose
      the LARGEST single contributor to full-arm stagnation.
"""

import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = "data/humaneval_eval.jsonl"

NEW_EXPERIMENTS = [
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-126: GEPO — Group Entropy-Controlled Policy Optimization
    #          (arXiv:2607.16850, Guangran Cheng et al., July 18 2026)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_21_001_gepo_group_entropy_controlled_humaneval",
        "priority": 8,
        "gpu": "auto",
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2607.16850 — 'Group Entropy-Controlled Policy Optimization' (GEPO) "
            "(Guangran Cheng et al., July 18, 2026). Identifies that GRPO-style normalized "
            "advantages create an entropy-dependent bias on heterogeneous task mixtures: "
            "easy-task groups (low entropy, all rollouts succeed → uniform R=1 → zero "
            "advantage) and hard-task groups (high entropy, partial success) produce "
            "z-score magnitudes that are not statistically comparable, allowing easy-task "
            "gradients to crowd out hard-task learning. GEPO applies entropy-conditioned "
            "asymmetric attenuation: attenuate positive advantages in low-entropy groups "
            "(reduce over-exploitation of mastered tasks) and attenuate negative advantages "
            "in high-entropy groups (preserve exploration stability when all rollouts fail). "
            "Thresholds H_low / H_high are set as the 25th/75th percentile of group entropies "
            "in the current mini-batch — self-calibrating, no hyperparameter tuning. "
            "Direct relevance to our pipeline: our HumanEval task mix is maximally "
            "heterogeneous (62/82 groups routed small=easy; 20/82 hard), ACR=52.4% means "
            "43 groups have zero advantage. GEPO is orthogonal to all prior ACR fixes: "
            "AVSPO (EXP-122) injects virtual samples; EDGE-GRPO (EXP-123) scales per-sample; "
            "FADE (EXP-124) rebalances batch-level M+/M-; GEPO is the only group-entropy-"
            "conditioned asymmetric method addressing easy-vs-hard group gradient imbalance. "
            "Implementation: ~20 lines in grpo_train_simple.py."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 4,
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "gepo_enabled": True,
            "gepo_H_low_percentile": 25,
            "gepo_H_high_percentile": 75,
            "grpo_temperature": 1.0,
            "n_generations": 8,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "analysis": [
                "group_entropy_distribution_per_cycle",
                "low_entropy_group_count_vs_baseline",
                "high_entropy_group_count_vs_baseline",
                "per_group_positive_advantage_attenuation_factor",
                "per_group_negative_advantage_attenuation_factor",
                "full_arm_pass_at_1_per_cycle_gepo_vs_standard",
                "acr_rate_per_cycle_gepo_vs_standard",
                "gradient_norm_per_step_gepo_vs_standard",
            ],
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (baseline): full 92.68%, ACR=52.4%",
                "EXP-122 (AVSPO): group-level virtual injection",
                "EXP-123 (EDGE-GRPO): per-sample entropy scaling",
                "EXP-124 (FADE): batch-level M+/M- sign rebalancing",
                "arxiv:2607.16850 (GEPO): group-entropy asymmetric attenuation",
            ],
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
            ],
            "implementation_note": (
                "In grpo_train_simple.py, after GRPO advantages are computed for the "
                "current mini-batch (grouped by prompt, K rollouts each): "
                "1. For each group g, compute group entropy H_g from the K rollout "
                "   token distributions: H_g = -mean_over_rollouts(sum_t p_t * log p_t). "
                "   Use the already-computed per-token logits from the rollout forward pass; "
                "   no extra forward pass needed. "
                "2. Compute H_low = percentile(all_H_g, GEPO_H_LOW_PERCENTILE=25), "
                "           H_high = percentile(all_H_g, GEPO_H_HIGH_PERCENTILE=75). "
                "3. For each group g: "
                "   If H_g < H_low:  A_i *= (H_g / H_low)  for all A_i > 0 in group g. "
                "   If H_g > H_high: |A_i| *= (H_high / H_g) for all A_i < 0 in group g "
                "                    (i.e., A_i *= (H_high / H_g) for A_i < 0). "
                "   Otherwise: no change. "
                "4. Proceed with the standard GRPO loss using the modified advantages. "
                "Add env vars: GEPO_ENABLED (default 0), "
                "GEPO_H_LOW_PERCENTILE (default 25), GEPO_H_HIGH_PERCENTILE (default 75). "
                "Total: ~20 lines. H_g computation reuses existing rollout logit tensors — "
                "zero overhead for the forward pass. "
                "Implementation reference: https://arxiv.org/abs/2607.16850."
            ),
            "arxiv_ref": "2607.16850",
            "estimated_gpu_hours": 3.0,
        },
    },
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-127: Rise-and-Collapse within-cycle GRPO diagnostic
    #          (arXiv:2606.21090, June 2026)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_21_002_rise_collapse_within_cycle_grpo_eval",
        "priority": 9,
        "gpu": "auto",
        "kind": "forgetting_eval",
        "rationale": (
            "arxiv:2606.21090 — 'Self-Improvement Can Self-Regress: The Rise-and-Collapse "
            "Failure Mode of LLM Self-Training' (June 2026). Documents that in REINFORCE/GRPO "
            "post-training for code generation, pass@1 peaks within tens of gradient steps "
            "and then collapses back to near-baseline within the SAME training run — not "
            "cross-task catastrophic forgetting, but within-task policy over-optimization on "
            "a fixed distribution. KL constraints and EWC do NOT prevent it. The mechanism: "
            "the model concentrates probability mass on its top-1 rollout → reduces diversity "
            "→ reduces GRPO group-relative advantage variance → reduces gradient → triggers "
            "ACR collapse — a self-reinforcing feedback loop within a single cycle. "
            "Critical relevance: our full arm stagnates at 92.68% = router arm, implying "
            "GRPO adds zero net value across 4 cycles. The standard explanation is ACR=52.4% "
            "(Root Problem A), but this paper raises an alternative: GRPO DOES improve early "
            "in each cycle's training run, but the checkpoint we save (end of GRPO_EPOCHS) "
            "is already collapsed. If peak pass@1 occurs at epoch 1–2 but we save epoch 5, "
            "we always load a collapsed model as the next cycle's start. This single change "
            "(save best-epoch checkpoint instead of final) would fix the stagnation without "
            "any advantage modification. Proposed diagnostic: add per-epoch eval of pass@1 "
            "on the HumanEval eval set DURING each cycle's GRPO training; log the curve; "
            "if rise-then-collapse is confirmed, switch grpo_train_simple.py to save the "
            "best-epoch checkpoint by eval pass@1. Distinct from EXP-111 (cross-cycle "
            "forgetting); orthogonal to AVSPO/FADE/GEPO (advantage modifications). "
            "Priority 9: this is the highest-leverage diagnostic remaining — if confirmed, "
            "the remedy is trivial (5-line checkpoint selection change) and directly closes "
            "the full-arm stagnation gap that has been the paper's main open problem."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 4,
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "grpo_per_epoch_eval": True,
            "grpo_eval_every_n_steps": 10,
            "grpo_save_best_epoch_checkpoint": False,
            "grpo_temperature": 1.0,
            "n_generations": 8,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "diagnostic_outputs": [
                "per_epoch_pass_at_1_curve_per_cycle",
                "grpo_step_at_peak_pass_at_1_per_cycle",
                "steps_from_peak_to_final_checkpoint",
                "pass_at_1_degradation_peak_to_final_per_cycle",
                "group_reward_variance_per_step",
                "acr_rate_per_step",
                "policy_entropy_per_step",
            ],
            "hypothesis": (
                "If pass@1 peaks at step K < final step for each cycle, "
                "confirming the rise-and-collapse pattern, recommend: "
                "(a) Enable grpo_save_best_epoch_checkpoint=True in all subsequent runs. "
                "(b) Reduce GRPO_EPOCHS from 5 to 2 as a simpler fixed early-stop. "
                "Both changes are compatible with all advantage modifications (AVSPO, FADE, GEPO)."
            ),
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (baseline): full 92.68% = router arm, GRPO no net gain",
                "EXP-111 (forgetting_eval): cross-cycle forgetting, orthogonal measurement",
                "arxiv:2606.21090: rise-and-collapse in REINFORCE code LLMs, KL does not fix it",
            ],
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
            ],
            "implementation_note": (
                "In grpo_train_simple.py, add an eval callback inside the GRPO training loop: "
                "1. After every GRPO_EVAL_EVERY_N_STEPS (default 10) gradient steps, run "
                "   pass@1 evaluation on eval_data (HumanEval eval set, n=82 problems, "
                "   temperature=0 greedy decode) and log: step, pass@1, group_reward_var, "
                "   acr_rate (fraction of groups with reward_std < 1e-6). "
                "2. Also log policy entropy over the eval set at each checkpoint. "
                "3. After training, plot pass@1 curve vs. step for each cycle and save to "
                "   results/<run_id>/cycle_N/grpo_within_cycle_pass_curve.json. "
                "4. If GRPO_SAVE_BEST_EPOCH_CHECKPOINT=1: save intermediate checkpoint at "
                "   the step with highest eval pass@1 (in addition to the final checkpoint). "
                "Add env vars: GRPO_PER_EPOCH_EVAL (default 0), "
                "GRPO_EVAL_EVERY_N_STEPS (default 10), "
                "GRPO_SAVE_BEST_EPOCH_CHECKPOINT (default 0). "
                "Eval cost: 82 HumanEval problems × greedy decode × ~5 tokens/step × "
                "ceil(GRPO_STEPS / 10) calls. At ~3s per problem, 82 evals × 10 checkpoints "
                "= 820 calls = ~41 min overhead per cycle — acceptable for a diagnostic run. "
                "For production runs, set GRPO_EVAL_EVERY_N_STEPS=50 to reduce to ~10 min. "
                "Implementation reference: https://arxiv.org/abs/2606.21090."
            ),
            "arxiv_ref": "2606.21090",
            "estimated_gpu_hours": 4.0,
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
