#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-18 (EXP-122, EXP-123).

A800 connectivity: offline since 2026-05-14 (day ~65). Apply when restored.

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

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_18.py            # EXP-122, EXP-123

Queue was ~131 pending on 2026-07-17 (+2 added: EXP-120, EXP-121).
Queue cap applied: >20 → max 2 new experiments.

Data motivating today's proposals
----------------------------------
results/e2e_4cyc_gpt55/cycle_3/e2e_ablation_summary.json (unchanged — A800 offline day 65):
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

AAAI 2027 deadline: 2026-08-15 (28 days from today).
A800 offline since 2026-05-14 (day 65).

arxiv:2605.21125 — "Advantage Collapse in Group Relative Policy Optimization:
    Diagnosis and Mitigation" (Xixiang He et al., ICML 2026, May 30, 2026):

    THE ACR METRIC (directly matches our diagnostic):
    Advantage Collapse Rate = fraction of training batches where within-group
    reward variance is zero (i.e., all K rollouts have the same reward).
    Paper shows: early-stage ACR explains 62% of variance in final pass@1
    performance, making it a strong early-training predictor of stagnation.
    Our baseline ACR=52.4% is in the "severe collapse" regime the paper targets.

    AVSPO METHOD (Adaptive Virtual Sample Policy Optimization):
    Rather than filtering collapsed groups (DAPO), ignoring them via zero-gradient
    (Dr.GRPO when all-same), or reweighting (EMA/MDP-GRPO), AVSPO *injects*
    virtual reward samples into collapsed groups to artificially restore diversity:

      For an all-fail group (all K rollouts reward=0):
        Inject m virtual samples with reward=1 (m guided by real-time ACR level).
        The group now has K+m samples with reward ∈ {0,1} → non-zero variance →
        non-zero advantage → gradient flows.
      For an all-pass group (all K rollouts reward=1):
        Inject m virtual samples with reward=0 → group remains informative.
      Injection count m is adaptive: m = ceil(ACR / threshold * m_max), so when
        ACR is high (many collapsed groups), more virtual samples are injected;
        when ACR drops below a target, injection ceases.

    The virtual samples are NOT added to the replay buffer or used in future rollouts;
    they are one-step gradient patches applied only to the collapsed group.

    KEY RESULTS:
      - AVSPO reduces ACR by 58-63% relative to standard GRPO across 0.5B to 14B
        models.
      - Consistent pass@1 gains of 4-6 pp across all model scales on math/code.
      - Outperforms DAPO, Dr.GRPO, and sign-advantage baselines in the paper's ablation.
      - Accepted to ICML 2026 with code released.

    DISTINCTION FROM PRIOR EXPERIMENTS:
      EXP-108 (sign-advantage): for all-fail groups only, adds a fixed negative
          advantage rather than injecting virtual samples. No adaptive injection rate.
      EXP-120 (Dr.GRPO): removes std division → all-same groups get zero gradient.
          This is the opposite philosophy to AVSPO — zero vs. inject.
      EXP-118 (RAPER): replays past high-signal rollouts to diversify the batch.
          Different from intra-group injection — RAPER rebalances across prompts.
      EXP-119 (MDP-GRPO): uses EMA cross-batch std as normalization denominator.
      EXP-122 (this): direct AVSPO implementation — adaptive virtual sample
          injection INTO collapsed groups, ACR-driven injection rate. Orthogonal
          to all prior experiments; targets problem (A) from a complementary angle.

arxiv:2507.21848 — "EDGE-GRPO: Entropy-Driven GRPO with Guided Error Correction
    for Advantage Diversity" (Xingjian Zhang et al., July 2025):

    CORE OBSERVATION:
    Within a GRPO training step, the relationship between a sample's policy entropy
    and its correctness/reward is NOT monotone: some correct responses show low
    entropy (confident-correct — the model already "knows" the answer, low marginal
    learning value), while some incorrect responses show low entropy too (confident-
    wrong — the model is confidently locked into a wrong solution, high risk of
    reinforcing). Symmetrically, some incorrect responses have high entropy (uncertain-
    wrong — the model is genuinely searching, highest learning value). Standard GRPO
    treats all samples in a group uniformly after reward normalization, ignoring these
    qualitative differences.

    EDGE-GRPO METHOD (two components):
    1. Entropy-Driven Advantage (EDA):
       Scale each sample's advantage by its per-token entropy H(π|x):
         A_edge(i) = A_grpo(i) × f(H(π_i))   where f is a monotone function
       For correct samples: low-entropy correct → f(H) → smaller update (we already
         know this; pushing harder risks over-specialization). High-entropy correct →
         larger update (model is uncertain; teacher signal is valuable).
       For incorrect samples: low-entropy wrong → larger penalty scale (model is
         confidently wrong; must counteract). High-entropy wrong → smaller penalty
         (model is exploring; let it search).
       Net effect: advantage diversity WITHIN the non-collapsed group increases;
         updates are concentrated on the most informative samples.
    2. Guided Error Correction (GEC):
       For all-fail groups (every rollout reward=0), the paper uses a "guided" correct
       solution — either from the teacher model or by sampling the policy at T=0 (greedy)
       and using the task verifier — to construct a single reference response with
       reward=1. This response is added to the group as a hard example, breaking the
       all-fail deadlock and providing a concrete direction for the gradient update.
       For our pipeline, the teacher trace from collect_traces.py already provides
       exactly this reference response (the large-model solution stored as the teacher
       rollout). GEC amounts to injecting one teacher trace per collapsed all-fail group.

    KEY RELEVANCE:
      - Targets both problem (A) (GEC for collapsed groups) and problem (B)
        (EDA for non-collapsed groups), in one algorithm.
      - The GEC component is nearly free to implement: we already have teacher traces
        per task from SCALING_FORCE_BOTH=1; injecting the teacher trace as a reward=1
        virtual sample into the all-fail group is ~5 lines in grpo_train_simple.py.
      - The EDA component requires computing per-token entropy over the policy logits,
        which are already materialized in the GRPO forward pass (no extra inference).
      - Distinct from AVSPO (EXP-122): AVSPO uses SYNTHETIC virtual samples with
        assigned rewards; EDGE-GRPO's GEC uses the actual teacher solution verified by
        the task checker. GEC is a "real" sample, not a virtual patch — higher quality
        diversity signal at the cost of requiring teacher traces (which we have).
"""

import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = "data/humaneval_eval.jsonl"

NEW_EXPERIMENTS = [
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-122: AVSPO — Adaptive Virtual Sample injection for collapsed GRPO groups
    #          (ICML 2026, arXiv:2605.21125)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_18_001_avspo_virtual_sample_injection_humaneval",
        "priority": 9,
        "gpu": "auto",
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2605.21125 — 'Advantage Collapse in Group Relative Policy "
            "Optimization: Diagnosis and Mitigation' (Xixiang He et al., ICML 2026, "
            "May 30, 2026). Proposes AVSPO: for each GRPO training batch, identify "
            "collapsed groups (within-group reward std = 0) and inject m virtual samples "
            "with the opposite reward to restore intra-group diversity. The injection "
            "count m is adaptive: driven by real-time ACR monitoring, m scales up when "
            "ACR is high and turns off when ACR falls below a target rate. Paper reports "
            "58-63% ACR reduction and +4-6pp pass@1 gains over standard GRPO across 0.5B "
            "to 14B models on math/code tasks; accepted ICML 2026 with code released. "
            "This directly attacks our Root Problem (A): ACR=52.4% in the 4-cycle baseline "
            "(43/82 groups collapsed, Full arm = Router arm). "
            "AVSPO is philosophically opposite to Dr.GRPO (EXP-120) — instead of setting "
            "gradient to zero for collapsed groups (Dr.GRPO) or filtering them (DAPO), "
            "AVSPO creates gradient signal via synthetic diversity injection. It is also "
            "distinct from RAPER (EXP-118, cross-batch replay) and MDP-GRPO (EXP-119, "
            "EMA normalization anchor) — those target the normalization denominator; "
            "AVSPO targets the reward distribution within each group. "
            "Target: ACR < 22% (58% reduction from 52.4%), full arm pass@1 > 93%."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 4,
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "avspo_enabled": True,
            "avspo_acr_target": 0.20,
            "avspo_m_max": 2,
            "avspo_injection_mode": "adaptive",
            "grpo_temperature": 1.0,
            "n_generations": 8,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "analysis": [
                "acr_per_step_with_without_avspo",
                "acr_injection_rate_over_training",
                "full_arm_pass_at_1_vs_baseline",
                "virtual_sample_fraction_per_batch",
                "acr_early_stage_vs_final_performance_correlation",
            ],
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (baseline): full 92.68%, ACR=52.4%",
                "EXP-108 (sign-advantage): fixed negative adv for all-fail, no injection rate",
                "EXP-120 (Dr.GRPO): zero gradient for collapsed, opposite philosophy",
                "EXP-118 (RAPER): cross-batch replay, different collapse strategy",
                "arxiv:2605.21125 (AVSPO): 58-63% ACR reduction, +4-6pp pass@1",
            ],
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
            ],
            "implementation_note": (
                "In grpo_train_simple.py advantage computation block: "
                "1. Compute group_std for each prompt group as before. "
                "2. Detect collapsed groups: collapsed = (group_std < eps). "
                "3. For collapsed groups, compute injection count m: "
                "   acr_current = collapsed.float().mean(); "
                "   m = ceil(acr_current / avspo_acr_target * avspo_m_max) if acr_current > avspo_acr_target else 0. "
                "4. For each collapsed group: if all rewards == 0 (all-fail), inject m "
                "   virtual samples with reward=1 and loss_mask=0 (virtual, not LM-trained on). "
                "   If all rewards == 1 (all-pass), inject m virtual samples with reward=0. "
                "5. Recompute group mean/std including virtual samples; compute advantages "
                "   using the augmented statistics (standard GRPO normalization on the "
                "   augmented group). Drop virtual samples after advantage computation — "
                "   they only affect the advantage denominator, not the token-level update. "
                "6. Add env vars AVSPO_ENABLED (default 0), AVSPO_ACR_TARGET (default 0.20), "
                "   AVSPO_M_MAX (default 2). Total change: ~20 lines. "
                "Note: virtual samples should NOT be added to the model's forward pass "
                "for token-level loss — they are pure group-statistics patches. "
                "Implementation reference: https://github.com/QingyongHu/AVSPO (ICML 2026)."
            ),
            "arxiv_ref": "2605.21125",
            "estimated_gpu_hours": 2.5,
        },
    },
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-123: EDGE-GRPO — Entropy-Driven Advantage + Guided Error Correction
    #          (arXiv:2507.21848, July 2025)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_18_002_edge_grpo_entropy_driven_adv_gec_humaneval",
        "priority": 8,
        "gpu": "auto",
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2507.21848 — 'EDGE-GRPO: Entropy-Driven GRPO with Guided Error "
            "Correction for Advantage Diversity' (Xingjian Zhang et al., July 2025). "
            "Addresses two distinct failure modes of standard GRPO: "
            "(1) Advantage collapse in all-same-reward groups (our ACR=52.4% problem): "
            "Guided Error Correction (GEC) injects the teacher's verified solution as a "
            "real reward=1 sample into each all-fail group. For our pipeline, teacher "
            "traces from collect_traces.py (SCALING_FORCE_BOTH=1) already provide one "
            "large-model solution per task, verified by the HumanEval test suite. GEC "
            "simply adds this trace to the all-fail group — no synthetic samples. "
            "(2) Advantage diversity within non-collapsed groups (our Root Problem B): "
            "Entropy-Driven Advantage (EDA) scales each sample's advantage by its "
            "per-token policy entropy H(π|x,y): confident-correct (low H, already known) "
            "gets a smaller update; confidently-wrong (low H, the model is locked in a "
            "wrong belief) gets a larger penalty; uncertain-wrong (high H, exploring) "
            "gets a smaller penalty to encourage continued exploration. Advantage "
            "computation becomes: A_edge(i) = A_grpo(i) * tanh(alpha * H(π_i)). "
            "This targets Root Problem B: the 47.6% of non-collapsed groups should "
            "produce stronger gradients with EDA than with uniform advantage. "
            "Distinct from AVSPO (EXP-122): AVSPO uses synthetic virtual samples with "
            "assigned rewards; EDGE-GRPO's GEC uses actual teacher solutions verified "
            "by the task checker — real samples, not patches. The combination of GEC "
            "(problem A) + EDA (problem B) makes EDGE-GRPO the most architecturally "
            "ambitious collapse fix in the queue. Target: ACR < 35% (via GEC), "
            "full arm pass@1 > 93.5% (combined A+B improvement)."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 4,
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "edge_grpo_enabled": True,
            "edge_gec_enabled": True,
            "edge_eda_enabled": True,
            "edge_eda_alpha": 1.0,
            "edge_eda_temperature_scale": "tanh",
            "grpo_temperature": 1.0,
            "n_generations": 8,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "analysis": [
                "acr_per_cycle_with_gec_vs_baseline",
                "entropy_distribution_correct_vs_incorrect_per_step",
                "eda_advantage_distribution_vs_standard_grpo",
                "gec_teacher_injection_rate_per_cycle",
                "full_arm_pass_at_1_per_cycle",
                "problem_a_vs_problem_b_contribution_ablation",
            ],
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (baseline): full 92.68%, ACR=52.4%",
                "EXP-108 (sign-advantage): all-fail negative adv (no teacher trace)",
                "EXP-120 (Dr.GRPO): zero gradient for collapsed (no injection)",
                "EXP-122 (AVSPO): synthetic virtual samples for collapsed",
                "arxiv:2507.21848 (EDGE-GRPO): entropy + teacher-guided correction",
            ],
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
                "src/pipeline/collect_traces.py",
            ],
            "implementation_note": (
                "GEC component (grpo_train_simple.py): "
                "At training time, load teacher traces from traces.jsonl for the current "
                "cycle (already generated by collect_traces with SCALING_FORCE_BOTH=1). "
                "For each prompt in the GRPO batch, if all K rollout rewards == 0 "
                "(all-fail collapsed), look up the teacher trace for that task_id; "
                "if found and test-suite verified (reward=1), add it to the group as an "
                "additional sample. Recompute group mean/std including this sample. "
                "The teacher trace IS included in the token-level loss (unlike AVSPO "
                "virtual samples) since it is a real, high-quality solution. "
                "EDA component (grpo_train_simple.py): "
                "After computing standard GRPO advantages, compute per-token entropy "
                "H_i = -sum_v [p_v * log(p_v)] from the already-materialized logits "
                "(mean over tokens for sequence-level H_i). Scale: "
                "A_edge(i) = A_grpo(i) * tanh(edge_eda_alpha * H_i). "
                "For correct samples (reward=1): high H → larger update (uncertain correct "
                "is more valuable). For incorrect (reward=0): high H → SMALLER penalty "
                "(exploring), low H → LARGER penalty (confidently wrong). "
                "Add env vars EDGE_GRPO_ENABLED, EDGE_GEC_ENABLED, EDGE_EDA_ENABLED, "
                "EDGE_EDA_ALPHA (default 1.0). Ablate GEC-only, EDA-only, combined. "
                "~30 lines total; logits are already in memory during GRPO forward pass. "
                "Teacher trace lookup requires a task_id→trace index built once at "
                "the start of GRPO training (dict keyed on task_id from traces.jsonl)."
            ),
            "arxiv_ref": "2507.21848",
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
