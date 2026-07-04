#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-04 (EXP-106, EXP-107).

A800 connectivity: offline since 2026-05-14 (day 51). Apply when restored.

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

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_04.py            # EXP-106, EXP-107

Queue was ~109 pending on 2026-07-03 (+2 added: EXP-104, EXP-105).
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
    Root problem: ACR=52.4% — 43/82 GRPO groups have zero within-group reward
    variance → zero gradient → Full = Router.
    Skills gap: 75.61% vs large 96.34% — 20.7pp gap. Skills arm biggest
    headroom, still only receiving a global single-cluster procedure.

arxiv:2604.02288 — Unifying Group-Relative and Self-Distillation Policy
    Optimization via Sample Routing (SRPO, April 2026):
    Proposes routing GRPO samples by outcome: correct-outcome rollouts go to
    GRPO's advantage-weighted reinforcement path; failed-outcome rollouts go to
    SDPO (Self-Distillation Policy Optimization), which applies targeted
    logit-level correction using the reference policy's distribution as the
    "teacher" signal for failed tokens.
    Result: ACR-equivalent groups (all-fail) no longer contribute zero gradient
    — they receive a SDPO correction signal that nudges the policy toward the
    reference distribution's high-probability tokens at positions where the
    policy diverged. Empirically: +3.2% pass@1 on HumanEval-Instruct (CodeQwen
    1.5B baseline), +4.7% on MBPP-plus, compared to standard DAPO.
    Critical relevance: our 43/82 all-fail groups (52.4% ACR) currently waste
    all their backward-pass compute generating zero gradient. SRPO gives each
    failed rollout a logit-correction signal from π_ref — the very teacher model
    we already have available (Qwen2.5-Coder-1.5B base weights, stored as
    π_ref). No extra model, no extra rollouts, no reward engineering — just
    route the failed rollouts to a different loss head.
    Orthogonal to all other queued ACR approaches: EXP-090/094/096/098/101/102/104
    all modify the reward function, group sampling, or loss weighting within the
    GRPO path. SRPO adds a complementary SDPO path for the zero-reward samples,
    leaving the GRPO path unchanged.

arxiv:2607.02502 — DemoPSD: Disagreement-Modulated Policy Self-Distillation
    (July 2, 2026 — 2 days ago):
    Identifies that standard on-policy self-distillation (OPSD, which our SFT
    pipeline approximates) distills from the large model uniformly, including
    on easy tasks where the small model already succeeds — causing overfitting
    to in-domain patterns and suppressing exploration on hard tasks.
    DemoPSD's core idea: selectively distill only on tasks where the student
    (small model) and teacher (large model) DISAGREE — i.e., tasks where
    π_small generates wrong answers but π_large generates correct ones.
    This disagreement region is EXACTLY the capability gap our pipeline cares
    about. On tasks where both models succeed, SFT is redundant (small model
    already knows) and may crowd out hard-task gradient. On tasks where both
    fail, SFT on large-model output is also meaningless (large model output is
    wrong).
    Implementation in our pipeline: after SCALING_FORCE_BOTH=1 trace collection,
    filter SFT training data to keep only (query, large_trace) pairs where
    small_rollout_results had 0/8 passing rollouts (small model consistently
    failed) AND large_model_result was correct. This is a pure data-filtering
    step — no architecture change, no new model.
    Note: preserve SFT_INCLUDE_SUCCESS=1 behavior for numeric stability (EWC
    effect from success traces), but apply a loss weight of λ_agree=0.1 for
    agreed-success pairs vs λ_disagree=1.0 for disagreement pairs, so the
    disagreement gradient dominates.
    Direct benefit: the skills arm's pass@1 (currently 75.61%) is bottlenecked
    by the SFT model learning the easy tasks deeply while being data-starved on
    hard tasks. DemoPSD reweights toward hard tasks, which is where the 20.7pp
    skills-vs-large gap lives.

AAAI 2027 context (deadline 2026-08-15, 42 days):
    EXP-106 (SRPO, today): highest-priority new ACR fix — directly converts
    wasted all-fail compute to a training signal. If it pushes Full>Router even
    by 1pp, that's the AAAI System Improvement section's headline result.
    EXP-107 (DemoPSD, today): skills arm improvement. The 20.7pp skills-vs-large
    gap (75.61% vs 96.34%) is currently reported honestly but unexplained.
    DemoPSD provides a principled fix with a fresh arxiv reference (2607.02502,
    Jul 2 2026), supporting the "future work" / "ablation" narrative in §5.

Additional diagnostic note (arxiv:2606.20881):
    "When Do Intrinsic Rewards Work for Code Reasoning? A Comprehensive Study"
    (June 2026) finds that certainty/entropy-based intrinsic rewards (our queued
    STARE/EXP-090, SA-AH-GRPO/EXP-098, EP-GRPO/EXP-102) yield early gains but
    COLLAPSE: models progressively shorten outputs and lose reasoning capability.
    Collapse speed is sensitive to sample size and temperature.
    ACTION: when these queued experiments run, add output-length tracking and
    per-epoch reasoning capability retention metrics to detect early collapse.
    This is a metric annotation for existing queued experiments, not a new exp.
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
        "id": "exp_2026_07_04_001_srpo_sample_routing_allfail_grpo_humaneval",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2604.02288 — Unifying Group-Relative and Self-Distillation "
            "Policy Optimization via Sample Routing (SRPO, April 2026). "
            ""
            "CORE IDEA: SRPO adds a second loss head for failed rollouts. "
            "  Correct-outcome rollouts → GRPO path (advantage-weighted reinforcement). "
            "  Failed-outcome rollouts → SDPO path (logit-level correction toward π_ref). "
            "The SDPO loss for a failed rollout r is: "
            "  L_SDPO(r) = KL(π_ref(·|ctx_t) ‖ π_θ(·|ctx_t)) for each token t "
            "              where π_θ(r_t|ctx) < π_ref(r_t|ctx) * margin "
            "i.e., push the policy back toward π_ref specifically at tokens where "
            "the policy has moved away from the reference distribution on failed samples. "
            ""
            "WHY THIS DIRECTLY FIXES ACR=52.4%: "
            "Our 43/82 all-fail groups each produce G=8 failed rollouts. "
            "Under standard DAPO: these rollouts compute reward_var=0 → gradient=0 "
            "→ the backward pass runs but contributes nothing. Pure compute waste. "
            "Under SRPO: these same 43×8=344 failed rollouts each produce a SDPO "
            "correction gradient that nudges π_θ toward π_ref's high-probability "
            "tokens at the divergence points. No new rollouts needed. No new model. "
            "The reference weights (Qwen2.5-Coder-1.5B base) are already loaded "
            "in memory during GRPO training (for KL penalty computation). "
            ""
            "IMPLEMENTATION (grpo_train_simple.py): "
            "After computing per-rollout rewards, split: "
            "  passed = rollouts where reward > 0 "
            "  failed = rollouts where reward == 0 "
            "Compute standard DAPO loss on `passed` rollouts (unchanged). "
            "Compute SDPO loss on `failed` rollouts: "
            "  for each failed rollout, at each token position t: "
            "    ref_logp = log π_ref(r_t | ctx_t)      # already computed for KL "
            "    pol_logp = log π_θ(r_t | ctx_t)        # from forward pass "
            "    sdpo_token_loss = max(0, ref_logp - pol_logp - margin) "
            "    # only correct when policy has drifted below reference "
            "  L_SDPO(rollout) = mean_t(sdpo_token_loss) "
            "  L_total = L_DAPO + λ_sdpo * L_SDPO(mean over failed rollouts) "
            "Suggested λ_sdpo=0.5, margin=log(1.5) (from SRPO paper defaults). "
            "Implementation: ~20 lines of additional code in the GRPO training loop. "
            "All required tensors (ref_logp, pol_logp) are already computed for "
            "the existing KL penalty term in DAPO — no extra forward passes. "
            ""
            "VARIANTS (single run, reported separately): "
            "  (A) SRPO full: DAPO (passed) + SDPO (failed), λ_sdpo=0.5 "
            "  (B) SRPO λ sweep: λ_sdpo ∈ {0.1, 0.5, 1.0} "
            "  (C) DAPO baseline (standard, reproduce cycle_3 ACR=52.4%) "
            ""
            "RELATIONSHIP TO OTHER QUEUED ACR EXPERIMENTS: "
            "All queued ACR approaches (EXP-090/094/096/098/101/102/104) modify "
            "the reward function, group sampling, or loss weighting WITHIN the "
            "GRPO path — they try to reduce how many groups collapse to ACR. "
            "SRPO leaves the GRPO path untouched and ADDS a SDPO path for collapsed "
            "groups. The two approaches are architecturally orthogonal and combinable: "
            "e.g., STARE (EXP-090) + SRPO would reduce ACR AND give correction signal "
            "to residual collapsed groups. "
            ""
            "ARXIV PAPER RESULTS: "
            "+3.2% pass@1 on HumanEval-Instruct (CodeQwen 1.5B baseline), "
            "+4.7% on MBPP-plus vs standard DAPO. "
            "If we achieve even +1pp on HumanEval (93.68% vs 92.68%), "
            "that is the AAAI headline: Full > Router for the first time. "
            ""
            "NOTE ON COLLAPSE RISK (arxiv:2606.20881): "
            "SRPO is NOT an entropy-based intrinsic reward method, so the "
            "certainty-based collapse finding (output shortening, reasoning degradation) "
            "does NOT apply. SDPO is a reference-anchored KL correction — "
            "it pulls the policy back toward π_ref, which naturally prevents "
            "entropy collapse (π_ref is the exploration reference). "
        ),
        "spec": {
            "pipeline": "humaneval_srpo_sample_routing",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 1,
            "grpo_algo": "srpo",
            "srpo_lambda_sdpo": 0.5,
            "srpo_sdpo_margin": 0.405,
            "srpo_lambda_sweep": [0.1, 0.5, 1.0],
            "variants": [
                "srpo_lambda_0.5",
                "srpo_lambda_0.1",
                "srpo_lambda_1.0",
                "dapo_baseline",
            ],
            "n_generations": 8,
            "grpo_temperature": 1.0,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (DAPO binary reference: 92.68% HE, ACR=52.4%)",
                "arxiv:2604.02288 (SRPO +3.2pp HumanEval-Instruct, +4.7pp MBPP-plus)",
            ],
            "implementation_note": (
                "SDPO loss uses already-computed ref_logp tensors from DAPO's "
                "KL penalty — no extra forward passes needed. "
                "Split rollouts by reward > 0 (GRPO path) vs == 0 (SDPO path). "
                "Combined loss: L_total = L_DAPO + lambda_sdpo * L_SDPO. "
                "~20 lines additional code in grpo_train_simple.py."
            ),
            "primary_metrics": [
                "humaneval_pass_at_1_post_1cycle",
                "grpo_acr_mean",
                "grpo_sdpo_active_fraction",
                "grpo_sdpo_loss_mean",
                "grpo_dapo_loss_mean",
                "grpo_total_loss_mean",
                "training_gradient_norm",
                "variant_delta_pass_at_1",
                "output_length_mean",
                "output_length_std",
            ],
            "target_outcome": (
                "Variant (A) passes DAPO baseline (pass@1 > 92.68%). "
                "SDPO active fraction confirms all-fail groups receive non-zero gradient. "
                "ACR-in-loss = 0% by construction (SRPO routes all groups). "
                "If pass@1 >= 93.68% (+1pp), Full > Router achieved for AAAI paper. "
                "Output length stable (SRPO does not cause entropy collapse via ref anchor). "
            ),
            "gpu": "auto",
            "estimated_gpu_hours": 1.75,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_07_04_002_demopsd_disagreement_distill_skills_humaneval",
        "priority": 6,
        "kind": "joint_cycle_multiseed",
        "rationale": (
            "arxiv:2607.02502 — DemoPSD: Disagreement-Modulated Policy "
            "Self-Distillation (Yunhe Li, Hao Shi et al., July 2, 2026 — "
            "submitted 2 days ago). "
            ""
            "PROBLEM WITH CURRENT SFT (skills arm bottleneck): "
            "Our SFT training (Phase 3a) uses ALL traces where large model succeeded, "
            "regardless of whether the small model could already solve the problem. "
            "With SCALING_FORCE_BOTH=1, we collect: "
            "  Category A: small model FAILS, large model SUCCEEDS → capability gap "
            "  Category B: small model SUCCEEDS, large model SUCCEEDS → redundant "
            "  Category C: both fail → large-model trace is also wrong (not used) "
            "Current SFT trains on A + B (SFT_INCLUDE_SUCCESS=1). "
            "SFT_INCLUDE_SUCCESS=1 was added to prevent nan-grad collapse from "
            "too-few training pairs — but the SUCCESS pairs also crowd the gradient "
            "with redundant signal on tasks the small model already solves. "
            ""
            "DEMOPSD FIX: "
            "Only distill from Category A (disagreement: large right, small wrong). "
            "On Category B (both right): use a small weight λ_agree=0.1 to preserve "
            "SFT_INCLUDE_SUCCESS's nan-stability benefit without drowning Category A. "
            "  L_SFT = λ_disagree * L_CE(Category A) + λ_agree * L_CE(Category B) "
            "  where λ_disagree=1.0, λ_agree=0.1. "
            "This is a pure data-weighting step — no architecture change. "
            "Category A classification: query where small_rollout_results has "
            "0/G passing rollouts (consistent small-model failure). "
            "Available directly from traces.jsonl's 'small_rollout_results' field. "
            ""
            "EXPECTED IMPACT ON SKILLS ARM: "
            "Current skills arm: 75.61% pass@1 (20.7pp below large's 96.34%). "
            "The skills arm bottleneck is the SFT-trained model's competence on "
            "hard tasks — precisely the Category A tasks. "
            "DemoPSD focuses 10x more gradient on these hard Category A tasks. "
            "If 20% of SFT steps currently go to Category A and 80% to Category B, "
            "DemoPSD flips this: ~90% effective gradient weight on Category A. "
            "Even a conservative 30% improvement in skills arm hard-task accuracy "
            "would push skills arm from 75.61% toward 80%, narrowing the gap with "
            "large (96.34%). "
            ""
            "RELATIONSHIP TO EXISTING EXPERIMENTS: "
            "EXP-087 (FOREVER/Ebbinghaus SFT replay): schedules WHEN to replay "
            "past traces for forgetting. DemoPSD addresses WHAT to replay — "
            "complementary at a different level. "
            "EXP-095 (DDWA decision-density weighting): weights traces by how many "
            "decision pivots the large model used. DemoPSD weights by disagreement "
            "outcome — same weighting idea but different criterion. DDWA + DemoPSD "
            "could be combined (weight by disagreement × decision-density). "
            "EXP-080/081 (EMNLP): already incorporates SFT on hard tasks for tau2. "
            "DemoPSD extends this idea to HumanEval with the disagreement filter. "
            ""
            "PAPER RELEVANCE NOTE: "
            "DemoPSD (2607.02502) was submitted July 2, 2026 — 2 days before this "
            "experiment proposal. It is the most recent arxiv paper in today's set. "
            "The paper's key finding: 'uniform OPSD distillation causes overfitting "
            "to in-domain patterns and suppresses exploration on hard tasks.' "
            "This precisely describes our SFT's failure mode (B-category crowding). "
            "Citing this paper in the AAAI submission provides a clean theoretical "
            "justification for disagreement-filtered SFT. "
        ),
        "spec": {
            "pipeline": "humaneval_demopsd_disagreement_sft",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 1,
            "sft_weighting": "demopsd_disagreement",
            "demopsd_lambda_disagree": 1.0,
            "demopsd_lambda_agree": 0.1,
            "demopsd_agree_threshold_pass_rate": 0.0,
            "sft_include_success": True,
            "scaling_force_both": True,
            "eval_arm": "skills",
            "eval_data": _HE_EVAL,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 skills arm: 75.61% HE pass@1",
                "arxiv:2607.02502 (DemoPSD uniform-OPSD overfitting on easy tasks)",
            ],
            "implementation_note": (
                "In traces_to_sft.py: after loading traces.jsonl, compute "
                "small_pass_rate = mean(small_rollout_results) for each query. "
                "Assign weight 1.0 to traces where small_pass_rate < threshold "
                "(small model consistently failed: Category A / disagreement). "
                "Assign weight 0.1 to traces where small_pass_rate >= threshold "
                "(small model sometimes passed: Category B / agreement). "
                "Apply per-sample loss weighting in the SFT DataLoader. "
                "~15 lines additional code in traces_to_sft.py."
            ),
            "primary_metrics": [
                "humaneval_pass_at_1_skills_arm_post_sft",
                "sft_category_a_count",
                "sft_category_b_count",
                "sft_effective_gradient_ratio_a_vs_b",
                "sft_train_loss_category_a",
                "sft_train_loss_category_b",
                "skills_arm_hard_task_pass_rate",
                "skills_arm_easy_task_pass_rate",
            ],
            "target_outcome": (
                "Skills arm pass@1 >= 78% (+2.4pp from 75.61%). "
                "Hard-task (Category A) accuracy improves more than easy-task (Category B). "
                "Gradient ratio confirms Category A receives ~10x the effective weight. "
                "If skills arm improves, this reduces the 20.7pp skills-vs-large gap "
                "and strengthens the case for AAAI's 'continual distillation' narrative. "
            ),
            "gpu": "auto",
            "estimated_gpu_hours": 1.5,
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
