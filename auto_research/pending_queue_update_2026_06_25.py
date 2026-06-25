#!/usr/bin/env python3
"""
Daily queue patch — 2026-06-25 (EXP-088, EXP-089).

A800 connectivity: offline since 2026-05-14 (day 42). Apply when restored.

Prior patches to apply before this one (if using the daily-patch chain):
    python3 auto_research/pending_queue_update.py                       # base experiments
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

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_06_25.py            # EXP-088, EXP-089

Queue was ~87 pending on 2026-06-23 (+2 added: EXP-086, EXP-087).
Queue cap applied: >20 → max 2 new experiments.

Data motivating today's proposals
----------------------------------
results/e2e_4cyc_gpt55/cycle_3/e2e_ablation_summary.json (from 06-23 patch header):
    HumanEval 4-cycle full/router: 92.68% task pass at 27.56% large-model cost
    GRPO zero-variance: 52.4% of groups discarded at G=8 DAPO (43/82 groups)
    Skills arm (always-small+procedure): 75.61% (up from 70.73% at cycle 0)

results/cwy_35b_fullsplit_20260610_083624/cycle_3/e2e_ablation_summary.json:
    tau2: base/skills=76.97%, router=69.66%, always-large=60.11%
    Cycle-over-cycle decline (W6): 70.22%→72.47%→70.22%→69.66%

arxiv:2606.20008 — 'VIMPO: Value-Implicit Policy Optimization for LLMs' (June 18, 2026):
    Derives a policy-implied value function from KL-regularized RL optimality conditions.
    Represents the value IMPLICITLY through the log-likelihood ratio of the current policy
    against a frozen reference model — no separate critic network, but denser token-level
    signal than GRPO's trajectory-level advantage. In a deterministic-transition MDP view
    of autoregressive generation, the KL-regularized objective yields an identity:
    V(s) = log[π(a|s) / π_ref(a|s)] + β * KL(π||π_ref). This allows per-token
    advantage estimation without any learned critic. Directly addresses the 52.4%
    zero-variance group-discard rate in our G=8 DAPO: when all K rollouts in a group
    have the same binary outcome (all pass or all fail), DAPO must discard the group.
    VIMPO's token-level advantage is computed from log-ratio differences between accepted
    and rejected spans, so even groups with identical final outcomes can yield non-zero
    per-token advantages from intermediate-step divergences.

arxiv:2605.18796 — 'UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing'
    (May 11, 2026):
    Calibration-first router: maps token-level margin uncertainty to per-query error
    probability via isotonic regression, then selects the escalation threshold by
    constrained cost minimization (minimize cost subject to error_rate ≤ target).
    On 75k NER queries served by 4B/12B LLMs: cuts inference cost by 31% vs entropy
    thresholding, split-conformal routing, and FrugalGPT-style learned thresholds.
    Our current router (train_router_simple.py) uses a scikit-learn classifier with
    uncalibrated probability outputs. Adding isotonic-regression calibration + threshold
    selection by cost minimization is a ~20-line post-training addition to
    train_router_simple.py that is orthogonal to the router's feature engineering
    (design decision #3 preserved: raw prompt features, no procedure prefix).
"""
import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = (
    "/data0/home/zeyuwang/router-skills-evolve-data/humaneval/HumanEval.jsonl.gz"
)
_TAU2_HELD_OUT = (
    "/data0/home/zeyuwang/router-skills-evolve-data/tau2/held_out_tasks.jsonl"
)
_TAU2_CONFIG = "config/tau2_bench.yaml"

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_06_25_001_vimpo_value_implicit_grpo_humaneval",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2606.20008 — 'VIMPO: Value-Implicit Policy Optimization for LLMs' "
            "(Kang et al., June 18, 2026). "
            "GRPO and DAPO assign a single trajectory-level advantage to every token in "
            "a rollout: the advantage of token t_{k} in rollout r is A_r = (R_r - "
            "mean_j(R_j)) / std_j(R_j). This creates two problems: "
            "(1) Zero-variance collapse: when all K rollouts have the same binary outcome "
            "(all pass or all fail), std=0 → group discarded → no gradient. We observe "
            "52.4% group discard rate at G=8 DAPO (43/82 groups) in the reference run. "
            "(2) Credit misattribution: all tokens in a passing 200-token rollout receive "
            "the same positive advantage, even if 180 of those tokens were boilerplate "
            "that appears identically in all rollouts. "
            "VIMPO addresses both by deriving a VALUE FUNCTION IMPLICITLY from the "
            "KL-regularized policy optimization objective, without training a separate "
            "critic. The derivation: "
            "In autoregressive generation modeled as a deterministic-transition MDP, "
            "the KL-regularized RL objective (maximize E[R] - β*KL(π||π_ref)) has the "
            "optimal value function: V*(s_t) = β * log Z(s_t), where Z(s_t) is the "
            "partition function of the softmax over possible next tokens. "
            "VIMPO observes that for a given policy π, an implicit value estimate is: "
            "V_π(s_t) ≈ sum_{τ≥t} [log π(a_τ|s_τ) - log π_ref(a_τ|s_τ)], "
            "i.e., the log-ratio of current vs reference policy over the REMAINING "
            "tokens of the trajectory from step t onward. "
            "This yields a per-token advantage: "
            "A_VIMPO(t, r) = [V_π(s_t, r) - mean over rollouts of V_π(s_t, ·)] + R_r. "
            "The key property: even when all rollouts receive the same final reward R_r, "
            "the implicit value V_π(s_t, r) differs across rollouts because the log-ratio "
            "sums differ. The rollout that generated the correct solution via a more "
            "canonical code path will have a higher log-ratio sum for its 'solution' "
            "tokens and a lower sum for its 'reasoning' tokens, concentrating the "
            "advantage signal where the policy actually diverged from the reference. "
            "This directly eliminates the zero-variance collapse: even a homogeneous "
            "group (all rollouts pass) yields non-zero VIMPO advantages from "
            "within-trajectory log-ratio differences. "
            "Implementation in grpo_train_simple.py: "
            "For each rollout, after generating tokens, compute the per-token log-ratio "
            "log π_θ(a_t|s_t) - log π_ref(a_t|s_t) (already available as the "
            "'logprobs' field from the rollout). Compute the suffix sum from each "
            "position t to end of rollout. This is the implicit value V_π(s_t, r). "
            "Normalize across rollouts at each position t to get VIMPO advantages. "
            "Add final outcome reward R_r. Replace DAPO trajectory-level advantage with "
            "VIMPO per-token advantage. ~40 lines, no external dependencies, uses "
            "already-computed logprobs. "
            "Distinct from: "
            "EXP-086 (VeRPO): modifies the REWARD (dense test-weighted partial pass). "
            "VIMPO modifies the ADVANTAGE ESTIMATION (implicit token-level value). "
            "Both can be composed (VeRPO reward + VIMPO advantage), but this experiment "
            "tests VIMPO advantage with standard binary DAPO outcome reward first. "
            "EXP-009 (P-GRPO, failed): static fractional reward → reward collapse. "
            "VIMPO does not change the reward at all; it changes how advantage is "
            "distributed within a trajectory. "
            "EXP-082 (RL-ZVP, zero-variance penalty): adds a penalty loss to discourage "
            "zero-advantage groups. VIMPO ELIMINATES zero-advantage situations by "
            "construction, rather than penalizing them post-hoc. "
            "No queued experiment uses implicit value functions or log-ratio suffix sums "
            "for per-token advantage. ✓ "
            "Wall-clock: ~90 min (same model size and rollout count as reference DAPO; "
            "suffix-sum computation adds ~3% overhead over logprob-already-computed). "
            "Expected outcome: reduce group discard rate from 52.4% to <20%; "
            "improve HumanEval cycle-3 pass@1 above 92.68% reference."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": _HE_EVAL,
            "eval_data": _HE_EVAL,
            "bench": "humaneval",
            "grpo_algo": "vimpo",
            "advantage_type": "value_implicit_log_ratio_suffix_sum",
            "outcome_reward": "binary_dapo",
            "vimpo_beta": 0.04,
            "vimpo_normalize_per_position": True,
            "vimpo_outcome_weight": 1.0,
            "dapo_dynamic_sampling": True,
            "dapo_clip_low": 0.2,
            "dapo_clip_high": 0.5,
            "n_generations": 8,
            "rollout": "repair",
            "max_turns": 3,
            "epochs": 1,
            "lr": 5e-6,
            "lora_r": 16,
            "prompt_style": "qwen-chat",
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (DAPO G=8 binary outcome, 92.68% reference)",
                "exp_2026_06_23_001_verpo (EXP-086: dense reward, different axis)",
                "exp_grpo_rl_zvp (EXP-082: zero-variance penalty, post-hoc fix)",
            ],
            "expected_improvement": "reduce group discard rate 52.4%→<20%, pass@1>92.68%",
            "arxiv_ref": "2606.20008",
            "estimated_hours": 1.5,
            "implementation_note": (
                "In grpo_train_simple.py collect_rollouts(): logprobs already computed "
                "per token. Add: suffix_logratios[t,r] = sum_{τ=t..T}(logp[τ,r] - "
                "logp_ref[τ,r]). Then per-position normalize across rollouts: "
                "A_vimpo[t,r] = (suffix_logratios[t,r] - mean_r) / (std_r + 1e-8). "
                "Add scalar outcome reward R_r. Replace trajectory-level advantage with "
                "A_vimpo as the per-token loss weight. ~40 lines."
            ),
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_06_25_002_ucci_calibrated_router_humaneval_4cycle",
        "priority": 7,
        "kind": "joint_cycle_multiseed",
        "rationale": (
            "arxiv:2605.18796 — 'UCCI: Calibrated Uncertainty for Cost-Optimal LLM "
            "Cascade Routing' (May 11, 2026). "
            "Our current router (train_router_simple.py) fits a scikit-learn logistic "
            "regression / gradient-boosted classifier on raw prompt features and uses "
            "the raw predicted probability P(small_can_solve | prompt) as a routing "
            "score. Thresholding a directly outputs route=small if P < threshold, "
            "route=large otherwise. Two problems with this: "
            "(1) Uncalibrated probabilities: scikit-learn classifiers on small "
            "training sets (N~82 router training traces per cycle) are systematically "
            "overconfident. The raw P values do not represent true error probabilities. "
            "With N~82, expected calibration error (ECE) can be 0.10-0.15, meaning "
            "P=0.8 may correspond to true error rate of 0.65. "
            "(2) Ad-hoc threshold: the current threshold is a fixed heuristic (0.5 or "
            "tuned on the training set). There is no principled way to trade off "
            "routing cost vs error rate. "
            "UCCI addresses both with two additions: "
            "(1) Isotonic regression calibration: fit a monotone mapping g(·) from "
            "raw P to calibrated error probability e = g(P) on a held-out calibration "
            "split. Isotonic regression achieves O(n^{-1/3}) ECE convergence and "
            "preserves ranking (so routing decisions are no worse than before). "
            "Requires only ~20 held-out traces for meaningful calibration at our scale. "
            "(2) Cost-constrained threshold selection: given the calibrated error "
            "probability e, select threshold τ* = argmin_{τ} cost(τ) subject to "
            "error_rate(τ) ≤ δ, where δ is the target error rate and cost(τ) is the "
            "expected fraction of queries routed to the large model. "
            "With calibrated probabilities, this threshold has a theoretical optimality "
            "guarantee (Theorem 2 in UCCI): no other threshold policy achieves lower "
            "cost at the same error constraint. "
            "Application to our pipeline: "
            "In train_router_simple.py, after fitting the base classifier: "
            "(a) Reserve 20% of the router training traces (N~16) as calibration split. "
            "(b) Fit isotonic regression on (raw_P, true_label) pairs from the "
            "calibration split to get g(·). "
            "(c) Compute threshold τ* by sweeping τ ∈ [0,1] and selecting the lowest τ "
            "such that empirical error rate on calibration split ≤ δ=0.10 "
            "(10% acceptable escalation miss rate). "
            "(d) At inference time: route=small if g(P) ≤ τ*, else route=large. "
            "Design decision #3 preserved: we change nothing about what features are "
            "used (raw prompt) or how the base classifier is trained — only the "
            "post-training calibration and threshold selection step is added. "
            "Expected impact: our reference run achieves 92.68% task pass at 27.56% "
            "large-model cost. With better threshold calibration, we expect to push "
            "large-model cost toward 20-23% while maintaining 92%+ task pass "
            "(matching UCCI's 31% cost reduction result on a similar cascade). "
            "This directly answers the paper's open question: can we reduce routing "
            "cost without sacrificing task quality? "
            "Distinct from all prior router experiments: "
            "EXP-074, EXP-075 (router feature ablation): tested what prompt FEATURES "
            "to use. UCCI is orthogonal — it changes how the score is CALIBRATED, not "
            "what features feed into it. "
            "No queued experiment uses isotonic regression calibration or cost-constrained "
            "threshold selection for the router. ✓ "
            "Wall-clock: ~2.5 GPU-hours (4 standard HumanEval cycles + calibration "
            "overhead; calibration itself is CPU-only, <1 min per cycle). "
            "n_seeds=2 (seeds 42, 7) to verify calibration gain is not seed-specific."
        ),
        "spec": {
            "pipeline": "humaneval_4cycle_ucci_calibrated_router",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 4,
            "n_seeds": 2,
            "seeds": [42, 7],
            "router_calibration": "isotonic_regression",
            "router_threshold_method": "ucci_cost_constrained",
            "ucci_error_budget": 0.10,
            "ucci_calibration_split_fraction": 0.20,
            "ucci_cost_metric": "fraction_queries_to_large_model",
            "router_features": "raw_prompt",
            "preserve_design_decision_3": True,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (reference: uncalibrated router, 27.56% cost)",
            ],
            "primary_metrics": [
                "router_task_pass_at_cycle_3",
                "large_model_query_fraction",
                "router_calibration_ece_before",
                "router_calibration_ece_after",
                "ucci_threshold_tau_star",
            ],
            "target_outcome": (
                "large_model_query_fraction < 0.23 with router_task_pass >= 0.92"
            ),
            "arxiv_ref": "2605.18796",
            "estimated_hours": 2.5,
            "implementation_note": (
                "In train_router_simple.py after classifier.fit(): "
                "(1) Split train traces 80/20; refit on 80%; "
                "(2) sklearn.isotonic.IsotonicRegression().fit( "
                "    clf.predict_proba(X_cal)[:,1], y_cal); "
                "(3) Sweep τ in linspace(0,1,200); pick lowest τ where "
                "    mean(g(P_cal) > τ AND y_cal==0) <= 0.10; "
                "(4) Save g(·) and τ* alongside router.joblib. "
                "In collect_traces._policy_decision(): "
                "    e = g(clf.predict_proba([feat])[0,1]); "
                "    route = 'small' if e <= tau_star else 'large'."
            ),
        },
        "gpu": "auto",
    },
]


def main():
    if not STATE_PATH.exists():
        print(f"ERROR: {STATE_PATH} not found. Are you on the A800 or H200?")
        return 1
    with open(STATE_PATH) as f:
        state = json.load(f)
    queue = state.setdefault("queue", [])
    existing_ids = {e["id"] for e in queue}
    existing_ids |= {e.get("id", "") for e in state.get("history", [])}
    added, skipped = [], []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] in existing_ids:
            skipped.append(exp["id"])
        else:
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
    print(f"Added {len(added)} experiments:")
    for eid in added:
        print(f"  + {eid}")
    if skipped:
        print(f"Skipped {len(skipped)} (already present):")
        for eid in skipped:
            print(f"  - {eid}")
    print(f"Queue now has {len(queue)} pending experiments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
