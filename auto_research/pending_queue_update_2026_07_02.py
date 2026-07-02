#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-02 (EXP-102, EXP-103).

A800 connectivity: offline since 2026-05-14 (day 49). Apply when restored.

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

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_02.py            # EXP-102, EXP-103

Queue was ~103 pending on 2026-07-01 (+2 added: EXP-100, EXP-101).
Queue cap applied: >20 → max 2 new experiments.

Data motivating today's proposals
----------------------------------
results/e2e_4cyc_gpt55/cycle_3/e2e_ablation_summary.json:
    HumanEval 4-cycle:
      large (always-large):        task_pass=96.34%, cost_vs_large=100%
      skills (always-small+proc):  task_pass=75.61%, cost_vs_large=10%
      router (logistic, cycle 3):  task_pass=92.68%, routing_acc=92.68%,
                                   cost_vs_large=27.56%, fallback=6.10%
      full (router+GRPO):          task_pass=92.68% ← SAME as router
    Root problem 1 (GRPO): ACR=52.4% — 43/82 groups have zero within-group
    reward variance → zero gradient → Full = Router. Queue has 10+ ACR
    mitigation approaches but NONE uses implicit process signals derived from
    the model's own intermediate policy divergence (EP-GRPO angle, 2605.04960).
    Root problem 2 (Router): logistic regression trained on binary labels (is
    task solved by small model at all?). Binary labels are noisy single-shot
    observations; K=8 GRPO rollouts already provide a distribution over pass/fail.
    DARS (arxiv:2606.06924, June 2026) shows that distribution-aware routing
    supervision from K samples outperforms single-shot binary labels.

arxiv:2605.04960 — EP-GRPO: Entropy-Progress Aligned Group Relative Policy
    Optimization with Implicit Process Guidance (May 6, 2026):
    EP-GRPO addresses three GRPO credit assignment failures: (1) uniform
    token-level granularity ignoring heterogeneous informational value,
    (2) uniform polarity penalizing correct steps and rewarding incorrect ones,
    (3) zero-variance collapse erasing outcome-driven gradients.
    It adds: entropy-gated modulation (amplify high-entropy decision pivots),
    implicit process signals (policy divergence anchored to outcome advantages
    for directional token-level feedback without external reward models), and
    cumulative entropy mapping (progress-aligned advantage normalization).
    Key to our problem: the implicit process signal creates within-group
    differentiation from the model's intermediate generation dynamics, even
    when all rollouts receive identical final rewards. For all-fail groups
    (R_pass=0 for all K=8), individual rollouts diverge from the reference
    policy differently at key intermediate tokens — these divergence patterns
    constitute a genuine process-level advantage signal. No external reward
    model or additional rollouts required.

arxiv:2606.06924 — From Sampled Outcomes to Capability Distributions:
    Rethinking Supervision for LLM Routing (June 2026):
    Proposes DARS (Distribution-Aware Routing Supervision). Key finding:
    single-shot binary labels for router training are noisy observations of
    underlying model capability because LLM generation is stochastic. DARS
    replaces single-shot labels with a distributional capability estimate:
    sample K outputs per task, compute empirical pass rate p̂_K, use p̂_K as
    soft routing supervision (Brier score loss instead of cross-entropy on
    binary labels). DARS also considers input-side semantic variation (query
    paraphrases), but the output-side distributional signal alone improves
    routing. For our project: SCALING_FORCE_BOTH=1 already runs K=8 GRPO
    rollouts from the small model per task. The empirical pass rate across
    these 8 rollouts (p̂_8 ∈ {0, 1/8, 2/8, ..., 1}) is DARS-style soft
    supervision available at zero extra compute. Our current binary label
    "does the small model solve it?" is a single-shot observation that
    conflates genuinely hard tasks (p̂_8=1/8) with inconsistently solved
    tasks (p̂_8=5/8) — DARS distinguishes them.

AAAI 2027 context:
    Deadline: 2026-08-15 (44 days from today).
    EXP-102 (EP-GRPO, today): fresh ACR mitigation angle not yet in queue.
    EXP-103 (DARS router, today): distribution-aware router supervision using
    already-collected K=8 rollouts — zero additional GPU cost, potentially
    +1-2pp routing accuracy for the AAAI routing accuracy table.
"""
import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = (
    "/data0/home/zeyuwang/router-skills-evolve-data/humaneval/HumanEval.jsonl.gz"
)
_TAU2_DATA = (
    "/data0/home/zeyuwang/router-skills-evolve-data/tau2"
)

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_07_02_001_epgrpo_process_guidance_acr_humaneval",
        "priority": 7,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2605.04960 — EP-GRPO: Entropy-Progress Aligned GRPO with "
            "Implicit Process Guidance (Song Yu et al., May 6, 2026). "
            ""
            "ROOT PROBLEM: ACR=52.4% on HumanEval GRPO at cycle 3 — 43/82 groups "
            "have zero within-group reward variance → zero gradient → Full=Router. "
            "All 10+ queued ACR mitigations operate on ONE of: "
            "  (a) reward function modifications (EXP-094 AVSPO, EXP-096 length, "
            "      EXP-098 SA-AH-GRPO, EXP-101 static-analysis) "
            "  (b) token/advantage reweighting (EXP-082 RLZVP, EXP-085 GDPO, "
            "      EXP-090 STARE, EXP-088 VIMPO) "
            "  (c) virtual/synthetic sample injection (EXP-094 AVSPO) "
            "  (d) group count scaling (EXP-077 G=16, EXP-081 scaling curve) "
            "MISSING angle: creating within-group differentiation from the MODEL'S "
            "OWN INTERMEDIATE GENERATION DYNAMICS — without modifying rewards, "
            "without injecting virtual samples, without scaling rollout count. "
            ""
            "HOW EP-GRPO WORKS (three additions to GRPO/DAPO objective): "
            "(1) ENTROPY-GATED MODULATION: for each token position t in a rollout, "
            "compute H_t = entropy of policy distribution at t. Apply an entropy "
            "gate g_t = sigmoid(H_t - H_ref_t) where H_ref_t is the reference "
            "policy's entropy at position t. Only tokens where g_t > 0.5 (the "
            "model is MORE uncertain than the reference at this position) are "
            "'decision pivots' — their advantage signal is amplified. "
            "Effect: concentrates gradient on tokens where the policy has moved "
            "most from the reference, regardless of final outcome. "
            "(2) IMPLICIT PROCESS SIGNALS: for each token position t, compute "
            "the KL divergence D_t = KL(π_θ(·|ctx_t) || π_ref(·|ctx_t)). "
            "Use D_t as an implicit process reward, scaled by the final outcome "
            "advantage A_rollout. Sign(A_rollout) * D_t gives a DIRECTIONAL "
            "process signal: good rollouts (A>0) are rewarded for diverging from "
            "reference at key tokens; bad rollouts (A<0) are penalized for it. "
            "Effect for ALL-FAIL groups: even when A_rollout=0 (collapsed), "
            "D_t varies across rollouts because individual rollouts diverge "
            "differently from the reference at intermediate tokens. This residual "
            "process divergence provides a gradient signal when binary reward is 0. "
            "(3) CUMULATIVE ENTROPY MAPPING: instead of normalizing advantages "
            "within a group by subtracting the group mean (standard GRPO), "
            "normalize by the CUMULATIVE ENTROPY PROGRESS — a trajectory-level "
            "scalar measuring how the model's entropy evolved from start to end. "
            "High-entropy-progress trajectories (model explored more, ended up "
            "at different entropy than reference) receive higher normalization weight. "
            "Effect: progress-aligned normalization replaces the flat mean-subtraction "
            "in collapsed groups, creating a non-trivial advantage distribution. "
            ""
            "WHY THIS IS ORTHOGONAL TO ALL QUEUED APPROACHES: "
            "STARE (EXP-090): reweights existing advantages by token surprisal "
            "  (P(token) under reference). EP-GRPO's entropy-gated modulation uses "
            "  entropy CHANGE (delta from reference), not absolute surprisal. "
            "SA-AH-GRPO (EXP-098): discounts gradient on negative-advantage rollouts "
            "  by adaptive horizon. EP-GRPO creates advantage signal FOR those "
            "  rollouts via implicit process divergence — complementary directions. "
            "AVSPO (EXP-094): injects virtual samples with synthetic rewards. "
            "  EP-GRPO uses only real rollouts; no synthetic data needed. "
            "RLZVP (EXP-082): entropy advantage on model outputs. EP-GRPO uses "
            "  entropy PROGRESS (delta between start and end of rollout) and "
            "  KL divergence from reference — different quantities. "
            "Static-analysis (EXP-101): creates reward spread from CODE STRUCTURE. "
            "  EP-GRPO creates gradient from GENERATION DYNAMICS independent of "
            "  code structure quality. Stackable (could combine, but this exp isolates). "
            ""
            "IMPLEMENTATION: "
            "In grpo_train_simple.py, add three loss terms alongside DAPO loss: "
            "  # (1) Entropy gate "
            "  H_t = -sum(p * log(p+1e-9)) over policy logits at position t "
            "  H_ref_t = -sum(p_ref * log(p_ref+1e-9)) over reference logits "
            "  gate_t = torch.sigmoid(H_t - H_ref_t)  # [batch, seq_len] "
            "  # (2) Implicit process signal "
            "  D_t = F.kl_div(log_p_ref_t, p_t, reduction='none').sum(-1) "
            "  process_signal = sign(A_rollout) * (gate_t * D_t).sum(-1)  # [batch] "
            "  # (3) Cumulative entropy progress normalization "
            "  H_progress = cumulative_entropy_delta(rollout_logits)  # scalar/rollout "
            "  advantages_normalized = (A - A.mean()) / (H_progress + 1e-8) "
            "  # Combined loss: L_EP = L_DAPO + lambda_proc * process_signal.mean() "
            "  #                       - lambda_ent * advantages_normalized.mean() "
            "  lambda_proc = 0.1; lambda_ent = 0.05 (from paper defaults) "
            "Memory overhead: one extra forward pass through reference model to "
            "compute D_t (already done for KL penalty in DAPO). "
            "Wall-clock overhead: ~15% per step vs. DAPO; 1 cycle ≈ 1.75h total. "
            ""
            "EXPECTED OUTCOME: "
            "ACR reduced from 52.4% toward 35-45% via process-divergence signal "
            "  in all-fail groups. "
            "HumanEval pass@1 after 1 cycle: >= 0.83 (baseline 0.756 at skills, "
            "  target same or better than DAPO binary reward 0.927 with less collapse). "
            "Process signal active: entropy-gate activations > 0.5 on >=30% of "
            "  tokens in collapsed-group rollouts. "
        ),
        "spec": {
            "pipeline": "humaneval_epgrpo_process_guidance",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 1,
            "grpo_algo": "ep_grpo",
            "ep_grpo_lambda_proc": 0.1,
            "ep_grpo_lambda_ent": 0.05,
            "ep_grpo_entropy_gate": True,
            "ep_grpo_implicit_process_signal": True,
            "ep_grpo_cumulative_entropy_norm": True,
            "dapo_dynamic_sampling": True,
            "n_generations": 8,
            "grpo_temperature": 1.0,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (DAPO binary reference: 92.68% HE, ACR=52.4%)",
                "exp_2026_06_26_001_stare_surprisal_token_advantage_humaneval (EXP-090)",
                "exp_2026_06_30_001_sa_ah_grpo_entropy_discount (EXP-098)",
                "exp_2026_07_01_002_static_analysis_reward_grpo_humaneval (EXP-101)",
            ],
            "primary_metrics": [
                "humaneval_pass_at_1_post_1cycle",
                "grpo_acr_mean",
                "grpo_zero_variance_group_rate",
                "grpo_within_group_reward_variance_mean",
                "ep_grpo_entropy_gate_activation_rate",
                "ep_grpo_process_signal_magnitude_collapsed_groups",
                "ep_grpo_cumulative_entropy_progress_mean",
            ],
            "target_outcome": (
                "ACR <= 0.40 (from 0.524 baseline; implicit process divergence "
                "provides non-trivial gradient in all-fail groups). "
                "HE pass@1 >= 0.83 after 1 cycle. "
                "Entropy gate active (>0.5) on >= 30% of tokens in collapsed "
                "groups — confirms process signal is non-trivial. "
            ),
            "gpu": "auto",
            "estimated_gpu_hours": 1.75,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_07_02_002_dars_distribution_router_softlabel_humaneval",
        "priority": 6,
        "kind": "forgetting_eval",
        "rationale": (
            "arxiv:2606.06924 — From Sampled Outcomes to Capability Distributions: "
            "Rethinking Supervision for LLM Routing (Guannan Lai, Haoran Hu, et al., "
            "Nanjing University / HKUST, June 2026). "
            "Proposes DARS: Distribution-Aware Routing Supervision. "
            ""
            "ROOT PROBLEM: Our logistic router is trained on binary oracle labels "
            "'label=1 if small model fails, label=0 if small model solves it'. "
            "These binary labels are single-shot noisy observations of the underlying "
            "capability distribution. Concretely (from e2e_4cyc_gpt55 cycle_0 traces): "
            "  - Some 'label=0' tasks (small model solves): the small model only "
            "    passes 2-3 out of 8 GRPO rollouts (p̂_8 = 0.25-0.375). These are "
            "    borderline tasks where the small model sometimes fails — the binary "
            "    label 0 underclaims confidence in the routing decision. "
            "  - Some 'label=1' tasks (small model fails): rarely, the small model "
            "    passes 1-2 of 8 rollouts (p̂_8 = 0.125-0.25). These are hard but "
            "    occasionally solvable — the binary label 1 (always escalate) is "
            "    overconfident. "
            "DARS replaces binary labels with p̂_K: the empirical pass rate across K "
            "sampled rollouts. Train the router to minimize Brier score "
            "(p_router - p̂_K)^2 instead of binary cross-entropy. "
            ""
            "WHY THIS IS ZERO ADDITIONAL GPU COST: "
            "SCALING_FORCE_BOTH=1 already runs K=8 GRPO rollouts per task from the "
            "small model in Phase 1 (collect_traces). These rollouts are logged in "
            "traces.jsonl with pass/fail per rollout. The DARS soft label is: "
            "  p̂_8(task_i) = count(pass) / 8 across the 8 small-model rollouts "
            "No additional sampling, no additional API calls, no additional GPU time. "
            "The only change: router training loss function (binary CE → Brier score) "
            "and routing labels (0/1 → p̂_8 ∈ {0, 0.125, 0.25, ..., 1.0}). "
            ""
            "CURRENT ROUTER TRAINING CONTEXT: "
            "Router trains on raw prompt features + binary label from oracle "
            "(GRPO-rollout-based label exists but is unused for routing). "
            "Router threshold: 0.5 (logistic output > 0.5 → route to large). "
            "With DARS: router learns to output a calibrated capability estimate. "
            "Threshold strategy: instead of p > 0.5, use p < alpha where alpha "
            "is optimized on validation set — allows trading cost for accuracy "
            "along the ROC curve. "
            ""
            "DISTINCTNESS FROM ALL QUEUED ROUTER EXPERIMENTS: "
            "EXP-066 (tau2 router regularization): adds L2/dropout regularization "
            "  to existing binary-label logistic router. Does NOT change supervision. "
            "EXP-083 (tau2_random50_routing_baseline_eval): evaluates routing on "
            "  random 50-task subset. Does NOT change router training. "
            "EXP-089 (ucci_calibrated_router): calibrates router OUTPUT probabilities "
            "  (Platt scaling) to be well-calibrated. Does NOT change training labels. "
            "EXP-097 (kNN router): changes algorithm family (logistic → kNN). "
            "  Still uses binary labels. DARS changes WHAT the router is trained on. "
            "EXP-091 (trust_uncertainty): adds uncertainty signal to routing decisions. "
            "  Does NOT change the core supervision labels. "
            "This is the ONLY experiment that changes the routing SUPERVISION from "
            "binary hard labels to distributional soft labels from K=8 rollouts. "
            ""
            "IMPLEMENTATION: "
            "In train_router_simple.py: "
            "  1. Load traces.jsonl; for each task extract per-rollout pass/fail "
            "     from small model's GRPO rollouts (field: 'small_rollout_results'). "
            "  2. Compute p̂_8 = mean(small_rollout_results) per task. "
            "  3. Replace hard label (0 or 1) with p̂_8 as soft target. "
            "  4. Change loss from sklearn LogisticRegression (cross-entropy) to "
            "     scikit-learn GradientBoostingRegressor with MSE loss (Brier score). "
            "     Alternative: train logistic on threshold-ed soft labels with "
            "     sample_weight = |p̂_8 - 0.5| * 2 (weight by confidence). "
            "  5. Tune threshold alpha on validation split (maximize routing_acc - "
            "     lambda * cost_vs_large) using the existing threshold sweep in "
            "     train_router_simple.py. "
            "Wall-clock: router retraining is <5 minutes on CPU; total eval ~1h "
            "(including trace collection evaluation). "
            ""
            "EXPECTED OUTCOME: "
            "Routing accuracy improves from 92.68% to 93.5-95% on HumanEval cycle 3 "
            "  by correctly handling borderline tasks (small model sometimes solves). "
            "Cost vs large may shift by ±2pp as the optimal threshold changes. "
            "Brier score of router predictions improves vs. binary-label baseline "
            "  (directly measuring calibration quality for AAAI Table 2 appendix). "
            "If improvement is substantial (>1pp): strengthens the AAAI routing "
            "  accuracy claim and motivates a DARS ablation table. "
        ),
        "spec": {
            "pipeline": "humaneval_dars_distribution_router",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "router_supervision": "dars_soft_label",
            "dars_k": 8,
            "dars_soft_label_field": "small_rollout_results",
            "router_loss": "brier_score",
            "router_threshold_strategy": "validation_sweep",
            "use_existing_traces": True,
            "traces_source": "e2e_4cyc_gpt55/cycle_3",
            "n_cycles": 0,
            "eval_data": _HE_EVAL,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (binary-label logistic router: 92.68% routing_acc)",
                "exp_2026_06_25_002_ucci_calibrated_router_humaneval_4cycle (EXP-089: output calibration)",
                "exp_2026_06_29_002_knn_router_ablation (EXP-097: algorithm family change)",
            ],
            "primary_metrics": [
                "routing_accuracy_dars_vs_binary",
                "router_brier_score",
                "cost_vs_large_at_optimal_threshold",
                "fallback_rate_at_optimal_threshold",
                "borderline_task_routing_accuracy",
            ],
            "target_outcome": (
                "Routing accuracy >= 93.5% (from 92.68% binary-label baseline). "
                "Router Brier score decreases vs. binary-label router. "
                "Borderline tasks (p̂_8 in [0.25, 0.75]) routed more accurately "
                "than with binary labels. "
            ),
            "gpu": "auto",
            "estimated_gpu_hours": 1.0,
        },
        "gpu": "auto",
    },
]


def atomic_write(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


def main() -> None:
    if not STATE_PATH.exists():
        raise FileNotFoundError(f"state.json not found at {STATE_PATH}")

    with STATE_PATH.open() as fh:
        state = json.load(fh)

    queue = state.setdefault("queue", [])
    existing_ids = {e["id"] for e in queue}

    added = []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] in existing_ids:
            print(f"[skip] {exp['id']} already in queue")
            continue
        queue.append(exp)
        existing_ids.add(exp["id"])
        added.append(exp["id"])
        print(f"[add]  {exp['id']} (priority={exp['priority']}, kind={exp['kind']})")

    atomic_write(STATE_PATH, state)
    print(f"\nDone. Added {len(added)} experiments: {added}")
    print(f"Total queue size: {len(queue)}")


if __name__ == "__main__":
    main()
