#!/usr/bin/env python3
"""
Daily queue patch — 2026-06-29 (EXP-096, EXP-097).

A800 connectivity: offline since 2026-05-14 (day 46). Apply when restored.

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

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_06_29.py            # EXP-096, EXP-097

Queue was ~95 pending on 2026-06-28 (+2 added: EXP-094, EXP-095).
Queue cap applied: >20 → max 2 new experiments.

Data motivating today's proposals
----------------------------------
results/e2e_4cyc_gpt55/cycle_3/e2e_ablation_summary.json:
    HumanEval 4-cycle full/router: 92.68% task pass at 27.56% large-model cost
    GRPO zero-variance / advantage collapse rate (ACR): 52.4% of groups at G=8 DAPO
    Skills arm (always-small+procedure): 75.61% (up from 70.73% at cycle 0)
    Full = Router (GRPO cannot surpass router; entropy collapse prevents gradient flow)
    Router accuracy: 92.68%, fallback rate: 6.1%, cost vs always-large: 27.56%

results/cwy_35b_joint_20260606_165203/cycle_3/e2e_ablation_summary.json:
    tau2: always-small=68.91%, router=72.97%, always-large=68.91%
    tau2 router accuracy: 89.19% at 35.5% large-model cost
    tau2 Full = Router (GRPO net-neutral on tau2 across cycle 3)

Deferred item now due (from ideas-2026-06-17.md):
    arxiv:2505.12601 kNN router ablation: explicitly deferred post-EMNLP (deadline
    2026-06-19). Today is 2026-06-29 — 10 days post-EMNLP. AAAI 2027 deadline is
    2026-08-15 (47 days). The ablation is now the highest-priority architectural
    question for the AAAI submission: does the 92.68% routing accuracy come from the
    learned logistic regression classifier or from embedding-space task structure that
    any nearest-neighbor method would capture?

Key gap in ACR-mitigation queue:
    EXP-090 (STARE): token-level surprisal reweighting (changes how advantage is
        distributed over existing tokens)
    EXP-094 (AVSPO): virtual sample injection for collapsed groups (changes training
        dynamics by adding synthetic group members)
    EXP-088 (VIMPO): suffix log-ratio as implicit value baseline (changes the baseline
        subtracted from raw return)
    MISSING: no experiment changes the REWARD FUNCTION itself to introduce continuous
        variation within groups. Adding a normalized length penalty to the binary
        pass/fail reward creates within-group reward spread even when all rollouts pass,
        reducing ACR from the reward-design side without requiring any new virtual
        samples or token reweighting machinery.

arxiv:2503.14476 — 'DAPO: An Open-Source LLM Reinforcement Learning System at Scale'
    (DeepSeek, Qiao et al., March 2025):
    Section 4.2 explicitly analyzes binary (0/1) reward functions for code and math
    tasks. Binary rewards cause two failure modes: (1) all-pass groups (trivially easy
    prompts) → zero advantage → zero gradient; (2) all-fail groups (too-hard prompts)
    → zero advantage → zero gradient. DAPO's dynamic sampling heuristic mitigates this
    by resampling prompts where all rollouts produce the same outcome, but as observed
    in our cycle-3 data, 52.4% ACR persists even with dynamic sampling. A secondary
    continuous reward component — e.g. normalized solution length or test-coverage
    fraction — supplements binary pass/fail with a within-group differentiating signal
    that survives even homogeneous pass/fail outcomes, directly addressing DAPO's
    Section 4.2 failure mode from the reward-design side.

arxiv:2505.12601 — 'Rethinking Predictive Modeling for LLM Routing: When Simple kNN
    Beats Complex Learned Routers' (May 2025):
    Demonstrates that embedding-based k-nearest-neighbor routing matches or exceeds
    trained neural routers and gradient-boosted classifiers on several routing
    benchmarks. The core argument: routing difficulty is driven by query-type similarity
    (is this a code task? a multi-hop reasoning task?) rather than fine-grained
    difficulty estimation, and kNN on pre-trained sentence embeddings captures
    query-type proximity as effectively as a trained model. Applied to our system:
    if kNN matches our 92.68% routing accuracy, it (a) validates that the routing
    decision is embedding-space-separable and (b) dramatically simplifies the MERA
    system description for AAAI reviewers — replacing a trained router with a
    parameter-free kNN underscores that the benefit comes from the skill-routing
    structure, not from the classifier complexity.
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
        "id": "exp_2026_06_29_001_length_penalty_reward_acr_humaneval",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2503.14476 — 'DAPO: An Open-Source LLM Reinforcement Learning "
            "System at Scale' (DeepSeek, Qiao et al., March 2025), Section 4.2 "
            "analysis of binary reward failure modes. "
            ""
            "ROOT PROBLEM: 52.4% ACR (Advantage Collapse Rate) persists at cycle 3 "
            "HumanEval even with DAPO G=8 dynamic sampling (43/82 groups have "
            "zero within-group reward variance → vanishing gradient → Full = Router). "
            "Three queued experiments attack ACR via TRAINING DYNAMICS: "
            "  EXP-090 (STARE): token-level surprisal reweighting on non-zero groups "
            "  EXP-094 (AVSPO): virtual sample injection into collapsed groups "
            "  EXP-088 (VIMPO): suffix log-ratio as improved value baseline "
            "NONE of these changes the REWARD FUNCTION itself. All three still rely "
            "on binary pass/fail as the only reward signal — so in an all-pass group "
            "every rollout gets R=1 and the GRPO advantage is still 0 before any "
            "modification. STARE's surprisal weights and AVSPO's virtual samples "
            "must compensate for this upstream binary reward limitation. "
            ""
            "THIS EXPERIMENT: Supplement the binary pass/fail reward with a "
            "normalized length penalty that introduces continuous within-group "
            "differentiation: "
            "  R_total(rollout k) = R_pass(k) - alpha * len_k / L_max "
            "where: "
            "  R_pass(k) ∈ {0, 1} — standard HumanEval binary reward "
            "  len_k = number of tokens generated in rollout k "
            "  L_max = 150 (95th percentile HumanEval solution length) "
            "  alpha = 0.1 (so max length penalty = 0.1, preserving 0/1 dominance) "
            ""
            "WHY THIS REDUCES ACR: "
            "In an all-pass group (R_pass=1 for all G=8 rollouts), the rollouts "
            "now have: R_total_k = 1.0 - 0.1 * len_k / 150. Since HumanEval "
            "solutions vary in length (σ_len ≈ 15-30 tokens across G=8 rollouts), "
            "the within-group variance of R_total is: "
            "  Var(R_total) = (0.1/150)^2 * Var(len_k) "
            "             ≈ (0.1/150)^2 * (20)^2 ≈ 1.8e-4 "
            "This is small but nonzero — the group no longer collapses to zero "
            "variance when all rollouts pass. The advantage signal now "
            "differentiates between compact and verbose correct solutions. "
            "For all-fail groups (R_pass=0 for all), the length variation similarly "
            "creates a nonzero within-group signal: shorter failed solutions get "
            "relatively less negative advantage (less length penalty paid). "
            ""
            "WHY alpha=0.1 IS CALIBRATED: "
            "The length penalty must not dominate the pass/fail signal (a passing "
            "100-token solution should still be preferred over a failing 20-token "
            "solution): R_total(pass,100t) = 0.933 >> R_total(fail,20t) = -0.013. "
            "The 10× gap ensures the pass/fail decision drives routing while the "
            "length variation creates within-group spread. "
            "alpha=0.05 (under-regularized): insufficient variance for collapsed "
            "all-pass groups. alpha=0.2 (over-regularized): may bias model toward "
            "syntactically minimal solutions that aren't readable (ablation: compare "
            "alpha in {0.05, 0.1, 0.2} if EXP-096 is promising). "
            ""
            "SECONDARY BENEFIT FOR AAAI: "
            "Length-aware rewards are a natural fit for the MERA paper's narrative: "
            "the system not only routes correctly but learns to generate concise "
            "solutions. If EXP-096 reduces ACR and improves pass@1, it also reduces "
            "average generated token count (a practical deployment benefit: fewer "
            "tokens → lower inference cost). "
            ""
            "DISTINCTNESS FROM PRIOR QUEUE: "
            "EXP-086 (VeRPO, arxiv:2601.03525): modifies reward by dynamic "
            "test-DIFFICULTY weighting — changes which tests count toward R, not "
            "adding a length term. "
            "EXP-090 (STARE): token-level surprisal reweighting of existing "
            "advantages — does not change the reward function, still zero advantage "
            "for all-pass groups before STARE applies. "
            "EXP-094 (AVSPO): injects virtual samples — operates after reward "
            "is assigned, not at the reward function design level. "
            "EXP-088 (VIMPO): suffix log-ratio baseline — a value-function "
            "correction, not a reward shaping approach. "
            "No queued experiment has reward_fn='pass_minus_length' or any "
            "length_penalty_alpha field. ✓ "
            ""
            "IMPLEMENTATION: ~10 lines in grpo_train_simple.py or collect_traces.py: "
            "  len_tokens = rollout_output_ids.shape[-1] "
            "  L_MAX = 150  # 95th pct HE solution length "
            "  length_penalty = ALPHA * (len_tokens / L_MAX) "
            "  reward = float(tests_pass) - length_penalty "
            "The rest of the GRPO pipeline (DAPO normalization, dynamic sampling) "
            "is unchanged. Estimated overhead: < 1% (token count is already available "
            "from the generation output). "
            ""
            "EXPECTED OUTCOME: "
            "ACR reduced from 52.4% baseline toward 35-40% (all-pass and all-fail "
            "groups now have nonzero R_total variance via length spread). "
            "HumanEval pass@1 gains 1-3 pp at cycle 1 (conservative estimate given "
            "the small within-group variance added). "
            "Average solution length decreases by 10-15 tokens (conciseness "
            "incentive active). "
            "Full > Router at cycle 1 if ACR reduction unblocks gradient flow. "
            "Wall-clock: 1 cycle × ~1.5 GPU-hours = 1.5 hours. "
        ),
        "spec": {
            "pipeline": "humaneval_length_penalty_reward",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 1,
            "grpo_algo": "dapo",
            "reward_fn": "pass_minus_length",
            "length_penalty_alpha": 0.1,
            "length_penalty_L_max": 150,
            "dapo_dynamic_sampling": True,
            "n_generations": 8,
            "grpo_temperature": 1.0,
            "eval_data": _HE_EVAL,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (DAPO binary reference: 92.68% HE, ACR=52.4%)",
                "exp_2026_06_28_001_avspo (EXP-094: group-level virtual sample injection)",
                "exp_2026_06_26_001_stare (EXP-090: token-level surprisal reweighting)",
            ],
            "primary_metrics": [
                "humaneval_pass_at_1_post_1cycle",
                "grpo_acr_mean",
                "grpo_zero_variance_group_rate",
                "grpo_within_group_reward_variance_mean",
                "generated_solution_length_mean_tokens",
                "generated_solution_length_std_tokens",
            ],
            "target_outcome": (
                "ACR ≤ 0.40 (from 0.524 baseline; length variation in all-pass/all-fail "
                "groups creates nonzero within-group spread). "
                "HE pass@1 >= 0.84 after 1 cycle (vs. ~0.83 expected from pure DAPO). "
                "Mean solution length decreases by >= 10 tokens vs. binary-reward baseline "
                "(length penalty incentive active). "
            ),
            "gpu": "auto",
            "estimated_gpu_hours": 1.5,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_06_29_002_knn_router_ablation_he_tau2",
        "priority": 7,
        "kind": "forgetting_eval",
        "rationale": (
            "arxiv:2505.12601 — 'Rethinking Predictive Modeling for LLM Routing: "
            "When Simple kNN Beats Complex Learned Routers' (May 2025). "
            ""
            "DEFERRED ITEM NOW DUE: In the ideas-2026-06-17.md report, this ablation "
            "was explicitly tagged 'Deferred — post-EMNLP' (EMNLP abstract deadline "
            "2026-06-19). It is now 2026-06-29 — 10 days post-EMNLP — and the AAAI "
            "2027 deadline is 2026-08-15 (47 days). This is the correct window to run "
            "the kNN router ablation: enough time to revise the paper if kNN matches "
            "the trained router (simplifying the narrative), or to strengthen the "
            "trained-router case if kNN underperforms. "
            ""
            "THE ABLATION QUESTION: "
            "Our trained router (scikit-learn logistic regression on prompt features "
            "+ cycle-N SFT embedding) achieves 92.68% routing accuracy on HumanEval "
            "and 89.19% on tau2 at cycle 3. Does this accuracy come from: "
            "(A) The learned decision boundary of the trained classifier, OR "
            "(B) The embedding-space structure of task types, which any proximity-"
            "    based method would exploit? "
            "arxiv:2505.12601 shows that across multiple routing benchmarks, a simple "
            "k-Nearest-Neighbor router operating on sentence-embedding similarities "
            "matches or exceeds trained gradient-boosted classifiers (GBT, MLP, "
            "logistic regression). The paper's central claim: routing difficulty is "
            "dominated by query TYPE (code / math / knowledge), and embedding-based "
            "kNN captures type proximity as well as a trained model. "
            ""
            "WHY THIS MATTERS FOR MERA/AAAI: "
            "(A) If kNN ≈ trained router: This is a positive finding. It tells AAAI "
            "reviewers that the 92.68% routing accuracy is a robust, stable property "
            "of the task space — not an artifact of careful classifier design. The "
            "MERA system simplifies: replace 'train a logistic regression router' "
            "with 'run kNN on prompt embeddings'. This makes Phase 4 (train_router) "
            "trivially reproducible and reduces the method description by ~2 paragraphs. "
            "(B) If trained router > kNN (e.g., +3–5 pp): This validates that the "
            "learning component of MERA's Phase 4 router is essential — the model "
            "learns something the embedding structure alone does not encode. This is "
            "a stronger paper claim. "
            "Either way, the comparison is an important ablation that belongs in the "
            "AAAI submission. "
            ""
            "IMPLEMENTATION: "
            "1. Extract cycle-3 raw prompts (already available in traces.jsonl). "
            "2. Encode prompts with a small sentence-BERT model "
            "   (all-MiniLM-L6-v2, ~22M params, CPU-runnable, no GPU needed). "
            "3. For each test prompt, find k=5 nearest neighbors in the training "
            "   traces by cosine similarity. "
            "4. Route to small model if majority of k neighbors' ground-truth "
            "   labels are 'small_can_solve'; else route to large model. "
            "5. Compute routing accuracy, fallback rate, and cost vs. always-large "
            "   (same metrics as existing router evaluation). "
            "6. Compare to trained logistic regression router at matching threshold. "
            "Entire implementation is a ~50-line Python script, CPU-only. "
            "Both HumanEval (82 test traces) and tau2 (74 test traces) can be "
            "evaluated in <15 minutes total. "
            ""
            "DISTINCTNESS FROM PRIOR QUEUE: "
            "EXP-089 (UCCI, arxiv:2605.18796): calibrates the EXISTING trained "
            "logistic regression router with isotonic regression — does not replace "
            "it with kNN. UCCI improves threshold selection on the trained model; "
            "this experiment tests whether the trained model is necessary at all. "
            "EXP-066 (tau2 router regularization): adds L2 regularization to the "
            "existing trained router to reduce overfitting — again, not kNN replacement. "
            "No queued experiment has knn_router=True or any knn-related routing field. ✓ "
            ""
            "KIND CHOICE — forgetting_eval: "
            "This is an evaluation-only experiment (no GRPO, no SFT, no gradient "
            "updates). The 'forgetting_eval' kind in runner.py handles evaluation "
            "sweeps on fixed checkpoints. We evaluate the kNN router on cycle-3 "
            "trained embeddings (from the cycle-3 SFT small model as encoder) as "
            "well as on pre-trained sentence-BERT embeddings — two kNN variants "
            "to test whether fine-tuned code embeddings improve over general "
            "sentence embeddings. "
            ""
            "EXPECTED OUTCOME: "
            "Option A (kNN matches trained): kNN accuracy 91–93% HumanEval "
            "(within 1 pp of 92.68% trained router). Paper narrative: 'embedding-"
            "space routing structure is robust; kNN provides equivalent routing "
            "accuracy with no training.' "
            "Option B (trained router wins): kNN accuracy 85–90% HumanEval "
            "(3–8 pp gap). Paper narrative: 'learned router captures difficulty "
            "signal beyond embedding proximity; Phase 4 training is essential.' "
            "Wall-clock: <30 minutes (CPU-only; sentence-BERT inference on 156 "
            "prompts is fast). "
        ),
        "spec": {
            "pipeline": "knn_router_ablation_he_tau2",
            "bench": "both",
            "eval_benches": ["humaneval", "tau2_bench"],
            "router_type": "knn",
            "knn_router": True,
            "knn_k": 5,
            "knn_similarity": "cosine",
            "knn_encoder_variants": [
                "all-MiniLM-L6-v2",
                "cycle3_sft_small_model_encoder",
            ],
            "comparison_router": "cycle3_trained_logistic_regression",
            "eval_data_he": _HE_EVAL,
            "eval_data_tau2": _TAU2_DATA,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 trained router: 92.68% HE accuracy",
                "cwy_35b_joint_20260606_165203 cycle_3 trained router: 89.19% tau2 accuracy",
                "exp_2026_06_25_002_ucci (EXP-089: calibrated logistic regression router)",
            ],
            "primary_metrics": [
                "he_knn_routing_accuracy_minilm",
                "he_knn_routing_accuracy_cycle3_sft",
                "tau2_knn_routing_accuracy_minilm",
                "tau2_knn_routing_accuracy_cycle3_sft",
                "he_trained_router_routing_accuracy_cycle3",
                "tau2_trained_router_routing_accuracy_cycle3",
                "routing_accuracy_delta_knn_vs_trained",
                "knn_large_model_fraction",
                "trained_router_large_model_fraction",
            ],
            "target_outcome": (
                "kNN routing accuracy within ±3 pp of trained router on HumanEval "
                "(92.68% trained → kNN >= 89.68% confirms embedding-structure hypothesis). "
                "kNN routing accuracy measured on both sentence-BERT and cycle-3 SFT "
                "encoder to isolate whether code fine-tuning improves routing embeddings. "
                "Result is a definitive ablation for the AAAI 2027 paper Section on router design. "
            ),
            "gpu": "auto",
            "estimated_gpu_hours": 0.1,
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
