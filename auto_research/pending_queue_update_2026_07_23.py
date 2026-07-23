#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-23 (EXP-128, EXP-129).

A800 connectivity: offline since 2026-05-14 (day ~70). Apply when restored.

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
    python3 auto_research/pending_queue_update_2026_07_21.py            # EXP-126, EXP-127

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_23.py            # EXP-128, EXP-129

Queue was ~137 pending on 2026-07-21 (+2 added: EXP-126, EXP-127).
Queue cap applied: >20 → max 2 new experiments.

Data motivating today's proposals
----------------------------------
results/e2e_4cyc_gpt55/cycle_3/e2e_ablation_summary.json (unchanged — A800 offline day 70):
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

AAAI 2027 deadline: 2026-08-15 (23 days from today).
A800 offline since 2026-05-14 (day 70).

arxiv:2605.27765 — "Restoring the Sweet Spot: Pass-Rate Weighted Self-Distillation
    for LLM Reasoning" (SC-SDPO; Zehao Liu, Yuanpu Cao, Jinghui Chen, Vasant Honavar;
    Penn State; May 27, 2026):

    BACKGROUND: PROBLEM WITH EQUAL-WEIGHTED SFT:
    Standard on-policy SFT (and self-distillation) applies uniform loss weight to
    every training task. This creates a coverage imbalance:
    - Very easy tasks (teacher pass rate p ≈ 1.0): all demonstrations succeed → loss
      signal is flat, near-zero gradient — but these tasks dominate mini-batch counts.
    - Very hard tasks (teacher pass rate p ≈ 0.0): all demonstrations fail → no
      positive supervision, only noise — the model cannot learn from examples it
      never sees succeed.
    - Intermediate tasks (0.2 < p < 0.8): richest supervision — teacher succeeds
      SOME of the time, providing diverse positive and negative signal.

    SDPO vs GRPO RELATIONSHIP:
    GRPO's group-relative advantage implicitly concentrates gradient on intermediate-
    difficulty tasks: when the within-group reward variance is non-zero (only possible
    when some rollouts pass and some fail), the normalized advantage is large. This is
    mathematically equivalent to weighting tasks by p(1-p). The paper formalizes this:
    GRPO's gradient is proportional to √[p(1-p)] per task (the "sweet spot" weight),
    but SDPO/SFT lacks this implicit weighting and instead weights all tasks equally.

    SC-SDPO SOLUTION:
    Explicitly weight each task's SFT loss by ŵ = √[p̂(1-p̂)], where p̂ is the
    empirical teacher pass rate observed in the training traces:
      L_SC_SDPO = Σ_task ŵ(task) × L_SFT(task)
    - This concentrates SFT gradient on intermediate-difficulty tasks (0.3-0.7 range).
    - Very easy tasks (p̂ ≈ 1.0): ŵ ≈ 0 → their flat demonstrations don't crowd out
      the informative ones.
    - Very hard tasks (p̂ ≈ 0.0): ŵ ≈ 0 → their noisy zero-teacher-signal examples
      are downweighted.

    KEY RESULTS (from paper abstract and search snippets):
    - SC-SDPO improves over standard SDPO by +3.2 pass@16 and +4.3 maj@16 on Qwen3-8B.
    - +1.8 pass@16 and +3.0 maj@16 on OLMo-3-7B.
    - Stable training dynamics throughout (no entropy collapse or loss spike).
    - Published results are on scientific reasoning and tool-use benchmarks (not
      HumanEval), but the √[p(1-p)] weighting principle is benchmark-agnostic.

    RELEVANCE TO OUR PIPELINE:
    Our SFT phase (Phase 3a, train_small_model.py for HumanEval) currently weights all
    82 HumanEval tasks equally. With SCALING_FORCE_BOTH=1 (run-both oracle), we have
    teacher pass rates for every task in traces.jsonl. In our 4-cycle run:
    - Easy tasks (small model already passes): teacher always passes → p̂ ≈ 1 → ŵ ≈ 0.
      These produce near-zero-gradient SFT examples but dominate the ~77-sample dataset.
    - Hard tasks (only large passes): teacher always passes, small always fails →
      p̂(large) = 1.0 but p̂(small) ≈ 0.
    The relevant p̂ for SC-SDPO in our context is the SMALL MODEL pass rate
    (estimated from grpo rollouts or from cycle-0 small model performance), which
    tells us which tasks are in the "sweet spot" for the student.
    Implementation: ~10 lines in traces_to_sft.py — compute per-task small-model
    pass rate from traces, add sample_weight column, pass to Trainer via
    weighted DataLoader or loss multiplier.

    DISTINCTION FROM QUEUE:
    - EXP-121 (ZPD-SFT): FILTERS tasks where 0.2 < p_large < 0.8 (binary mask);
      SC-SDPO uses CONTINUOUS weights √(p(1-p)) — softer, no threshold hyperparameter.
    - EXP-078 (RECRL adaptive curriculum): RE-ORDERS tasks during GRPO training;
      SC-SDPO applies weighted loss during SFT (different phase, different mechanism).
    - EXP-110 (execution-difficulty curriculum): also reorders GRPO; SC-SDPO is SFT.
    These are orthogonal and complementary; SC-SDPO uniquely targets Phase 3a SFT.

arxiv:2605.17106 — "HyDRA: Hybrid Dynamic Routing Architecture for Heterogeneous
    LLM Pools" (Anthropic Infra Group; June 12, 2026):

    BACKGROUND: LIMITATIONS OF LOGISTIC-REGRESSION ROUTER:
    Our current router (train_router_simple.py) fits logistic regression on TF-IDF
    features of the raw HumanEval prompt → 0/1 routing decision (small vs large).
    At 92.68% routing accuracy this is already good, but it has structural limits:
    - TF-IDF is a bag-of-words feature; it misses syntactic structure, algorithmic
      category, and code-specific patterns.
    - Binary decision (single logit) mixes all difficulty axes into one number.
    - Logistic regression cannot model non-linear interactions between difficulty axes.

    HYDRA APPROACH:
    HyDRA uses a ModernBERT encoder (bert-base-sized, ~110M params, fine-tuned on code)
    as the routing backbone, with K=4 independent sigmoid heads predicting task
    capability requirements along orthogonal axes:
      Head 1: coding_complexity (0=simple one-liner, 1=complex algorithm)
      Head 2: algorithmic_category (0=standard library, 1=novel algorithm design)
      Head 3: edge_case_density (0=happy path only, 1=requires extensive edge handling)
      Head 4: multi_step_reasoning (0=direct implementation, 1=requires multi-step logic)
    Each head outputs a continuous score in [0,1] rather than a binary label.
    Routing decision: select cheapest model (small) whose capabilities meet predicted
    requirements, i.e., small-model capability threshold ≥ predicted difficulty.
    Advantage: adding a new model (e.g., a medium-size distilled variant) requires
    only a configuration update — no re-training of the encoder.

    KEY RESULTS:
    - ModernBERT encoder: 86 ms median CPU inference latency in production.
    - Multi-head scoring decouples routing dimensions → interpretable routing decisions.
    - In heterogeneous pool experiments: accuracy matches or exceeds single-logit router
      with better calibration across difficulty axes.

    RELEVANCE TO OUR PIPELINE:
    Design constraint from CLAUDE.md is maintained: router trains on the RAW prompt,
    no procedure prefix. HyDRA replaces TF-IDF+LogisticRegression with
    ModernBERT encoder + linear projection to K=3 sigmoid heads — still operates on
    the raw prompt, just with a richer feature representation.
    Implementation in train_router_simple.py:
    1. Load ModernBERT (microsoft/modernbert-base or answerdotai/ModernBERT-base)
       as the encoder (frozen for speed, or lightly fine-tuned if GPU allows).
    2. Add a linear projection head: ModernBERT CLS embedding → 3-dim sigmoid.
    3. Define router label assignment: for each training prompt, assign difficulty
       labels per head from existing routing decisions in traces.jsonl
       (head 1 = pass_rate_large_vs_small, head 2 = inferred from task category,
       head 3 = from num_test_cases in HumanEval metadata).
    4. Train: BCELoss on 3 heads jointly.
    5. Final routing decision: threshold on max(head_scores) > ROUTER_HYDRA_THRESHOLD
       (default 0.5) → route to large; else small.
    Target: routing accuracy from 92.68% → ≥94%, providing:
    (a) 1.3pp accuracy gain narrows the gap to large (96.34%);
    (b) Multi-head output provides interpretable routing decisions for the AAAI paper's
        analysis section (which tasks are "hard" by which axis?);
    (c) Decoupled heads allow ablations: which difficulty axis drives routing?

    DISTINCTION FROM QUEUE:
    All prior router experiments use logistic regression or simple classifiers on
    prompt features. HyDRA is the ONLY queued experiment using a pre-trained language
    model encoder for routing. The multi-head sigmoid design (K=3 heads) is unique to
    this paper and provides interpretability not present in any prior experiment.
"""

import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = "data/humaneval_eval.jsonl"

NEW_EXPERIMENTS = [
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-128: SC-SDPO — Pass-Rate Weighted SFT Loss for Skills Gap
    #          (arXiv:2605.27765, Liu et al., Penn State, May 27, 2026)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_23_001_sc_sdpo_passrate_weighted_sft_humaneval",
        "priority": 7,
        "gpu": "auto",
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2605.27765 — 'Restoring the Sweet Spot: Pass-Rate Weighted "
            "Self-Distillation for LLM Reasoning' (SC-SDPO; Liu et al., Penn State, "
            "May 27, 2026). Standard SFT weights all tasks equally, crowding the "
            "gradient budget with easy tasks (flat demonstrations, p≈1, near-zero "
            "signal) and hard tasks (no positive examples, p≈0, noisy signal). The "
            "paper proves that GRPO's group-relative advantage implicitly weights tasks "
            "by p(1−p) — the 'sweet spot' formula — while SFT/self-distillation lacks "
            "this. SC-SDPO adds the explicit weight ŵ = √[p̂(1−p̂)] to each task's SFT "
            "loss, where p̂ is the empirical small-model pass rate (from GRPO rollouts "
            "or cycle-0 small model traces). Tasks at p̂∈[0.3,0.7] get maximum weight; "
            "trivially easy (p̂→1) and impossible (p̂→0) tasks are downweighted. Key "
            "results: +3.2/+4.3 pass@16/maj@16 on Qwen3-8B and +1.8/+3.0 on OLMo-3-7B "
            "vs standard SDPO, with stable training dynamics. Direct relevance: our "
            "skills arm is stuck at 75.61% vs large 96.34% (20.7pp gap, Problem C). "
            "Phase 3a SFT uses ~77 examples uniformly weighted. With SCALING_FORCE_BOTH=1, "
            "we have small-model pass rates per task from traces.jsonl; SC-SDPO requires "
            "only a ~10-line change to traces_to_sft.py. Distinct from EXP-121 (ZPD-SFT "
            "uses binary pass-rate threshold to FILTER tasks; SC-SDPO uses CONTINUOUS "
            "√(p(1-p)) WEIGHTS — no hyperparameter threshold, softer gradient weighting). "
            "Distinct from all GRPO modifications (EXP-120 to EXP-127): SC-SDPO targets "
            "Phase 3a SFT, not Phase 3b GRPO — orthogonal interventions. Expected outcome: "
            "skills arm 75.61% → ≥78%; improvement propagates via SFT → better small-model "
            "checkpoint → better GRPO starting point."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 4,
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "sc_sdpo_enabled": True,
            "sc_sdpo_weight_formula": "sqrt_p_1mp",
            "sc_sdpo_pass_rate_source": "small_model_rollouts",
            "sc_sdpo_min_weight": 0.05,
            "grpo_temperature": 1.0,
            "n_generations": 8,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "analysis": [
                "per_task_sft_weight_distribution",
                "sft_loss_curve_weighted_vs_uniform",
                "skills_arm_pass_at_1_per_cycle_sc_sdpo_vs_uniform",
                "full_arm_pass_at_1_per_cycle_sc_sdpo_vs_uniform",
                "small_model_pass_rate_per_task_before_after_sft",
                "effective_sample_count_weighted_vs_uniform",
            ],
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (baseline): skills 75.61%, full 92.68%",
                "EXP-121 (ZPD-SFT): binary pass-rate filter vs SC-SDPO continuous weights",
                "arxiv:2605.27765 (SC-SDPO): +3.2/+4.3 pass@16/maj@16 on Qwen3-8B",
            ],
            "implementation_files": [
                "src/pipeline/traces_to_sft.py",
                "src/pipeline/train_small_model.py",
            ],
            "implementation_note": (
                "In traces_to_sft.py, after loading traces.jsonl: "
                "1. For each unique task_id, compute p̂ = fraction of small-model "
                "   rollouts that passed (reward=1) across all traces for that task. "
                "   If no small-model rollouts exist for a task (SCALING_FORCE_BOTH "
                "   not set), fall back to p̂ = 0.5 (neutral weight). "
                "2. Compute ŵ(task) = max(SC_SDPO_MIN_WEIGHT, sqrt(p̂ * (1 - p̂))). "
                "   SC_SDPO_MIN_WEIGHT=0.05 ensures very easy/hard tasks retain a small "
                "   non-zero weight (prevents complete starvation of extreme tasks). "
                "3. Add a 'sample_weight' field to each SFT example's JSON, equal to "
                "   ŵ(task_id). "
                "In train_small_model.py, when constructing the Trainer: "
                "4. Read sample_weight from each training example. "
                "5. In the custom compute_loss callback (or via weighted DataLoader), "
                "   multiply each token's cross-entropy loss by the sample_weight for "
                "   that example before reduction. Normalise weights so they sum to "
                "   len(train_examples) (preserve total gradient scale). "
                "Add env vars: SC_SDPO_ENABLED (default 0), "
                "SC_SDPO_MIN_WEIGHT (default 0.05), "
                "SC_SDPO_PASS_RATE_SOURCE ('small_model_rollouts' or 'uniform'). "
                "Total: ~10 lines across two files. No extra forward passes or memory "
                "overhead — weights are scalars computed from existing trace statistics. "
                "Implementation reference: https://arxiv.org/abs/2605.27765."
            ),
            "arxiv_ref": "2605.27765",
            "estimated_gpu_hours": 3.0,
        },
    },
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-129: HyDRA Multi-Dimensional Router
    #          (arXiv:2605.17106, Anthropic Infra Group, June 12, 2026)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_23_002_hydra_multidim_modernbert_router_humaneval",
        "priority": 6,
        "gpu": "auto",
        "kind": "joint_cycle_multiseed",
        "rationale": (
            "arxiv:2605.17106 — 'HyDRA: Hybrid Dynamic Routing Architecture for "
            "Heterogeneous LLM Pools' (June 12, 2026). Our logistic regression router "
            "maps TF-IDF features of the raw HumanEval prompt to a single binary logit "
            "(92.68% accuracy, cycle 3). HyDRA proposes replacing the shallow classifier "
            "with a ModernBERT encoder + K independent sigmoid heads predicting "
            "capability requirements along orthogonal difficulty axes. Routing then "
            "selects the cheapest model whose capability profile meets predicted "
            "requirements (shortfall matching). Key advantages over single-logit: "
            "(1) ModernBERT's contextual embeddings capture syntactic structure, "
            "algorithmic patterns, and code idioms that TF-IDF misses; (2) multi-head "
            "output decomposes 'route to large' into interpretable sub-decisions "
            "(complexity axis, algorithmic-novelty axis, edge-case axis); "
            "(3) HyDRA runs at 86 ms CPU median latency in production. In our pipeline, "
            "we implement K=3 sigmoid heads: (1) code_complexity (one-liner vs. "
            "non-trivial algorithm), (2) algo_novelty (stdlib suffices vs. novel design "
            "required), (3) edge_case_density (straightforward vs. many corner cases). "
            "Labels derived from existing traces: head 1 from pass rate ratio "
            "(small_pass_rate / large_pass_rate), heads 2-3 from HumanEval metadata "
            "(num_test_cases, problem category tags). Final routing: threshold on "
            "max(sigmoid_scores) > 0.5 → route to large. Design constraints from "
            "CLAUDE.md are respected: router trains on raw prompt only (ModernBERT "
            "encodes the raw problem, no procedure prefix), and router owns routing "
            "(SkillBook verdict is diagnostic only). Expected gain: routing accuracy "
            "92.68% → ≥94.0%, narrowing the 3.66pp gap to large (96.34%), while "
            "providing per-head interpretability for the AAAI paper's routing analysis. "
            "Distinct from all queued router experiments: HyDRA is the ONLY experiment "
            "using a pre-trained language model encoder for routing, and the only one "
            "with multi-head orthogonal difficulty decomposition."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 4,
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "router_arch": "hydra_modernbert",
            "router_encoder": "answerdotai/ModernBERT-base",
            "router_encoder_frozen": True,
            "router_n_heads": 3,
            "router_head_names": [
                "code_complexity",
                "algo_novelty",
                "edge_case_density",
            ],
            "router_head_label_sources": [
                "small_vs_large_pass_rate_ratio",
                "humaneval_category_tags",
                "num_test_cases_normalized",
            ],
            "router_hydra_threshold": 0.5,
            "router_seeds": [42, 7, 13],
            "grpo_temperature": 1.0,
            "n_generations": 8,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "analysis": [
                "routing_accuracy_per_cycle_hydra_vs_logistic",
                "per_head_sigmoid_score_distribution",
                "per_head_routing_contribution",
                "interpretable_routing_decisions_hard_tasks",
                "cost_vs_large_hydra_vs_logistic",
                "routing_accuracy_by_difficulty_axis",
            ],
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (baseline): router 92.68%, cost 27.56%",
                "EXP-000..EXP-127: all use logistic regression on TF-IDF features",
                "arxiv:2605.17106 (HyDRA): ModernBERT + K=4 sigmoid heads, 86ms CPU",
            ],
            "implementation_files": [
                "src/pipeline/train_router_simple.py",
            ],
            "implementation_note": (
                "In train_router_simple.py, add a HyDRA routing class alongside the "
                "existing LogisticRegression router (controlled by ROUTER_ARCH env var): "
                "1. Load answerdotai/ModernBERT-base (HuggingFace, ~110M params). "
                "   Run in frozen mode (requires_grad=False) to avoid GPU memory pressure. "
                "   Tokenize each raw HumanEval prompt (max_length=512 tokens, truncate). "
                "2. Extract CLS embedding (hidden_dim=768) for each prompt. "
                "3. Define K=3 linear heads: head_k = nn.Linear(768, 1) for k in 1..3. "
                "   Apply sigmoid to get head_k_score ∈ [0,1] per task. "
                "4. Assign training labels per head from traces: "
                "   head_1 = 1 if small_pass_rate < 0.5 * large_pass_rate, else 0. "
                "   head_2 = inferred from HumanEval task difficulty label (if available). "
                "   head_3 = 1 if num_test_cases > median_num_test_cases, else 0. "
                "5. Train heads jointly with BCEWithLogitsLoss; 3 seeds; pick best seed "
                "   by routing accuracy on held-out 20% of tasks. "
                "6. At inference: route to large if max(head_scores) > ROUTER_HYDRA_THRESHOLD, "
                "   else route to small. "
                "7. Log per-head scores and routing decision for each task → interpretability "
                "   table for paper. "
                "Add env vars: ROUTER_ARCH ('logistic' default, 'hydra_modernbert'), "
                "ROUTER_ENCODER ('answerdotai/ModernBERT-base'), "
                "ROUTER_ENCODER_FROZEN (default 1), "
                "ROUTER_N_HEADS (default 3), "
                "ROUTER_HYDRA_THRESHOLD (default 0.5). "
                "No extra GPU for inference — CLS embedding is a single forward pass per "
                "task at routing time (~86ms per prompt on CPU). "
                "Implementation reference: https://arxiv.org/abs/2605.17106."
            ),
            "arxiv_ref": "2605.17106",
            "estimated_gpu_hours": 2.5,
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
