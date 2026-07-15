#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-15 (EXP-118, EXP-119).

A800 connectivity: offline since 2026-05-14 (day ~62). Apply when restored.

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

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_15.py            # EXP-118, EXP-119

Queue was ~125 pending on 2026-07-14 (+2 added: EXP-116, EXP-117).
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

AAAI 2027 deadline: 2026-08-15 (31 days from today).
A800 offline since 2026-05-14 (day 62).

arxiv:2606.04889 — GRAIL: Gradient-Reweighted Advantages for Reinforcement Learning
    with Verifiable Rewards (June 2026; Nanyang Technological University DeCLaRe Lab):

    PROBLEM THEY SOLVE:
    Current GRPO and most RLVR methods assign a single sequence-level advantage A to
    all tokens in a rollout (uniform advantage broadcast), or — for step-level methods —
    require expensive process reward models (PRMs). Uniform broadcast assumes all tokens
    contribute equally to the final reward, diluting the gradient signal: tokens encoding
    trivial filler ("print(", whitespace, closing brackets) are updated as strongly as
    tokens encoding the key algorithmic decision (the data structure choice, the
    boundary condition). This leads to two problems:
      1. Gradient dilution: much of the gradient budget is spent on uninformative tokens.
      2. Possible counter-productive reinforcement of filler tokens in correct rollouts.
    Both degrade sample efficiency and can destabilize training.

    GRAIL MECHANISM (gradient-activation saliency reweighting):
    For each token t_i in rollout r, GRAIL computes a gradient-activation saliency score:
      s_i = || grad_{h_i} L_GRPO || × || h_i ||
    where h_i is the hidden state (last transformer layer activation) at position i, and
    grad_{h_i} L_GRPO is the gradient of the (scalar) GRPO loss with respect to h_i.
    This saliency score measures how much a small perturbation to the representation at
    position i changes the overall training loss — i.e., how "locally sensitive" the loss
    is to the model's behavior at token i.

    The advantage is then reweighted:
      A_i = A × (1 + γ · normalize(s_i))
    where normalize(s) = (s − mean(s)) / (std(s) + ε) across all tokens in the rollout,
    and γ is a hyperparameter (γ=0.5 in the paper's best configuration). Tokens with
    above-average saliency receive amplified advantage; tokens with below-average
    saliency receive reduced advantage.

    KEY IMPLEMENTATION NOTE:
    Computing grad_{h_i} L_GRPO requires one backward pass w.r.t. the hidden states
    (not the model parameters — so no parameter update occurs from this backward pass).
    In PyTorch, this can be done by:
      (a) Registering h_i = last_hidden_state as a leaf requiring gradients before the
          forward pass (h_i.requires_grad_(True) and h_i.retain_grad()).
      (b) Running the forward pass to get the GRPO loss L.
      (c) Running L.backward(retain_graph=True) to get grad_{h_i}.
      (d) Computing s_i = ||grad_{h_i}|| × ||h_i|| for each position.
      (e) Reweighting A → A_i and recomputing the final policy loss with A_i.
    This adds one extra backward pass per rollout batch (not per rollout).
    On A800 80GB with Qwen2.5-Coder-1.5B: +~50% compute vs. standard GRPO.

    COMPARISON WITH TACO (EXP-116):
    TACO (arxiv:2607.07976) uses token probability as the weighting signal:
      τ_i = softmax(−log p_θ(t_i))     ← tail-risk: low p → high τ → suppressed credit
      A_i = A × (1 − α × τ_i)         ← suppression, not amplification
    TACO suppresses credit for low-probability tokens in WINNING rollouts. O(0) overhead.

    GRAIL uses gradient-activation saliency as the weighting signal:
      s_i = ||grad_{h_i}|| × ||h_i||  ← saliency: high grad-activation → amplified credit
      A_i = A × (1 + γ × normalize(s_i))  ← amplification, not suppression
    GRAIL amplifies credit for HIGH-saliency tokens in ALL rollouts. +50% compute.

    The two mechanisms are COMPLEMENTARY and COMPOSABLE:
      TACO-GRAIL: A_i = A × (1 − α × τ_i) × (1 + γ × normalize(s_i))
    This doubly-targets credit quality: suppressing low-probability noise (TACO) while
    amplifying high-saliency signal (GRAIL). If EXP-116 (TACO) and EXP-118 (GRAIL)
    both show positive results, EXP-120 (TACO-GRAIL composition) is the natural ceiling.

    DISTINCTION FROM PRIOR EXPERIMENTS:
    EXP-116 (TACO): suppresses positive credit for low-probability tokens in winning
      rollouts. O(0) overhead. Mechanism: token probability.
    EXP-108 (sign advantage): changes advantage formula for all-fail groups.
      Does not reweight per-token credit within groups.
    EXP-094 (AVSPO): injects virtual samples for collapsed groups.
      Does not touch non-collapsed group gradient quality.
    EXP-114 (RLCSD): contrastive self-distillation; dense signal independent of rollout
      pass/fail. No gradient-saliency weighting.
    EXP-118 (this): per-token gradient-activation saliency reweighting across ALL rollouts
      (not just winning, not just collapsed groups). Requires one extra backward pass.

    RESULTS FROM PAPER:
    Evaluated on Qwen3 (1.7B, 4B, 8B), R1-distilled, and OctoThinker families.
    GRAIL achieves +3.60% average accuracy improvement and +3.05% Pass@3 vs. vanilla GRPO.
    Outperforms uniform advantage, token-entropy reweighting, and step-level PRMs.
    Results on math-heavy benchmarks (AIME, AMC, MATH500, OlympiadBench).
    Code generation specifically: +1.8–2.2pp on HumanEval, MBPP (extrapolated from math gains).

    RELEVANCE TO OUR NON-COLLAPSED GROUPS (47.6% = 39/82):
    For a 1.5B Coder model at HumanEval: the non-collapsed groups contain ~40 tasks × 8 rollouts
    = ~320 non-trivially-advantaged rollouts per cycle. Within each rollout, there are typically
    200–500 tokens. Many of these tokens are filler (indentation, closing brackets, comments)
    that carry no algorithmic signal. Under uniform advantage broadcast, all 320 × ~350 = 112,000
    non-collapsed-group tokens receive equal gradient weight per step. Under GRAIL, ~top 30%
    (roughly the genuinely decision-bearing tokens) receive amplified gradient (×1.5 average).
    The expected effect is comparable to having 50% more effective training examples from the
    non-collapsed groups — which is the primary lever for breaking Full=Router.

    PREDICTED OUTCOME:
    Full arm pass@1: 92.68% → 94–95% (+1.3–2.3pp), breaking Full=Router.
    Skills arm pass@1: 75.61% → 77–79% (improved gradient quality per GRPO step).
    ACR stays at ~52.4% (GRAIL does not affect collapsed groups).
    Training loss variance decreases, fewer spikes (per paper results pattern).
    Estimated run time: ~3.5h (n_cycles=1; +50% overhead vs. 2.5h TACO baseline).

arxiv:2607.01763 — Denser ≠ Better: Limits of On-Policy Self-Distillation for Continual
    Post-Training (July 1, 2026; Meng Wang et al., NLPR/MAIS Institute):

    PROBLEM THEY DOCUMENT:
    On-policy self-distillation (SDPO) methods — which add a dense KL-matching term between
    the model's on-policy distribution and a teacher distribution to GRPO's reward signal —
    are widely used to increase training signal density. The intuition: denser per-token
    supervision should improve gradient quality and reduce training instability.

    KEY FINDING:
    This paper directly refutes that intuition in the continual post-training setting:
      1. In continual multi-task SDPO training, the dense self-distillation term causes
         STRONGER FORGETTING of prior-task capabilities than vanilla GRPO.
      2. SDPO can AMPLIFY HIGH-FREQUENCY FORMATTING ARTIFACTS through a self-reinforcing
         teacher-student loop: the model's own output distribution (on-policy) becomes the
         teacher for the next batch; formatting biases in earlier rounds get reinforced
         and amplified in later rounds.
      3. SDPO induces larger drift in BOTH parameter space (||Δθ||) and response space
         (KL divergence from the pre-training distribution) than vanilla GRPO.
      4. Critically: GRPO adapts more CONSERVATIVELY than SDPO and better preserves
         prior capabilities — a reversal of the commonly-assumed benefit of denser supervision.

    RELEVANCE TO OUR PIPELINE AND QUEUED EXPERIMENTS:
    We have two queued experiments that add dense self-distillation terms to GRPO:
      EXP-112 (forward-KL teacher distillation): adds KL(M_θ || GPT-5.5 teacher)
        as a dense term to GRPO. Motivation was arxiv:2507.15234. Single-cycle.
      EXP-114 (RLCSD): adds contrastive KL (correct_hint - wrong_hint) as a dense
        term to GRPO. Motivation was arxiv:2606.11709. Single-cycle.
    Both EXP-112 and EXP-114 are single-cycle experiments designed to address ACR=52.4%.
    But our DEPLOYMENT setup is multi-cycle (4 cycles). If EXP-112/EXP-114 are adopted
    in the full 4-cycle pipeline, the "Denser ≠ Better" finding predicts they would
    INCREASE forgetting rate across cycles and potentially amplify formatting artifacts.

    DIAGNOSTIC QUESTION:
    Does adding RLCSD's contrastive self-distillation term to our GRPO training in a
    1-cycle run already exhibit the early signatures of parameter-space drift that would
    cause forgetting in a multi-cycle setting? Specifically:
      (a) Does EXP-114's RLCSD checkpoint exhibit higher parameter drift
          (||Δθ_RLCSD|| vs ||Δθ_vanilla_GRPO||) than the EXP-116 TACO checkpoint?
      (b) Does EXP-114's RLCSD checkpoint show lower pass@1 on tasks that were
          ALREADY PASSING before cycle-1 training (early-forgetting signal)?

    IMPLEMENTATION (eval-only, depends_on EXP-114 and EXP-116):
    This is a post-hoc analysis that runs AFTER EXP-114 (RLCSD) and EXP-116 (TACO, or
    vanilla GRPO baseline) have produced checkpoints. It does NOT require new training.
    Steps:
      1. Load: (a) vanilla GRPO cycle-1 checkpoint (from e2e_4cyc_gpt55/cycle_1/),
                   (b) EXP-114 RLCSD checkpoint (after EXP-114 runs),
                   (c) base Qwen2.5-Coder-1.5B-Instruct (pre-training reference).
      2. Compute parameter drift: ||θ_cycle1_vanilla - θ_base|| vs ||θ_RLCSD - θ_base||
         (L2 norm of LoRA weight deltas relative to base model).
      3. Evaluate both checkpoints on: (a) ALL 82 HumanEval tasks; (b) ONLY the tasks
         that the vanilla GRPO cycle-0 model already solved (easy tasks, forgetting test).
      4. Compute early-forgetting signal: how many tasks that vanilla cycle-0 solved
         does each cycle-1 checkpoint LOSE relative to base GRPO?
      5. Output: parameter_drift_ratio (RLCSD/vanilla), early_forgetting_rate_comparison.
      6. Decision rule:
         If parameter_drift_ratio > 1.5 AND early_forgetting_rate(RLCSD) > early_forgetting_rate(vanilla)+2%:
           → "Denser ≠ Better confirmed for our pipeline" → de-prioritize EXP-112/EXP-114
              in multi-cycle setting; recommend TACO/GRAIL instead.
         Else: → "Denser ≠ Better does not generalize to our single-task (HumanEval) setting"
              → EXP-112/EXP-114 safe to include in multi-cycle ablation for AAAI paper.

    DISTINCT FROM EXP-115 (forgetting diagnostic):
    EXP-115: measures within-pipeline forgetting for the DEFAULT vanilla GRPO run across
      4 cycles. Asks: "how much forgetting does our baseline pipeline cause?"
    EXP-119 (this): compares RLCSD vs. vanilla GRPO forgetting after 1 cycle.
      Asks: "does adding dense distillation INCREASE forgetting vs. baseline?"
    EXP-115 characterizes the baseline; EXP-119 tests whether the distillation-augmented
    experiments are safe to promote to multi-cycle use.

    PRACTICAL IMPORTANCE FOR AAAI PAPER:
    If EXP-119 finds that RLCSD increases early forgetting by >2%, the paper must
    include a §5.x caveat: "Dense contrastive self-distillation improves single-cycle
    performance (EXP-114: +Xpp) but increases parameter drift in multi-cycle settings
    (consistent with arxiv:2607.01763). We recommend TACO/GRAIL over RLCSD for the
    full 4-cycle ablation." This is an actionable finding that strengthens the paper.
    Estimated run time: 0.5h (eval-only; depends on EXP-114 and vanilla cycle-1 already
    available from e2e_4cyc_gpt55/cycle_1/).

    PREDICTED OUTCOME:
    CASE A (drift ratio ≤ 1.5, forgetting Δ ≤ 2%): "Denser ≠ Better" does not generalize
      to single-task HumanEval code generation. The finding is specific to multi-task
      continual settings. EXP-112/EXP-114 are safe for multi-cycle use. Paper strengthens:
      "RLCSD adds dense signal safely in our single-task setting."
    CASE B (drift ratio > 1.5, forgetting Δ > 2%): "Denser ≠ Better" generalizes even in
      single-task code generation with iterative traces. De-prioritize EXP-112/EXP-114 for
      multi-cycle. Focus AAAI contribution on TACO (EXP-116) and GRAIL (EXP-118) instead.
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
_EXP114_DIR = "/data0/home/zeyuwang/auto_research/exp_results/exp_2026_07_13_001_rlcsd"

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_07_15_001_grail_gradient_saliency_advantage_reweighting_grpo_humaneval",
        "priority": 8,
        "kind": "grpo_continual",
        "gpu": "auto",
        "rationale": (
            "arxiv:2606.04889 — GRAIL: Gradient-Reweighted Advantages for Reinforcement "
            "Learning with Verifiable Rewards (June 2026; DeCLaRe Lab, NTU). "
            ""
            "THE CORE PROBLEM THIS SOLVES: "
            "Uniform advantage broadcast in GRPO assigns the same sequence-level advantage A "
            "to all tokens — including trivial filler (indentation, closing brackets) and "
            "genuinely decision-bearing tokens (data structure choice, boundary condition). "
            "This dilutes the gradient signal: most gradient budget is spent on tokens with "
            "near-zero information content for the outcome. "
            ""
            "GRAIL MECHANISM (gradient-activation saliency): "
            "For each token t_i, compute saliency s_i = ||grad_{h_i} L|| × ||h_i||, where "
            "h_i = last transformer hidden state at position i, and grad_{h_i} L = gradient "
            "of the GRPO loss w.r.t. h_i (one extra backward pass w.r.t. activations, not "
            "parameters — no parameter update occurs from this pass). "
            "Reweight advantage: A_i = A × (1 + γ × normalize(s_i)), γ=0.5. "
            "Tokens with above-average saliency (decision-bearing) receive amplified credit; "
            "tokens with below-average saliency (filler) receive reduced credit. "
            ""
            "COMPARISON WITH TACO (EXP-116): "
            "TACO suppresses credit for LOW-PROBABILITY tokens in WINNING rollouts only: "
            "  A_i = A × (1 − α × τ_i), τ_i = softmax(−log p_θ(t_i)). O(0) overhead. "
            "GRAIL amplifies credit for HIGH-SALIENCY tokens across ALL rollouts: "
            "  A_i = A × (1 + γ × normalize(s_i)), s_i = ||grad×activation||. +50% compute. "
            "Mechanisms are COMPLEMENTARY (different signals: probability vs gradient): "
            "  TACO removes noise; GRAIL boosts signal. Composable as EXP-120. "
            ""
            "RELEVANCE TO NON-COLLAPSED GROUPS (47.6% = 39/82): "
            "Our 39 non-collapsed GRPO groups contain ~320 non-trivially-advantaged rollouts "
            "per cycle, each ~350 tokens = ~112,000 tokens receiving gradient per cycle step. "
            "Under GRAIL: top ~30% saliency tokens (the algorithmic decision tokens) receive "
            "~1.5× average advantage. Effect ≈ 50% more gradient weight on the signal-bearing "
            "tokens — the primary lever for breaking Full=Router without increasing ACR. "
            ""
            "DISTINCT FROM ALL EXISTING EXPERIMENTS: "
            "EXP-116 (TACO): token probability → suppresses noise. O(0) overhead. "
            "  GRAIL uses gradient-activation → amplifies signal. Different mechanism. "
            "EXP-108 (sign advantage): advantage formula for ALL-FAIL groups. "
            "  GRAIL reweights within-group per-token, regardless of group type. "
            "EXP-094 (AVSPO): virtual samples for collapsed groups only. "
            "  GRAIL improves per-token gradient quality in non-collapsed groups. "
            "EXP-114 (RLCSD): contrastive KL distillation, style-drift-free. "
            "  GRAIL adds no distillation term; pure saliency reweighting. "
            "EXP-118 (this): unique mechanism — gradient-activation saliency per token, "
            "  orthogonal to all above. First gradient-based per-token weighting in queue. "
            ""
            "PAPER RESULTS: "
            "+3.60% average accuracy, +3.05% Pass@3 over vanilla GRPO across Qwen3 1.7B/4B/8B, "
            "R1-distilled, and OctoThinker on math-reasoning benchmarks. Outperforms: "
            "  uniform advantage, token-entropy reweighting, step-level PRMs. "
            "Extrapolated to code (HumanEval/MBPP): +1.8–2.2pp based on math/code transfer "
            "ratio observed in TACO paper (+0.8pp code per ~1.5pp math gain). "
            ""
            "PREDICTED OUTCOME: "
            "  Full arm pass@1: 92.68% → 94–95% (+1.3–2.3pp), breaking Full=Router. "
            "  Skills arm pass@1: 75.61% → 77–79% (improved GRPO gradient quality). "
            "  ACR stays at ~52.4% (GRAIL does not affect collapsed groups). "
            "  Training loss variance decreases (per paper pattern). "
            "  Est. run time: ~3.5h (n_cycles=1; +50% overhead from extra backward pass). "
            ""
            "EXP-120 NATURAL FOLLOW-UP: "
            "If both EXP-116 (TACO, +1pp est.) and EXP-118 (GRAIL, +1.5pp est.) show positive "
            "results, compose them: A_i = A × (1 − α × τ_i) × (1 + γ × normalize(s_i)). "
            "TACO removes noise; GRAIL boosts signal. Non-redundant: they use different token "
            "features (log-prob vs gradient-activation) with no mathematical overlap. "
            "EXP-120 (TACO-GRAIL) is deferred until individual results confirm value. "
        ),
        "spec": {
            "pipeline": "humaneval_grail_gradient_saliency_advantage_reweighting",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 1,
            "grpo_algo": "grail_dapo",
            "grail_gamma": 0.5,
            "grail_saliency_fn": "grad_activation_product",
            "grail_saliency_layer": "last_hidden_state",
            "grail_normalize_fn": "z_score_per_rollout",
            "grail_apply_to": "all_rollouts",
            "n_generations": 8,
            "grpo_temperature": 1.0,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (DAPO baseline): full 92.68%, skills 75.61%",
                "EXP-116 (TACO): token-probability suppression → O(0) overhead",
                "arxiv:2606.04889 (GRAIL): +3.60% avg acc on Qwen3 1.7B/4B/8B math",
                "EXP-108 (sign advantage): all-fail group formula change",
                "EXP-094 (AVSPO): collapsed group virtual samples",
            ],
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
            ],
            "implementation_note": (
                "In grpo_train_simple.py, after the standard forward pass to get log_probs "
                "and the scalar GRPO loss L, before the optimizer step: "
                "(1) Retain last_hidden_state from the forward pass (register hook or "
                "    set retain_grad() on last_hidden_state before forward pass). "
                "(2) L.backward(retain_graph=True) — computes grad w.r.t. hidden states "
                "    only, NOT w.r.t. parameters (zero_grad() before this call). "
                "(3) For each position i: s_i = norm(grad_h[i]) * norm(h[i]) (elementwise "
                "    L2 norms over the hidden dimension D). "
                "(4) Normalize: s_i = (s_i - mean(s)) / (std(s) + 1e-8) per rollout. "
                "(5) Reweight: A_i = A * (1 + 0.5 * s_i) for each token position. "
                "(6) Recompute policy loss with A_i: L_grail = -sum(A_i * log_prob_i). "
                "(7) L_grail.backward() for the actual parameter update. "
                "Memory overhead: retain hidden states for one batch (~200-500 MB on 1.5B). "
                "~30 lines added to the GRPO loss computation block. "
                "Extra compute: ~50% per step (one extra backward pass w.r.t. activations). "
            ),
            "arxiv_ref": "2606.04889",
            "estimated_gpu_hours": 3.5,
        },
    },
    {
        "id": "exp_2026_07_15_002_denser_not_better_rlcsd_vs_grpo_parameter_drift_forgetting",
        "priority": 6,
        "kind": "forgetting_eval",
        "gpu": "auto",
        "rationale": (
            "arxiv:2607.01763 — Denser ≠ Better: Limits of On-Policy Self-Distillation "
            "for Continual Post-Training (July 1, 2026; Meng Wang et al., NLPR/MAIS). "
            ""
            "KEY FINDING FROM THE PAPER: "
            "On-policy self-distillation (SDPO, which adds a dense KL-matching term to GRPO) "
            "causes STRONGER forgetting than vanilla GRPO in continual post-training: "
            "  (a) Larger parameter drift (||Δθ_SDPO|| > ||Δθ_GRPO||) "
            "  (b) Larger response-space drift (KL from pre-training distribution) "
            "  (c) Self-reinforcing formatting artifacts (teacher-student loop amplification) "
            "  (d) Worse preservation of prior-task capabilities after each new-task round. "
            "GRPO adapts MORE CONSERVATIVELY and better preserves prior capabilities. "
            ""
            "RELEVANCE TO QUEUED EXPERIMENTS EXP-112 AND EXP-114: "
            "Two queued experiments add dense distillation terms to our GRPO training: "
            "  EXP-112: L = L_GRPO + λ_fwd × KL(π_θ || π_teacher_gpt55) [forward-KL] "
            "  EXP-114: L = L_GRPO + λ × (KL_correct − KL_wrong) [RLCSD contrastive] "
            "Both add DENSER supervision — exactly the pattern arxiv:2607.01763 warns about. "
            "If this finding generalizes to our setting (single-task code generation, "
            "iterative trace collection, LoRA adapters), then adopting EXP-112/EXP-114 "
            "in our 4-cycle multi-cycle pipeline could increase cross-cycle forgetting "
            "and degrade the skills arm in later cycles. "
            ""
            "THIS DIAGNOSTIC (eval-only, ~0.5h): "
            "Compare EXP-114's RLCSD checkpoint vs. vanilla GRPO cycle-1 checkpoint on: "
            "(a) Parameter drift: ||ΔLoRA_RLCSD|| / ||ΔLoRA_vanilla_GRPO|| "
            "    (ratio of LoRA weight delta L2 norms from base model) "
            "(b) Response drift: KL(π_base || π_RLCSD) vs KL(π_base || π_vanilla_GRPO) "
            "    computed on a sample of HumanEval prompts not seen during training "
            "(c) Early-forgetting rate: pass@1 on easy tasks (solved by vanilla cycle-0) "
            "    under each checkpoint — does RLCSD's cycle-1 lose previously-passing tasks? "
            ""
            "DECISION RULE: "
            "If drift_ratio > 1.5 AND early_forgetting(RLCSD) > early_forgetting(vanilla)+2%: "
            "  → arxiv:2607.01763 generalizes to our single-task code generation setting. "
            "  → De-prioritize EXP-112 and EXP-114 for multi-cycle adoption. "
            "  → AAAI paper focuses on TACO (EXP-116) and GRAIL (EXP-118): gradient-quality "
            "    improvements that do NOT add dense distillation terms. "
            "  → Add §5.x: 'Dense distillation increases parameter drift in our continual "
            "    training; we find TACO/GRAIL provide gains without this downside.' "
            "Else: "
            "  → arxiv:2607.01763 is specific to multi-task settings; single-task code "
            "    generation with stable teacher traces does not exhibit the amplification loop. "
            "  → EXP-112 and EXP-114 are safe to include in multi-cycle ablation. "
            "  → AAAI paper: report RLCSD and forward-KL results without forgetting caveat. "
            ""
            "DISTINCT FROM EXP-115 (forgetting diagnostic): "
            "EXP-115: measures vanilla GRPO forgetting across 4 cycles. Asks how much "
            "  forgetting the BASELINE causes. "
            "EXP-119 (this): compares RLCSD vs. vanilla GRPO forgetting after 1 cycle. "
            "  Asks whether ADDING dense distillation INCREASES forgetting beyond baseline. "
            ""
            "DEPENDENCY: "
            "Requires EXP-114 (RLCSD, exp_2026_07_13_001) to have run and produced a "
            "checkpoint. Also uses e2e_4cyc_gpt55/cycle_1/grpo_adapter as the vanilla baseline "
            "(already available from prior runs). If EXP-114 has not run, skip this experiment "
            "and run EXP-114 first. "
            ""
            "PREDICTED OUTCOME: "
            "CASE A (drift_ratio ≤ 1.5, forgetting Δ ≤ 2%): "
            "  'Denser ≠ Better' is multi-task specific. EXP-112/EXP-114 safe for multi-cycle. "
            "  Strengthens AAAI paper: 'RLCSD adds dense signal without forgetting overhead.' "
            "CASE B (drift_ratio > 1.5, forgetting Δ > 2%): "
            "  Dense distillation problematic even in single-task code. "
            "  Focus AAAI contribution on gradient-reweighting (TACO, GRAIL) over distillation. "
            "  Novel finding: distillation amplifies drift in iterative code RL distillation. "
        ),
        "spec": {
            "pipeline": "humaneval_denser_not_better_rlcsd_drift_comparison",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "eval_mode": "parameter_drift_and_forgetting_comparison",
            "depends_on": [
                "exp_2026_07_13_001_rlcsd_contrastive_self_distill_grpo_humaneval",
            ],
            "checkpoints_to_compare": [
                {
                    "label": "vanilla_grpo_cycle1",
                    "path": f"{_CYCLE_DIR}/cycle_1/grpo_adapter",
                    "fallback": f"{_CYCLE_DIR}/cycle_1/llm_adapter/checkpoint-best",
                },
                {
                    "label": "rlcsd_cycle1",
                    "path": f"{_EXP114_DIR}/cycle_1/grpo_adapter",
                    "fallback": f"{_EXP114_DIR}/cycle_1/llm_adapter/checkpoint-best",
                },
            ],
            "base_model_path": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "eval_data": _HE_EVAL,
            "analysis": [
                "lora_weight_delta_l2_norm_ratio",
                "response_drift_kl_from_base",
                "per_task_pass_fail_matrix",
                "early_forgetting_rate_on_cycle0_passing_tasks",
                "early_forgetting_rate_comparison_rlcsd_vs_vanilla",
            ],
            "decision_thresholds": {
                "drift_ratio_alarm": 1.5,
                "forgetting_delta_alarm_pct": 2.0,
            },
            "compare_with": [
                "arxiv:2607.01763 (Denser ≠ Better): SDPO causes stronger forgetting than GRPO",
                "EXP-115 (forgetting diagnostic): 4-cycle vanilla GRPO forgetting rate",
                "EXP-114 (RLCSD): dense contrastive distillation in GRPO",
                "EXP-112 (forward-KL): teacher distillation term in GRPO",
                "EXP-116 (TACO): gradient-noise suppression, no dense distillation",
                "EXP-118 (GRAIL): gradient-saliency amplification, no dense distillation",
            ],
            "implementation_files": [
                "src/pipeline/forgetting_eval.py",
            ],
            "implementation_note": (
                "forgetting_eval kind with parameter_drift_and_forgetting_comparison mode. "
                "For each checkpoint pair (vanilla_grpo_cycle1, rlcsd_cycle1): "
                "(1) Load LoRA adapter weights. Compute L2 norm of (LoRA_A - base_LoRA_A) "
                "    and (LoRA_B - base_LoRA_B) for all LoRA modules. Sum across modules "
                "    → ||ΔLoRA||. Compute drift_ratio = ||ΔLoRA_RLCSD|| / ||ΔLoRA_vanilla||. "
                "(2) Serve both via vLLM. Run pass@1 greedy on all 82 HumanEval tasks. "
                "(3) Load cycle-0 pass/fail results (from e2e_4cyc_gpt55/cycle_0/). "
                "    Identify 'easy tasks' = those solved (pass=True) in cycle-0 checkpoint. "
                "(4) For each checkpoint: early_forgetting = "
                "    count(tasks solved in cycle_0 AND failed in this checkpoint) / total_easy. "
                "    Compare: RLCSD early forgetting vs. vanilla GRPO early forgetting. "
                "Total runtime: ~0.5h (two checkpoint loads + eval on 82 tasks × 2). "
                "No training required. Produces: drift_analysis.json, forgetting_comparison.json. "
            ),
            "arxiv_ref": "2607.01763",
            "estimated_gpu_hours": 0.5,
            "skip_if": "exp_2026_07_13_001 not completed",
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
