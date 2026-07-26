#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-26 (EXP-134, EXP-135).

A800 connectivity: offline since 2026-05-14 (day ~73). Apply when restored.

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
    python3 auto_research/pending_queue_update_2026_07_23.py            # EXP-128, EXP-129
    python3 auto_research/pending_queue_update_2026_07_24.py            # EXP-130, EXP-131
    python3 auto_research/pending_queue_update_2026_07_24_paper.py      # EXP-132, EXP-133

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_26.py            # EXP-134, EXP-135

Queue was ~143 pending on 2026-07-24 (+4 added: EXP-130, EXP-131, EXP-132, EXP-133).
Queue cap applied: >20 → max 2 new experiments.

Data motivating today's proposals
----------------------------------
results/e2e_4cyc_gpt55/cycle_3/e2e_ablation_summary.json (unchanged — A800 offline day 73):
    HumanEval 4-cycle:
      large (always-large):        task_pass=96.34%, cost_vs_large=100%
      skills (always-small+proc):  task_pass=75.61%, cost_vs_large=10%
      router (logistic, cycle 3):  task_pass=92.68%, routing_acc=92.68%,
                                   cost_vs_large=27.56%, fallback=6.10%
      full (router+GRPO):          task_pass=92.68%  ← identical to router
    Root problems:
      (A) ACR=52.4% — 43/82 GRPO groups have zero within-group reward variance
          → zero gradient → Full = Router.
      (B) Non-collapsed groups (47.6%) carry ALL GRPO signal; credit quality matters.
      (C) Skills gap: 75.61% vs large 96.34% — 20.7pp gap.
      (D) GRPO may forget SFT gains during Phase 3b (CPO-PMP tests this: EXP-130).
      (E) Rise-and-collapse: GRPO peaks at early steps then collapses (EXP-132 tests this).

AAAI 2027 deadline: 2026-08-15 (20 days from today 2026-07-26).
A800 offline since 2026-05-14 (day 73).

arxiv:2607.21273 — "The Dark Room in the Reward Channel: Dense Prediction Rewards Collapse
    GRPO-Trained LLM Agents — and What Actually Works" (Wang, July 23 2026)

    This paper formalizes the GRPO zero-variance collapse mechanism as the "dark room"
    pathology: when all K rollouts in a group receive identical terminal rewards, GRPO's
    within-group z-scoring produces 0/0 advantages → zero gradient → the policy is
    trapped. In the dense-reward case, even partially positive shaping signals become
    "dark room" generators because bounded reward values produce identical z-scores
    (the normalized advantage is scale-invariant). The paper shows:
    1. Dense per-step prediction rewards always collapse GRPO across Qwen3-1.7B/4B/8B.
    2. Removing ONLY the σ normalization (Dr. GRPO) turns catastrophe → baseline parity.
    3. Sparse terminal binary rewards are safe for GRPO even without σ removal.
    CRITICAL FOR OUR PAPER: Our tau2 multi-turn agent oscillated 89.19%→70.27%→72.97%
    across cycles. We have NEVER measured the ACR metric (dark-room fraction) for the
    tau2 GRPO setting — only for HumanEval. If tau2's dark-room fraction is comparably
    high (>50%), this links two empirically separate findings (HumanEval ACR=52.4%;
    tau2 oscillation) under one theoretical cause. A `forgetting_eval` diagnostic that
    logs per-cycle per-group reward variance in tau2 GRPO groups is < 1h, zero training,
    and gives us a quantitative §5.5 contribution for the AAAI paper with 20 days left.
    This paper also introduces the term "dark room" which we can adopt as terminology
    that strengthens our theoretical narrative (currently we call it "zero-variance
    collapse" and "ACR"; the dark room framing is more vivid and citable).

arxiv:2607.16244 — "CIGPO: Contextual Information-Gain Policy Optimization for
    Multi-Turn Evidence-Reading LLM Agents" (Dou, July 2026)

    CIGPO addresses GRPO collapse in multi-turn agentic settings where intermediate
    turns receive no direct credit. For code repair specifically, the per-turn
    information-gain signal can be naturally estimated as the change in test-suite
    pass rate from turn t-1 to turn t: IG_t = pass_rate(turn_t) - pass_rate(turn_t-1).
    This gives intermediate credit even for globally-failed trajectories — if turn 2
    fixed 3 of 10 failing tests and turn 3 fixed 2 more, both get partial positive
    credit even if the trajectory never achieves full pass. In our HumanEval DAPO
    multi-turn repair setting (G=8 repair turns), this addresses a structural gap:
    standard GRPO assigns zero credit to all intermediate repair turns that improve
    coverage but don't reach 100%, treating them identically to destructive turns.
    CIGPO-Code is complementary to SCCA (EXP-131), which operates WITHIN trajectories
    at token level using the nearest-successful-rollout divergence as anchor. CIGPO-Code
    operates at the TURN level using test-coverage improvement as the anchor, and applies
    to ALL groups (including ACR groups where SCCA falls back to standard GRPO). It is
    also orthogonal to Dr. GRPO (EXP-120), which removes σ normalization but does not
    modify how turn-level credit is distributed. The three form a hierarchy:
    Dr. GRPO (fixes σ normalization) → CIGPO-Code (adds turn-level credit) → SCCA
    (adds token-level credit within turns). No prior queued experiment combines turn-
    level and token-level credit injection. Expected: Full arm 92.68% → ≥94%;
    non-ACR groups benefit most (turn-level credit improves multi-step trajectories);
    ACR groups may partially break out of dark room if intermediate turns generate
    non-zero coverage gains (IG_t > 0) even when the terminal reward=0.
"""

import json
import os
import tempfile
from pathlib import Path

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

NEW_EXPERIMENTS = [
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-134: Dark Room Tau2 ACR Diagnostic
    #          Fast forgetting_eval: log per-group reward variance in tau2 GRPO
    #          to measure dark-room fraction across 4 tau2 cycles.
    #          (arXiv:2607.21273, Wang, July 23 2026)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_26_001_darkroom_tau2_acr_diagnostic",
        "priority": 8,
        "gpu": "auto",
        "kind": "forgetting_eval",
        "rationale": (
            "arxiv:2607.21273 — 'The Dark Room in the Reward Channel: Dense Prediction "
            "Rewards Collapse GRPO-Trained LLM Agents — and What Actually Works' (Wang, "
            "July 23 2026). Formalizes our ACR=52.4% finding as the 'dark room' pathology: "
            "all-same-reward groups trap GRPO in a zero-gradient absorbing state. We have "
            "measured ACR for HumanEval (52.4%), but NEVER for tau2 GRPO groups, even though "
            "the tau2 multi-turn oscillation (89.19%→70.27%→72.97%) is our paper's most "
            "striking failure mode. This diagnostic runs in < 1h with no model training: "
            "replay each of the 4 tau2 cycle checkpoints, generate K=8 rollouts for each "
            "tau2 task, record terminal reward per rollout, compute sigma^2 per group, and "
            "log the fraction of groups with sigma^2=0 (the dark-room fraction). If tau2 "
            "dark-room fraction > 40%, it links the HumanEval and tau2 findings under one "
            "theoretical cause and adds a new quantitative data point to AAAI Section 5.5 "
            "(20 days to deadline). The paper also introduces the 'dark room' terminology "
            "that we can adopt to replace 'zero-variance collapse' and 'ACR=52.4%' with a "
            "more vivid and citable framing. This diagnostic is zero-risk: no training, no "
            "model changes, eval-only replay of existing checkpoints. Distinct from EXP-115 "
            "(HumanEval forgetting diagnostic across checkpoints): EXP-134 specifically "
            "measures the dark-room fraction on the tau2 task distribution."
        ),
        "spec": {
            "bench": "tau2_bench",
            "mode": "diagnostic_only",
            "eval_checkpoints": [
                "results/tau2_4cyc/cycle_0/llm_adapter/checkpoint-best",
                "results/tau2_4cyc/cycle_1/llm_adapter/checkpoint-best",
                "results/tau2_4cyc/cycle_2/llm_adapter/checkpoint-best",
                "results/tau2_4cyc/cycle_3/llm_adapter/checkpoint-best",
            ],
            "n_generations": 8,
            "grpo_temperature": 0.9,
            "metrics": [
                "per_group_reward_variance_sigma2",
                "dark_room_fraction_sigma2_eq_0",
                "per_cycle_pass_at_1",
                "per_cycle_acr_fraction",
                "sigma2_distribution_histogram_per_cycle",
            ],
            "output": "darkroom_tau2_diagnostic.json",
            "analysis": [
                "dark_room_fraction_tau2_vs_humaneval_comparison",
                "per_cycle_dark_room_fraction_trend",
                "correlation_dark_room_fraction_vs_pass_at_1_delta",
            ],
            "compare_with": [
                "HumanEval ACR=52.4% (43/82 groups, cycle 3 baseline)",
                "tau2 oscillation: 89.19%→70.27%→72.97% across cycles",
                "arxiv:2607.21273: dark room pathology theory",
                "EXP-115: HumanEval forgetting diagnostic (orthogonal)",
            ],
            "implementation_files": [
                "src/pipeline/collect_traces.py",
                "src/pipeline/grpo_train_simple.py",
            ],
            "implementation_note": (
                "Add a new diagnostic mode to collect_traces.py (or a standalone "
                "src/pipeline/darkroom_diagnostic.py): "
                "1. For each checkpoint in eval_checkpoints: "
                "   a. Load the adapter onto the base model "
                "   b. For each tau2 task t in the eval set: "
                "      run K=8 rollouts with temperature=0.9 "
                "      record terminal_rewards = [r_1, ..., r_K] (binary pass/fail) "
                "      sigma2 = variance(terminal_rewards) "
                "      dark_room_t = (sigma2 == 0) "
                "   c. dark_room_fraction = mean(dark_room_t) over all tasks "
                "   d. log to darkroom_tau2_diagnostic.json "
                "2. Print comparison table: HumanEval ACR vs tau2 dark-room fraction "
                "   per cycle. "
                "Overhead: 4 checkpoints × N_tau2_tasks × 8 rollouts. "
                "If N_tau2_tasks ~ 100, total = 3200 rollouts ≈ 15 min. "
                "No gradient computation, no parameter updates — pure forward pass. "
                "Enable via: DARK_ROOM_DIAGNOSTIC=1 bench=tau2_bench "
                "This diagnostic is safe to run alongside any queued experiment. "
                "Implementation reference: https://arxiv.org/abs/2607.21273"
            ),
            "arxiv_ref": "2607.21273",
            "estimated_gpu_hours": 0.5,
        },
    },
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-135: CIGPO-Code — Per-Turn Test-Coverage Credit for Multi-Turn GRPO
    #          Injects per-turn intermediate reward via test-coverage improvement
    #          for HumanEval DAPO multi-turn code repair.
    #          (arXiv:2607.16244, Dou, July 2026)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_26_002_cigpo_code_perturn_coverage_humaneval",
        "priority": 7,
        "gpu": "auto",
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2607.16244 — 'CIGPO: Contextual Information-Gain Policy Optimization "
            "for Multi-Turn Evidence-Reading LLM Agents' (Dou, July 2026). CIGPO addresses "
            "the problem that multi-turn GRPO assigns zero credit to intermediate turns "
            "that make progress but don't achieve the terminal goal. In HotpotQA, CIGPO "
            "injects information-gain at each evidence-reading turn based on the reduction "
            "in the reference model's uncertainty. For HumanEval multi-turn code repair "
            "(our DAPO G=8 turns), the natural analogue is test-coverage information gain: "
            "IG_t = pass_rate(turn_t_code) - pass_rate(turn_{t-1}_code). Each repair turn "
            "gets credit proportional to how many additional tests it passes, even if the "
            "full trajectory never reaches 100%. The combined CIGPO-Code loss is: "
            "L = L_terminal_GRPO + alpha * sum_t(IG_t * log_prob_t). "
            "This is complementary to SCCA (EXP-131) which operates at token level within "
            "a single turn using the nearest-successful-rollout divergence. CIGPO-Code "
            "operates at the turn level and applies to ALL groups including ACR groups — "
            "if turn 2 fixes 3/10 tests and turn 3 fixes 2 more, both get partial credit "
            "even when the terminal reward=0. This breaks ACR groups' zero-gradient lock: "
            "if sigma^2 of terminal rewards is 0 but sigma^2 of coverage-gain sums across "
            "turns is non-zero, CIGPO-Code provides gradient signal that standard GRPO "
            "misses. No prior experiment in the queue (EXP-001–EXP-133) applies turn-level "
            "test-coverage credit injection. EXP-120 (Dr.GRPO) removes sigma normalization; "
            "EXP-122 (AVSPO) injects synthetic virtual samples; EXP-123 (EDGE-GRPO) adds "
            "entropy-driven advantage and guided error correction; EXP-131 (SCCA) uses "
            "token-level divergence. CIGPO-Code is the only turn-level coverage-gain "
            "approach. Expected: Full arm 92.68% → ≥93.5%; improvement concentrated in "
            "non-trivially-failing tasks (those with partial test passage at intermediate "
            "repair turns); faster GRPO convergence per cycle."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 4,
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "cigpo_code_enabled": True,
            "cigpo_alpha": 0.3,
            "cigpo_min_ig": 0.05,
            "cigpo_normalize_ig": True,
            "grpo_temperature": 1.0,
            "n_generations": 8,
            "scaling_force_both": True,
            "sft_include_success": True,
            "analysis": [
                "full_vs_router_pass_at_1_per_cycle_cigpo_vs_vanilla",
                "per_turn_ig_distribution_per_cycle",
                "acr_fraction_cigpo_vs_vanilla",
                "turn_level_credit_contribution_per_task",
                "grpo_loss_terminal_vs_cigpo_turn_component_per_step",
                "tasks_graduating_acr_due_to_cigpo_turn_credit",
            ],
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (baseline): full=router=92.68%",
                "EXP-131 (SCCA): token-level credit within turns (orthogonal)",
                "EXP-120 (Dr.GRPO): sigma normalization removal (orthogonal)",
                "EXP-122 (AVSPO): virtual-sample injection for ACR (orthogonal)",
                "EXP-123 (EDGE-GRPO): entropy-guided correction (orthogonal)",
                "arxiv:2607.16244 (CIGPO): +12pp HotpotQA F1 on Qwen2.5-3B",
            ],
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
                "src/pipeline/collect_traces.py",
            ],
            "implementation_note": (
                "In grpo_train_simple.py, add per-turn test-coverage tracking: "
                "1. Add env var CIGPO_CODE_ENABLED (default 0), CIGPO_ALPHA (default 0.3), "
                "   CIGPO_MIN_IG (default 0.05, ignore turns with coverage gain < 5%). "
                "2. In the rollout collection phase, for each multi-turn repair trajectory: "
                "   a. Run turn 0 (initial attempt): eval on test suite → pass_rate_0 "
                "   b. For each repair turn t in [1, G]: "
                "      run turn t: eval on test suite → pass_rate_t "
                "      ig_t = max(0, pass_rate_t - pass_rate_{t-1}) "
                "      if ig_t >= CIGPO_MIN_IG: record (turn_t_tokens, ig_t) "
                "   c. If terminal reward=0 but sum(ig_t) > 0: "
                "      the trajectory made partial progress; CIGPO-Code gives partial credit "
                "3. In the loss computation: "
                "   for each turn t with ig_t >= CIGPO_MIN_IG: "
                "     if CIGPO_NORMALIZE_IG: ig_t_norm = ig_t / max(0.01, sum_ig_trajectory) "
                "     else: ig_t_norm = ig_t "
                "     cigpo_loss_t = -CIGPO_ALPHA * ig_t_norm * sum(log_prob(tokens_t)) "
                "   total_loss = grpo_terminal_loss + sum(cigpo_loss_t across turns) "
                "4. Log per-turn IG distribution to cigpo_turn_ig_log.csv "
                "   and the fraction of previously-ACR groups that now receive CIGPO "
                "   turn-level credit (tasks_graduating_acr_due_to_cigpo_turn_credit). "
                "Overhead: per-turn test evaluation for each rollout. "
                "With G=8 turns × K=8 rollouts × 82 tasks × 4 cycles = 21,504 test runs. "
                "Each test run is a Python pytest call (< 1s). Total overhead: ~6h → "
                "use CIGPO_MAX_TURNS_EVAL=3 (eval only first 3 repair turns per rollout) "
                "to limit to 8,064 test runs (~2h overhead within a 4h total run). "
                "IMPORTANT: per-turn test eval requires the repair turns to generate "
                "intermediate code files, not just a final answer. Verify that collect_traces "
                "saves intermediate repair turn outputs before implementing. "
                "Implementation reference: https://arxiv.org/abs/2607.16244"
            ),
            "arxiv_ref": "2607.16244",
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
    atomic_save(Path(STATE_PATH), state)
    print(f"\nDone. Added {len(added)} experiments. Total queue: {len(queue)} pending.")


if __name__ == "__main__":
    main()
