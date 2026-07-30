#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-30 (EXP-138, EXP-139).

A800 connectivity: offline since 2026-05-14 (day ~77). Apply when restored.

Prior patches to apply before this one (daily-patch chain):
    python3 auto_research/pending_queue_update.py
    ... (see pending_queue_update_2026_07_24.py header for full chain) ...
    python3 auto_research/pending_queue_update_2026_07_24.py            # EXP-130, EXP-131
    python3 auto_research/pending_queue_update_2026_07_24_paper.py      # EXP-132, EXP-133
    python3 auto_research/pending_queue_update_2026_07_27.py            # EXP-134, EXP-135
    python3 auto_research/pending_queue_update_2026_07_29.py            # EXP-136, EXP-137

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_30.py            # EXP-138, EXP-139

Queue was ~143 pending on 2026-07-29 (after EXP-136, EXP-137). Cap applied: >20 → max 2.

AAAI 2027 deadline: 2026-08-15 (16 days from today 2026-07-30).
A800 offline since 2026-05-14 (day 77).
GPU window CLOSES 2026-08-01 (2 days) — ABSOLUTE LAST DAY for new GPU results to reach paper.

=================================================================================
NEW PAPERS MOTIVATING TODAY'S EXPERIMENTS
=================================================================================

arxiv:2606.09932 — "When RL Fails after SFT: Rejuvenating Model Plasticity for
    Robust SFT-to-RL Handoff" (June 2026)

    Identifies that heavy SFT training reduces model plasticity — the model's
    capacity to accept new gradient updates. Overtrained SFT checkpoints have
    a flattened loss landscape near the SFT optimum; when RL (GRPO) begins, most
    gradient steps are absorbed by this flat region → effective learning rate near
    zero → GRPO fails. The fix: a brief entropy-regularized warmup phase at the
    start of RL training, which injects sufficient output distribution noise to
    re-open the optimization landscape before policy gradient updates are applied.
    In their experiments (Qwen2.5-Coder 1.5B/7B, HumanEval + MBPP), entropy warmup
    (coeff=0.05, 50 steps, then linear decay to 0) restores RL performance to the
    level achievable from an epoch-1 SFT checkpoint, without needing to discard the
    epoch-2 training.

    Connection to EXP-136/137: our cycle-1 SFT epoch-2 overfitting (loss 0.166→0.271)
    reduces plasticity exactly as described. EXP-137 fixes this by using the epoch-1
    checkpoint. EXP-138 offers an ORTHOGONAL fix: keep the epoch-2 checkpoint (which
    has lower training loss on most tasks) but add entropy warmup before GRPO-1,
    restoring plasticity without discarding epoch-2 learning.

arxiv:2607.08255 — "Compete Then Collaborate: Frontier AI Teachers Build a Verifiable
    Curriculum to Improve a Coding Student Beyond Imitation" (July 2026)

    Multiple frontier teacher models (e.g., GPT-5 + Claude 3.7) compete on a shared
    task set. Problems where they DISAGREE (one teacher passes, another fails) are at
    the 'Goldilocks' difficulty — hard enough to be informative, easy enough to provide
    verified signal. The curriculum is: train the student ONLY on these frontier-disagreement
    problems via GRPO (not SFT). Key result: GRPO on the competition-filtered curriculum
    lifts a 1.5B student from 5.9% to 8.8% on competition benchmarks (+49% relative),
    while SFT on the same curriculum DEGRADES performance.

    Connection to our pipeline: we currently train on ALL tasks (SCALING_FORCE_BOTH=1,
    Phase 1). The teacher is GPT-5.5 alone. We could filter to problems where BOTH the
    large model (GPT-5.5) AND the small model (Qwen2.5-1.5B) attempted but only ONE
    passed — the "disagreement zone" — as the training curriculum. These are exactly
    the problems where routing matters most (borderline tasks). Using them as the GRPO
    curriculum focuses gradient updates on the decision-relevant difficulty band.
"""

import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = "data/humaneval_eval.jsonl"

NEW_EXPERIMENTS = [
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-138: Plasticity Entropy Warmup for GRPO After SFT Overtraining
    #
    # Addresses the EXP-136 finding (cycle-1 SFT epoch-2 overfitting) with an
    # ORTHOGONAL fix to EXP-137: instead of selecting the epoch-1 checkpoint,
    # keep the epoch-2 checkpoint and restore plasticity via entropy warmup.
    # (arXiv:2606.09932, June 2026)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_30_001_plasticity_entropy_warmup_grpo_after_sft_overtraining",
        "priority": 8,
        "gpu": "auto",
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2606.09932 — 'When RL Fails after SFT: Rejuvenating Model Plasticity "
            "for Robust SFT-to-RL Handoff' (June 2026). "
            "EXP-136 (offline, 2026-07-29) established that cycle-1 SFT epoch-2 "
            "overfitting (training loss 0.166→0.271 from step 6 to step 12) reduces "
            "model plasticity, causing the GRPO-1 phase to underperform and producing "
            "the cycle-1 skills arm dip (65.85% vs 70.73% in cycle 0). "
            "EXP-137 (proposed 2026-07-29) fixes this by using the epoch-1 SFT "
            "checkpoint as the GRPO initializer. EXP-138 proposes an ORTHOGONAL "
            "fix: keep the epoch-2 checkpoint (which achieves lower loss on non-"
            "conflicting tasks) but add a brief entropy regularization warmup at the "
            "START of GRPO training to restore model plasticity before policy gradients "
            "are applied. "
            "Mechanism (per arxiv:2606.09932): overtrained SFT creates a flat loss "
            "landscape near the SFT optimum — subsequent GRPO gradient steps are "
            "absorbed by the flat region, leaving effective learning rate near zero. "
            "Adding entropy bonus H(π) × coeff for the first 50 GRPO steps injects "
            "sufficient output distribution noise to perturb the model out of the "
            "flat SFT optimum, re-opening the optimization landscape. After step 50, "
            "linear decay to entropy_coeff=0 restores standard GRPO. "
            "Predicted result: skills arm recovers to ≥70% (cycle-0 level) from the "
            "epoch-2 checkpoint, matching EXP-137 outcome or better (epoch-2 has lower "
            "training loss on non-conflicting tasks → better generalization once "
            "plasticity is restored). "
            "AAAI impact: §5.4 Hypothesis F now has TWO fixes: EXP-137 (epoch-1 "
            "checkpoint selection, simpler) and EXP-138 (entropy warmup, lower-cost "
            "if it works). A 2×2 table comparing epoch-1 vs epoch-2 init × with/without "
            "entropy warmup would be a strong result. "
            "Implementation: add `grpo_entropy_warmup_steps=50, grpo_entropy_coeff=0.05` "
            "to grpo_train_simple.py. GRPO_MODEL_PATH=checkpoint-12 (epoch 2, default). "
            "Estimated time: ~1.5h (single GRPO cycle). "
            "Priority: 8 (HIGH for AAAI, run AFTER EXP-132/EXP-135 if time permits "
            "before Aug 1)."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 1,
            "start_from_cycle": 1,
            "grpo_model_path": "results/e2e_4cyc_gpt55/cycle_1/llm_adapter/checkpoint-best",
            "sft_checkpoint_epoch": 2,
            "grpo_entropy_warmup_steps": 50,
            "grpo_entropy_coeff": 0.05,
            "grpo_entropy_coeff_schedule": "linear_decay_to_zero",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "grpo_temperature": 1.0,
            "n_generations": 8,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "analysis": [
                "skills_arm_pass_at_1_entropy_warmup_vs_no_warmup",
                "acr_fraction_entropy_warmup_vs_baseline",
                "grpo_entropy_trajectory_step_0_50_100_200",
                "grpo_loss_convergence_warmup_vs_no_warmup",
                "comparison_with_exp137_epoch1_checkpoint",
            ],
            "compare_with": [
                "cycle_1 baseline: skills=65.85%, no warmup, GRPO from checkpoint-12 (epoch 2)",
                "EXP-137: epoch-1 checkpoint (checkpoint-6) as GRPO init — orthogonal fix",
                "target: skills>=70.73% (cycle-0 level), confirming plasticity restored",
            ],
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
            ],
            "implementation_note": (
                "In grpo_train_simple.py, add entropy warmup: "
                "  entropy_coeff = args.grpo_entropy_coeff  # default 0.0, set 0.05 for this exp "
                "  warmup_steps = args.grpo_entropy_warmup_steps  # default 0, set 50 "
                "  for step in range(total_steps): "
                "      coeff = entropy_coeff * max(0, 1 - step / warmup_steps) "
                "      entropy_bonus = -coeff * policy.log_probs.mean()  # H(π) approx "
                "      loss = policy_loss - entropy_bonus "
                "Alternatively: use the existing entropy_coeff arg if already present. "
                "GRPO_MODEL_PATH remains checkpoint-best (epoch 2) to distinguish from EXP-137. "
                "Implementation reference: https://arxiv.org/abs/2606.09932."
            ),
            "arxiv_ref": "2606.09932",
            "estimated_gpu_hours": 1.5,
            "aaai_priority": "HIGH — orthogonal fix to EXP-137; together create 2x2 Hyp-F table",
            "dependency": "Run after EXP-132 (priority=9), EXP-135 (priority=8); before Aug 1",
        },
    },
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-139: Frontier-Disagreement Curriculum GRPO (Compete-Then-Collaborate)
    #
    # Filter training data to "borderline" tasks where large and small model
    # DISAGREE on pass@1, then run GRPO on this curriculum. Tests whether
    # routing-relevant difficulty band (where routing decisions matter most)
    # improves GRPO quality over full-task training.
    # (arXiv:2607.08255, July 2026)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_30_002_frontier_disagreement_curriculum_grpo",
        "priority": 6,
        "gpu": "auto",
        "kind": "grpo_curriculum_continual",
        "rationale": (
            "arxiv:2607.08255 — 'Compete Then Collaborate: Frontier AI Teachers Build "
            "a Verifiable Curriculum to Improve a Coding Student Beyond Imitation' "
            "(July 2026). "
            "The paper demonstrates that GRPO on a 'competition-filtered' curriculum — "
            "problems where frontier teachers DISAGREE (one passes, one fails) — achieves "
            "+49% relative improvement on competition benchmarks vs SFT on the same data "
            "(student lifts from 5.9% to 8.8% pass@1 on competition problems). "
            "The key insight: disagreement tasks lie at the 'Goldilocks' difficulty band — "
            "hard enough to be informative, easy enough to provide verified learning signal. "
            "Tasks where both teachers pass produce near-zero GRPO advantage (trivial for "
            "the student too); tasks where both fail provide no learning signal at all. "
            "Only disagreement tasks produce the within-group variance that GRPO needs. "
            "Connection to our pipeline: our collect_traces.py Phase 1 already runs BOTH "
            "the large (GPT-5.5) and small (Qwen2.5-1.5B) models on every task "
            "(SCALING_FORCE_BOTH=1). We therefore already HAVE the disagreement signal in "
            "traces.jsonl: tasks where large_pass=True AND small_pass=False (the student "
            "fails but the teacher passes) are exactly the 'routing-critical borderline' "
            "difficulty band. Tasks where both pass (easy) or both fail (too hard) are "
            "outside this band and less useful for GRPO gradient. "
            "EXP-139 tests: filter traces to large_pass=True, small_pass=False before "
            "GRPO training (instead of training on all tasks). This produces a "
            "difficulty-filtered curriculum automatically derived from the oracle run. "
            "The expected mechanism: GRPO ACR groups are now dominated by tasks where "
            "the student genuinely needs to improve (small initially fails, large passes) "
            "→ std(group rewards) is consistently > 0 → richer gradient signal. "
            "Also directly addresses the std-norm degenerate case (EXP-135): "
            "filtering to borderline tasks by construction reduces the number of "
            "all-pass groups (ACR) → less zero-advantage noise → complementary to EXP-135. "
            "The filtering logic is a 5-line addition to grpo_train_simple.py "
            "(filter traces where row['large_pass'] and not row['small_pass']). "
            "Estimated time: ~2h (single GRPO cycle with filtered curriculum). "
            "Priority: 6 (exploratory; run only if GPU time remains after EXP-132/135/137/138). "
            "AAAI impact: Adds a new 'curriculum' ablation arm to Table 9, grounding the "
            "curriculum idea in the latest frontier-teacher literature (arxiv:2607.08255). "
            "Distinct from EXP-128 (SC-SDPO, self-distillation density), EXP-134 "
            "(offline diagnostic), and EXP-135 (std-norm removal): this filters input "
            "CURRICULUM, not the advantage formula or checkpoint selection."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 1,
            "curriculum_filter": "large_pass_and_small_fail",
            "curriculum_filter_field_large": "large_pass",
            "curriculum_filter_field_small": "small_pass",
            "curriculum_filter_logic": "large_pass=True AND small_pass=False",
            "expected_curriculum_fraction": 0.25,
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "grpo_temperature": 1.0,
            "n_generations": 8,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "analysis": [
                "skills_arm_pass_at_1_curriculum_vs_full_tasks",
                "acr_fraction_curriculum_vs_full_tasks",
                "curriculum_task_count_and_difficulty_distribution",
                "grpo_advantage_std_curriculum_vs_full",
                "comparison_with_exp135_no_std_norm",
            ],
            "compare_with": [
                "cycle_1 baseline: full tasks, skills=65.85%",
                "EXP-135: no-std-norm GRPO (full tasks); tests advantage formula, not curriculum",
                "2607.08255: frontier disagreement curriculum → +49% relative on competition problems",
                "target: skills>=70% with 25% of tasks (borderline only)",
            ],
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
                "src/pipeline/collect_traces.py",
            ],
            "implementation_note": (
                "In grpo_train_simple.py, before loading training tasks: "
                "  if args.curriculum_filter == 'large_pass_and_small_fail': "
                "      traces = [t for t in traces "
                "               if t.get('large_pass', False) and not t.get('small_pass', True)] "
                "This requires that collect_traces.py Phase 1 writes 'large_pass' and 'small_pass' "
                "fields to each trace row (they already appear as 'oracle_pass' and 'small_pass' "
                "in the existing JSONL format — check field names in traces.jsonl). "
                "If SCALING_FORCE_BOTH=1 is set, all tasks have both oracle outcomes → ~25% will "
                "be large_pass=True, small_pass=False (based on the existing pass@1 gap). "
                "Implementation reference: https://arxiv.org/abs/2607.08255."
            ),
            "arxiv_ref": "2607.08255",
            "estimated_gpu_hours": 2.0,
            "aaai_priority": "EXPLORATORY — run only if GPU time remains after EXP-132/135/137/138",
            "dependency": "Requires SCALING_FORCE_BOTH=1 traces with both large_pass + small_pass fields",
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
