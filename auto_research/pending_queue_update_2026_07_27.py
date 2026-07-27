#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-27 (EXP-134, EXP-135).

A800 connectivity: offline since 2026-05-14 (day ~74). Apply when restored.

Prior patches to apply before this one (daily-patch chain):
    python3 auto_research/pending_queue_update.py
    ... (see pending_queue_update_2026_07_24.py header for full chain) ...
    python3 auto_research/pending_queue_update_2026_07_24.py            # EXP-130, EXP-131
    python3 auto_research/pending_queue_update_2026_07_24_paper.py      # EXP-132, EXP-133

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_27.py            # EXP-134, EXP-135

Queue was ~143 pending on 2026-07-24 (after EXP-132, EXP-133 from paper patch).
Queue cap applied: >20 → max 2 new experiments.

AAAI 2027 deadline: 2026-08-15 (19 days from today 2026-07-27).
A800 offline since 2026-05-14 (day 74).
GPU window CLOSES 2026-08-01 (5 days) — new results must land by Aug 1 to reach paper.

New papers motivating this run:

arxiv:2607.21273 — "The Dark Room in the Reward Channel: Dense Prediction Rewards
    Collapse GRPO-Trained LLM Agents — and What Actually Works" (July 2026)

    BACKGROUND:
    GRPO's group-relative advantage uses z-score normalization:
        A_i = (r_i - mean(r_group)) / std(r_group)
    The paper demonstrates that within an all-pass or near-all-pass group, std(r_group)
    approaches zero. GRPO z-scoring then amplifies the residual ε-scale variation
    (numerical noise, minor formatting reward differences) to O(1) — creating spurious
    high-magnitude gradients from groups that should contribute zero gradient.
    This is termed the "Dark Room" pathology: the optimizer finds a degenerate fixed
    point where task success is maximized in a vacuous sense (100% prediction accuracy
    on a trivial sub-task) at the cost of overall task performance.

    ROOT CAUSE:
    The vulnerability is not the reward signal itself but GRPO's std normalization step.
    Single-factor ablation: removing only std normalization (keeping mean subtraction)
    eliminates the Dark Room collapse. The resulting advantage is:
        A_i = r_i - mean(r_group)   [no std division]
    For zero-variance groups, this produces exactly-zero advantages (not amplified noise).

    RELEVANCE TO OUR PIPELINE:
    Our 52.4% ACR finding: 43/82 GRPO groups have zero within-group reward variance
    (all 8 rollouts pass OR all 8 fail → reward = const). Under standard GRPO:
    - All-fail group (r=0 for all 8): mean=0, std→0 → A = (0 - 0) / ε = amplified noise
    - All-pass group (r=1 for all 8): mean=1, std→0 → A = (1 - 1) / ε = amplified noise
    The "zero-gradient" description of ACR assumes std is meaningful — but if std is
    numerically near-zero, GRPO does NOT produce zero gradient: it produces amplified noise.
    This means ACR groups may be actively HARMING training (not merely contributing zero).

    With no-std-norm (A_i = r_i - mean(r_group)):
    - All-fail group: A = 0 - 0 = 0 exactly (truly zero gradient — no noise amplification)
    - All-pass group: A = 1 - 1 = 0 exactly (truly zero gradient)
    - Non-ACR group: A = r_i - mean (still informative, sign correct, magnitude = reward gap)

    KEY RESULTS (paper, Qwen3 on ALFWorld):
    - Dense prediction reward + std-norm: collapses to Dark Room (0% task success)
    - Dense prediction reward + NO std-norm: performs at baseline parity (no collapse)
    Removing std normalization is a minimal one-line fix to prevent spurious gradient
    amplification in zero-variance groups.

    RELEVANCE TO HYPOTHESIS D/E:
    If ACR groups produce amplified noise gradients (not truly zero), they could be
    the mechanism behind: (D) GRPO forgetting SFT gains (noisy gradient degrades easy-task
    weights), and (E) within-cycle rise-and-collapse (noisy gradient causes model drift
    away from the SFT checkpoint without converging). No-std-norm fixes both by
    silencing zero-variance groups completely.

    PRIOR QUEUE CHECK:
    - EXP-126 (GEPO): entropy-controlled advantage ATTENUATION — scales down advantages
      for high-entropy groups, but does NOT remove std normalization.
    - EXP-130 (CPO-PMP): L2-SP regularization — limits parameter drift, does not fix
      the spurious gradient amplification at the advantage computation level.
    - EXP-131 (SCCA): self-conditioned token-level credit — refines credit within non-ACR
      groups, but still uses std normalization for the group-level scaling.
    - EXP-132 (rise-and-collapse): best-epoch checkpointing — saves best eval checkpoint,
      but does not address the source of the within-cycle collapse.
    EXP-135 is the ONLY queued experiment that removes std normalization as the fix,
    directly testing the Dark Room hypothesis in our codebase.

arxiv:2607.01763 — "Denser ≠ Better: Limits of On-Policy Self-Distillation for
    Continual Post-Training" (July 2026)

    BACKGROUND:
    Self-Distillation Policy Optimization (SDPO) augments GRPO/PPO with a dense
    token-level KL distillation loss toward the current model's own greedy predictions
    (the "teacher" is the current model's argmax rollout). In single-cycle post-training,
    SDPO accelerates in-domain specialization: the dense supervision signal gives more
    gradient per rollout than sparse verifiable rewards alone.

    KEY FINDING:
    In CONTINUAL post-training (multiple sequential cycles, as in our pipeline), SDPO
    exhibits stronger forgetting and can collapse:
    - Denser distillation induces larger parameter-space drift per cycle.
    - A self-reinforcing teacher–student loop amplifies high-frequency formatting
      artifacts across cycles: the "teacher" greedy rollout learns a formatting bias,
      distils it to the student, the student adopts it, the next cycle's "teacher"
      is even more biased, etc.
    - GRPO (without self-distillation) is more conservative: sparse verifiable rewards
      constrain learning to task-relevant updates, limiting cross-cycle drift.

    RELEVANCE TO OUR PIPELINE:
    EXP-128 (SC-SDPO, pass-rate weighted SFT) is already queued. This paper is a
    WARNING: SC-SDPO's pass-rate weighting ŵ = √[p̂(1−p̂)] concentrates SFT supervision
    on intermediate-difficulty tasks, which is good for single-cycle performance.
    But in our 4-cycle continual pipeline:
    - Cycles 1–3 use the previous cycle's model as the SFT student.
    - The SC-SDPO teacher traces are from the large model (GPT-5.5), not the student.
    - However, the ŵ weighting is derived from the SMALL model's own pass-rates — i.e.,
      it is a form of self-conditioned signal that adapts each cycle.
    - This is precisely the self-reinforcing loop the paper warns about: if the small
      model's pass-rate distribution shifts across cycles, ŵ shifts too, potentially
      amplifying biases rather than correcting them.

    MITIGATION EXPERIMENT (EXP-134):
    Run a 1-cycle forgetting_eval of SC-SDPO weighting vs uniform SFT weighting:
    measure per-category pass@1 after SFT on easy tasks (p̂ ≈ 1.0) that are down-weighted
    by ŵ. If easy-task performance regresses under SC-SDPO weighting (even with the same
    teacher traces), this confirms the paper's warning and should delay or modify EXP-128.
    This is a zero-GPU diagnostic using existing trace data.

    PRIOR QUEUE CHECK:
    EXP-128 (SC-SDPO, already queued) tests the full 4-cycle run.
    EXP-134 is a pre-diagnostic for EXP-128: if this eval shows easy-task forgetting
    from just 1-cycle ŵ weighting, EXP-128 should be deprioritized in favor of a
    more conservative SFT weighting approach (e.g., linear ŵ or capped ŵ_max=0.5).
"""

import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = "data/humaneval_eval.jsonl"

NEW_EXPERIMENTS = [
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-134: Dark Room Variance-Profile Diagnostic (offline, 0h GPU)
    #          Tests whether ACR groups produce amplified-noise gradient under
    #          GRPO std normalization — the "Dark Room" mechanism.
    #          (arXiv:2607.21273, July 2026)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_27_001_dark_room_variance_profile_offline",
        "priority": 7,
        "gpu": "auto",
        "kind": "forgetting_eval",
        "rationale": (
            "arxiv:2607.21273 — 'The Dark Room in the Reward Channel: Dense Prediction "
            "Rewards Collapse GRPO-Trained LLM Agents — and What Actually Works' (July 2026). "
            "Demonstrates that GRPO's std normalization amplifies near-zero within-group "
            "variance to O(1) spurious gradients in zero-variance groups. Our 52.4% ACR "
            "groups are assumed to produce exactly-zero gradient, but if std(r_group) is "
            "numerically near-zero (not exactly zero), z-scoring amplifies ε-scale noise "
            "to O(1) — meaning ACR groups may be actively degrading easy-task weights "
            "(consistent with Hypothesis D: GRPO forgets SFT gains). This offline analysis "
            "uses existing grpo_info.json group reward log data from all 4 cycles to: "
            "(1) compute the empirical std(r_group) distribution for all 82×4 groups, "
            "(2) identify the fraction of 'ACR' groups with std < ε vs std exactly 0, "
            "(3) estimate the implied advantage magnitude under std normalization for these "
            "groups vs the no-std-norm alternative. Zero GPU required; uses existing "
            "results/e2e_4cyc_gpt55/cycle_*/grpo_adapter/grpo_info.json. If many ACR "
            "groups have std in [1e-6, 1e-2] (numerically near-zero but nonzero), the "
            "Dark Room mechanism is active in our pipeline and EXP-135 (no-std-norm GRPO) "
            "becomes AAAI-critical. If ACR groups have exactly-zero std (all 8 rewards "
            "identical bit-for-bit), then std-norm is safe and EXP-135 is lower priority. "
            "Paper impact: grounds the '52.4% zero-variance' claim in §7 with a quantitative "
            "analysis distinguishing exactly-zero from near-zero variance groups; adds the "
            "Dark Room mechanism as a new component of Hypothesis D."
        ),
        "spec": {
            "bench": "humaneval",
            "offline_analysis": True,
            "estimated_gpu_hours": 0,
            "data_sources": [
                "results/e2e_4cyc_gpt55/cycle_0/grpo_adapter/grpo_info.json",
                "results/e2e_4cyc_gpt55/cycle_1/grpo_adapter/grpo_info.json",
                "results/e2e_4cyc_gpt55/cycle_2/grpo_adapter/grpo_info.json",
                "results/e2e_4cyc_gpt55/cycle_3/grpo_adapter/grpo_info.json",
            ],
            "analysis": [
                "group_reward_std_distribution_histogram_per_cycle",
                "acr_group_near_zero_std_fraction",
                "implied_amplified_advantage_magnitude_std_norm_vs_no_std_norm",
                "correlation_near_zero_std_groups_with_cycle1_skills_arm_dip",
                "dark_room_risk_score_per_cycle",
            ],
            "thresholds": {
                "exactly_zero_std": 1e-9,
                "near_zero_std_upper": 1e-2,
                "high_amplification_risk": "std < 1e-3 with any reward nonzero",
            },
            "compare_with": [
                "arxiv:2607.21273 Dark Room: std collapse → O(1) spurious gradient",
                "Our ACR=52.4%: assumed zero-gradient, but may be near-zero-std amplified",
                "EXP-135 (no-std-norm GRPO): the GPU fix, gated on this offline diagnostic",
            ],
            "implementation_files": [
                "auto_research/analyze_dark_room_variance.py",
            ],
            "implementation_note": (
                "Create auto_research/analyze_dark_room_variance.py: "
                "1. Load grpo_info.json for each cycle (key 'group_rewards': list of "
                "   lists, shape [n_tasks, K=8] where K is rollouts per task). "
                "2. For each group g: compute std_g = np.std(rewards[g]). "
                "   Classify: 'exactly_zero' if std_g < 1e-9, 'near_zero' if std_g < 1e-2, "
                "   'informative' otherwise. "
                "3. Compute amplified advantage magnitude under std-norm: "
                "   if std_g > 0: max_advantage = max(|r_i - mean|) / std_g "
                "   If std_g ∈ [1e-6, 1e-2]: max_advantage can be >> 1 (spurious amplification). "
                "4. Histogram of std_g values and max_advantage values, per cycle. "
                "5. Output: dark_room_variance_report.json with per-cycle statistics "
                "   and a 'dark_room_risk_active' boolean (True if >5% groups have "
                "   std ∈ [1e-6, 1e-2]). "
                "Python only, no GPU, ~50 lines. "
                "Implementation reference: https://arxiv.org/abs/2607.21273."
            ),
            "arxiv_ref": "2607.21273",
        },
    },
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-135: No-Std-Norm GRPO — Silencing Zero-Variance Groups
    #          Removes std normalization from GRPO advantage to prevent Dark Room
    #          spurious gradient amplification in ACR groups.
    #          (arXiv:2607.21273, July 2026)
    #          Priority: RUN AFTER EXP-134 CONFIRMS dark_room_risk_active=True.
    #          ~2h GPU — fits within August 1 GPU window if A800 restores by July 30.
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_27_002_no_std_norm_grpo_single_cycle",
        "priority": 8,
        "gpu": "auto",
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2607.21273 — 'The Dark Room in the Reward Channel' (July 2026). "
            "GRPO's standard advantage computation divides by group std: "
            "A_i = (r_i − mean) / std. In zero-variance groups (std → 0), this amplifies "
            "numerical noise to O(1) gradient, potentially destabilizing easy-task weights "
            "(Dark Room mechanism). No-std-norm replaces this with A_i = r_i − mean, which "
            "produces exactly zero gradient for all-pass and all-fail groups (our 52.4% ACR). "
            "This is the minimal one-line fix in grpo_train_simple.py "
            "(`advantages = rewards - rewards.mean(-1, keepdim=True)` without std division). "
            "Run a single GRPO cycle starting from the cycle-3 SFT checkpoint (same "
            "starting point as the baseline cycle-3 GRPO run), with no-std-norm advantage. "
            "Compare full arm pass@1 and ACR fraction vs baseline (cycle-3: full=92.68%, "
            "ACR=52.4%). If no-std-norm improves full arm to ≥93.5%, the Dark Room "
            "mechanism is confirmed as a contributor to Full=Router, and no-std-norm "
            "is a zero-cost fix across all future cycles. "
            "Distinct from all queued experiments: EXP-126 (GEPO) attenuates by entropy "
            "but still uses std; EXP-130 (CPO-PMP) regularizes parameters not advantages; "
            "EXP-131 (SCCA) refines token-level credit but preserves std-norm; EXP-132 "
            "(best-epoch) changes checkpoint selection but not gradient computation. "
            "EXP-135 is the only experiment directly testing std-norm removal as the fix. "
            "Single-cycle (~2h GPU) fits within the August 1 GPU window. Paper impact: "
            "if confirmed, adds a 2-line fix to §3.4 and a new 'no-std-norm' full-arm "
            "data point to Table 9 — directly strengthens soundness with a positive result "
            "on the Full=Router gap. Dependency: run EXP-134 (offline, 0h) first to "
            "confirm dark_room_risk_active before spending GPU."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 1,
            "start_from_cycle": 3,
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "grpo_normalize_advantage_by_std": False,
            "grpo_temperature": 1.0,
            "n_generations": 8,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "analysis": [
                "full_arm_pass_at_1_no_std_norm_vs_baseline_cycle3",
                "acr_fraction_no_std_norm_vs_baseline",
                "grpo_loss_curve_no_std_norm_vs_std_norm",
                "advantage_magnitude_distribution_no_std_norm",
                "easy_task_pass_at_1_before_after_no_std_norm_grpo",
            ],
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (baseline): full=router=92.68%, ACR=52.4%",
                "arxiv:2607.21273: removing std-norm turns Dark Room from 0% → baseline parity",
                "EXP-126 (GEPO): entropy-attenuation but std-norm retained",
                "EXP-130 (CPO-PMP): parameter-space fix orthogonal to advantage computation",
            ],
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
            ],
            "implementation_note": (
                "In grpo_train_simple.py, locate the advantage computation step: "
                "  # Current (std-norm): "
                "  advantages = (rewards - rewards.mean(-1, keepdim=True)) / "
                "               (rewards.std(-1, keepdim=True) + 1e-8) "
                "  # No-std-norm fix: "
                "  advantages = rewards - rewards.mean(-1, keepdim=True) "
                "Add env var: GRPO_NORMALIZE_BY_STD (default 1 for backward compat; "
                "set to 0 to enable no-std-norm). "
                "Note: with no-std-norm, advantage magnitudes are in [0, 1] for binary "
                "rewards (all-fail: 0; all-pass: 0; mixed: r_i - p̂ ∈ [-p̂, 1-p̂]). "
                "Learning rate may need a compensating increase to maintain gradient "
                "magnitude (e.g., GRPO_LR *= 2.0 with no-std-norm). Run both with and "
                "without LR compensation if time allows; otherwise use default LR first. "
                "This is a 2-line change to grpo_train_simple.py. "
                "Implementation reference: https://arxiv.org/abs/2607.21273."
            ),
            "arxiv_ref": "2607.21273",
            "estimated_gpu_hours": 2.0,
            "aaai_priority": "HIGH — 2h GPU, fits Aug 1 window; direct fix for Full=Router gap",
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
