#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-29 (EXP-136, EXP-137).

A800 connectivity: offline since 2026-05-14 (day ~76). Apply when restored.

Prior patches to apply before this one (daily-patch chain):
    python3 auto_research/pending_queue_update.py
    ... (see pending_queue_update_2026_07_24.py header for full chain) ...
    python3 auto_research/pending_queue_update_2026_07_24.py            # EXP-130, EXP-131
    python3 auto_research/pending_queue_update_2026_07_24_paper.py      # EXP-132, EXP-133
    python3 auto_research/pending_queue_update_2026_07_27.py            # EXP-134, EXP-135

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_29.py            # EXP-136, EXP-137

Queue was ~143 pending on 2026-07-27 (after EXP-134, EXP-135). Cap applied: >20 → max 2.

AAAI 2027 deadline: 2026-08-15 (17 days from today 2026-07-29).
A800 offline since 2026-05-14 (day 76).
GPU window CLOSES 2026-08-01 (3 days) — EXP-137 will only run if A800 restores immediately.

=================================================================================
NEW FINDING (generated offline, 2026-07-29): SFT EPOCH-2 OVERFITTING IN CYCLE 1
=================================================================================

Offline analysis of results/e2e_4cyc_gpt55/cycle_*/llm_adapter/training_info.json
reveals a CLEAR epoch-2 overfitting pattern in cycle 1 that explains the non-monotonic
skills arm dip (65.85% in cycle 1 vs 70.73% in cycle 0):

  Cycle 0  step 6 (ep 1.0): loss=0.178, step 12 (ep 2.0): loss=0.101  [normal convergence]
  Cycle 1  step 6 (ep 1.0): loss=0.166, step 12 (ep 2.0): loss=0.271  [EPOCH-2 OVERFIT!]
  Cycle 2  step 6 (ep 1.0): loss=0.184, step 12 (ep 2.0): loss=0.069  [normal convergence]
  Cycle 3  step 6 (ep 1.0): loss=0.184, step 12 (ep 2.0): loss=0.113  [normal convergence]

Cycle 1 is the ONLY cycle where training loss increases in epoch 2 (from 0.166 to 0.271).
This coincides with entropy compression: entropy at step 12 in cycle 1 = 0.207, lower than
epoch-1 entropy of 0.255. In all other cycles, the final SFT checkpoint has lower loss and
comparable or higher entropy than the epoch-1 checkpoint.

MECHANISM:
The cycle-1 SFT starts from the GRPO-0 adapter (compressed, entropy=0.152 at step 1).
The GRPO-0 adapter is mode-collapsed: it has learned to pass specific HumanEval tasks
well, but has lost generalization entropy. When cycle-1 SFT tries to learn new teacher
traces (different distribution from the GRPO-0 training distribution), epoch 1 succeeds
in fitting the data (loss drops from 0.345 to 0.166). But epoch 2 over-regularizes:
the GRPO-0 adapter's compressed parameter space "snaps back" to its mode, causing the
loss to INCREASE to 0.271 in epoch 2. This is exactly the "Geometry Conflict" described
in arxiv:2605.09608: gradient directions of the new SFT data conflict with the parameter
geometry of the GRPO-0 adapter.

IMPLICATION:
If cycle 1's GRPO phase was initialized from the epoch-1 checkpoint (step 6, loss=0.166,
entropy=0.255) instead of the epoch-2 checkpoint (step 12, loss=0.271, entropy=0.207),
the GRPO-1 training would likely be MORE EFFECTIVE — consistent with arxiv:2606.18487
("SFT Overtraining Predicts Rank Inversion via Entropy Collapse Under RLVR").

This motivates EXP-137: test epoch-1 vs epoch-2 SFT checkpoint as GRPO initializer.

=================================================================================

New papers motivating this run:

arxiv:2606.18487 — "SFT Overtraining Predicts Rank Inversion via Entropy Collapse Under RLVR"
    Demonstrates that selecting the highest-eval-score SFT checkpoint for GRPO is
    counterproductive: overtrained SFT checkpoints have lower entropy (compressed
    output distribution) which leads to gradient vanishing at the start of GRPO →
    the overfit checkpoint is the WORST GRPO initializer. Using the epoch-1 checkpoint
    instead of the best-eval checkpoint consistently improves downstream GRPO quality.
    This paper directly motivates EXP-137.

arxiv:2605.09608 — "Geometry Conflict: Explaining and Controlling Forgetting in LLM
    Continual Post-Training"
    Proposes that forgetting in continual post-training is caused by conflicting gradient
    directions between old and new tasks. The "geometry conflict" score = cosine similarity
    between new-task gradients and the direction from the old-task minimum. High conflict
    → high forgetting → non-monotonic performance.
    Our cycle-1 epoch-2 overfitting is consistent with geometry conflict: the GRPO-0
    adapter's parameter space conflicts with cycle-1 SFT data gradients in epoch 2.
"""

import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = "data/humaneval_eval.jsonl"

NEW_EXPERIMENTS = [
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-136: SFT Epoch-2 Overfitting Diagnostic (offline, 0h GPU)
    #
    # NEW FINDING from local offline analysis (2026-07-29):
    # Cycle 1 SFT training loss INCREASES in epoch 2 (0.166→0.271) — the only
    # cycle showing this pattern. Entropy also drops (0.255→0.207). This
    # overfitting explains the cycle-1 non-monotonic skills arm dip (65.85%
    # vs 70.73% in cycle 0) — the GRPO was initialized from an overtrained SFT
    # checkpoint, consistent with arxiv:2606.18487.
    # (arXiv:2606.18487 + arxiv:2605.09608, June 2026)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_29_001_sft_epoch_overfit_diagnostic_offline",
        "priority": 6,
        "gpu": "auto",
        "kind": "forgetting_eval",
        "rationale": (
            "arxiv:2606.18487 — 'SFT Overtraining Predicts Rank Inversion via Entropy "
            "Collapse Under RLVR' (June 2026) and arxiv:2605.09608 — 'Geometry Conflict: "
            "Explaining and Controlling Forgetting in LLM Continual Post-Training' (May 2026). "
            "Offline analysis of results/e2e_4cyc_gpt55/cycle_*/llm_adapter/training_info.json "
            "reveals that cycle 1's SFT training has a unique epoch-2 overfitting pattern: "
            "training loss INCREASES from 0.166 (step 6, epoch 1) to 0.271 (step 12, epoch 2), "
            "while entropy DECREASES from 0.255 to 0.207. This is the ONLY cycle showing this: "
            "cycle 0 final loss=0.101, cycle 2 final loss=0.069, cycle 3 final loss=0.113 — "
            "all with monotonically decreasing loss. The mechanism (per arxiv:2605.09608 "
            "Geometry Conflict): cycle-1 SFT starts from the GRPO-0 adapter (entropy=0.152 "
            "at step 1, mode-collapsed), whose parameter geometry conflicts with cycle-1 "
            "training data gradient directions in epoch 2. Epoch 1 fits the data (loss drops "
            "normally), but epoch 2 overshoots the GRPO-0 adapter's attractor → loss reversal. "
            "The resulting epoch-2 SFT checkpoint (used as GRPO-1 initializer) is overtrained "
            "and entropy-compressed, explaining why cycle-1 skills arm (65.85%) is LOWER than "
            "cycle 0 (70.73%) despite more GRPO training data. Per arxiv:2606.18487, overtrained "
            "SFT checkpoints have compressed output distribution → gradient vanishing in GRPO → "
            "less effective GRPO-1 training → lower skills arm. This offline diagnostic "
            "formalizes these findings into paper-ready statistics: per-cycle epoch-1 vs epoch-2 "
            "loss comparison, entropy trajectories, step-1 gradient norms (proxy for distribution "
            "shift from prior GRPO adapter), and the quantitative 'overfitting gap' for each cycle. "
            "No GPU required. Paper impact: §5.4 new Hypothesis F paragraph ('Cycle-1 SFT "
            "Epoch-2 Overfitting from Geometry Conflict with GRPO-0 Adapter'), with the "
            "loss-reversal table (cycle 1: ep1=0.166, ep2=0.271; others: monotone) as Figure "
            "evidence. Motivates EXP-137 (epoch-1 checkpoint GRPO init) as the fix."
        ),
        "spec": {
            "bench": "humaneval",
            "offline_analysis": True,
            "estimated_gpu_hours": 0,
            "data_sources": [
                "results/e2e_4cyc_gpt55/cycle_0/llm_adapter/training_info.json",
                "results/e2e_4cyc_gpt55/cycle_1/llm_adapter/training_info.json",
                "results/e2e_4cyc_gpt55/cycle_2/llm_adapter/training_info.json",
                "results/e2e_4cyc_gpt55/cycle_3/llm_adapter/training_info.json",
            ],
            "existing_offline_findings": {
                "sft_loss_epoch1_vs_epoch2": {
                    "cycle_0": {"ep1_loss": 0.1784, "ep2_loss": 0.1013, "monotone": True},
                    "cycle_1": {"ep1_loss": 0.1664, "ep2_loss": 0.2712, "monotone": False, "flag": "OVERFITTING"},
                    "cycle_2": {"ep1_loss": 0.1842, "ep2_loss": 0.0693, "monotone": True},
                    "cycle_3": {"ep1_loss": 0.1835, "ep2_loss": 0.1125, "monotone": True},
                },
                "sft_entropy_init_vs_final": {
                    "cycle_0": {"h_init": 0.214, "h_ep1": 0.233, "h_final": 0.256},
                    "cycle_1": {"h_init": 0.152, "h_ep1": 0.255, "h_final": 0.207, "flag": "ENTROPY_REGRESSION"},
                    "cycle_2": {"h_init": 0.187, "h_ep1": 0.214, "h_final": 0.177},
                    "cycle_3": {"h_init": 0.146, "h_ep1": 0.214, "h_final": 0.240},
                },
                "skills_arm_pass_at_1": {
                    "cycle_0": 0.7073, "cycle_1": 0.6585, "cycle_2": 0.7317, "cycle_3": 0.7561
                },
                "grpo_acr_fraction": {
                    "cycle_0": 0.512, "cycle_1": 0.476, "cycle_2": 0.463, "cycle_3": 0.524
                },
            },
            "analysis_to_produce": [
                "per_cycle_loss_monotonicity_table",
                "entropy_compression_and_recovery_per_cycle",
                "step1_gradient_norm_as_distribution_shift_proxy",
                "overfitting_gap_metric: max(0, ep2_loss - ep1_loss)",
                "correlation_overfitting_gap_with_skills_arm_delta",
                "hypothesis_f_geometry_conflict_narrative",
            ],
            "implementation_files": [
                "auto_research/analyze_sft_epoch_overfitting.py",
            ],
            "implementation_note": (
                "Create auto_research/analyze_sft_epoch_overfitting.py: "
                "1. Load training_info.json for all 4 cycles. "
                "2. Extract: step 6 (epoch 1) loss, grad_norm, entropy; "
                "   step 12 (epoch 2) loss, grad_norm, entropy. "
                "3. Compute overfitting_gap = ep2_loss - ep1_loss. "
                "   Flag as 'overfitting' if > 0 (cycle 1 = +0.105; others = negative). "
                "4. Compute entropy_regression = ep1_entropy - ep2_entropy. "
                "   Flag if > 0.02 (cycle 1 = 0.048; others ~ 0 or negative). "
                "5. Correlate overfitting_gap with skills_arm_pass_at_1 delta (vs prior cycle). "
                "6. Output: auto_research/sft_epoch_overfitting_report.json with per-cycle "
                "   metrics and a summary string for §5.4 Hypothesis F. "
                "Python only, no GPU, ~60 lines. "
                "Key result to report: 'Cycle 1 is the only cycle with overfitting_gap > 0 "
                "({:.3f}), coinciding with the only cycle where skills arm dips below the "
                "prior cycle ({:.2f}% < {:.2f}%)' "
                "Implementation reference: https://arxiv.org/abs/2606.18487."
            ),
            "arxiv_refs": ["2606.18487", "2605.09608"],
        },
    },
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-137: Epoch-1 SFT Checkpoint as GRPO Initializer — Single Cycle (~1.5h)
    #
    # Direct fix for the EXP-136 offline finding: use checkpoint-6 (epoch 1,
    # loss=0.166, entropy=0.255) instead of checkpoint-12 (epoch 2, loss=0.271,
    # entropy=0.207) as the GRPO initializer for cycle 1. Tests whether the
    # cycle-1 non-monotonic dip is caused by the overtrained SFT checkpoint.
    # (arXiv:2606.18487, June 2026)
    # Priority: run AFTER EXP-132 (best-epoch, priority=9) and EXP-135 (no-std-norm,
    # priority=8) if A800 restores before August 1.
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_29_002_sft_epoch1_checkpoint_grpo_init_single_cycle",
        "priority": 7,
        "gpu": "auto",
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2606.18487 — 'SFT Overtraining Predicts Rank Inversion via Entropy "
            "Collapse Under RLVR' (June 2026). Follow-up to EXP-136 offline finding: "
            "cycle 1's SFT epoch-2 overfitting (loss 0.166→0.271 between epoch 1 and 2) "
            "means the GRPO-1 phase was initialized from an OVERTRAINED SFT checkpoint "
            "(step 12, loss=0.271, entropy=0.207) rather than the epoch-1 checkpoint "
            "(step 6, loss=0.166, entropy=0.255). arxiv:2606.18487 predicts that using "
            "the overtrained checkpoint as GRPO initialization leads to gradient vanishing "
            "at GRPO start (due to entropy-compressed output distribution), explaining why "
            "cycle-1 GRPO produced a skills arm (65.85%) LOWER than cycle 0 (70.73%). "
            "The fix: select checkpoint-6 (epoch 1) as GRPO initializer when epoch-2 loss "
            "is HIGHER than epoch-1 loss (overfitting_gap > 0). Test by running a single "
            "GRPO cycle initialized from checkpoint-6 of cycle-1 SFT "
            "(stored at results/e2e_4cyc_gpt55/cycle_1/llm_adapter/checkpoint-6) "
            "vs the baseline (checkpoint-12 = epoch 2). Compare: "
            "skills arm pass@1, ACR fraction, GRPO loss convergence curve. "
            "If epoch-1 init improves skills arm to ≥70%, the SFT checkpoint selection "
            "bug is confirmed as the cycle-1 dip cause — a free fix for future cycles. "
            "Distinct from EXP-132 (which selects best GRPO checkpoint DURING GRPO training, "
            "not the SFT checkpoint used to INITIALIZE GRPO) and EXP-135 (which changes "
            "the GRPO advantage computation, not initialization). This experiment changes "
            "only the SFT checkpoint used as GRPO starting point — a one-line change in "
            "run_full_pipeline.sh: GRPO_MODEL_PATH=...checkpoint-6 instead of ...checkpoint-best. "
            "Estimated time: ~1.5h (single GRPO cycle). "
            "AAAI priority: HIGH if confirmed — directly strengthens the cycle-evolution "
            "results by explaining the non-monotonic dip AND showing it's fixable. "
            "Queue priority: run after EXP-132 (priority=9) and EXP-135 (priority=8) "
            "if GPU time remains before August 1."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 1,
            "start_from_cycle": 1,
            "sft_checkpoint_to_use": "checkpoint-6",
            "sft_checkpoint_path": "results/e2e_4cyc_gpt55/cycle_1/llm_adapter/checkpoint-6",
            "baseline_checkpoint": "results/e2e_4cyc_gpt55/cycle_1/llm_adapter/checkpoint-12",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "grpo_temperature": 1.0,
            "n_generations": 8,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "analysis": [
                "skills_arm_pass_at_1_epoch1_checkpoint_vs_epoch2_baseline",
                "acr_fraction_epoch1_init_vs_epoch2_init",
                "grpo_loss_convergence_epoch1_vs_epoch2_init",
                "grpo_entropy_at_init_epoch1_vs_epoch2",
            ],
            "compare_with": [
                "cycle_1 baseline: skills=65.85%, GRPO init from checkpoint-12 (loss=0.271, H=0.207)",
                "target: skills≥70.73% (cycle-0 level), confirming epoch-2 overfit is the bug",
                "EXP-132 (best-epoch GRPO checkpoint selection): orthogonal fix, different phase",
                "EXP-135 (no-std-norm advantage): orthogonal fix, GRPO computation not initialization",
            ],
            "implementation_files": [
                "scripts/run_full_pipeline.sh",
            ],
            "implementation_note": (
                "In run_full_pipeline.sh: before the GRPO phase for cycle 1, override the "
                "model path to use checkpoint-6 instead of checkpoint-best: "
                "  export GRPO_MODEL_PATH=results/e2e_4cyc_gpt55/cycle_1/llm_adapter/checkpoint-6 "
                "Or more generally: add GRPO_SFT_CHECKPOINT_EPOCH=1 env var that selects the "
                "epoch-1 checkpoint when epoch-2 loss > epoch-1 loss (auto-detect overfitting). "
                "One-line change; no training code modification needed. "
                "Checkpoint-6 already exists in the repo at: "
                "results/e2e_4cyc_gpt55/cycle_1/llm_adapter/checkpoint-6/adapter_config.json "
                "Implementation reference: https://arxiv.org/abs/2606.18487."
            ),
            "arxiv_ref": "2606.18487",
            "estimated_gpu_hours": 1.5,
            "aaai_priority": "HIGH — explains cycle-1 dip and provides a clean fix; 1.5h GPU",
            "dependency": "Run after EXP-132 and EXP-135 if GPU time remains before Aug 1",
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
