#!/usr/bin/env python3
"""
Daily queue patch — 2026-08-03 (EXP-148, EXP-149).

A800 connectivity: offline since 2026-05-14 (day ~81). Apply when restored.

Prior patches to apply before this one (daily-patch chain):
    python3 auto_research/pending_queue_update.py
    ... (see pending_queue_update_2026_07_30.py header for full chain) ...
    python3 auto_research/pending_queue_update_2026_07_31.py            # EXP-140, EXP-141
    python3 auto_research/pending_queue_update_2026_07_31_paper.py      # EXP-142, EXP-143
    python3 auto_research/pending_queue_update_2026_08_01.py            # EXP-144, EXP-145
    python3 auto_research/pending_queue_update_2026_08_02.py            # EXP-146, EXP-147

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_08_03.py            # EXP-148, EXP-149

Queue was ~151 pending on 2026-08-02 (after EXP-146, EXP-147). Cap applied: >20 -> max 2.

AAAI 2027 deadline: 2026-08-15 (12 days from today 2026-08-03).
A800 offline since 2026-05-14 (day 81).
GPU window CLOSED 2026-08-01 — new GPU results cannot reach AAAI paper.
EXP-146 is 100% OFFLINE — runs against existing training logs, no GPU required.
EXP-147 is GPU-only (~2h) — queued for post-deadline execution or rebuttal.

=================================================================================
NEW PAPERS MOTIVATING TODAY'S EXPERIMENTS
=================================================================================

arxiv:2605.20005 — "Fine-Tuning Without Forgetting via Loss-Adaptive Learning
    Rates" (FINCH), Prashant et al., May 19, 2026.

    Key theorem: per-step forgetting F_t is bounded by
        F_t ≤ η_t · √loss_t · C
    where η_t is the learning rate at step t and loss_t is the batch training
    loss at step t. High-loss batches are thus the primary driver of forgetting,
    not just total steps. FINCH adapts η_t to counteract this: it reduces the
    learning rate when loss is high and increases it as the model converges.
    FINCH reduces catastrophic forgetting by 93% on average while preserving
    target-task accuracy.

    Connection to Hypothesis F and our cycle-1 SFT overfit:
    EXP-136 established offline that cycle-1 SFT epoch-2 produces a loss
    REVERSAL (loss increases from 0.166 to 0.271 across the epoch), a 62.7%
    loss elevation. By the FINCH bound, each step in epoch-2 introduces
    ≥√(0.271/0.166) ≈ 1.28× more forgetting per unit of LR than epoch-1 steps.
    Because our SFT schedule uses a cosine warmup-decay LR that does NOT adapt
    to batch loss, it violates the FINCH invariant at exactly the epoch-2
    overfit boundary.

    EXP-146 provides the first QUANTITATIVE upper-bound on cycle-1 forgetting:
    compute step-level FINCH bound (η_t · √loss_t) from the saved training log
    and compare against the observed 4.88pp skills arm dip. If the step-level
    bound integral exceeds the observed dip, the FINCH mechanism is a sufficient
    explanation for Hypothesis F and can be cited as a novel forgetting metric
    in §5.3 without any additional GPU experiments. EXP-146 is 100% offline
    and can complete before the AAAI deadline.

arxiv:2607.26862 — "ReCo: Reweighting GRPO Against Distributional
    Concentration", July 2026.

    GRPO training over multiple steps concentrates probability mass onto a
    shrinking subset of reasoning paths, reducing Pass@k coverage (the "Dark
    Room" effect in our language). ReCo applies importance-ratio reweighting
    to the GRPO objective so that underrepresented rollout groups contribute
    more gradient signal, countering distributional concentration without
    changing the reward signal.

    Connection to our cycle-3 ACR=52.4% collapse:
    Our cycle-3 GRPO (Phase 3b) shows 52.4% of training groups with
    std(reward)=0 — the zero-variance collapse consistent with distributional
    concentration. EXP-135 (no-std-norm) removes the normalization division
    at zero groups; EXP-139 (frontier-disagreement curriculum) diversifies
    the training distribution entering GRPO. ReCo offers a THIRD orthogonal
    remedy: reweighting groups that are currently under-explored. If ReCo
    reduces zero-variance groups from 52.4% to <25%, it directly increases
    effective GRPO gradient signal and should recover the cycle-3 skills arm
    plateau (full arm 76.83% ≈ skills arm 75.61% at cycle 4, Table 9).

    EXP-147 applies ReCo to Phase 3b at cycle 3 (the worst-ACR cycle) with
    the same n=1 budget as all other experiments, producing a new Table 9 row
    if the A800 is ever restored.
"""

import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

NEW_EXPERIMENTS = [
    # -----------------------------------------------------------------------
    # EXP-148: FINCH Forgetting Bound — Offline Validation on Cycle-1 SFT Log
    #          (arxiv:2605.20005, May 19, 2026)
    # -----------------------------------------------------------------------
    {
        "id": "exp_2026_08_03_001_finch_forgetting_bound_offline_validation_sft_cycle1",
        "priority": 8,
        "gpu": "auto",
        "kind": "forgetting_eval",
        "rationale": (
            "arxiv:2605.20005 (FINCH, May 2026) proves per-step forgetting is bounded by "
            "η_t·√loss_t. Our cycle-1 SFT epoch-2 shows loss REVERSAL (0.166→0.271, "
            "+62.7%); by the FINCH bound each epoch-2 step introduces ≥1.28× more forgetting "
            "than an epoch-1 step at equal LR. EXP-146 is a 100% OFFLINE analysis: load the "
            "cycle-1 SFT training log (results/e2e_4cyc_gpt55/cycle_1/sft_train_log.json or "
            "equivalent), compute step-level FINCH forgetting proxy η_t·√loss_t for all steps "
            "across both epochs, and compare epoch-1 vs epoch-2 cumulative proxy sums. "
            "Prediction: epoch-2 cumulative FINCH bound exceeds epoch-1 by ≥1.28× — a "
            "quantitative lower-bound on the excess forgetting introduced by the second epoch. "
            "If confirmed, this adds a new mechanistic figure (Figure 3: step-level forgetting "
            "proxy vs observed loss) and a quantitative §5.3 paragraph to the AAAI submission "
            "WITHOUT requiring any GPU time. The Python analysis script can run in ~1h on any "
            "CPU-only machine. Script: src/pipeline/finch_forgetting_analysis.py."
        ),
        "spec": {
            "bench": "humaneval",
            "eval_only": True,
            "script": "src/pipeline/finch_forgetting_analysis.py",
            "input_log": "results/e2e_4cyc_gpt55/cycle_1/train_log.jsonl",
            "fallback_inputs": [
                "results/e2e_4cyc_gpt55/cycle_1/llm_adapter/trainer_state.json",
                "results/e2e_4cyc_gpt55/cycle_1/sft_epoch_loss.json",
            ],
            "metric": "finch_forgetting_bound_per_step",
            "sft_lr_schedule": "cosine_warmup",
            "sft_lr_peak": 2e-4,
            "sft_warmup_steps": 10,
            "gpu_required": False,
            "gpu_hours": 0,
            "notes": (
                "Load SFT training log. For each step t, compute proxy = lr_t * sqrt(loss_t). "
                "Aggregate into per-epoch cumulative sums and per-step plots. "
                "Test: does epoch-2 cumulative proxy / epoch-1 cumulative proxy >= 1.28? "
                "Plot: step-level proxy vs step-level loss (two-axis plot), epoch boundary marked. "
                "Output: finch_analysis_cycle1.json + finch_forgetting_proxy.png. "
                "If trainer_state.json is available but not step-level LR, reconstruct LR from "
                "cosine schedule (warmup_steps=10, total_steps=N_epochs*steps_per_epoch). "
                "Implements FINCH bound from arxiv:2605.20005 §3 Theorem 1 in offline mode. "
                "100% offline — no GPU, no API call, ~1h CPU."
            ),
        },
    },
    # -----------------------------------------------------------------------
    # EXP-149: ReCo GRPO Reweighting — Distributional Concentration Fix
    #          (arxiv:2607.26862, July 2026)
    # -----------------------------------------------------------------------
    {
        "id": "exp_2026_08_03_002_reco_grpo_coverage_reweighting_distributional_concentration",
        "priority": 6,
        "gpu": "auto",
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2607.26862 (ReCo, July 2026) shows GRPO concentrates probability mass onto "
            "a shrinking subset of high-reward paths, reducing Pass@k coverage — the 'distributional "
            "concentration' effect. ReCo applies importance-ratio reweighting to underrepresented "
            "rollout groups so they contribute more gradient signal, reducing concentration without "
            "changing the reward signal. Our cycle-3 GRPO Phase 3b shows ACR=52.4% (52.4% of groups "
            "with std(reward)=0, zero-gradient Dark Room groups), the strongest signature of "
            "distributional concentration in our runs. EXP-135 (no-std-norm) disables the normalization "
            "divide-by-std that converts zero-variance groups into NaN gradients; EXP-139 "
            "(frontier-disagreement curriculum) diversifies inputs BEFORE GRPO. ReCo is an orthogonal "
            "WITHIN-GRPO remedy: it reweights training groups by inverse coverage frequency, so "
            "paths the policy has recently collapsed away from receive higher gradients. "
            "EXP-147 adds ReCo reweighting (reco_concentration_threshold=0.1) to Phase 3b at "
            "cycle 3 (worst-ACR cycle) and evaluates skills arm pass@1 vs EXP-135 baseline. "
            "Predicted: if ReCo reduces cycle-3 zero-variance groups from 52.4% to <25%, skills arm "
            "recovers 2-3pp above the current cycle-4 plateau (75.61%). Requires A800 (~2h GPU). "
            "Queue priority 6 (post-deadline; valuable for rebuttal or follow-up submission)."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 1,
            "start_cycle": 3,
            "reco_reweighting": True,
            "reco_concentration_threshold": 0.1,
            "reco_coverage_window": 200,
            "scaling_force_both": 1,
            "skip_grpo": 0,
            "skip_sft": 0,
            "grpo_temperature": 1.0,
            "gpu_hours_estimate": 2.0,
            "notes": (
                "Apply ReCo reweighting to grpo_train_simple.py advantage computation: "
                "track per-prompt rollout group coverage over a sliding window of 200 steps; "
                "upweight groups whose average advantage magnitude fell below reco_concentration_threshold "
                "in the last window. Formula: weight_g = 1 / max(coverage_g, reco_concentration_threshold). "
                "Normalize weights within each batch so total gradient scale is unchanged. "
                "Use cycle-3 SFT checkpoint (results/e2e_4cyc_gpt55/cycle_3/llm_adapter/checkpoint-best) "
                "as GRPO init (skip cycle-3 SFT — already trained). "
                "Evaluate with run_e2e_ablation_simple.py variant=skills. "
                "Implements ReCo from arxiv:2607.26862 §3.2. "
                "Apply AFTER A800 restoration; GPU window CLOSED for AAAI paper."
            ),
        },
    },
]


def main():
    state_path = Path(STATE_PATH)
    with open(state_path, "r") as f:
        state = json.load(f)

    existing_ids = {e["id"] for e in state.get("queue", [])} | {
        e.get("id", "") for e in state.get("history", [])
    }

    added = []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] in existing_ids:
            print(f"SKIP (already exists): {exp['id']}")
        else:
            state["queue"].append(exp)
            added.append(exp["id"])
            print(f"ADDED: {exp['id']}")

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(state_path.parent), suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, str(state_path))
        print(f"state.json updated. Added {len(added)} experiments: {added}")
    except Exception as e:
        os.unlink(tmp_path)
        raise e


if __name__ == "__main__":
    main()
