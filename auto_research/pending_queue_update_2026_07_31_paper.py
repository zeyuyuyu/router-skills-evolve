#!/usr/bin/env python3
"""
Weekly paper pipeline queue patch — 2026-07-31 (EXP-142, EXP-143).

AAAI 2027 deadline: 2026-08-15 (15 days).
A800 offline since 2026-05-14 (day 78).
GPU window: CLOSED (Aug 1 was the last viable day; apply immediately on A800 restore).

This patch adds 2 experiments from the weekly paper pipeline INJECTOR pass.
Queue was ~147 pending before this patch (after EXP-140, EXP-141 from daily patches).

New findings motivating this run:
- Offline analysis confirms Hypothesis F: cycle-1 SFT epoch-2 geometry conflict
  (overfit_gap=+0.105, epoch-2 loss 0.166→0.271, entropy regression 0.255→0.207).
  Epoch-1 checkpoint (checkpoint-6) is SAVED LOCALLY — EXP-142 evaluates it immediately.
- Dark Room effect (arxiv:2607.21273): GRPO's σ→0 amplifies noise to O(1) spurious gradients,
  meaning ACR groups may actively HARM training (not merely contribute zero gradient).
  EXP-143 quantifies the spurious gradient magnitude in our cycle-3 GRPO run from existing logs.

Apply when A800 is restored:
    python3 auto_research/pending_queue_update_2026_07_31_paper.py  # EXP-142, EXP-143
"""

import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = "data/humaneval_eval.jsonl"

NEW_EXPERIMENTS = [
    # -----------------------------------------------------------------------
    # EXP-142: Hypothesis F Validation — Epoch-1 SFT Checkpoint Skill Eval
    # Addresses: W3 (Full=Router explanation, mechanism H-F), W6 (cycle-1 dip)
    # GPU required: ~0.5h (eval only, no training)
    # Notes: checkpoint-6 is ALREADY SAVED locally at
    #        results/e2e_4cyc_gpt55/cycle_1/llm_adapter/checkpoint-6/
    #        → runnable on ANY system with Qwen3-35B-A3B, not just A800
    # -----------------------------------------------------------------------
    {
        "id": "exp_2026_07_31_paper_001_hypothesis_f_epoch1_checkpoint_eval",
        "kind": "forgetting_eval",
        "priority": 9,
        "gpu_hours": 0.5,
        "status": "pending",
        "rationale": (
            "Offline analysis confirms Hypothesis F: cycle-1 SFT epoch-2 loss increases "
            "from 0.166 to 0.271 (overfit_gap=+0.105), the only cycle with positive overfit_gap, "
            "coinciding with the only cycle-1 skills arm dip (65.85% vs 70.73%). "
            "The epoch-1 SFT checkpoint (checkpoint-6) is saved at "
            "results/e2e_4cyc_gpt55/cycle_1/llm_adapter/checkpoint-6/ in the local repo. "
            "EXP-142 runs the skills arm eval directly on checkpoint-6 and compares to "
            "checkpoint-12 (65.85%) — if checkpoint-6 yields ≥70% skills arm, Hypothesis F "
            "is confirmed and epoch-1 selection recovers the dip with zero algorithm change. "
            "Addresses W3 (cycle-1 dip mechanism) and W6 (skills gap explanation). "
            "Refs: arxiv:2606.18487 (SFT overtraining), arxiv:2605.09608 (geometry conflict)."
        ),
        "spec": {
            "experiment_name": "hypothesis_f_epoch1_checkpoint_eval",
            "description": "Eval skills arm on cycle-1 epoch-1 SFT checkpoint (checkpoint-6) vs epoch-2 (checkpoint-12)",
            "checkpoint_epoch1": "results/e2e_4cyc_gpt55/cycle_1/llm_adapter/checkpoint-6",
            "checkpoint_epoch2": "results/e2e_4cyc_gpt55/cycle_1/llm_adapter/checkpoint-12",
            "eval_script": "src/pipeline/run_e2e_ablation_simple.py",
            "eval_benchmark": "humaneval",
            "eval_mode": "skills_arm_only",
            "variants": ["epoch1_ckpt", "epoch2_ckpt"],
            "expected_result": {
                "epoch1_skills_arm": "≥70%  (Hypothesis F confirmed if ≥70%)",
                "epoch2_skills_arm": "65.85% (known baseline)",
                "delta": "+4.88pp predicted if H-F holds"
            },
            "aaai_impact": "If confirmed: adds concrete fix for cycle-1 dip to §5.3; "
                           "strengthens Hypothesis F from 'offline-confirmed' to 'experimentally validated'; "
                           "adds row to Table tab:he4cyc footnote; closes W6 completely.",
            "fallback_if_no_a800": "Run on any system with Qwen3-35B-A3B model; "
                                    "checkpoint-6 available locally; ~0.5h on a single GPU"
        }
    },
    # -----------------------------------------------------------------------
    # EXP-143: Dark Room Spurious Gradient Audit — Offline Analysis
    # Addresses: W1 (zero-gradient interpretation update), §7 Dark Room paragraph
    # GPU required: 0h (offline analysis of existing GRPO logs)
    # -----------------------------------------------------------------------
    {
        "id": "exp_2026_07_31_paper_002_dark_room_spurious_gradient_audit",
        "kind": "analysis_offline",
        "priority": 7,
        "gpu_hours": 0.0,
        "status": "pending",
        "rationale": (
            "The Dark Room effect (arxiv:2607.21273) reveals that GRPO's σ→0 amplifies "
            "finite-precision residual variance (σ≈ε≈1e-6) to O(1) spurious advantages. "
            "Our cycle-3 GRPO log shows 43/82 groups as 'zero-variance'. "
            "Under standard GRPO (without DAPO σ=0 drop), these groups would produce "
            "spurious gradient ≈ (r_i - mean) / ε for each token in each rollout. "
            "EXP-143 computes the expected spurious gradient magnitude from our existing "
            "logs (per-group reward statistics in results/e2e_4cyc_gpt55/cycle_3/phase3b_grpo.log) "
            "and estimates the effective learning-rate inflation caused by spurious updates. "
            "This quantifies whether DAPO's σ=0 filter is the critical safeguard in our "
            "pipeline vs standard GRPO without DAPO. "
            "Zero GPU hours; uses existing log data. Adds quantitative backing for §7 "
            "Dark Room paragraph and Proposition 1."
        ),
        "spec": {
            "experiment_name": "dark_room_spurious_gradient_audit",
            "description": "Offline audit: quantify spurious gradient magnitude in cycle-3 GRPO zero-variance groups",
            "input_log": "results/e2e_4cyc_gpt55/cycle_3/phase3b_grpo.log",
            "analysis_script": "auto_research/analyze_dark_room_gradients.py",  # to be created
            "outputs": [
                "Per-group sigma values for all 43 ACR groups",
                "Estimated spurious advantage magnitude under standard GRPO (no DAPO)",
                "Ratio of spurious to informative gradient magnitude",
                "Effective LR inflation factor due to σ→0 amplification"
            ],
            "aaai_impact": "Adds quantitative evidence to Proposition 1 and §7 Dark Room paragraph; "
                           "justifies DAPO's σ=0 filter as a critical safety mechanism, not just a 'remove silent groups' trick.",
            "implementation": {
                "approach": "Parse GRPO log for per-group reward std estimates; "
                             "compute A = (r - mean) / sigma for near-zero-sigma groups; "
                             "compare to A = (r - mean) for no-std-norm groups",
                "estimated_time": "2 hours implementation + 0h runtime"
            }
        }
    }
]


def main():
    if not STATE_PATH.exists():
        print(f"state.json not found at {STATE_PATH} — A800 offline; saving patch for later.")
        patch_path = Path(__file__).parent / "pending_queue_update_2026_07_31_paper_offline.json"
        patch_path.write_text(json.dumps(NEW_EXPERIMENTS, indent=2))
        print(f"Saved {len(NEW_EXPERIMENTS)} experiments to {patch_path}")
        return

    with open(STATE_PATH) as f:
        state = json.load(f)

    queue = state.get("experiment_queue", [])
    existing_ids = {e["id"] for e in queue}

    added = 0
    for exp in NEW_EXPERIMENTS:
        if exp["id"] not in existing_ids:
            queue.append(exp)
            added += 1
            print(f"  + Queued {exp['id']} (priority={exp['priority']})")
        else:
            print(f"  = Already present: {exp['id']}")

    state["experiment_queue"] = queue

    with tempfile.NamedTemporaryFile(
        mode="w", dir=STATE_PATH.parent, suffix=".tmp", delete=False
    ) as f:
        json.dump(state, f, indent=2)
        tmp = f.name

    os.replace(tmp, STATE_PATH)
    print(f"\nDone. Added {added} experiments. Queue size: {len(queue)}")


if __name__ == "__main__":
    main()
