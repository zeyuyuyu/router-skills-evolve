#!/usr/bin/env python3
"""
Weekly paper pipeline queue patch — 2026-07-10 (EXP-110, EXP-111).

A800 connectivity: offline since 2026-05-14 (day 57). Apply when restored.
RUSH MODE ACTIVATED: AAAI deadline 2026-08-15, 36 days remaining.

Prior patches to apply before this one (see pending_queue_update_2026_07_05.py for full chain):
    ...
    python3 auto_research/pending_queue_update_2026_07_05.py            # EXP-108, EXP-109
Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_10_paper.py      # EXP-110, EXP-111

Queue was ~115 pending before this run.
"""

import json
import pathlib

STATE_PATH = pathlib.Path("/data0/home/zeyuwang/auto_research/state.json")

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_07_10_paper_001_sign_advantage_grpo_humaneval_4cycle",
        "kind": "grpo_continual",
        "priority": 9,
        "rationale": (
            "W4: arxiv:2605.07689 sign-advantage (A=2r-1) eliminates zero-advantage groups "
            "for binary reward (+45pp GSM8K, 7 seeds). Our 43/82 (52.4%) zero-variance "
            "HumanEval groups would all receive non-zero gradient. EXP-108 tests MBPP "
            "standalone; this tests it in the 4-cycle MERA pipeline where the bottleneck "
            "was originally identified. If skills arm >78%, it changes the headline result."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 4,
            "grpo_advantage": "sign",
            "grpo_advantage_formula": "A = 2*r - 1",
            "grpo_rollouts": 8,
            "grpo_temperature": 1.0,
            "dapo_dynamic_sampling": True,
            "lora_rank": 16,
            "lr": 5e-6,
            "small_model": "Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "gpt-5.5",
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 skills arm: 75.61% HE pass@1 (standard DAPO)",
                "EXP-108: sign-advantage standalone MBPP",
            ],
            "primary_metrics": [
                "humaneval_pass_at_1_skills_arm_post_grpo",
                "zero_variance_group_rate",
                "grpo_informative_group_count",
            ],
            "target_outcome": (
                "Skills arm pass@1 >= 78% (+2.4pp from 75.61%). "
                "Zero-variance group rate drops below 20% (from 52.4% with standard DAPO). "
                "If achieved, replaces the zero-variance finding's punchline: not just a "
                "bottleneck identified, but one solved within this paper."
            ),
            "estimated_gpu_hours": 4.0,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_07_10_paper_002_tau2_sft_to_rft_adapter",
        "kind": "grpo_continual",
        "priority": 8,
        "rationale": (
            "W6: tau2 cycle oscillation (89.19%->70.27%->72.97%) attributed to SFT-driven "
            "catastrophic forgetting (arxiv:2507.05386). Converting tau2 adapter training "
            "from SFT to GRPO/RFT should stabilize cycles. Directly tests the theoretical "
            "SFT-vs-RFT forgetting hypothesis on the agentic multi-turn setting."
        ),
        "spec": {
            "bench": "tau2_bench",
            "n_cycles": 4,
            "adapter_training": "grpo",
            "grpo_temperature": 0.9,
            "grpo_rollouts": 4,
            "small_model": "Qwen3.6-35B-A3B",
            "large_model": "gpt-5.4",
            "compare_with": [
                "tau2 joint SFT run cycles 0-3: 78.38%->89.19%->70.27%->72.97%",
            ],
            "primary_metrics": [
                "tau2_task_pass_per_cycle",
                "cycle_to_cycle_delta",
                "tau2_task_pass_final_cycle",
            ],
            "target_outcome": (
                "Monotonic or near-monotonic improvement over 4 cycles (no regression >5pp). "
                "Final cycle task pass >= 85% (vs 72.97% SFT final cycle). "
                "If achieved: validates SFT-forgetting hypothesis empirically AND provides "
                "a better agentic MERA result for the paper."
            ),
            "note": (
                "Requires tau2-bench multi-turn GRPO rollout harness. Check if existing "
                "grpo_train_simple.py supports multi-turn agent trajectories or if "
                "tau2 step-budget harness needs extension. May need ~2h engineering."
            ),
            "estimated_gpu_hours": 16.0,
        },
        "gpu": "auto",
    },
]


def main() -> None:
    if not STATE_PATH.exists():
        print(f"ERROR: {STATE_PATH} not found. Is the A800 mounted?")
        raise SystemExit(1)

    with STATE_PATH.open() as f:
        state = json.load(f)

    existing_ids = {e["id"] for e in state.get("queue", [])}
    existing_ids |= {e["id"] for e in state.get("history", [])}

    added = []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] in existing_ids:
            print(f"  SKIP (duplicate): {exp['id']}")
            continue
        state["queue"].append(exp)
        added.append(exp["id"])
        print(f"  QUEUED: {exp['id']}  (priority={exp['priority']}, "
              f"kind={exp['kind']})")

    if not added:
        print("No new experiments added (all duplicates).")
        return

    tmp = STATE_PATH.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(STATE_PATH)
    print(f"\nSaved {STATE_PATH}. Added {len(added)} experiment(s): {added}")


if __name__ == "__main__":
    main()
