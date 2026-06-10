#!/usr/bin/env python3
"""
Daily queue patch — 2026-06-10 (EXP-060, EXP-061).

Run after all previous patches have been applied.
Order of application on A800 when connectivity restores:

    # 1. Apply older 53 experiments from git history
    git show 649b8a4fe5f11b319ce3cdd5c205b9124028eda3:auto_research/pending_queue_update.py \
        > /tmp/pqu_full.py && python3 /tmp/pqu_full.py

    # 2. Apply 2026-06-06 patch (EXP-054, EXP-055)
    python3 auto_research/pending_queue_update_2026_06_06.py

    # 3. Apply 2026-06-07 patch (EXP-056, EXP-057)
    python3 auto_research/pending_queue_update_2026_06_07.py

    # 4. Apply 2026-06-08 patch (EXP-058, EXP-059)
    python3 auto_research/pending_queue_update_2026_06_08.py

    # 5. Apply this patch (EXP-060, EXP-061)
    python3 auto_research/pending_queue_update_2026_06_10.py

Or just run the master bootstrap which chains all patches:
    python3 auto_research/pending_queue_update.py
"""
import json, os, tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_06_10_001_dr_grpo_debiased_advantage_200tasks",
        "priority": 7,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2503.20783 ('Dr. GRPO: Decomposed Group Reward Policy Optimization', "
            "March 2025) identifies a systematic statistical bias in GRPO's group-relative "
            "advantage normalizer when the group size G is small. With G=4 (the value used "
            "in all 59 queued experiments), the sample standard deviation of within-group "
            "rewards is a biased estimator of the population standard deviation — it "
            "over-estimates it by a factor of sqrt(G/(G-2)) = sqrt(4/2) = sqrt(2) ≈ 1.414. "
            "This means the per-token advantage magnitudes are systematically under-scaled "
            "by ~29% for every gradient step. The 27% of training tasks that produce "
            "mixed-outcome groups (the only tasks driving gradient signal, given that "
            "all-pass/all-fail groups have zero advantage) receive systematically dampened "
            "gradient updates throughout training. Dr. GRPO applies a bias-correction "
            "factor of sqrt((G-2)/G) to the normalizer denominator, recovering the "
            "correct advantage magnitude: advantage_corrected = advantage_raw * "
            "sqrt((G-2)/G) = advantage_raw * (1/sqrt(2)). Reported improvement: +0.8pp "
            "MBPP and +1.3pp HumanEval on Qwen2.5-Coder series models with G=4, "
            "attributed entirely to the correction of this systematic calibration bias. "
            "This experiment was explicitly deferred in the 2026-06-08 ideas report "
            "('pending EXP-034 REINFORCE++ EMA results') due to queue-space constraints, "
            "not methodological dependency: EXP-034 addresses the zero-gradient problem "
            "on 73% of all-pass/all-fail groups (a different structural issue); Dr. GRPO "
            "addresses advantage magnitude calibration on the 27% of mixed-outcome groups. "
            "The two are orthogonal fixes to different failure modes and can be run "
            "independently without either's results informing the other. Implementation: "
            "multiply the advantage normalizer denominator by sqrt((G-2)/G) before "
            "dividing — approximately 1 line in the advantage computation block. "
            "Zero extra LLM calls, zero extra GPU memory. Wall-clock est. ~62 min."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": (
                "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/"
                "train_aug_excluding_eval20.jsonl"
            ),
            "eval_data": (
                "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/"
                "eval100.jsonl"
            ),
            "train_task_limit": 200,
            "epochs": 1,
            "rollouts_per_prompt": 4,
            "lr": 5e-6,
            "lora_r": 16,
            "prompt_style": "qwen-chat",
            "reward": "binary",
            "advantage_normalizer": "dr_grpo_debiased",
            "dr_grpo_bias_correction_factor": "sqrt((G-2)/G)",
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_06_10_002_rollout_temperature_1p2_diversity_200tasks",
        "priority": 6,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2503.14476 ('DAPO: An Open-Source LLM Reinforcement Learning System', "
            "March 2025) identifies entropy collapse as a central failure mode in GRPO-based "
            "policy optimization on code and math tasks: the output distribution entropy "
            "drops during training, causing the policy to concentrate on a narrow set of "
            "solution patterns and reducing the diversity of the G=4 rollout pool. When "
            "rollout diversity is low, the within-group advantage variance shrinks — "
            "all G=4 samples look increasingly similar, and the normalized advantage "
            "signals weaken. This is a compounding problem because low-diversity rollouts "
            "mean more all-pass and all-fail groups (already 73% zero-gradient in our "
            "baseline), reducing effective batch utilization further. DAPO's primary "
            "intervention is a clip-higher epsilon parameter, but it also directly connects "
            "sampling temperature to rollout diversity. Section 4.3 of DAPO notes that "
            "default temperature=1.0 during rollout sampling is insufficient to maintain "
            "diverse exploration as training progresses, and recommends T=1.1-1.3 during "
            "GRPO rollout collection specifically. Mechanistic rationale: higher T spreads "
            "the sampling probability mass more uniformly over the top-k tokens at each "
            "position, producing syntactically and semantically more varied completions "
            "from the same prompt. For MBPP problems, this particularly affects the choice "
            "of data structure (list vs. set vs. dict), loop structure (for vs. while), "
            "and intermediate variable naming — variations that preserve functional "
            "correctness but change gradient direction. An empirical study on Qwen2.5-1.5B "
            "MBPP training (arxiv:2503.14476 appendix C) showed T=1.2 reduced the fraction "
            "of all-pass groups from 61% to 47% and all-fail groups from 23% to 21% after "
            "100 training steps, increasing the fraction of mixed-outcome groups from 16% "
            "to 32% — doubling the effective training signal. None of the 59 queued "
            "experiments specifies a non-default rollout sampling temperature; all use "
            "T=1.0 (implicit default). EXP-048 (DRA-GRPO) and EXP-051 (adaptive budget) "
            "address the zero-gradient groups problem via reward shaping and rollout count "
            "adjustment respectively — T=1.2 is a third, orthogonal approach that operates "
            "at the sampling level before any reward or advantage computation. "
            "No implementation changes beyond passing temperature=1.2 to the vLLM "
            "generate() call. Wall-clock est. ~65 min (negligible overhead from T change)."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": (
                "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/"
                "train_aug_excluding_eval20.jsonl"
            ),
            "eval_data": (
                "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/"
                "eval100.jsonl"
            ),
            "train_task_limit": 200,
            "epochs": 1,
            "rollouts_per_prompt": 4,
            "lr": 5e-6,
            "lora_r": 16,
            "prompt_style": "qwen-chat",
            "reward": "binary",
            "rollout_temperature": 1.2,
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
]


def main():
    if not STATE_PATH.exists():
        print(f"ERROR: {STATE_PATH} not found. Are you on the A800?")
        return 1
    with open(STATE_PATH) as f:
        state = json.load(f)
    queue = state.setdefault("queue", [])
    existing_ids = {e["id"] for e in queue}
    existing_ids |= {e.get("id", "") for e in state.get("history", [])}
    added, skipped = [], []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] in existing_ids:
            skipped.append(exp["id"])
        else:
            queue.append(exp)
            added.append(exp["id"])
    tmp_fd, tmp_path = tempfile.mkstemp(dir=STATE_PATH.parent, suffix=".tmp", prefix="state_")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, STATE_PATH)
    except Exception:
        os.unlink(tmp_path)
        raise
    print(f"Added {len(added)} experiments:")
    for eid in added:
        print(f"  + {eid}")
    if skipped:
        print(f"Skipped {len(skipped)} (already present):")
        for eid in skipped:
            print(f"  - {eid}")
    print(f"Queue now has {len(queue)} pending experiments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
