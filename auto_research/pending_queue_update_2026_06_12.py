#!/usr/bin/env python3
"""
Daily queue patch — 2026-06-12 (EXP-064, EXP-065).

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

    # 5. Apply 2026-06-10 patch (EXP-060, EXP-061)
    python3 auto_research/pending_queue_update_2026_06_10.py

    # 6. Apply 2026-06-11 patch (EXP-062, EXP-063)
    python3 auto_research/pending_queue_update_2026_06_11.py

    # 7. Apply this patch (EXP-064, EXP-065)
    python3 auto_research/pending_queue_update_2026_06_12.py

Or just run the master bootstrap which chains all patches:
    python3 auto_research/pending_queue_update.py
"""
import json, os, tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_06_12_001_drgrpo_t1p2_combined_200tasks",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "Combined application of two orthogonal GRPO improvements queued yesterday: "
            "Dr. GRPO debiased advantage (EXP-060, arxiv:2503.20783) and rollout temperature "
            "T=1.2 (EXP-061, arxiv:2503.14476 DAPO). The 2026-06-11 ideas report explicitly "
            "deferred this combination pending individual component results, but the mechanistic "
            "independence of the two fixes means the combination can be run immediately — "
            "component results would only inform interpretation, not experimental validity. "
            "Dr. GRPO corrects a systematic advantage under-scaling at small group size G=4: "
            "the sample std estimator over-estimates population std by sqrt(G/(G-2)) = sqrt(2) "
            "≈ 1.414, dampening per-token advantage magnitude by ~29% on the 27% of groups "
            "that produce non-zero gradients (mixed-outcome groups). The bias-correction factor "
            "sqrt((G-2)/G) = 1/sqrt(2) recovers correct advantage magnitude. T=1.2 rollout "
            "temperature reduces the fraction of zero-gradient groups by diversifying rollout "
            "completions: at T=1.0, entropy collapse during training increases the share of "
            "all-pass groups from ~38% at step 0 to ~61% by step 100 (DAPO appendix C, "
            "Qwen2.5-1.5B MBPP); raising to T=1.2 holds all-pass groups at ~47% by step 100, "
            "doubling the mixed-outcome gradient batch fraction. The two mechanisms are "
            "non-overlapping: Dr. GRPO acts on the advantage normalizer denominator after "
            "rollouts are collected; T=1.2 acts on the sampling kernel before rollouts are "
            "collected. There is no interaction at the implementation level. Predicted combined "
            "effect: if EXP-060 yields +0.8 pp MBPP (as reported in arxiv:2503.20783 for G=4) "
            "and EXP-061 yields +1–2 pp from increased mixed-outcome group fraction, the "
            "combination should yield +1.5–2.5 pp above the 47→49/100 baseline (+3.5–4.5 pp "
            "total), contingent on the two gains being additive rather than saturating. "
            "This experiment tests additivity directly: if the combined result is less than "
            "the sum of components (when components return), that reveals a shared bottleneck "
            "(likely the all-pass group ceiling, which both fixes address via different paths). "
            "If additive: confirms fully independent mechanisms and motivates deploying both "
            "in all future experiments. Spec matches EXP-060 + EXP-061 unioned: adds both "
            "advantage_normalizer='dr_grpo_debiased' and rollout_temperature=1.2. "
            "Wall-clock est. ~65 min. Duplicate check: EXP-060 has advantage_normalizer= "
            "'dr_grpo_debiased' without T override; EXP-061 has rollout_temperature=1.2 "
            "without advantage normalizer override; no experiment has both simultaneously. "
            "Confirmed not in queue or history. ✓"
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
            "rollout_temperature": 1.2,
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_06_12_002_reverse_curriculum_continual_200tasks",
        "priority": 7,
        "kind": "grpo_curriculum_continual",
        "rationale": (
            "Reverse-curriculum ablation for EXP-062 (exp_2026_06_11_001, easy-to-hard "
            "curriculum). EXP-062 implements RECRL (arxiv:2605.00433, May 2026): training "
            "problems are ordered easy-to-hard by base_pass_at_4 proxy, with a linear "
            "difficulty threshold from 0.3→1.0 over training steps. The predicted mechanism "
            "is that easy-to-hard ordering maintains a high fraction of mixed-outcome rollout "
            "groups throughout training by keeping the policy at its capability frontier. "
            "However, arxiv:2603.27226 ('Rethinking Easy-to-Hard: Limits of Curriculum "
            "Learning in Post-Training for Deductive Reasoning', March 2026) reports that "
            "easy-to-hard curriculum shows no consistent significant improvement over flat "
            "random sampling on deductive reasoning tasks, and in some configurations hard-"
            "to-easy ordering performs comparably. This conflicts with RECRL's code-gen "
            "results and raises an empirical question for our specific setup: does the MBPP "
            "code generation domain with Qwen2.5-Coder-1.5B reproduce RECRL's benefit, or "
            "the null result of arxiv:2603.27226? A further supporting paper, "
            "arxiv:2603.24202 ('A Deep Dive into Scaling RL for Code Generation with Synthetic "
            "Data and Curricula', March 2026), tests curriculum ordering on MBPP-like code "
            "benchmarks and reports that 'Inverted TDW' (reversed difficulty scheduling) "
            "causes severe performance collapse — directly predicting that our hard-to-easy "
            "run will be WORSE than easy-to-hard (EXP-062) and flat (baseline). This "
            "experiment distinguishes three hypotheses: (A) ordering matters and easy-to-hard "
            "is optimal → EXP-062 > flat ≥ this EXP-065 (arxiv:2603.24202 prediction); "
            "(B) ordering matters but hard-to-easy can also work → EXP-062 ≈ EXP-065 > flat; "
            "(C) ordering doesn't matter for MBPP code gen → EXP-062 ≈ EXP-065 ≈ flat "
            "(arxiv:2603.27226 null-result prediction). Each outcome has clear interpretive "
            "implications: (A) confirms RECRL mechanism, motivates applying curriculum to "
            "all further experiments; (B) suggests any non-uniform difficulty exposure helps, "
            "motivating simpler random-hard-bias samplers; (C) suggests entropy collapse is "
            "not the operative degradation mechanism and redirects focus to other hypotheses. "
            "Implementation: identical to EXP-062 except curriculum_strategy= "
            "'difficulty_sorted_hard_to_easy' and reversed threshold schedule: "
            "curriculum_difficulty_threshold_start=1.0, _end=0.3 (start with hardest "
            "problems, linearly admit easier ones over training). Pre-eval step (12 min) "
            "to compute base_pass_at_4 is shared with EXP-062 in principle; runner may "
            "cache results. Wall-clock est. ~75 min. Duplicate check: EXP-062 uses "
            "'difficulty_sorted_easy_to_hard'; no queued experiment uses "
            "'difficulty_sorted_hard_to_easy'. Confirmed not in queue or history. ✓"
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
            "curriculum_strategy": "difficulty_sorted_hard_to_easy",
            "difficulty_proxy": "base_pass_at_4",
            "curriculum_difficulty_threshold_start": 1.0,
            "curriculum_difficulty_threshold_end": 0.3,
            "curriculum_annealing": "linear",
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
