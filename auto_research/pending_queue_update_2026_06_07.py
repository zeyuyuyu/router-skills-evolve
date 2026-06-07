#!/usr/bin/env python3
"""
Daily queue patch — 2026-06-07 (EXP-056, EXP-057).

Run after pending_queue_update_2026_06_06.py has been applied.
Order of application on A800 when connectivity restores:

    # 1. Apply older 53 experiments from git history
    git show 649b8a4fe5f11b319ce3cdd5c205b9124028eda3:auto_research/pending_queue_update.py \
        > /tmp/pqu_full.py && python3 /tmp/pqu_full.py

    # 2. Apply 2026-06-06 patch (EXP-054, EXP-055)
    python3 auto_research/pending_queue_update_2026_06_06.py

    # 3. Apply this patch (EXP-056, EXP-057)
    python3 auto_research/pending_queue_update_2026_06_07.py

Or just run the master bootstrap which chains all patches:
    python3 auto_research/pending_queue_update.py
"""
import json, os, tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_06_07_001_lora_rank8_400tasks_cliff_test",
        "priority": 7,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2603.02224 (Subspace Geometry and Catastrophic Forgetting in LoRA, "
            "cited as rationale for EXP-053) proves forgetting severity scales inversely "
            "with the minimum principal angle (MPA) between consecutive task gradient "
            "subspaces. All prior experiments use LoRA r=16; the known 200→400-task "
            "degradation (47→46 at 400 tasks, flat binary GRPO) is consistent with "
            "subspace collapse as task batches accumulate. Lower LoRA rank (r=8) reduces "
            "the parameter subspace dimensionality, making each task batch's gradient "
            "subspace geometrically smaller and less likely to overlap with adjacent tasks "
            "— effectively widening the MPA and suppressing the overwrite mechanism. "
            "This is the first rank-ablation experiment in the queue; all 55 prior "
            "entries use r=16. Runs at 400 tasks (the known degradation point). If r=8 "
            "avoids the −1pp cliff, lower rank is a cheap regularizer that requires zero "
            "implementation overhead. Est. wall-clock ~95 min (fewer adapter params → "
            "faster steps)."
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
            "train_task_limit": 400,
            "epochs": 1,
            "rollouts_per_prompt": 4,
            "lr": 5e-6,
            "lora_r": 8,
            "prompt_style": "qwen-chat",
            "reward": "binary",
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_06_07_002_captrack_subspace_orthogonal_400tasks",
        "priority": 6,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2603.06610 (CapTrack: Capacity-Tracking LoRA for Continual Fine-Tuning, "
            "identified in project hotspot table 2026-06-05 at ★★★ alongside 2603.02224) "
            "proposes tracking LoRA subspace capacity utilization across task boundaries "
            "and adding an orthogonality penalty L_orth = captrack_coeff * ||A^T A_prev||_F "
            "that penalizes the current LoRA column space A from aligning with the previous "
            "task's subspace A_prev. This is the direct algorithmic intervention for the "
            "mechanism diagnosed by arxiv:2603.02224 and being measured by EXP-053 "
            "(forgetting_eval with HumanEval + MPA probe): if EXP-053 confirms H1 "
            "(catastrophic forgetting via subspace collapse), CapTrack is the prescribed "
            "fix. Run at 400 tasks to test whether the orthogonality penalty preserves "
            "the +2pp gain while avoiding the −1pp degradation. Priority 6 (below EXP-056) "
            "because CapTrack requires a new regularization term in the GRPO loss vs "
            "EXP-056's single integer hyperparameter change. Est. wall-clock ~115 min "
            "(orthogonality term adds one extra matmul per gradient step)."
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
            "train_task_limit": 400,
            "epochs": 1,
            "rollouts_per_prompt": 4,
            "lr": 5e-6,
            "lora_r": 16,
            "prompt_style": "qwen-chat",
            "reward": "binary",
            "lora_regularization": "captrack",
            "captrack_orthogonality_coeff": 0.01,
            "captrack_task_boundary_interval": 50,
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
