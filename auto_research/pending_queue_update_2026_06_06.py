#!/usr/bin/env python3
"""
Daily queue patch — 2026-06-06 (EXP-054, EXP-055).
Run this AFTER restoring pending_queue_update.py from git history:
    git checkout 649b8a4fe5f11b319ce3cdd5c205b9124028eda3 -- auto_research/pending_queue_update.py
    python3 auto_research/pending_queue_update.py  # apply all 53 older experiments first
    python3 auto_research/pending_queue_update_2026_06_06.py  # then apply today's 2 new ones

Or run directly against state.json if the older experiments are already applied.
"""
import json, os, tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_06_06_001_egca_execution_grounded_credit_15b",
        "priority": 9,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2603.16158 (EGCA, ICLR 2026 SPOT Workshop) shows that GRPO coarse "
            "credit assignment spreads a single binary outcome uniformly across all tokens. "
            "EGCA localises the GRPO update via execution traces: runs both candidate and "
            "canonical MBPP solution under identical statement-level tracing, identifies "
            "earliest runtime-state divergence, assigns advantage only to that token span. "
            "Results: +3.1pp HumanEval, +1.5pp MBPP vs vanilla GRPO; 18% wall-clock overhead. "
            "Orthogonal to EXP-052 (GRPO-lambda temporal discounting): EGCA is semantic and "
            "execution-grounded, not positional. Uses MBPP canonical_solution field -- no "
            "new data required. Est. wall-clock ~105 min."
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
            "credit_assignment": "execution_grounded",
            "egca_divergence_granularity": "statement",
            "egca_canonical_source": "mbpp_canonical_solutions",
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_06_06_002_sa_ah_grpo_entropy_adaptive_horizon_15b",
        "priority": 7,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2606.05434 (SA-AH-GRPO, June 2026) introduces entropy-adaptive horizon "
            "discounting for GRPO: weights each token's policy gradient by cumulative "
            "entropy discount delta_t = prod_{k<=t} exp(-beta*H(pi(.|s_k))), applied "
            "asymmetrically ONLY to negative-advantage (failing) rollouts. Positive-advantage "
            "trajectories are unattenuated. On Qwen2.5-1.5B LoRA GSM8K: peak Pass@1=0.686 "
            "vs zero-shot 0.637 (+4.9pp); training variance reduced 3.6x. Distinct from "
            "EXP-052 (GRPO-lambda, fixed temporal decay) -- this uses model entropy as "
            "data-adaptive discount signal. Distinct from EXP-013 (DAPO, drops groups) -- "
            "SA-AH-GRPO retains all rollouts but attenuates token-level gradient. "
            "Entropy computed from existing forward-pass logits: zero extra LLM calls. "
            "Caveat: paper evaluates on math (GSM8K); first code-generation test. "
            "Est. wall-clock ~95 min."
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
            "advantage_mode": "selective_entropy_adaptive_horizon",
            "entropy_discount_base": 0.9,
            "selective_advantage_polarity": "negative_only",
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
