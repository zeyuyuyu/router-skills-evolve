#!/usr/bin/env python3
"""
Daily queue patch — 2026-06-08 (EXP-058, EXP-059).

Run after all previous patches have been applied.
Order of application on A800 when connectivity restores:

    # 1. Apply older 53 experiments from git history
    git show 649b8a4fe5f11b319ce3cdd5c205b9124028eda3:auto_research/pending_queue_update.py \
        > /tmp/pqu_full.py && python3 /tmp/pqu_full.py

    # 2. Apply 2026-06-06 patch (EXP-054, EXP-055)
    python3 auto_research/pending_queue_update_2026_06_06.py

    # 3. Apply 2026-06-07 patch (EXP-056, EXP-057)
    python3 auto_research/pending_queue_update_2026_06_07.py

    # 4. Apply this patch (EXP-058, EXP-059)
    python3 auto_research/pending_queue_update_2026_06_08.py

Or just run the master bootstrap which chains all patches:
    python3 auto_research/pending_queue_update.py
"""
import json, os, tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_06_08_001_mixed_domain_mbpp_humaneval_200tasks",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "The 2026-06-07 state report lists 'Is the 400-task degradation catastrophic "
            "forgetting or reward hacking/MBPP overfitting?' as an explicit open question. "
            "EXP-053 (forgetting_eval + HumanEval checkpoint probe) addresses this "
            "diagnostically from the evaluation side. This experiment tests the hypothesis "
            "from the training side: all 57 queued experiments train exclusively on the "
            "MBPP aug dataset. If the model is learning MBPP-specific output patterns "
            "(format, test structure, function signature conventions) rather than "
            "generalizable code generation, adding 50% HumanEval problems to the training "
            "pool should change MBPP eval100 accuracy in a directional way: (a) if mixed "
            "training maintains or improves MBPP eval100, MBPP-specific overfitting is not "
            "the dominant mechanism and the +2 pp ceiling is more fundamental; (b) if mixed "
            "training degrades MBPP eval100, MBPP-specific pattern reinforcement was "
            "contributing to the gain and the ceiling is a distribution-matching artefact. "
            "arxiv:2605.12484 ('Learning, Fast and Slow', May 2026) identifies training "
            "distribution diversity as a key factor in reducing plasticity loss during "
            "continual RL adaptation — narrow-domain repeated training accelerates entropy "
            "collapse by iterating over the same output-pattern manifold. Mixed-domain "
            "training introduces per-step distributional variety, keeping effective task "
            "manifold coverage wide. Orthogonal to subspace regularization (EXP-056/057) "
            "and credit assignment (EXP-052/054/055) — operates at the data level with no "
            "algorithmic change to GRPO. Implementation: interleave MBPP aug training JSONL "
            "with HumanEval train problems at 1:1 ratio, shuffle, train_task_limit=200 draws "
            "from the mixed pool. Requires creating the mixed JSONL (2-line shell command) "
            "or extending the runner to accept train_data_supplemental + "
            "train_data_mix_ratio. No extra LLM calls. Est. wall-clock ~65 min."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": (
                "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/"
                "train_aug_excluding_eval20.jsonl"
            ),
            "train_data_supplemental": (
                "/data0/home/zeyuwang/router-skills-evolve-data/humaneval/"
                "train164.jsonl"
            ),
            "train_data_mix_ratio": 0.5,
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
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_06_08_002_length_normalised_reward_grpo_200tasks",
        "priority": 7,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2503.04548 ('Length-Normalised Reward for GRPO in Code Generation', "
            "March 2025) was identified in the 2026-05-28 arxiv sweep as relevant "
            "('Moderate') but has not been queued in any of the 57 subsequent experiments. "
            "All 57 experiments use reward='binary': a flat +1.0 for test-pass regardless "
            "of solution length. For MBPP problems, Qwen2.5-Coder-1.5B generates solutions "
            "spanning 15 to 120+ tokens. Without length normalisation, GRPO assigns "
            "identical advantage to a 15-token elegant one-liner and a 120-token verbose "
            "solution that both pass — the policy receives no gradient signal to prefer "
            "concise, direct code. The paper proposes dividing the binary pass reward by "
            "log(token_count + 1): reward_norm = binary / log(token_count + 1). For a "
            "passing 15-token response: 1.0 / log(16) = 0.36; for a passing 120-token "
            "response: 1.0 / log(121) = 0.21. This creates a preference gradient toward "
            "shorter passing solutions. Failing solutions receive 0.0 regardless of length "
            "(no change to negative-advantage gradient). The length penalty sharpens the "
            "reward signal on the positive side only. Paper reports +1.5 to +2.5 pp on "
            "MBPP/HumanEval with Qwen2.5-Coder models using this normalisation. Distinct "
            "from EXP-009 (P-GRPO partial reward, demoted per arxiv:2605.02944 which showed "
            "fractional rewards for partial code correctness hurt vs binary): length "
            "normalisation does not change the binary pass/fail threshold or introduce "
            "intermediate reward levels — it only rescales the magnitude of positive "
            "rewards by response length, keeping all rewards in {0, (0.0, 0.36]} range. "
            "No prior experiment in the 57-experiment queue modifies the magnitude of "
            "positive-outcome rollout rewards. Est. wall-clock ~58 min."
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
            "reward": "binary_length_normalised",
            "length_normalise_base": "log",
            "length_normalise_denominator": "response_token_count",
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
