#!/usr/bin/env python3
"""
Daily queue patch — 2026-06-11 (EXP-062, EXP-063).

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

    # 6. Apply this patch (EXP-062, EXP-063)
    python3 auto_research/pending_queue_update_2026_06_11.py

Or just run the master bootstrap which chains all patches:
    python3 auto_research/pending_queue_update.py
"""
import json, os, tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_06_11_001_recrl_curriculum_continual_200tasks",
        "priority": 8,
        "kind": "grpo_curriculum_continual",
        "rationale": (
            "arxiv:2605.00433 ('Improving LLM Code Generation via Requirement-Aware "
            "Curriculum Reinforcement Learning', May 2026) introduces RECRL: a curriculum "
            "RL framework that estimates per-problem difficulty as the fraction of base "
            "model rollouts that fail (difficulty_proxy = 1 - pass@G), then constructs "
            "training batches that expose the model to problems in a smoothed easy-to-hard "
            "order (linear annealing of difficulty percentile threshold over training). "
            "Experiments on Qwen2.5-Coder-1.5B across MBPP, MBPP+, HumanEval, HumanEval+, "
            "and LiveCodeBench show average Pass@1 improvement of +1.23%–5.62% over "
            "standard GRPO (flat ordering) on the same task sets. The mechanistic argument "
            "is consistent with arxiv:2605.12484 (Learning Fast and Slow): when the model "
            "first consolidates easy problems — those it can solve with ~50% rollout "
            "success rate — it enters the harder subset with a higher effective gradient "
            "rank, because the easy-task gradient directions are already near-converged and "
            "the optimiser shifts budget toward the harder, under-explored directions. "
            "Flat GRPO (all 200 tasks in fixed random order, every problem equally likely "
            "each batch) never achieves this because it spends gradient steps on already-"
            "mastered easy tasks throughout training. None of the 61 queued experiments "
            "use grpo_curriculum_continual kind — all use flat grpo_continual, "
            "grpo_multi_seed_staircase (task-count staircase, not intra-budget difficulty "
            "ordering), forgetting_eval, or joint_cycle_multiseed. The curriculum ordering "
            "is orthogonal to credit assignment (EXP-052/054/055), regularization "
            "(EXP-056/057), data diversity (EXP-058), and reward shaping (EXP-059-061): "
            "it changes only the sequence in which 200 tasks are sampled during one epoch, "
            "with no change to GRPO algorithm, LoRA config, rollout count, or reward. "
            "Implementation: (1) pre-compute base_pass_at_4 for all training problems "
            "using the frozen base model before training starts (~12 min, included in "
            "wall-clock), (2) sort problems by ascending difficulty "
            "(difficulty = 1 - base_pass_at_4), (3) linearly increase difficulty "
            "threshold from p=0.3 to p=1.0 over training steps, sampling only problems "
            "whose difficulty <= current threshold at each batch. Wall-clock est. ~75 min "
            "(12 min pre-eval + 63 min training, comparable to 200-task baseline)."
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
            "curriculum_strategy": "difficulty_sorted_easy_to_hard",
            "difficulty_proxy": "base_pass_at_4",
            "curriculum_difficulty_threshold_start": 0.3,
            "curriculum_difficulty_threshold_end": 1.0,
            "curriculum_annealing": "linear",
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_06_11_002_recrl_curriculum_continual_400tasks",
        "priority": 7,
        "kind": "grpo_curriculum_continual",
        "rationale": (
            "Direct extension of EXP-062 (exp_2026_06_11_001) to the 400-task regime "
            "where flat grpo_continual is known to degrade: the best-observed flat GRPO "
            "result drops from 47→46/100 at train_task_limit=400 vs. 47→49/100 at "
            "train_task_limit=200. This degradation is one of three explicit open "
            "questions in the project (state.json, history section): 'Is the 400-task "
            "degradation catastrophic forgetting or reward hacking/MBPP overfitting?' "
            "EXP-053 (forgetting_eval) and EXP-058 (mixed MBPP+HumanEval) probe the "
            "forgetting and overfitting hypotheses respectively from diagnostic and "
            "data-diversity angles. This experiment probes a third hypothesis from "
            "arxiv:2605.00433 (RECRL) and arxiv:2605.12484 (Learning Fast and Slow, "
            "Section 3.2): the 400-task degradation may be an entropy collapse artefact. "
            "Entropy collapse in GRPO proceeds as follows: the policy rapidly satisfies "
            "the easy problems in the mixed random training pool during early steps, then "
            "continues to produce near-identical rollouts for those problems (all-pass, "
            "zero gradient), while the genuinely hard problems — which the policy has "
            "not yet solved — become an increasingly small fraction of the effective "
            "gradient batch. By step ~300 the policy is mostly reinforcing already-correct "
            "patterns, accelerating entropy collapse into a narrow solution manifold that "
            "scores well on the training set but degrades on held-out eval. The RECRL "
            "curriculum fix addresses this directly: by sorting easy problems first and "
            "hard problems last, the policy spends gradient budget on the most informative "
            "frontier at each point in training, maintaining a higher fraction of "
            "mixed-outcome rollout groups throughout. At 400 tasks the curriculum is "
            "especially important because the task difficulty range is wider: the top-200 "
            "hardest problems in the 400-task set are genuinely harder than those in the "
            "200-task set, and without curriculum ordering, these hardest problems are "
            "diluted by easy ones that produce zero gradient. Spec matches EXP-062 except "
            "train_task_limit=400. Wall-clock est. ~120 min (12 min pre-eval + 108 min "
            "training at 2× task count). Within 4-hour A800 limit. "
            "Duplicate check: EXP-056 trains at 400 tasks with lora_r=8 (flat GRPO, no "
            "curriculum); no other queued experiment uses grpo_curriculum_continual at "
            "any task count. ✓"
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
            "curriculum_strategy": "difficulty_sorted_easy_to_hard",
            "difficulty_proxy": "base_pass_at_4",
            "curriculum_difficulty_threshold_start": 0.3,
            "curriculum_difficulty_threshold_end": 1.0,
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
