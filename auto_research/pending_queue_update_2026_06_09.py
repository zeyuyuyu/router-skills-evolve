#!/usr/bin/env python3
"""
Daily queue patch — 2026-06-09 (EXP-060, EXP-061).

Run after pending_queue_update_2026_06_08.py has been applied.
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
    python3 auto_research/pending_queue_update_2026_06_09.py

Or just run the master bootstrap which chains all patches:
    python3 auto_research/pending_queue_update.py
"""
import json, os, tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_06_09_001_code_quality_reward_grpo_200tasks",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2506.02211 ('Improving LLM-Generated Code Quality with GRPO', June 2, "
            "2026; Robeyns & Aitchison, Univ. of Bristol) identifies a fundamental blind "
            "spot in binary pass/fail reward training: execution feedback confirms "
            "correctness but provides zero gradient signal toward maintainable, readable, "
            "or safe code. All 59 experiments in the current queue use reward='binary' or "
            "reward='binary_length_normalised' — both evaluate only whether tests pass, "
            "not HOW the solution is written. The paper develops a code quality library "
            "that quantifies maintainability, reliability, and style via static analysis "
            "(cyclomatic complexity, Halstead volume, maintainability index — all computable "
            "via the `radon` Python package, already available in the project environment) "
            "and uses this as the GRPO reward signal. Human evaluation confirms the "
            "trained model produces code rated higher on maintainability and safety. "
            "This experiment implements a WEIGHTED COMBINATION rather than a quality-only "
            "reward: reward = 0.7 * binary_pass_fail + 0.3 * quality_score, where "
            "quality_score = 0.5 * normalised_maintainability_index + "
            "0.5 * (1 - normalised_cyclomatic_complexity), both normalised to [0,1] across "
            "the rollout group. The 0.7/0.3 weighting preserves functional correctness as "
            "the dominant gradient while adding a differentiating signal within the "
            "passing-rollout group — where binary reward is currently uniform (+1.0 for all "
            "passing solutions regardless of quality). This is orthogonal to EXP-059 "
            "(length-normalised reward): EXP-059 penalises verbosity; EXP-060 rewards "
            "code structure quality independent of length. Failing rollouts still receive "
            "0.0 (the quality term only applies if the solution passes at least one test, "
            "avoiding the demoted EXP-009 mechanism of rewarding non-passing code). "
            "For the ~47% of rollouts that currently pass (≈376/800 positive rewards per "
            "200-task run), quality differentiation adds a previously absent gradient "
            "direction. Est. wall-clock ~80 min: quality metrics add ~15 ms per rollout "
            "via radon AST traversal (CPU, no extra LLM calls)."
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
            "reward": "binary_plus_quality",
            "quality_weight": 0.3,
            "quality_metrics": ["maintainability_index", "cyclomatic_complexity"],
            "quality_provider": "radon",
            "quality_normalise": "group",
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_06_09_002_scrl_test_progressive_curriculum_200tasks",
        "priority": 6,
        "kind": "grpo_curriculum_continual",
        "rationale": (
            "arxiv:2605.22074 ('From Reasoning Chains to Verifiable Subproblems: Curriculum "
            "RL Enables Credit Assignment for LLM Reasoning', May 21, 2026; Jiang et al., "
            "Tsinghua LeapLab) proposes SCRL: decompose hard RL tasks into verifiable "
            "subproblems to convert sparse final-answer rewards into dense curriculum "
            "signals. The paper was identified in the 2026-05-30 hotspot sweep as 'Moderate "
            "relevance' (tag: scrl-subproblem-curriculum) and deferred 'pending implementation "
            "assessment'. This experiment provides the concrete implementation for MBPP: "
            "each MBPP problem has 3 test cases (ordered by difficulty: test_0 tests the "
            "most common case; test_1 an edge case; test_2 a stress/corner case). "
            "The SCRL curriculum maps naturally to a test-case progressive schedule: "
            "Stage 1 (tasks 0-66): pass reward requires only test_0 to pass (easy target; "
            "~70% of the model's current rollouts pass test_0 alone). "
            "Stage 2 (tasks 67-133): reward requires tests {0,1} both to pass. "
            "Stage 3 (tasks 134-199): reward requires all tests {0,1,2} to pass (full MBPP "
            "standard). "
            "This is mechanistically distinct from all three existing reward-shaped "
            "experiments: (a) VPO (EXP-038) uses per-test-case advantage WITHIN a single "
            "reward computation — no curriculum, no stage progression; (b) P-GRPO (EXP-009, "
            "demoted) assigns fractional scalar credit for partial pass across all stages "
            "simultaneously — no curriculum progression and hurt vs binary per "
            "arxiv:2605.02944; (c) RECRL (EXP-001) uses difficulty-adaptive task sampling "
            "based on model pass rate but does NOT change which test cases serve as the "
            "reward criterion — the reward target is always 'pass all tests'. "
            "SCRL test-progressive curriculum changes the reward THRESHOLD progressively, "
            "making the task easier at the start and harder at the end. The mechanism: "
            "by training against an achievable single-test target first, the model develops "
            "correct algorithmic structure (test_0 typically verifies the core algorithm); "
            "subsequent stages then refine edge-case handling with full gradient signal, "
            "rather than training from scratch against a hard 3-test criterion where most "
            "rollouts fail and advantage collapses to 0. If the ~26% of all-fail tasks at "
            "stage 3 were all-fail at stage 1 too (~5-8% estimated), this intervention "
            "maintains gradient signal across more task batches than flat binary GRPO. "
            "Implementation: spec field test_case_curriculum_schedule is a list of "
            "[task_start, task_end, required_tests_count] triples; runner selects the "
            "relevant test-case subset from the MBPP eval tuple per stage. "
            "Est. wall-clock ~70 min (same as baseline; no extra model calls)."
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
            "curriculum_type": "test_case_progressive",
            "test_case_curriculum_schedule": [
                {"task_range": [0, 66], "required_tests": 1},
                {"task_range": [66, 133], "required_tests": 2},
                {"task_range": [133, 200], "required_tests": "all"},
            ],
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
