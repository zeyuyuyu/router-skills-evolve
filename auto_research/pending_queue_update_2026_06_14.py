#!/usr/bin/env python3
"""
Daily queue patch — 2026-06-14 (EXP-068, EXP-069).

A800 connectivity: offline since 2026-05-14. Apply when restored.

Prior patches to apply before this one (if using the daily-patch chain):
    python3 auto_research/pending_queue_update_2026_06_06.py   # EXP-054, EXP-055
    python3 auto_research/pending_queue_update_2026_06_07.py   # EXP-056, EXP-057
    python3 auto_research/pending_queue_update_2026_06_08.py   # EXP-058, EXP-059
    python3 auto_research/pending_queue_update_2026_06_10.py   # EXP-060, EXP-061
    python3 auto_research/pending_queue_update_2026_06_11.py   # EXP-062, EXP-063
    python3 auto_research/pending_queue_update_2026_06_12.py   # EXP-064, EXP-065
    python3 auto_research/pending_queue_update_2026_06_12_v2.py  # EXP-066, EXP-067

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_06_14.py   # EXP-068, EXP-069

Queue was 67 pending on 2026-06-13 (65 from ideas-06-12 + 2 from 06-12-v2).
Queue cap applied: >20 → max 2 new experiments.
"""
import json, os, tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_06_14_001_ep_grpo_entropy_gated_200tasks",
        "priority": 7,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2605.04960 ('EP-GRPO: Entropy-Progress Aligned Group Relative Policy "
            "Optimization with Implicit Process Guidance', May 2026) identifies three "
            "structural credit assignment failures in standard GRPO: (1) uniform token-level "
            "granularity — every token in a generated completion receives an identical "
            "advantage scalar regardless of its contribution to the outcome, ignoring "
            "heterogeneous informational value; (2) uniform polarity — the same group-relative "
            "normalizer can misattribute advantage sign (penalizing correct reasoning steps "
            "within a failed completion, rewarding incorrect steps within a passing one); "
            "(3) zero-variance collapse — when all G rollouts within a group are identical "
            "(all-pass or all-fail), the group-relative advantage collapses to zero, wiping "
            "out the gradient signal for 73% of our training prompts at 200 tasks and nearly "
            "all prompts at 400 tasks. EP-GRPO addresses all three by: (A) entropy-gated "
            "advantage modulation — tokens with high entropy (genuine decision points, "
            "e.g., branching code structure choices) receive a scaled-up advantage; "
            "low-entropy deterministic tokens (whitespace, closing brackets) receive scaled-"
            "down advantage; (B) implicit process signals — the log-probability ratio between "
            "the current policy and reference model is used as a dense per-token quality "
            "signal anchored to the final outcome advantage, providing directional supervision "
            "between gradient steps without any external reward model. Both signals are "
            "computable from quantities already available during GRPO training: token "
            "log-probs (for entropy) and policy/reference log-prob ratios (already computed "
            "for the KL penalty). Implementation cost: zero additional LLM calls, minimal "
            "memory overhead. Published results: on Qwen2.5-3B across 5 math reasoning "
            "benchmarks, EP-GRPO improves average accuracy from GRPO's 18.14% to 22.93% "
            "(+4.79 pp). The mechanism is domain-agnostic and directly targets the "
            "zero-variance collapse identified as our primary 400-task degradation hypothesis. "
            "This experiment is distinct from EXP-052 (GRPO-lambda, temporal discount by "
            "token position) and from GRPO-S (arxiv:2508.04349, simpler entropy weighting "
            "without implicit process signal). Wall-clock estimate: ~65 min."
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
            "advantage_normalizer": "ep_grpo",
            "entropy_gated_advantage": True,
            "entropy_gate_scale": "adaptive",
            "implicit_process_signal": "policy_divergence",
            "ep_grpo_ref_model": "base",
            "arxiv_ref": "2605.04960",
            "estimated_hours": 1.08,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_06_14_002_clpo_online_curriculum_200tasks",
        "priority": 6,
        "kind": "grpo_curriculum_continual",
        "rationale": (
            "arxiv:2509.25004 ('CLPO: Curriculum Learning meets Policy Optimization for "
            "LLM Reasoning', updated June 8, 2026) introduces a self-evolving curriculum "
            "framework that re-estimates task difficulty from on-policy rollout accuracy "
            "at regular training intervals, then restructures the task distribution "
            "according to the model's current capability frontier. Unlike the static "
            "curricula in EXP-062 (easy-to-hard, base_pass_at_4 computed once before "
            "training) and EXP-063 (same, 400 tasks), CLPO updates the difficulty proxy "
            "every K gradient steps using the model's current rollout pass rates. This "
            "addresses the key limitation of static easy-to-hard curricula: after the "
            "model masters the initially-easy tasks, it continues sampling them at high "
            "frequency under a fixed schedule, producing all-pass groups with zero gradient "
            "signal. CLPO detects this capability shift online and re-classifies those tasks "
            "as 'solved', replacing them with medium-difficulty tasks that produce mixed-"
            "outcome rollout groups (informative gradient signal). In parallel, CLPO "
            "diversifies hard tasks by generating perturbed variants of problems the model "
            "consistently fails, providing additional gradient signal at the upper tail. "
            "Concretely: every 20 gradient steps, collect 2 rollouts per training task "
            "(half the standard rollout budget to limit overhead), recompute difficulty as "
            "current_pass_at_2, and rebuild the sampling priority distribution. This "
            "experiment directly tests whether the gap between our best static-curriculum "
            "result (EXP-062, expected ~2.1pp over baseline per arxiv:2605.00433) and "
            "CLPO's online adaptation can be closed by dynamic re-estimation. "
            "Corroborating evidence: arxiv:2506.06632 (E2H Reasoner) provides convergence "
            "guarantees showing that models trained with gradually fading easy tasks achieve "
            "lower sample complexity than both flat and static easy-to-hard curricula, "
            "with improvements concentrated in 1.5B-3B models. Online re-estimation is "
            "the mechanism enabling this fading to track the actual capability frontier. "
            "Implementation: the `grpo_curriculum_continual` kind in runner.py already "
            "supports periodic difficulty refresh via `curriculum_update_freq_steps`. "
            "Novel fields: `curriculum_strategy='online_clpo'` with "
            "`curriculum_update_freq_steps=20` and `curriculum_rollout_proxy='current_pass_at_2'` "
            "(vs EXP-062's `'base_pass_at_4'` pre-computed once). Wall-clock: ~85 min "
            "(includes overhead of periodic 2-rollout re-evaluation every 20 steps)."
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
            "curriculum_strategy": "online_clpo",
            "curriculum_update_freq_steps": 20,
            "curriculum_rollout_proxy": "current_pass_at_2",
            "curriculum_difficulty_threshold_start": 0.3,
            "curriculum_difficulty_threshold_end": 1.0,
            "curriculum_annealing": "linear",
            "arxiv_ref": "2509.25004",
            "estimated_hours": 1.42,
        },
        "gpu": "auto",
    },
]


def main():
    if not STATE_PATH.exists():
        print(f"ERROR: {STATE_PATH} not found. Are you on the A800 or H200?")
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
