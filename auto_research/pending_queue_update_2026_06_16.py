#!/usr/bin/env python3
"""
Daily queue patch — 2026-06-16 (EXP-072, EXP-073).

A800 connectivity: offline since 2026-05-14 (day 33). Apply when restored.

Prior patches to apply before this one (if using the daily-patch chain):
    python3 auto_research/pending_queue_update.py                  # base experiments
    python3 auto_research/pending_queue_update_2026_06_06.py       # EXP-054, EXP-055
    python3 auto_research/pending_queue_update_2026_06_07.py       # EXP-056, EXP-057
    python3 auto_research/pending_queue_update_2026_06_08.py       # EXP-058, EXP-059
    python3 auto_research/pending_queue_update_2026_06_10.py       # EXP-060, EXP-061
    python3 auto_research/pending_queue_update_2026_06_11.py       # EXP-062, EXP-063
    python3 auto_research/pending_queue_update_2026_06_12.py       # EXP-064, EXP-065
    python3 auto_research/pending_queue_update_2026_06_12_v2.py    # EXP-066, EXP-067
    python3 auto_research/pending_queue_update_2026_06_14.py       # EXP-068, EXP-069
    python3 auto_research/pending_queue_update_2026_06_15.py       # EXP-070, EXP-071

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_06_16.py       # EXP-072, EXP-073

Queue was ~71 pending on 2026-06-15 (+2 added: EXP-070, EXP-071).
Queue cap applied: >20 → max 2 new experiments.
"""
import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_06_16_001_infinite_sampling_g8_200tasks",
        "priority": 7,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2506.22950 ('Infinite Sampling: Efficient and Stable Grouped RL "
            "Training for Large Language Models', Wang et al., June 2025) addresses a "
            "critical under-explored dimension of GRPO: the group size G. All prior "
            "experiments in the queue use G=4 (rollouts_per_prompt=4) because larger G "
            "requires proportionally more GPU memory. At G=4 on our MBPP training set, "
            "zero-variance collapse affects an estimated 25-73% of all prompt groups at "
            "any given point in training (when pass_rate≈0.7, the probability that all "
            "four rollouts pass is 0.7^4≈24%; at pass_rate≈0.9 from mid-training, it "
            "is 0.9^4≈66%). This is the dominant gradient-signal pathology in our "
            "setting — identified since the initial failure-mode analysis and now "
            "addressed from the token-level by EP-GRPO (EXP-068, per-token entropy "
            "gating). Infinite Sampling proposes a complementary structural fix: "
            "decompose a large logical group of G=N samples into multiple micro-sampling "
            "rounds of g < N samples each, running generation in memory-feasible g-size "
            "batches and then computing group-relative advantages across all N combined "
            "completions. The paper also introduces 'continuous sampling' that pipelines "
            "the generation of round k+1 with the training step of round k, keeping the "
            "GPU continuously occupied and limiting actual wall-clock overhead to roughly "
            "1.5-1.8× (not the naive 2× that would result from purely sequential rounds). "
            "At G=8 (2 micro rounds of g=4): P(all-pass at rate 0.7) = 0.7^8 ≈ 6%; "
            "P(all-pass at rate 0.9) = 0.9^8 ≈ 43%. This is a 4-6× reduction in "
            "zero-variance collapse rate compared to G=4, achieved without any change to "
            "the reward signal, LoRA configuration, or training schedule. The larger "
            "group also yields a lower-variance advantage estimator: since advantages are "
            "normalized within the group, using more samples makes the mean and standard "
            "deviation estimates more reliable, reducing gradient noise. Mechanistically "
            "distinct from EP-GRPO (EXP-068): EP-GRPO injects process signal via the "
            "policy/reference log-ratio to compensate for zero-advantage groups after "
            "they form; Infinite Sampling (EXP-072) prevents zero-advantage groups from "
            "forming in the first place by increasing group diversity. The two approaches "
            "are also orthogonal and could in principle be combined after results are "
            "available. Distinct from all other queued experiments: no queued experiment "
            "has rollouts_per_prompt > 4 or micro_sampling=True. Wall-clock estimate: "
            "~110 min (1.57× overhead from G=8 micro-sampling on A800 80GB, based on "
            "Infinite Sampling paper benchmarks: generation-to-train ratio ≈ 60:40 for "
            "1.5B model, pipeline overlap reduces overhead to ~1.57× baseline)."
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
            "rollouts_per_prompt": 8,
            "micro_sampling": True,
            "micro_sampling_group_size": 4,
            "lr": 5e-6,
            "lora_r": 16,
            "prompt_style": "qwen-chat",
            "reward": "binary",
            "arxiv_ref": "2506.22950",
            "estimated_hours": 1.83,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_06_16_002_frpo_robust_grpo_400tasks",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2602.08813 ('Robust Policy Optimization to Prevent Catastrophic "
            "Forgetting', February 2026) introduces FRPO (Fine-tuning Robust Policy "
            "Optimization), a modified RLHF/GRPO objective that directly targets "
            "catastrophic forgetting in multi-stage post-training. The key insight: "
            "standard GRPO maximizes E_π[R] at the current policy θ, but is oblivious "
            "to the fact that subsequent task updates will move θ to θ', and E_{θ'}[R_old] "
            "may collapse. FRPO modifies the objective to optimize reward not just at θ "
            "but across a KL-bounded neighborhood of reachable downstream policies: "
            "max_θ  min_{KL(π', π) ≤ ε}  E_{π'}[R]. The resulting policy has a "
            "'flat reward landscape' in the KL ball around it — small perturbations "
            "(caused by learning new tasks) cause smaller reward degradation on the "
            "old distribution. Implementation: FRPO augments the GRPO loss with a "
            "minimax regularization term parameterized by kl_neighborhood_radius ε; "
            "the inner minimization is solved via a dual-variable update (Lagrangian "
            "relaxation) and adds negligible compute overhead (~5-8% per step). The "
            "paper reports code at github.com/Helloworld10011/FRPO. This directly "
            "addresses our principal open question: why does GRPO degrade from 47→46 "
            "MBPP at 400 tasks despite using LoRA r=16 and standard KL regularization? "
            "Our current hypothesis tree has three branches — entropy collapse (targeted "
            "by EXP-070, EXP-071), zero-variance groups (EXP-068, EXP-072), and true "
            "catastrophic forgetting of early-task representations. FRPO directly tests "
            "the third branch: if the 400-task degradation is due to sequential policy "
            "drift, FRPO's KL-neighborhood objective should stabilize the drift and "
            "allow 400-task training to match or exceed the 200-task result (47→49→?). "
            "Testing at 400 tasks (not 200) is critical: at 200 tasks GRPO already "
            "improves, so the forgetting mechanism is only exposed at higher task counts. "
            "Control comparison: standard GRPO at 400 tasks (existing result: 47→46, "
            "-1pp) vs. FRPO at 400 tasks (expected: ≥47 if robust objective works). "
            "Distinct from all queued experiments: no queued experiment uses "
            "loss_type='frpo', kl_neighborhood_radius > 0, or any minimax robust "
            "objective. EXP-071 (GTPO) removes the reference model entirely (kl_coeff=0); "
            "FRPO uses the reference model but augments the objective with a worst-case "
            "term — the two are mechanistically and mathematically distinct. Wall-clock "
            "estimate: ~120 min (400 tasks × standard G=4 per-prompt + ~6% FRPO "
            "regularization overhead; same as projected EXP-063 curriculum 400 tasks)."
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
            "loss_type": "frpo",
            "kl_neighborhood_radius": 0.1,
            "arxiv_ref": "2602.08813",
            "estimated_hours": 2.0,
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
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=STATE_PATH.parent, suffix=".tmp", prefix="state_"
    )
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
