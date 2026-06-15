#!/usr/bin/env python3
"""
Daily queue patch — 2026-06-15 (EXP-070, EXP-071).

A800 connectivity: offline since 2026-05-14 (day 32). Apply when restored.

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

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_06_15.py       # EXP-070, EXP-071

Queue was ~69 pending on 2026-06-14 (+2 added: EXP-068, EXP-069).
Queue cap applied: >20 → max 2 new experiments.
"""
import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_06_15_001_entrocraft_entropy_linear_annealing_200tasks",
        "priority": 7,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2604.26326 ('Addressing Performance Saturation for LLM RL via Precise "
            "Entropy Curve Control', Bolian Li et al., April 2026) identifies performance "
            "saturation in LLM RL — including GRPO — as a direct consequence of entropy "
            "collapse: once the policy's entropy falls below a critical threshold, rollout "
            "diversity drops to near-zero, gradient signal vanishes, and training improvement "
            "stalls. This is distinct from the zero-variance group collapse targeted by "
            "EP-GRPO (EXP-068): zero-variance collapse affects individual prompt groups "
            "when G rollouts agree, whereas entropy collapse is a global phenomenon that "
            "accumulates progressively across training and becomes catastrophic at high "
            "task counts (explaining our 400-task degradation as well as the +2pp ceiling "
            "at 200 tasks). The paper introduces Entrocraft, a rejection-sampling technique "
            "that realizes a user-specified entropy schedule by biasing the advantage "
            "distribution: completions whose per-step entropy is inconsistent with the "
            "target curve are rejected from the batch without computing their gradient, "
            "maintaining entropy within a controlled corridor throughout training. Unlike "
            "entropy regularization (adding an entropy bonus to the loss), Entrocraft "
            "directly controls the empirical entropy trajectory without requiring an "
            "auxiliary coefficient that must be tuned. The paper's systematic sweep of "
            "entropy schedule shapes shows that linear annealing — starting at high entropy "
            "(encouraging exploration early) and decaying linearly to a slightly lower "
            "target (consolidating solutions late) — outperforms all alternatives. Key "
            "empirical results: (1) a 4B model with Entrocraft sustains performance gains "
            "for up to 4× longer before saturation compared to standard GRPO; (2) pass@K "
            "diversity is raised by 50% over baseline; (3) the 4B model surpasses an 8B "
            "baseline that uses standard GRPO. These results suggest the +2pp ceiling we "
            "observe at 200 tasks (47→49 MBPP) may be addressable without increasing task "
            "count or changing the reward signal — purely by controlling the entropy "
            "trajectory. Distinction from existing queue: EXP-068 (EP-GRPO, 2605.04960) "
            "uses token-level entropy gating to re-weight per-token advantages — it does "
            "not control the global entropy curve and does not use rejection sampling. "
            "EXP-052 (GRPO-lambda) uses temporal token-position discounting. No queued "
            "experiment uses entropy schedule control or rejection-sampling advantage "
            "filtering. Implementation: Entrocraft requires computing per-completion "
            "entropy from the model's logits (already available) and comparing to a "
            "schedule target; completions outside a tolerance band are masked from the "
            "gradient update. The schedule is parameterized as start_entropy=3.0, "
            "end_entropy=1.5, schedule='linear' across training steps. Wall-clock: "
            "~70 min (same compute as standard GRPO; rejection adds negligible overhead "
            "since rejected completions are already generated but not back-propped)."
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
            "entropy_control": "entrocraft_rejection",
            "entropy_schedule": "linear_annealing",
            "entropy_schedule_start": 3.0,
            "entropy_schedule_end": 1.5,
            "entropy_schedule_tolerance": 0.2,
            "entrocraft_reject_outside_band": True,
            "arxiv_ref": "2604.26326",
            "estimated_hours": 1.17,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_06_15_002_gtpo_no_reference_model_200tasks",
        "priority": 6,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2508.03772 ('GTPO: Stabilizing Group Relative Policy Optimization via "
            "Gradient and Entropy Control', August 2025) identifies two structural "
            "instabilities in GRPO that are distinct from those addressed by EP-GRPO "
            "(EXP-068) and Entrocraft (EXP-070). (1) Conflicting gradient updates on "
            "shared tokens: when a valuable token (a correct branching keyword like `if`, "
            "`return`, or `def`) appears in both a passing and a failing completion within "
            "the same group G, the token receives both a positive gradient update (from "
            "the passing completion) and a negative update (from the failing completion). "
            "The net effect depends on advantage magnitudes and can reduce the probability "
            "of a token that the model has already learned is useful — an anti-gradient "
            "signal that degrades learned skills. GTPO addresses this by skipping gradient "
            "updates for shared tokens in negatively-rewarded completions while amplifying "
            "gradient from shared tokens in positively-rewarded completions. (2) Entropy "
            "collapse via negatively-rewarded trajectories: completions that receive "
            "negative rewards in GRPO shift the policy away from the sampled tokens "
            "toward the uniform distribution, but since the space is extremely large, "
            "this shift is diffuse and destabilizing. GTPO filters out completions whose "
            "trajectory-level entropy exceeds a provable threshold derived from the group "
            "advantage statistics, preventing entropy blowup while still applying learning "
            "signal from stable completions. Crucially, GTPO does not use KL-divergence "
            "regularization against a reference model — the stability guarantees come "
            "entirely from the gradient and entropy filters. This eliminates the reference "
            "model forward pass (saving ~10-15% wall-clock) and removes the KL "
            "hyperparameter as a confound. Validated on GSM8K, MATH, AIME 2024/2025, "
            "AMC 2023 (math reasoning). This is the first code generation test. "
            "Distinction from existing queue: EP-GRPO (EXP-068, 2605.04960) uses "
            "token-level entropy WEIGHTING (scaling advantage by per-token entropy) and "
            "policy/reference divergence as a process signal — it retains the reference "
            "model and KL regularization. GTPO removes the reference model entirely and "
            "operates at the trajectory level. GRPO-S (2508.04349, deferred) uses "
            "sequence-level entropy bonus — it is additive to the loss, not a gradient "
            "filter, and still uses KL/reference. No queued experiment has "
            "kl_coeff=0.0 or use_reference_model=False. Wall-clock: ~55 min "
            "(no reference model forward pass per training step)."
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
            "advantage_normalizer": "gtpo",
            "gtpo_skip_negative_shared_tokens": True,
            "gtpo_amplify_positive_shared_tokens": True,
            "gtpo_entropy_filter_threshold": "auto",
            "kl_coeff": 0.0,
            "use_reference_model": False,
            "arxiv_ref": "2508.03772",
            "estimated_hours": 0.92,
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
