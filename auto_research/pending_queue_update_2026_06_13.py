#!/usr/bin/env python3
"""
Daily queue patch — 2026-06-13 (EXP-068, EXP-069).

Experiments motivated by two papers found via web-search arxiv hotspot scan
(hotspot file unavailable; A800 offline day 30):

  EXP-068: F-GRPO focal advantage weighting (arxiv:2602.06717)
  EXP-069: S2L-PO small-model explorer for rollout diversity (arxiv:2605.30789)

Queue cap: 67 pending > 20 → max 2 new experiments added.

Run after all previous patches have been applied.
Order of application on A800 when connectivity restores:

    # 1. Apply older 53 experiments from git history
    git show 649b8a4fe5f11b319ce3cdd5c205b9124028eda3:auto_research/pending_queue_update.py \
        > /tmp/pqu_full.py && python3 /tmp/pqu_full.py

    # 2-7. Apply 2026-06-06 through 2026-06-12 patches in order
    python3 auto_research/pending_queue_update_2026_06_06.py
    python3 auto_research/pending_queue_update_2026_06_07.py
    python3 auto_research/pending_queue_update_2026_06_08.py
    python3 auto_research/pending_queue_update_2026_06_10.py
    python3 auto_research/pending_queue_update_2026_06_11.py
    python3 auto_research/pending_queue_update_2026_06_12.py
    python3 auto_research/pending_queue_update_2026_06_12_v2.py

    # 8. Apply this patch (EXP-068, EXP-069)
    python3 auto_research/pending_queue_update_2026_06_13.py

Or just run the master bootstrap which chains all patches:
    python3 auto_research/pending_queue_update.py
"""
import json, os, tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_06_13_001_fgrpo_focal_advantage_200tasks",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2602.06717 ('F-GRPO: Don't Let Your Policy Learn the Obvious and Forget "
            "the Rare', Feb 2026) introduces focal advantage weighting for GRPO: a difficulty-"
            "aware coefficient alpha_i = (1 - group_pass_rate)^gamma applied per-group before "
            "the advantage normalizer, where group_pass_rate = mean reward over the G=4 rollouts "
            "for prompt i. At gamma=2 (default), this down-weights gradient updates on easy "
            "prompts (group_pass_rate near 1.0, alpha→0) and up-weights updates on hard prompts "
            "(group_pass_rate near 0.0, alpha→1.0). The mechanism is distinct from and "
            "orthogonal to both Dr. GRPO (EXP-060, arxiv:2503.20783) and rollout temperature "
            "(EXP-061, arxiv:2503.14476): Dr. GRPO corrects statistical bias in the advantage "
            "normalizer denominator; T=1.2 diversifies rollout completions at sampling time; "
            "F-GRPO reweights the gradient *distribution across prompts* after rollouts are "
            "collected. Standard GRPO spends equal gradient mass on every prompt that happens "
            "to produce a mixed-outcome group, regardless of whether that prompt is in the "
            "half-mastered easy zone (group_pass_rate ~0.75, yielding large gradients on the "
            "already-learned solution) or the hard zone (group_pass_rate ~0.25, where the "
            "model actually needs improvement). F-GRPO shifts gradient mass toward prompts "
            "where the model is weakest, mirroring the function of focal loss in imbalanced "
            "classification. Empirical results on Qwen2.5-7B code generation (arxiv:2602.06717 "
            "Table 3): pass@1 holds (38.6% GRPO vs comparable F-GRPO), pass@256 improves "
            "from 64.1% → 70.3%; same pattern holds for F-DAPO (+3.2 pp) and F-CISPO (+3.6 pp). "
            "The pass@256 improvement confirms the method recovers rare correct trajectories "
            "that standard GRPO under-learns. For our setup: we are stuck at 47→49/100 pass@1 "
            "(eval100, n=1 seed, Qwen2.5-Coder-1.5B). The F-GRPO gain may be larger at 1.5B "
            "than at 7B since smaller models have a higher fraction of hard prompts in the "
            "MBPP training set. Duplicate check: no queued or historical experiment uses "
            "advantage_weighting='focal' or any per-prompt difficulty scaling coefficient. "
            "EXP-054 (EGCA, arxiv:2603.16158) uses execution-grounded credit assignment "
            "across tokens within a single completion, not across prompts. Completely distinct "
            "mechanism. Wall-clock estimate: ~65 min (identical to standard 200-task grpo_continual "
            "baseline — the focal weighting adds O(1) per-group multiply operations). ✓"
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
            "advantage_weighting": "focal",
            "focal_gamma": 2.0,
            "focal_groupwise_scaling": True,
            "eval_limit": 100,
            "max_new_tokens": 192,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_06_13_002_s2lpo_explorer_0p5b_policy_1p5b_200tasks",
        "priority": 7,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2605.30789 ('Smaller Models are Natural Explorers for Policy-Level "
            "Diversity in GRPO', May 2026) introduces S2L-PO (Small-to-Large Policy "
            "Optimization): rollouts for GRPO advantage estimation are generated by a "
            "smaller frozen model from the same family, while gradient updates train the "
            "larger target model. The key insight is that small models within the same "
            "family (e.g., Qwen2.5-Coder-0.5B vs 1.5B) naturally produce more diverse "
            "completions because their lower-entropy base distribution hasn't been "
            "RLHF-sharpened to the same degree. Unlike token-level temperature increase "
            "(EXP-061, T=1.2, arxiv:2503.14476), S2L-PO diversity is policy-level: the "
            "0.5B model explores structurally different solution paths (different algorithm "
            "choices, different variable names, different control flow) rather than "
            "token-level noise that injects incoherent variations while preserving the "
            "1.5B model's preferred solution skeleton. Paper reports +8.8% AIME24 accuracy "
            "using a 1.7B explorer to guide the 8B model, with reduced rollout compute. "
            "Relevance to our setup: our history shows Qwen2.5-Coder-0.5B consistently "
            "degrades under SFT and GRPO training (exp history entries for 0.5B base_model "
            "all show pass@1 degradation). S2L-PO uses the 0.5B ONLY as a frozen rollout "
            "generator — no gradient updates are applied to it. The 0.5B degradation history "
            "is irrelevant: we exploit its diverse inference distribution without fine-tuning "
            "it. The 1.5B model is trained normally with LoRA r=16. Memory: 0.5B at FP16 "
            "≈ 1.0 GB, 1.5B at FP16 with LoRA optimizer states ≈ 12 GB, both fit "
            "comfortably on A800 80 GB. Implementation: (1) for each training batch, generate "
            "G=4 rollouts using the 0.5B model (frozen, no gradients); (2) score each rollout "
            "with the binary correctness reward; (3) compute GRPO group-relative advantages "
            "over the 4 rollouts; (4) compute policy gradient loss against the 1.5B model "
            "(using the 0.5B rollouts as the behavior policy). This is an importance-weighted "
            "policy gradient: since the behavior policy (0.5B) differs from the target policy "
            "(1.5B), a clipping ratio is needed — we use the same PPO epsilon=0.2 clip as "
            "standard GRPO. Addresses the entropy collapse mechanism identified in the "
            "400-task degradation (open question since 2026-06-12 report, targeted by "
            "EXP-063) from a complementary angle: EXP-063 uses easy-to-hard curriculum to "
            "space out hard problems; S2L-PO provides fresh diverse rollouts that prevent "
            "the 1.5B model from collapsing to a narrow high-pass-rate solution distribution "
            "even without curriculum scheduling. Duplicate check: no queued or historical "
            "experiment uses explorer_model or rollout_generator='explorer_model'. "
            "Wall-clock estimate: ~80 min (0.5B rollout generation adds ~15 min vs "
            "standard 200-task baseline of ~65 min). ✓"
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "explorer_model": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            "rollout_generator": "explorer_model",
            "explorer_rollout_temperature": 1.0,
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
            "importance_weighting": "ppo_clip",
            "ppo_epsilon": 0.2,
            "eval_limit": 100,
            "max_new_tokens": 192,
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
