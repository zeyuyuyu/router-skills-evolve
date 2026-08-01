"""
Pending queue update for 2026-08-01.
Run this on the A800 when SSH access is restored:
  python3 /data0/home/zeyuwang/auto_research/pending_queue_update_2026_08_01.py

Adds EXP-144 and EXP-145 to state["queue"].
"""

import json
import os
import tempfile

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_08_01_001_grsd_intra_cycle_skill_reflection_from_grpo_rollouts",
        "priority": 6,
        "gpu": "auto",
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2607.28076 (GRSD, Jul 30 2026) shows that contrasting successful vs failed "
            "GRPO rollouts within a training batch yields richer skill supervision (+4.1pp on WebArena). "
            "Our SkillBook distills skills from GPT-5.5 teacher traces between cycles (inter-cycle). "
            "EXP-144 adds a post-GRPO intra-cycle skill refinement step: after GRPO training completes "
            "for cycle N, extract 3-5 rollout-differential skills by prompting the large model to contrast "
            "successful vs failed rollout pairs from the GRPO training log, then append to skillbook.json "
            "before cycle N+1 trace collection. Tests whether intra-cycle skill refinement (capturing small "
            "model failure modes) complements inter-cycle distillation from teacher traces. ~20 lines hooking "
            "into grpo_train_simple.py end; no change to GRPO algorithm or reward signal."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 1,
            "start_cycle": 1,
            "grpo_intra_cycle_skill_reflection": True,
            "grpo_skill_reflection_n_pairs": 5,
            "grpo_skill_reflection_model": "large",
            "scaling_force_both": 1,
            "skip_grpo": 0,
            "skip_sft": 0,
            "notes": (
                "Post-GRPO: sample 5 success/fail rollout contrast pairs from grpo training log, "
                "call large model with update_skill prompt, append results to skillbook.json. "
                "Implements GRSD-style intra-cycle skill extraction (arxiv:2607.28076)."
            ),
        },
    },
    {
        "id": "exp_2026_08_01_002_circuit_forgetting_probe_sft_vs_grpo_cka_analysis",
        "priority": 7,
        "gpu": "auto",
        "kind": "forgetting_eval",
        "rationale": (
            "arxiv:2605.28860 (Mechanistic Origins of Catastrophic Forgetting, ACL 2026) shows SFT "
            "causes greater attention circuit disruption than RL (Dr.GRPO) using differential circuit "
            "vulnerability (DCV). Our cycle-1 SFT epoch-2 overfit (loss reversal 0.178->0.271) is "
            "hypothesized to cause the 4.88pp skills arm dip via plasticity loss (Hypothesis F). "
            "EXP-145 provides mechanistic validation: load the 4 cycle-1 checkpoint stages "
            "(cycle-0-base, SFT-ep1/ckpt-6, SFT-ep2/ckpt-12, GRPO-1/grpo_adapter), run 82 HumanEval "
            "probes (forward pass only, no generation), compute CKA across decoder layers vs cycle-0 "
            "baseline. If SFT-ep2 shows greater attention-layer CKA drop than SFT-ep1, and GRPO-1 shows "
            "partial restoration, this mechanistically confirms Hypothesis F. Inference-only (~0.5h), "
            "uses existing checkpoints in results/e2e_4cyc_gpt55/cycle_1/. Implements "
            "src/pipeline/circuit_forgetting_probe.py."
        ),
        "spec": {
            "bench": "humaneval",
            "eval_only": True,
            "checkpoints": [
                "results/e2e_4cyc_gpt55/cycle_0/grpo_adapter",
                "results/e2e_4cyc_gpt55/cycle_1/llm_adapter/checkpoint-6",
                "results/e2e_4cyc_gpt55/cycle_1/llm_adapter/checkpoint-12",
                "results/e2e_4cyc_gpt55/cycle_1/grpo_adapter",
            ],
            "metric": "cka_per_layer",
            "baseline_checkpoint": "results/e2e_4cyc_gpt55/cycle_0/grpo_adapter",
            "n_probe_tasks": 82,
            "script": "src/pipeline/circuit_forgetting_probe.py",
            "notes": (
                "Load each checkpoint, run 82 HumanEval inputs as forward-pass probes (no sampling), "
                "extract decoder layer hidden states, compute centered kernel alignment (CKA) vs cycle-0 "
                "baseline. Plot per-layer CKA heatmap. No new training. Grounds Hypothesis F with "
                "mechanistic figure for AAAI submission §5.4."
            ),
        },
    },
]


def main():
    with open(STATE_PATH, "r") as f:
        state = json.load(f)

    existing_ids = {e["id"] for e in state.get("queue", [])} | {
        e.get("id", "") for e in state.get("history", [])
    }

    added = []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] in existing_ids:
            print(f"SKIP (already exists): {exp['id']}")
        else:
            state["queue"].append(exp)
            added.append(exp["id"])
            print(f"ADDED: {exp['id']}")

    # Atomic write
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(STATE_PATH), suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, STATE_PATH)
        print(f"state.json updated. Added {len(added)} experiments: {added}")
    except Exception as e:
        os.unlink(tmp_path)
        raise e


if __name__ == "__main__":
    main()
