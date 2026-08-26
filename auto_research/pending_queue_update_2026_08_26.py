"""
Pending queue update — 2026-08-26
Run on A800 when connectivity is restored:
  python /data0/home/zeyuwang/auto_research/pending_queue_update_2026_08_26.py
Appends EXP-192 and EXP-193 to state["queue"] and saves atomically.

A800 offline since 2026-05-14 (day ~104). Queue ~191 pending (>20 cap → 2 today).
AAAI 2027 camera-ready: due ~2026-08-29 (3 days). CRITICAL WINDOW.
GPU window: CLOSED. Both experiments are OFFLINE / 0h GPU.

NEW PAPERS TODAY:
  EXP-192: Distilled-RL-Audit — MERA's SFT→GRPO Two-Phase Pipeline as a Distilled RL Instance
           (arxiv:2607.17247 "Distilled Reinforcement Learning for LLM Post-training", July 2026)
           Offline audit: map MERA's Phase 1 (teacher corpus) + Phase 3a (SFT init) + Phase 3b (GRPO)
           onto Distilled RL's four components (teacher corpus, negative sample reset, selective
           transfer, fine-grained credit). Provides §5.1 camera-ready citation grounding MERA's
           SFT→GRPO design choice in Distilled RL theory. Priority 8. No GPU.

  EXP-193: Cue-GRPO-Audit — Bright-Room Gradient Loss Analysis for MERA Phase 3b
           (arxiv:2608.03467 "When Correct Solutions Repeat: Rarity-Aware Credit Redistribution
            for GRPO", August 2026)
           Offline audit: count all-success rollout groups in MERA's cycle-1/3 results; compute
           fraction of gradient mass lost to bright-room collapse (all rollouts pass → uniform
           advantage → zero gradient). Validates that GRPO_TEMPERATURE=1.0 prevents bright-room
           collapse. Provides §5.4 camera-ready two-sided temperature motivation. Priority 7. No GPU.

Apply chain before this patch:
    python3 auto_research/pending_queue_update_2026_08_25.py  # EXP-190, EXP-191
"""

import json, os, shutil

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

NEW_EXPERIMENTS = [
    {
        "id": "EXP-192",
        "priority": 8,
        "title": "Distilled-RL-Audit: MERA's SFT→GRPO Two-Phase Pipeline as a Distilled RL Instance",
        "paper": "arxiv:2607.17247",
        "paper_title": "Distilled Reinforcement Learning for LLM Post-training",
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": "results/e2e_4cyc_gpt55/cycle_*/",
            "metric": (
                "Map MERA pipeline stages onto Distilled RL components: "
                "(1) teacher corpus = Phase-1 GPT-5.5 traces (SCALING_FORCE_BOTH=1); "
                "(2) negative sample reset = SFT on teacher solutions for all-fail tasks (Phase 3a); "
                "(3) selective transfer = router confidence threshold; "
                "(4) fine-grained credit = GRPO rollout advantage at temperature=1.0. "
                "Verify: SFT-initialized GRPO outperforms GRPO-only in MERA's ablation table."
            ),
            "expected_output": (
                "Formal mapping of MERA → Distilled RL components. "
                "The `skills` arm (always-small + SFT) vs `full` arm (SFT + GRPO + router) "
                "comparison directly tests whether Distilled RL's teacher-initialized RL "
                "gains hold in MERA's coding domain. Camera-ready §5.1 sentence: "
                "'MERA's two-phase SFT→GRPO design instantiates Distilled RL "
                "[arxiv:2607.17247], where Phase-1 teacher traces serve as negative sample "
                "reset initialization, reducing GRPO's credit-sparsity bootstrap cost.'"
            ),
            "camera_ready_target": "§5.1 SFT initialization motivation + §5.3 two-phase design justification",
            "estimated_time": "0.5h",
        },
        "rationale": (
            "arxiv:2607.17247 ('Distilled Reinforcement Learning for LLM Post-training', July 2026) "
            "proposes Distilled RL, which integrates teacher-model supervision directly into the RL "
            "training loop via negative sample reset: for all-fail student rollout groups, the teacher's "
            "correct solution replaces the student's trajectory, providing non-zero credit signal where "
            "GRPO produces zero advantage. MERA's architecture is a close structural match: Phase 1 runs "
            "GPT-5.5 (teacher) on every task via SCALING_FORCE_BOTH=1, producing teacher solutions that "
            "SFT-initialize the small model (Phase 3a) before GRPO (Phase 3b). Distilled RL's theory "
            "predicts that teacher-initialization reduces GRPO's credit-sparsity bootstrap cost — the "
            "exact mechanism MERA's ablation table (`skills` vs `full` arm) demonstrates empirically. "
            "An offline mapping of MERA's pipeline stages onto Distilled RL's four components provides "
            "a camera-ready §5.1 citation that positions MERA's SFT→GRPO ordering as a principled "
            "design choice (not an engineering convenience), strengthening the §5 narrative beyond the "
            "current empirical ablation. Not previously cited; 0h GPU; 0.5h analysis."
        ),
        "added": "2026-08-26T00:00:00Z",
        "camera_ready_priority": True,
    },
    {
        "id": "EXP-193",
        "priority": 7,
        "title": "Cue-GRPO-Audit: Bright-Room Gradient Loss Analysis for MERA Phase 3b",
        "paper": "arxiv:2608.03467",
        "paper_title": (
            "When Correct Solutions Repeat: Rarity-Aware Credit Redistribution for GRPO"
        ),
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": "results/e2e_4cyc_gpt55/cycle_*/rollouts/",
            "metric": (
                "Count all-success rollout groups (all K responses pass) in cycle-1 and cycle-3. "
                "Compute: bright_room_fraction = |all-pass groups| / |total groups|; "
                "bright_room_grad_loss = bright_room_fraction * avg_group_weight; "
                "Compare against dark_room_grad_loss (EXP-190 ACR=52.4% baseline). "
                "Verify GRPO_TEMPERATURE=1.0 provides within-group diversity on all-pass tasks "
                "(distinct solution strategies even when all pass → non-uniform natural advantage)."
            ),
            "expected_output": (
                "Expected: bright_room_fraction << dark_room_fraction (52.4%), because "
                "HumanEval hard tasks (which dominate cycle-3 training) rarely all pass. "
                "GRPO_TEMPERATURE=1.0 diversity ensures even all-pass groups have solution "
                "variation, partially mitigating uniform-advantage collapse. "
                "Camera-ready §5.4 sentence: 'MERA's GRPO_TEMPERATURE=1.0 prevents both "
                "dark-room collapse (zero credit, ACR=52.4%) and bright-room collapse "
                "(uniform credit for all-success groups [arxiv:2608.03467]), providing a "
                "complete two-sided motivation for the temperature hyperparameter (Design Rule #8).'"
            ),
            "camera_ready_target": "§5.4 GRPO temperature motivation — two-sided dark-room + bright-room framing",
            "estimated_time": "0.5h",
        },
        "rationale": (
            "arxiv:2608.03467 ('When Correct Solutions Repeat: Rarity-Aware Credit Redistribution "
            "for GRPO', August 2026) identifies the symmetric counterpart to the GRPO dark room "
            "problem: rollout groups where all K responses are correct ('bright room') produce "
            "uniform positive advantages → zero discriminative gradient signal. Cue-GRPO partitions "
            "correct responses into clusters using deterministic strategy cues and redistributes "
            "positive advantages proportional to cluster rarity. MERA's §5.3/5.4 dark room analysis "
            "(ACR=52.4%) currently addresses only the all-fail case; the bright-room case is "
            "unaddressed. An offline audit of MERA's cycle-1/3 rollout logs to count all-success "
            "groups and estimate bright-room gradient loss would: (a) provide the symmetric "
            "framing for §5.4, (b) validate that MERA's GRPO_TEMPERATURE=1.0 (Design Rule #8) "
            "provides enough within-group diversity to mitigate bright-room collapse, and "
            "(c) give a camera-ready §5.4 citation that completes the dark-room/bright-room "
            "temperature motivation story. The two-sided framing (dark room = all-fail → "
            "temperature avoids zero advantage; bright room = all-pass → temperature avoids "
            "uniform advantage) is cleaner and stronger than the current single-sided motivation "
            "in §5.4. Not previously cited; 0h GPU; 0.5h analysis."
        ),
        "added": "2026-08-26T00:00:00Z",
        "camera_ready_priority": True,
    },
]


def main():
    with open(STATE_PATH) as f:
        state = json.load(f)
    existing_ids = {e.get("id") for e in state.get("queue", [])}
    existing_ids |= {e.get("id") for e in state.get("history", [])}
    added = []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] not in existing_ids:
            state["queue"].append(exp)
            added.append(exp["id"])
        else:
            print(f"Skipping {exp['id']} — already in queue/history.")
    if not added:
        print("All experiments already queued — nothing to do.")
        return
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    shutil.move(tmp, STATE_PATH)
    print(f"Added: {', '.join(added)}. Queue length now: {len(state['queue'])}")


if __name__ == "__main__":
    main()
