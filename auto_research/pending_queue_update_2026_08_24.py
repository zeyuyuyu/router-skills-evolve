"""
Pending queue update — 2026-08-24
Run on A800 when connectivity is restored:
  python /data0/home/zeyuwang/auto_research/pending_queue_update_2026_08_24.py
Appends EXP-187 and EXP-188 to state["queue"] and saves atomically.

A800 offline since 2026-05-14 (day ~102). Queue ~186 pending (>20 cap → 2 today).
AAAI 2027 camera-ready window (2026-08-16 to ~2026-08-29), day 9. 5 days remain. CRITICAL.
GPU window: CLOSED. EXP-188 offline, no GPU needed.

NEW PAPERS TODAY:
  EXP-187: CoDistill-GRPO — Co-Distillation Reward Signal Audit for MERA Phase 3b (arxiv:2604.02288)
           KD reward from large model log-probs on Phase-1 rollouts + importance-reweighted large
           model training on small model traces. Offline audit using existing results/ traces. Priority 8.
  EXP-188: Skill-SD — Gated Multi-Skill vs. Single Global Skill Ablation (arxiv:2605.28791)
           Cluster HumanEval tasks by failure mode; measure procedure divergence across clusters.
           Validates Design Rule #1 (single global skill) or quantifies theoretical multi-skill ceiling.
           Offline, no code changes. Priority 7.

Apply chain before this patch:
    python3 auto_research/pending_queue_update_2026_08_23.py  # EXP-185, EXP-186
"""

import json, os, time, tempfile, shutil

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

NEW_EXPERIMENTS = [
    {
        "id": "EXP-187",
        "priority": 8,
        "title": "CoDistill-GRPO: Co-Distillation Reward Signal Audit for MERA Phase 3b",
        "paper": "arxiv:2604.02288",
        "type": "offline_analysis",
        "gpu": False,
        "description": (
            "Audit CoDistill-GRPO's KD reward signal (large-model log-prob ratio) on existing "
            "MERA Phase-1 run-both oracle traces. Estimate advantage-weight changes and routing "
            "impact. No GPU. Validates Phase 3b GRPO reward augmentation direction."
        ),
        "added": "2026-08-24T00:00:00Z",
    },
    {
        "id": "EXP-188",
        "priority": 7,
        "title": "Skill-SD: Gated Multi-Skill vs. Single Global Skill Ablation Audit",
        "paper": "arxiv:2605.28791",
        "type": "offline_analysis",
        "gpu": False,
        "description": (
            "Cluster HumanEval tasks (results/) by failure mode; measure SkillBook procedure "
            "divergence across clusters. Validates CLAUDE.md Design Rule #1 (single coding skill) "
            "or quantifies theoretical ceiling of multi-skill. No code changes."
        ),
        "added": "2026-08-24T00:00:00Z",
    },
]

def main():
    with open(STATE_PATH) as f:
        state = json.load(f)
    existing_ids = {e.get("id") for e in state.get("queue", [])}
    added = []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] not in existing_ids:
            state["queue"].append(exp)
            added.append(exp["id"])
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
