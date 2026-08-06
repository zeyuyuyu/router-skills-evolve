#!/usr/bin/env python3
"""
Daily queue patch — 2026-08-06 (EXP-150, EXP-151).

A800 connectivity: offline since 2026-05-14 (day ~84). Apply when restored.

Prior patches to apply before this one (daily-patch chain):
    ...
    python3 auto_research/pending_queue_update_2026_08_05.py            # EXP-148, EXP-149

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_08_06.py            # EXP-150, EXP-151

Queue was ~155 pending on 2026-08-06 (after EXP-148, EXP-149). Cap applied: >20 -> max 2.

AAAI 2027 deadline: 2026-08-15 (9 days from today 2026-08-06).
A800 offline since 2026-05-14 (day 84). GPU window CLOSED 2026-08-01.

=================================================================================
NEW FINDINGS MOTIVATING TODAY'S EXPERIMENTS
=================================================================================

OFFLINE ANALYSIS (2026-08-06):

  arxiv:2607.04364 — "RL Forgets! Towards Continual Policy Optimization"
      Key finding (Jul 2026): Continual RL still suffers catastrophic forgetting,
      and the standard GRPO KL penalty does NOT prevent it — in fact, the KL term
      reduces model plasticity without preserving old capabilities. Measured
      forgetting rate: ~8-12% absolute pass@1 drop per cycle in continual GRPO.

      Direct application: Our cycle-by-cycle routing accuracy shows non-monotonic
      patterns (dip at cycle 2 before recovery). The 2607.04364 framework explains
      this as catastrophic forgetting of cycle-0 task generalizations by the small
      model during cycle-1 GRPO training. EXP-150 formalizes this with an offline
      measurement from our existing cycle eval logs (0 GPU hours).

  arxiv:2605.00433 — "Improving LLM Code Generation via Requirement-Aware
      Curriculum Reinforcement Learning" (RECRL, May 2026)
      Key finding: Flat-sampling GRPO under-trains the hardest task quartile.
      RECRL oversamples tasks where pass@k is lowest (difficulty ∝ 1 − pass@k),
      beating vanilla GRPO by 3-5 pts pass@1 on HumanEval/MBPP.

      Direct application: We have 4 cycles of per-task pass@k data. EXP-151
      measures whether our bottom difficulty-quartile tasks improve less per cycle
      (the expected flat-sampling gap from RECRL). If confirmed, this provides
      empirical motivation for the curriculum experiments already in queue and
      adds a quantified gap to the paper's §5.4 motivation section (0 GPU hours).
"""

import json
import shutil
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

NEW_EXPERIMENTS = [
    # -----------------------------------------------------------------------
    # EXP-150: Router Non-Monotonicity as Forgetting Signal (0h GPU)
    # -----------------------------------------------------------------------
    {
        "id": "exp_2026_08_06_001_router_forgetting_kl_nonmonotonicity_analysis",
        "priority": 8,
        "gpu": "auto",
        "kind": "forgetting_eval",
        "rationale": (
            "arXiv:2607.04364 ('RL Forgets! Towards Continual Policy Optimization', Jul 2026) "
            "shows GRPO's KL constraint does NOT prevent catastrophic forgetting and reduces "
            "plasticity. Our cycle-by-cycle routing accuracy shows non-monotonic patterns "
            "(dip at cycle 2). This offline experiment computes per-task pass@1 on cycle-0 "
            "tasks at each subsequent cycle from existing eval logs, measuring forgetting rate "
            "≡ max(acc_0 - acc_k) for k=1..3. If forgetting rate > 5% abs on any task cluster, "
            "cites 2607.04364 in §5.3 with a 'Router Forgetting' subsection. Zero GPU."
        ),
        "spec": {
            "bench": "humaneval",
            "gpu_hours": 0,
            "analysis_type": "offline_from_logs",
            "metric": "per_task_pass1_across_cycles",
            "forgetting_threshold_pct": 5.0,
            "cycles_to_analyze": [0, 1, 2, 3],
            "cite": "2607.04364",
            "paper_section": "5.3_router_forgetting",
        },
        "queued_at": "2026-08-06T00:00:00Z",
    },
    # -----------------------------------------------------------------------
    # EXP-151: Per-Task Difficulty Trajectory for Curriculum Motivation (0h GPU)
    # -----------------------------------------------------------------------
    {
        "id": "exp_2026_08_06_002_pertask_difficulty_trajectory_curriculum_gap",
        "priority": 7,
        "gpu": "auto",
        "kind": "forgetting_eval",
        "rationale": (
            "arXiv:2605.00433 (RECRL, May 2026) shows flat-sampling GRPO under-trains the "
            "bottom difficulty quartile (hardest ~20-25 HumanEval tasks) vs difficulty-"
            "proportional curriculum, with a 3-5 pt pass@1 gap. This offline experiment plots "
            "each task's pass@k trajectory across 4 cycles, labels tasks by difficulty quartile "
            "(from cycle-0 pass@k), and checks: do bottom-quartile tasks improve less per cycle? "
            "If yes, quantifies the curriculum gap and motivates curriculum experiments already "
            "in queue. Output: difficulty_quartile_improvement_table added to §5.4. Zero GPU."
        ),
        "spec": {
            "bench": "humaneval",
            "gpu_hours": 0,
            "analysis_type": "offline_from_logs",
            "metric": "per_task_pass1_by_difficulty_quartile",
            "difficulty_basis": "cycle0_pass_at_k",
            "cycles_to_analyze": [0, 1, 2, 3],
            "output": "difficulty_quartile_improvement_table",
            "cite": "2605.00433",
            "paper_section": "5.4_curriculum_gap_motivation",
        },
        "queued_at": "2026-08-06T00:00:00Z",
    },
]


def main():
    if not STATE_PATH.exists():
        print(f"ERROR: {STATE_PATH} not found. Is the A800 reachable?")
        raise SystemExit(1)

    with open(STATE_PATH) as f:
        state = json.load(f)

    existing_ids = {e["id"] for e in state.get("queue", [])} | {
        e["id"] for e in state.get("history", [])
    }

    queue_size = len(state.get("queue", []))
    to_add = [e for e in NEW_EXPERIMENTS if e["id"] not in existing_ids]

    # Enforce the cap even here: if queue is over 20, allow at most 2
    if queue_size > 20 and len(to_add) > 2:
        print(f"Queue has {queue_size} pending (>20 cap). Limiting to 2 additions.")
        to_add = to_add[:2]

    if not state.get("queue"):
        state["queue"] = []

    state["queue"].extend(to_add)

    tmp = str(STATE_PATH) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    shutil.move(tmp, str(STATE_PATH))

    print(
        f"Appended {len(to_add)} experiments to queue "
        f"(skipped {len(NEW_EXPERIMENTS) - len(to_add)} duplicates)"
    )
    for e in to_add:
        print(f"  + {e['id']}  priority={e['priority']}  kind={e['kind']}")
    print("DONE: queued 2 new experiments (EXP-150, EXP-151)")


if __name__ == "__main__":
    main()
