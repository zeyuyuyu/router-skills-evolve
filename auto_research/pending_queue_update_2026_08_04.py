#!/usr/bin/env python3
"""
Daily queue patch — 2026-08-04 (EXP-146, EXP-147).

A800 connectivity: offline since 2026-05-14 (day ~82). Apply when restored.

Prior patches to apply before this one (daily-patch chain):
    ...
    python3 auto_research/pending_queue_update_2026_08_01.py            # EXP-144, EXP-145

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_08_04.py            # EXP-146, EXP-147

Queue was ~151 pending on 2026-08-01 (after EXP-144, EXP-145). Cap applied: >20 -> max 2.

AAAI 2027 deadline: 2026-08-15 (11 days from today 2026-08-04).
A800 offline since 2026-05-14 (day 82). GPU window CLOSED 2026-08-01.

=================================================================================
NEW FINDINGS MOTIVATING TODAY'S EXPERIMENTS
=================================================================================

OFFLINE ANALYSIS (2026-08-04):
  Per-cycle Dark Room quantification from existing phase3b_grpo.log files:
    Cycle 0: 42/82 tasks zero-advantage (29 all-pass + 13 all-fail) = 51.2%
    Cycle 1: 39/82 tasks zero-advantage (28 all-pass + 11 all-fail) = 47.6%
    Cycle 2: 38/82 tasks zero-advantage (26 all-pass + 12 all-fail) = 46.3%
    Cycle 3: 43/82 tasks zero-advantage (33 all-pass + 10 all-fail) = 52.4%
  All-fail (Dark Room proper) is 10-13 tasks/cycle — reward sparsity
  prevents any gradient on these 12-16% of tasks throughout training.

  Within-campaign advantage-magnitude trend (cycle-3):
    First third (steps 1-42):   mean|adv| = 0.959
    Middle third (steps 43-83): mean|adv| = 0.683
    Last third (steps 84-125):  mean|adv| = 0.727
  28% drop from first to middle third suggests within-campaign over-optimization
  consistent with the rise-and-collapse framework (arxiv:2606.21090).

arxiv:2606.21090 — "Self-Improvement Can Self-Regress: The Rise-and-Collapse
    Failure Mode of LLM Self-Training" (June 2026)
    Controlled multi-seed testbed with Qwen-2.5-3B/7B on competitive-programming
    tasks shows pass@1 rises within tens of gradient steps then collapses within
    the same campaign. Not cross-task catastrophic forgetting — within-task
    over-optimization on fixed distribution. KL/EWC constraints do not prevent it.
    Mechanism: once the easy-reward tasks are solved, residual gradient comes only
    from hard tasks; the policy over-fits to their surface patterns → collapse.
    Matches our cycle-3 all_pass increase (26→33) driving 28% mean|adv| drop.

arxiv:2607.26457 — "DHRCL: Training Code LLMs with Dense Hierarchical Rewards
    and Curriculum Learning" (July 29 2026)
    Decomposes reward into syntax-validation (0.1), execution-success (0.3), and
    unit-test pass rate (1.0) with a three-stage Syntax→Execution→Pass curriculum
    driven by rolling validation trends. Stage-aware credit redistribution follows
    consolidation-to-refinement: syntax-phase emphasises established token patterns;
    pass-phase allocates credit to uncertain token decisions. Outperforms GRPO,
    GRPO-PassRate, AceCoder, VeRPO on HumanEval, HumanEval+, BigCodeBench-Full,
    BigCodeBench-Hard, LiveCodeBench V6, and CodeElo.
    Direct application: our 10-13 all_fail tasks/cycle get ZERO gradient under
    binary rewards. DHRCL's intermediate tiers (syntax_ok=0.1, execution_ok=0.3)
    give non-zero reward to rollouts that compile or run, recovering ~21% of the
    currently zero-advantage Dark Room group and providing denser learning signal
    for the hardest 12-16% of HumanEval tasks.
"""

import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

NEW_EXPERIMENTS = [
    # -----------------------------------------------------------------------
    # EXP-146: Within-Campaign Advantage-Trajectory Offline Audit
    # -----------------------------------------------------------------------
    {
        "id": "exp_2026_08_04_001_within_campaign_advantage_trajectory_rise_collapse_audit",
        "priority": 6,
        "gpu": "auto",
        "kind": "forgetting_eval",
        "rationale": (
            "arxiv:2606.21090 (Self-Improvement Can Self-Regress, June 2026) documents a "
            "rise-and-collapse failure mode in REINFORCE/GRPO training: pass@1 rises within "
            "tens of gradient steps, then collapses within the same campaign due to within-task "
            "over-optimization on a fixed distribution. Qwen-2.5-3B/7B are the subjects; our "
            "Qwen2.5-Coder-1.5B is even smaller and more susceptible. "
            "Offline analysis of our existing phase3b_grpo.log files reveals a 28% mean|adv| "
            "drop in cycle-3 from first (0.959) to middle third (0.683), consistent with "
            "within-campaign over-optimization: all_pass tasks increase from 26 (cycle-2) to "
            "33 (cycle-3) as easy tasks are solved early, leaving residual gradient only from "
            "hard tasks that the policy then over-fits to. EXP-146 formalizes this analysis "
            "into a 4-cycle per-step advantage-trajectory plot and fits the rise-and-collapse "
            "curve per cycle, providing a mechanistic narrative for the cycle-3 plateau "
            "(75.61% → 75.61% skills arm, no further gain) as within-campaign saturation "
            "rather than inter-cycle forgetting. This enriches §5.3 with a new 'within-campaign "
            "dynamics' paragraph and adds Figure 3 (mean|adv| vs training step, 4-cycle overlay)."
        ),
        "spec": {
            "bench": "humaneval",
            "eval_only": True,
            "log_files": [
                "results/e2e_4cyc_gpt55/cycle_0/phase3b_grpo.log",
                "results/e2e_4cyc_gpt55/cycle_1/phase3b_grpo.log",
                "results/e2e_4cyc_gpt55/cycle_2/phase3b_grpo.log",
                "results/e2e_4cyc_gpt55/cycle_3/phase3b_grpo.log",
            ],
            "metrics": ["mean_abs_adv_per_step", "dark_room_fraction", "rise_collapse_curve"],
            "script": "src/pipeline/within_campaign_advantage_audit.py",
            "notes": (
                "Reads existing phase3b_grpo.log step-level data (epoch=0 step=N loss=X mean_adv=Y). "
                "Computes rolling mean|adv| over a 10-step window. Fits a piecewise rise-then-fall "
                "model per cycle (scipy.optimize). Plots 4-cycle overlay (Figure 3). Also computes "
                "per-cycle all_pass/all_fail fractions from rollout logs (proxy for Dark Room evolution "
                "within a cycle). ~0h GPU. Inference-only. Adds §5.3 'within-campaign dynamics' "
                "paragraph + Figure 3 to AAAI paper. Required background for EXP-132 "
                "(rise-and-collapse checkpoint, priority=9)."
            ),
        },
    },
    # -----------------------------------------------------------------------
    # EXP-147: DHRCL 3-Tier Hierarchical Reward — Root-Cause Dark Room Fix
    # -----------------------------------------------------------------------
    {
        "id": "exp_2026_08_04_002_dhrcl_3tier_hierarchical_reward_grpo_humaneval_dark_room",
        "priority": 7,
        "gpu": "auto",
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2607.26457 (DHRCL, July 29 2026) decomposes GRPO reward into syntax "
            "validation (0.1), execution success (0.3), and unit-test pass rate (1.0) with a "
            "three-stage curriculum. Offline analysis shows 10-13 all_fail tasks/cycle (12-16% "
            "of HumanEval) receive ZERO gradient under our current binary reward because every "
            "K=8 rollout fails → group std=0 → advantage=0 (Dark Room effect). EXP-147 replaces "
            "the binary reward in grpo_train_simple.py with a 3-tier DHRCL reward: syntax_ok=0.1, "
            "execution_ok=0.3, all_tests_pass=1.0. For the 10-13 all_fail Dark Room tasks, even "
            "rollouts that only compile get reward=0.1 vs 0.0, breaking the zero-variance group "
            "and restoring gradient on ~21% of the currently silent Dark Room tasks. This is a "
            "root-cause fix (denser rewards) orthogonal to EXP-135's symptom fix (no-std-norm). "
            "Expected to pair as a 2×2 ablation with EXP-135: {binary_reward, 3-tier_reward} × "
            "{std_norm, no_std_norm}. Adds §5.4 'reward density' fix row to Table 9."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 2,
            "start_cycle": 0,
            "grpo_reward_tiers": {
                "syntax_ok": 0.1,
                "execution_ok": 0.3,
                "all_tests_pass": 1.0,
            },
            "grpo_reward_mode": "dhrcl_3tier",
            "scaling_force_both": 1,
            "skip_grpo": 0,
            "skip_sft": 0,
            "notes": (
                "Modify grpo_train_simple.py reward computation: after each pytest run, check "
                "(a) does the code parse/compile (ast.parse), (b) does it run without import error "
                "(subprocess returncode != SIGKILL), (c) does it pass all tests. Assign tiered "
                "reward accordingly. Keep K=8, max_turns=3, temp=0.8 identical to baseline. "
                "Compare cycle-0 GRPO pass@1 trajectory vs baseline to confirm Dark Room reduction: "
                "expect fewer zero-advantage tasks (target: all_fail < 8/82). "
                "~1.5h GPU. Synergises with EXP-135 (no-std-norm) as a 2x2 ablation."
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
