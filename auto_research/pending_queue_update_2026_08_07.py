#!/usr/bin/env python3
"""
Daily queue patch — 2026-08-07 (EXP-150, EXP-151).

A800 connectivity: offline since 2026-05-14 (day ~85). Apply when restored.

Prior patches to apply before this one (daily-patch chain):
    ...
    python3 auto_research/pending_queue_update_2026_08_05.py            # EXP-148, EXP-149

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_08_07.py            # EXP-150, EXP-151

Queue was ~155 pending on 2026-08-05 (after EXP-148, EXP-149). Cap applied: >20 -> max 2.

AAAI 2027 deadline: 2026-08-15 (8 days from today 2026-08-07).
A800 offline since 2026-05-14 (day 85). GPU window CLOSED 2026-08-01.
Both experiments are OFFLINE / 0h GPU — immediately runnable on local repo.

=================================================================================
NEW FINDINGS MOTIVATING TODAY'S EXPERIMENTS
=================================================================================

arxiv:2608.00782 — "Distill Where You Fail: Recovering Learning Signals of Negative
    RL-Groups from Adaptive Teacher Guidance" (August 2026, RSTG)
    Standard GRPO silently discards gradient on zero-variance groups (all rollouts
    receive identical reward — both all-pass AND all-fail cases). RSTG detects these
    negative zero-variance groups and injects targeted teacher supervision: for
    partial-failure groups it uses on-policy distillation (OPD) with per-token
    weighting by teacher confidence; for fully all-fail groups it applies SFT on
    correct teacher trajectories. The result: gradient is recovered on the previously
    dead groups without changing the reward function or the policy architecture.
    This is the 4th orthogonal dark room mechanism (EXP-135: std-norm fix;
    EXP-147: DHRCL 3-tier reward; EXP-148: OC-GRPO privileged prefix; EXP-150: RSTG
    teacher distillation on all-fail). The approach requires teacher traces for all-fail
    tasks — with SCALING_FORCE_BOTH=1 our pipeline ran GPT-5.5 on every task, so
    teacher completions should be available in traces.jsonl for the 10-13 all_fail tasks
    per cycle. EXP-150 verifies this coverage as a pre-requisite feasibility check.

arxiv:2607.07847 — "When Does Continual Learning Require Learning?" (July 2026)
    Benchmarks continual post-training methods (GEPA, ACE, SFT/SDFT, GRPO, SDPO)
    across a sequential-task curriculum. Key finding: RL-based methods (GRPO in
    particular) exhibit the sharpest forgetting curves across earlier tasks, even
    when replay is added. The benchmark directly tests the interaction of learning
    efficiency vs. retention that our pipeline is designed to address via the router
    arm. EXP-151 maps our 4-cycle data onto the paper's per-task retention metric,
    providing a concrete citation anchor for §5.2 (Forgetting section).
"""

import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

NEW_EXPERIMENTS = [
    # -----------------------------------------------------------------------
    # EXP-150: RSTG All-Fail Teacher Trace Feasibility Audit
    # -----------------------------------------------------------------------
    {
        "id": "exp_2026_08_07_001_rstg_dark_room_teacher_trace_feasibility_audit",
        "priority": 8,
        "gpu": "auto",
        "kind": "forgetting_eval",
        "rationale": (
            "arxiv:2608.00782 (RSTG, August 2026) proposes injecting targeted teacher "
            "distillation for zero-variance GRPO groups: SFT on correct teacher trajectories "
            "for confirmed all-fail prompts. This is the 4th orthogonal dark room mechanism, "
            "complementing EXP-135 (std-norm fix), EXP-147 (DHRCL reward tiers), and "
            "EXP-148 (OC-GRPO privileged prefix). RSTG requires teacher traces to be "
            "available for all-fail tasks. With SCALING_FORCE_BOTH=1, our pipeline ran "
            "GPT-5.5 on every HumanEval task in every cycle, storing completions in "
            "traces.jsonl. EXP-150 audits these trace files to verify: (a) coverage — "
            "what fraction of confirmed all_fail tasks (identified from phase3b_grpo.log) "
            "have a corresponding large_model.completion entry in traces.jsonl; (b) quality "
            "— mean teacher trace length and pass rate for all_fail tasks (RSTG SFT quality "
            "floor); (c) RSTG pipeline compatibility — can RSTG be grafted onto "
            "grpo_train_simple.py with the existing traces.jsonl format? "
            "Output: feasibility table (4 cycles × 3 metrics) + Y/N compatibility verdict. "
            "Zero GPU. Immediately runnable from local repo on results/e2e_4cyc_gpt55/. "
            "AAAI value: §5.4 ('Dark Room Remedies') gains a 4th mechanism cite for a "
            "more complete survey; Future Work gains a concrete implementable next step "
            "with evidence that the data prerequisite is already satisfied."
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
            "trace_files": [
                "results/e2e_4cyc_gpt55/cycle_0/traces.jsonl",
                "results/e2e_4cyc_gpt55/cycle_1/traces.jsonl",
                "results/e2e_4cyc_gpt55/cycle_2/traces.jsonl",
                "results/e2e_4cyc_gpt55/cycle_3/traces.jsonl",
            ],
            "metrics": [
                "allfail_task_ids_per_cycle",
                "teacher_trace_coverage_rate",
                "mean_teacher_trace_length_allfail",
                "teacher_pass_rate_on_allfail_tasks",
                "rstg_compatibility_verdict",
            ],
            "script": "src/pipeline/rstg_feasibility_audit.py",
            "notes": (
                "New ~40-line script. Steps: "
                "(1) Parse phase3b_grpo.log for each cycle: identify all_fail task IDs "
                "(tasks where all K=8 rollouts have reward=0). "
                "(2) Load traces.jsonl: for each all_fail task_id, check if a "
                "large_model.completion entry exists (SCALING_FORCE_BOTH=1 should give ~100%). "
                "(3) For covered tasks: measure teacher completion length (token count) and "
                "teacher pass rate (was the teacher correct on that task?). "
                "(4) Compatibility check: confirm traces.jsonl fields match what "
                "grpo_train_simple.py would need for SFT insertion. "
                "Expected: ~100% coverage (SCALING_FORCE_BOTH=1), teacher length >>  "
                "student rollout length, teacher pass rate ~0.7-0.9 on all_fail tasks "
                "(hard tasks — teacher may also fail some). "
                "If teacher pass rate < 0.5 on all_fail tasks, flag: RSTG SFT signal "
                "may be weak for the hardest tasks — privilege prefix (EXP-148) may be "
                "preferred. Cite arxiv:2608.00782 §3.2 for the OPD/SFT split criterion."
            ),
        },
    },
    # -----------------------------------------------------------------------
    # EXP-151: Continual GRPO Per-Task Retention Profile vs. arxiv:2607.07847
    # -----------------------------------------------------------------------
    {
        "id": "exp_2026_08_07_002_continual_grpo_per_task_retention_profile_vs_2607_07847",
        "priority": 7,
        "gpu": "auto",
        "kind": "forgetting_eval",
        "rationale": (
            "arxiv:2607.07847 ('When Does Continual Learning Require Learning?', Jul 2026) "
            "benchmarks RL-based continual post-training methods and finds that GRPO exhibits "
            "the sharpest per-task forgetting curves among all methods tested — sharper than "
            "SFT/SDFT and far sharper than prompt-based methods (GEPA, ACE). Their Figure 3 "
            "plots per-task retention rate (pass@1 at cycle T / pass@1 at cycle 0) across "
            "sequential tasks for each method. EXP-151 replicates this analysis on our "
            "4-cycle continual GRPO run: for each of the 82 HumanEval tasks, compute the "
            "per-cycle retention rate from the eval logs in results/e2e_4cyc_gpt55/, then "
            "split by routing arm (large-only, small+skill, router, full). The hypothesis "
            "is that the router arm preserves retention better than the skills arm (always "
            "small) because routing hard tasks to the large model prevents the small model "
            "from over-specializing on them and forgetting easy tasks. Zero GPU. "
            "Output: 4-panel Figure (one panel per arm) showing per-task pass@1 curves "
            "across 4 cycles + mean retention rate per arm. AAAI value: §5.2 (Forgetting) "
            "gains a concrete citation ('GRPO's sharp forgetting profile, consistent with "
            "arxiv:2607.07847 Figure 3, motivates the router arm...') and Figure 4 of "
            "our paper is populated with a real retention plot."
        ),
        "spec": {
            "bench": "humaneval",
            "eval_only": True,
            "eval_log_dir": "results/e2e_4cyc_gpt55/",
            "metrics": [
                "per_task_pass_at_1_cycle_0_to_3",
                "per_task_retention_rate",
                "mean_retention_by_arm",
                "forgetting_task_count_per_cycle",
                "retention_distribution_iqr",
            ],
            "output": [
                "results/retention_profile_4cyc.png",
                "results/retention_table_by_arm.csv",
            ],
            "script": "src/pipeline/per_task_retention_profile.py",
            "notes": (
                "New ~50-line script. Steps: "
                "(1) For each cycle c in [0,1,2,3], load the per-task eval results from "
                "results/e2e_4cyc_gpt55/cycle_c/e2e_ablation_summary.json (the 4-arm "
                "ablation results already contain per-task pass@1 by arm). "
                "(2) Compute retention_rate[arm][task][c] = pass1[c] / pass1[0] "
                "(clip denominator at 0.05 to avoid zero-div on very hard tasks). "
                "(3) For each arm, plot mean retention ± stderr across all 82 tasks over "
                "4 cycles. Expected: router and full arms show flatter retention curves "
                "(less forgetting) than skills (always-small) and large (never fine-tuned "
                "small model) arms. "
                "(4) Compute 'forgetting count' per cycle: tasks where retention < 0.8. "
                "Cite arxiv:2607.07847 Figure 3 in §5.2: our router arm retention curve "
                "is predicted to be shallower, consistent with the hypothesis that "
                "adaptive routing prevents GRPO over-specialization on hard tasks. "
                "~0h GPU. Immediately runnable if eval logs are present."
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
