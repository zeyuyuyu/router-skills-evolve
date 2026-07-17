#!/usr/bin/env python3
"""
Weekly paper-pipeline queue patch — 2026-07-17 (EXP-122, EXP-123).

A800 connectivity: offline since 2026-05-14 (day ~64). Apply when restored.

Added by: AAAI paper pipeline (weekly, Friday 11:00 UTC)

Prior patches to apply before this one (daily-patch chain):
    python3 auto_research/pending_queue_update.py
    python3 auto_research/pending_queue_update_2026_06_06.py
    ... (EXP-054 through EXP-119 — see pending_queue_update_2026_07_16.py for full chain)
    python3 auto_research/pending_queue_update_2026_07_16.py            # EXP-118, EXP-119
    python3 auto_research/pending_queue_update_2026_07_17.py            # EXP-120, EXP-121

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_17_paper.py      # EXP-122, EXP-123

Queue was ~131 pending on 2026-07-17 (+2 added: EXP-120, EXP-121 by daily patch).

Data motivating today's proposals
----------------------------------
results/e2e_4cyc_gpt55/cycle_3/e2e_ablation_summary.json (unchanged — A800 offline day 64):
    HumanEval 4-cycle:
      large (always-large):        task_pass=96.34%, cost_vs_large=100%
      skills (always-small+proc):  task_pass=75.61%, cost_vs_large=10%
      router (logistic, cycle 3):  task_pass=92.68%, routing_acc=92.68%,
                                   cost_vs_large=27.56%, fallback=6.10%
      full (router+GRPO):          task_pass=92.68% ← SAME as router
    Root problems:
      (A) ACR=52.4% — 43/82 GRPO groups have zero within-group reward variance
      (B) Skills gap: 75.61% vs large 96.34% — 20.7pp gap

AAAI 2027 deadline: 2026-08-15 (29 days from today).
A800 offline since 2026-05-14 (day 64).

New papers motivating these experiments:
-----------------------------------------
arxiv:2607.00152 — "GRPO, Dr. GRPO, and DAPO Are Three Operations on One Number:
    The Group-Standard-Deviation Identity" (Bay & Yearick, June 30, 2026)

    KEY RESULT: GRPO/Dr.GRPO/DAPO differ only in treatment of within-group σ.
    Dr. GRPO drops the 1/σ division → update magnitude ∝ p*(1-p) for binary rewards.
    This is the CLEANEST fix for our ACR=52.4% problem: no extra hyperparameters,
    no external curriculum, ~5 lines of code change in grpo_train_simple.py.

    EXP-120 (daily patch, July 17) tests Dr.GRPO + sign-advantage fallback in isolation.
    EXP-122 (this patch) tests Dr.GRPO JOINTLY with ZPD-SFT — testing whether the
    two orthogonal fixes (A: zero-variance collapse, C: skills gap) are additive or
    interact. This is the highest-priority combined experiment for AAAI narrative.

arxiv:2507.00054 — "Enhancing Reasoning Capabilities in SLMs with Reward Guided
    Dataset Distillation" (July 2026)

    KEY RESULT: ZPD filtering of SFT traces (r_small ∈ [lo, hi]) gives +3.1-5.6pp
    over naive SFT at identical example counts on code benchmarks.

    EXP-121 (daily patch, July 17) tests ZPD-SFT in isolation.
    EXP-122 (this patch) tests ZPD-SFT + Dr.GRPO jointly.
    EXP-123 (this patch) is a zero-GPU analysis experiment using existing trace logs.
"""

import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = "data/humaneval_eval.jsonl"

NEW_EXPERIMENTS = [
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-122: Joint ZPD-SFT + Dr. GRPO — tests interaction of A+C fixes
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_17_paper_001_joint_zpd_sft_dr_grpo_humaneval_4cycle",
        "priority": 9,
        "gpu": "auto",
        "kind": "grpo_continual",
        "rationale": (
            "Addresses reviewer weaknesses W9 (Group-SD Identity creates gap — why not "
            "use Dr.GRPO?) and W10 (skills gap 20.7pp unexplained). EXP-120 (Dr.GRPO "
            "alone) and EXP-121 (ZPD-SFT alone) are now queued as of 2026-07-17. "
            "However, AAAI reviewers will ask: do these two orthogonal fixes (problem A: "
            "zero-variance collapse; problem C: SFT data quality) interact when combined? "
            "If A is the bottleneck to Full=Router, fixing C alone may not show in the "
            "Full arm. If C is the bottleneck to the skills arm, fixing A alone may not "
            "improve the skills arm. The joint experiment distinguishes these cases and "
            "provides the paper's strongest possible result: if skills arm ≥ 78% AND "
            "Full arm > 93%, the joint fix narrative (two complementary fixes, additive "
            "benefits, each grounded by a 2026 paper) becomes the paper's headline. "
            "arxiv:2607.00152 (Dr.GRPO, Group-SD Identity) + arxiv:2507.00054 (ZPD). "
            "AAAI 2027 deadline 2026-08-15 (29 days). A800 must restore by ~July 28 "
            "for this result to make it into the submission."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 4,
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "dr_grpo_enabled": True,
            "dr_grpo_no_std_division": True,
            "dr_grpo_zero_var_fallback": "sign_advantage",
            "zpd_sft_enabled": True,
            "zpd_lo": 0.20,
            "zpd_hi": 0.80,
            "zpd_filter_field": "small_pass_rate",
            "grpo_temperature": 1.0,
            "n_generations": 8,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "analysis": [
                "skills_arm_pass_at_1_joint_vs_individual_fixes",
                "full_arm_pass_at_1_joint_vs_baseline",
                "acr_per_cycle_joint_vs_dr_grpo_only",
                "n_sft_examples_per_cycle_zpd_filtered",
                "interaction_effect_dr_grpo_x_zpd",
            ],
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (baseline): full 92.68%, skills 75.61%, ACR=52.4%",
                "EXP-120 (Dr.GRPO alone): problem A fix",
                "EXP-121 (ZPD-SFT alone): problem C fix",
                "Target: skills arm >= 78%, full arm >= 93.0%, ACR < 45%",
            ],
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
                "src/pipeline/traces_to_sft.py",
            ],
            "arxiv_refs": ["2607.00152", "2507.00054"],
            "estimated_gpu_hours": 2.5,
            "run_after": ["EXP-120", "EXP-121"],
            "note": (
                "Run after EXP-120 and EXP-121 complete to confirm interaction. "
                "If EXP-120 and EXP-121 individually show no improvement, this joint "
                "experiment may still show benefit via the interaction effect. "
                "If EITHER individual experiment shows improvement, this is likely additive."
            ),
        },
    },
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-123: Zero-GPU Group-SD ceiling analysis (offline trace analysis)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_17_paper_002_group_sd_ceiling_analysis_offline",
        "priority": 8,
        "gpu": "none",
        "kind": "analysis_offline",
        "rationale": (
            "arxiv:2607.00152 Group-SD Identity predicts: for binary rewards, GRPO update "
            "magnitude ∝ 1/σ (unstable), Dr.GRPO ∝ p*(1-p) (stable, informativeness-weighted). "
            "Given our existing cycle_3 trace logs (which store per-prompt binary rewards for "
            "G=8 DAPO rollouts), we can COMPUTE the σ distribution directly from those logs "
            "and verify: (1) that 52.4% have σ=0 (all-pass or all-fail), (2) the distribution "
            "of p for the remaining 47.6% (i.e., the Dr.GRPO update magnitude distribution), "
            "(3) the THEORETICAL maximum gradient ratio: if we had used Dr.GRPO instead of "
            "GRPO, what fraction of the σ=0 groups would have received non-zero gradient? "
            "This is a ZERO-GPU experiment runnable on existing saved trace data. "
            "Output: a table of p-values and σ-values per HumanEval task (cycle 3), "
            "a histogram of p*(1-p) vs current 1/σ effective magnitudes, and the theoretical "
            "ceiling analysis for Dr.GRPO. This directly supports the §7 addition in v6 and "
            "provides concrete data for the paper without requiring A800 to be online. "
            "CRITICAL: This can run locally on the existing e2e_4cyc_gpt55 trace files "
            "in results/e2e_4cyc_gpt55/cycle_3/ — check for traces.jsonl or rollout_rewards "
            "files in that directory."
        ),
        "spec": {
            "bench": "humaneval",
            "data_source": "results/e2e_4cyc_gpt55/cycle_3/traces.jsonl",
            "analysis_type": "group_sd_identity_verification",
            "compute_fields": [
                "per_prompt_pass_rate_p",
                "per_prompt_sigma",
                "grpo_effective_magnitude_1_over_sigma",
                "dr_grpo_effective_magnitude_p_times_1_minus_p",
                "zero_variance_group_count",
                "histogram_p_values_for_nonzero_sigma",
                "theoretical_gradient_ratio_dr_grpo_vs_grpo",
            ],
            "output_files": [
                "results/e2e_4cyc_gpt55/cycle_3/group_sd_analysis.json",
                "results/e2e_4cyc_gpt55/cycle_3/group_sd_analysis_plot.png",
            ],
            "paper_target": (
                "If analysis confirms theoretical predictions from 2607.00152: add as "
                "Figure 1 (p*(1-p) distribution curve) or Table 9 (Group-SD analysis). "
                "This is the only zero-GPU experiment that could strengthen soundness "
                "without A800, and it directly validates our §7 theoretical claims."
            ),
            "arxiv_ref": "2607.00152",
            "estimated_gpu_hours": 0.0,
            "estimated_cpu_hours": 0.1,
            "note": (
                "Check if traces.jsonl for cycle_3 exists locally in results/ first. "
                "If not, check if rollout_rewards are stored in grpo_info.json or "
                "training_info.json under cycle_3/grpo_adapter/. "
                "Implementation: Python script analyzing existing JSON files, "
                "no new model runs required. Could be run TODAY on local machine."
            ),
        },
    },
]


def atomic_save(path: Path, obj) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


def main():
    with open(STATE_PATH) as f:
        state = json.load(f)

    queue = state.get("queue", [])
    history = state.get("history", [])
    existing_ids = {e["id"] for e in queue} | {e["id"] for e in history}

    added = []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] in existing_ids:
            print(f"  SKIP (already queued): {exp['id']}")
            continue
        queue.append(exp)
        existing_ids.add(exp["id"])
        added.append(exp["id"])
        print(f"  ADDED: {exp['id']}")

    state["queue"] = queue
    atomic_save(STATE_PATH, state)
    print(f"\nDone. Added {len(added)} experiments. Total queue: {len(queue)} pending.")


if __name__ == "__main__":
    main()
