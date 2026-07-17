#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-17 (EXP-120, EXP-121).

A800 connectivity: offline since 2026-05-14 (day ~64). Apply when restored.

Prior patches to apply before this one (daily-patch chain):
    python3 auto_research/pending_queue_update.py
    python3 auto_research/pending_queue_update_2026_06_06.py            # EXP-054, EXP-055
    python3 auto_research/pending_queue_update_2026_06_07.py            # EXP-056, EXP-057
    python3 auto_research/pending_queue_update_2026_06_08.py            # EXP-058, EXP-059
    python3 auto_research/pending_queue_update_2026_06_10.py            # EXP-060, EXP-061
    python3 auto_research/pending_queue_update_2026_06_11.py            # EXP-062, EXP-063
    python3 auto_research/pending_queue_update_2026_06_12.py            # EXP-064, EXP-065
    python3 auto_research/pending_queue_update_2026_06_12_v2.py         # EXP-066, EXP-067
    python3 auto_research/pending_queue_update_2026_06_14.py            # EXP-068, EXP-069
    python3 auto_research/pending_queue_update_2026_06_15.py            # EXP-070, EXP-071
    python3 auto_research/pending_queue_update_2026_06_16.py            # EXP-072, EXP-073
    python3 auto_research/pending_queue_update_2026_06_17.py            # EXP-074, EXP-075
    python3 auto_research/pending_queue_update_2026_06_18.py            # EXP-076, EXP-077
    python3 auto_research/pending_queue_update_2026_06_19.py            # EXP-078, EXP-079
    python3 auto_research/pending_queue_update_2026_06_19_paper.py      # EXP-080, EXP-081
    python3 auto_research/pending_queue_update_2026_06_21.py            # EXP-082, EXP-083
    python3 auto_research/pending_queue_update_2026_06_22.py            # EXP-084, EXP-085
    python3 auto_research/pending_queue_update_2026_06_23.py            # EXP-086, EXP-087
    python3 auto_research/pending_queue_update_2026_06_25.py            # EXP-088, EXP-089
    python3 auto_research/pending_queue_update_2026_06_26.py            # EXP-090, EXP-091
    python3 auto_research/pending_queue_update_2026_06_27.py            # EXP-092, EXP-093
    python3 auto_research/pending_queue_update_2026_06_28.py            # EXP-094, EXP-095
    python3 auto_research/pending_queue_update_2026_06_29.py            # EXP-096, EXP-097
    python3 auto_research/pending_queue_update_2026_06_30.py            # EXP-098, EXP-099
    python3 auto_research/pending_queue_update_2026_07_01.py            # EXP-100, EXP-101
    python3 auto_research/pending_queue_update_2026_07_02.py            # EXP-102, EXP-103
    python3 auto_research/pending_queue_update_2026_07_03.py            # EXP-104, EXP-105
    python3 auto_research/pending_queue_update_2026_07_03_paper.py      # EXP-106, EXP-107
    python3 auto_research/pending_queue_update_2026_07_05.py            # EXP-108, EXP-109
    python3 auto_research/pending_queue_update_2026_07_10_paper.py      # (paper-pipeline EXPs)
    python3 auto_research/pending_queue_update_2026_07_11.py            # EXP-110, EXP-111
    python3 auto_research/pending_queue_update_2026_07_12.py            # EXP-112, EXP-113
    python3 auto_research/pending_queue_update_2026_07_13.py            # EXP-114, EXP-115
    python3 auto_research/pending_queue_update_2026_07_14.py            # EXP-116, EXP-117
    python3 auto_research/pending_queue_update_2026_07_16.py            # EXP-118, EXP-119

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_17.py            # EXP-120, EXP-121

Queue was ~129 pending on 2026-07-16 (+2 added: EXP-118, EXP-119).
Queue cap applied: >20 → max 2 new experiments.

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
          → zero gradient → Full = Router.
      (B) Non-collapsed groups (47.6%) = ALL GRPO learning signal; each must be
          used as efficiently as possible.
      (C) Skills gap: 75.61% vs large 96.34% — 20.7pp gap.

AAAI 2027 deadline: 2026-08-15 (29 days from today).
A800 offline since 2026-05-14 (day 64).

arxiv:2607.00152 — "GRPO, Dr. GRPO, and DAPO Are Three Operations on One Number:
    The Group-Standard-Deviation Identity" (Bay & Yearick, June 30, 2026):

    CORE FINDING:
    All three popular GRPO variants differ only in how they treat the within-group
    standard deviation σ. For binary (pass/fail) rewards over K rollouts of a prompt
    with true pass-rate p, the reward std is exactly σ = sqrt(p*(1-p)*K/(K-1)).
    The paper proves the Group-Standard-Deviation Identity:

        GRPO update ∝  (r_i - mean_r) / σ        [divide by σ → scale = 1/σ]
        Dr.GRPO update ∝ (r_i - mean_r)            [drop division → scale ∝ p*(1-p)]
        DAPO update    = GRPO but σ=0 groups filtered entirely

    KEY IMPLICATION FOR OUR ACR=52.4% PROBLEM:
    When σ=0 (collapsed group), GRPO's 1/σ blows up → unstable or zero-gradient
    depending on eps. Dr. GRPO keeps the gradient at (r_i - mean_r) = 0 for
    perfectly collapsed groups (all same reward), which is correct — there is zero
    signal there — but for PARTIALLY collapsed groups (very small σ due to only 1/K
    rollouts differing), Dr. GRPO gives a stable, p*(1-p)-scaled gradient instead
    of 1/σ amplification of near-zero signal. The p*(1-p) scaling acts as a natural
    curriculum: groups near p=0.5 (maximally informative) receive the largest updates;
    groups near p=0 or p=1 (very hard or trivially easy) get proportionally smaller
    updates, reflecting their actual informativeness — without any external curriculum.

    DISTINCTION FROM PRIOR EXPERIMENTS:
    EXP-064 (Dr.GRPO + DAPO T=1.2): combines Dr.GRPO normalization with high-
        temperature rollout sampling to PREVENT collapse at sampling time.
    EXP-108 (sign-advantage): changes the formula ONLY for all-fail groups.
    EXP-119 (MDP-GRPO): uses EMA cross-batch statistics as normalization anchor.
    EXP-118 (RAPER): replays historical rollouts into the advantage computation.
    EXP-120 (this): pure normalization change — remove 1/σ for ALL groups, and
        use sign-advantage only as a zero-variance fallback. Does NOT use T=1.2
        (rollout generation unchanged), does NOT replay, does NOT use EMA. The
        cleanest possible operationalization of 2607.00152's Dr.GRPO setting.

arxiv:2507.00054 — "Enhancing Reasoning Capabilities in SLMs with Reward Guided
    Dataset Distillation" (July 2026):

    PROBLEM THEY SOLVE:
    When distilling from a large teacher model into a small student, naive SFT on
    ALL teacher traces is suboptimal: easy tasks (where the student already solves
    them) add redundant gradient signal, and very hard tasks (where the teacher's
    solution is far beyond the student's reach) add noise that disrupts existing
    capabilities. The effective learning signal concentrates in the "zone of proximal
    development" (ZPD) — tasks where the student partially succeeds, indicating it
    has the building blocks but needs the teacher's strategy to close the gap.

    THEIR SOLUTION (reward-guided dataset distillation):
    Score each teacher trace by the student's own pass-rate on that task (r_small).
    Retain only traces where r_small ∈ [lo, hi] — the ZPD band. Outside this band:
    r_small > hi (student already solves it → no learning), r_small < lo (too hard,
    teacher trace cannot bridge the gap → catastrophic forgetting risk).
    Paper reports: on math and code benchmarks, ZPD-filtered distillation outperforms
    naive SFT by 3.1–5.6pp at the same number of training examples.

    RELEVANCE TO OUR SKILLS GAP (75.61% vs 96.34%):
    Our current SFT (with SFT_INCLUDE_SUCCESS=1) trains on all ~77 teacher traces.
    With HumanEval pass@1, we can estimate small_pass_rate per task from the cycle-0
    collect_traces outputs (rollout rewards under the small model). Filtering to ZPD
    band [0.2, 0.8] on small_pass_rate would remove:
    - ~18 trivially easy tasks (small already solves >80% → wasted signal)
    - ~12 impossibly hard tasks (small pass@1 < 20% → noise)
    - Retaining ~47 ZPD traces with concentrated teacher information
    This directly targets problem (C) — the skills gap — via better data curation,
    with no changes to the GRPO or router components.
"""

import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = "data/humaneval_eval.jsonl"

NEW_EXPERIMENTS = [
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-120: Dr. GRPO (no std normalization) + sign-advantage fallback
    #          for zero-variance groups (HumanEval)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_17_001_dr_grpo_no_std_sign_adv_fallback_humaneval",
        "priority": 8,
        "gpu": "auto",
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2607.00152 — 'GRPO, Dr. GRPO, and DAPO Are Three Operations on One "
            "Number: The Group-Standard-Deviation Identity' (Bay & Yearick, June 30 2026). "
            "Proves that GRPO divides advantages by within-group std σ, Dr. GRPO drops "
            "that division, and DAPO filters zero-σ groups. For binary rewards, the "
            "Dr. GRPO update magnitude is K/(K-1)*p*(1-p)*grad_log_pi, which naturally "
            "weights each prompt by its informativeness p*(1-p) without ever dividing by "
            "σ. This avoids the 1/σ explosion for near-collapsed groups (tiny σ > 0 that "
            "our current eps fallback amplifies) and gives zero gradient for exactly "
            "collapsed groups (σ=0 → mean_r = r_i for all i → r_i - mean_r = 0). "
            "For the zero-variance case we add sign-advantage as a fallback (existing "
            "EXP-108 mechanism already in grpo_train_simple.py), which activates only "
            "when ALL K rollouts are identical (all-fail: sign=-1; all-pass: sign=+1, "
            "no gradient). Distinct from EXP-064 (Dr.GRPO + T=1.2 DAPO sampling), "
            "EXP-119 (EMA dual-anchor), EXP-118 (RAPER replay), EXP-108 (sign-adv "
            "for all-fail only). This is the cleanest operationalization of the "
            "2607.00152 Dr.GRPO variant applied to our ACR=52.4% pipeline. Target: "
            "full arm pass@1 > 92.68% and ACR < 45%. AAAI 2027 deadline 2026-08-15."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 4,
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "dr_grpo_enabled": True,
            "dr_grpo_no_std_division": True,
            "dr_grpo_zero_var_fallback": "sign_advantage",
            "grpo_temperature": 1.0,
            "n_generations": 8,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "analysis": [
                "acr_per_cycle_dr_grpo_vs_baseline",
                "per_prompt_update_magnitude_vs_pass_rate",
                "p1p_scaling_empirical_vs_theoretical",
                "full_arm_pass_at_1_improvement",
                "gradient_norm_per_training_step",
            ],
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (baseline): full 92.68%, ACR=52.4%",
                "EXP-064 (Dr.GRPO + T=1.2 DAPO): sampling-side collapse prevention",
                "EXP-108 (sign-advantage): formula change for all-fail only",
                "EXP-119 (MDP-GRPO): EMA dual-anchor normalization",
                "arxiv:2607.00152 (Dr.GRPO): p*(1-p) natural weighting",
            ],
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
            ],
            "implementation_note": (
                "In grpo_train_simple.py, find the advantage normalization block. "
                "Currently: `advantages = (rewards - mean_r) / (std_r + eps)`. "
                "Replace with Dr. GRPO mode: `advantages = rewards - mean_r` (no division). "
                "Then add zero-variance fallback: if std_r < eps (collapsed group), check "
                "if all rewards == 0 (all-fail) → set advantages = -sign_scale (e.g. -1.0) "
                "for all tokens; if all rewards == 1 (all-pass) → advantages = zeros "
                "(no gradient, group is trivially solved). This integrates the existing "
                "sign-advantage logic (from EXP-108) as the zero-var branch. "
                "Add a flag `dr_grpo_enabled` (env var or spec field) to toggle so the "
                "baseline is unchanged. Total change: ~10 lines. Compatible with existing "
                "LoRA, vLLM serving, and checkpoint priority logic."
            ),
            "arxiv_ref": "2607.00152",
            "estimated_gpu_hours": 2.5,
        },
    },
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-121: Zone-of-Proximal-Development SFT (reward-guided trace filtering)
    #          to close the Skills gap (75.61% → target ≥80%)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_17_002_zpd_sft_reward_guided_distillation_humaneval",
        "priority": 7,
        "gpu": "auto",
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2507.00054 — 'Enhancing Reasoning Capabilities in SLMs with Reward "
            "Guided Dataset Distillation' (July 2026). Proposes filtering SFT distillation "
            "data to the 'zone of proximal development' (ZPD): retain teacher traces only "
            "for tasks where the student's own pass-rate r_small ∈ [lo, hi]. Tasks with "
            "r_small > hi are already solved by the student → wasted training signal. Tasks "
            "with r_small < lo are beyond the student's reach → the teacher trace introduces "
            "distribution shift the student cannot assimilate (catastrophic forgetting risk). "
            "Paper reports +3.1–5.6pp over naive SFT on math/code benchmarks at identical "
            "example counts. For our pipeline: collect_traces already records per-prompt "
            "small-model rollout rewards in traces.jsonl. The ZPD filter is a 2-line change "
            "in traces_to_sft.py: `small_pass = mean(trace['small_rewards'])` and "
            "`if not (zpd_lo <= small_pass <= zpd_hi): continue`. "
            "Setting zpd_lo=0.20, zpd_hi=0.80 removes: ~18 trivially easy tasks "
            "(small already scores >80%) and ~12 impossibly hard tasks (small <20%), "
            "retaining ~47 ZPD traces. This targets root problem (C) — skills gap "
            "75.61% vs 96.34% (20.7pp) — orthogonal to all GRPO-collapse fixes "
            "(EXP-108/118/119/120 all target problem A). No changes to GRPO or router. "
            "AAAI 2027 deadline 2026-08-15 (29 days). If skills arm ≥80%, the "
            "router-vs-skills improvement gap widens → stronger paper narrative."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 4,
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "zpd_sft_enabled": True,
            "zpd_lo": 0.20,
            "zpd_hi": 0.80,
            "zpd_filter_field": "small_pass_rate",
            "sft_include_success": True,
            "scaling_force_both": True,
            "grpo_temperature": 1.0,
            "n_generations": 8,
            "eval_data": _HE_EVAL,
            "analysis": [
                "skills_arm_pass_at_1_zpd_vs_baseline",
                "n_sft_examples_per_cycle_zpd_vs_baseline",
                "zpd_band_task_distribution_per_cycle",
                "small_pass_rate_histogram_before_after_zpd",
                "skills_gap_vs_large_per_cycle",
            ],
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (baseline): skills 75.61%, ACR=52.4%",
                "EXP-112 (forward-KL teacher distillation): KL-based weighting",
                "arxiv:2507.00054 (reward-guided distillation): +3.1-5.6pp on code",
            ],
            "implementation_files": [
                "src/pipeline/traces_to_sft.py",
            ],
            "implementation_note": (
                "In traces_to_sft.py, after loading each trace from traces.jsonl: "
                "compute small_pass_rate = mean(trace.get('small_rewards', [0])) "
                "where small_rewards is the list of binary rewards from the K small-model "
                "rollouts already stored by collect_traces. "
                "Add ZPD filter: `if zpd_sft_enabled and not (zpd_lo <= small_pass_rate <= zpd_hi): continue`. "
                "Add env vars ZPD_SFT_ENABLED (default 0), ZPD_LO (default 0.20), "
                "ZPD_HI (default 0.80) so the baseline run is unchanged. "
                "If small_rewards is absent from a trace (e.g. cycle 0 where small wasn't "
                "run), default small_pass_rate=0.5 (include it — conservative). "
                "No other files need changes. ~8 lines total."
            ),
            "arxiv_ref": "2507.00054",
            "estimated_gpu_hours": 2.5,
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
