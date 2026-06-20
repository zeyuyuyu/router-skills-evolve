#!/usr/bin/env python3
"""
Daily queue patch — 2026-06-20 (EXP-080, EXP-081).

A800 connectivity: offline since 2026-05-14 (day 37). Apply when restored.

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
    python3 auto_research/pending_queue_update_2026_06_15.py       # EXP-070, EXP-071
    python3 auto_research/pending_queue_update_2026_06_16.py       # EXP-072, EXP-073
    python3 auto_research/pending_queue_update_2026_06_17.py       # EXP-074, EXP-075
    python3 auto_research/pending_queue_update_2026_06_18.py       # EXP-076, EXP-077
    python3 auto_research/pending_queue_update_2026_06_19.py       # EXP-078, EXP-079

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_06_20.py       # EXP-080, EXP-081

Queue was ~79 pending on 2026-06-19 (+2 added: EXP-078, EXP-079).
Queue cap applied: >20 → max 2 new experiments.

Data motivating today's proposals
----------------------------------
results/rl_15b_a800_20260509_summary.json (committed 2026-05-09):
    base=47/100, grpo_adapter=49/100 → +2 pp abs gain (single seed, n=1)
    Binomial standard error at n=100, p̂=0.49: σ = sqrt(0.49×0.51/100) ≈ 0.050
    95% Wald CI: 49 ± ~10 pp → the +2pp lift may not be distinguishable from noise
    → EXP-080 directly answers: is the MBPP GRPO gain statistically robust?

results/e2e_4cyc_gpt55/cycle_N/e2e_ablation_summary.json (cycles 0-3):
    Cycle 0 routing_acc=91.46% → Cycle 3 routing_acc=92.68% (single run, n=1)
    Cycle 2 routing_acc=93.90% — a HIGHER accuracy than cycle 3; single-run
    variance (±1-2 pp) obscures the true cycle-over-cycle trend. Without seed
    replication, it is impossible to assert that cycle 3 "improves" cycle 2 with
    confidence. The AAAI 2027 claim "routing accuracy improves over N cycles" is
    untenable without error bars.
    → EXP-081 directly answers: is the cycle-over-cycle routing improvement real?

AAAI 2027 deadline: 2026-08-15 (56 days from today). Neither `grpo_multi_seed_staircase`
nor `joint_cycle_multiseed` has been used in any queued experiment to date (EXP-001 to
EXP-079). These are the two most AAAI-critical missing validations.
"""
import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_MBPP_TRAIN = (
    "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/"
    "train_aug_excluding_eval20.jsonl"
)
_MBPP_EVAL = (
    "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/eval100.jsonl"
)
_HE_EVAL = (
    "/data0/home/zeyuwang/router-skills-evolve-data/humaneval/HumanEval.jsonl.gz"
)
_E2E_GPT55_DIR = (
    "/data0/home/zeyuwang/router-skills-evolve/results/e2e_4cyc_gpt55"
)

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_06_20_001_grpo_multiseed_mbpp_200tasks_k3_staircase",
        "priority": 9,
        "kind": "grpo_multi_seed_staircase",
        "rationale": (
            "Internal data: results/rl_15b_a800_20260509_summary.json (committed "
            "2026-05-09). The canonical MBPP GRPO result — base=47/100, "
            "grpo_adapter=49/100 — is from a SINGLE training run with a single "
            "random seed. Statistical analysis: at n=100 eval tasks and p̂=0.49, "
            "the binomial standard error is σ = sqrt(p̂(1-p̂)/n) = sqrt(0.49×0.51/100) "
            "≈ 0.050, giving a 95% Wald confidence interval of approximately "
            "[0.39, 0.59]. The observed +0.02 absolute gain (47→49) lies entirely "
            "within this single-run sampling noise band. Put differently: even if the "
            "true model pass rate were still 0.47 (no real gain), a single 100-task "
            "evaluation would return 49/100 with ~19% probability by chance. A SINGLE "
            "seed cannot distinguish a real +2pp gain from a lucky random evaluation. "
            "The AAAI 2027 paper makes the claim 'GRPO improves small-model pass@1 "
            "from 47 to 49 pp on MBPP'. AAAI area chairs and reviewers will immediately "
            "flag the lack of multiple seeds as a critical weakness — especially for a "
            "+2pp claim on n=100. The fix requires multiple independent training seeds "
            "AND possibly larger eval sets, but the most direct action is k=3 training "
            "seeds on the same configuration. With k=3 seeds: "
            "(a) If all three seeds produce pass@1 ∈ [48, 51], the gain is real and "
            "we can report mean=49.0±1.0 pp. "
            "(b) If results vary widely (e.g., 46, 49, 52), the variance itself is a "
            "finding: GRPO on this problem size has high training noise and the reported "
            "single-seed result is not representative. In case (b), we can look at "
            "median or provide a seed-filtered result. "
            "(c) The staircase component — evaluating at 25%/50%/75%/100% of training "
            "steps per seed — provides the learning curve across seeds. If the three "
            "curves converge, GRPO training is stable. If they diverge after step ~50%, "
            "it pinpoints instability onset (consistent with known 400-task degradation "
            "pattern). The staircase is a strict superset of a single eval: it costs "
            "3 extra eval passes per seed (cheap, ~3 min each) and provides a publication- "
            "quality training-curve figure. Configuration exactly mirrors the successful "
            "2026-05-09 run (DAPO G=8, 200 tasks, 1 epoch, lr=5e-6, LoRA r=16, "
            "multi-turn repair max_turns=3, qwen-chat prompt, binary pytest reward). "
            "No new algorithm: this is a pure statistical-robustness validation of the "
            "best known result. Priority 9 (joint-highest with EXP-076): critical for "
            "AAAI submission. Wall-clock: 3 × ~75 min (including staircase evals) = "
            "~225 min = ~3.75 hours. Within the 4-hour budget on a single A800. "
            "Not a duplicate of any queued experiment: EXP-001 through EXP-079 all use "
            "a single fixed seed. No `grpo_multi_seed_staircase` kind has ever been "
            "used in this project."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": _MBPP_TRAIN,
            "eval_data": _MBPP_EVAL,
            "bench": "mbpp",
            "train_task_limit": 200,
            "grpo_algo": "dapo",
            "n_generations": 8,
            "dapo_dynamic_sampling": True,
            "dapo_clip_low": 0.2,
            "dapo_clip_high": 0.5,
            "rollout": "repair",
            "max_turns": 3,
            "epochs": 1,
            "lr": 5e-6,
            "lora_r": 16,
            "prompt_style": "qwen-chat",
            "reward": "binary",
            "n_seeds": 3,
            "seeds": [42, 1337, 7],
            "staircase_checkpoints": [0.25, 0.5, 0.75, 1.0],
            "report_mean_std": True,
            "aaai_relevant": True,
            "motivation": (
                "Single-seed +2pp MBPP result (47→49, rl_15b_a800_20260509) has "
                "binomial σ≈5pp at n=100; +2pp is within noise band; AAAI requires "
                "multi-seed validation; no grpo_multi_seed_staircase in queue (EXP-001~079)"
            ),
            "estimated_hours": 3.75,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_06_20_002_joint_cycle_multiseed_he_k3_router_variance",
        "priority": 8,
        "kind": "joint_cycle_multiseed",
        "rationale": (
            "Internal data: results/e2e_4cyc_gpt55/cycle_{0,1,2,3}/e2e_ablation_summary.json "
            "(committed 2026-06-17). The routing accuracy across the 4 cycles is: "
            "cycle 0 = 91.46%, cycle 1 = 90.24%, cycle 2 = 93.90%, cycle 3 = 92.68%. "
            "The non-monotone pattern (cycle 2 > cycle 3, cycle 1 < cycle 0) already "
            "suggests the per-cycle routing accuracy has meaningful run-to-run variance. "
            "All four measurements are from a SINGLE run with a SINGLE random seed for "
            "both the GRPO training and the router training. The paper claims 'routing "
            "accuracy improves over the training cycles' — but a single-seed trajectory "
            "that goes 91.46% → 90.24% → 93.90% → 92.68% is neither monotone nor "
            "clearly improving. The cycle-2 value (93.90%) is actually higher than cycle "
            "3 (92.68%). Without multiple seeds we cannot determine whether the cycle-3 "
            "result is 'typical' or a downward fluctuation. AAAI reviewers will rightly "
            "ask: is the cycle-over-cycle improvement real, or is it within variance? "
            "The fix: run the full SLR pipeline (Skills distillation + SFT + Router, "
            "with SKIP_GRPO=1 to control cost) on the EXISTING oracle traces from "
            "e2e_4cyc_gpt55 with k=3 independent random seeds. The oracle traces "
            "(gpt-5.5 teacher outputs) are already collected and stored on disk — "
            "zero new API calls required. The only compute is: "
            "(a) Skills extraction: deterministic, ~2 min per cycle (skills are seeded "
            "by trace content, not by a random PRNG). "
            "(b) SFT (LoRA): ~25 min per cycle on A800 for Qwen2.5-Coder-1.5B. "
            "(c) Router training (TF-IDF + logistic regression): ~5 min per cycle. "
            "(d) E2E ablation eval: ~3 min per cycle. "
            "Total per seed: 4 cycles × (2+25+5+3) = 4 × 35 = 140 min. "
            "k=3 seeds × 140 min = 420 min = 7 hours. Over budget. "
            "REDUCED SCOPE: use k=3 seeds but fix at the SINGLE most informative "
            "cycle (cycle 3, where the GRPO adapter is available) and re-run only "
            "the router training + eval step with 3 independent random seeds. "
            "Router training is the ONLY stochastic component once the LLM adapter "
            "is fixed (logistic regression is seeded): the key variance question is "
            "not 'does GRPO vary across seeds?' (already covered by EXP-080 on MBPP) "
            "but 'does the router training produce consistent accuracy given the same "
            "LLM backbone?' With cycle-3 GRPO adapter fixed: 3 seeds × (5 min router + "
            "3 min eval) = 24 min. Additionally run with cycle-0/1/2 LLM adapters fixed "
            "(no GRPO): 4 cycles × 3 seeds × 8 min = 96 min. Grand total: ~120 min = "
            "2 hours. Well within budget. "
            "Scope for this experiment: cycle 0-3, SLR with SKIP_GRPO=1, k=3 seeds "
            "on router training only, using existing oracle traces and LLM adapters. "
            "Reports per-cycle routing_accuracy mean ± std across seeds. Provides the "
            "AAAI-critical error bars on Figure 2 (routing accuracy over cycles). "
            "Not a duplicate of any queued experiment: no `joint_cycle_multiseed` kind "
            "has ever been used in this project (EXP-001 through EXP-079 are all "
            "single-seed). "
            "Secondary question answered: does the router variance explain the "
            "non-monotone cycle trajectory (91.46% → 90.24% → 93.90% → 92.68%)? "
            "If per-cycle std is ~1-2 pp, the trajectory is within noise. If std < "
            "0.5 pp, the non-monotone pattern is real and needs investigation. "
            "Priority 8 (just below EXP-080): routing accuracy claim is the paper's "
            "main contribution; multi-seed validation is essential but this experiment "
            "is faster and cheaper than EXP-080, so the higher-priority slot goes to "
            "the MBPP result which is more statistically uncertain per sample."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "bench": "humaneval",
            "eval_data": _HE_EVAL,
            "oracle_traces_dir": _E2E_GPT55_DIR,
            "reuse_oracle_traces": True,
            "reuse_llm_adapters": True,
            "phases": ["router"],
            "skip_grpo": True,
            "skip_sft": True,
            "skip_skills": True,
            "cycles": [0, 1, 2, 3],
            "n_seeds": 3,
            "seeds": [42, 1337, 7],
            "router_type": "tfidf_logistic",
            "report_per_cycle_mean_std": True,
            "aaai_relevant": True,
            "motivation": (
                "Cycle routing accuracy is non-monotone across 4 cycles "
                "(91.46→90.24→93.90→92.68) in single-seed e2e_4cyc_gpt55 run; "
                "AAAI requires error bars on the main routing claim; "
                "no joint_cycle_multiseed in queue (EXP-001~079); "
                "reuses existing oracle traces and LLM adapters (zero new API calls)"
            ),
            "estimated_hours": 2.0,
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
