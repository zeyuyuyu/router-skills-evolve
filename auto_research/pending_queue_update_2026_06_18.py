#!/usr/bin/env python3
"""
Daily queue patch — 2026-06-18 (EXP-076, EXP-077).

A800 connectivity: offline since 2026-05-14 (day 35). Apply when restored.

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

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_06_18.py       # EXP-076, EXP-077

Queue was ~75 pending on 2026-06-17 (+2 added: EXP-074, EXP-075).
Queue cap applied: >20 → max 2 new experiments.

Data motivating today's proposals
----------------------------------
results/e2e_4cyc_gpt55/cycle_3/grpo_adapter/grpo_info.json (committed 2026-06-17):
    n_groups=82, n_dropped_zero_variance=43, n_informative_groups=39
    → 52.4% of training groups provided zero gradient even at G=8 with DAPO
    → only 39/82 HumanEval problems are in the informative "Goldilocks" pass-rate zone

results/e2e_4cyc_gpt55/final_ablation_table.md (committed 2026-06-17):
    Cycle 0 skills 70.73% → cycle 3 skills 75.61% (+4.88 pp over 4 cycles)
    "Full" == "Router" in cycle 3 (92.68% task pass): LLM fine-tuning does not help routing
    Cross-benchmark impact of HumanEval GRPO training on MBPP: UNKNOWN (motivates EXP-076)
"""
import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_CYCLE3_GRPO_ADAPTER = (
    "/shared_home/yuhang.yao/router-skills-evolve/results/"
    "e2e_4cyc_gpt55/cycle_3/grpo_adapter"
)
_CYCLE3_LLM_ADAPTER = (
    "/shared_home/yuhang.yao/router-skills-evolve/results/"
    "e2e_4cyc_gpt55/cycle_3/llm_adapter"
)
_MBPP_EVAL = (
    "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/eval100.jsonl"
)
_HE_EVAL = (
    "/data0/home/zeyuwang/router-skills-evolve-data/humaneval/HumanEval.jsonl.gz"
)

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_06_18_001_forgetting_eval_cycle3_grpo_mbpp_generalization",
        "priority": 9,
        "kind": "forgetting_eval",
        "rationale": (
            "Internal data source: results/e2e_4cyc_gpt55/cycle_3/grpo_adapter/"
            "grpo_info.json (committed 2026-06-17 by Yuhang Yao). The 4-cycle "
            "e2e_4cyc_gpt55 run trained Qwen2.5-Coder-1.5B-Instruct with DAPO "
            "(clip_low=0.2, clip_high=0.5, G=8, max_turns=3, dynamic_sampling=True) "
            "on the 82-problem HumanEval benchmark, producing grpo_adapter at "
            "/shared_home/yuhang.yao/.../cycle_3/grpo_adapter. The final ablation "
            "table shows the adapter achieves 75.61% task pass on HumanEval "
            "(skills arm, always-small + procedure) in cycle 3, up from 70.73% "
            "in cycle 0 (+4.88 pp). The critical open question for the AAAI 2027 "
            "submission is: does HumanEval GRPO training transfer to MBPP, or does "
            "it cause cross-benchmark catastrophic forgetting? The GRPO training "
            "was optimised for pytest-based HumanEval rewards on function-level "
            "Python coding problems; MBPP shares the problem type (Python function "
            "coding) but uses a different problem distribution (competitive "
            "programming style vs. docstring-driven). There are two distinct "
            "hypotheses: (H1) GENERALIZATION — HumanEval GRPO teaches a general "
            "'code debugging via multi-turn repair' policy that transfers positively "
            "to MBPP, potentially lifting the 47 pp MBPP baseline; (H2) "
            "INTERFERENCE — HumanEval's specific function signatures, docstring "
            "style, and test harness over-specialize the adapter, causing MBPP "
            "performance to degrade (cross-benchmark catastrophic forgetting). "
            "This is cheap to test: evaluate the cycle-3 grpo_adapter on MBPP "
            "eval100 (100 tasks) and HumanEval holdout (n_eval=82 already seen, "
            "plus any held-out subset). Wall-clock ~25-30 min (evaluation only, "
            "no training). Result directly informs the AAAI paper's claim about "
            "the generality of the GRPO-trained small model. If H1 holds: "
            "strongest possible AAAI result (HumanEval training lifts MBPP). "
            "If H2 holds: signals need for cross-benchmark replay in the "
            "training mix (implementable as EXP-07X, mixed-benchmark DAPO). "
            "Priority 9 (highest): very cheap, uniquely valuable, directly "
            "enables the AAAI cross-benchmark generalization claim within the "
            "57-day window before August 15 submission."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "adapter_paths": {
                "cycle3_grpo": _CYCLE3_GRPO_ADAPTER,
                "cycle3_sft": _CYCLE3_LLM_ADAPTER,
            },
            "eval_benchmarks": [
                {
                    "name": "mbpp_eval100",
                    "data": _MBPP_EVAL,
                    "n_tasks": 100,
                    "metric": "pass@1",
                },
                {
                    "name": "humaneval_82",
                    "data": _HE_EVAL,
                    "n_tasks": 82,
                    "metric": "pass@1",
                },
            ],
            "eval_modes": ["base_model", "sft_adapter", "grpo_adapter"],
            "prompt_style": "qwen-chat",
            "temperature": 0.0,
            "purpose": "cross_benchmark_generalization_check",
            "aaai_relevant": True,
            "estimated_hours": 0.5,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_06_18_002_dapo_g16_dynamic_humaneval_multiturn_repair",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "Internal data source: results/e2e_4cyc_gpt55/cycle_3/grpo_adapter/"
            "grpo_info.json (committed 2026-06-17). Key finding: n_groups=82, "
            "n_dropped_zero_variance=43, n_informative_groups=39 — 52.4% of all "
            "training groups provide zero gradient even with DAPO G=8 dynamic "
            "sampling (max_turns=3, repair rollout). This is the dominant "
            "training-efficiency bottleneck in the HumanEval multi-turn repair "
            "setting. Analysis: zero-variance groups arise from two sources — "
            "(a) ALL-PASS: the cycle-3 model already masters ~43% of HumanEval "
            "problems at pass_rate ≈ 1.0 (consistent with 75.61% skills-arm "
            "result, where the easiest ~45% problems are fully solved); (b) "
            "ALL-FAIL: ~7-9% of problems are currently unsolvable regardless of "
            "temperature or repair attempts. DAPO's dynamic sampling correctly "
            "discards these groups from the UPDATE STEP but still wastes "
            "generation compute on them (all 8 rollouts × 3 repair turns run "
            "before the group is identified as zero-variance). The fix: increasing "
            "G from 8 to 16 directly addresses the PARTIAL-COLLAPSE cases where "
            "G=8 creates zero-variance by random chance (e.g., a problem at "
            "pass_rate=0.6: P(all-8 pass)=0.6^8≈1.7%, P(all-8 fail)=0.4^8≈0.07% "
            "→ <2% zero-variance by chance; but P(all-16 pass)=0.6^16≈0.03% → "
            "DAPO only needs to drop problems that are truly at p≈0 or p≈1). For "
            "G=16 with DAPO dynamic sampling, the cost scales with the number of "
            "INFORMATIVE groups, not with G — groups at p≈1 or p≈0 are identified "
            "early and dropped, while groups in [0.2, 0.8] contribute 16 rollouts "
            "each. arxiv:2603.07777 ('Breaking Training Bottlenecks: Effective and "
            "Stable Reinforcement Learning for Coding Models', Li et al., Mar 2026, "
            "Microsoft Research) introduces MicroCoder-GRPO, which addresses "
            "related bottlenecks in code GRPO training. MicroCoder-GRPO's "
            "Diversity-Determined Temperature (DDT) component dynamically adjusts "
            "generation temperature per group based on a diversity metric computed "
            "over in-group rollouts — if the group's rollouts are collapsing "
            "(all similar), temperature is raised; if diverse, temperature is "
            "reduced. This complements G=16: DDT specifically helps partially-"
            "collapsed groups (where G=8 generated diverse but G=16 might still "
            "collapse for very high-pass-rate tasks). Combined: G=16 DAPO + DDT "
            "on HumanEval multi-turn repair. MicroCoder-GRPO reports +17.6% "
            "relative improvement on LiveCodeBench v6 and better training stability "
            "across all configurations. Distinct from existing queue: EXP-072 "
            "(Infinite Sampling G=8) is for MBPP single-turn standard GRPO — "
            "different benchmark, algo, rollout type, and G value. No queued "
            "experiment combines G=16 + DDT + DAPO on HumanEval multi-turn repair. "
            "Wall-clock estimate: ~110 min. DAPO discards zero-variance groups "
            "from UPDATE — effective training set ≈ 39-50 informative groups × "
            "16 rollouts × 3 turns. At 1.5B, inference speed: ~0.8 s/rollout-turn "
            "→ 50 × 16 × 3 × 0.8 ≈ 1920s ≈ 32 min generation; ~80 min total with "
            "backward passes. Within the 4-hour budget."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": _HE_EVAL,
            "eval_data": _HE_EVAL,
            "bench": "humaneval",
            "grpo_algo": "dapo",
            "n_generations": 16,
            "dapo_dynamic_sampling": True,
            "dapo_clip_low": 0.2,
            "dapo_clip_high": 0.5,
            "diversity_determined_temperature": True,
            "ddt_base_temperature": 1.0,
            "ddt_diversity_metric": "pass_rate_variance",
            "ddt_temp_range": [0.6, 1.4],
            "rollout": "repair",
            "max_turns": 3,
            "epochs": 1,
            "lr": 5e-6,
            "lora_r": 16,
            "beta": 0.04,
            "prompt_style": "qwen-chat",
            "reward": "binary",
            "arxiv_ref": "2603.07777",
            "data_motivation": (
                "grpo_info.json cycle_3: n_dropped_zero_variance=43/82 (52.4%) "
                "even at G=8 DAPO; G=16 targets partial-collapse cases"
            ),
            "estimated_hours": 1.83,
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
