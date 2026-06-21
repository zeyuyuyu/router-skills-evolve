#!/usr/bin/env python3
"""
Daily queue patch — 2026-06-21 (EXP-082, EXP-083).

A800 connectivity: offline since 2026-05-14 (day 38). Apply when restored.

Prior patches to apply before this one (if using the daily-patch chain):
    python3 auto_research/pending_queue_update.py                       # base experiments
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

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_06_21.py            # EXP-082, EXP-083

Queue was ~81 pending on 2026-06-19 (+2 added: EXP-080, EXP-081).
Queue cap applied: >20 → max 2 new experiments.

Data motivating today's proposals
----------------------------------
results/e2e_4cyc_gpt55/cycle_3/grpo_adapter/grpo_info.json (committed 2026-06-17):
    n_groups=82, n_dropped_zero_variance=43, n_informative_groups=39
    → 52.4% of training groups had zero gradient (DAPO drops them entirely)
    → EXP-077 (DDT G=16) reduces chance-collapse; EXP-078 (RECRL) avoids sampling
      zero-variance problems pre-generation. Neither extracts signal from them post-rollout.
    → EXP-082 (RL-ZVP) closes this gap: assign entropy-scaled advantage for all-pass /
      all-fail groups AFTER rollout, instead of discarding them entirely.

results/cwy_35b_fullsplit_20260610_083624/cycle_3/e2e_ablation_summary.json:
    tau2 held-out: base/skills=76.97%, always-large=60.11%, router=69.66%, full=69.66%
    → Router hurts vs. always-small on tau2 held-out (69.66% vs. 76.97%)
    → Large model underperforms small on tau2 (60.11% < 76.97%); any routing to large
      is costly; no random-routing or oracle baseline to contextualize this finding
    → EXP-083 adds the random-50% baseline (W5 in paper review-2026-06-19.md)

arxiv:2509.21880 (RL-ZVP, ICLR 2026):
    'No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning
    via Entropy-Guided Advantage Shaping'. Proposes entropy-guided advantage for BOTH
    all-pass and all-fail groups: positive (all-pass) or negative (all-fail) advantage
    scaled by per-token entropy. +8.61 pp on math reasoning benchmarks vs GRPO baseline.

arxiv:2506.06579 (multi-LLM routing survey, Jun 2025):
    'Towards Efficient Multi-LLM Inference: Characterization and Analysis of LLM Routing
    and Hierarchical Techniques'. Documents that random-proportion routing (Bernoulli-p
    policy) is a standard minimum baseline for learned routing evaluation; absence of this
    baseline inflates perceived gain of learned routers.
"""
import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = (
    "/data0/home/zeyuwang/router-skills-evolve-data/humaneval/HumanEval.jsonl.gz"
)
_TAU2_HELD_OUT = (
    "/data0/home/zeyuwang/router-skills-evolve-data/tau2/held_out_tasks.jsonl"
)
_TAU2_FULLSPLIT_CYCLE3 = (
    "/data0/home/zeyuwang/router-skills-evolve/results/"
    "cwy_35b_fullsplit_20260610_083624/cycle_3"
)

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_06_21_001_rlzvp_entropy_advantage_humaneval_dapo",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2509.21880 — 'No Prompt Left Behind: Exploiting Zero-Variance Prompts "
            "in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping' "
            "(RL-ZVP, ICLR 2026, Sep 2025 preprint). "
            "Standard GRPO and DAPO discard zero-variance groups — prompts where all G "
            "rollouts share the same reward (all pass or all fail). DAPO's dynamic "
            "sampling already filters these AFTER generation, but grpo_info.json "
            "(cycle_3, commit 2026-06-17) shows 43/82 groups (52.4%) are still "
            "discarded at G=8. RL-ZVP proposes to EXPLOIT these groups instead: "
            "(a) All-pass groups: assign a POSITIVE advantage scaled by per-token "
            "entropy H(token). High-entropy tokens within a correct response were "
            "uncertain choices — reinforcing them tightens the policy toward the "
            "high-reward region. Low-entropy tokens (already peaked) get near-zero "
            "advantage, avoiding over-reinforcement. "
            "(b) All-fail groups: assign a NEGATIVE advantage scaled by per-token "
            "entropy H(token). High-entropy tokens within a failing response represent "
            "uncertain wrong choices — penalising them explicitly steers the policy "
            "away from the failure mode. "
            "This operates POST-rollout: it does not change which problems are sampled "
            "(RECRL/EXP-078 handles that) or the generation temperature (DDT/EXP-077 "
            "handles that). It is therefore orthogonal to both EXP-077 and EXP-078 — "
            "all three can be combined, but this experiment tests RL-ZVP alone vs "
            "the baseline DAPO G=8 (reference: e2e_4cyc_gpt55 cycle_3 run). "
            "Direct data motivation: 43 zero-variance groups × 8 rollouts = 344 "
            "completions generated but discarded per training step in the reference "
            "run. RL-ZVP recaptures gradient signal from all 344 completions, "
            "increasing effective batch utilisation by up to 2× (344/(312 kept) = 1.10× "
            "from discarded alone, or 344/(628 total) = 54.8% recaptured rollouts). "
            "RL-ZVP paper reports +8.61 pp on math (MATH500) and +7.77 pp pass rate "
            "across 6 benchmarks with Qwen2.5-7B-Instruct and Llama-3.1-8B. No prior "
            "queued experiment uses entropy-guided advantage shaping on zero-variance "
            "groups — EXP-077 (DDT) changes generation temperature, EXP-078 (RECRL) "
            "filters problems pre-generation, EXP-079 (OPR) adds replay, EXP-074 "
            "(EGCA) reallocates credit across turns, none modifies advantage for "
            "zero-variance groups post-rollout. Implementation: ~25 lines in "
            "grpo_train_simple.py replacing the current 'drop zero-variance group' "
            "branch with the RL-ZVP entropy-scaled advantage assignment. Token entropy "
            "is already computable from the completion logits (already logged during "
            "the DAPO forward pass). Estimated wall-clock: ~90 min for Qwen2.5-Coder- "
            "1.5B-Instruct, G=8, 1 epoch on HumanEval 82 problems (now processing all "
            "82 groups vs 39 in reference run — more update steps, but rollout cost "
            "is unchanged)."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": _HE_EVAL,
            "eval_data": _HE_EVAL,
            "bench": "humaneval",
            "grpo_algo": "dapo",
            "n_generations": 8,
            "dapo_dynamic_sampling": True,
            "dapo_clip_low": 0.2,
            "dapo_clip_high": 0.5,
            "rlzvp_enabled": True,
            "rlzvp_alpha": 0.5,
            "rlzvp_use_token_entropy": True,
            "rlzvp_zero_var_advantage_mode": "entropy_guided",
            "rlzvp_all_pass_sign": 1,
            "rlzvp_all_fail_sign": -1,
            "rollout": "repair",
            "max_turns": 3,
            "epochs": 1,
            "lr": 5e-6,
            "lora_r": 16,
            "beta": 0.04,
            "prompt_style": "qwen-chat",
            "reward": "binary",
            "arxiv_ref": "2509.21880",
            "data_motivation": (
                "grpo_info.json cycle_3: n_dropped_zero_variance=43/82 (52.4%); "
                "RL-ZVP recaptures gradient from all 43 discarded groups via "
                "entropy-scaled advantage"
            ),
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (DAPO G=8 reference)",
                "exp_2026_06_18_002_dapo_g16_dynamic_humaneval_multiturn_repair",
                "exp_2026_06_19_001_recrl_adaptive_curriculum_grpo_humaneval",
            ],
            "estimated_hours": 1.5,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_06_21_002_tau2_random50_routing_baseline_eval",
        "priority": 7,
        "kind": "forgetting_eval",
        "rationale": (
            "arxiv:2506.06579 — 'Towards Efficient Multi-LLM Inference: Characterization "
            "and Analysis of LLM Routing and Hierarchical Techniques' (Jun 2025). "
            "This routing survey documents that random-proportion routing "
            "(Bernoulli-p policy: route to large with probability p, else small) is "
            "a standard minimum baseline for learned routing evaluation; without it, "
            "the perceived gain of a learned router cannot be attributed to the router "
            "logic vs. the trivial effect of 'how often large is used'. "
            "Direct motivation from local data: "
            "results/cwy_35b_fullsplit_20260610_083624/cycle_3/e2e_ablation_summary.json "
            "shows a striking result: always-large = 60.11% task pass on tau2 held-out, "
            "while always-small = 76.97%. The large model (GPT-5.4/5.5) is materially "
            "WORSE than the specialized small model (Qwen3.6-35B-A3B) on held-out tau2 "
            "tasks. The learned router (cycle 3) routes only 1.1% of tasks to large "
            "(cost_vs_large=0.469 means 46.9% of cost → ~47% of tasks to large, not "
            "1.1% — re-reading: fallback=0.011 means 1.1% fallback to API, i.e. large "
            "model called for 1.1% of tasks). Router achieves 69.66% task pass — "
            "WORSE than always-small (76.97%) but BETTER than always-large (60.11%). "
            "A random-50% routing baseline on the SAME held-out tasks would get "
            "approximately: 0.5 × 76.97% + 0.5 × 60.11% ≈ 68.5%. The learned router "
            "(69.66%) outperforms random-50% (~68.5%) by ~1.2pp but significantly "
            "underperforms always-small (76.97%). This baseline is essential for "
            "AAAI paper W5: it contextualizes whether the router's tau2 performance "
            "is due to 'learning to route' or simply 'using large rarely', and it "
            "directly answers whether the held-out drop (80%→66% routing accuracy "
            "as noted in review-2026-06-19.md W2) is worse than chance-level routing. "
            "Additionally, a random-10% baseline (route to large with p=0.1, matching "
            "the router's ~1.1% fallback rate) provides a fair comparison: if random-10% "
            "also achieves ~69.5% (since calling large rarely is nearly always-small), "
            "then the router's improvement is trivially explained by its low large-call "
            "rate, not by learning which tasks need large. "
            "This is the cheapest possible AAAI-relevant experiment: pure evaluation "
            "on existing traces + adapter, no training, ~20 min wall-clock on A800 "
            "or H200. It directly addresses paper W5 and provides the critical "
            "denominator for interpreting all tau2 routing results."
        ),
        "spec": {
            "mode": "routing_eval_only",
            "no_training": True,
            "bench": "tau2_bench",
            "eval_split": "held_out",
            "eval_data": _TAU2_HELD_OUT,
            "adapter_dir": _TAU2_FULLSPLIT_CYCLE3,
            "small_model": "Qwen/Qwen3.6-35B-A3B",
            "large_model": "gpt-5.4",
            "routing_baselines": [
                {
                    "name": "random_50",
                    "policy": "bernoulli",
                    "p_large": 0.5,
                    "n_seeds": 3,
                    "description": "Route each task to large with p=0.5, independent",
                },
                {
                    "name": "random_10",
                    "policy": "bernoulli",
                    "p_large": 0.1,
                    "n_seeds": 3,
                    "description": "Route each task to large with p=0.1, ~matching router fallback rate",
                },
                {
                    "name": "random_oracle_p",
                    "policy": "bernoulli",
                    "p_large": "match_router_fallback",
                    "n_seeds": 5,
                    "description": "Use the router's actual large-call rate as p; isolates routing logic",
                },
            ],
            "compare_with": ["base", "large", "skills", "router", "full"],
            "report_metrics": ["task_pass", "routing_acc", "cost_vs_large", "fallback"],
            "arxiv_ref": "2506.06579",
            "data_motivation": (
                "cwy_35b_fullsplit cycle_3: always-large=60.11% < always-small=76.97%; "
                "router=69.66% (fallback=1.1%); random-50% baseline missing (paper W5)"
            ),
            "addresses_paper_weakness": "W5",
            "paper_claim_to_verify": (
                "Does the learned router outperform random routing at matched cost, "
                "or does its tau2 gain simply reflect low large-call rate?"
            ),
            "estimated_hours": 0.5,
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
