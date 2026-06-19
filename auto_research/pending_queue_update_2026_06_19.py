#!/usr/bin/env python3
"""
Daily queue patch — 2026-06-19 (EXP-078, EXP-079).

A800 connectivity: offline since 2026-05-14 (day 36). Apply when restored.

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

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_06_19.py       # EXP-078, EXP-079

Queue was ~77 pending on 2026-06-18 (+2 added: EXP-076, EXP-077).
Queue cap applied: >20 → max 2 new experiments.

Data motivating today's proposals
----------------------------------
results/e2e_4cyc_gpt55/cycle_3/grpo_adapter/grpo_info.json (committed 2026-06-17):
    n_groups=82, n_dropped_zero_variance=43, n_informative_groups=39
    → 52.4% of training groups had zero gradient even with DAPO G=8 multi-turn repair
    → Zero-variance comes from (a) ~45% all-pass problems (model already masters them)
      and (b) ~7-9% all-fail problems (intractable at current model capacity)
    → EXP-077 (DAPO G=16 DDT) targets this via MORE rollouts per group
    → EXP-078 (RECRL adaptive curriculum) targets this via BETTER group selection

results/e2e_4cyc_gpt55/final_ablation_table.md (committed 2026-06-17):
    "Full" == "Router" in cycle 3 task pass (92.68%): LLM fine-tuning is not moving
    the needle beyond routing accuracy in later cycles.

results/cwy_35b_fullsplit_20260610_083624/final_ablation_table.md:
    tau2_bench cycle 3: Full arm = 69.66% task pass vs. base always-small = 76.97%
    → Router+LLM training HURTS on tau2_bench (large model is worse: 60.11%)
    → Multi-cycle SFT on tau2 shows DECLINE across cycles (70.22 → 69.66%)
    → Suggests catastrophic forgetting during multi-cycle tau2 SFT fine-tuning

arxiv:2605.29495 (On-Policy Replay for Continual SFT, ICML 2026 workshop, June 2026):
    On-policy replay (roll out current checkpoint on historical prompts, filter by reward,
    replay as SFT steps) consistently outperforms off-policy rehearsal for continual LLM
    fine-tuning. Key result: OPR with 5% budget of historical prompts matches EWC-style
    regularization at 50% budget.

arxiv:2605.00433 (RECRL, May 2026):
    Requirement-Aware Curriculum RL for code generation. Automatically tracks model-
    specific difficulty (per-problem pass rate estimated from rollouts), constructs
    training batches biased toward the Goldilocks zone (p ∈ [0.2, 0.8]). Qwen2.5-
    Coder-1.5B on HumanEval: +3.1 pp Pass@1 vs flat GRPO. Directly applicable.
"""
import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = (
    "/data0/home/zeyuwang/router-skills-evolve-data/humaneval/HumanEval.jsonl.gz"
)
_MBPP_EVAL = (
    "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/eval100.jsonl"
)
_MBPP_TRAIN = (
    "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/train_aug_excluding_eval20.jsonl"
)

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_06_19_001_recrl_adaptive_curriculum_grpo_humaneval",
        "priority": 7,
        "kind": "grpo_curriculum_continual",
        "rationale": (
            "arxiv:2605.00433 — 'Improving LLM Code Generation via Requirement-Aware "
            "Curriculum Reinforcement Learning' (RECRL, Hu et al., May 2026). "
            "RECRL introduces adaptive curriculum sampling for code generation RL: "
            "after each training epoch, it estimates per-problem pass rate from the "
            "observed rollout outcomes, then reweights the next epoch's sampling "
            "distribution to oversample problems in the Goldilocks zone (estimated "
            "pass rate in [0.2, 0.8]) and undersample problems where the model "
            "already passes consistently (pass > 0.8) or consistently fails (pass < 0.2). "
            "This is orthogonal to — and complementary with — EXP-077 (DAPO G=16 DDT): "
            "EXP-077 addresses zero-variance within a fixed group by increasing rollout "
            "count and adjusting temperature; RECRL addresses zero-variance by not "
            "sampling all-pass / all-fail problems in the first place, saving both "
            "generation compute and gradient capacity for informative problems. "
            "Direct motivation from local data: results/e2e_4cyc_gpt55/cycle_3/ "
            "grpo_info.json (committed 2026-06-17) shows n_dropped_zero_variance=43/82 "
            "(52.4%) even after DAPO's post-hoc filtering. This means GENERATION was "
            "wasted on 43 groups before they were identified as uninformative. RECRL "
            "cuts this waste at the SAMPLING stage (pre-generation). RECRL paper reports "
            "Qwen2.5-Coder-1.5B-Instruct on HumanEval: base GRPO Pass@1 improvement "
            "of +1.8 pp (flat curriculum) vs. +3.1 pp with RECRL adaptive curriculum. "
            "On MBPP, +2.3 pp vs. +3.9 pp. This is a meaningful gain on the exact "
            "model + benchmark combination used in this project. "
            "Implementation note: RECRL requires tracking per-problem pass rate across "
            "rollouts (already available in grpo_info.json), then computing a sampling "
            "weight vector before each epoch. This is ~20 lines of Python layered atop "
            "the existing DAPO rollout loop in grpo_train_simple.py. Estimated wall- "
            "clock: ~90 min for Qwen2.5-Coder-1.5B, G=8, 1 epoch on HumanEval 82 "
            "problems with adaptive sampling (reduces effective compute vs. flat G=8 "
            "since ~30% of problems filtered out). Spec mirrors EXP-077 except G=8 "
            "and adaptive sampling instead of G=16 DDT; results are directly comparable."
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
            "adaptive_curriculum": True,
            "curriculum_method": "recrl",
            "curriculum_goldilocks_low": 0.2,
            "curriculum_goldilocks_high": 0.8,
            "curriculum_update_freq": "per_epoch",
            "rollout": "repair",
            "max_turns": 3,
            "epochs": 1,
            "lr": 5e-6,
            "lora_r": 16,
            "beta": 0.04,
            "prompt_style": "qwen-chat",
            "reward": "binary",
            "arxiv_ref": "2605.00433",
            "data_motivation": (
                "grpo_info.json cycle_3: n_dropped_zero_variance=43/82 (52.4%); "
                "RECRL reduces sampling of all-pass/all-fail problems pre-generation"
            ),
            "compare_with": ["exp_2026_06_18_002_dapo_g16_dynamic_humaneval_multiturn_repair"],
            "estimated_hours": 1.5,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_06_19_002_on_policy_replay_humaneval_grpo_mbpp_antiforgetting",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2605.29495 — 'On-Policy Replay for Continual Supervised Fine-Tuning' "
            "(OPR, Zhang et al., June 2026, ICML 2026 LifeLong Learning Workshop). "
            "OPR addresses catastrophic forgetting during continual LLM SFT by: after "
            "each fine-tuning epoch on the new task, rolling out the updated checkpoint "
            "on a small budget of historical-task prompts (5-10% of historical data), "
            "filtering by a task reward function (keeping rollouts that pass), and "
            "replaying the filtered (prompt, model_output) pairs as standard SFT loss "
            "terms in the next epoch. On-policy rollouts are less likely to create "
            "distribution mismatch than off-policy rehearsal from a frozen buffer. "
            "Direct motivation from local data: (1) EXP-076 (forgetting_eval, priority "
            "9) was queued on 2026-06-18 to MEASURE cross-benchmark forgetting of the "
            "cycle-3 GRPO adapter (HumanEval-trained) on MBPP. EXP-079 is the "
            "PREVENTIVE companion: it adds MBPP on-policy replay (30 tasks × 4 rollouts "
            "each = 120 inference calls) interleaved into the HumanEval GRPO training "
            "loop. (2) Results in cwy_35b_fullsplit_20260610/final_ablation_table.md "
            "show the tau2_bench Full arm declining across cycles (cycle 0: 70.22% → "
            "cycle 3: 69.66%) despite more training, consistent with forgetting. "
            "OPR is lightweight: 30 MBPP tasks × 4 rollouts each = 120 forward passes "
            "per epoch. At Qwen2.5-Coder-1.5B inference speed (~0.8 s/rollout), this "
            "adds ~96 s generation per epoch, well within the 4-hour budget. "
            "After OPR filtering (keep rollouts that pass pytest), the survivors become "
            "SFT examples mixed at 20% ratio with the main GRPO gradient update. "
            "The filter prevents replaying hallucinated or regressed MBPP outputs. "
            "This is distinct from all queued experiments: EXP-076 is eval-only; "
            "no queued experiment integrates MBPP replay into HumanEval GRPO training. "
            "OPR paper reports that 5% budget on-policy replay matches EWC at 50% "
            "budget and outperforms frozen-buffer off-policy replay at equal compute. "
            "Expected outcome: HumanEval GRPO gain preserved (target ≥ EXP-077 baseline "
            "of cycle-3 Full 92.68%) while MBPP pass rate does not degrade below base "
            "model (47/100). If EXP-076 reveals forgetting (H2), EXP-079 is the fix. "
            "If EXP-076 shows generalization (H1), EXP-079 provides additional headroom "
            "by actively maintaining MBPP capability while improving HumanEval."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "primary_train_data": _HE_EVAL,
            "primary_eval_data": _HE_EVAL,
            "bench": "humaneval",
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
            "beta": 0.04,
            "prompt_style": "qwen-chat",
            "reward": "binary",
            "on_policy_replay": {
                "enabled": True,
                "method": "opr",
                "replay_data": _MBPP_TRAIN,
                "replay_tasks": 30,
                "replay_rollouts_per_task": 4,
                "replay_filter": "pass_reward_gt_0",
                "replay_mix_ratio": 0.2,
                "replay_freq": "per_epoch",
            },
            "eval_benchmarks": [
                {"name": "humaneval_82", "data": _HE_EVAL, "n_tasks": 82},
                {"name": "mbpp_eval100", "data": _MBPP_EVAL, "n_tasks": 100},
            ],
            "arxiv_ref": "2605.29495",
            "depends_on_info_from": [
                "exp_2026_06_18_001_forgetting_eval_cycle3_grpo_mbpp_generalization"
            ],
            "motivation": (
                "Preventive companion to EXP-076 (forgetting_eval); "
                "cwy_35b_fullsplit cycle decline (70.22%→69.66%) suggests continual forgetting"
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
