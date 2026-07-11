#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-11 (EXP-110, EXP-111).

A800 connectivity: offline since 2026-05-14 (day ~58). Apply when restored.

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
    python3 auto_research/pending_queue_update_2026_07_04.py            # EXP-106, EXP-107
    python3 auto_research/pending_queue_update_2026_07_05.py            # EXP-108, EXP-109

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_11.py            # EXP-110, EXP-111

Queue was ~113 pending on 2026-07-05 (+2 added: EXP-108, EXP-109).
Queue cap applied: >20 → max 2 new experiments.

Data motivating today's proposals
----------------------------------
results/e2e_4cyc_gpt55/cycle_3/e2e_ablation_summary.json (unchanged — A800 offline):
    HumanEval 4-cycle:
      large (always-large):        task_pass=96.34%, cost_vs_large=100%
      skills (always-small+proc):  task_pass=75.61%, cost_vs_large=10%
      router (logistic, cycle 3):  task_pass=92.68%, routing_acc=92.68%,
                                   cost_vs_large=27.56%, fallback=6.10%
      full (router+GRPO):          task_pass=92.68% ← SAME as router
    Root problem: ACR=52.4% — 43/82 GRPO groups have zero within-group reward
    variance → zero gradient → Full = Router.
    Skills gap: 75.61% vs large 96.34% — 20.7pp gap.

arxiv:2605.00433 — RECRL: Requirement-Aware Curriculum Reinforcement Learning
    for Code Generation (May 1, 2026):
    Uses execution correctness as a principled difficulty proxy to dynamically
    order GRPO training tasks from easy (high pass_rate) to hard (low pass_rate).
    Achieves +1.23%–+5.62% Pass@1 over five state-of-the-art baselines across
    5 LLMs × 5 benchmarks. Key insight: training on harder tasks before the model
    is ready collapses gradients and destabilizes GRPO; easy tasks first gives
    positive gradient signal that warms up the policy before hard tasks are
    introduced.

    RELEVANCE TO OUR ACR=52.4%:
    Our all-fail group problem (52.4% ACR) is partly an ordering problem: if the
    G=8 rollouts for a hard task all fail, the group contributes zero gradient.
    RECRL's difficulty-aware ordering defers hard tasks until the model can
    occasionally pass them (pass_rate > 0), converting previously-all-fail groups
    into mixed groups with non-zero gradient.
    Distinct from EXP-104 (selective-group GRPO filters zero-variance groups
    after they are sampled) and EXP-108 (sign advantage changes the advantage
    formula for zero-variance groups).
    RECRL prevents zero-variance groups from arising in the first place by
    ensuring the model has been warmed up on easier tasks.

    IMPLEMENTATION:
    1. After Phase 1 collect_traces, compute per-task difficulty:
       difficulty_i = 1 - pass_rate_cycle_k_small  (from traces.jsonl)
       (pass_rate = fraction of G=8 small-model rollouts that passed pytest)
    2. Sort HumanEval tasks by difficulty ascending (easy first).
    3. Partition into 3 buckets: easy (difficulty < 0.33), medium (0.33–0.67),
       hard (difficulty > 0.67).
    4. Feed to GRPO in staircase: epoch-1 easy only, epoch-2 easy+medium,
       epoch-3 all tasks.
    ~20 lines in grpo_train_simple.py (DataLoader curriculum sampler).

arxiv:2507.05386 — Reinforcement Fine-Tuning Naturally Mitigates Forgetting in
    Continual Post-Training (July 7, 2025; updated June 2026):
    Comparative study of SFT vs RFT (GRPO-based) on continual post-training across
    7 diverse tasks. Key finding: SFT leads to catastrophic forgetting of previously
    learned tasks when new tasks are introduced; RFT (online GRPO) naturally
    preserves earlier knowledge because reward maximization requires the model to
    maintain generalizable capabilities.
    Results on Qwen2.5-VL-7B-Instruct: SFT shows 8–15% forgetting of cycle-0 tasks
    by cycle-3; RFT shows <3% forgetting under the same continual training schedule.

    RELEVANCE TO OUR FOUR-ARM ABLATION:
    We have never formally measured forgetting in our 4-cycle run.
    Our ablation table tracks task_pass at each cycle, but not "retention of
    cycle-0 capabilities." The paper predicts:
      - skills arm (SFT only): should exhibit forgetting of cycle-0 easy tasks
        by cycle-3 (SFT crowds on cycle-3 hard traces → overwrites cycle-0 patterns)
      - full arm (SFT+GRPO): should exhibit less forgetting (GRPO preserves)
    If forgetting is measured and confirmed, it becomes a diagnostic that explains
    part of the 20.7pp skills gap (75.61% vs 96.34%): the skills arm not only
    fails hard tasks but also forgets some easy tasks it solved in cycle 0.
    This is a cheap experiment: evaluation only (no new training), runs in <30 min.
    The forgetting measurement also strengthens the paper's GRPO motivation:
    "GRPO-based routing + training preserves task diversity; SFT alone forgets."
"""
import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = (
    "/data0/home/zeyuwang/router-skills-evolve-data/humaneval/HumanEval.jsonl.gz"
)

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_07_11_001_recrl_execution_difficulty_curriculum_grpo_humaneval",
        "priority": 7,
        "kind": "grpo_curriculum_continual",
        "gpu": "auto",
        "rationale": (
            "arxiv:2605.00433 — Improving LLM Code Generation via Requirement-Aware "
            "Curriculum Reinforcement Learning (RECRL; May 1, 2026). "
            ""
            "BACKGROUND — GRPO ORDERING PROBLEM: "
            "At cycle k, our HumanEval GRPO sampler presents tasks uniformly at random. "
            "Hard tasks (difficulty ≥ 0.67 = small model fails ≥67% of rollouts) make "
            "up ~40% of HumanEval. When these appear early in training, all G=8 rollouts "
            "fail → zero variance → zero gradient under DAPO group-mean advantage. "
            "This contributes to ACR=52.4% (43/82 all-fail groups per cycle). "
            ""
            "RECRL APPROACH: "
            "Use execution correctness at cycle k as a principled difficulty proxy: "
            "  difficulty_i = 1 - pass_rate_{k,i}  (fraction of G=8 small rollouts that fail) "
            "Sort tasks by difficulty ascending. Feed to GRPO in a staircase: "
            "  Epoch 1: easy tasks (difficulty < 0.33, pass_rate > 0.67) "
            "    → model is likely to get some rollouts right → non-zero gradient "
            "  Epoch 2: easy + medium tasks (difficulty < 0.67) "
            "    → model has warmed up; medium tasks now occasionally pass "
            "  Epoch 3: all tasks (difficulty ≤ 1.0) "
            "    → hard tasks introduced after the model has consolidated easy skills "
            "This prevents hard all-fail groups from dominating early gradients. "
            "It is orthogonal to EXP-108 (sign advantage changes the formula for "
            "zero-variance groups) and complementary to EXP-104 (selective-group GRPO "
            "filters zero-variance groups at sample time). RECRL acts earlier: it "
            "reduces the fraction of zero-variance groups by scheduling easier tasks first. "
            ""
            "EMPIRICAL RESULTS (paper, arxiv:2605.00433): "
            "RECRL achieves +1.23%–+5.62% Pass@1 over five SOTA baselines across "
            "5 LLMs × 5 benchmarks (including HumanEval, MBPP, LeetCode). "
            "Gains are largest on medium-difficulty tasks (0.3 < difficulty < 0.7) — "
            "exactly the 'mixed group' regime where curriculum ordering matters most. "
            "Easy tasks (already solved) and very hard tasks (never solved) benefit less. "
            ""
            "RELATIONSHIP TO EXISTING EXPERIMENTS: "
            "EXP-104 (selective-group GRPO): filters groups with zero within-group "
            "reward variance at sampling time. Reactive: a zero-variance group is "
            "sampled and then discarded. "
            "EXP-108 (sign advantage): changes A formula so zero-variance groups "
            "still contribute gradient. Tolerates zero-variance groups. "
            "RECRL (this experiment): prevents zero-variance groups by warming up "
            "the model on easy tasks before hard tasks are introduced. Proactive. "
            "All three are composable; RECRL + sign advantage (EXP-108) is a natural "
            "combination if both show positive results. "
            ""
            "IMPLEMENTATION: "
            "In grpo_train_simple.py: "
            "1. After loading traces.jsonl, compute per-task difficulty: "
            "   pass_rate_i = sum(r == 1 for r in group_i) / len(group_i) "
            "   difficulty_i = 1 - pass_rate_i "
            "2. Sort tasks by difficulty_i ascending. "
            "3. Partition into three buckets: "
            "   easy   = difficulty_i < 0.33 "
            "   medium = 0.33 <= difficulty_i < 0.67 "
            "   hard   = difficulty_i >= 0.67 "
            "4. Custom DataLoader: epoch 1 = easy only, epoch 2 = easy+medium, "
            "   epoch 3 = all tasks. "
            "~20 lines of Python. No change to advantage computation or loss. "
            ""
            "METRICS TO TRACK: "
            "  (a) Fraction of zero-variance groups per epoch (should decrease in epoch 1 vs random) "
            "  (b) Per-task pass@1 at end of GRPO (vs DAPO baseline at 92.68%) "
            "  (c) Medium-difficulty task improvement (RECRL's largest benefit zone) "
            "  (d) Hard-task pass@1 (does curriculum ordering help hard tasks or not?) "
        ),
        "spec": {
            "pipeline": "humaneval_recrl_curriculum_grpo",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 1,
            "grpo_algo": "dapo_default",
            "curriculum": "recrl_execution_difficulty",
            "curriculum_buckets": {
                "easy":   {"difficulty_max": 0.33, "epochs": [1, 2, 3]},
                "medium": {"difficulty_min": 0.33, "difficulty_max": 0.67, "epochs": [2, 3]},
                "hard":   {"difficulty_min": 0.67, "epochs": [3]},
            },
            "n_epochs": 3,
            "n_generations": 8,
            "grpo_temperature": 1.0,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (DAPO uniform random: 92.68% HE, ACR=52.4%)",
                "arxiv:2605.00433 (RECRL +1.23%–+5.62% vs SOTA on HumanEval/MBPP)",
                "EXP-104 (selective-group GRPO: reactive zero-variance filter)",
                "EXP-108 (sign advantage: tolerant zero-variance formula change)",
            ],
            "implementation_file": "src/pipeline/grpo_train_simple.py",
            "implementation_note": (
                "Add curriculum sampler: compute difficulty = 1 - pass_rate per task "
                "from traces.jsonl; define easy/medium/hard buckets (thresholds 0.33/0.67); "
                "wrap the DataLoader with a per-epoch bucket filter that introduces "
                "progressively harder tasks across epochs 1→2→3. "
                "~20 lines. No change to advantage formula or model architecture. "
            ),
            "estimated_gpu_hours": 2.0,
        },
    },
    {
        "id": "exp_2026_07_11_002_forgetting_eval_rft_vs_sft_cycle_retention_humaneval",
        "priority": 7,
        "kind": "forgetting_eval",
        "gpu": "auto",
        "rationale": (
            "arxiv:2507.05386 — Reinforcement Fine-Tuning Naturally Mitigates Forgetting "
            "in Continual Post-Training (July 2025; updated June 2026). "
            ""
            "BACKGROUND — UNMEASURED FORGETTING IN OUR 4-CYCLE RUN: "
            "We track task_pass@1 per cycle using the full HumanEval set, but we do "
            "not measure 'retention' — whether cycle-3 models still solve tasks that "
            "were solved in cycle-0. Our ablation table shows skills arm at 75.61%, "
            "but does not disaggregate: "
            "  (a) tasks NEVER solved (hard tasks the small model can't crack), vs "
            "  (b) tasks solved in cycle-0 but FORGOTTEN by cycle-3 (catastrophic forgetting). "
            ""
            "THE PAPER'S FINDING: "
            "arxiv:2507.05386 compares SFT vs GRPO-based RFT in continual post-training "
            "across 7 diverse tasks on Qwen2.5-VL-7B-Instruct. Key result: "
            "  - SFT: 8–15% forgetting of cycle-0 task accuracy by cycle-3 "
            "  - RFT (GRPO): <3% forgetting under the same continual training schedule "
            "The paper explains this as: GRPO reward maximization requires the model to "
            "maintain generalizable coding patterns (because reward is task-agnostic); "
            "SFT on new traces overwrites cycle-0 weight patterns without that constraint. "
            ""
            "PREDICTED PATTERN IN OUR ABLATION: "
            "  - skills arm (SFT only, no GRPO): likely forgets some tasks it solved in "
            "    cycle 0 by cycle 3. Magnitude: if consistent with paper's 8–15%, "
            "    skills arm forgetting could account for 6–11pp of the 20.7pp gap vs large. "
            "  - router arm (routing only, trained small-model-cycle0): no GRPO → "
            "    uses cycle-3 SFT checkpoint → same forgetting pattern as skills arm. "
            "  - full arm (SFT+GRPO): GRPO should preserve cycle-0 capabilities; "
            "    expected forgetting <3% per paper. "
            "  - large arm: gold standard, no forgetting. "
            ""
            "WHY THIS MATTERS FOR THE PAPER: "
            "If SFT forgetting explains 6–11pp of the skills gap, the narrative becomes: "
            "'SFT-based skills distillation loses two ways: it struggles with hard tasks "
            "(Category-A problem) AND it forgets easy tasks already mastered in earlier "
            "cycles (catastrophic forgetting). GRPO-based full training mitigates both.' "
            "This motivates the full arm and provides a mechanistic explanation for "
            "skills arm underperformance beyond just 'hard tasks.' "
            "Directly cites arxiv:2507.05386 as the theoretical grounding. "
            ""
            "IMPLEMENTATION — EVAL ONLY (cheap, <30 min): "
            "1. Load cycle-0 checkpoints for each arm: "
            "   skills: llm_adapter/checkpoint-best (cycle 0) "
            "   full:   grpo_adapter/ (cycle 0) "
            "2. Load cycle-3 checkpoints for each arm (same paths, cycle 3). "
            "3. Identify 'cycle-0 solved tasks': tasks where the small model (base) "
            "   passed all G=1 greedy-decode rollouts in cycle-0 traces.jsonl. "
            "   Typically ~60% of HumanEval (easy tasks). "
            "4. Run cycle-3 model on these cycle-0-solved tasks; compute pass@1. "
            "5. Forgetting metric: "
            "   retention_rate = pass@1_{cycle-3 model on cycle-0 tasks} "
            "   forgetting_rate = 1 - retention_rate "
            "6. Report per arm: skills / router / full / large (reference). "
            "No training required; this is pure evaluation on existing checkpoints. "
            "Run on the A800 with the cycle-3 checkpoints already saved. "
            ""
            "EXPECTED TIMELINE: "
            "~20 min: load 4 model checkpoints (1.5B each) + evaluate on ~100 HE tasks "
            "→ 4 arms × ~60 cycle-0-solved tasks × greedy decode = fast. "
            ""
            "OPEN QUESTIONS THIS ANSWERS: "
            "  Q: Does GRPO mitigate forgetting in our specific 4-cycle setup? "
            "  Q: How much of the 20.7pp skills gap is forgetting vs never-solved? "
            "  Q: Does the router arm (which uses cycle-3 SFT small model) exhibit "
            "     forgetting, and does this affect routing quality? "
        ),
        "spec": {
            "pipeline": "humaneval_forgetting_eval_cycle_retention",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "eval_type": "forgetting_retention",
            "n_cycles_to_compare": [0, 3],
            "arms": ["skills", "router", "full", "large"],
            "cycle0_solved_task_selection": "pass_rate_cycle0_small >= 1.0",
            "decode_strategy": "greedy",
            "metrics": [
                "retention_rate_per_arm",
                "forgetting_rate_per_arm",
                "decomposition_skills_gap_forgetting_vs_never_solved",
            ],
            "eval_data": _HE_EVAL,
            "checkpoint_paths": {
                "skills_cycle0": "results/e2e_4cyc_gpt55/cycle_0/llm_adapter/checkpoint-best",
                "skills_cycle3": "results/e2e_4cyc_gpt55/cycle_3/llm_adapter/checkpoint-best",
                "full_cycle0": "results/e2e_4cyc_gpt55/cycle_0/grpo_adapter",
                "full_cycle3": "results/e2e_4cyc_gpt55/cycle_3/grpo_adapter",
            },
            "compare_with": [
                "arxiv:2507.05386 (RFT <3% forgetting vs SFT 8–15% in continual PTing)",
                "e2e_4cyc_gpt55 cycle_3 skills arm: 75.61% (20.7pp below large 96.34%)",
            ],
            "implementation_file": "src/pipeline/benches/forgetting_eval.py",
            "implementation_note": (
                "Add a forgetting_eval runner: (1) load cycle-0 and cycle-3 checkpoints "
                "per arm; (2) identify cycle-0-solved tasks from traces.jsonl pass_rate; "
                "(3) run greedy eval on cycle-3 model over cycle-0-solved tasks; "
                "(4) compute retention_rate and forgetting_rate per arm; "
                "(5) decompose skills gap: forgetting contribution vs never-solved. "
                "~50 lines; eval-only, no training, no GPU memory issues. "
            ),
            "estimated_gpu_hours": 0.5,
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
    existing_ids = {e["id"] for e in queue}

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
