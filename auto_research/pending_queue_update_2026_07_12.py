#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-12 (EXP-112, EXP-113).

A800 connectivity: offline since 2026-05-14 (day ~59). Apply when restored.

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
    python3 auto_research/pending_queue_update_2026_07_10_paper.py      # (paper-pipeline EXPs)
    python3 auto_research/pending_queue_update_2026_07_11.py            # EXP-110, EXP-111

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_12.py            # EXP-112, EXP-113

Queue was ~115 pending on 2026-07-11 (+2 added: EXP-110, EXP-111).
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
    Root problems:
      (A) ACR=52.4% — 43/82 GRPO groups have zero within-group reward
          variance → zero gradient → Full = Router.
      (B) Skills gap: 75.61% vs large 96.34% — 20.7pp gap.

AAAI 2027 deadline: 2026-08-15 (34 days from today).
A800 offline since 2026-05-14 (day 59).

arxiv:2605.12483 — Beyond GRPO and On-Policy Distillation: An Empirical
    Sparse-to-Dense Reward Principle for Language-Model Post-Training
    (May 12, 2026; revised May 14, 2026):

    CENTRAL CLAIM:
    Each labeled example should be allocated to the model and reward density
    where it is most informative. Sparse sequence-level reward (GRPO) is most
    informative for strong models that can explore (high entropy), while dense
    token-level teacher supervision (forward-KL / soft-label SFT) is better
    suited for compressing that behavior into smaller deployment models.

    FOUR-STAGE WORKFLOW:
    Stage 1 — Teacher RL:   Train strong teacher with sparse GRPO.
                            Teacher gains exploration headroom (in our setup:
                            GPT-5.5 plays this role implicitly via oracle traces)
    Stage 2 — KL Warmup:   Distill teacher → student using FORWARD-KL loss
                            (student aligns its full distribution to teacher's
                            predicted distribution, not just argmax).
                            Unlike SFT (cross-entropy on one-hot teacher argmax),
                            forward-KL transfers the teacher's uncertainty over
                            alternative correct tokens — multi-modal distributions,
                            whitespace variants, style alternatives.
    Stage 3 — On-Policy:   Optional on-policy distillation with student rollouts.
    Stage 4 — Student RL:  Optional sparse GRPO on student, now initialized
                           better by the KL warmup.

    KEY FINDING:
    An RL-improved 8B teacher distilled through the KL warmup (Stage 2) into a
    3B student outperforms direct GRPO on the 3B student across all 5 evaluated
    benchmarks (GSM8K, MATH, MBPP, HumanEval, LiveCodeBench). The gap is largest
    on tasks where the student's exploration probability is near zero (hard tasks
    with pass_rate ≈ 0 under direct GRPO sampling) — exactly our ACR=52.4% regime.

    RELEVANCE TO OUR SKILLS GAP (B):
    Our skills arm (SFT on GPT-5.5 traces → Phase 3a) uses hard-label cross-entropy:
      loss = CE(student output, teacher argmax)
    This discards the teacher's soft token distribution — if GPT-5.5 places 30%
    probability on an equivalent but stylistically different solution, SFT binarizes
    this to 0 and the student gets no signal about alternative valid patterns.
    Forward-KL transfers this: the student is encouraged to produce not just the
    teacher's most likely token, but to match the full distribution. For HumanEval
    with 82 tasks and ~60% of tasks solved by GPT-5.5 with multiple valid approaches,
    this could close part of the 20.7pp skills gap by giving the student more signal.

    RELEVANCE TO ACR=52.4% PROBLEM (A):
    The paper explicitly notes: "direct sparse GRPO on small models that cannot
    explore hard tasks (pass_rate ≈ 0) is statistically wasteful." The KL-warmup
    approach skips the all-fail GRPO groups entirely for the warmup phase, then
    optionally applies GRPO after warmup. If the student's pass_rate on hard tasks
    increases during KL-warmup (because it learns more multi-modal solution patterns),
    some previously all-fail GRPO groups become mixed-variance groups after warmup —
    reducing ACR.

    DISTINCT FROM EXISTING EXPERIMENTS:
    EXP-VeRPO (exp_2026_06_23_001): uses partial test-case pass rates (dense scalar
      reward) at the sequence level. Changes the reward function, not the loss.
    EXP-EGCA (exp_2026_06_06_001, exp_2026_06_17_001): token-level advantage masking
      based on execution divergence. Changes credit assignment within GRPO.
    EXP-112 (this): replaces the GRPO loss entirely with forward-KL distillation
      for the Phase 3a-b block. Changes the training objective from "reward
      maximization from scratch" to "compress teacher distribution."
    All three are composable.

arxiv:2605.07689 (Sign Advantage) + arxiv:2605.00433 (RECRL Curriculum):
    Both are already individually queued (EXP-108: sign advantage; EXP-110: RECRL
    curriculum). Their mechanisms are non-overlapping in the code:
    - EXP-108 (sign advantage): modifies the advantage formula A = 2r - 1, a 2-line
      change in grpo_train_simple.py that makes A = -1 for failed rollouts (instead
      of 0). All groups contribute gradient; zero-variance groups now get negative
      gradient signal toward correct behavior.
    - EXP-110 (RECRL curriculum): modifies the DataLoader to sort tasks by difficulty
      ascending (easy first), feeding hard tasks only in epoch 3. Prevents all-fail
      groups from dominating early training.
    The composition EXP-113 runs both simultaneously:
    - RECRL staircase reduces the FREQUENCY of zero-variance groups (fewer hard groups
      in early epochs).
    - Sign advantage handles the REMAINING zero-variance groups (gives them gradient).
    Predicted to be stronger than either alone; if additive, could push from 92.68%
    to ≥ 94% pass@1.
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
        "id": "exp_2026_07_12_001_fwdkl_teacher_distill_bridge_humaneval",
        "priority": 7,
        "kind": "grpo_continual",
        "gpu": "auto",
        "rationale": (
            "arxiv:2605.12483 — Beyond GRPO and On-Policy Distillation: An Empirical "
            "Sparse-to-Dense Reward Principle for Language-Model Post-Training "
            "(May 12, 2026). "
            ""
            "THE SPARSE-TO-DENSE PRINCIPLE: "
            "Sparse sequence-level reward (GRPO) is best for strong models that can "
            "explore. Dense token-level teacher supervision (forward-KL) is better for "
            "compressing teacher behavior into smaller students. For our Qwen2.5-Coder-1.5B "
            "student, direct GRPO suffers ACR=52.4% (hard tasks never pass, zero gradient). "
            "Forward-KL distillation from GPT-5.5 bypasses this: it aligns the student's "
            "full token distribution to the teacher's, capturing multi-modal solutions "
            "that binary SFT (hard-label CE) discards. "
            ""
            "CURRENT PIPELINE (Phase 3a): "
            "SFT loss = CE(student_output, one_hot(teacher_argmax)) "
            "This trains the student to reproduce the teacher's most likely token at each "
            "step but discards the teacher's uncertainty about alternative correct tokens. "
            ""
            "PROPOSED CHANGE (Phase 3a modified): "
            "Forward-KL loss = KL(teacher_distribution || student_distribution) "
            "  = sum_t [ teacher_prob(t) * log(teacher_prob(t) / student_prob(t)) ] "
            "Student minimizes this by matching teacher's full predicted distribution. "
            "For HumanEval, if GPT-5.5 assigns 30% to an equivalent solution variant, "
            "the student is trained to capture this distribution, not just the argmax. "
            "NOTE: GPT-5.5 does not expose full logprobs; we approximate the teacher "
            "distribution using temperature-scaled sampling (T=0.7) with K=8 diverse "
            "GPT-5.5 traces per task (already collected with SCALING_FORCE_BOTH=1). "
            "We construct an empirical teacher distribution per token position using "
            "token frequency across the K traces. "
            ""
            "IMPLEMENTATION: "
            "1. In traces_to_sft.py: instead of writing a single teacher trace per task, "
            "   collect all K GPT-5.5 traces per task and compute empirical token "
            "   frequencies at each position (after alignment). "
            "2. In train_small_model.py: replace CE loss with forward-KL using the "
            "   empirical teacher distribution as the target. "
            "   implementation_note: use soft-label cross-entropy "
            "   loss = -sum_t [ teacher_empirical(t) * log_softmax(student_logits)[t] ] "
            "   (equivalent to forward-KL up to entropy constant). "
            "3. Phase 3b (GRPO): keep unchanged, but the student now starts from a "
            "   better initialization — richer code patterns from multi-modal distillation. "
            "~30 lines: 10 lines in traces_to_sft.py (aggregation), 10 in "
            "train_small_model.py (loss change), 10 in the data collator. "
            ""
            "METRICS TO TRACK: "
            "  (a) Skills arm pass@1 after Phase 3a only (forward-KL vs SFT) "
            "  (b) ACR after Phase 3b GRPO (fraction of zero-variance groups; "
            "      predicted to decrease if student starts with better coverage) "
            "  (c) Full arm pass@1 after Phase 3b GRPO (composition benefit) "
            "  (d) Hard-task pass@1 (difficulty >= 0.67): largest expected gain zone "
            "      (these are tasks where teacher distributes mass over multiple approaches) "
            ""
            "COMPARISON BASELINE: "
            "  e2e_4cyc_gpt55 cycle_3: skills 75.61%, full 92.68% (ACR=52.4%) "
            "  EXP-VeRPO (dense scalar reward via partial test-case pass rates): "
            "    modifies reward, not loss; orthogonal to forward-KL. "
            "  EXP-EGCA (execution-grounded token masking): modifies credit assignment "
            "    within GRPO rollouts; orthogonal to forward-KL distillation. "
            ""
            "EXPECTED OUTCOME: "
            "  Skills arm pass@1 >= 78% (+2.4pp) from richer teacher distillation. "
            "  ACR drop to ≤ 45% (from 52.4%) due to better student initialization "
            "    on hard tasks after forward-KL warmup. "
            "  Full arm pass@1 >= 93.5% if GRPO benefits from the warmed-up student. "
            "  If forward-KL improves skills but not full, the bottleneck is confirmed "
            "  to be GRPO credit assignment (not distillation quality), motivating "
            "  EXP-108/EXP-110 as the higher-priority fix. "
        ),
        "spec": {
            "pipeline": "humaneval_fwdkl_teacher_distill_then_grpo",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 1,
            "phase_3a_loss": "forward_kl",
            "phase_3a_teacher_traces_k": 8,
            "phase_3a_teacher_distribution": "empirical_token_frequency",
            "phase_3a_temperature_for_teacher_sampling": 0.7,
            "phase_3b_grpo": True,
            "phase_3b_algo": "dapo_default",
            "grpo_temperature": 1.0,
            "n_generations": 8,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3: skills 75.61%, full 92.68% (ACR=52.4%, hard-CE SFT)",
                "arxiv:2605.12483 (forward-KL dense bridge outperforms direct GRPO on 5 benchmarks)",
                "EXP-VeRPO (2026-06-23-001): dense scalar reward via partial test pass rate",
                "EXP-EGCA (2026-06-06-001): token-level credit assignment",
            ],
            "implementation_files": [
                "src/pipeline/traces_to_sft.py",
                "src/pipeline/train_small_model.py",
            ],
            "implementation_note": (
                "traces_to_sft.py: add multi-trace aggregation mode: for each task, "
                "collect all K GPT-5.5 traces (from traces.jsonl SCALING_FORCE_BOTH), "
                "compute empirical token frequency distribution per position after "
                "sequence alignment (pad shorter sequences). "
                "train_small_model.py: replace CE loss with soft-label CE "
                "(loss = -sum_t [teacher_freq[t] * log_softmax(logits)[t]]); "
                "this equals forward-KL(teacher||student) up to the teacher's entropy. "
                "~30 lines total. Phase 3b GRPO unchanged. "
            ),
            "arxiv_ref": "2605.12483",
            "estimated_gpu_hours": 2.5,
        },
    },
    {
        "id": "exp_2026_07_12_002_recrl_sign_advantage_composition_humaneval",
        "priority": 7,
        "kind": "grpo_curriculum_continual",
        "gpu": "auto",
        "rationale": (
            "Composition of arxiv:2605.00433 (RECRL curriculum) and arxiv:2605.07689 "
            "(sign advantage). Both mechanisms are individually queued (EXP-110: RECRL; "
            "EXP-108: sign advantage) and expected to improve the ACR=52.4% problem. "
            "This experiment tests whether they compose additively or supraadditively. "
            ""
            "BACKGROUND — TWO ORTHOGONAL MECHANISMS FOR ZERO-VARIANCE GROUPS: "
            ""
            "Mechanism A (RECRL, arxiv:2605.00433, EXP-110): "
            "Prevent zero-variance groups by curriculum ordering. "
            "Sort tasks by difficulty_i = 1 - pass_rate_i (ascending). "
            "Epoch 1: easy tasks only (difficulty < 0.33) — mostly non-zero variance "
            "Epoch 2: easy + medium tasks — model warmed up on easier tasks "
            "Epoch 3: all tasks — hard tasks introduced after the model can "
            "occasionally solve medium-difficulty ones "
            "EFFECT: reduces the fraction of zero-variance groups in epoch 1 from "
            "~52.4% to ~15–20% (predicted). Hard tasks remain all-fail even in epoch 3 "
            "for tasks that the base model is far from solving. "
            ""
            "Mechanism B (Sign Advantage, arxiv:2605.07689, EXP-108): "
            "Handle remaining zero-variance groups by replacing group-mean centering: "
            "  Standard DAPO: A = (r - mean(group_rewards)) → 0 if all rewards equal "
            "  Sign Advantage: A = 2r - 1 → +1 if r=1 (correct), -1 if r=0 (wrong) "
            "EFFECT: ALL groups contribute gradient, including all-fail groups (A=-1 "
            "pushes away from wrong patterns) and all-pass groups (A=+1 reinforces). "
            "Confirmed: +45.4pp GSM8K vs standard DAPO (7 seeds, p<0.0001). "
            ""
            "WHY COMPOSE THEM: "
            "RECRL and sign advantage are code-level non-overlapping: "
            "  RECRL: modifies the DataLoader (task ordering by difficulty) "
            "  Sign Advantage: modifies the advantage formula (A = 2r - 1 vs A = r - mean) "
            "Both target the ACR=52.4% problem from different angles: "
            "  RECRL: proactive (reduces frequency of zero-variance groups) "
            "  Sign Advantage: reactive (makes zero-variance groups still contribute) "
            "If both work independently: "
            "  RECRL alone: epoch-1 has ~15% zero-variance groups (prediction) "
            "  Sign Advantage alone: zero-variance group rate is moot (all groups learn) "
            "  Composition: RECRL provides gradient diversity by ordering; "
            "    sign advantage ensures even hard all-fail groups get learning signal "
            "    → maximum coverage of the training distribution "
            ""
            "PREDICTED COMPOSITION BENEFIT: "
            "RECRL improves medium-difficulty tasks most (+2–4pp, per 2605.00433). "
            "Sign advantage improves hard all-fail tasks most (A=-1 gradient signal "
            "on hard tasks that RECRL defers to epoch 3). "
            "Together: RECRL handles medium tasks → sign advantage handles hard tasks "
            "→ combined expected improvement > either alone. "
            "If EXP-108 (sign advantage standalone) achieves ~93.5% and EXP-110 "
            "(RECRL standalone) achieves ~93.5%, composition may reach 94–95%. "
            ""
            "IMPLEMENTATION (two additive changes from the DAPO baseline): "
            "1. DataLoader change from RECRL (EXP-110): "
            "   difficulty_i = 1 - pass_rate_i from cycle-0 traces.jsonl "
            "   Epoch 1: easy (difficulty < 0.33) "
            "   Epoch 2: easy + medium (difficulty < 0.67) "
            "   Epoch 3: all tasks "
            "2. Advantage formula change from Sign Advantage (EXP-108): "
            "   A = 2r - 1  (replaces standard group-mean centering) "
            "   r=1 → A=+1; r=0 → A=-1 "
            "Total: ~22 lines combined (20 from RECRL, 2 from sign advantage). "
            "These changes are in different code regions and cannot conflict. "
            ""
            "RELATIONSHIP TO INDIVIDUAL EXPERIMENTS: "
            "EXP-108 (sign advantage, standalone MBPP): priority 9 — runs first if A800 restores "
            "EXP-110 (RECRL curriculum, HumanEval): priority 7 — runs in parallel if possible "
            "EXP-113 (this, composition): priority 7 — most informative AFTER both individuals "
            "  run, but queued now so it can run immediately after if both are positive. "
            ""
            "METRICS TO TRACK: "
            "  (a) Zero-variance group fraction per epoch (RECRL effect: should be low in E1) "
            "  (b) Pass@1 per difficulty bucket: "
            "      easy (difficulty < 0.33): should be high in both individual and composition "
            "      medium (0.33–0.67): RECRL's main gain zone "
            "      hard (> 0.67): sign advantage's main gain zone (A=-1 gradient signal) "
            "  (c) Overall pass@1 vs 92.68% baseline and vs EXP-108, EXP-110 individually "
            "  (d) Gradient norm per epoch: sign advantage adds gradient even in epoch 3 "
            "      hard tasks; RECRL ensures epoch 1-2 gradient comes from tractable tasks "
        ),
        "spec": {
            "pipeline": "humaneval_recrl_sign_advantage_composition_grpo",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 1,
            "grpo_algo": "sign_advantage_dapo",
            "grpo_advantage_formula": "A = 2*r - 1",
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
                "e2e_4cyc_gpt55 cycle_3 (DAPO uniform random): 92.68%, ACR=52.4%",
                "EXP-108 (sign advantage standalone, MBPP): pending",
                "EXP-110 (RECRL curriculum standalone, HumanEval): pending",
                "arxiv:2605.07689 (+45.4pp GSM8K vs standard DAPO, 7 seeds)",
                "arxiv:2605.00433 (+1.23%–+5.62% HumanEval/MBPP vs SOTA)",
            ],
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
            ],
            "implementation_note": (
                "Two additive changes from DAPO baseline: "
                "(1) RECRL DataLoader: compute difficulty = 1 - pass_rate per task "
                "from traces.jsonl; bucket into easy/medium/hard; filter per epoch. "
                "(2) Sign advantage: replace DAPO group-mean centering with A = 2r - 1. "
                "These changes are in distinct code regions (DataLoader vs loss compute) "
                "and cannot interact. Total: ~22 lines. No change to model architecture "
                "or SFT phase. "
            ),
            "arxiv_refs": ["2605.00433", "2605.07689"],
            "estimated_gpu_hours": 2.0,
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
