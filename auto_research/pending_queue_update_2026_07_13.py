#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-13 (EXP-114, EXP-115).

A800 connectivity: offline since 2026-05-14 (day ~60). Apply when restored.

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
    python3 auto_research/pending_queue_update_2026_07_12.py            # EXP-112, EXP-113

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_13.py            # EXP-114, EXP-115

Queue was ~119 pending on 2026-07-12 (+2 added: EXP-112, EXP-113).
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

AAAI 2027 deadline: 2026-08-15 (33 days from today).
A800 offline since 2026-05-14 (day 60).

arxiv:2606.11709 — RLCSD: Reinforcement Learning with Contrastive On-Policy
    Self-Distillation (June 11, 2026; Tsinghua / Alibaba Tongyi Lab):

    PROBLEM THEY SOLVE:
    Standard on-policy self-distillation (OPSD) methods provide dense token-level
    supervision by aligning the model's distribution toward its outputs under
    "privileged context" (e.g., a verified correct solution hint).
    But they suffer from "privilege-induced style drift": the learning signal
    concentrates on style tokens (shorter, more direct outputs under the hint)
    rather than task-bearing tokens (the reasoning steps that matter).
    This destabilizes training and causes response length collapse.

    CONTRASTIVE SOLUTION:
    RLCSD contrasts the distributional gap under a CORRECT hint vs. the gap under
    a WRONG hint. Both hint conditions induce style drift; the contrastive difference
    cancels the style component and yields a signal that is concentrated on
    task-bearing tokens — the specific reasoning transitions that the correct hint
    enables and the wrong hint suppresses.

    FORMAL LOSS:
    L_RLCSD = L_GRPO + λ · [KL(M(·|x, correct_hint) || M_θ(·|x))
                             - KL(M(·|x, wrong_hint) || M_θ(·|x))]
    where M(·|x, hint) is the model's own distribution conditioned on the hint.
    The contrastive term cancels style drift and focuses gradient on the
    reasoning-critical token positions.

    RESULTS:
    Outperforms GRPO and prior OPSD methods on Qwen3 (1.7B/4B/8B) and OLMo-3-7B-Think
    across mathematical and logical reasoning benchmarks. The contrastive principle is
    modular and can plug into existing OPSD methods.

    RELEVANCE TO OUR ACR=52.4% PROBLEM:
    Standard GRPO suffers from zero gradient on all-fail groups (ACR=52.4%):
      if all G=8 rollouts fail, the group mean = 0, all advantages = 0, no gradient.
    RLCSD's contrastive term provides DENSE TOKEN-LEVEL signal even for groups
    where all rollouts fail: the model is compared to its own outputs under a
    correct hint (which we can construct from the GPT-5.5 traces in traces.jsonl),
    and the wrong-hint contrastive baseline uses a perturbed or alternate wrong trace.
    This gives gradient signal on every token in every training example, independent
    of whether any rollout passed.

    ADAPTATION TO CODE GENERATION (distinct from math):
    For code tasks: "correct hint" = first 3 lines of the GPT-5.5 correct trace
    (already available from SCALING_FORCE_BOTH=1 traces.jsonl).
    "Wrong hint" = first 3 lines of a failing rollout from the G=8 samples.
    The contrastive signal cancels stylistic differences (indentation, spacing)
    and concentrates on algorithmic-bearing tokens (loop structure, data structure
    selection, boundary conditions).
    λ=0.05 (small coefficient; GRPO loss dominates to preserve explore-and-exploit).

    DISTINCT FROM EXISTING EXPERIMENTS:
    EXP-112 (fwd-KL distillation): uses full GPT-5.5 teacher trace, matches
      full token distribution. No contrastive cancellation of style drift.
    EXP-EGCA (execution-grounded credit): masks tokens post-hoc based on execution
      divergence points. No dense signal for all-fail groups (still zero gradient
      where execution never runs).
    EXP-108 (sign advantage, A=2r-1): provides A=−1 gradient for all-fail groups,
      but at the SEQUENCE level (one scalar per rollout). RLCSD provides
      token-level gradient at every position via the contrastive distillation term.
    EXP-114 (this): RLCSD provides dense token-level gradient independent of
      rollout pass/fail, canceling style drift via contrastive wrong-hint baseline.
    All four are composable; EXP-114 is the only one that eliminates style drift.

arxiv:2601.20861 — Evolutionary Strategies lead to Catastrophic Forgetting in LLMs
    (January 2026):

    KEY FINDING:
    ES and GRPO reach comparable accuracy on math/reasoning tasks, but the updates
    are geometrically opposite:
      ES updates: dense (broadly distributed across all parameters), large ℓ2 norm
      GRPO updates: highly sparse (concentrated in a small fraction of parameters)
    This sparsity explains GRPO's lower forgetting: sparse updates minimally
    interfere with parameter regions encoding prior task knowledge.

    IMPLICATION FOR OUR MULTI-CYCLE PIPELINE:
    Our pipeline runs 4 GRPO cycles, each on traces from the previous cycle.
    Even with GRPO's inherently sparse updates, each cycle's gradient concentrates
    on tasks where the policy changed most (the hardest tasks, non-zero advantage).
    This may cause "within-GRPO forgetting": cycle-k GRPO updates on hard tasks
    could interfere with easy tasks the model solved in cycle-(k-1).

    DIAGNOSTIC QUESTION:
    Is the Full=Router equivalence (92.68%) partly due to catastrophic forgetting?
    Specifically: does GRPO in cycle k forget some previously-solvable easy tasks?
    If so, the ACR=52.4% explanation (zero-variance groups) may be partial:
    the other component is forgetting of previously-passing tasks.
    If forgetting explains, say, 5pp of the Full=Router gap, then sign advantage
    (EXP-108) won't fully close it — we'd need replay (EWC or experience replay)
    in addition.

    DISTINCTION FROM EXP-111 (RFT vs SFT cycle retention):
    EXP-111 compares METHODS: does RFT or SFT preserve cycle-k abilities better?
    EXP-115 (this) characterizes WITHIN-METHOD forgetting in the default pipeline
    (GRPO) across cycles, correlating forgetting with gradient sparsity.
    EXP-111 is prescriptive (which method is better); EXP-115 is diagnostic
    (which tasks are forgotten and why).

    IMPLEMENTATION (eval-only, ~0.5h):
    1. Load each cycle's LLM checkpoint: cycle_0/llm_adapter/, cycle_1/, cycle_2/, cycle_3/
    2. For each checkpoint, run pass@1 eval on ALL 82 HumanEval tasks (not just the
       current cycle's subset).
    3. Build a (4 checkpoints × 82 tasks) pass/fail matrix.
    4. For each task i: compute "forgetting score" = max(pass_in_earlier_cycles)
       - pass_in_cycle_3 (1 if task was solved before but not in cycle 3 checkpoint).
    5. Load gradient update magnitude (||grad||_2 per task group) from GRPO training logs.
    6. Correlate forgetting scores with per-task gradient magnitudes from later cycles.
    7. Report: which task categories suffer highest forgetting? Correlation with ACR?
    Key output: is the Full=Router gap explainable by forgetting alone, or does ACR=52.4%
    account for all of it?
"""
import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = (
    "/data0/home/zeyuwang/router-skills-evolve-data/humaneval/HumanEval.jsonl.gz"
)
_CYCLE_DIR = "/data0/home/zeyuwang/router-skills-evolve-data/e2e_4cyc_gpt55"

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_07_13_001_rlcsd_contrastive_self_distill_grpo_humaneval",
        "priority": 7,
        "kind": "grpo_continual",
        "gpu": "auto",
        "rationale": (
            "arxiv:2606.11709 — RLCSD: Reinforcement Learning with Contrastive "
            "On-Policy Self-Distillation (June 11, 2026; Tsinghua / Alibaba Tongyi Lab). "
            ""
            "THE CORE PROBLEM THIS SOLVES: "
            "Standard OPSD methods align the model toward its own outputs under a "
            "correct-hint context, but the learning signal concentrates on STYLE tokens "
            "(shorter, more direct responses) rather than task-bearing reasoning tokens. "
            "RLCSD fixes this by contrasting correct-hint vs wrong-hint distributional "
            "gaps — canceling the shared style drift and isolating the task-bearing signal. "
            ""
            "RELEVANCE TO ACR=52.4%: "
            "GRPO produces zero gradient on all-fail groups (43/82 groups, ACR=52.4%): "
            "all G=8 rollouts fail → group mean = 0 → all advantages = 0 → zero gradient. "
            "RLCSD's contrastive term provides dense token-level signal INDEPENDENT of "
            "rollout pass/fail: the model is compared to its own outputs conditioned on "
            "a correct hint (available from SCALING_FORCE_BOTH GPT-5.5 traces), "
            "and the wrong-hint baseline cancels style. Every token in every training "
            "step receives a gradient, eliminating the zero-signal regime. "
            ""
            "HOW WE ADAPT RLCSD TO CODE (not math): "
            "Correct hint = first 3 lines of the GPT-5.5 correct trace for task i "
            "  (from traces.jsonl, already collected via SCALING_FORCE_BOTH=1). "
            "Wrong hint = first 3 lines of the WORST failing rollout from the G=8 GRPO samples. "
            "Both hints induce style drift (e.g., indentation choices, comment style); "
            "the contrastive difference cancels these and concentrates gradient on "
            "algorithmic-bearing token positions (loop structure, data structure, boundaries). "
            ""
            "IMPLEMENTATION (~40 lines in grpo_train_simple.py): "
            "1. Load correct traces from traces.jsonl (already done in Phase 3b setup). "
            "2. For each task in the GRPO batch: "
            "   a. Sample G=8 rollouts (standard GRPO path). "
            "   b. Forward pass M_θ on (prompt + correct_hint): get log-probs → KL_correct. "
            "   c. Forward pass M_θ on (prompt + wrong_hint): get log-probs → KL_wrong. "
            "   d. Contrastive loss: L_contrast = KL_correct - KL_wrong. "
            "   e. Total loss: L = L_GRPO + λ * L_contrast (λ=0.05). "
            "3. 2 extra forward passes per step (hint-conditioned). "
            "   GPU cost: +~20% vs standard GRPO. Estimated 2.5h total (vs 2.0h baseline). "
            ""
            "DISTINCT FROM EXISTING EXPERIMENTS: "
            "EXP-112 (fwd-KL distillation): uses full GPT-5.5 trace as external teacher; "
            "  no contrastive cancellation; dense signal only on teacher-passing tasks. "
            "EXP-EGCA (exec-grounded credit): masks token gradients post-hoc in GRPO; "
            "  zero signal still for all-fail groups (execution never runs). "
            "EXP-108 (sign advantage, A=2r-1): sequence-level gradient on all-fail groups; "
            "  no token-level density; no style-drift cancellation. "
            "EXP-114 (this): token-level, style-drift-free signal via contrastive hints; "
            "  independent of rollout pass/fail. All four are composable. "
            ""
            "PREDICTED OUTCOME: "
            "  ACR reduction from 52.4% toward ≤ 40% (all-fail groups now receive gradient). "
            "  Full arm pass@1 > 92.68% (GRPO lift > 0, breaking Full=Router equivalence). "
            "  Skills arm unchanged (this modifies Phase 3b only). "
            "  If EXP-114 and EXP-108 (sign advantage) are both positive, they are composable "
            "  (different mechanisms: token-level contrastive vs sequence-level advantage). "
        ),
        "spec": {
            "pipeline": "humaneval_rlcsd_contrastive_opsd_grpo",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 1,
            "grpo_algo": "rlcsd_dapo",
            "rlcsd_lambda": 0.05,
            "rlcsd_correct_hint_source": "gpt55_traces_first3lines",
            "rlcsd_wrong_hint_source": "worst_failing_rollout_first3lines",
            "rlcsd_hint_len_tokens": 48,
            "n_generations": 8,
            "grpo_temperature": 1.0,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (DAPO): full 92.68%, ACR=52.4%",
                "EXP-108 (sign advantage, A=2r-1): sequence-level gradient on all-fail groups",
                "EXP-112 (fwd-KL distillation): external teacher distillation",
                "arxiv:2606.11709 (RLCSD on Qwen3 1.7B/4B/8B: outperforms GRPO + OPSD)",
            ],
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
            ],
            "implementation_note": (
                "In grpo_train_simple.py: after sampling G=8 rollouts (standard GRPO path), "
                "for each task: (a) load correct_hint = first 48 tokens of the GPT-5.5 trace "
                "from traces.jsonl; (b) forward-pass the model on (prompt + correct_hint) to "
                "get per-token log-probs; (c) forward-pass on (prompt + wrong_hint, where "
                "wrong_hint = first 48 tokens of the lowest-reward rollout); (d) compute "
                "L_contrast = mean[log_softmax(logits_correct) - log_softmax(logits_wrong)] "
                "at each position; (e) total loss = L_GRPO + 0.05 * L_contrast. "
                "2 extra forward passes per step; no extra parameters. ~40 lines. "
            ),
            "arxiv_ref": "2606.11709",
            "estimated_gpu_hours": 2.5,
        },
    },
    {
        "id": "exp_2026_07_13_002_forgetting_multicycle_gradient_sparsity_diagnostic",
        "priority": 6,
        "kind": "forgetting_eval",
        "gpu": "auto",
        "rationale": (
            "arxiv:2601.20861 — Evolutionary Strategies lead to Catastrophic Forgetting "
            "in LLMs (January 2026). "
            ""
            "KEY INSIGHT FROM THE PAPER: "
            "GRPO updates are highly sparse (concentrated in a small fraction of parameters) "
            "compared to ES updates (dense, large ℓ2 norm), explaining GRPO's lower "
            "catastrophic forgetting. But 'lower' is not 'zero': even sparse GRPO updates "
            "can cause within-method forgetting across cycles when each cycle's gradient "
            "concentrates on different task-subsets. "
            ""
            "OUR DIAGNOSTIC QUESTION: "
            "Is the Full=Router equivalence (92.68%) partly explained by catastrophic "
            "forgetting across GRPO cycles — not solely by ACR=52.4% (zero-variance groups)? "
            ""
            "Concretely: our pipeline runs 4 GRPO cycles. Each cycle's gradient concentrates "
            "on tasks where the policy changed (high advantage, non-zero reward variance). "
            "If cycle-2 GRPO updates on medium-hard tasks interfere with parameters encoding "
            "easy tasks solved in cycle-1, the model may lose pass@1 on easy tasks by cycle 3. "
            "If, say, 5% of the 82 tasks are forgotten by cycle 3, this accounts for 4.1pp "
            "of the Full=Router gap (92.68% could be 96%+ without forgetting). "
            ""
            "DISTINCTION FROM EXP-111 (RFT vs SFT cycle retention): "
            "EXP-111 compares methods: does RFT or SFT preserve capabilities better? "
            "It compares two different training methods at cycle boundaries. "
            "EXP-115 (this): within the DEFAULT pipeline (GRPO), tracks per-task pass "
            "rates across checkpoints, correlates forgetting with gradient update magnitude. "
            "EXP-111 is prescriptive (which method is better); EXP-115 is diagnostic "
            "(quantifies the forgetting already happening in our production setup). "
            ""
            "IMPLEMENTATION (eval-only, no training): "
            "1. Load the 4 cycle LLM checkpoints from cycle_{0,1,2,3}/llm_adapter/checkpoint-best "
            "   (or grpo_adapter/ where available). "
            "2. For each checkpoint, run vLLM pass@1 eval on all 82 HumanEval tasks. "
            "   Result: a (4 cycles × 82 tasks) pass/fail matrix. "
            "3. For each task i, compute: "
            "   first_solved_cycle = min{k : pass[k][i] = 1} (or None if never solved) "
            "   forgetting_flag[i] = 1 if first_solved_cycle < 3 and pass[3][i] = 0 "
            "   forgetting_score = count(forgetting_flag) / 82 "
            "4. Load GRPO training logs (grpo_train_simple.py saves per-group stats). "
            "   Extract per-task gradient norm proxy: reward_variance * advantage_magnitude "
            "   for each task in each cycle's GRPO run. "
            "5. Spearman correlation: forgetting_flag ↔ gradient_norm_later_cycles. "
            "   Prediction: high gradient norm in cycle 2-3 on task i → higher forgetting "
            "   probability for task i (per 2601.20861's sparsity ↔ forgetting link). "
            "6. Report: "
            "   (a) forgetting_rate overall and per difficulty bucket "
            "   (b) correlation coefficient (Spearman ρ) "
            "   (c) counterfactual: if forgotten tasks were restored, what would full-arm "
            "       pass@1 be? (upper bound on forgetting's contribution to Full=Router) "
            "   (d) recommendation: if forgetting explains ≥ 3pp, add EWC / replay buffer "
            "       to the AAAI ablation (new §5.6). "
            ""
            "EXPECTED OUTCOME: "
            "CASE A — Low forgetting (forgetting_rate < 2%): "
            "  ACR=52.4% (zero-variance groups) is the dominant explanation for Full=Router. "
            "  EXP-108 (sign advantage) is the right fix; no replay needed. "
            "  Forgetting is a non-finding: report briefly in §5.1 as a sanity check. "
            "CASE B — Moderate forgetting (forgetting_rate 2–8%): "
            "  Forgetting contributes 1–6pp to Full=Router gap. "
            "  Sign advantage alone insufficient; EWC or replay needed in addition. "
            "  New experiment: EWC-regularized GRPO (natural EXP-116). "
            "CASE C — High forgetting (forgetting_rate > 8%): "
            "  Forgetting is a major factor; this is novel and publishable on its own. "
            "  The claim 'GRPO in multi-cycle on-policy distillation causes significant "
            "  forgetting' is a new finding not in the literature (2601.20861 covers "
            "  within-task forgetting, not cross-cycle forgetting in iterative distillation). "
            ""
            "RESOURCE COST: "
            "  Eval-only: 4 checkpoints × 82 tasks × pass@1 (greedy) = ~0.5h on A800. "
            "  Log parsing: ~0.1h. "
            "  Total: ~0.6h. Lowest-cost diagnostic available. "
            "  Priority 6 (not 7): results depend on A800 availability; "
            "  run after EXP-108, EXP-099, EXP-100 (higher AAAI priority). "
        ),
        "spec": {
            "pipeline": "humaneval_forgetting_multicycle_diagnostic",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "eval_checkpoints": [
                f"{_CYCLE_DIR}/cycle_0/grpo_adapter",
                f"{_CYCLE_DIR}/cycle_0/llm_adapter/checkpoint-best",
                f"{_CYCLE_DIR}/cycle_1/grpo_adapter",
                f"{_CYCLE_DIR}/cycle_1/llm_adapter/checkpoint-best",
                f"{_CYCLE_DIR}/cycle_2/grpo_adapter",
                f"{_CYCLE_DIR}/cycle_2/llm_adapter/checkpoint-best",
                f"{_CYCLE_DIR}/cycle_3/grpo_adapter",
                f"{_CYCLE_DIR}/cycle_3/llm_adapter/checkpoint-best",
            ],
            "eval_checkpoint_priority": "grpo_adapter > llm_adapter/checkpoint-best",
            "eval_data": _HE_EVAL,
            "eval_mode": "pass_at_1_greedy",
            "analysis": [
                "per_task_pass_fail_matrix",
                "forgetting_flag_per_task",
                "forgetting_rate_overall",
                "forgetting_rate_by_difficulty_bucket",
                "spearman_correlation_forgetting_vs_gradient_norm",
                "counterfactual_full_arm_pass_rate_if_no_forgetting",
            ],
            "gradient_norm_source": "grpo_train_simple_training_logs",
            "compare_with": [
                "EXP-111 (RFT vs SFT method comparison): prescriptive method comparison",
                "arxiv:2601.20861 (ES vs GRPO forgetting: sparsity is protective)",
                "e2e_4cyc_gpt55 cycle_3 full arm: 92.68% (full=router; unexplained gap)",
            ],
            "implementation_files": [
                "src/pipeline/forgetting_eval.py",
                "src/pipeline/run_e2e_ablation_simple.py",
            ],
            "implementation_note": (
                "forgetting_eval kind: for each checkpoint in eval_checkpoints, run "
                "vLLM-served pass@1 on all HumanEval tasks (greedy decoding). "
                "Load per-task gradient norms from grpo_train_simple.py's reward_var logs. "
                "Output: forgetting_matrix.json (cycles × tasks), forgetting_stats.json "
                "(overall rate, correlation). ~0.6h total. No training needed. "
            ),
            "arxiv_ref": "2601.20861",
            "estimated_gpu_hours": 0.6,
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
