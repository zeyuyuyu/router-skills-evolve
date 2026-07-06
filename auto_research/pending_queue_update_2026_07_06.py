#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-06 (EXP-110, EXP-111).

A800 connectivity: offline since 2026-05-14 (day ~53). Apply when restored.

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
    python3 auto_research/pending_queue_update_2026_07_06.py            # EXP-110, EXP-111

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
    Root problem 1: ACR=52.4% — 43/82 GRPO groups have zero within-group reward
    variance → zero gradient → Full = Router.
    Root problem 2: Skills gap — 75.61% vs large 96.34% (20.7pp). The global
    single-cluster procedure is insufficient; distillation misses decision-level
    and evidence-level token structure.

arxiv:2606.22830 — DEAR: Decision-Evidence Aware Reasoning Distillation
    (Meituan Longcat / Nanjing Univ. / TJUNLP, June 26, 2026):
    Standard on-policy distillation applies uniform per-token KL supervision.
    DEAR identifies two privileged token classes:
      Decision tokens: positions of high student entropy (the model is uncertain
        about which branch to take — e.g., choosing `if` vs. `while`, `return`
        vs. `continue`). These are the forks that determine code structure.
      Evidence tokens: positions near decision anchors where teacher-student
        divergence is highest and hidden-state cosine similarity to the nearest
        decision anchor is high. These are the supporting steps that make the
        decision right.
    DEAR applies a weighted KL loss: weight=λ_decision at decision tokens,
    weight=λ_evidence at evidence tokens, weight=0 at routine tokens.
    Empirical results (Qwen2.5-1.5B ← Qwen2.5-14B):
      HumanEval: +2.13pp pass@1 vs standard OPD
      MBPP+:     +5.74pp pass@1 vs standard OPD
      APPS:      +2.56pp pass@1 vs standard OPD
    Relevance: our SFT (Phase 3a) uses standard token-uniform KL. DEAR identifies
    exactly the token class our skills arm struggles with: decision tokens in code
    correspond to control-flow choice points. When the student model is uncertain
    at a control-flow decision (e.g., loop vs. recursion), the teacher's guidance
    at those positions is most valuable — DEAR ensures those positions receive the
    highest supervision weight. Implementation in train_small_model.py: after
    standard forward pass, compute per-token student entropy H_t = -sum(p*log(p));
    identify decision positions where H_t > threshold (e.g., top-20% entropy per
    sequence); for evidence, compute hidden-state cosine similarity to decision
    anchors and teacher-student KL; apply 3x weight at decision+evidence positions.
    ~40 lines of additional code in train_small_model.py or traces_to_sft.py.
    Orthogonal to MOPD (EXP-109, peer context) and DemoPSD (EXP-107, trace
    selection): those change what traces are used; DEAR changes how each trace
    is supervised at the token level.

arxiv:2604.23577 — RouteNLP: Closed-Loop LLM Routing with Conformal Cascading
    and Distillation Co-Optimization (HKU/Stellaris AI, April 26, 2026):
    Standard LLM routing treats the model portfolio as fixed: train a router,
    deploy, never retrain. RouteNLP breaks this assumption by closing the loop:
    (1) Route queries through small → large cascade.
    (2) Log all escalations (queries where small model failed → router escalated).
    (3) Cluster escalation logs by query embedding to find failure modes.
    (4) Apply targeted distillation to the small model on those failure clusters
        (using the large model's answers on those queries as teacher signals).
    (5) Recalibrate the router's confidence thresholds via conformal prediction.
    Repeat each cycle. Result: >2x cost reduction at equal quality vs. untargeted
    distillation across 4 routing benchmarks.
    KEY INSIGHT: The router's escalation decisions are a perfect oracle for which
    tasks the small model currently cannot handle. Using those exact tasks as
    distillation targets gives a focused, high-signal SFT dataset — rather than
    uniform distillation over the full task distribution.
    RELEVANCE TO OUR PROJECT: In cycle 3, our logistic router escalates 6.10%
    of tasks to the large model. Those ~5 tasks (out of 82) are the ones where
    the small model is most deficient. Currently our SFT (Phase 3a) uses uniform
    distillation from ALL tasks regardless of router signal. RouteNLP prescribes
    upweighting the distillation loss on the router-escalated tasks — these
    teacher traces are most valuable because they address the small model's exact
    failure modes. Implementation: in traces_to_sft.py, add per-sample weights
    derived from traces.jsonl router_decision field:
      if trace["routing"] == "large" and trace["small_reward"] == 0:
        weight = ESCALATION_WEIGHT   # suggested 5.0
      else:
        weight = 1.0
    Then pass sample_weights to the SFT DataLoader. ~15 lines in traces_to_sft.py.
    This is orthogonal to DEAR (token-level weighting) and MOPD (peer context).
    Potential combination: DEAR + RouteNLP weights = "focus distillation effort
    on the most important tasks AND the most important token positions within
    those tasks." This could be EXP-112 if both EXP-110 and EXP-111 succeed.
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
        "id": "exp_2026_07_06_001_dear_decision_evidence_token_selective_sft_humaneval",
        "priority": 7,
        "kind": "joint_cycle_multiseed",
        "rationale": (
            "arxiv:2606.22830 — DEAR: Decision-Evidence Aware Reasoning Distillation "
            "(Meituan Longcat Team / Nanjing University / TJUNLP, June 26, 2026). "
            ""
            "PROBLEM: UNIFORM TOKEN SUPERVISION MISSES CRITICAL CODE STRUCTURE "
            "Standard on-policy distillation in our Phase 3a SFT applies the same "
            "per-token KL weight to every position in the training sequence. "
            "However, code generation has two structurally distinct token classes: "
            "  (A) Decision tokens: control-flow choice points where the model selects "
            "      the next branch (if/while/for/return/recursion). Student entropy "
            "      H_t = -sum(p_t * log p_t) is high at these positions. These are "
            "      the forks that determine whether the code is structurally correct. "
            "  (B) Evidence tokens: positions that implement the decision just made. "
            "      Student entropy is low (model confidently fills in the syntax), "
            "      but teacher-student KL divergence remains high — the student has "
            "      learned the wrong implementation of the right branch. Hidden-state "
            "      cosine similarity to the nearest decision anchor identifies these. "
            "DEAR shows that uniform supervision wastes budget on routine tokens "
            "(spacing, closing brackets, routine API calls) and under-supervises the "
            "decision + evidence positions that actually determine task success. "
            ""
            "DEAR APPROACH: "
            "For each training token position t in sequence s: "
            "  1. Compute student entropy H_t (forward pass, no teacher needed). "
            "  2. Identify decision positions D_s = {t : H_t > top-20% of H in s}. "
            "  3. For each decision position d in D_s, compute: "
            "       evidence_score(t) = cos_sim(h_t^student, h_d^student) * KL(t) "
            "     where h_t^student is the student hidden state and KL(t) is the "
            "     teacher-student KL at position t. "
            "  4. Identify evidence positions E_s = top-20% by evidence_score. "
            "  5. Apply weighted KL loss: "
            "       loss(t) = lambda_decision * KL(t)  if t in D_s "
            "               = lambda_evidence * KL(t)  if t in E_s and t not in D_s "
            "               = 0                         otherwise "
            "     Suggested weights: lambda_decision=3.0, lambda_evidence=2.0. "
            "     Only D_s union E_s tokens contribute to the SFT loss. "
            ""
            "IMPLEMENTATION (~40 lines in train_small_model.py): "
            "1. After student forward pass, extract per-token logits and compute H_t. "
            "2. Compute decision mask: top-20% entropy positions per sequence. "
            "3. Extract student hidden states at decision positions for cosine sim. "
            "4. Compute evidence scores at all non-decision positions. "
            "5. Apply position weights to the per-token KL loss before reduction. "
            ""
            "EMPIRICAL RESULTS (arxiv:2606.22830, Qwen2.5-1.5B ← Qwen2.5-14B): "
            "  HumanEval:  +2.13pp pass@1 vs standard OPD (from 72.4% to 74.53%) "
            "  MBPP+:      +5.74pp pass@1 vs standard OPD (largest gain) "
            "  APPS:       +2.56pp pass@1 vs standard OPD "
            "The MBPP+ gain (+5.74pp) exceeds MOPD gains (2-4%) and is concentrated "
            "on hard tasks where decision token count per sequence is highest. "
            "Our target: skills arm pass@1 ≥ 77.74% (75.61% + 2.13pp). "
            ""
            "ORTHOGONALITY TO EXISTING EXPERIMENTS: "
            "EXP-107 DemoPSD: changes WHICH traces to distill (Category-A disagreement). "
            "EXP-109 MOPD:    changes HOW each trace is conditioned (peer context). "
            "EXP-110 DEAR:    changes WHICH TOKEN POSITIONS within each trace to "
            "                  supervise (decision + evidence). "
            "All three address the skills gap via orthogonal mechanisms; a three-way "
            "combination (DemoPSD+MOPD+DEAR) would address trace selection, context, "
            "and token-level focus simultaneously. "
            ""
            "AAAI 2027 NARRATIVE VALUE: "
            "DEAR positions our SFT work as 'structure-aware distillation': rather "
            "than imitating every token equally, we identify and supervise the "
            "semantically critical positions. This aligns with the paper's theme of "
            "skills distilling PROCEDURE, not just outputs — decision tokens are where "
            "the procedure is instantiated. A result of skills arm ≥ 77.7% would "
            "reduce the skills-large gap from 20.7pp to 18.6pp, strengthening the "
            "argument that distillation quality matters. "
        ),
        "spec": {
            "pipeline": "humaneval_dear_token_selective_sft",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 1,
            "sft_mode": "dear_token_selective",
            "dear_decision_threshold_percentile": 80,
            "dear_evidence_threshold_percentile": 80,
            "dear_lambda_decision": 3.0,
            "dear_lambda_evidence": 2.0,
            "dear_cosine_sim_window": 32,
            "dear_variants": [
                "dear_decision_evidence",
                "dear_decision_only",
                "standard_opd_baseline",
            ],
            "sft_include_success": True,
            "scaling_force_both": True,
            "eval_arm": "skills",
            "eval_data": _HE_EVAL,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 skills arm: 75.61% HE pass@1",
                "arxiv:2606.22830 (DEAR: +2.13pp HumanEval, +5.74pp MBPP vs standard OPD)",
                "EXP-107 (DemoPSD: orthogonal WHAT-to-distill filter)",
                "EXP-109 (MOPD: orthogonal peer-context conditioning)",
            ],
            "implementation_note": (
                "In train_small_model.py, override the per-token KL loss computation: "
                "1. After forward pass, compute H_t = -(student_logits.softmax(-1) * "
                "   student_logits.log_softmax(-1)).sum(-1) for each token position. "
                "2. Identify decision positions: top-20% H_t per sequence → decision_mask. "
                "3. Extract student hidden states at decision positions. "
                "4. For each non-decision position, compute "
                "   evidence_score = cos_sim(h_t, nearest_decision_h) * KL_t. "
                "5. Identify evidence positions: top-20% evidence_score → evidence_mask. "
                "6. Loss: (3.0 * KL * decision_mask + 2.0 * KL * evidence_mask).sum() "
                "   / (decision_mask.sum() + evidence_mask.sum() + 1e-8). "
                "~40 lines in train_small_model.py. No change to architecture or "
                "hyperparameters. Teacher model stays Qwen2.5-Coder-14B-Instruct "
                "(identical to baseline SFT). "
                "Variant B (decision-only) uses only decision_mask to isolate "
                "the contribution of evidence supervision. "
            ),
            "primary_metrics": [
                "humaneval_pass_at_1_skills_arm_post_sft",
                "sft_decision_token_fraction_mean",
                "sft_evidence_token_fraction_mean",
                "sft_supervised_token_fraction_mean",
                "skills_arm_hard_task_pass_rate",
                "skills_arm_decision_token_density_correlation",
                "variant_delta_pass_at_1",
            ],
            "target_outcome": (
                "Best variant (dear_decision_evidence) achieves skills arm pass@1 "
                ">= 77.74% (75.61% + 2.13pp, matching DEAR paper's HumanEval gain). "
                "Decision-only variant (B) < Decision+Evidence variant (A) — confirms "
                "that evidence supervision adds value beyond decision supervision alone. "
                "Hard-task improvement > easy-task improvement (decision token density "
                "is higher in complex multi-branch code tasks). "
                "Supervised token fraction: 30-40% of tokens per sequence (top-20% "
                "decision + top-20% evidence, with overlap ~10%). "
            ),
            "gpu": "auto",
            "estimated_gpu_hours": 1.5,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_07_06_002_routenlp_escalation_targeted_sft_humaneval",
        "priority": 7,
        "kind": "grpo_curriculum_continual",
        "rationale": (
            "arxiv:2604.23577 — RouteNLP: Closed-Loop LLM Routing with Conformal "
            "Cascading and Distillation Co-Optimization "
            "(Dongxin Guo, Jikun Wu, Siu Ming Yiu — HKU / Stellaris AI, "
            "April 26, 2026). "
            ""
            "PROBLEM: UNIFORM DISTILLATION IGNORES THE ROUTER'S ORACLE SIGNAL "
            "Our Phase 3a SFT uses the same per-sample distillation weight for all "
            "tasks regardless of whether the router escalated them to the large model "
            "in the previous cycle. But the router's escalation decision is a perfect "
            "oracle: if the logistic router sent task T to the large model in cycle 3, "
            "it means the router's classifier estimated P(small fails | T) is high. "
            "This implies task T is at the current frontier of the small model's "
            "capability — precisely the most valuable teacher trace target for SFT. "
            "Tasks the router routes to small (confident the small model will succeed) "
            "are already well-handled — distilling them more is redundant. "
            ""
            "ROUTENLP CLOSED-LOOP APPROACH: "
            "RouteNLP formalizes this as a distillation-routing co-optimization loop: "
            "  Cycle k: collect traces (router escalates ~6.10% to large). "
            "  Distillation targeting: cluster escalation failures by embedding; "
            "    apply targeted SFT on those clusters (large-model traces for "
            "    tasks that were escalated). "
            "  Router recalibration: refit conformal thresholds using new small model. "
            "Result: >2x cost reduction at equal quality vs. untargeted distillation. "
            ""
            "OUR EXPERIMENT (simplified RouteNLP-style targeted SFT): "
            "We already have the router's cycle-3 escalation decisions in traces.jsonl "
            "as the 'routing' field. The targeted SFT experiment: "
            "  1. Load traces.jsonl from cycle 3. "
            "  2. For each trace, assign sample weight: "
            "       weight = ESCALATION_WEIGHT (suggested 5.0) "
            "         if trace['routing'] == 'large' AND trace['small_reward'] == 0 "
            "         (router escalated AND small model actually failed — these are "
            "          the hardest tasks where teacher traces are most valuable) "
            "       weight = 1.0 otherwise "
            "  3. Pass weights to SFT DataLoader (WeightedRandomSampler). "
            "  4. Run standard SFT training on all traces with these sample weights. "
            "  5. Evaluate skills arm pass@1 with focus on the escalated task subset. "
            ""
            "THREE VARIANTS to isolate the mechanism: "
            "  (A) Escalation-targeted (weight=5.0 for escalated+failed tasks). "
            "  (B) Hard-task upweighting (weight=5.0 for all tasks where "
            "      small_reward=0, regardless of routing — 'oracle difficulty'). "
            "  (C) Standard uniform SFT (weight=1.0 for all tasks, baseline). "
            "Comparing (A) vs (B): is the router's escalation decision more "
            "informative than pure reward-based difficulty? "
            "Comparing (A) vs (C): does targeted weighting improve the skills arm? "
            ""
            "RELATIONSHIP TO EXISTING EXPERIMENTS: "
            "EXP-107 DemoPSD: restricts SFT to disagreement traces (task categories). "
            "EXP-109 MOPD:    adds peer context to each training example. "
            "EXP-110 DEAR:    weights loss by decision/evidence token positions. "
            "EXP-111 (this):  weights loss by router escalation decision per SAMPLE. "
            "All four operate on different axes: trace selection (DemoPSD), "
            "input context (MOPD), token position (DEAR), sample weight (RouteNLP). "
            "They are fully composable: the ultimate 'skills arm v2' experiment "
            "would combine all four — EXP-112 or EXP-113 after A800 returns. "
            ""
            "ESTIMATED TASK COUNT FOR ESCALATED+FAILED TASKS: "
            "Cycle-3 router: fallback=6.10% → ~5 escalated tasks (out of 82 HumanEval). "
            "Of those ~5, all likely have small_reward=0 (why the router escalated). "
            "With weight=5.0 on 5 tasks and weight=1.0 on 77 tasks: "
            "  effective dataset size = 5*5 + 77 = 102 examples (vs 82 baseline). "
            "This is a mild 25% increase in effective training examples, focusing on "
            "the exact tasks where the small model currently fails and the router "
            "already knows it. "
            ""
            "AAAI 2027 NARRATIVE VALUE: "
            "If variant (A) outperforms variant (C): this directly supports the paper's "
            "routing-skills co-evolution narrative — the router's signal is useful not "
            "just for inference routing but for improving the small model via targeted "
            "distillation. This closes the distillation loop: router trains ON the "
            "small model's failures, AND its decisions GUIDE the small model's next "
            "SFT cycle. Clean co-evolution story for §3 and §4 of the AAAI paper. "
        ),
        "spec": {
            "pipeline": "humaneval_routenlp_escalation_targeted_sft",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 1,
            "sft_mode": "escalation_targeted",
            "escalation_weight": 5.0,
            "escalation_target_condition": "routing=='large' AND small_reward==0",
            "sft_variants": [
                "escalation_targeted_weight5",
                "oracle_hardtask_weight5",
                "uniform_baseline",
            ],
            "sft_include_success": True,
            "scaling_force_both": True,
            "eval_arm": "skills",
            "eval_sub_eval": "escalated_tasks_only",
            "eval_data": _HE_EVAL,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 skills arm: 75.61% HE pass@1",
                "e2e_4cyc_gpt55 cycle_3 router: fallback=6.10% (~5 tasks escalated)",
                "arxiv:2604.23577 (RouteNLP: >2x cost reduction vs untargeted distillation)",
                "EXP-107 (DemoPSD: orthogonal trace-selection filter)",
                "EXP-110 (DEAR: orthogonal token-level weighting)",
            ],
            "implementation_note": (
                "In traces_to_sft.py, after loading traces.jsonl: "
                "  escalated_failed = { "
                "    t['task_id'] for t in traces "
                "    if t.get('routing') == 'large' and t.get('small_reward', 1) == 0 "
                "  } "
                "  sample_weights = [ "
                "    ESCALATION_WEIGHT if t['task_id'] in escalated_failed else 1.0 "
                "    for t in sft_dataset "
                "  ] "
                "Pass sample_weights to torch.utils.data.WeightedRandomSampler. "
                "~15 lines in traces_to_sft.py. "
                "Variant B uses small_reward==0 condition only (no routing condition). "
                "Variant C uses all weights=1.0 (standard baseline, should match "
                "existing cycle_3 skills arm result of 75.61% as a sanity check). "
                "Sub-evaluation: compute pass@1 separately on the ~5 escalated tasks "
                "to directly measure improvement on the targeted tasks. "
            ),
            "primary_metrics": [
                "humaneval_pass_at_1_skills_arm_post_sft",
                "humaneval_pass_at_1_escalated_tasks_only",
                "humaneval_pass_at_1_non_escalated_tasks",
                "sft_escalated_sample_fraction_effective",
                "router_escalation_rate_post_cycle",
                "variant_delta_pass_at_1",
            ],
            "target_outcome": (
                "Variant (A) achieves overall skills arm pass@1 >= 76.5% and "
                "escalated-task pass@1 >= 40% (these tasks are currently ~0% for "
                "small model, so any improvement is significant). "
                "Variant (A) outperforms Variant (B) on escalated tasks: router's "
                "judgment adds signal beyond pure reward-based difficulty. "
                "Non-escalated task performance maintained or improved (no regression). "
                "If both DEAR (EXP-110) and this experiment succeed, a combined "
                "EXP-112 applying both DEAR token weights + RouteNLP sample weights "
                "would be the skills arm v2 experiment. "
            ),
            "gpu": "auto",
            "estimated_gpu_hours": 1.5,
        },
        "gpu": "auto",
    },
]


def main() -> None:
    if not STATE_PATH.exists():
        print(f"ERROR: {STATE_PATH} not found. Is the A800 mounted?")
        raise SystemExit(1)

    with STATE_PATH.open() as f:
        state = json.load(f)

    existing_ids = {e["id"] for e in state.get("queue", [])}
    existing_ids |= {e["id"] for e in state.get("history", [])}

    added = []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] in existing_ids:
            print(f"  SKIP (duplicate): {exp['id']}")
            continue
        state["queue"].append(exp)
        added.append(exp["id"])
        print(f"  QUEUED: {exp['id']}  (priority={exp['priority']}, "
              f"kind={exp['kind']})")

    if not added:
        print("No new experiments added (all duplicates).")
        return

    tmp = STATE_PATH.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(STATE_PATH)
    print(f"\nSaved {STATE_PATH}. Added {len(added)} experiment(s): {added}")


if __name__ == "__main__":
    main()
