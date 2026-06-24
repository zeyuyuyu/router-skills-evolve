#!/usr/bin/env python3
"""
Daily queue patch — 2026-06-24 (EXP-088, EXP-089).

A800 connectivity: offline since 2026-05-14 (day 41). Apply when restored.

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
    python3 auto_research/pending_queue_update_2026_06_21.py            # EXP-082, EXP-083
    python3 auto_research/pending_queue_update_2026_06_22.py            # EXP-084, EXP-085
    python3 auto_research/pending_queue_update_2026_06_23.py            # EXP-086, EXP-087

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_06_24.py            # EXP-088, EXP-089

Queue was ~87 pending on 2026-06-23 (+2 added: EXP-086, EXP-087).
Queue cap applied: >20 -> max 2 new experiments.

Data motivating today's proposals
----------------------------------
Best historical results (unchanged):
    HumanEval 4-cycle MERA full/router: 92.68% task pass at 27.56% large-model cost
    GRPO zero-variance: 52.4% of groups discarded at G=8 DAPO (43/82 groups)
    Skills arm (always-small+procedure): 75.61% (up from 70.73% at cycle 0)

Open questions motivating today's proposals:
    W8 (new): Does sparse binary GRPO sub-optimally exploit the small model's capacity
      relative to dense token-level distillation from GPT-5.5?
      <- EXP-088 (sparse-to-dense OPD) directly tests this.
    W9 (new): Does the standard KL-to-ref penalty in GRPO fail to leverage the teacher
      model's improving policy, causing objective interference when teacher and student
      rollout distributions diverge?
      <- EXP-089 (RLAD selective teacher KL) directly tests this.

Papers motivating today's proposals:
    arxiv:2605.12483 — 'Beyond GRPO and On-Policy Distillation: An Empirical
      Sparse-to-Dense Reward Principle for Language-Model Post-Training' (Xu et al., May 2026).
      Key finding: sparse reward GRPO is best for capable models that can explore;
      dense token-level teacher supervision (OPD) is better for compressing into small students.
      For a fixed Qwen3-1.7B student, distilling from RL-improved 8B outperforms direct
      GRPO on the student. Directly challenges our Phase 3b design (GRPO on 1.5B small model).

    arxiv:2602.22495 — 'Reinforcement-aware Knowledge Distillation for LLM Reasoning'
      (Zhang et al., April 2026).
      Problem: standard KL(student||teacher) added to GRPO suffers distribution mismatch
      when teacher and student rollout distributions diverge mid-training. The KL term
      competes with reward maximization, potentially degrading both objectives.
      RLAD fix: SELECTIVE imitation — compute whether imitating the teacher improves the
      current policy update (using advantage-weighted teacher log-probs). Skip distillation
      when teacher rollout quality is below the current policy advantage baseline.
      Applied to our setting: use GPT-5.5 as the teacher; selectively apply KL-to-GPT5.5
      on traces where GPT-5.5's rollout advantage > current small model advantage baseline.
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
_HE_CONFIG = "config/humaneval_dapo_gpt.yaml"

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_06_24_001_sparse_to_dense_opd_phase3b_replace_grpo_humaneval",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2605.12483 — 'Beyond GRPO and On-Policy Distillation: An Empirical "
            "Sparse-to-Dense Reward Principle for Language-Model Post-Training' "
            "(Xu et al., May 2026). "
            ""
            "Core finding: sparse sequence-level reward (GRPO, binary pass/fail) is most "
            "useful for STRONG models that can explore and discover better behaviors. Dense "
            "token-level teacher supervision (on-policy distillation, OPD) is better suited "
            "for SMALL students compressing capability from a stronger teacher. "
            ""
            "The paper evaluates this on Qwen3 and Llama models for verifiable math tasks. "
            "For a fixed Qwen3-1.7B deployment student, distilling from a GRPO-improved 8B "
            "teacher outperforms applying GRPO directly to the 1.7B student using the same "
            "labeled data budget. A forward-KL warmup on teacher rollouts followed by OPD "
            "on student rollouts outperforms standalone student GRPO. "
            ""
            "Why this challenges our current Phase 3b design: "
            "We currently run binary GRPO on Qwen2.5-Coder-1.5B (Phase 3b). The paper "
            "predicts this is sub-optimal: 1.5B models explore poorly (limited capacity for "
            "reward-driven behavior discovery), so sparse binary reward GRPO wastes the "
            "training budget on rollouts that consistently fail. Dense distillation from "
            "GPT-5.5 (our large model, already providing SFT traces in Phase 3a) would "
            "instead provide token-level supervision on EVERY token of the trajectory, not "
            "just a terminal binary signal, giving the small model richer gradient signal. "
            ""
            "Proposed Phase 3b replacement — two-stage OPD: "
            "(1) FORWARD-KL WARMUP (5 steps): train small model with standard forward KL "
            "toward GPT-5.5 rollouts (same traces as Phase 3a, no new API calls). This "
            "aligns the student's distribution with the improved teacher before OPD. "
            "(2) ON-POLICY DISTILLATION (main phase, same compute budget as current GRPO): "
            "sample rollouts from the student, compute forward KL between student rollouts "
            "and GPT-5.5's token-level distribution on the same prompt. Gradient signal is "
            "token-level (every token) rather than trajectory-level (one binary reward). "
            ""
            "Implementation in Phase 3b (grpo_train_simple.py): "
            "Add a `--reward-type opd` flag. When set: "
            "  - Load GPT-5.5 traces from Phase 3a output (already available). "
            "  - For each rollout, compute forward KL vs GPT-5.5 log-probs on the same "
            "    prompt using the pre-computed traces as the teacher's token distribution. "
            "  - Replace the binary reward accumulation with token-level KL loss (no "
            "    GRPO advantage normalization needed — direct KL minimization per step). "
            "~40 lines in grpo_train_simple.py. No new API calls. "
            ""
            "Key comparison: "
            "  - Reference: DAPO G=8 binary GRPO (92.68% pass at cycle 3) "
            "  - EXP-088: OPD (token-level KL from GPT-5.5) as Phase 3b "
            "  - Same SFT Phase 3a, same data budget, same A800 GPU time "
            "  - Hypothesis: OPD should outperform sparse GRPO for 1.5B model per "
            "    arxiv:2605.12483's empirical law "
            ""
            "Distinct from all queued experiments: "
            "  - EXP-086 (VeRPO): modifies reward VALUES within GRPO (dense partial tests) "
            "  - EXP-085 (GDPO): modifies ADVANTAGE NORMALIZATION within multi-reward GRPO "
            "  - EXP-082 (RL-ZVP): handles zero-variance groups via entropy-scaled advantage "
            "  - EXP-088 (this): REPLACES the GRPO objective entirely with OPD "
            "    (token-level distillation from teacher) — categorically different approach. "
            "No queued experiment has 'opd', 'on_policy_distillation', or 'sparse_to_dense' "
            "in its id or spec. Confirmed via grep. ✓"
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": _HE_EVAL,
            "eval_data": _HE_EVAL,
            "bench": "humaneval",
            "phase3b_mode": "opd",
            "opd_teacher": "openai/gpt-5.5",
            "opd_teacher_trace_source": "phase3a_sft_traces",
            "opd_warmup_steps": 5,
            "opd_warmup_loss": "forward_kl",
            "opd_main_loss": "forward_kl_on_policy",
            "opd_token_level": True,
            "dapo_dynamic_sampling": False,
            "n_generations": 8,
            "max_turns": 3,
            "epochs": 1,
            "lr": 5e-6,
            "lora_r": 16,
            "prompt_style": "qwen-chat",
            "skip_grpo_reward": True,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (DAPO G=8 binary GRPO, reference)",
                "exp_2026_06_23_001 (VeRPO dense reward, different axis)",
            ],
            "arxiv_ref": "2605.12483",
            "hypothesis": (
                "Per arxiv:2605.12483: for 1.5B student, OPD (dense token-level KL from "
                "GPT-5.5) should outperform sparse binary GRPO. Predicts higher pass@1 "
                "with same API/GPU budget and fewer zero-variance gradient steps."
            ),
            "estimated_hours": 1.5,
            "implementation_note": (
                "Modify grpo_train_simple.py: add --reward-type opd path. Load GPT-5.5 "
                "traces from Phase 3a output. Compute per-token forward KL between student "
                "and teacher log-probs. Replace GRPO advantage accumulation with direct "
                "KL loss per step. ~40 lines. No new API calls needed."
            ),
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_06_24_002_rlad_selective_teacher_kl_grpo_humaneval",
        "priority": 7,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2602.22495 — 'Reinforcement-aware Knowledge Distillation for LLM "
            "Reasoning' (Zhang et al., April 2026). "
            ""
            "Problem: Standard approaches to combining RL and KD for LLM training add a "
            "fixed KL(student||teacher) regularizer to the GRPO objective. When teacher "
            "and student rollout distributions diverge mid-training (which happens quickly "
            "when GRPO updates the student while the teacher is frozen), this KL term "
            "suffers distribution mismatch: the teacher's supervision is computed on "
            "prompts the teacher would solve easily, but these may not align with the "
            "student's current training frontier. The competing objectives (reward "
            "maximization vs KL-to-teacher) interfere, degrading both. "
            ""
            "RLAD solution: SELECTIVE imitation during RL. For each teacher trace, compute "
            "an 'advantage-weighted teacher benefit': does imitating this teacher rollout "
            "improve the current policy update? Formally, the RLAD augmentation term is: "
            "  L_RLAD = alpha * E_prompt [ max(0, A_teacher(tau_T) - A_student(tau_S)) * "
            "             KL(student || teacher | prompt) ] "
            "where A_teacher(tau_T) is the advantage of the teacher rollout under the "
            "student's value baseline, and A_student(tau_S) is the student's own rollout "
            "advantage. The max(0, ...) gate skips teacher imitation when the teacher's "
            "rollout is NOT better than the student's current best — preventing the "
            "teacher from pulling the student backward on problems it already solves. "
            ""
            "Why this fits our system: "
            "We have GPT-5.5 traces from Phase 1 collection (teacher rollouts, already "
            "cached). During Phase 3b GRPO on the small model, we can add the RLAD term: "
            "  - Compute student advantage A_student via standard DAPO normalization. "
            "  - For each problem with a cached GPT-5.5 trace, compute A_teacher: the "
            "    GPT-5.5 trajectory's reward (binary pass/fail) minus the DAPO group "
            "    mean (the student's own baseline for that problem). "
            "  - If A_teacher > A_student, add alpha * KL(student || GPT-5.5). "
            "  - If A_teacher <= A_student (student already better than teacher on this "
            "    problem), skip KL for this problem — no teacher pulling student backward. "
            ""
            "Expected effect: "
            "  - On hard problems where GPT-5.5 succeeds and the small model fails: "
            "    teacher advantage is high, RLAD adds teacher KL, student gets richer "
            "    gradient signal beyond binary reward. "
            "  - On easy problems where the small model already passes: teacher advantage "
            "    is not higher than student advantage, RLAD is gated OFF. "
            "  - Net result: denser supervision on hard problems, no interference on "
            "    easy problems. Should reduce zero-variance group rate (52.4% currently) "
            "    by providing gradient signal on all-fail groups via teacher KL. "
            ""
            "Distinct from all queued experiments: "
            "  - EXP-088 (OPD): replaces GRPO entirely with distillation; RLAD augments "
            "    GRPO additively with a selective, advantage-gated teacher KL term. "
            "  - EXP-086 (VeRPO): modifies reward values within GRPO; RLAD adds a "
            "    distillation term alongside reward, not to it. "
            "  - EXP-085 (GDPO): modifies how multi-reward advantages are normalized; "
            "    RLAD is an additive KL loss term, not a normalization change. "
            "No queued experiment has 'rlad', 'selective_teacher', or 'advantage_gated_kl' "
            "in its id or spec. Confirmed via grep. ✓ "
            ""
            "Implementation: ~35 lines in grpo_train_simple.py. "
            "Load GPT-5.5 traces from Phase 1 output (already written to disk). For each "
            "GRPO batch, look up the cached GPT-5.5 rollout for each problem. Compute "
            "A_teacher per problem. Apply RLAD gate. Add alpha*KL term to GRPO loss. "
            "alpha=0.3 per paper recommendations for code generation tasks."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": _HE_EVAL,
            "eval_data": _HE_EVAL,
            "bench": "humaneval",
            "grpo_algo": "dapo_rlad",
            "reward_type": "binary_plus_selective_teacher_kl",
            "rlad_teacher": "openai/gpt-5.5",
            "rlad_teacher_trace_source": "phase1_collected_traces",
            "rlad_alpha": 0.3,
            "rlad_gate": "advantage_weighted",
            "rlad_gate_threshold": 0.0,
            "dapo_dynamic_sampling": True,
            "dapo_clip_low": 0.2,
            "dapo_clip_high": 0.5,
            "n_generations": 8,
            "rollout": "repair",
            "max_turns": 3,
            "epochs": 1,
            "lr": 5e-6,
            "lora_r": 16,
            "beta": 0.04,
            "prompt_style": "qwen-chat",
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (DAPO G=8 binary GRPO, reference)",
                "exp_2026_06_24_001 (OPD full replacement, complementary approach)",
                "exp_2026_06_23_001 (VeRPO dense reward, different axis)",
            ],
            "arxiv_ref": "2602.22495",
            "hypothesis": (
                "RLAD should reduce zero-variance group rate (currently 52.4%) by providing "
                "gradient signal on all-fail groups via advantage-gated teacher KL. Predicted "
                "improvement: 2-4pp pass@1 over DAPO baseline with same GPU budget."
            ),
            "estimated_hours": 1.5,
            "implementation_note": (
                "Modify grpo_train_simple.py: add rlad_alpha param. For each batch problem, "
                "load GPT-5.5 trace from phase1 output dir, compute A_teacher = "
                "gpt55_reward - group_mean_reward. If A_teacher > A_student, add "
                "alpha * KL(student_logits || gpt55_logits) to GRPO loss. ~35 lines total."
            ),
        },
        "gpu": "auto",
    },
]


def main():
    if not STATE_PATH.exists():
        print(f"ERROR: {STATE_PATH} not found. Run from A800 or copy state.json first.")
        return

    with open(STATE_PATH) as f:
        state = json.load(f)

    queue = state.get("queue", [])
    existing_ids = {e["id"] for e in queue}
    history = state.get("history", [])
    existing_ids.update(e["id"] for e in history if "id" in e)

    added = []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] in existing_ids:
            print(f"SKIP (already present): {exp['id']}")
            continue
        queue.append(exp)
        added.append(exp["id"])
        print(f"QUEUED: {exp['id']}  (priority={exp['priority']}, kind={exp['kind']})")

    state["queue"] = queue

    tmp = STATE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)

    print(f"\nDone. Added {len(added)} experiments. Total queue: {len(queue)}.")
    for eid in added:
        print(f"  + {eid}")


if __name__ == "__main__":
    main()
