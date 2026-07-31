#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-31 (EXP-140, EXP-141).

A800 connectivity: offline since 2026-05-14 (day ~78). Apply when restored.

Prior patches to apply before this one (daily-patch chain):
    python3 auto_research/pending_queue_update.py
    ... (see pending_queue_update_2026_07_30.py header for full chain) ...
    python3 auto_research/pending_queue_update_2026_07_27.py            # EXP-134, EXP-135
    python3 auto_research/pending_queue_update_2026_07_29.py            # EXP-136, EXP-137
    python3 auto_research/pending_queue_update_2026_07_30.py            # EXP-138, EXP-139

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_31.py            # EXP-140, EXP-141

Queue was ~145 pending on 2026-07-30 (after EXP-138, EXP-139). Cap applied: >20 -> max 2.

AAAI 2027 deadline: 2026-08-15 (15 days from today 2026-07-31).
A800 offline since 2026-05-14 (day 78).
GPU window CLOSES 2026-08-01 (1 day) — ABSOLUTE LAST DAY for new GPU results to reach paper.

=================================================================================
NEW PAPERS MOTIVATING TODAY'S EXPERIMENTS
=================================================================================

arxiv:2607.12640 — "A Learning-Rate-Gated Failure of GRPO in a Small Language and
    Vision-Language Model Web Agent: A Controlled Null and Its Mechanism" (July 2026)

    Across 18 runs varying LR, KL weight, seed, init, and clipping at 4B-8B scale,
    no configuration credibly improved success rate on tasks the agent had largely
    mastered. Moderate-to-high LRs made performance credibly worse (attention + MLP
    parameter drift). GRPO only helps when the sampled policy succeeds MORE often than
    greedy — i.e., when there is genuine "headroom". The failure mode partitions into
    two regimes: a "degrade" regime at moderate LR (localised to attention + MLP) and
    a "collapse" regime at high LR (diffuse parameter drift). Our model is 1.5B —
    even smaller than the 4B threshold studied — making it MORE susceptible to the
    degrade regime at standard LR settings.

    Connection to cycle-1 dip: EXP-137 (epoch-1 checkpoint) and EXP-138 (entropy
    warmup) target the PLASTICITY explanation for the cycle-1 skills arm dip
    (65.85%). EXP-140 tests the ORTHOGONAL explanation from arxiv:2607.12640: the
    default GRPO LR (typically 1e-5) places the 1.5B model in the "degrade regime".
    Reducing LR to 5e-7 (a 20x reduction) is predicted to move the policy updates
    into the sub-threshold range where GRPO does not destroy existing capabilities.

arxiv:2607.01763 — "Denser != Better: Limits of On-Policy Self-Distillation for
    Continual Post-Training" (July 2026)

    Finds that SDPO (on-policy self-distillation with a frozen teacher copy) causes
    MORE catastrophic forgetting than GRPO in continual post-training. The mechanism:
    denser self-distillation induces larger parameter-space and response-space drift,
    and can amplify high-frequency formatting artifacts through a self-reinforcing
    teacher-student loop. By contrast, GRPO adapts more conservatively and better
    preserves prior capabilities because the policy gradient only fires on rollouts
    that actually change the reward — on-policy distillation fires on every token.

    Connection to our pipeline: our SFT phase IS teacher distillation (we clone from
    GPT-5.5 teacher traces). The paper predicts that our SFT phase introduces more
    forgetting than the GRPO phase that follows it. EXP-141 tests this directly:
    run cycle 1 WITHOUT the SFT phase (GRPO-only from the cycle-0 base model) and
    compare the forgetting profile vs the standard SFT+GRPO pipeline. If the paper's
    finding generalises to our offline distillation setting, GRPO-only should retain
    more cycle-0 skill (the base model's HumanEval pass@1 before any training)
    while still improving on cycle-1 tasks — a novel argument for shortening the
    SFT phase in our distillation loop.
"""

import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = "data/humaneval_eval.jsonl"

NEW_EXPERIMENTS = [
    # -----------------------------------------------------------------------
    # EXP-140: Learning-Rate-Gated GRPO Failure — LR Reduction for Cycle-1
    #
    # Addresses cycle-1 skills arm dip (65.85%) via the LR-regime explanation
    # orthogonal to EXP-137/138. (arXiv:2607.12640, July 2026)
    # -----------------------------------------------------------------------
    {
        "id": "exp_2026_07_31_001_lr_reduction_grpo_cycle1_degrade_regime",
        "priority": 8,
        "gpu": "auto",
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2607.12640 — 'A Learning-Rate-Gated Failure of GRPO in a Small "
            "Language and Vision-Language Model Web Agent' (July 2026). "
            "The paper demonstrates that GRPO fails — and actively degrades performance "
            "— at moderate-to-high learning rates for 4B-8B models, due to parameter "
            "drift in attention + MLP blocks ('degrade regime'). GRPO is only beneficial "
            "when the sampled policy already succeeds more often than greedy, i.e., when "
            "genuine headroom exists. Our model (Qwen2.5-Coder-1.5B, half the 4B "
            "threshold studied) is likely MORE susceptible to the degrade regime. "
            "The cycle-1 skills arm dip (65.85% vs 70.73% cycle-0) has been attributed "
            "to SFT epoch-2 plasticity loss (EXP-136 finding), motivating EXP-137 "
            "(epoch-1 checkpoint selection) and EXP-138 (entropy warmup). EXP-140 "
            "tests a THIRD orthogonal hypothesis: the standard GRPO LR places the "
            "1.5B model in the degrade regime, destroying cycle-0 capabilities. "
            "Fix: reduce GRPO LR from default (~1e-5) to 5e-7 (20x reduction), which "
            "is predicted to move policy updates into the sub-threshold range where "
            "reward-positive rollouts improve the model without triggering degrade-regime "
            "drift. Per the paper's controlled null, only LR matters for the failure mode "
            "(KL weight, clipping, seed are secondary) — making this a targeted 1-variable "
            "perturbation. "
            "AAAI impact: §5.4 Hypothesis F now has THREE orthogonal fixes — epoch "
            "selection (EXP-137), entropy warmup (EXP-138), and LR reduction (EXP-140). "
            "If all three recover the skills arm to >=70%, Table 9 gains a 3-row fix "
            "block showing convergent evidence for the cycle-1 plasticity/LR hypothesis. "
            "If only LR reduction works, it implicates the 'degrade regime' as the primary "
            "cause rather than SFT overfitting, shifting the paper's mechanism discussion. "
            "Estimated time: ~1h (single GRPO cycle with reduced LR). "
            "Priority: 8 (HIGH — last day before Aug 1 GPU window closes)."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 1,
            "start_from_cycle": 1,
            "grpo_lr": 5e-7,
            "grpo_lr_note": "20x reduction from default 1e-5; targets sub-degrade-regime per 2607.12640",
            "grpo_model_path": "results/e2e_4cyc_gpt55/cycle_1/llm_adapter/checkpoint-best",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "grpo_temperature": 1.0,
            "n_generations": 8,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "analysis": [
                "skills_arm_pass_at_1_lr_reduced_vs_default",
                "acr_fraction_lr_reduced_vs_default",
                "parameter_drift_attention_mlp_lr5e7_vs_lr1e5",
                "grpo_loss_convergence_lr_sweep",
                "comparison_with_exp137_exp138_three_fix_table",
            ],
            "compare_with": [
                "cycle_1 baseline: skills=65.85%, default LR~1e-5",
                "EXP-137: epoch-1 checkpoint, default LR (plasticity fix)",
                "EXP-138: epoch-2 checkpoint + entropy warmup (plasticity fix)",
                "EXP-140 (this): epoch-2 checkpoint + LR=5e-7 (degrade-regime fix)",
                "target: skills>=70.73% (cycle-0 level), confirming LR regime as cause",
            ],
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
                "scripts/run_full_pipeline.sh",
            ],
            "implementation_note": (
                "Pass GRPO_LR=5e-7 to grpo_train_simple.py or set in config. "
                "All other hyperparameters identical to cycle-1 baseline. "
                "Only change is the optimizer LR — single-variable control. "
                "Implementation reference: https://arxiv.org/abs/2607.12640."
            ),
            "arxiv_ref": "2607.12640",
            "estimated_gpu_hours": 1.0,
            "aaai_priority": "HIGH — orthogonal fix #3 for Hypothesis F; run before Aug 1",
            "dependency": "Run after EXP-132 (priority=9), EXP-135 (priority=8); before Aug 1",
        },
    },
    # -----------------------------------------------------------------------
    # EXP-141: Denser!=Better — Forgetting Profile: GRPO-Only vs SFT+GRPO
    #
    # Tests whether our SFT teacher-distillation phase causes MORE forgetting
    # than GRPO alone, per arXiv:2607.01763. Uses forgetting_eval kind.
    # -----------------------------------------------------------------------
    {
        "id": "exp_2026_07_31_002_forgetting_eval_grpo_only_vs_sft_grpo",
        "priority": 7,
        "gpu": "auto",
        "kind": "forgetting_eval",
        "rationale": (
            "arxiv:2607.01763 — 'Denser != Better: Limits of On-Policy Self-Distillation "
            "for Continual Post-Training' (July 2026). "
            "The paper shows that on-policy self-distillation (SDPO) causes MORE "
            "catastrophic forgetting than GRPO in continual post-training. The mechanism: "
            "denser teacher-student learning fires gradients on every token, inducing "
            "large parameter-space and response-space drift; by contrast, GRPO fires only "
            "on reward-changing rollouts, adapting more conservatively. The result: GRPO "
            "better preserves prior capabilities across tasks while achieving similar or "
            "better in-distribution improvement. "
            "Connection to our pipeline: our SFT phase IS teacher distillation — we clone "
            "from GPT-5.5 teacher traces using a cross-entropy (SFT) loss that applies "
            "gradient to EVERY token in every teacher trajectory. Per arxiv:2607.01763, "
            "this is the exact 'denser' regime that induces more forgetting than GRPO. "
            "Our observed forgetting pattern (cycle-1 skills arm dip, ACR decline from "
            "cycle 0 to cycle 1) may be partly caused by the SFT phase, not solely by "
            "GRPO. EXP-141 tests this by running cycle-1 WITHOUT the SFT phase "
            "(GRPO-only, starting from the cycle-0 base checkpoint) and measuring: "
            "(a) forgetting of base model HumanEval pass@1 (pre-training capabilities), "
            "(b) cycle-1 skills arm pass@1, "
            "(c) ACR fraction vs SFT+GRPO pipeline. "
            "If GRPO-only forgets LESS than SFT+GRPO but achieves similar cycle-1 skill, "
            "this motivates removing or shortening the SFT phase in future cycles — "
            "a structurally significant result for our iterative distillation loop. "
            "If GRPO-only forgets MORE (counter to 2607.01763), it suggests our offline "
            "teacher distillation actually serves as an implicit regulariser against "
            "forgetting — also a novel positive finding. "
            "AAAI impact: new §5.5 'Role of Teacher Distillation in Forgetting' paragraph. "
            "Grounded in the latest continual post-training literature (2607.01763). "
            "Distinct from EXP-130 (CPO-PMP inter-cycle forgetting with KL regularizer): "
            "EXP-141 tests forgetting from the SFT PHASE itself, not from the GRPO phase. "
            "Estimated time: ~2h (GRPO-only cycle from cycle-0 checkpoint). "
            "Priority: 7 (novel angle; run after EXP-132/135/140/137/138 if time permits)."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 1,
            "start_from_cycle": 1,
            "skip_sft": True,
            "skip_sft_note": "GRPO initialised from cycle-0 base checkpoint, no SFT phase",
            "grpo_model_path": "results/e2e_4cyc_gpt55/cycle_0/llm_adapter/checkpoint-best",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "grpo_temperature": 1.0,
            "n_generations": 8,
            "scaling_force_both": True,
            "eval_data": _HE_EVAL,
            "forgetting_metrics": [
                "pass_at_1_on_held_out_base_tasks",
                "pass_at_1_on_cycle_1_tasks",
                "acr_fraction",
                "parameter_drift_norm_vs_cycle0_checkpoint",
                "response_distribution_kl_cycle0_vs_cycle1",
            ],
            "compare_with": [
                "SFT+GRPO pipeline (cycle-1 baseline): skills=65.85%, ACR=?",
                "GRPO-only (EXP-141, this): no SFT phase, pure GRPO from cycle-0",
                "2607.01763 prediction: GRPO-only < SFT+GRPO forgetting",
                "target: confirm/refute 2607.01763 in offline-distillation regime",
            ],
            "implementation_files": [
                "scripts/run_full_pipeline.sh",
                "src/pipeline/train_small_model.py",
                "src/pipeline/grpo_train_simple.py",
            ],
            "implementation_note": (
                "Set SKIP_SFT=1 in run_full_pipeline.sh (or add --skip-sft flag) to skip "
                "Phase 3a (SFT). GRPO Phase 3b then initialises from cycle_0/llm_adapter/"
                "checkpoint-best instead of the cycle-1 SFT output. "
                "Forgetting measured on a held-out split of base HumanEval tasks not seen "
                "in cycle-1 GRPO rollouts (use the existing eval_data split if available). "
                "Implementation reference: https://arxiv.org/abs/2607.01763."
            ),
            "arxiv_ref": "2607.01763",
            "estimated_gpu_hours": 2.0,
            "aaai_priority": "EXPLORATORY — novel forgetting angle; run if GPU time after EXP-132/135/140",
            "dependency": "Requires cycle-0 llm_adapter checkpoint-best to be available",
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
