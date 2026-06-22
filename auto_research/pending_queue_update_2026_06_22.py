#!/usr/bin/env python3
"""
Daily queue patch — 2026-06-22 (EXP-084, EXP-085).

A800 connectivity: offline since 2026-05-14 (day 39). Apply when restored.

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

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_06_22.py            # EXP-084, EXP-085

Queue was ~83 pending on 2026-06-21 (+2 added: EXP-082, EXP-083).
Queue cap applied: >20 → max 2 new experiments.

Data motivating today's proposals
----------------------------------
results/cwy_35b_fullsplit_20260610_083624/cycle_3/e2e_ablation_summary.json:
    tau2 multi-cycle held-out: 70.22% (c0) → 72.47% (c1) → 70.22% (c2) → 69.66% (c3)
    → Consistent post-peak decline across cycles, despite SFT improving on train split
    → Both SFT (Phase 3a) and GRPO (Phase 3b) occur each cycle; root cause undiagnosed
    → Paper W6 flags this unexplained decline; AAAI reviewers will ask about it

arxiv:2605.28860 (deferred from 2026-06-21, "too costly today"):
    'Mechanistic Origins of Catastrophic Forgetting: Why RL Preserves Circuits Better
    than SFT?' (Rojas Nunez et al., June 2026). RL fine-tuning preserves a larger
    fraction of base model circuits (measured via differential circuit vulnerability at
    the attention-head level) compared to SFT. SFT adapts faster but disrupts more
    circuits. Prediction: tau2 cycle decline is SFT-phase-induced, not GRPO-phase.

arxiv:2601.18699 (new today):
    'Mechanistic Analysis of Catastrophic Forgetting in Large Language Models During
    Continual Fine-tuning' (January 2026). Studies six LLM architectures (Llama 4
    Scout 109B, Llama 4 Maverick 400B, GPT-5.1, Claude Opus 4.5, Gemini 2.5 Pro,
    Qwen3.6-35B). Identifies three primary forgetting mechanisms: gradient interference
    in attention weights, representational drift in intermediate layers, and loss
    landscape flattening. Gradient interference in attention weights is the primary
    mechanism for LoRA-based SFT (relevant: our tau2 SFT uses LoRA).

arxiv:2601.05242 (new today):
    'GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL
    Optimization' (NVLabs, January 2026). GRPO/DAPO normalizes the SUM of rewards across
    the group, which collapses distinct reward components (e.g., correctness + format)
    into identical advantage values when one reward dominates. GDPO instead normalizes
    each reward component INDEPENDENTLY within the group before aggregation — a drop-in
    replacement for GRPO in TRL and verl. Tested on tool calling, math reasoning, and
    coding reasoning: GDPO consistently outperforms GRPO across all settings.
    Code: https://github.com/NVlabs/GDPO.
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
_TAU2_CONFIG = "config/tau2_bench.yaml"

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_06_22_001_tau2_sft_only_2cycle_forgetting_diagnostic",
        "priority": 8,
        "kind": "joint_cycle_multiseed",
        "rationale": (
            "arxiv:2605.28860 — 'Mechanistic Origins of Catastrophic Forgetting: Why RL "
            "Preserves Circuits Better than SFT?' (Rojas Nunez et al., Jun 2026). RL "
            "fine-tuning preserves significantly more of the model's base computational "
            "circuits than SFT, measured via differential circuit vulnerability at "
            "attention-head level. SFT adapts faster but disrupts substantially more "
            "circuits. This paper directly predicts that the SFT phases (Phase 3a) are "
            "the primary cause of the tau2 held-out decline "
            "(70.22% → 72.47% → 70.22% → 69.66% over 4 cycles), not the GRPO phases "
            "(Phase 3b). See also arxiv:2601.18699 (Jan 2026, Mechanistic Analysis of "
            "Catastrophic Forgetting): for LoRA-based continual SFT specifically, "
            "gradient interference in attention weights is the primary forgetting "
            "mechanism — exactly the architecture used in our tau2 SFT pipeline. "
            "Diagnostic experiment: run the tau2 2-cycle pipeline with SKIP_GRPO=1, "
            "disabling Phase 3b (GRPO) entirely. Each cycle performs Phase 3a (SFT) "
            "and Phase 4 (router training) only. Compare held-out performance across "
            "cycles to the existing joint run (cwy_35b_fullsplit_20260610_083624): "
            "(a) If SFT-only ALSO declines across cycles → SFT is the forgetting cause. "
            "Actionable paper fix: cite arxiv:2506.09428 (Improved SFT via prior "
            "distribution reconstruction + data mixing) as mitigation direction; state "
            "in §7 that SFT-phase circuit disruption (arxiv:2601.18699) explains W6. "
            "(b) If SFT-only holds or improves → GRPO is the forgetting cause. "
            "Actionable fix: reduce GRPO KL strength or add on-policy replay "
            "(EXP-079 already queued for HumanEval; extend to tau2). "
            "This deferred experiment from 2026-06-21 report ('too costly today') is "
            "now the highest-priority diagnostic for W6. Running today because: "
            "(1) it is within the 4-hour wall-clock limit (SKIP_GRPO reduces per-cycle "
            "time from ~105 min to ~65 min; 2 cycles ≈ 2.2 GPU-hours); "
            "(2) W6 is an open AAAI reviewer objection with no current answer; "
            "(3) no queued experiment addresses tau2 forgetting root-cause isolation. "
            "Compare with: cwy_35b_fullsplit (reference, with GRPO), which shows "
            "consistent post-peak decline. If SFT-only shows no decline, the paper can "
            "assert GRPO plasticity loss as the cause with direct ablation evidence."
        ),
        "spec": {
            "pipeline": "tau2_2cycle_ablation",
            "bench": "tau2_bench",
            "config": _TAU2_CONFIG,
            "n_cycles": 2,
            "n_seeds": 1,
            "seeds": [42],
            "small_model": "Qwen/Qwen3.6-35B-A3B",
            "large_model": "gpt-5.4",
            "skip_grpo": True,
            "env_flags": {"SKIP_GRPO": "1", "SKIP_LLM": "0"},
            "sft_include_success": True,
            "eval_split": "held_out",
            "eval_data": _TAU2_HELD_OUT,
            "primary_metrics": [
                "cycle1_base_task_pass",
                "cycle1_router_task_pass",
                "cycle2_base_task_pass",
                "cycle2_router_task_pass",
            ],
            "compare_with": [
                "cwy_35b_fullsplit_20260610_083624 (reference: SFT+GRPO joint, 4 cycles)"
            ],
            "paper_weakness_addressed": "W6",
            "paper_claim_to_test": (
                "Is the tau2 cycle decline (70.22% → 69.66%) caused by SFT-phase "
                "circuit disruption (arxiv:2605.28860) or GRPO-phase plasticity loss?"
            ),
            "arxiv_ref": ["2605.28860", "2601.18699"],
            "estimated_hours": 2.2,
            "note": (
                "Run with SKIP_GRPO=1. Two cycles of: collect traces → skills evolve → "
                "SFT (Phase 3a) → router train (Phase 4) → ablation eval (Phase 5). "
                "No GRPO weight update. Compare cycle 0→1→2 held-out task pass to "
                "reference run to isolate SFT-only forgetting contribution."
            ),
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_06_22_002_gdpo_decoupled_normalization_humaneval_binary_format",
        "priority": 7,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2601.05242 — 'GDPO: Group reward-Decoupled Normalization Policy "
            "Optimization for Multi-reward RL Optimization' (NVLabs, January 2026; "
            "code: github.com/NVlabs/GDPO). Standard GRPO and DAPO normalize the SUM "
            "of reward components within a group before computing advantages. When two "
            "reward components have different scales or training dynamics, this collapses "
            "their relative gradient signal: one reward dominates early (easy binary "
            "signal), suppressing the gradient from the harder-to-optimize second "
            "reward. GDPO resolves this by normalizing each reward component "
            "INDEPENDENTLY within the group, computing a per-component advantage, "
            "then aggregating via a learnable or fixed weighting. "
            "Direct motivation from our codebase: grpo_train_simple.py currently uses "
            "'reward': 'binary' (single component: test-pass 0/1). In the repair "
            "rollout setting (max_turns=3), a partial reward structure emerges implicitly: "
            "the model can produce syntactically valid code that fails tests (format "
            "correct, correctness incorrect) or invalid code (format incorrect). "
            "EXP-009 (P-GRPO) showed that fractional correctness rewards hurt vs binary "
            "— but that was a single-component fractional reward. GDPO addresses "
            "MULTI-component reward interaction, which is different. "
            "This experiment adds an explicit second reward component: "
            "R2 = format compliance (binary: 1 if response contains a ```python``` code "
            "fence with valid Python syntax, 0 otherwise). R2 is already computed "
            "implicitly during evaluation but discarded. With DAPO normalization of "
            "sum(R1+R2): when R1=0 for all rollouts (zero-variance group), "
            "sum = R2 only, and the advantage signal degrades to a format-only "
            "gradient. GDPO normalizes R1 and R2 independently — even a zero-variance "
            "R1 group retains its proper (zero) contribution, while R2 provides a clean "
            "format signal scaled by R2's within-group variance, not polluted by R1. "
            "Expected outcome: improved code format compliance (fewer parse failures in "
            "repair turns) and potentially higher pass@1 from cleaner code structure. "
            "The format reward is almost always 1 by turn 2 (models learn fencing "
            "quickly), so GDPO's advantage is mainly in cycle 0 / early training where "
            "format errors are more common and R1/R2 interaction is highest. "
            "Implementation: ~25 lines in grpo_train_simple.py replacing the single "
            "advantage computation with per-component GDPO normalization "
            "(per-component zero-mean unit-variance within group, then sum with "
            "weight w_format=0.3). GDPO is a drop-in in TRL, so no architectural change. "
            "Wall-clock: ~90 min (same as reference DAPO G=8 HumanEval run). "
            "Duplicate check: no queued experiment uses GDPO or multi-component "
            "independent reward normalization. EXP-009 (P-GRPO fractional correctness) "
            "and EXP-068 (EP-GRPO entropy gating) are different mechanisms. "
            "No prior experiment adds a second reward component to HumanEval training."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": _HE_EVAL,
            "eval_data": _HE_EVAL,
            "bench": "humaneval",
            "grpo_algo": "gdpo",
            "dapo_dynamic_sampling": True,
            "dapo_clip_low": 0.2,
            "dapo_clip_high": 0.5,
            "n_generations": 8,
            "rewards": [
                {
                    "name": "correctness",
                    "type": "binary_test_pass",
                    "weight": 1.0,
                    "normalize_independently": True,
                },
                {
                    "name": "format_compliance",
                    "type": "binary_code_fence",
                    "weight": 0.3,
                    "normalize_independently": True,
                    "description": (
                        "1 if response contains a ```python``` fence with parseable "
                        "Python AST, 0 otherwise"
                    ),
                },
            ],
            "gdpo_aggregation": "weighted_sum",
            "rollout": "repair",
            "max_turns": 3,
            "epochs": 1,
            "lr": 5e-6,
            "lora_r": 16,
            "beta": 0.04,
            "prompt_style": "qwen-chat",
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (DAPO G=8 single binary reward, reference)",
                "exp_2026_06_21_001_rlzvp_entropy_advantage_humaneval_dapo",
            ],
            "arxiv_ref": "2601.05242",
            "data_motivation": (
                "EXP-009 (P-GRPO fractional reward) showed fractional SINGLE-component "
                "reward hurts; GDPO uses INDEPENDENT normalization of MULTIPLE components — "
                "a categorically different mechanism. First multi-reward GRPO experiment "
                "on HumanEval in the queue."
            ),
            "estimated_hours": 1.5,
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
