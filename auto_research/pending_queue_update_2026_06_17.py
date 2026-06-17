#!/usr/bin/env python3
"""
Daily queue patch — 2026-06-17 (EXP-074, EXP-075).

A800 connectivity: offline since 2026-05-14 (day 34). Apply when restored.

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

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_06_17.py       # EXP-074, EXP-075

Queue was ~73 pending on 2026-06-16 (+2 added: EXP-072, EXP-073).
Queue cap applied: >20 → max 2 new experiments.
"""
import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_06_17_001_egca_execution_credit_200tasks",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2603.16158 ('Execution-Grounded Credit Assignment for GRPO in Code "
            "Generation', Kumar et al., March 2026; accepted ICLR 2026 Workshop on "
            "Scaling Post-Training for LLMs, SPOT) identifies coarse credit assignment "
            "as GRPO's dominant failure mode for code: a single binary outcome reward "
            "is spread uniformly across all tokens in a program, regardless of which "
            "tokens caused the failure. For a 30-token failing function, all 30 tokens "
            "receive equal negative advantage — but typically only 2-5 tokens are "
            "semantically responsible for the test failure. This dilution directly "
            "reduces signal-to-noise ratio and forces the model to strengthen many "
            "correct tokens alongside the buggy ones. EGCA (Execution-Grounded Credit "
            "Assignment) addresses this via deterministic execution-trace localization: "
            "(1) Failure mode gates — samples are classified into syntax errors, "
            "algorithmic constraint violations, or logic errors. Syntax and constraint "
            "failures receive standard uniform GRPO advantage (their token-level cause "
            "is typically early and deterministic). (2) For logic failures (the "
            "majority on MBPP), EGCA executes both the candidate program and a "
            "canonical reference solution under identical test instrumentation, then "
            "walks the execution trace to find the earliest state-divergence point — "
            "the first variable assignment, conditional branch, or return value where "
            "the candidate deviates from the reference. (3) Advantage is assigned "
            "exclusively to the token span corresponding to this divergence point; "
            "all downstream tokens are masked to zero gradient. The reference "
            "solutions are those already present in the MBPP dataset (used as "
            "canonical ground truth, not as supervision — the reference is never "
            "shown to the model). EGCA adds: zero critic model, zero auxiliary loss, "
            "zero extra LLM calls. The only overhead is two Python executions per "
            "failing rollout (~18% wall-clock on HumanEval/MBPP-scale programs). "
            "Published results: HumanEval 82.1% pass@1 (+3.1 over GRPO); "
            "MBPP 68.9% pass@1 (+1.5 over GRPO). The MBPP +1.5 pp result is on the "
            "same benchmark (MBPP) and evaluation protocol we use — directly "
            "comparable to our 47→49 (+2 pp) best result. If both effects are "
            "additive, EGCA could push our baseline from 49 toward 50-51 pp, "
            "which would be statistically distinguishable from the base 47 and "
            "strengthen our AAAI submission. Implementation note for runner.py: "
            "spec field `advantage_assignment='egca'` should (a) classify failing "
            "rollouts by failure mode gate, (b) for logic failures, run candidate "
            "and MBPP canonical solution in subprocess under the existing "
            "HumanEval-style harness, walk the trace, find first divergence, "
            "output a token-mask tensor, and pass it to the advantage module; "
            "for syntax/constraint failures, use standard uniform advantage. "
            "The canonical solution per MBPP task is available as "
            "`dataset['canonical_solution']` in the HuggingFace MBPP dataset. "
            "Distinct from EP-GRPO (EXP-068): EP-GRPO weights advantage by "
            "per-token entropy (higher entropy = more important token) — an "
            "information-theoretic proxy that does not use execution state. "
            "EGCA uses actual execution divergence — a causal, program-semantic "
            "criterion. The two are complementary and could be combined (EGCA "
            "localization + EP-GRPO entropy weighting within the localized span) "
            "but should be tested separately first. Distinct from GTPO (EXP-071): "
            "GTPO's positive-only shared-token filter operates on the group level "
            "across rollouts; EGCA's mask operates within each individual rollout "
            "on the execution-trace level. Wall-clock estimate: ~85 min "
            "(200 tasks × G=4 rollouts; logic-failure gate + 2 subprocess calls "
            "per failing rollout ≈ 18% overhead over ~70 min baseline)."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": (
                "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/"
                "train_aug_excluding_eval20.jsonl"
            ),
            "eval_data": (
                "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/"
                "eval100.jsonl"
            ),
            "train_task_limit": 200,
            "epochs": 1,
            "rollouts_per_prompt": 4,
            "lr": 5e-6,
            "lora_r": 16,
            "prompt_style": "qwen-chat",
            "reward": "binary",
            "advantage_assignment": "egca",
            "egca_reference_solutions": "mbpp_canonical",
            "egca_failure_mode_gates": ["syntax", "constraint", "logic"],
            "egca_logic_trace_localization": True,
            "arxiv_ref": "2603.16158",
            "estimated_hours": 1.42,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_06_17_002_tarot_testtiered_intra_problem_200tasks",
        "priority": 7,
        "kind": "grpo_curriculum_continual",
        "rationale": (
            "arxiv:2602.15449 ('TAROT: Test-driven and Capability-adaptive Curriculum "
            "Reinforcement Fine-tuning for Code Generation with Large Language Models', "
            "Park et al., February 2026; accepted OpenReview ICLR 2026 track) introduces "
            "a fundamentally different axis of curriculum design: intra-problem test "
            "difficulty tiering, as opposed to the inter-problem difficulty ordering "
            "used by all other curricula in our queue (EXP-062 easy-to-hard static, "
            "EXP-063/065 curriculum/reverse-curriculum at 400 tasks, EXP-069 CLPO online). "
            "Key insight: existing GRPO-based code training treats all test cases for a "
            "given problem as interchangeable — a binary pass@1 over the full test suite "
            "is the reward signal. But different test cases for the same problem vary "
            "enormously in difficulty: a basic correctness check (input=[1,2], "
            "expected=[1,2]) is qualitatively easier than an edge case (empty input, "
            "large numbers, aliased objects). Mixing all test cases uniformly leads to "
            "imbalanced reward signals and biased gradient updates: the model gets "
            "rewarded for passing easy cases even when it fails hard ones, which provides "
            "a false training signal about its actual capability. TAROT addresses this "
            "by constructing, for each problem, a four-tier test suite: (1) Basic — "
            "minimal input/output pairs validating the core algorithmic logic; "
            "(2) Intermediate — moderately complex inputs testing boundary conditions "
            "and typical use cases; (3) Complex — stress tests with large inputs, "
            "nested structures, and multi-constraint scenarios; (4) Edge — corner cases "
            "including empty inputs, degenerate cases, type edge cases, and aliased "
            "data structures. The tier assignment is automated: TAROT uses a "
            "capability-adaptive scoring function based on the model's current "
            "pass-rate per tier to identify the capability frontier and train on tiers "
            "at the frontier (e.g., start with Basic → progress to Intermediate "
            "as pass-rate on Basic saturates → etc.). This is a within-problem "
            "curriculum rather than a between-problem one. Published results: TAROT "
            "consistently improves pass@1 over the base checkpoint on Qwen2.5-Instruct "
            "and Qwen2.5-Coder-Instruct (1.5B, 3B, 7B) on HumanEval and MBPP. The "
            "1.5B Qwen2.5-Coder configuration is directly our model and benchmark. "
            "Mechanism is distinct from and orthogonal to all entropy/advantage "
            "experiments (EXP-068 EP-GRPO, EXP-070 Entrocraft, EXP-071 GTPO, "
            "EXP-072 Infinite Sampling): those modify the GRPO advantage computation; "
            "TAROT modifies which test cases are used to compute the binary pass/fail "
            "reward and which tasks are presented during training. Implementation note "
            "for runner.py: spec field `curriculum='tarot_testtiered'` should (a) "
            "construct or load a pre-built four-tier test expansion of the MBPP "
            "training set (MBPP-aug can be extended: original test cases become "
            "'basic'; intermediate/complex/edge tiers generated via mutation-based "
            "test augmentation or LLM-assisted generation); (b) implement a "
            "capability-adaptive tier scheduler that re-evaluates per-tier pass-rate "
            "every N steps and advances the tier when saturation is detected; "
            "(c) report per-tier pass-rate trajectory alongside the standard eval100 "
            "eval. Dependency: requires a four-tier MBPP test expansion; if not "
            "pre-built on A800, runner.py must generate it (one-time cost, ~10 min). "
            "Wall-clock estimate: ~90 min (200 tasks × G=4 + tier scheduling overhead; "
            "within 4-hour budget)."
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": (
                "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/"
                "train_aug_excluding_eval20.jsonl"
            ),
            "eval_data": (
                "/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/"
                "eval100.jsonl"
            ),
            "train_task_limit": 200,
            "epochs": 1,
            "rollouts_per_prompt": 4,
            "lr": 5e-6,
            "lora_r": 16,
            "prompt_style": "qwen-chat",
            "reward": "binary",
            "curriculum": "tarot_testtiered",
            "tarot_tiers": ["basic", "intermediate", "complex", "edge"],
            "tarot_tier_advancement": "capability_adaptive",
            "tarot_tier_saturation_threshold": 0.85,
            "arxiv_ref": "2602.15449",
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
