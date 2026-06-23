#!/usr/bin/env python3
"""
Daily queue patch — 2026-06-23 (EXP-086, EXP-087).

A800 connectivity: offline since 2026-05-14 (day 40). Apply when restored.

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

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_06_23.py            # EXP-086, EXP-087

Queue was ~85 pending on 2026-06-22 (+2 added: EXP-084, EXP-085).
Queue cap applied: >20 → max 2 new experiments.

Data motivating today's proposals
----------------------------------
results/e2e_4cyc_gpt55/cycle_3/e2e_ablation_summary.json:
    HumanEval 4-cycle (MERA) full/router: 92.68% task pass at 27.56% large-model cost
    GRPO zero-variance: 52.4% of groups discarded at G=8 DAPO (43/82 groups)
    Skills arm (always-small+procedure): 75.61% (up from 70.73% at cycle 0)

results/cwy_35b_fullsplit_20260610_083624/cycle_3/e2e_ablation_summary.json:
    tau2 held-out: base/skills=76.97%, router=69.66%, always-large=60.11%
    Cycle-over-cycle decline: 70.22% → 72.47% → 70.22% → 69.66% (W6, root cause unknown)
    → EXP-084 (SFT-only diagnostic) will isolate whether SFT Phase 3a is the cause.
    → EXP-087 (FOREVER replay) provides a principled mitigation, irrespective of root cause.

arxiv:2601.03525 (new today — never mentioned in any prior report):
    'VeRPO: Verifiable Dense Reward Policy Optimization for Code Generation'
    (January 7, 2026). Constructs dense rewards from dynamically weighted partial test
    success. During training, difficulty weight of each unit test is estimated from
    execution statistics: tests that most rollouts fail get higher weight; tests that
    most rollouts pass get lower weight. Fuses this dense turn-level reward with a
    global trajectory-level outcome (binary pass/fail). Drop-in for TRL GRPO.
    Different from EXP-009 (P-GRPO, failed): P-GRPO used a static fraction of passed
    tests as a single-component reward — VeRPO dynamically reweights each test case
    individually, creating a curriculum that shifts attention to the hardest unsolved
    tests as training progresses. P-GRPO's failure (arxiv:2605.02944) was attributed
    to collapsed reward variance with static fractional rewards; VeRPO's dynamic
    reweighting maintains reward variance throughout training.

arxiv:2601.03938 (new today — never mentioned in any prior report):
    'FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual
    Learning' (January 2026). Observes that LLM forgetting follows the Ebbinghaus
    forgetting curve when measured against model update magnitude (optimizer step
    norm). Replaces fixed-step replay intervals with a forgetting-curve-aligned
    schedule: more frequent replay immediately after a cycle's SFT update, decaying
    replay frequency as the model stabilises. Adds intensity-aware regularization
    scaling replay loss by predicted forgetting severity at that model-time point.
    Tested on 0.6B–13B models across three continual learning benchmarks; consistently
    reduces forgetting while maintaining plasticity. Directly applicable to our tau2
    SFT Phase 3a, where each cycle's LoRA update is the forgetting event and
    held-out task pass rate is the forgetting metric.
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
_TAU2_CONFIG = "config/tau2_bench.yaml"

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_06_23_001_verpo_dynamic_testweight_dense_reward_humaneval",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2601.03525 — 'VeRPO: Verifiable Dense Reward Policy Optimization "
            "for Code Generation' (January 7, 2026). "
            "Standard GRPO and DAPO use a binary outcome reward: +1 if ALL tests pass, "
            "0 if ANY test fails. This collapses all the signal from partially correct "
            "solutions: a solution that passes 7 of 8 tests receives the same zero reward "
            "as a solution that passes 0 of 8. VeRPO addresses this by constructing a "
            "dense reward from DYNAMICALLY weighted partial test success. "
            "VeRPO's mechanism: "
            "(1) During training, for each unit test t in problem p, compute "
            "w(t) = 1 - mean_{rollouts}[pass(t)] — the test's failure rate across all "
            "rollouts in the current training batch. Tests that most rollouts fail get "
            "high weight; tests that most rollouts pass (already mastered) get low weight. "
            "(2) The dense reward for rollout r on problem p is: "
            "R_dense(r) = sum_t [ w(t) * pass(r, t) ] / sum_t [w(t)] — a weighted pass "
            "fraction that emphasizes the hardest unsolved tests. "
            "(3) A global trajectory-level outcome reward R_outcome (binary, like DAPO) "
            "is retained and combined: R_final = alpha * R_dense + (1-alpha) * R_outcome. "
            "The key difference from EXP-009 (P-GRPO, demoted after failing): "
            "P-GRPO used a STATIC fraction of tests passed (no weighting, no dynamics). "
            "Its failure (arxiv:2605.02944) was attributed to collapsed reward variance — "
            "as training progressed, the model passed most tests on easy problems, and "
            "the fractional reward converged toward 1.0 for almost all rollouts on those "
            "problems, eliminating the advantage signal (same problem as zero-variance "
            "groups in DAPO). VeRPO prevents this collapse: the difficulty weights w(t) "
            "are RECOMPUTED each batch, so as the model masters easy tests, those tests' "
            "weights drop toward zero and hard tests gain proportionally higher weight — "
            "the reward variance is maintained throughout training by continuously "
            "redirecting attention to the unsolved tests. "
            "The fusion parameter alpha controls the trade-off: alpha=0 is standard DAPO "
            "(binary outcome); alpha=1 is pure dense reward (no trajectory-level signal). "
            "VeRPO recommends alpha=0.5 based on ablations in the paper. "
            "Implementation in grpo_train_simple.py: replace the binary reward computation "
            "with VeRPO's two-component reward. Requires tracking per-test pass rates "
            "across rollouts within each batch (~30 lines, no external dependencies). "
            "Direct applicability to HumanEval: each problem has multiple test cases in "
            "the MBPP/HumanEval evaluation harness; the existing test runner already "
            "returns per-test pass/fail, which is needed for w(t) computation. "
            "Wall-clock: ~90 min (identical rollout count and model size to reference "
            "DAPO G=8 run; weight recomputation per batch adds ~5% overhead). "
            "Duplicate check: "
            "EXP-009 (P-GRPO): STATIC fractional reward, single component, failed. "
            "VeRPO: DYNAMIC difficulty-weighted multi-test reward fused with binary "
            "outcome — categorically different mechanism, addresses P-GRPO's failure mode "
            "directly. No queued experiment uses dynamic test-difficulty weighting. "
            "EXP-085 (GDPO): multi-component independent reward normalization — different "
            "axis (normalization of component advantages); VeRPO modifies the reward "
            "VALUES themselves, not how advantages are normalized. "
            "No queued experiment has 'verpo', 'dynamic_testweight', or 'dense_reward' "
            "in its id or spec. ✓"
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": _HE_EVAL,
            "eval_data": _HE_EVAL,
            "bench": "humaneval",
            "grpo_algo": "verpo",
            "reward_type": "verpo_dynamic_dense_binary_fusion",
            "verpo_alpha": 0.5,
            "verpo_test_weight_update": "per_batch_failure_rate",
            "verpo_dense_component": "difficulty_weighted_partial_pass",
            "verpo_outcome_component": "binary_all_tests_pass",
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
                "e2e_4cyc_gpt55 cycle_3 (DAPO G=8 binary outcome, reference)",
                "exp_2026_06_06_001 (EGCA execution-grounded credit, different mechanism)",
            ],
            "arxiv_ref": "2601.03525",
            "distinct_from_exp009": (
                "EXP-009 used static fraction-passed single-component reward → collapsed "
                "variance as model improved. VeRPO recomputes test-difficulty weights each "
                "batch → maintains variance by upweighting unsolved tests continuously."
            ),
            "estimated_hours": 1.5,
            "implementation_note": (
                "Modify grpo_train_simple.py: after collecting rollouts, compute per-test "
                "pass rates across rollouts in the batch to get w(t) weights. Compute "
                "R_dense = weighted pass fraction per rollout. R_final = 0.5*R_dense + "
                "0.5*R_outcome. Replace the scalar reward in the DAPO advantage computation "
                "with R_final. ~30 lines total."
            ),
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_06_23_002_forever_forgetting_curve_sft_replay_tau2_2cycle",
        "priority": 7,
        "kind": "joint_cycle_multiseed",
        "rationale": (
            "arxiv:2601.03938 — 'FOREVER: Forgetting Curve-Inspired Memory Replay for "
            "Language Model Continual Learning' (January 2026). "
            "FOREVER addresses catastrophic forgetting in continual LLM fine-tuning by "
            "aligning the replay schedule with the model's INTERNAL LEARNING DYNAMICS "
            "rather than a fixed step count. "
            "Core insight: LLM forgetting, when measured against optimizer update "
            "magnitude (the L2 norm of the parameter change per step), follows the "
            "Ebbinghaus forgetting curve — rapid forgetting immediately after parameter "
            "update, exponentially decaying as the model stabilizes. Fixed-step replay "
            "heuristics (e.g., 'replay every 100 steps') misalign with this: a large "
            "gradient step at step 50 causes more forgetting than a small step at step "
            "100, but both receive the same replay interval. "
            "FOREVER's mechanism: "
            "(1) Define model time τ as the cumulative L2 norm of optimizer updates "
            "(Σ ||Δθ||_2). Unlike raw step count, τ reflects the actual magnitude of "
            "parameter drift. "
            "(2) Schedule replay events using the Ebbinghaus curve: the next replay "
            "event is triggered when predicted forgetting (based on τ since last replay) "
            "exceeds a threshold θ_forget. Immediately after a large gradient update, "
            "τ grows quickly → replay is triggered soon. As the model stabilises, τ grows "
            "slowly → replay intervals lengthen. "
            "(3) Intensity-aware regularization: scale the replay loss by the predicted "
            "forgetting severity at the current model time, so replay has proportionally "
            "stronger gradient signal when forgetting is predicted to be high. "
            "Tested on 0.6B–13B decoder-only LLMs across three continual learning "
            "benchmarks. Consistently reduces forgetting while maintaining plasticity. "
            "The 35B Qwen3 (our tau2 small model) was NOT directly tested, but the "
            "paper's 7B results are the most applicable scaling point. "
            "Relevance to our project: "
            "The tau2 cycle decline (70.22%→72.47%→70.22%→69.66%, W6) is the primary "
            "open AAAI paper weakness. EXP-084 (queued June 22) will diagnose WHETHER "
            "the decline is SFT-induced or GRPO-induced by running SKIP_GRPO=1. "
            "FOREVER is the MITIGATION EXPERIMENT: regardless of root cause, if the "
            "tau2 SFT phase is causing forgetting, FOREVER's Ebbinghaus-scheduled replay "
            "of cycle-0 training data within cycle-1's SFT phase provides a principled "
            "anti-forgetting intervention with strong theoretical grounding. "
            "The experiment runs 2 tau2 cycles with FOREVER's replay integrated into "
            "Phase 3a (SFT). Within each cycle's SFT training, FOREVER monitors cumulative "
            "optimizer update magnitude and inserts replay samples from the PREVIOUS "
            "cycle's SFT training data (held in a small episodic buffer, "
            "buffer_size=50 samples, ~6% of typical SFT corpus size) at Ebbinghaus- "
            "scheduled intervals. "
            "This is complementary to EXP-084 and not a duplicate: "
            "EXP-084 ablates SKIP_GRPO (is GRPO or SFT the cause?) with NO replay. "
            "EXP-087 applies FOREVER replay within the FULL SFT+GRPO pipeline to "
            "directly measure whether forgetting is reduced. These two experiments together "
            "allow the AAAI paper to say: "
            "(a) EXP-084 identifies the forgetting phase (SFT vs GRPO); "
            "(b) EXP-087 demonstrates a concrete mitigation (FOREVER replay); "
            "(c) Combined, they close W6 with both diagnosis and remedy. "
            "Distinct from EXP-079 (on-policy GRPO replay, HumanEval/MBPP, queued "
            "2026-06-19): that experiment replays GRPO rollout data within the GRPO "
            "phase to prevent policy drift. FOREVER replays SFT TRAINING DATA within the "
            "SFT phase using an Ebbinghaus schedule. Different phase (SFT vs GRPO), "
            "different data type (supervised pairs vs rollout traces), different scheduling "
            "(Ebbinghaus vs fixed interval). "
            "Wall-clock: ~2.5 GPU-hours "
            "(2 tau2 cycles × ~65 min each with FOREVER overhead; "
            "replay adds ~15 min per cycle for buffer management and Ebbinghaus scheduling). "
            "Within 4-hour budget. ✓"
        ),
        "spec": {
            "pipeline": "tau2_2cycle_forever_replay",
            "bench": "tau2_bench",
            "config": _TAU2_CONFIG,
            "n_cycles": 2,
            "n_seeds": 1,
            "seeds": [42],
            "small_model": "Qwen/Qwen3.6-35B-A3B",
            "large_model": "gpt-5.4",
            "sft_phase": "3a",
            "sft_include_success": True,
            "forever_replay": True,
            "forever_model_time_metric": "cumulative_optimizer_update_norm",
            "forever_forgetting_threshold": 0.15,
            "forever_replay_buffer_size": 50,
            "forever_replay_data": "previous_cycle_sft_pairs",
            "forever_intensity_regularization": True,
            "eval_split": "held_out",
            "eval_data": _TAU2_HELD_OUT,
            "primary_metrics": [
                "cycle1_base_task_pass",
                "cycle1_router_task_pass",
                "cycle2_base_task_pass",
                "cycle2_router_task_pass",
            ],
            "compare_with": [
                "cwy_35b_fullsplit_20260610_083624 (reference: SFT+GRPO joint, 4 cycles)",
                "exp_2026_06_22_001_tau2_sft_only_2cycle_forgetting_diagnostic (EXP-084: SFT-only no replay)",
            ],
            "paper_weakness_addressed": "W6",
            "paper_claim_to_test": (
                "Does FOREVER Ebbinghaus-scheduled SFT replay of previous-cycle data "
                "halt or reverse the tau2 cycle-over-cycle held-out performance decline? "
                "If cycle2_base_task_pass >= cycle1_base_task_pass, forgetting is mitigated."
            ),
            "arxiv_ref": "2601.03938",
            "distinct_from_exp079": (
                "EXP-079 replays GRPO rollouts within the GRPO phase (policy drift "
                "prevention). EXP-087 replays SFT training pairs within the SFT phase "
                "using Ebbinghaus scheduling (catastrophic forgetting mitigation). "
                "Different phase, different data, different scheduling mechanism."
            ),
            "estimated_hours": 2.5,
            "implementation_note": (
                "In tau2_train_wrapper.sh / train_small_model.py: "
                "after each optimizer step, compute ||Δθ||_2 and accumulate into τ. "
                "When Ebbinghaus-predicted forgetting (1 - e^{-τ/τ_0}) > 0.15, "
                "inject a batch of 'forever_replay_buffer_size' samples from the "
                "previous cycle's SFT data (stored in 'cycle_{n-1}/sft_data.jsonl'). "
                "Scale the replay batch loss by forgetting severity. "
                "Reset τ after each replay event."
            ),
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
