#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-01 (EXP-100, EXP-101).

A800 connectivity: offline since 2026-05-14 (day 48). Apply when restored.

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

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_01.py            # EXP-100, EXP-101

Queue was ~99 pending on 2026-06-30 (+2 added: EXP-098, EXP-099).
Queue cap applied: >20 → max 2 new experiments.

Data motivating today's proposals
----------------------------------
results/e2e_4cyc_gpt55/cycle_3/e2e_ablation_summary.json:
    HumanEval 4-cycle:
      large (always-large):        task_pass=96.34%, cost_vs_large=100%
      skills (always-small+proc):  task_pass=75.61%, cost_vs_large=10%
      router (logistic, cycle 3):  task_pass=92.68%, routing_acc=92.68%,
                                   cost_vs_large=27.56%, fallback=6.10%
      full (router+GRPO):          task_pass=92.68% ← SAME as router
    Full = Router: GRPO contributes zero lift over the router.
    Cause: 52.4% ACR (advantage collapse rate) at G=8 DAPO — 43/82
    groups have zero within-group reward variance → zero gradient.
    Label distribution: 62 small-solvable / 20 large-only (75.6% easy).
    All results: single-seed. AAAI 2027 requires confidence intervals.

results/cwy_35b_joint_20260606_165203/cycle_3/e2e_ablation_summary.json:
    tau2 (cycle 3):
      always-small (base/skills):  task_pass=68.92%, cost_vs_large=10%
      router:                      task_pass=72.97%, routing_acc=89.19%,
                                   cost_vs_large=35.54%, fallback=6.76%
      full:                        task_pass=72.97% ← SAME as router
    All results: single-seed. AAAI 2027 requires confidence intervals on tau2 too.
    EXP-099 (added 2026-06-30) covers HumanEval 3-seed variance.
    Tau2 3-seed variance has no EXP assigned yet — today is the correct time.

arxiv:2605.17174 — 'Beyond Execution: Static-Analysis Rewards and
    Hint-Conditioned Diffusion RL for Code Generation' (May 2026):
    Demonstrates that static code analysis metrics (syntax validity, mypy type
    correctness, pylint style score) are the strongest execution-free reward signals
    for code-generation RL. The key finding: static-analysis rewards create
    within-group reward variation even in ALL-FAIL groups (where all G rollouts fail
    unit tests). Syntax failure, type-error, and lint-warning patterns vary across
    rollouts of the same failing task, providing discriminative structure the binary
    pass/fail reward cannot capture. Applied to our HumanEval ACR problem: the 43/82
    groups where all rollouts fail unit tests (ACR all-fail component) have zero binary
    reward variance, but their AST validity and type-check pass rate vary. Adding a
    static-analysis score S ∈ [0.0, 1.0] supplements binary reward without requiring
    any additional rollouts, synthetic samples, or gradient-weight changes.

AAAI 2027 context:
    Deadline: 2026-08-15 (45 days from today).
    EXP-099 (HumanEval 3-seed, added 2026-06-30): addresses HumanEval variance.
    Tau2 3-seed (EXP-100, today): addresses tau2 variance.
    Both are needed for Table 2 confidence intervals in the AAAI submission.
    Combined GPU budget: 6h for tau2 3-seed + 1.5h for static-analysis = 7.5h.
    These are the two highest-priority experiments for the AAAI paper.
"""
import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = (
    "/data0/home/zeyuwang/router-skills-evolve-data/humaneval/HumanEval.jsonl.gz"
)
_TAU2_DATA = (
    "/data0/home/zeyuwang/router-skills-evolve-data/tau2"
)

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_07_01_001_tau2_multiseed_variance_aaai",
        "priority": 9,
        "kind": "joint_cycle_multiseed",
        "rationale": (
            "DEFERRED ITEM NOW DUE: The 2026-06-30 ideas report explicitly deferred "
            "tau2 multi-seed variance because EXP-099 (HumanEval 3-seed) was just "
            "added that day, with the note 'Tau2 multi-seed is the logical follow-up "
            "once HumanEval variance is known.' It is now 2026-07-01 — one day "
            "later — and the AAAI 2027 deadline is 2026-08-15 (45 days). "
            ""
            "WHAT EXP-099 COVERS (added 2026-06-30): "
            "joint_cycle_multiseed on HumanEval, seeds=[42,123,999], 1 cycle, "
            "DAPO G=8, SCALING_FORCE_BOTH=1, SFT_INCLUDE_SUCCESS=1. Purpose: "
            "establish confidence intervals for HumanEval routing accuracy "
            "(92.68% single-seed) and task_pass_at_1 for AAAI Table 2. "
            ""
            "WHAT THIS EXP COVERS: "
            "The same 3-seed design on tau2_bench. Tau2 results are also "
            "single-seed: routing_acc=89.19%, task_pass=72.97%, "
            "cost_vs_large=35.54% at cycle 3. AAAI reviewers will compare "
            "HumanEval and tau2 arms side-by-side in Table 2; presenting CI "
            "for HumanEval but not tau2 would invite the comment 'why does "
            "only one benchmark show confidence intervals?' Tau2 variance is "
            "also independently valuable: the multi-turn repair setting (up to "
            "3 turns per task, with ~30% of tasks reaching turn 2) adds "
            "non-trivial stochasticity relative to the single-turn HumanEval "
            "setting, so tau2 CI may be wider than HumanEval CI. "
            ""
            "WHY MULTI-SEED ON TAU2 IS NOT DUPLICATIVE WITH EXP-099: "
            "EXP-099 runs on HumanEval adapter (local models + pytest reward). "
            "This experiment runs on tau2_bench adapter (remote agent API + "
            "FSDP2 SFT). The pipeline, data path, and reward function are "
            "different. Seeds control different sources of randomness: "
            "on HumanEval — GRPO rollout sampling and SFT data order; "
            "on tau2 — additionally the multi-turn API call order and the "
            "turn-level repair sampling. "
            ""
            "DISTINCTNESS FROM PRIOR QUEUE: "
            "EXP-093 (MDP-GRPO antiforgetting staircase): multi-seed but varies "
            "training objective across seeds (not fixed-config variance). "
            "EXP-099: HumanEval, not tau2. "
            "No queued experiment has joint_cycle_multiseed on tau2_bench with "
            "seeds=[42,123,999] and the canonical cwy_35b config. "
            ""
            "TIMELINE: "
            "tau2 1 cycle × 3 seeds ≈ 3 × 2h = 6 GPU-hours (serial on A800). "
            "If A800 restored 2026-07-01: results by 2026-07-02 morning. "
            "Leaves 44 days for Table 2 revision before AAAI 2026-08-15. "
        ),
        "spec": {
            "pipeline": "tau2_multiseed_variance_aaai",
            "bench": "tau2_bench",
            "base_model": "Qwen/Qwen3-4B-Instruct",
            "n_cycles": 1,
            "seeds": [42, 123, 999],
            "grpo_algo": "dapo",
            "dapo_dynamic_sampling": True,
            "n_generations": 8,
            "grpo_temperature": 1.0,
            "scaling_force_both": True,
            "sft_include_success": True,
            "variance_estimation": True,
            "aaai_variance_bound": True,
            "eval_data": _TAU2_DATA,
            "compare_with": [
                "cwy_35b_joint_20260606_165203 cycle_3 (single-seed tau2 reference: 72.97% task_pass, 89.19% routing_acc)",
                "exp_2026_06_30_002_multiseed_variance_dapo_1cycle_aaai (EXP-099: HumanEval 3-seed)",
            ],
            "primary_metrics": [
                "tau2_task_pass_mean_3seed",
                "tau2_task_pass_std_3seed",
                "tau2_routing_accuracy_mean_3seed",
                "tau2_routing_accuracy_std_3seed",
                "tau2_cost_vs_large_mean_3seed",
                "tau2_cost_vs_large_std_3seed",
                "tau2_fallback_rate_mean_3seed",
                "tau2_sft_pass_at_1_mean_3seed",
                "tau2_grpo_acr_mean_3seed",
            ],
            "target_outcome": (
                "tau2 routing accuracy CI: 89.19% ± σ (target σ < 3 pp for "
                "AAAI Table 2 credibility). "
                "tau2 task_pass CI: 72.97% ± σ (target σ < 2 pp). "
                "Result directly enables AAAI Table 2 tau2 column with mean ± std. "
                "If σ_tau2 >> σ_humaneval: flag multi-turn stochasticity as a "
                "limitation (multi-turn variance is a known issue in agentic RL). "
            ),
            "gpu": "auto",
            "estimated_gpu_hours": 6.0,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_07_01_002_static_analysis_reward_grpo_humaneval",
        "priority": 7,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2605.17174 — 'Beyond Execution: Static-Analysis Rewards and "
            "Hint-Conditioned Diffusion RL for Code Generation' (May 2026). "
            ""
            "ROOT PROBLEM: 52.4% ACR on HumanEval GRPO at cycle 3 (43/82 groups "
            "have zero within-group reward variance → zero gradient → Full = Router). "
            "The 43 collapsed groups decompose into: "
            "  (a) ALL-PASS groups: all G=8 rollouts pass all unit tests → R=1 for all "
            "  (b) ALL-FAIL groups: all G=8 rollouts fail at least one unit test → R=0 "
            "    for all (binary reward) "
            "The ACR mitigation queue so far: "
            "  EXP-090 (STARE): token-level surprisal reweighting on NON-COLLAPSED groups "
            "  EXP-094 (AVSPO): virtual sample injection into COLLAPSED groups "
            "  EXP-096: length penalty on binary reward (creates within-group spread via "
            "    token count variation) "
            "  EXP-098 (SA-AH-GRPO): entropy-adaptive gradient discount on failed rollouts "
            "MISSING attack vector: creating within-group reward spread from CODE STRUCTURE "
            "rather than token count (EXP-096) or gradient weights (EXP-098). "
            ""
            "WHAT STATIC-ANALYSIS REWARDS DO: "
            "arxiv:2605.17174 shows that three static analysis metrics are the strongest "
            "execution-free code quality signals: "
            "  (1) Syntax validity: does ast.parse() succeed? (binary, 0 or 1) "
            "  (2) Type correctness: does mypy --strict succeed? (binary, 0 or 1, but "
            "      error count normalizable: 1 - min(mypy_errors, 5) / 5) "
            "  (3) Lint quality: pylint score (0–10, normalized to [0, 1]) "
            "Combined as a weighted static-analysis score: "
            "  S(rollout) = 0.4 * syntax_valid + 0.3 * type_score + 0.3 * lint_score "
            "where each component is in [0, 1]. "
            ""
            "SUPPLEMENTED REWARD: "
            "  R_total = R_pass + beta * S(rollout) "
            "where: "
            "  R_pass ∈ {0, 1} — standard HumanEval binary pass/fail reward "
            "  S(rollout) ∈ [0, 1] — static analysis score "
            "  beta = 0.15 — static analysis weight (preserves pass/fail dominance: "
            "    passing solution always preferred over failing: 1+0.15*S_pass > "
            "    0+0.15*S_fail since 1 > 0.15) "
            ""
            "WHY THIS REDUCES ACR (ALL-FAIL GROUPS): "
            "In an all-fail group, all G=8 rollouts have R_pass=0. BUT their static "
            "analysis scores vary: some solutions are syntactically invalid (didn't "
            "parse → S=0.0), some parse but fail mypy (S≈0.4), some pass mypy but "
            "have high lint score (S≈0.7), some are near-perfect structurally except "
            "for one algorithmic error (S≈0.9). HumanEval solutions from a 1.5B model "
            "at cycle 0 have realistic variance in type correctness and lint quality "
            "(~30% of solutions fail mypy; ~40% have lint warnings). The within-group "
            "reward variance in an all-fail group becomes: "
            "  Var(R_total) = beta^2 * Var(S) = 0.0225 * Var(S) "
            "If Var(S) ≈ 0.04 (std(S) ≈ 0.2 across G=8 rollouts), then "
            "  Var(R_total) ≈ 0.0225 * 0.04 = 0.0009 "
            "This is small but nonzero — enough to produce non-trivial GRPO advantages "
            "that guide the model toward structurally better failed solutions. "
            ""
            "WHY THIS REDUCES ACR (ALL-PASS GROUPS): "
            "In an all-pass group, all G=8 rollouts have R_pass=1. Their static scores "
            "also vary (a concise, lint-clean passing solution gets S≈0.9; a verbose, "
            "lint-noisy passing solution gets S≈0.6). This supplements EXP-096's "
            "length penalty: length penalizes verbose solutions; static analysis "
            "penalizes structurally noisy solutions independent of length. Together "
            "they provide two orthogonal within-group differentiating signals. "
            "(This experiment does NOT combine them — beta replaces alpha from EXP-096 "
            "so we can isolate the static-analysis signal.) "
            ""
            "ORTHOGONALITY TO ALL QUEUED ACR APPROACHES: "
            "EXP-090 (STARE): operates on existing advantages after reward → unchanged "
            "  by the reward function (STARE still sees binary R; no within-group spread "
            "  in collapsed groups). Static analysis creates the spread BEFORE STARE. "
            "EXP-094 (AVSPO): injects virtual samples with synthetic rewards. Static "
            "  analysis changes real rewards for existing rollouts. "
            "EXP-096: length penalty (token count) as reward supplement. Static analysis "
            "  is code quality, not length — orthogonal signal. "
            "EXP-098 (SA-AH-GRPO): entropy discount on gradient weights. No interaction "
            "  with static-analysis reward design. "
            "EXP-086 (VeRPO, 2601.03525): test-difficulty weighting (changes WHICH tests "
            "  count toward binary reward). Static analysis creates a CONTINUOUS reward "
            "  component independent of test-level results. "
            ""
            "DISTINCTNESS FROM PRIOR QUEUE: "
            "No queued experiment has static_analysis_reward=True, mypy, pylint, or "
            "  ast.parse fields. "
            "The static-analysis reward is the only approach in the queue that creates "
            "  within-group spread from CODE STRUCTURE independent of test execution. "
            ""
            "IMPLEMENTATION: "
            "In grpo_train_simple.py or a reward-wrapper: "
            "  import ast, subprocess "
            "  def static_score(code_str): "
            "      try: ast.parse(code_str); syntax = 1.0 "
            "      except SyntaxError: return 0.0  # all other scores = 0 "
            "      mypy_result = subprocess.run(['mypy', '--no-error-summary', ...]) "
            "      type_score = max(0, 1.0 - mypy_result.error_count / 5) "
            "      pylint_score = run_pylint_score(code_str) / 10.0 "
            "      return 0.4 * syntax + 0.3 * type_score + 0.3 * pylint_score "
            "  reward = float(tests_pass) + BETA * static_score(solution) "
            "Overhead: ~50ms per rollout (mypy + pylint on short HumanEval solutions). "
            "G=8 × 50ms × 82 tasks = ~33 seconds overhead per cycle (negligible). "
            ""
            "EXPECTED OUTCOME: "
            "ACR reduced from 52.4% toward 35-45% (all-fail groups now have nonzero "
            "  within-group static-analysis variance). "
            "HumanEval pass@1 improves by 1-3 pp if the model learns to write "
            "  syntactically cleaner, type-correct solutions under the static signal. "
            "Average mypy error rate decreases (model trained to write type-correct "
            "  code even when solutions are wrong on tests). "
            "Wall-clock: 1 cycle × ~1.5h + ~33s overhead = ~1.5 GPU-hours total. "
        ),
        "spec": {
            "pipeline": "humaneval_static_analysis_reward",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 1,
            "grpo_algo": "dapo",
            "reward_fn": "pass_plus_static_analysis",
            "static_analysis_reward": True,
            "static_analysis_components": ["syntax", "mypy", "pylint"],
            "static_analysis_weights": {"syntax": 0.4, "mypy": 0.3, "pylint": 0.3},
            "static_analysis_beta": 0.15,
            "dapo_dynamic_sampling": True,
            "n_generations": 8,
            "grpo_temperature": 1.0,
            "eval_data": _HE_EVAL,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (DAPO binary reference: 92.68% HE, ACR=52.4%)",
                "exp_2026_06_29_001_length_penalty_reward_acr_humaneval (EXP-096: beta=length)",
                "exp_2026_06_26_001_stare (EXP-090: token-level surprisal reweighting)",
                "exp_2026_06_28_001_avspo (EXP-094: virtual sample injection)",
            ],
            "primary_metrics": [
                "humaneval_pass_at_1_post_1cycle",
                "grpo_acr_mean",
                "grpo_zero_variance_group_rate",
                "grpo_within_group_reward_variance_mean",
                "solution_mypy_error_rate_mean",
                "solution_pylint_score_mean",
                "solution_syntax_validity_rate",
                "static_analysis_score_std_within_group",
            ],
            "target_outcome": (
                "ACR <= 0.42 (from 0.524 baseline; all-fail groups gain nonzero "
                "within-group static-analysis variance). "
                "HE pass@1 >= 0.84 after 1 cycle. "
                "mypy error rate decreases vs. binary-reward baseline (structural "
                "quality incentive active). "
            ),
            "gpu": "auto",
            "estimated_gpu_hours": 1.5,
        },
        "gpu": "auto",
    },
]


def atomic_write(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


def main() -> None:
    if not STATE_PATH.exists():
        raise FileNotFoundError(f"state.json not found at {STATE_PATH}")

    with STATE_PATH.open() as fh:
        state = json.load(fh)

    queue = state.setdefault("queue", [])
    existing_ids = {e["id"] for e in queue}

    added = []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] in existing_ids:
            print(f"[skip] {exp['id']} already in queue")
            continue
        queue.append(exp)
        existing_ids.add(exp["id"])
        added.append(exp["id"])
        print(f"[add]  {exp['id']} (priority={exp['priority']}, kind={exp['kind']})")

    atomic_write(STATE_PATH, state)
    print(f"\nDone. Added {len(added)} experiments: {added}")
    print(f"Total queue size: {len(queue)}")


if __name__ == "__main__":
    main()
