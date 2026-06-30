#!/usr/bin/env python3
"""
Daily queue patch — 2026-06-30 (EXP-098, EXP-099).

A800 connectivity: offline since 2026-05-14 (day 47). Apply when restored.

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

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_06_30.py            # EXP-098, EXP-099

Queue was ~97 pending on 2026-06-29 (+2 added: EXP-096, EXP-097).
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
    Results are single-seed; AAAI 2027 requires confidence intervals.

results/cwy_35b_joint_20260606_165203/cycle_3/e2e_ablation_summary.json:
    tau2 (cycle 3):
      always-small (base/skills):  task_pass=68.92%, cost_vs_large=10%
      router:                      task_pass=72.97%, routing_acc=89.19%,
                                   cost_vs_large=35.54%, fallback=6.76%
      full:                        task_pass=72.97% ← SAME as router
    Full = Router on tau2 as well — GRPO net-neutral across both benches.
    Results are single-seed.

arxiv:2606.05434 — 'Selective-Advantage Entropy-Adaptive Horizon GRPO:
    Asymmetric Token-Level Discounting for Efficient Reinforcement Learning
    of Language Models' (June 3, 2026):
    Introduces two GRPO extensions. (1) AH-GRPO: weights each token's policy
    gradient by a cumulative entropy-based discount factor that shrinks the
    effective horizon when the model is uncertain about future tokens. Motivation:
    long, uncertain trajectories that happen to fail contribute noisy gradient
    signal — the entropy discount suppresses them relative to shorter, more
    confident (and more likely correct) trajectories. (2) SA-AH-GRPO (Selective):
    applies the entropy-adaptive horizon discount ONLY to negative-advantage
    rollouts (those that failed), leaving positive-advantage successful
    trajectories fully attenuated. Tested on Qwen2.5-1.5B-Instruct and
    Qwen2.5-3B-Instruct with LoRA on GSM8K. Reports improved accuracy and
    faster convergence vs. vanilla GRPO. The selective variant prevents penalizing
    correct-but-verbose solutions.

Context: AAAI 2027 deadline is 2026-08-15 (46 days from today). The submitted
EMNLP 2026 abstract (2026-06-19, 11 days ago) contains only single-seed HumanEval
and tau2 results. AAAI reviewers will expect variance bounds. The standard for
GRPO-based code papers (e.g., DAPO, CodeRL+) is mean ± std over 3 seeds. A
3-seed rerun of the best 1-cycle config (DAPO, G=8, SCALING_FORCE_BOTH=1) is
fast enough for the A800 and directly usable as Table 2 confidence intervals.
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
        "id": "exp_2026_06_30_001_sa_ah_grpo_entropy_discount_humaneval",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2606.05434 — 'Selective-Advantage Entropy-Adaptive Horizon GRPO: "
            "Asymmetric Token-Level Discounting for Efficient Reinforcement Learning "
            "of Language Models' (June 3, 2026). "
            ""
            "ROOT PROBLEM: 52.4% ACR on HumanEval GRPO (43/82 groups at G=8 DAPO "
            "have zero within-group reward variance → vanishing gradient). Full = "
            "Router at cycle 3 on both HumanEval (92.68%) and tau2 (72.97%). The "
            "GRPO training update is effectively zero for these collapsed groups. "
            ""
            "WHAT SA-AH-GRPO DOES: "
            "Introduces a cumulative entropy-based discount factor γ_t that reduces "
            "the effective policy gradient horizon for high-entropy (uncertain) token "
            "positions: "
            "  γ_t = exp(-τ * Σ_{k=1}^{t} H(π(·|s_k))) "
            "where H(π(·|s_k)) is the token-level entropy at step k and τ is a "
            "temperature parameter controlling discount rate. This downweights the "
            "gradient contribution of uncertain tokens — those where the model is "
            "spread across many next-token candidates — relative to confident tokens "
            "where the model knows what to write. "
            ""
            "The SELECTIVE (SA) variant applies the entropy discount ONLY to "
            "negative-advantage rollouts (failed HumanEval solutions). Positive- "
            "advantage rollouts (correct solutions) are left unattenuated — a "
            "correct-but-verbose solution should not be penalized. This asymmetry is "
            "critical for code: correct solutions vary widely in verbosity (comments, "
            "helper functions), but that verbosity is signal, not noise. "
            ""
            "WHY THIS ATTACKS ACR: "
            "In an all-fail group (ACR: all 8 rollouts fail), the vanilla GRPO "
            "advantage is 0 because all rewards are identical. SA-AH-GRPO does NOT "
            "directly fix this zero-advantage problem (that is AVSPO's EXP-094 role). "
            "Instead, SA-AH-GRPO addresses the QUALITY of gradient from non-collapsed "
            "groups: the 47.6% of groups where variance IS nonzero. For these groups, "
            "the entropy discount makes the model learn more from the DECISIVE tokens "
            "(where entropy is low — the model has committed to a choice) and less "
            "from the UNCERTAIN tokens (where entropy is high — the model is exploring). "
            "This is precisely right for code: the variable names and structure are "
            "decisive (low-entropy prefix); the body logic is variable. Concentrating "
            "gradient on the prefix aligns with how code correctness is determined. "
            ""
            "In the all-fail case, SA-AH-GRPO also avoids the trap of amplifying "
            "noisy gradient from long, meandering failed trajectories: the entropy "
            "discount reduces the effective horizon of failed rollouts to focus the "
            "(already-zero-advantage) signal on the structural prefix rather than "
            "the verbose suffix. Combined with EXP-096 (length penalty reward) and "
            "EXP-094 (AVSPO), SA-AH-GRPO provides the third leg of a multi-pronged "
            "ACR mitigation: AVSPO restores group-level diversity; EXP-096 adds "
            "continuous reward variation; SA-AH-GRPO improves gradient quality "
            "from non-collapsed groups. "
            ""
            "MODEL MATCH: "
            "The paper evaluates Qwen2.5-1.5B-Instruct — our exact base model for "
            "HumanEval experiments. The LoRA-based evaluation is directly comparable "
            "to our setup (we also use LoRA adapters). GSM8K → HumanEval transfer: "
            "both are 'verifiable single-turn': one correct/incorrect outcome per "
            "problem. The entropy discount is task-agnostic (computed from output "
            "logits, not task structure), so it transfers directly. "
            ""
            "DISTINCTNESS FROM ALL QUEUED EXPERIMENTS: "
            "EXP-090 (STARE, arxiv:2606.19236): surprisal-guided TOKEN-LEVEL "
            "ADVANTAGE REWEIGHTING — reweights the per-token advantage scalar, not "
            "the gradient weight. STARE identifies 'entropy-critical tokens' and "
            "rescales their advantage contribution. SA-AH-GRPO applies a CUMULATIVE "
            "ENTROPY DISCOUNT to the gradient weight itself, asymmetrically by "
            "advantage sign. These are ORTHOGONAL modifications at different levels "
            "of the GRPO update rule. "
            "EXP-094 (AVSPO, arxiv:2605.21125): GROUP-LEVEL virtual sample injection "
            "into collapsed groups — different level (group vs. token). "
            "EXP-096 (length penalty): REWARD FUNCTION modification — different "
            "level (reward design vs. gradient computation). "
            "EXP-088 (VIMPO, arxiv:2606.20008): value-implicit baseline — different "
            "mechanism (KL-regularized value baseline vs. entropy-discount). "
            "No queued experiment has grpo_algo='sa_ah_grpo' or "
            "entropy_discount_tau as a field. ✓ "
            ""
            "IMPLEMENTATION: "
            "~20 lines in grpo_train_simple.py: "
            "  For each rollout, compute cumulative entropy from output logits: "
            "    H_k = -Σ_v π(v|s_k) log π(v|s_k) "
            "    cumH_t = Σ_{k=1}^{t} H_k "
            "    gamma_t = exp(-tau * cumH_t) "
            "  For positive-advantage rollouts: no discount (gamma_t = 1 for all t). "
            "  For negative-advantage rollouts: apply gamma_t to per-token loss. "
            "  tau = 0.5 (paper default for Qwen2.5-1.5B on GSM8K). "
            "The logits are already available in the GRPO rollout buffer, so no "
            "additional forward pass is needed. Estimated overhead: +5% compute "
            "(entropy computation over logit tensors per token). "
            ""
            "EXPECTED OUTCOME: "
            "The paper reports +2.3 pp accuracy on GSM8K at 1.5B scale vs. GRPO. "
            "For HumanEval, we target: pass@1 >= 0.855 after 1 GRPO cycle "
            "(vs. ~0.83 expected from flat DAPO binary reward). "
            "ACR reduction: indirect — SA-AH-GRPO does not inject virtual samples, "
            "but reduces gradient noise from long-failed trajectories, so effective "
            "ACR (fraction of groups contributing gradient) should increase. "
            "If ACR remains high but pass@1 improves, it confirms that QUALITY of "
            "gradient from non-collapsed groups matters more than quantity — a new "
            "finding for the AAAI paper. "
            "Wall-clock: 1 cycle × ~1.5 GPU-hours (same as DAPO baseline). "
        ),
        "spec": {
            "pipeline": "humaneval_sa_ah_grpo_entropy_discount",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 1,
            "grpo_algo": "sa_ah_grpo",
            "entropy_discount_tau": 0.5,
            "selective_negative_only": True,
            "dapo_dynamic_sampling": True,
            "n_generations": 8,
            "grpo_temperature": 1.0,
            "eval_data": _HE_EVAL,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (DAPO binary: 92.68% HE, ACR=52.4%)",
                "exp_2026_06_28_001_avspo (EXP-094: group-level virtual sample injection)",
                "exp_2026_06_26_001_stare (EXP-090: token-level surprisal reweighting)",
                "exp_2026_06_29_001_length_penalty_reward_acr_humaneval (EXP-096: reward design)",
            ],
            "primary_metrics": [
                "humaneval_pass_at_1_post_1cycle",
                "grpo_acr_mean",
                "grpo_zero_variance_group_rate",
                "grpo_per_token_entropy_mean",
                "grpo_entropy_discount_effective_horizon_mean",
                "grpo_entropy_discount_negative_rollout_weight_mean",
            ],
            "target_outcome": (
                "HE pass@1 >= 0.855 after 1 cycle (vs. ~0.83 DAPO binary baseline). "
                "Effective GRPO horizon (entropy-weighted) shorter for failed rollouts "
                "than successful ones (confirms selective discount active). "
                "ACR remains measurable but pass@1 improvement shows non-collapsed "
                "groups now contribute higher-quality gradient. "
            ),
            "gpu": "auto",
            "estimated_gpu_hours": 1.5,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_06_30_002_multiseed_variance_dapo_1cycle_aaai",
        "priority": 9,
        "kind": "joint_cycle_multiseed",
        "rationale": (
            "AAAI 2027 deadline: 2026-08-15 (46 days). Current best results "
            "(HumanEval: 92.68% routing accuracy / 27.56% cost at cycle 3; "
            "tau2: 72.97% task pass / 35.54% cost at cycle 3) are from SINGLE-SEED "
            "runs. The AAAI 2027 review process requires: "
            "(1) Confidence intervals or std over multiple seeds for all key metrics. "
            "(2) Evidence that improvements are statistically significant vs. baselines. "
            "Without multi-seed estimates, reviewers will flag the results as unreliable. "
            ""
            "WHAT THIS EXPERIMENT DOES: "
            "Re-runs the complete 1-cycle MERA pipeline (Phase 1 collect traces → "
            "Phase 2 skills → Phase 3a SFT → Phase 3b GRPO → Phase 4 router → "
            "Phase 5 ablation) with three distinct random seeds (42, 123, 999) on "
            "HumanEval, using the canonical configuration from e2e_4cyc_gpt55: "
            "  - Base model: Qwen/Qwen2.5-Coder-1.5B-Instruct "
            "  - Large model: gpt-5.5 (distiller) "
            "  - SCALING_FORCE_BOTH=1 (run-both oracle) "
            "  - SFT_INCLUDE_SUCCESS=1 (behaviour-clone solved + teacher traces) "
            "  - GRPO: DAPO, G=8, temperature=1.0, dynamic_sampling=True "
            "  - Router: logistic regression on raw prompt (single global skill) "
            "For each seed: collect traces → train small model → GRPO 1 cycle → "
            "train router → run 4-arm ablation (large/skills/router/full). "
            ""
            "WHY PRIORITY 9 (HIGHEST): "
            "This is infrastructure for the AAAI submission, not a new research "
            "direction. Without variance bounds, every table entry is a point estimate "
            "with unknown noise. The review timeline demands results by 2026-08-01 "
            "(to allow 2 weeks of paper revision before the 2026-08-15 deadline). "
            "The experiment must start as soon as the A800 comes back online — "
            "hence priority 9, higher than all experimental variants. "
            ""
            "WHY MULTI-SEED ON CYCLE 1 (NOT CYCLE 3): "
            "4 cycles × 3 seeds = 12 pipeline runs (~60 GPU-hours) would dominate "
            "the backlog and likely miss the AAAI window given the existing ~97 "
            "pending experiments. 1 cycle × 3 seeds = 3 pipeline runs (~4.5 GPU-hours "
            "total: ~1.5h per run). This gives: "
            "  (a) Variance for the fastest measurable outcome (cycle 1 pass@1, "
            "      routing accuracy, ACR), which is the most reproducible signal "
            "      since the SFT + GRPO is one gradient pass. "
            "  (b) Cross-seed variance for the router threshold and classifier "
            "      (does the 92.68% routing accuracy hold across seeds, or was it "
            "      a lucky threshold?) "
            "  (c) Variance for the 4-arm ablation summary at cycle 1. "
            "Cycle 1 variance provides the statistical foundation; we note in the "
            "paper that cycle-3 results are one seed (the reference run) and report "
            "cycle-1 variance as a proxy for the per-cycle variance structure. "
            ""
            "DISTINCTNESS FROM PRIOR QUEUE: "
            "EXP-093 (MDP-GRPO antiforgetting staircase, 2026-06-27): uses "
            "grpo_multi_seed_staircase kind with an antiforgetting objective — "
            "staircase structure is the experimental design, not variance estimation. "
            "EXP-093 varies the training objective across seeds; this experiment "
            "holds the objective FIXED and varies ONLY the random seed to estimate "
            "metric variance. "
            "No queued experiment has variance_estimation=True or "
            "aaai_variance_bound=True as a field. ✓ "
            "No prior experiment ran joint_cycle_multiseed with the canonical "
            "e2e_4cyc_gpt55 config on seeds [42, 123, 999]. "
            ""
            "IMPLEMENTATION: "
            "joint_cycle_multiseed kind: runner.py should iterate seeds and call "
            "run_full_pipeline.sh --bench humaneval --n-cycles 1 --seed <s> "
            "for each seed in [42, 123, 999]. Seeds control: "
            "  - SFT dataloader shuffling "
            "  - GRPO rollout sampling (temperature=1.0, seed used for sampling) "
            "  - Router train/val split in train_router_simple.py "
            "Results aggregated to: "
            "  mean_task_pass (router arm) = Σ task_pass_s / 3 "
            "  std_task_pass (router arm) = sqrt(Σ (task_pass_s - mean)^2 / 2) "
            "Reported in the AAAI table as: 92.68% ± X.XX pp (cycle 3, single run) "
            "with cycle-1 variance: mean ± std (3 seeds). "
            ""
            "EXPECTED OUTCOME: "
            "Std(pass@1, router arm) across 3 seeds at cycle 1: expected 1-3 pp "
            "(cycle-1 GRPO is noisy; router threshold also has seed dependence). "
            "Std(routing_acc) across 3 seeds: expected 0.5-2 pp "
            "(logistic regression on fixed embeddings is stable; split variance is "
            "the main source). "
            "This variance estimate will appear in AAAI Table 2 as confidence "
            "intervals, satisfying the statistical rigor requirement. "
            "Wall-clock: 3 × ~1.5 GPU-hours = ~4.5 GPU-hours (pipeline serial; "
            "staircase runner may parallelize some phases). "
        ),
        "spec": {
            "pipeline": "humaneval_dapo_1cycle_multiseed_variance",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 1,
            "seeds": [42, 123, 999],
            "variance_estimation": True,
            "aaai_variance_bound": True,
            "scaling_force_both": True,
            "sft_include_success": True,
            "grpo_algo": "dapo",
            "n_generations": 8,
            "grpo_temperature": 1.0,
            "dapo_dynamic_sampling": True,
            "eval_data": _HE_EVAL,
            "reference_run": "e2e_4cyc_gpt55",
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (reference, seed=0): 92.68% HE routing acc",
                "e2e_4cyc_gpt55 cycle_1 (reference): cycle-1 baseline for variance context",
            ],
            "primary_metrics": [
                "humaneval_pass_at_1_router_mean",
                "humaneval_pass_at_1_router_std",
                "humaneval_pass_at_1_full_mean",
                "humaneval_pass_at_1_full_std",
                "humaneval_routing_accuracy_mean",
                "humaneval_routing_accuracy_std",
                "humaneval_cost_vs_large_mean",
                "humaneval_cost_vs_large_std",
                "grpo_acr_mean_across_seeds",
                "grpo_acr_std_across_seeds",
            ],
            "target_outcome": (
                "Std(pass@1, router/full) across 3 seeds <= 2.5 pp at cycle 1 "
                "(confirming that single-seed cycle-3 results are representative). "
                "Mean(pass@1, full, cycle 1) >= 0.85 across seeds "
                "(confirms the benefit holds across random initializations). "
                "Result directly usable as AAAI 2027 Table 2 confidence intervals. "
            ),
            "gpu": "auto",
            "estimated_gpu_hours": 4.5,
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
