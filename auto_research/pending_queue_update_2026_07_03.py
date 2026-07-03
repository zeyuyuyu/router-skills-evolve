#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-03 (EXP-104, EXP-105).

A800 connectivity: offline since 2026-05-14 (day 50). Apply when restored.

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

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_03.py            # EXP-104, EXP-105

Queue was ~107 pending on 2026-07-02 (+2 added: EXP-102, EXP-103).
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
    Root problem: ACR=52.4% — 43/82 GRPO groups have zero within-group reward
    variance → zero gradient → Full = Router.

arxiv:2507.05386 — Reinforcement Fine-Tuning Naturally Mitigates Forgetting in
    Continual Post-Training (July 2026):
    Compares SFT vs. RFT (GRPO) across 7 multimodal continual learning tasks.
    Key finding: SFT leads to catastrophic forgetting; RFT inherently preserves
    prior task knowledge and matches multi-task training without explicit replay.
    Mechanism: RFT's implicit regularization effect is theoretically grounded —
    updates are CONSERVATIVE IN PARAMETER SUBSPACES SENSITIVE TO PRIOR TASKS,
    with conservatism NATURALLY SCALED BY THE VARIANCE OF THE REWARD SIGNAL.
    Explicit mechanisms (KL penalty, CoT) contribute but are not the primary
    driver — the variance-scaling of the implicit regularization is.
    Critical implication for our project: in zero-variance GRPO groups (ACR=52.4%),
    reward variance = 0 → conservatism is maximal → NO gradient update happens
    for those task-parameter subspaces. This means: (1) GRPO neither improves NOR
    forgets on zero-variance tasks — it is a pure no-op for them; (2) the 52.4%
    collapsed groups are wasting rollout compute with zero training signal; (3)
    the paper implies that selectively masking zero-variance groups (skipping their
    gradient contribution) would not change training dynamics — they already
    contribute nothing — but WOULD reveal whether non-collapsed groups (47.6%)
    alone provide enough improvement signal to push Full > Router.

arxiv:2507.20673 — Geometric-Mean Policy Optimization (GMPO, July 2025):
    Identifies that standard GRPO optimizes the ARITHMETIC MEAN of token-level
    importance-weighted rewards, making it vulnerable to outlier tokens with
    extreme importance sampling ratios (|π_θ/π_ref| >> 1 or << 1).
    GMPO substitutes the geometric mean: instead of mean(w_t * A), compute
    exp(mean(log(w_t * A + ε))) for positive advantage, with sign preservation
    for negative advantage rollouts.
    Geometric mean is inherently less sensitive to outliers and maintains a more
    stable range of importance sampling ratios throughout training.
    Results: GMPO-7B outperforms GRPO by +4.1% on math benchmarks (AIME24,
    AMC, MATH500, OlympiadBench) and +1.4% on multimodal reasoning.
    Relevance to our project: HumanEval code generation has a specific outlier
    pattern — syntax-critical tokens (closing brackets, indentation, `return`
    statements) often diverge between rollouts precisely at the positions where
    correctness is decided. When a rollout diverges at a critical syntactic token,
    the importance ratio |π_θ/π_ref| at that token can be extremely large.
    Under arithmetic-mean GRPO, this single token's outlier importance ratio
    dominates the group-level gradient, destabilizing the policy update.
    Geometric mean (GMPO) bounds this influence: log(extreme_ratio) is still
    large but does not swamp the average. This is ORTHOGONAL to ACR mitigation
    (which targets zero-variance groups) — GMPO targets high-variance, non-collapsed
    groups where outlier tokens inflate gradient variance.
    In our setting, non-collapsed groups (47.6% of tasks) are where all GRPO
    learning happens. GMPO could make these updates more stable and effective.

AAAI 2027 context:
    Deadline: 2026-08-15 (43 days from today).
    EXP-104 (Selective-Group GRPO, today): tests whether non-collapsed groups alone
    provide enough GRPO signal to push Full > Router. Directly diagnostic for the
    Full=Router bottleneck described in our AAAI analysis section.
    EXP-105 (GMPO, today): drop-in GRPO variant that stabilizes the 47.6% of
    groups that DO provide learning signal, potentially improving HE pass@1 on
    the non-collapsed subset.
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
        "id": "exp_2026_07_03_001_selective_group_grpo_rft_forgetting_humaneval",
        "priority": 7,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2507.05386 — Reinforcement Fine-Tuning Naturally Mitigates "
            "Forgetting in Continual Post-Training (July 2026). "
            ""
            "KEY FINDING FROM PAPER: RFT's implicit regularization conservatism is "
            "SCALED BY REWARD VARIANCE. In zero-variance groups (all-fail or all-pass): "
            "variance=0 → conservatism=max → gradient=0 → model unchanged for those "
            "tasks. In positive-variance groups: conservatism proportional to variance "
            "→ model updated in a way that naturally preserves prior task subspaces. "
            ""
            "CONSEQUENCE FOR OUR PIPELINE: ACR=52.4% means 43/82 groups are zero-variance. "
            "Per the paper's theory, these 43 groups contribute NOTHING to GRPO training "
            "— neither learning nor forgetting. They are pure compute waste. The 39/82 "
            "non-collapsed groups (47.6%) are where ALL GRPO learning happens. "
            ""
            "EXPERIMENT DESIGN — Selective-Group GRPO: "
            "Hypothesis: if zero-variance groups contribute zero gradient anyway, then "
            "masking them (skipping their loss contribution) should have NEAR-ZERO "
            "effect on training dynamics. If true: training on only non-collapsed groups "
            "is equivalent to standard GRPO but uses 52.4% fewer backward passes, "
            "reducing memory pressure and potentially allowing larger effective batch size "
            "or more epochs on the informative 47.6%. "
            "If false (selective masking HELPS): the zero-variance groups were actually "
            "contributing a net-negative signal (e.g., gradient noise from numeric "
            "imprecision in collapsed groups, or the normalization denominator being "
            "near-zero causing gradient explosion in adjacent groups). Selective masking "
            "would then reveal the clean signal from non-collapsed groups. "
            ""
            "IMPLEMENTATION: "
            "In grpo_train_simple.py, after computing group advantages, add a mask: "
            "  reward_var = rollout_rewards.var(dim=1, keepdim=True)  # [n_groups, 1] "
            "  active_mask = (reward_var > 1e-6).float()              # skip zero-var "
            "  loss = (grpo_loss * active_mask).sum() / (active_mask.sum() + 1e-8) "
            "This is a 3-line change. Run 1 cycle on HumanEval with G=8, "
            "SCALING_FORCE_BOTH=1, same hyperparams as e2e_4cyc_gpt55 cycle_3. "
            ""
            "VARIANTS (same run, reported separately): "
            "  (A) selective_mask_zero_var: mask ACR groups (reward_var < 1e-6) "
            "  (B) selective_mask_allpass: mask all-pass groups only (all R=1), "
            "      keep all-fail groups (GRPO still penalizes consistently failing tasks) "
            "  (C) standard DAPO baseline (reproduce cycle_3 ACR=52.4%) "
            "All three share the same rollouts (collect once, reuse). "
            ""
            "EXPECTED OUTCOMES: "
            "If (A) matches baseline: confirms zero-variance groups contribute nothing. "
            "If (A) improves over baseline: zero-variance groups were adding noise. "
            "Either way, ACR becomes trivially 0% (mask eliminates zero-var groups "
            "from loss, so ACR-in-loss = 0%). True metric: HE pass@1 and whether "
            "the gradient norm on non-collapsed groups increases with the freed budget. "
            "Target: HE pass@1 >= 0.93 for variant (A) (Full > Router for first time). "
        ),
        "spec": {
            "pipeline": "humaneval_selective_group_grpo",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 1,
            "grpo_algo": "dapo_selective_group_mask",
            "selective_mask_threshold": 1e-6,
            "selective_mask_mode": "zero_variance",
            "variants": [
                "selective_mask_zero_var",
                "selective_mask_allpass_only",
                "dapo_baseline",
            ],
            "n_generations": 8,
            "grpo_temperature": 1.0,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (DAPO binary reference: 92.68% HE, ACR=52.4%)",
                "arxiv:2507.05386 (RFT conservatism scaled by reward variance)",
            ],
            "primary_metrics": [
                "humaneval_pass_at_1_post_1cycle",
                "grpo_acr_mean",
                "grpo_active_group_fraction",
                "grpo_gradient_norm_active_groups",
                "grpo_gradient_norm_all_groups_baseline",
                "variant_delta_pass_at_1",
            ],
            "target_outcome": (
                "Selective-mask variant (A) matches or exceeds DAPO baseline pass@1. "
                "If gradient norm on active groups increases, confirms freed compute "
                "per group improves signal quality. "
                "If pass@1 > 0.93 in any variant, Full > Router achieved for AAAI. "
            ),
            "gpu": "auto",
            "estimated_gpu_hours": 1.5,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_07_03_002_gmpo_geometric_mean_policy_optimization_humaneval",
        "priority": 6,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2507.20673 — Geometric-Mean Policy Optimization (GMPO, "
            "Yuzhong Zhao, Yue Liu, Junpeng Liu et al., July 2025). "
            ""
            "PROBLEM ADDRESSED: Standard GRPO/DAPO optimizes the ARITHMETIC MEAN "
            "of token-level importance-weighted rewards across a sequence: "
            "  L_GRPO = mean_t [ clip(π_θ/π_ref, 1-ε, 1+ε) * A_rollout ] "
            "This arithmetic mean is dominated by outlier tokens with extreme "
            "importance ratios |π_θ(a_t|ctx) / π_ref(a_t|ctx)| >> 1. "
            ""
            "GMPO SOLUTION: Replace arithmetic mean with geometric mean: "
            "  For A > 0: L_GMPO = exp( mean_t[ log(clip(r_t, 1-ε, 1+ε) * A) ] ) "
            "  For A < 0: L_GMPO = -exp( mean_t[ log(clip(r_t, 1-ε, 1+ε) * |A|) ] ) "
            "The log-mean is inherently less sensitive to outliers — a token with "
            "r_t=10 contributes log(10)=2.3 rather than 10 to the average. "
            ""
            "RELEVANCE TO CODE GENERATION (HUMANEVAL): "
            "HumanEval code has a specific outlier pattern: "
            "  (1) SYNTACTIC PIVOTS: tokens like `return`, `:`, `\\n    ` (indentation) "
            "      often differ between rollouts that pass vs. fail tests. When a "
            "      rollout correctly indents where another does not, π_θ(correct_indent) "
            "      may be much higher than π_ref(correct_indent) after SFT → large r_t. "
            "  (2) RARE-BUT-CORRECT TOKENS: mathematical operators, specific library "
            "      calls (e.g., `bisect.insort`) that the small model rarely uses but "
            "      that solve edge cases — when these appear in a correct rollout, "
            "      their π_θ/π_ref can be extreme. "
            "Under arithmetic GRPO, these outlier tokens cause gradient spikes that "
            "push the policy hard on single tokens rather than globally improving "
            "code structure. Result: the policy learns to insert specific high-reward "
            "tokens rather than improving overall code generation competence. "
            "GMPO's geometric mean bounds this: log(extreme_ratio) is large but "
            "finite, distributing gradient signal across all tokens rather than "
            "concentrating it on outliers. "
            ""
            "RELATIONSHIP TO ACR PROBLEM: "
            "GMPO does NOT directly address ACR (zero-variance groups). It targets "
            "the COMPLEMENTARY problem: instability in the 47.6% of NON-COLLAPSED "
            "groups where GRPO IS receiving gradient signal. If these non-collapsed "
            "groups have outlier tokens causing gradient instability, GMPO stabilizes "
            "them and potentially increases their effective learning contribution. "
            "Combined analysis: if EXP-104 shows selective masking frees up compute "
            "for non-collapsed groups, and EXP-105 shows GMPO stabilizes those groups, "
            "then EXP-104+EXP-105 combined could be the AAAI System Improvement section. "
            ""
            "IMPLEMENTATION: "
            "In grpo_train_simple.py, replace the per-token loss aggregation: "
            "  # Standard DAPO: "
            "  loss = -(ratio * advantages).mean(dim=-1)  # arithmetic mean over tokens "
            "  # GMPO replacement: "
            "  pos_mask = (advantages > 0).float() "
            "  neg_mask = (advantages <= 0).float() "
            "  # Geometric mean for positive-advantage rollouts "
            "  log_ratios = torch.log(ratio.clamp(min=1e-8)) "
            "  geo_pos = torch.exp((log_ratios + torch.log(advantages.abs().clamp(1e-8))) "
            "                      .mean(dim=-1)) * pos_mask.squeeze(-1) "
            "  geo_neg = -torch.exp((log_ratios + torch.log(advantages.abs().clamp(1e-8))) "
            "                       .mean(dim=-1)) * neg_mask.squeeze(-1) "
            "  loss = -(geo_pos + geo_neg) "
            "Note: GMPO uses clipped ratios (same clip as DAPO, ε=0.2). "
            "Numerical stability: add ε=1e-8 inside log; importance ratios are "
            "already clipped by DAPO's clip(r_t, 1-ε, 1+ε). "
            "Wall-clock overhead: negligible (log/exp ops are cheap vs. forward pass). "
            ""
            "EXPECTED OUTCOME: "
            "More stable training curves (lower gradient norm variance across steps). "
            "HE pass@1 >= 0.93 after 1 cycle (target: match or exceed DAPO 92.68%). "
            "Importance ratio histogram: tighter distribution (fewer extreme ratios). "
            "ACR unaffected (GMPO does not change zero-variance group behavior). "
            "Paper reports +4.1% on math benchmarks — if HumanEval code is similarly "
            "outlier-sensitive, expect +1-3% pass@1 improvement. "
        ),
        "spec": {
            "pipeline": "humaneval_gmpo_geometric_mean",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 1,
            "grpo_algo": "gmpo",
            "gmpo_clip_eps": 0.2,
            "gmpo_log_eps": 1e-8,
            "dapo_dynamic_sampling": True,
            "n_generations": 8,
            "grpo_temperature": 1.0,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (DAPO arithmetic-mean reference: 92.68% HE)",
                "arxiv:2507.20673 (GMPO +4.1% on math benchmarks vs GRPO baseline)",
            ],
            "primary_metrics": [
                "humaneval_pass_at_1_post_1cycle",
                "grpo_gradient_norm_mean",
                "grpo_gradient_norm_std",
                "grpo_importance_ratio_p95",
                "grpo_importance_ratio_p99",
                "grpo_acr_mean",
                "training_loss_smoothed",
            ],
            "target_outcome": (
                "HE pass@1 >= 0.93 (match or exceed DAPO 92.68%). "
                "Gradient norm std reduced vs. DAPO baseline (more stable updates). "
                "Importance ratio P99 reduced vs. DAPO baseline (outlier bounding). "
                "Training loss variance reduced (smoother convergence). "
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
