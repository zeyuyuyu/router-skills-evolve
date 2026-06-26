#!/usr/bin/env python3
"""
Daily queue patch — 2026-06-26 (EXP-090, EXP-091).

A800 connectivity: offline since 2026-05-14 (day 43). Apply when restored.

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

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_06_26.py            # EXP-090, EXP-091

Queue was ~89 pending on 2026-06-25 (+2 added: EXP-088, EXP-089).
Queue cap applied: >20 → max 2 new experiments.

Data motivating today's proposals
----------------------------------
results/e2e_4cyc_gpt55/cycle_3/e2e_ablation_summary.json:
    HumanEval 4-cycle full/router: 92.68% task pass at 27.56% large-model cost
    GRPO zero-variance: 52.4% of groups discarded at G=8 DAPO (43/82 groups)
    Skills arm (always-small+procedure): 75.61% (up from 70.73% at cycle 0)
    Full = Router (entropy collapse prevents GRPO surpassing router)

results/cwy_35b_fullsplit_20260610_083624/cycle_3/e2e_ablation_summary.json:
    tau2: always-small=76.97%, router=69.66%, always-large=60.11%
    multi-turn repair rollouts: up to 3 turns per tau2 task

arxiv:2606.19236 — 'STARE: Surprisal-Guided Token-Level Advantage Reweighting for
    Policy Entropy Stability' (Luo et al., Tsinghua + Tencent Hunyuan, June 17, 2026):
    First-order gradient analysis of token-level entropy dynamics under GRPO reveals
    advantage-surprisal four-quadrant structure: tokens with (high advantage, high
    surprisal) cause explosive entropy increase; (high advantage, low surprisal)
    reinforce boilerplate paths; (low advantage, high surprisal) cause entropy collapse
    on already-uncertain tokens. STARE weights each token's advantage by its batch-
    internal surprisal S_t = -log pi_ref(a_t|s_t), concentrating gradient on tokens
    where the policy and reference diverge most significantly. Prevents entropy collapse
    without an auxiliary entropy loss term (which can interfere with task reward).

arxiv:2606.06976 — 'Exploring Agentic Tool-Calling Decisions via Uncertainty-Aligned
    Reinforcement Learning' (TRUST, June 2026):
    Multi-turn agent trajectories suffer from credit assignment dilution: the same binary
    outcome reward is applied to all turns, including boilerplate turns (parameter filling,
    syntax completion) and decision turns (which tool to call next, whether to retry).
    TRUST quantifies per-turn uncertainty as the policy entropy averaged over that turn's
    tokens, then upweights the gradient contribution of high-entropy (decision) turns and
    applies a repulsive uncertainty-separation regularizer that maintains distinct action
    distributions across turns with similar context but different optimal decisions.
    In a 4-turn tool-calling benchmark, TRUST reduces hallucinated tool invocations by
    37% and unsupported direct responses by 29% vs standard multi-turn GRPO.
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
        "id": "exp_2026_06_26_001_stare_surprisal_token_advantage_humaneval",
        "priority": 9,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2606.19236 — 'STARE: Surprisal-Guided Token-Level Advantage "
            "Reweighting for Policy Entropy Stability' "
            "(Luo et al., Tsinghua University & Tencent Hunyuan, June 17, 2026). "
            ""
            "ROOT PROBLEM: GRPO/DAPO assign a uniform trajectory-level advantage A_r "
            "to every token in rollout r, regardless of each token's individual "
            "information content. This creates three credit-assignment pathologies "
            "identified via first-order gradient analysis in STARE: "
            "(P1) Entropy collapse: tokens with high surprisal (low reference prob) "
            "that happen to belong to a zero-advantage group receive no gradient, so "
            "the policy's distribution over these uncertain tokens collapses toward "
            "the reference — removing productive exploration. "
            "(P2) Boilerplate reinforcement: tokens with low surprisal (boilerplate "
            "like `def solution():`, closing parentheses, import statements) in "
            "high-advantage rollouts receive the same positive advantage as the "
            "creative tokens that actually determined the solution's correctness. "
            "(P3) Advantage-surprisal mismatch: the four-quadrant structure "
            "[(+adv,+surprisal)→explosive entropy, (+adv,-surprisal)→boilerplate "
            "reinforcement, (-adv,+surprisal)→collapse, (-adv,-surprisal)→benign] "
            "means that without surprisal-weighting, GRPO simultaneously worsens "
            "entropy collapse (P1) and dilutes signal on the creative tokens that "
            "matter most (P2). "
            ""
            "STARE SOLUTION: Weight each token's effective advantage by its "
            "batch-internal surprisal S_t = -log pi_ref(a_t | s_t): "
            "A_STARE(t, r) = A_GRPO(r) * S_t / (mean_t(S_t) + eps). "
            "The product A_GRPO(r) * S_t concentrates gradient on high-surprisal, "
            "high-advantage tokens (the creative steps that made a rollout correct) "
            "and down-weights low-surprisal boilerplate even in high-advantage "
            "rollouts. The batch-internal normalization (dividing by mean surprisal "
            "across the batch) prevents the scale of S_t from dominating the loss "
            "when the reference policy is very sharp. "
            "For zero-advantage groups (all-pass or all-fail): A_GRPO(r) = 0 for all "
            "r, so A_STARE(t,r) = 0 — STARE does not directly eliminate the "
            "zero-variance collapse for fully-uniform-outcome groups. However, it "
            "reduces entropy collapse in the non-zero-variance groups (the remaining "
            "47.6%), so the policy retains more entropy in subsequent rollouts → "
            "fewer groups collapse to all-pass or all-fail in the NEXT batch. "
            "Entropy stability is measured by the near-criticality property: the "
            "authors show that STARE maintains dH/dt ≈ 0 (entropy neither grows nor "
            "collapses) throughout training, while vanilla GRPO has dH/dt < 0 after "
            "~500 steps. "
            ""
            "IMPLEMENTATION IN grpo_train_simple.py: "
            "After computing per-token logprobs log pi_ref(a_t|s_t) (already done "
            "in the reference-logprob pass): "
            "  surprisal = -log_pi_ref  # shape: (batch, seq_len) "
            "  mean_surp = surprisal.mean(dim=-1, keepdim=True)  # per-sequence "
            "  stare_weights = surprisal / (mean_surp + 1e-8)  # normalized "
            "  loss = -((stare_weights * A_GRPO.unsqueeze(-1)) * log_pi_theta).mean() "
            "~15 lines replacing the standard advantage-weighted policy gradient loss. "
            ""
            "DISTINCTNESS FROM ALL QUEUED EXPERIMENTS: "
            "EXP-086 (VeRPO, arxiv:2601.03525): Changes the REWARD R_r via dynamic "
            "test-difficulty weighting. STARE changes how the ADVANTAGE A_r is "
            "distributed across tokens — orthogonal axes. Can be composed. "
            "EXP-088 (VIMPO, arxiv:2606.20008): Uses suffix log-ratio sums as an "
            "implicit value function. STARE uses raw surprisal (one-step, not "
            "suffix). Both are token-level but with different information: VIMPO's "
            "suffix log-ratio captures future-trajectory divergence; STARE's surprisal "
            "captures the local token's surprise under the reference. "
            "EXP-082 (RL-ZVP): Penalizes zero-variance groups post-hoc. STARE "
            "addresses entropy collapse in the non-zero-variance groups — a different "
            "mechanism. "
            "EXP-055 (SA-AH-GRPO, arxiv:2606.05434): Applies entropy-adaptive "
            "HORIZON DISCOUNTING to reduce the effective gradient window when the "
            "model is uncertain. STARE applies surprisal-based ADVANTAGE WEIGHTING "
            "at each token position — distinct axes (window vs. weight). "
            "No queued experiment uses batch-internal surprisal as advantage weights. ✓ "
            ""
            "EXPECTED OUTCOME: "
            "Reduce entropy collapse in non-zero-variance groups; maintain policy "
            "entropy near criticality; improve per-token credit assignment for creative "
            "steps in passing rollouts. Target: HumanEval cycle-3 pass@1 > 92.68% "
            "(reference); entropy trajectory stable through cycle 3 (dH/dt ≈ 0). "
            "Wall-clock: ~80 min (same model + rollout count as DAPO reference; "
            "surprisal computation adds one reference-logprob forward pass, already "
            "present in the KL-penalty computation, ~2% overhead). "
        ),
        "spec": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "train_data": _HE_EVAL,
            "eval_data": _HE_EVAL,
            "bench": "humaneval",
            "grpo_algo": "stare",
            "advantage_type": "surprisal_weighted_trajectory",
            "stare_surprisal_source": "reference_model_logprobs",
            "stare_normalize_per_sequence": True,
            "stare_eps": 1e-8,
            "base_advantage": "dapo_group_normalized",
            "dapo_dynamic_sampling": True,
            "dapo_clip_low": 0.2,
            "dapo_clip_high": 0.5,
            "n_generations": 8,
            "rollout": "repair",
            "max_turns": 3,
            "epochs": 1,
            "lr": 5e-6,
            "lora_r": 16,
            "prompt_style": "qwen-chat",
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (DAPO G=8 binary, 92.68% reference, 52.4% zero-var)",
                "exp_2026_06_25_001_vimpo (EXP-088: suffix log-ratio, different token axis)",
                "exp_2026_06_06_001_sa_ah_grpo (EXP-055: entropy-adaptive horizon, orthogonal)",
            ],
            "primary_metrics": [
                "humaneval_pass_at_1_cycle3",
                "grpo_zero_variance_group_rate",
                "policy_entropy_trajectory",
                "dH_dt_estimate_steps_0_500_1000",
            ],
            "expected_improvement": (
                "Stable entropy (dH/dt ≈ 0 vs DAPO dH/dt < 0 after step 500); "
                "pass@1 > 92.68%; zero-variance rate < 52.4%"
            ),
            "arxiv_ref": "2606.19236",
            "estimated_hours": 1.4,
            "implementation_note": (
                "In grpo_train_simple.py compute_loss(): "
                "ref_logp already computed for KL term. "
                "surprisal = -ref_logp  # (B, T) "
                "stare_w = surprisal / (surprisal.mean(-1, keepdim=True) + 1e-8) "
                "policy_grad = -(stare_w * advantages.unsqueeze(-1) * log_pi).mean() "
                "Replace standard policy_grad = -(advantages.unsqueeze(-1) * log_pi).mean()"
            ),
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_06_26_002_trust_uncertainty_multiturn_reward_humaneval_4cycle",
        "priority": 7,
        "kind": "joint_cycle_multiseed",
        "rationale": (
            "arxiv:2606.06976 — 'Exploring Agentic Tool-Calling Decisions via "
            "Uncertainty-Aligned Reinforcement Learning' (TRUST, June 2026). "
            ""
            "ROOT PROBLEM: Multi-turn GRPO in our HumanEval Phase 3b assigns the same "
            "final binary outcome reward R ∈ {0, 1} to every token across all turns in "
            "a repair trajectory (max 3 turns). This creates CREDIT ASSIGNMENT DILUTION "
            "for multi-turn rollouts: "
            "(D1) Turn-level credit ignored: the crucial decision at turn 2 ('should I "
            "refactor the loop or just fix the off-by-one?') receives the same gradient "
            "signal as the boilerplate at turn 1 ('import sys; def solution(n): '). "
            "(D2) Inter-turn confusion: rollouts with different decision sequences at "
            "turn 2 but identical outcomes receive identical per-token advantages, so "
            "the model cannot distinguish good decisions from lucky ones. "
            "(D3) Wasted rollouts: in a group where all 8 rollouts succeed (zero-variance "
            "group), the multi-turn structure provides no additional discriminative signal "
            "because all turns from all rollouts receive zero advantage. "
            ""
            "TRUST SOLUTION: "
            "For each turn k in a trajectory (k=1..K), compute the per-turn policy "
            "entropy H_k = -mean_{t in turn k}[sum_v pi(v|s_t) log pi(v|s_t)]. "
            "High H_k indicates turn k is a decision point (model is uncertain about "
            "what to generate next). Low H_k indicates boilerplate. "
            "TRUST weights each turn's gradient contribution by its uncertainty: "
            "  w_k = softmax(H_k)_k  (normalized across turns in the trajectory) "
            "The final multi-turn GRPO loss becomes: "
            "  L_TRUST = -sum_k w_k * A_r * mean_{t in turn k}[log pi(a_t|s_t)] "
            "Additionally, TRUST adds an uncertainty-separation regularizer: "
            "  L_sep = -mean_{(k,k') s.t. H_k > H_{k'}}[D_KL(pi_k || pi_{k'})] "
            "This encourages the model to maintain distinct (high-entropy) action "
            "distributions at decision turns. Weight: lambda_sep = 0.01 (from paper). "
            ""
            "APPLICATION TO HUMANEVAL MULTI-TURN REPAIR: "
            "Turn 1: initial solution attempt (often high entropy — model choosing "
            "algorithm / data structure approach) "
            "Turn 2 (if turn 1 fails tests): error message read + repair decision "
            "(extremely high entropy — model choosing whether to refactor or patch) "
            "Turn 3 (if turn 2 fails): second repair attempt (moderate entropy) "
            "TRUST should upweight turn 2 gradient (the hardest repair decision) and "
            "down-weight turn 1 boilerplate structure tokens. In our reference run "
            "(e2e_4cyc_gpt55, G=8 DAPO), ~30% of problems require multi-turn repair "
            "(HumanEval, 1.5B model). For these ~25 problems, TRUST provides turn-level "
            "credit attribution that standard DAPO lacks. "
            ""
            "IMPLEMENTATION: "
            "In collect_rollouts(): after generating each turn, compute H_k using "
            "token logits from the forward pass (already available). Store H_k per turn "
            "in the rollout buffer. "
            "In compute_loss(): "
            "  turn_weights = F.softmax(H_k_tensor, dim=-1)  # (B, K) "
            "  per_turn_loss = [-(log_pi_turn_k * adv).mean() for k in turns] "
            "  loss = (turn_weights * per_turn_loss_tensor).sum(-1).mean() "
            "  loss += lambda_sep * uncertainty_separation_loss(H_k_tensor, pi_turns) "
            "~50 lines, requires tracking turn boundaries in the rollout buffer "
            "(turn boundaries already available as stop tokens in repair setup). "
            ""
            "SCOPE: joint_cycle_multiseed (4 cycles, seeds 42 and 7) to verify that "
            "better turn-level credit assignment accumulates benefits across cycles. "
            "Using 2 seeds because TRUST's entropy-based turn weighting introduces "
            "seed-level variance in which turns are upweighted — multi-seed check "
            "ensures the gain is systematic. "
            ""
            "DISTINCTNESS FROM ALL QUEUED EXPERIMENTS: "
            "EXP-089 (UCCI, joint_cycle_multiseed): changes ROUTER calibration "
            "post-training. TRUST changes GRPO multi-turn reward — orthogonal. "
            "EXP-031 (MURPHY, multi-turn GRPO): MURPHY extends GRPO to multi-turn via "
            "retrospective credit assignment using execution trace divergence. TRUST "
            "uses uncertainty quantification (entropy) at decision boundaries — a "
            "different source of per-turn weight. MURPHY uses success/failure of "
            "individual test cases across turns; TRUST uses policy entropy. "
            "No queued experiment uses per-turn entropy to weight gradient contributions "
            "in multi-turn GRPO. ✓ "
            ""
            "EXPECTED OUTCOME: "
            "Better credit assignment for multi-turn problems → improved repair "
            "efficiency → lower turn-2/3 usage → lower large-model escalation. "
            "Target: HumanEval cycle-3 task_pass > 92.68% (reference); "
            "large_model_query_fraction < 0.27 (improvement over 27.56% reference). "
            "Wall-clock: ~3.0 GPU-hours (4-cycle × 2 seeds; turn entropy computation "
            "adds ~5% overhead over DAPO per-token forward pass). "
        ),
        "spec": {
            "pipeline": "humaneval_4cycle_trust_multiturn_reward",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 4,
            "n_seeds": 2,
            "seeds": [42, 7],
            "grpo_algo": "trust_uncertainty_aligned",
            "trust_turn_weight_source": "policy_entropy_per_turn",
            "trust_weight_fn": "softmax_normalized",
            "trust_lambda_sep": 0.01,
            "trust_sep_kl_direction": "high_entropy_to_low_entropy",
            "base_advantage": "dapo_group_normalized",
            "dapo_dynamic_sampling": True,
            "n_generations": 8,
            "rollout": "repair",
            "max_turns": 3,
            "compare_with": [
                "e2e_4cyc_gpt55 (4-cycle DAPO reference: 92.68% task_pass, 27.56% cost)",
                "exp_2026_06_25_002_ucci (EXP-089: UCCI router calibration, orthogonal)",
                "exp_murphy_multiturn (EXP-031: execution-trace divergence credit, different source)",
            ],
            "primary_metrics": [
                "router_task_pass_at_cycle_3",
                "large_model_query_fraction",
                "multi_turn_usage_fraction",
                "grpo_zero_variance_group_rate",
                "per_turn_gradient_contribution_k1_k2_k3",
            ],
            "target_outcome": (
                "task_pass > 0.9268 with large_model_query_fraction < 0.27; "
                "turn-2 gradient contribution > turn-1 (validates TRUST mechanism)"
            ),
            "arxiv_ref": "2606.06976",
            "estimated_hours": 3.0,
            "implementation_note": (
                "Track turn boundaries in repair rollout buffer via stop-token indices. "
                "After forward pass for logprob computation: "
                "  H_k = -sum_t(pi_t * log_pi_t).mean(turn_k_mask) "
                "  turn_w = F.softmax(H_k_stacked, dim=-1)  # shape: (B, K) "
                "In compute_loss(): "
                "  per_turn_pg = [-(log_pi[turn_k_mask] * adv).mean() for k in K] "
                "  loss_pg = (turn_w * torch.stack(per_turn_pg, -1)).sum(-1).mean() "
                "  # uncertainty separation (low cost): "
                "  pairs = [(k,k') for k < k' if H_k[k] > H_k[k']] "
                "  loss_sep = -mean[kl_div(pi_k, pi_kp) for (k,kp) in pairs] "
                "  loss = loss_pg + lambda_sep * loss_sep"
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
