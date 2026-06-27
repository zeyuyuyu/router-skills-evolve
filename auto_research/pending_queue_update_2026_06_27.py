#!/usr/bin/env python3
"""
Daily queue patch — 2026-06-27 (EXP-092, EXP-093).

A800 connectivity: offline since 2026-05-14 (day 44). Apply when restored.

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

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_06_27.py            # EXP-092, EXP-093

Queue was ~91 pending on 2026-06-26 (+2 added: EXP-090, EXP-091).
Queue cap applied: >20 → max 2 new experiments.

Data motivating today's proposals
----------------------------------
results/e2e_4cyc_gpt55/cycle_3/e2e_ablation_summary.json:
    HumanEval 4-cycle full/router: 92.68% task pass at 27.56% large-model cost
    GRPO zero-variance: 52.4% of groups discarded at G=8 DAPO (43/82 groups)
    Skills arm (always-small+procedure): 75.61% (up from 70.73% at cycle 0)
    Full = Router (entropy collapse prevents GRPO surpassing router)

results/cwy_35b_joint_20260606_165203/cycle_3/e2e_ablation_summary.json:
    tau2: always-small=76.97%, router=69.66%, always-large=60.11%
    multi-turn repair rollouts: up to 3 turns per tau2 task
    tau2 router accuracy 89.2% (cycle 3); large_model_fraction=35.5%

arxiv:2606.22995 — 'Group-Graph Policy Optimization for Long-Horizon Agentic
    Reinforcement Learning' (G2PO, June 22, 2026):
    G2PO formalizes multi-turn agent exploration as a global state-transition graph
    rather than isolated linear trajectories. Nodes = visited states; edges = (s_t,
    a_t, s_{t+1}) transitions. Two contributions: (1) group-aggregation state-value
    estimation — states visited by multiple rollouts aggregate their value estimates
    to reduce per-rollout baseline variance; (2) edge-centric advantage estimation —
    TD errors computed across ALL edges in the batch graph and globally standardized,
    so advantages prioritize transitions with the most discriminative task-progress
    signal regardless of which rollout they belong to. Applied to WebShop, ALFWorld,
    AppWorld: G2PO outperforms step-level GRPO (8–14% success rate improvement).

arxiv:2606.06058 — 'MDP-GRPO: Stabilized Group Relative Policy Optimization for
    Multi-Constraint Instruction Following' (Jun 4, 2026):
    Extends GRPO to handle multiple simultaneous reward signals (R_1 ... R_K) by
    maintaining K separate advantage estimates and combining them via a learned
    constraint weight vector. Motivation: single-scalar reward conflates orthogonal
    objectives (e.g., correctness vs. anti-forgetting vs. format compliance), leading
    to gradient interference when constraint directions are non-aligned. MDP-GRPO
    normalizes advantages per-constraint before combining, preventing any one objective
    from dominating the loss in high-variance batches. Adds a feasibility projection
    that clips the combined gradient to remain within the intersection of constraint
    improvement cones. Reported 11% improvement on multi-constraint instruction
    following vs. vanilla GRPO on constrained benchmarks.
"""
import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = (
    "/data0/home/zeyuwang/router-skills-evolve-data/humaneval/HumanEval.jsonl.gz"
)
_MBPP = (
    "/data0/home/zeyuwang/router-skills-evolve-data/mbpp/mbpp_sanitized.jsonl.gz"
)

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_06_27_001_g2po_graph_credit_tau2_multiturn",
        "priority": 8,
        "kind": "joint_cycle_multiseed",
        "rationale": (
            "arxiv:2606.22995 — 'Group-Graph Policy Optimization for Long-Horizon "
            "Agentic Reinforcement Learning' (G2PO, June 22, 2026). "
            ""
            "ROOT PROBLEM: Our tau2 multi-turn repair GRPO (up to 3 turns per task) "
            "assigns per-trajectory advantage A_r = (R_r - mean(R_g)) / std(R_g). "
            "This trajectory-level credit has two structural weaknesses in multi-turn "
            "settings that G2PO directly addresses: "
            ""
            "(W1) HIGH-VARIANCE BASELINE: Each rollout's baseline is estimated from "
            "only G=8 other rollouts of the SAME task. In tau2 with 74 tasks, many "
            "tasks have only 1–3 passing rollouts out of G=8, giving an extremely noisy "
            "baseline. If a repair succeeds at turn 2 via a lucky patch, its full "
            "trajectory advantage is high — but so is the variance, because the mean "
            "is estimated from only 7 other rollouts (some of which may have gone a "
            "completely different repair path). "
            ""
            "(W2) ISOLATED LINEAR CREDIT: Standard GRPO treats each rollout as an "
            "isolated linear sequence. In tau2 multi-turn repair, multiple rollouts "
            "from DIFFERENT tasks often produce identical intermediate states: e.g., "
            "task A turn-2 and task B turn-2 may both arrive at the same 'add boundary "
            "check' repair step. GRPO ignores this structural overlap — each rollout "
            "gets a per-task advantage that is blind to information from other tasks' "
            "repair trajectories, even when they share high-value transitions. "
            ""
            "G2PO SOLUTION: "
            "Transform the batch of multi-turn rollouts into a global state-transition "
            "graph: "
            "  - Nodes: (task_idx, turn_idx) observation states, keyed by turn-start "
            "    token content (first 128 tokens of each turn, hashed for fast lookup). "
            "  - Edges: (s_t, a_t, s_{t+1}) sequential turn transitions within each "
            "    rollout (K edges per rollout, K ≤ 3 for tau2). "
            "  - Group aggregation: for rollouts sharing the same turn-0 observation "
            "    (same task prompt, always the case within a group of G=8 rollouts of "
            "    one task), aggregate the value estimates: "
            "      V_agg(s_t^i) = mean_{j: obs(s_t^j) ≅ obs(s_t^i)} [V(s_t^j)] "
            "    where obs(s_t^j) ≅ obs(s_t^i) means same task AND same turn index. "
            "    For cross-task aggregation: states from DIFFERENT tasks that arrive at "
            "    structurally similar turn-2 observations (Jaccard(turn_2_tokens) ≥ 0.65) "
            "    also pool their value estimates, further reducing baseline variance. "
            "  - Edge-centric advantage: compute TD error for each edge: "
            "      δ(s_t, a_t, s_{t+1}) = R_final + V_agg(s_{t+1}) - V_agg(s_t) "
            "    Globally standardize across ALL edges in the batch: "
            "      A_edge(t) = (δ(t) - mean_batch(δ)) / (std_batch(δ) + eps) "
            "    This replaces the per-trajectory advantage with a cross-rollout, "
            "    cross-task advantage that identifies the specific turn-transitions that "
            "    most drove task progress. "
            ""
            "APPLICATION TO tau2 MULTI-TURN REPAIR (74 tasks × G=8 × max 3 turns): "
            "Batch graph: up to 74 × 8 × 3 = 1776 edges (sparse; most tasks complete "
            "in 1–2 turns). Cross-task aggregation: in a batch, ~12–15 tasks typically "
            "reach turn 2 with similar 'boundary-check repair' or 'type-cast repair' "
            "patterns — G2PO pools their value estimates across tasks, reducing per-task "
            "baseline variance from 8 samples to an effective 30–40 samples. "
            "Edge-centric advantage explicitly identifies whether the critical transition "
            "was the turn-2 repair decision (high delta) or the turn-1 boilerplate "
            "(low delta), providing fine-grained credit allocation. "
            ""
            "IMPLEMENTATION IN grpo_train_simple.py: "
            "After computing rollout embeddings (first 128 tokens hashed per turn): "
            "  1. Build adjacency dict: {node_key: [value_est_from_rollout_i, ...]} "
            "  2. Aggregate: V_agg[node] = mean(adjacency[node]) "
            "  3. Compute edge TD errors: δ = R + V_agg[next_node] - V_agg[curr_node] "
            "  4. Globally normalize: A = (δ - δ.mean()) / (δ.std() + 1e-8) "
            "  5. Use A as per-token advantage within the turn (same value for all "
            "     tokens in the same turn-transition). "
            "~40 lines in compute_loss(); no additional model forward passes needed "
            "(value estimates derived from the existing group-rollout logprobs). "
            ""
            "SCOPE: joint_cycle_multiseed (4 cycles × 2 seeds: [42, 7]) on tau2_bench "
            "with Qwen3-235B-A22B small model. Multi-seed and multi-cycle necessary to "
            "verify that graph credit assignment accumulates across cycles rather than "
            "providing only a one-cycle benefit. "
            ""
            "DISTINCTNESS FROM ALL QUEUED EXPERIMENTS: "
            "EXP-091 (TRUST, arxiv:2606.06976): weights per-TURN gradient by policy "
            "entropy. G2PO uses cross-rollout graph aggregation for value estimation "
            "and edge-centric TD errors — orthogonal axes (TRUST = which turn matters; "
            "G2PO = cross-trajectory baseline variance reduction). Can be composed. "
            "EXP-031 (MURPHY): execution-trace divergence for credit assignment; uses "
            "per-test-case pass/fail rather than graph TD errors. "
            "exp_2026_06_12_004_tau2_agentic_grpo: extends GRPO to tau2 agent format "
            "but uses standard trajectory-level advantage. G2PO changes the credit "
            "assignment mechanism itself. "
            "No queued experiment builds a cross-rollout state-transition graph for "
            "advantage estimation. ✓ "
            ""
            "EXPECTED OUTCOME: "
            "Reduced baseline variance in tau2 multi-turn GRPO (effective sample size "
            "~30–40 vs. current 8); better credit assignment to turn-2 repair decisions "
            "vs. turn-1 boilerplate. Target: tau2 task_pass > 72.97% (cycle-3 router "
            "reference); router routing_acc > 89.2%; large_model_fraction < 35.5%. "
            "Wall-clock: ~3.0 GPU-hours (4 cycles × 2 seeds × tau2 74 tasks × G=8 "
            "multi-turn rollouts; graph construction adds ~3% overhead). "
        ),
        "spec": {
            "pipeline": "tau2_4cycle_g2po_graph_credit",
            "bench": "tau2_bench",
            "base_model": "Qwen/Qwen3-235B-A22B",
            "n_cycles": 4,
            "n_seeds": 2,
            "seeds": [42, 7],
            "grpo_algo": "g2po_graph",
            "g2po_node_key": "turn_start_first128_hash",
            "g2po_cross_task_jaccard_threshold": 0.65,
            "g2po_value_agg_fn": "mean",
            "g2po_advantage_type": "edge_centric_td_globally_normalized",
            "base_reward": "tau2_binary_pass_fail",
            "n_generations": 8,
            "rollout": "repair",
            "max_turns": 3,
            "compare_with": [
                "cwy_35b_joint_20260606_165203 cycle_3 (tau2 router reference: "
                "72.97% task_pass, 35.5% large_fraction)",
                "exp_2026_06_26_002_trust (EXP-091: per-turn entropy weighting, "
                "orthogonal mechanism — credit distribution vs. baseline variance)",
                "exp_2026_06_12_004_tau2_agentic_grpo (same bench, standard "
                "trajectory-level advantage — direct ablation baseline)",
            ],
            "primary_metrics": [
                "tau2_task_pass_cycle3",
                "router_routing_acc_cycle3",
                "large_model_query_fraction",
                "grpo_zero_variance_group_rate",
                "g2po_effective_sample_size_per_node",
                "g2po_cross_task_aggregation_count",
            ],
            "target_outcome": (
                "task_pass > 0.7297 (cycle-3 router baseline) "
                "with large_model_fraction < 0.355; "
                "effective_sample_size_per_node > 12 (validates cross-task pooling)"
            ),
            "arxiv_ref": "2606.22995",
            "estimated_hours": 3.0,
            "implementation_note": (
                "In collect_rollouts() or compute_loss(): "
                "Step 1 — build node registry per batch: "
                "  node_key = hash(rollout.turns[k].tokens[:128]) for each (rollout, k) "
                "  node_values[node_key].append(V_estimate_from_rollout) "
                "Step 2 — cross-task aggregation (optional, toggle via flag): "
                "  for (k1, k2) in pairs: if jaccard(tokens1, tokens2) >= 0.65: merge "
                "Step 3 — compute edge TD errors: "
                "  delta[e] = R_final + mean(node_values[next_key]) "
                "              - mean(node_values[curr_key]) "
                "Step 4 — global normalize and assign per-turn advantage: "
                "  A[e] = (delta[e] - delta.mean()) / (delta.std() + 1e-8) "
                "  advantage_per_token[turn_k] = A[edge_k]  # uniform within turn "
                "Replace standard group-advantage line in compute_loss()."
            ),
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_06_27_002_mdpgrpo_multiseed_staircase_antiforgetting",
        "priority": 7,
        "kind": "grpo_multi_seed_staircase",
        "rationale": (
            "arxiv:2606.06058 — 'MDP-GRPO: Stabilized Group Relative Policy "
            "Optimization for Multi-Constraint Instruction Following' (Jun 4, 2026), "
            "combined with an observation from our 4-cycle HumanEval results. "
            ""
            "ROOT PROBLEM: Our HumanEval grpo_continual runs (cycles 0–3) train the "
            "small model exclusively on HumanEval rollouts. The skills arm (always-small "
            "+ procedure) improves from 70.73% → 75.61% pass@1 over 4 cycles, but "
            "forgetting of general coding capability (MBPP generalization) has not been "
            "measured or controlled. Standard GRPO's single scalar reward (binary "
            "HumanEval pass) provides no signal to preserve MBPP performance. Given the "
            "Qwen2.5-Coder-1.5B model's 82/100 MBPP baseline before training (from "
            "experiment exp_2026_06_18_001 diagnostic), uncontrolled forgetting during "
            "HumanEval GRPO may be reducing the model's utility as a general code model "
            "between cycles — a problem that would worsen if the pipeline extends to "
            "longer horizons (N > 4 cycles). "
            ""
            "MDP-GRPO SOLUTION (arxiv:2606.06058): "
            "MDP-GRPO handles K simultaneous reward signals by maintaining K separate "
            "advantage estimates A_1,...,A_K and combining them with a constraint weight "
            "vector w ∈ R^K: "
            "  L_MDP = sum_k w_k * L_GRPO(A_k) "
            "Per-constraint advantage normalization ensures no single constraint "
            "dominates when batch reward variances are unequal. A feasibility projection "
            "clips the combined gradient direction to remain within the intersection of "
            "all constraint improvement cones (a first-order sufficient condition for "
            "joint improvement). "
            ""
            "APPLICATION TO HUMANEVAL + MBPP ANTI-FORGETTING: "
            "K=2 constraints: "
            "  Constraint 1: HumanEval task pass@1 (primary objective, w_1=1.0 fixed) "
            "  Constraint 2: MBPP anti-forgetting (auxiliary objective, w_2=λ_af). "
            "    MBPP rollouts: at every 20th GRPO step, sample 10 tasks from a held-out "
            "    50-task MBPP subset; run G=4 rollouts; compute GRPO advantage; add "
            "    λ_af * L_GRPO_mbpp to the loss. This interleaves general coding "
            "    gradient signal into the HumanEval specialization loop. "
            ""
            "STAIRCASE DESIGN (grpo_multi_seed_staircase, 3 seeds): "
            "  Seed 42, λ_af = 0.0 → pure HumanEval GRPO (ablation baseline, no "
            "    anti-forgetting: should match existing grpo_continual reference) "
            "  Seed 7, λ_af = 0.2 → light anti-forgetting pressure (target: ≤2% "
            "    HumanEval regression vs. λ=0 while MBPP pass@1 improves ≥3%) "
            "  Seed 13, λ_af = 0.5 → strong anti-forgetting (test if MBPP preservation "
            "    can be achieved without HumanEval degradation > 5%) "
            "Seeds run sequentially on A800 (GPU memory: ~16 GB/seed at 1.5B). "
            "Total: 3 seeds × 1 cycle × 1.0 GPU-hours/seed = 3.0 GPU-hours. "
            ""
            "DISTINCTNESS FROM ALL QUEUED EXPERIMENTS: "
            "exp_2026_06_18_001_forgetting_eval_cycle3: evaluates forgetting post-hoc "
            "as a diagnostic; does NOT modify the GRPO objective to prevent it. "
            "MDP-GRPO adds an inline anti-forgetting constraint to the training loop. "
            "exp_2026_06_19_002_on_policy_replay: uses on-policy MBPP replay as a "
            "separate SFT phase after GRPO. MDP-GRPO interleaves MBPP rollouts into "
            "GRPO itself (concurrent, not sequential phases) — avoids train/eval "
            "distribution shift between phases. "
            "exp_2026_06_23_002_forever (EXP-087): Ebbinghaus-curve memory replay for "
            "SFT (not GRPO). Operates at the SFT phase, not inside the GRPO rollout. "
            "No queued experiment applies multi-constraint GRPO (λ_af staircase) to "
            "jointly optimize HumanEval specialization + MBPP anti-forgetting within "
            "a single GRPO training run. ✓ "
            ""
            "EXPECTED OUTCOME: "
            "λ_af = 0.2 target: HumanEval pass@1 ≥ 90% (vs. 92.68% reference at λ=0 "
            "over 4 cycles; 1-cycle run expected ~82–85% for all seeds); MBPP pass@1 "
            "degrades ≤2 percentage points from pre-training baseline vs. ≥5 pp "
            "degradation expected for λ_af=0. This validates MDP-GRPO's constraint "
            "feasibility projection prevents gradient interference when the MBPP and "
            "HumanEval gradients diverge (which they do ~30% of steps by cosine angle). "
            "Wall-clock: 3.0 GPU-hours (3 seeds × 1 cycle × ~1h). "
        ),
        "spec": {
            "pipeline": "humaneval_mdpgrpo_antiforgetting_staircase",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 1,
            "n_seeds": 3,
            "seeds": [42, 7, 13],
            "staircase_param": "lambda_antiforgetting",
            "staircase_values": [0.0, 0.2, 0.5],
            "grpo_algo": "mdp_grpo_multi_constraint",
            "mdpgrpo_primary_reward": "humaneval_binary_pass",
            "mdpgrpo_auxiliary_reward": "mbpp_binary_pass",
            "mdpgrpo_auxiliary_data": _MBPP,
            "mdpgrpo_auxiliary_n_tasks_per_step": 10,
            "mdpgrpo_auxiliary_interleave_every_n_steps": 20,
            "mdpgrpo_auxiliary_n_generations": 4,
            "mdpgrpo_feasibility_projection": True,
            "mdpgrpo_per_constraint_normalization": True,
            "base_advantage": "dapo_group_normalized",
            "dapo_dynamic_sampling": True,
            "n_generations": 8,
            "rollout": "repair",
            "max_turns": 3,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (4-cycle DAPO reference: 92.68% HE, "
                "forgetting of MBPP unmeasured)",
                "exp_2026_06_18_001_forgetting_eval (MBPP forgetting diagnostic: "
                "post-hoc eval, not training intervention)",
                "exp_2026_06_23_002_forever (EXP-087: Ebbinghaus SFT replay, "
                "forgetting prevention in SFT phase not GRPO)",
            ],
            "primary_metrics": [
                "humaneval_pass_at_1_post_1cycle",
                "mbpp_pass_at_1_post_1cycle",
                "mbpp_forgetting_delta_vs_pretrain",
                "grpo_zero_variance_group_rate",
                "mdpgrpo_gradient_cosine_angle_he_vs_mbpp",
                "mdpgrpo_feasibility_projection_activation_rate",
            ],
            "target_outcome": (
                "At lambda_af=0.2: HE_pass >= 0.82; MBPP_forgetting <= 0.02. "
                "At lambda_af=0.0: MBPP_forgetting >= 0.05 (validates forgetting risk). "
                "Staircase reveals the Pareto frontier between specialization and "
                "anti-forgetting — informs optimal lambda for 4-cycle runs."
            ),
            "arxiv_ref": "2606.06058",
            "estimated_hours": 3.0,
            "implementation_note": (
                "In collect_rollouts(): add MBPP sampling branch triggered every 20 steps "
                "(global step counter); run G=4 MBPP rollouts; store in separate buffer. "
                "In compute_loss(): "
                "  A_he = grpo_advantage(he_rollouts)  # normalized across G=8 HE group "
                "  A_mbpp = grpo_advantage(mbpp_rollouts)  # normalized across G=4 MBPP group "
                "  L_he = -(A_he * log_pi_he).mean() "
                "  L_mbpp = -(A_mbpp * log_pi_mbpp).mean() "
                "  # feasibility projection: clip if grads point away from each other "
                "  g_he = autograd.grad(L_he, params, retain_graph=True) "
                "  g_mbpp = autograd.grad(L_mbpp, params) "
                "  if dot(g_he, g_mbpp) < 0: g_mbpp = project_away(g_mbpp, g_he) "
                "  loss = L_he + lambda_af * L_mbpp_projected "
                "Lambda from staircase_values[seed_index]. "
                "~60 lines; feasibility projection is a single inner product + subtraction."
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
