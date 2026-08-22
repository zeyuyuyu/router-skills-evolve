"""
Pending queue update — 2026-08-22
Run on A800 when connectivity is restored:
  python /data0/home/zeyuwang/auto_research/pending_queue_update_2026_08_22.py
Appends EXP-183 and EXP-184 to state["queue"] and saves atomically.

A800 offline since 2026-05-14 (day 100). Queue ~182 pending (>20 cap → 2 today).
AAAI 2027 camera-ready window (2026-08-16 to ~2026-08-29), day 7.
GPU window: CLOSED. EXP-183 offline only; EXP-184 needs GPU (queued for restoration).

NEW PAPERS TODAY:
  EXP-183: Task- and Session-Level Model Routing benchmark (arxiv:2608.14641) —
           Hybrid evaluation of 4 open-source routers across RouterBench, BFCL v4,
           tau2-bench, WebArena. Key finding: 3 of 4 routers emit near-constant or
           zero-variance tier assignments (always-same routing). Camera-ready §2 audit:
           demonstrate MERA's router varies with prompt content and achieves lower cost
           at comparable quality vs. constant-assignment baselines. Priority 8.
  EXP-184: Latent Thought Credit multi-answer GRPO (arxiv:2608.01593) —
           Multi-answer credit assignment for GRPO via latent-space aggregation of K
           rollout answers; more informative advantage estimates where standard per-rollout
           scalar collapses to zero-variance for easy problems. Addresses MERA's
           ACR=52.4% zero-variance group issue. GPU needed. Priority 6.

Apply chain before this patch:
    python3 auto_research/pending_queue_update_2026_08_11.py  # EXP-158, EXP-159
    python3 auto_research/pending_queue_update_2026_08_12.py  # EXP-160, EXP-161
    python3 auto_research/pending_queue_update_2026_08_13.py  # EXP-162, EXP-163
    python3 auto_research/pending_queue_update_2026_08_14.py  # EXP-164, EXP-165
    python3 auto_research/pending_queue_update_2026_08_15.py  # EXP-166, EXP-167
    python3 auto_research/pending_queue_update_2026_08_16.py  # EXP-168, EXP-169
    python3 auto_research/pending_queue_update_2026_08_17.py  # EXP-170, EXP-171
    python3 auto_research/pending_queue_update_2026_08_18.py  # EXP-172, EXP-173
    python3 auto_research/pending_queue_patch_2026-08-19.py   # EXP-174..178 (block)
    python3 auto_research/pending_queue_update_2026_08_20.py  # EXP-179, EXP-180
    python3 auto_research/pending_queue_update_2026_08_21.py  # EXP-181, EXP-182
    python3 auto_research/pending_queue_update_2026_08_22.py  # EXP-183, EXP-184 (this file)
"""
import json, os, tempfile, shutil

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

new_experiments = [
    {
        "id": "exp_2026_08_22_001_router_benchmark_audit_constant_assignment_positioning",
        "priority": 8,
        "kind": "offline_analysis",
        "gpu": "none",
        "rationale": (
            "arxiv:2608.14641 ('Task- and Session-Level Model Routing: A Common-Interface "
            "Hybrid Evaluation of Four Open-Source Routers Across Four Benchmarks', Kumar "
            "& Saminathan, Indiana University, August 2026) provides the most comprehensive "
            "public benchmark of open-source LLM routers to date: 4 routers (MatryoshkaRouter, "
            "Aurelio, vLLM Semantic Router, and a 5th commercial) evaluated on 290 frozen "
            "tasks × 2,610 candidate outcomes across RouterBench, BFCL v4, tau2-bench, and "
            "WebArena. The study's central finding is damning for naive router deployment: "
            "three of four open-source routers emit near-constant or zero-variance tier "
            "assignments — they route almost all queries to the same model tier regardless "
            "of prompt content. The fourth (vLLM Semantic Router) varies with content but "
            "has the lowest success rate on none of the four benchmarks. Always-Mid (a "
            "constant-assignment baseline that always routes to the middle tier) matches "
            "Aurelio exactly on three benchmarks and within 0.003 on the fourth.\n\n"
            "MERA camera-ready §2 significance: this benchmark reveals that learned routing "
            "variability — not just routing infrastructure — is the open problem. MERA's "
            "router is trained on per-task (prompt, large/small oracle) labels from "
            "SCALING_FORCE_BOTH=1 runs, producing a signal-bearing, prompt-sensitive routing "
            "function. The camera-ready §2 'Router Design' paragraph should cite 2608.14641 "
            "to establish that MERA's training approach (oracle labels from run-both, TF-IDF "
            "feature encoding, LogReg trained per cycle) yields a router that genuinely varies "
            "with prompt content, contrasting with the constant-assignment failure mode "
            "documented here.\n\n"
            "Proposed experiment (offline, ~1h):\n"
            "  (1) Load cycle-3 router (results/e2e_4cyc_gpt55/cycle_3/router/router.joblib).\n"
            "  (2) Extract P_router(large) distribution over all 164 HumanEval test prompts.\n"
            "  (3) Compute: (a) variance of P_router(large), (b) fraction of prompts with "
            "      |P_router(large) − mean| > 0.05 (non-constant fraction), (c) fraction with "
            "      P_router(large) > 0.7 (confident large) vs. < 0.3 (confident small).\n"
            "  (4) Compare vs. Always-Large (100% cost), Always-Small (47% MBPP baseline), "
            "      and Always-Mid (50%/50% random split).\n"
            "  (5) Report: routing entropy H(P_router), cost at parity quality, routing "
            "      variability vs. the constant-assignment baselines from 2608.14641.\n\n"
            "Output: a 'Router Variability' table for §2 and §4 of the camera-ready draft "
            "(to be placed in reports/router_variability_audit_2026_08_22.md).\n\n"
            "Also serves as: diagnostic for whether current router is collapsing toward "
            "constant assignment as training cycles progress (a risk if the small model's "
            "quality gap closes).\n\n"
            "No GPU required. ~1h. Priority 8 (camera-ready §2, directly addresses a gap "
            "in the current draft's router-positioning argument)."
        ),
        "spec": {
            "router_checkpoint": "results/e2e_4cyc_gpt55/cycle_3/router/router.joblib",
            "bench": "humaneval",
            "eval_type": "router_variability_audit",
            "metrics": [
                "router_probability_variance",
                "non_constant_fraction",
                "routing_entropy",
                "cost_at_parity",
            ],
            "baselines": ["always_large", "always_small", "always_mid"],
            "output_report": "reports/router_variability_audit_2026_08_22.md",
            "paper": "arxiv:2608.14641",
        },
    },
    {
        "id": "exp_2026_08_22_002_latent_thought_credit_multi_answer_grpo_advantage",
        "priority": 6,
        "kind": "grpo_continual",
        "gpu": "auto",
        "rationale": (
            "arxiv:2608.01593 ('Latent Thought Credit: Multi-Answer Credit Assignment for "
            "Latent Reasoning', Zhang, Li & Huang, Peking University, August 2026) addresses "
            "a credit assignment failure mode in standard GRPO that is distinct from the "
            "solution-structure frequency issue targeted by Cue-GRPO (EXP-181): in standard "
            "GRPO, the advantage for each rollout is computed as (reward − group_mean_reward) "
            "/ group_std. For easy problems where K/K rollouts are correct, group_std → 0 "
            "and advantages collapse to zero, producing no gradient signal (the 'zero-variance "
            "group' problem). For hard problems where 0/K rollouts are correct, the same "
            "collapse occurs.\n\n"
            "Latent Thought Credit proposes replacing the scalar per-rollout reward with a "
            "multi-answer credit vector computed as follows: (1) for each rollout, extract "
            "the final answer token's latent representation h_i from the last transformer "
            "layer; (2) compute pairwise cosine similarity between correct-answer latent "
            "vectors (h_correct_j); (3) assign credit proportional to the average cosine "
            "distance from the rollout's latent representation to the correct-answer manifold "
            "— rollouts that are 'close to correct in latent space' receive partial credit "
            "even when their surface-form answer is wrong. This prevents zero-advantage "
            "collapse for hard problems (where no rollout matches the surface answer) by "
            "providing a smooth gradient signal from the latent-space proximity.\n\n"
            "MERA connection — why this matters for HumanEval GRPO:\n"
            "  MERA's GRPO Phase 3b reports ACR=52.4% (fraction of rollout groups with "
            "non-zero advantage variance). This means 47.6% of training steps contribute "
            "no gradient. On HumanEval with Qwen3-4B, easy problems (task difficulty < 0.4) "
            "likely produce K/K correct rollouts → zero advantage groups. Latent Thought "
            "Credit would provide a smooth advantage signal for these easy problems by "
            "rewarding rollouts that reach a 'more correct' latent representation even among "
            "all-correct groups.\n\n"
            "For hard problems (0/K correct rollouts), the latent-proximity credit provides "
            "partial gradient toward the correct-answer manifold, improving sample efficiency "
            "compared to zero-gradient steps.\n\n"
            "Proposed MERA implementation:\n"
            "  - After each rollout batch, extract final-token latent vectors h_i from "
            "    Qwen3-4B (layer -1) for all K rollouts.\n"
            "  - For correct rollouts: h_correct = {h_i : reward_i == 1.0}.\n"
            "  - Latent credit for rollout i: lc_i = mean_j cosine_sim(h_i, h_correct_j) "
            "    if h_correct is non-empty, else 0.\n"
            "  - Blended advantage: A_i = α * (reward_i - mean(reward)) / std(reward) + "
            "    (1-α) * lc_i, where α is annealed from 0.5 to 1.0 over training.\n"
            "  - Fallback for zero-variance groups: use lc_i directly as the advantage.\n\n"
            "This is DISTINCT from:\n"
            "  - EXP-181 (Cue-GRPO): reweights among correct-form clusters; requires at "
            "    least some correct rollouts; does not provide gradient for all-fail groups.\n"
            "  - EXP-175 (G²RPO-A): adapts the prompt; does not change advantage computation.\n\n"
            "GPU cost: ~3-4h (requires forward pass to extract latent vectors, then GRPO "
            "update; 2 cycles from cycle-3 checkpoint). Queued for GPU window restoration.\n"
            "Priority 6."
        ),
        "spec": {
            "base_model": "05_qwen3_5_4b_273",
            "from_checkpoint": "results/e2e_4cyc_gpt55/cycle_3/grpo_adapter",
            "bench": "humaneval",
            "n_cycles": 2,
            "grpo_temperature": 0.9,
            "grpo_batch_size": 1,
            "n_generations": 8,
            "algo": "dapo",
            "credit_method": "latent_thought_credit",
            "latent_layer": -1,
            "alpha_anneal_start": 0.5,
            "alpha_anneal_end": 1.0,
            "zero_variance_fallback": "latent_credit_direct",
            "cost_target": 0.30,
            "paper": "arxiv:2608.01593",
        },
    },
]


def already_exists(state, exp_id):
    for item in state.get("queue", []) + state.get("history", []):
        if item.get("id") == exp_id:
            return True
    return False


with open(STATE_PATH, "r") as f:
    state = json.load(f)

added = []
for exp in new_experiments:
    if already_exists(state, exp["id"]):
        print(f"SKIP (already exists): {exp['id']}")
    else:
        state.setdefault("queue", []).append(exp)
        added.append(exp["id"])
        print(f"ADDED: {exp['id']} (priority={exp['priority']})")

if added:
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    shutil.move(tmp, STATE_PATH)
    print(f"\nSaved {STATE_PATH} with {len(added)} new experiment(s).")
else:
    print("\nNo new experiments added (all duplicates).")
