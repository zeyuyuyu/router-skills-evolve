"""
Pending queue update — 2026-08-21
Run on A800 when connectivity is restored:
  python /data0/home/zeyuwang/auto_research/pending_queue_update_2026_08_21.py
Appends EXP-176 and EXP-177 to state["queue"] and saves atomically.

A800 offline since 2026-05-14 (day 99). Queue ~180 pending (>20 cap → 2 today).
AAAI 2027 camera-ready window (2026-08-16 to ~2026-08-29), day 6.
GPU window: CLOSED. EXP-176 needs GPU (queued for restoration); EXP-177 is offline + 1h GPU.

NEW PAPERS TODAY:
  EXP-176: Cue-GRPO / Rarity-Aware Credit Redistribution (arxiv:2608.03467) —
           GRPO credit concentrates on recurring correct solution forms, leaving rare but
           valid forms under-credited. Cue-GRPO partitions rollout-correct traces by
           solution structure and inverse-frequency-weights the advantage. In MERA: AST-
           level clustering of correct rollouts → rarity-weighted advantages. Distinct from
           G²RPO-A (EXP-175, prompt adaptation) and curriculum hard-first (EXP-177 below).
           Priority 7.
  EXP-177: FlyRoute Self-Evolving Router Profiling (arxiv:2605.22057) —
           FlyRoute grows agent capability profiles from successful routing traffic via a
           data flywheel, closest published prior to MERA's router co-evolution. Key
           differentiator: FlyRoute uses a targeted exploration policy (uncertainty × BM25
           relevance × novelty) to grow evidence only for under-profiled agents on plausible
           queries. MERA analog: identify "uncertain" routing frontier (|P_router−0.5|<0.1),
           force-run both models on those tasks, update router only on frontier traces.
           Camera-ready §2 citation + offline analysis + 1h targeted router retrain.
           Priority 6.

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
    python3 auto_research/pending_queue_update_2026_08_21.py  # EXP-181, EXP-182 (this file)
"""
import json, os, tempfile, shutil

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

new_experiments = [
    {
        "id": "exp_2026_08_21_001_cue_grpo_rarity_aware_credit_redistribution_ast_clustering",
        "priority": 7,
        "kind": "grpo_continual",
        "gpu": "auto",
        "rationale": (
            "arxiv:2608.03467 ('When Correct Solutions Repeat: Rarity-Aware Credit "
            "Redistribution for GRPO', Cao, Wen & Chen, August 2026) identifies a "
            "structural failure mode in standard GRPO: when the rollout group contains "
            "multiple correct solutions that share the same solution structure (e.g., the "
            "same algorithmic pattern or boilerplate), positive advantage mass accumulates "
            "proportionally to how frequently each structure is sampled. Rare but valid "
            "solution forms (e.g., a novel algorithm that solves the same problem differently) "
            "receive limited gradient signal. The result: the model overfits to common "
            "solution patterns and underexplores rare ones.\n\n"
            "Cue-GRPO addresses this with a partition-conditioned rarity reweighting rule: "
            "verify-correct rollouts are grouped into clusters by 'Strategy Cues' (deterministic "
            "structural labels derived without an auxiliary model), and positive advantages are "
            "reweighted by the inverse cluster size within each group. This transfers "
            "advantage mass from over-represented solution structures to under-represented "
            "ones. The combined rule also includes positive-credit stabilization (prevents "
            "advantage collapse for the dominant cluster) and within-group mean restoration "
            "(keeps the overall advantage distribution zero-mean). Results: +improvement on "
            "AIME repeated-sampling at high sampling budgets on Qwen2.5-Math-7B and "
            "Llama-3.1-8B-Instruct.\n\n"
            "MERA connection — why this matters for HumanEval GRPO:\n"
            "  In our GRPO Phase 3b on HumanEval, the small model (Qwen3-4B) is likely "
            "generating repeated correct solution structures for easy problems (e.g., simple "
            "list comprehensions or string slicing patterns that satisfy tests). These "
            "over-represented patterns dominate the positive advantage mass, while more "
            "complex but correct alternatives (recursion, generator patterns, custom "
            "comparators) receive minimal gradient signal. This structural homogeneity could "
            "explain the test-split degradation (92.68% train routing acc vs 60.98% test): "
            "the model learns to apply a narrow set of patterns rather than generalizing to "
            "unseen problem structures.\n\n"
            "Proposed MERA implementation: after rollout sampling (K=8 per problem), cluster "
            "the verify-correct rollouts by AST-level structural cues extracted via Python's "
            "`ast.walk()` — specifically the top-level function-body statement sequence type "
            "(e.g., 'for-loop→return', 'list-comp→return', 'recursion→base-case'). This "
            "produces a deterministic partition without any auxiliary model. Apply inverse "
            "cluster-size weights to positive advantages. For problems where all K rollouts "
            "share one structure (structure frequency = K), weight is 1/K, pushing the model "
            "to seek alternative valid forms in future rounds.\n\n"
            "This is DISTINCT from:\n"
            "  - EXP-175 (G²RPO-A adaptive guidance): changes prompt CONTENT for hard "
            "    problems; Cue-GRPO changes ADVANTAGE WEIGHTING for repeated correct forms.\n"
            "  - EXP-176 (curriculum hard-first): reorders batches by difficulty; Cue-GRPO "
            "    operates within a single group's reward normalization.\n"
            "  - EXP-174 (replay): prevents forgetting via trace reuse; Cue-GRPO targets "
            "    diversity within each rollout group.\n\n"
            "Continuing from cycle-3 GRPO adapter. 2 cycles (4-5). Batch=1, K=8, temp=0.9, "
            "DAPO algo. Expected: improved pass@1 on structurally diverse HumanEval problems "
            "(test split); compare routing accuracy improvement vs. baseline GRPO."
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
            "credit_redistribution": "cue_grpo",
            "cue_method": "ast_top_level_statement_sequence",
            "positive_credit_stabilization": True,
            "within_group_mean_restoration": True,
            "cost_target": 0.30,
            "paper": "arxiv:2608.03467",
        },
    },
    {
        "id": "exp_2026_08_21_002_flyroute_self_evolving_router_profiling_uncertainty_frontier",
        "priority": 6,
        "kind": "joint_cycle_multiseed",
        "gpu": "auto",
        "rationale": (
            "arxiv:2605.22057 ('FlyRoute: Self-Evolving Agent Profiling via Data Flywheel "
            "for Adaptive Task Routing', Li, Zhou & Wu, Huawei, May 2026) addresses the "
            "problem that enterprise routers rely on static agent profiles that become stale "
            "as agents evolve. FlyRoute is a self-evolving profiling framework: (1) dispatch "
            "candidates via current profile, (2) quality-gate successful routing pairs into "
            "each agent's success store, (3) periodically distill evidence into learned "
            "capability descriptions, (4) inject descriptions + BM25-retrieved successes into "
            "an LLM router. Results: zero-shot LLM router from 72.57% → 78.04% on proprietary "
            "enterprise developer-support queries.\n\n"
            "FlyRoute's KEY INNOVATION over naive evidence accumulation is the targeted "
            "exploration policy that combines three signals: (a) profile uncertainty (high when "
            "few successful examples exist for an agent), (b) BM25 relevance (plausible queries "
            "for that agent), and (c) lexical novelty (not duplicating existing evidence). This "
            "prevents redundant evidence collection and focuses exploration on the frontier where "
            "the router is uncertain.\n\n"
            "MERA's router is the closest structural analog to FlyRoute: MERA's TF-IDF + "
            "LogReg router is retrained each cycle on (prompt, large/small label) pairs from "
            "run-both oracle traces. Like FlyRoute's static profile problem, MERA's router "
            "faces a data coverage gap: the run-both oracle covers all tasks (with "
            "SCALING_FORCE_BOTH=1), but without it only ~25% of tasks have teacher traces. "
            "The router's soft-probability output identifies an 'uncertainty frontier': tasks "
            "where |P_router(large) − 0.5| < 0.1 are tasks the router cannot confidently route.\n\n"
            "Proposed experiment:\n"
            "  (1) Load cycle-3 router; extract uncertain-frontier prompts from HumanEval "
            "      (|P_router(large) − 0.5| < 0.1).\n"
            "  (2) Force-run BOTH small and large models on only those frontier prompts "
            "      (targeted oracle, not full SCALING_FORCE_BOTH=1 sweep).\n"
            "  (3) Retrain router using: cycle-3 full training set + new frontier traces.\n"
            "  (4) Measure: (a) routing accuracy on test split, (b) fraction of frontier "
            "      prompts that resolve to confident routing (|P| > 0.7).\n\n"
            "This is distinct from SCALING_FORCE_BOTH=1 (which forces both models on ALL "
            "tasks regardless of certainty) and from SeqRoute (2605.25424, budget-aware "
            "sequential routing). FlyRoute's targeted exploration principle is new in this "
            "project context and directly motivated by the test-split routing gap (92.68% "
            "train vs. 60.98% test routing accuracy seen in cycle-3 results).\n\n"
            "Camera-ready §2 citation: FlyRoute belongs in the 'Co-Evolving Router' "
            "paragraph alongside MERA's router co-evolution as the closest published prior. "
            "Differentiator: MERA co-evolves the router with the model and skill simultaneously "
            "(3-component joint evolution), while FlyRoute evolves only the router profile "
            "(agent model is fixed). This makes MERA a strictly richer co-evolution loop.\n\n"
            "GPU cost: ~1h (router training is CPU-dominant; GPU needed only for the forced "
            "small/large inference on ~20 uncertain frontier prompts). Offline analysis ~30min."
        ),
        "spec": {
            "base_model": "05_qwen3_5_4b_273",
            "router_checkpoint": "results/e2e_4cyc_gpt55/cycle_3/router/router.joblib",
            "bench": "humaneval",
            "n_cycles": 1,
            "flyroute_mode": True,
            "uncertainty_frontier_threshold": 0.1,
            "force_both_on_frontier": True,
            "retrain_router_with_frontier": True,
            "eval_metrics": [
                "routing_accuracy_test",
                "frontier_fraction_resolved",
            ],
            "cost_target": 0.05,
            "paper": "arxiv:2605.22057",
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
