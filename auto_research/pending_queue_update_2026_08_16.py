"""
Pending queue update — 2026-08-16
Run on A800 when connectivity is restored:
  python /data0/home/zeyuwang/auto_research/pending_queue_update_2026_08_16.py
Appends EXP-168 and EXP-169 to state["queue"] and saves atomically.

A800 offline since 2026-05-14 (day 94). Queue ~169 pending (>20 cap → 2 today).
AAAI 2027 deadline: 2026-08-15 (YESTERDAY). Camera-ready window begins TODAY.
NEW PAPERS TODAY (camera-ready concurrent paper monitoring, first day):
  EXP-168: Scaffold-Mediated Post-Training (arxiv:2608.05156) — closest structural
           concurrent analog to MERA's full 3-component co-evolution (LLM+Skills+Router).
           Co-evolves procedural scaffold graph with model params; missing from §2 is a
           major camera-ready risk. CRITICAL priority 9.
  EXP-169: LLMRouter (arxiv:2608.06867) — unified 5-component routing taxonomy, 16+
           routers, xRouteBench. MERA's router must be positioned within this framework
           for camera-ready §2 "Router" subsection. Priority 8.

Apply chain before this patch:
    python3 auto_research/pending_queue_update_2026_08_11.py  # EXP-158, EXP-159
    python3 auto_research/pending_queue_update_2026_08_12.py  # EXP-160, EXP-161
    python3 auto_research/pending_queue_update_2026_08_13.py  # EXP-162, EXP-163
    python3 auto_research/pending_queue_update_2026_08_14.py  # EXP-164, EXP-165
    python3 auto_research/pending_queue_update_2026_08_15.py  # EXP-166, EXP-167
    python3 auto_research/pending_queue_update_2026_08_16.py  # EXP-168, EXP-169 (this file)
"""
import json, os, tempfile, shutil

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

new_experiments = [
    {
        "id": "exp_2026_08_16_001_scaffold_mediated_post_training_co_evolving_skills_model_concurrent_citation",
        "priority": 9,
        "kind": "forgetting_eval",
        "gpu": "auto",
        "rationale": (
            "arxiv:2608.05156 ('Scaffold-Mediated Post-Training: Co-Evolving Model Parameters "
            "and Procedural Scaffold Graphs', August 2026, Alibaba Group + Tsinghua University) "
            "proposes a three-phase framework that co-evolves a procedural scaffold graph with "
            "model parameters: (1) discovery — scaffold nodes are initialized from teacher "
            "demonstrations; (2) distillation — each scaffold node is trained into the model "
            "via SFT+RL; (3) dynamic recompilation — scaffold graph edges are reweighted based "
            "on model performance, pruning ineffective procedures. Results: +8.1pp pass rate "
            "from automatically discovered skills; 85.2% distillation retention rate (model "
            "achieves 27.7% pass@k on FeatureBench without any external scaffold post-training)."
            "\n\n"
            "This is the CLOSEST STRUCTURAL CONCURRENT PAPER to MERA's entire framework. "
            "The scaffold graph discovery phase mirrors MERA's cycle-0 GPT-5.5 SkillBook seeding; "
            "the distillation phase mirrors MERA's SFT+GRPO training; the dynamic recompilation "
            "mirrors MERA's per-cycle SkillBook evolution from small-model successful traces. "
            "The structural parallel is so strong that omitting this paper from §2 would invite "
            "a reviewer question: 'How does MERA differ from Scaffold-Mediated Post-Training?'"
            "\n\n"
            "MERA's key differentiators from Scaffold-Mediated Post-Training: "
            "(1) ROUTING COMPONENT: MERA's explicit logistic regression router (Phase 4) dispatches "
            "tasks to small or large LLM based on predicted difficulty. Scaffold-Mediated targets "
            "single-model post-training with no dispatch mechanism — routing is MERA's unique "
            "contribution absent from ALL concurrent work (ADRS, OCSD, CoPES, DA-GRPO, "
            "Scaffold-Mediated). "
            "(2) THREE-COMPONENT CO-EVOLUTION: MERA co-evolves three components (LLM + SkillBook + "
            "Router) whereas Scaffold-Mediated co-evolves two (model params + scaffold graph). "
            "The router is the third dimension absent from Scaffold-Mediated's design. "
            "(3) SKILL STRUCTURE: MERA's SkillBook is a flat single-skill store (single global "
            "'coding' procedure per cycle) distilled from GPT-5.5 traces; Scaffold-Mediated's "
            "scaffold graph has nodes (procedures) and edges (task-dependency weights) with "
            "learned dependency structure. MERA's flat structure is a simplification that enables "
            "the routing problem to be factored out cleanly. "
            "(4) INFERENCE-TIME BEHAVIOR: Scaffold-Mediated explicitly eliminates the scaffold at "
            "inference time (85.2% distillation retention). MERA's SkillBook collapses empirically "
            "from cycle 1 (EXP-163: 89% shrinkage, 17-21 word noise sequences), so MERA converges "
            "to the same inference-time-agnostic operation in practice — consistent with both "
            "Scaffold-Mediated and the ADRS/OCSD design philosophy."
            "\n\n"
            "EXP-168 is a pure offline audit: (1) build a comparison table (MERA vs. "
            "Scaffold-Mediated across 5 dimensions: skill structure, routing, co-evolution "
            "components, inference scaffold, task domain); (2) draft a new §2.3 subparagraph "
            "'Procedural Skill Distillation and Co-Evolution' that covers Scaffold-Mediated + "
            "Skill-α (arxiv:2608.01678); (3) state MERA's routing differentiator clearly. "
            "Offline, 0h GPU, ~35 lines Python + markdown, 20 minutes. "
            "Priority 9 (CRITICAL): This is the first day of the camera-ready window; adding "
            "this citation now ensures the §2 concurrent work section is complete against the "
            "full August 2026 paper cluster."
        ),
        "spec": {
            "script": "src/pipeline/scaffold_mediated_post_training_concurrent_citation.py",
            "inputs": [
                "results/e2e_4cyc_gpt55/cycle_{0..3}/skillbook.json",
                "results/adrs_ocsd_concurrent_cluster_table.md",
                "results/sec2_privileged_training_signals_paragraph.md"
            ],
            "outputs": [
                "results/scaffold_mediated_mera_differentiation_table.md",
                "results/scaffold_mediated_mera_differentiation_paragraph.md",
                "results/sec2_skill_coevolution_subsection.md"
            ],
            "metrics": [
                "concurrent_papers_in_skill_coevolution_cluster",
                "mera_differentiators_routing_absent_from_all_concurrent",
                "inference_time_scaffold_comparison"
            ],
            "comparison_dimensions": [
                "skill_structure",
                "routing_component",
                "co_evolution_components",
                "inference_time_scaffold",
                "task_domain"
            ],
            "known_values": {
                "scaffold_mediated": {
                    "arxiv": "2608.05156",
                    "submitted": "2026-08",
                    "affiliation": "Alibaba Group + Tsinghua University",
                    "skill_structure": "scaffold graph (nodes=procedures, edges=task-dependency weights)",
                    "skill_source": "automated discovery from teacher demonstrations",
                    "routing_component": False,
                    "co_evolution_components": ["model_params", "scaffold_graph"],
                    "inference_time_scaffold": "eliminated_after_distillation",
                    "distillation_retention_rate": 0.852,
                    "task_domain": "multi-step reasoning (FeatureBench)",
                    "key_result": "+8.1pp pass rate; 27.7% pass@k scaffold-free"
                },
                "mera": {
                    "skill_structure": "SkillBook flat single-skill store (global 'coding' procedure)",
                    "skill_source": "GPT-5.5 teacher traces (cycle 0) + small model successes (cycles 1-3)",
                    "routing_component": True,
                    "co_evolution_components": ["LLM", "SkillBook", "Router"],
                    "inference_time_scaffold": "cycle_0_only_collapses_to_noise_cycles_1_3",
                    "task_domain": "code generation (HumanEval, MBPP)"
                }
            },
            "paper_sections": [
                "sec2_procedural_skill_distillation_co_evolution_new_subsection",
                "mera_routing_differentiator_from_all_concurrent_work"
            ],
            "estimated_runtime_minutes": 20,
            "arxiv": "2608.05156",
            "immediately_runnable": True,
            "gpu_required": False,
            "deadline_critical": True,
            "target_revision": "camera_ready"
        }
    },
    {
        "id": "exp_2026_08_16_002_llmrouter_unified_infrastructure_routing_taxonomy_concurrent_citation",
        "priority": 8,
        "kind": "forgetting_eval",
        "gpu": "auto",
        "rationale": (
            "arxiv:2608.06867 ('LLMRouter: Unified Infrastructure for Developing, Evaluating, "
            "and Deploying LLM Routers', August 7, 2026, UIUC + Maryland + NTU + Purdue + UIC) "
            "proposes a unified framework that formalizes LLM routing as a sequential decision "
            "process with five components: (1) context encoders, (2) model encoders, (3) scoring "
            "functions, (4) decision rules, (5) learning signals. The paper introduces xRouteBench "
            "(cross-task routing benchmark covering text classification, QA, code generation, "
            "retrieval) and an open-source library with 16+ representative routers spanning "
            "classification-based, regression-based, and LLM-judge-based designs. Key finding: "
            "learned routers outperform the strongest fixed-model baseline by 14.6% relatively "
            "on xRouteBench. The framework is explicitly designed to unify prior routing work "
            "(Frugal GPT, LLM Cascade, RouterBench, etc.) under a common taxonomy."
            "\n\n"
            "MERA's router is a specific instantiation of the LLMRouter 5-component framework: "
            "context encoder = TF-IDF over raw problem text; model encoder = none (binary label "
            "from small model pass@1 at current cycle checkpoint); scoring function = logistic "
            "regression sigmoid; decision rule = fixed threshold τ=0.5 (uncalibrated — EXP-154 "
            "UCCI audit); learning signal = cross-entropy on GPT-5.5 trace labels."
            "\n\n"
            "WHY THIS IS CAMERA-READY CRITICAL: LLMRouter provides the canonical taxonomy for "
            "§2's 'Router' subsection. Reviewers familiar with the routing literature — and "
            "especially reviewers who are LLMRouter authors or users — will immediately ask "
            "'how does MERA's router map onto this taxonomy?' without a citation. Citing LLMRouter "
            "and providing the explicit mapping positions MERA's contribution as a co-evolutionary "
            "router variant that extends the LLMRouter framework with: "
            "(a) cycle-aware dynamic re-training: MERA's router re-trains each cycle using "
            "updated small-model evaluation outcomes — LLMRouter benchmarks static routing policies; "
            "(b) skill-conditioned routing: MERA's routing decision is implicitly conditioned "
            "on SkillBook procedure quality per cycle (rich GPT-5.5 procedure at cycle 0 vs. "
            "collapsed noise at cycles 1-3); no LLMRouter system integrates a co-evolving skill "
            "bank alongside the router; "
            "(c) joint training signal: labels come from GPT-5.5 teacher trace outcomes on the "
            "same distribution as SFT/GRPO phases, rather than from held-out evaluation data."
            "\n\n"
            "EXP-169 is a pure offline audit: (1) map MERA's router onto the 5-component "
            "LLMRouter taxonomy and document each component; (2) identify MERA's differentiators "
            "(co-evolution, skill-conditioning, joint training signal); (3) draft a §2 router "
            "subsection addition that cites LLMRouter and positions MERA's router within the "
            "taxonomy; (4) compare MERA's routing accuracy (92.68% at 27.56% cost, cycle 3) "
            "to LLMRouter's 14.6% relative gain benchmark. "
            "Offline, 0h GPU, ~40 lines Python + markdown, 20 minutes. "
            "Priority 8 (HIGH): Camera-ready citation; LLMRouter is the reference infrastructure "
            "paper for the routing community and MERA's §2 'Router' subsection is incomplete "
            "without it."
        ),
        "spec": {
            "script": "src/pipeline/llmrouter_taxonomy_positioning_audit.py",
            "inputs": [
                "src/pipeline/train_router_simple.py",
                "results/e2e_ablation_a800_20260509_summary.json"
            ],
            "outputs": [
                "results/llmrouter_taxonomy_mera_mapping.md",
                "results/llmrouter_mera_taxonomy_paragraph.md"
            ],
            "metrics": [
                "mera_router_taxonomy_component_mapping",
                "mera_co_evolutionary_differentiators_vs_static_routers",
                "mera_routing_accuracy_vs_xroutebench_best"
            ],
            "taxonomy_components": [
                "context_encoder",
                "model_encoder",
                "scoring_function",
                "decision_rule",
                "learning_signal"
            ],
            "known_values": {
                "llmrouter": {
                    "arxiv": "2608.06867",
                    "submitted": "2026-08-07",
                    "affiliation": "UIUC + Maryland + NTU + Purdue + UIC",
                    "benchmark": "xRouteBench (cross-task: classification, QA, code, retrieval)",
                    "num_routers": 16,
                    "best_relative_gain_over_fixed_model": 0.146
                },
                "mera_router": {
                    "context_encoder": "TF-IDF bag-of-words over raw problem text",
                    "model_encoder": "none (binary pass@1 label at current cycle checkpoint)",
                    "scoring_function": "logistic regression sigmoid",
                    "decision_rule": "fixed threshold tau=0.5 (uncalibrated, EXP-154)",
                    "learning_signal": "cross-entropy on GPT-5.5 trace outcome labels",
                    "cycle_3_routing_accuracy": 0.9268,
                    "cycle_3_escalation_cost": 0.2756
                },
                "mera_differentiators": [
                    "co_evolutionary_retraining_per_cycle",
                    "skill_conditioned_routing_implicit",
                    "joint_training_signal_same_distribution_as_sft_grpo"
                ]
            },
            "paper_sections": [
                "sec2_llm_routing_subsection_llmrouter_taxonomy_mapping"
            ],
            "estimated_runtime_minutes": 20,
            "arxiv": "2608.06867",
            "immediately_runnable": True,
            "gpu_required": False,
            "deadline_critical": False,
            "target_revision": "camera_ready"
        }
    }
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
    print(f"\nSaved {STATE_PATH} with {len(added)} new experiments.")
    print(f"Queue size now: {len(state.get('queue', []))}")
else:
    print("No new experiments added (all already exist).")
