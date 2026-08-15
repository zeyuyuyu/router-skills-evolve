"""
Pending queue update — 2026-08-15
Run on A800 when connectivity is restored:
  python /data0/home/zeyuwang/auto_research/pending_queue_update_2026_08_15.py
Appends EXP-166 and EXP-167 to state["queue"] and saves atomically.

A800 offline since 2026-05-14 (day 93). Queue ~167 pending (>20 cap → 2 today).
AAAI 2027 deadline: 2026-08-15 (TODAY). Post-submission camera-ready window begins.
NEW PAPERS TODAY:
  EXP-166: OCSD (arxiv:2608.04788) — second concurrent agentic RL paper; extends
           EXP-164's ADRS §2 paragraph with OCSD as part of the same concurrent cluster.
           Submitted Aug 5 2026; addresses identical design tension (privileged training
           vs. inference-time info) from a different angle (future env observations vs.
           procedural skills).
  EXP-167: Post-AAAI Camera-Ready Roadmap — offline analysis to plan which GPU experiments
           and paper sections need attention in the 2-week camera-ready window.

Apply chain before this patch:
    python3 auto_research/pending_queue_update_2026_08_11.py  # EXP-158, EXP-159
    python3 auto_research/pending_queue_update_2026_08_12.py  # EXP-160, EXP-161
    python3 auto_research/pending_queue_update_2026_08_13.py  # EXP-162, EXP-163
    python3 auto_research/pending_queue_update_2026_08_14.py  # EXP-164, EXP-165
    python3 auto_research/pending_queue_update_2026_08_15.py  # EXP-166, EXP-167 (this file)
"""
import json, os, tempfile, shutil

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

new_experiments = [
    {
        "id": "exp_2026_08_15_001_ocsd_agentic_rl_observation_calibrated_self_distillation_concurrent_citation",
        "priority": 8,
        "kind": "forgetting_eval",
        "gpu": "auto",
        "rationale": (
            "arxiv:2608.04788 ('Agentic Reinforcement Learning with Observation-Calibrated "
            "Self-Distillation', August 5, 2026) proposes OCSD, a framework that trains LLM agents "
            "via reinforcement learning using token-level dense supervision obtained by contrasting "
            "two structured replay views: a 'Full' replay view (which includes future environment "
            "observations as privileged context) and an 'Observation-Ablated' view (which removes "
            "those future observations). The difference between the two — the 'observation-calibrated "
            "gap' — provides a cleaner signal for identifying which tokens were steered by the "
            "privileged information vs. which were produced by the agent's own policy. The resulting "
            "token-level supervision avoids the confounding between privileged-info support and "
            "genuine policy improvement that plagues standard On-Policy Self-Distillation (OPSD)."
            "\n\n"
            "OCSD is a CONCURRENT system that shares the same core design tension as ADRS "
            "(arxiv:2608.03223, EXP-164): both papers grapple with when and how to expose "
            "privileged training information (procedural skills in ADRS; future environment "
            "observations in OCSD) without creating train/inference distribution shift. Together "
            "ADRS + OCSD form a coherent 'concurrent cluster' on the theme of privileged-info-"
            "calibrated RL for LLM agents. Including both in a single §2 paragraph is more "
            "informative than citing ADRS alone."
            "\n\n"
            "MERA vs. OCSD key differentiators: "
            "(a) OCSD's privileged signal is ENVIRONMENT STATE (future observations — what the "
            "environment returns after the agent's action); MERA's privileged signal is a "
            "PROCEDURAL SKILL extracted from teacher traces (what a stronger model does on the "
            "same task). These are structurally different privileged information types. "
            "(b) OCSD operates on MULTI-TURN interactive agents (web agents, tool-use, multi-step "
            "tasks); MERA targets SINGLE-TURN code generation (one response per HumanEval problem). "
            "(c) OCSD uses privileged info only at TRAINING TIME to calibrate token-level rewards; "
            "MERA uses the SkillBook procedure at BOTH training (prefix in SFT/GRPO examples) and "
            "inference (small model's prompt), with the procedure collapsing after cycle 0 (EXP-163). "
            "(d) OCSD has NO routing component — it trains a single-model policy. MERA's explicit "
            "logistic regression router (Phase 4) is MERA's unique contribution absent in both "
            "ADRS and OCSD. "
            "(e) MERA co-evolves three components (LLM, SkillBook, Router) across N cycles; "
            "OCSD and ADRS are single-run RL training frameworks."
            "\n\n"
            "Why the inference-time conditioning design is still defensible for MERA (compared to "
            "both ADRS and OCSD, which avoid inference-time privileged info): MERA's procedure "
            "prefix collapses after cycle 0 (EXP-163: 89% shrinkage, 17–21 word de facto noise "
            "from cycle 1 onward). At cycle 0, the GPT-5.5 procedure (1102 chars, rich Python "
            "patterns) serves the same function as ADRS's TVA gate or OCSD's full-replay view: "
            "it provides task-matched privileged knowledge to bootstrap quality. From cycles 1-3, "
            "the inference-time prefix is de facto empty — MERA operates without meaningful "
            "privileged conditioning, effectively converging to the same training-only-privileges "
            "design that ADRS and OCSD advocate. This is an honest nuance that strengthens the "
            "§2 concurrent work narrative: 'ADRS and OCSD independently validate the principle "
            "that inference-time privileged conditioning is unnecessary; MERA's empirical SkillBook "
            "collapse (EXP-163) provides a practical corroboration of this principle in code "
            "generation.'"
            "\n\n"
            "EXP-166 is a pure offline audit: (1) extend EXP-164's MERA vs. ADRS comparison "
            "table with a third column for OCSD; (2) draft an updated §2 concurrent work paragraph "
            "that covers ADRS + OCSD as a cluster under the heading 'Privileged Training Signals "
            "for Agentic RL'; (3) highlight MERA's three differentiators from the cluster: routing, "
            "co-evolution, single-turn code generation. "
            "Offline, 0h GPU, ~20 lines Python + markdown, 15 minutes. "
            "Priority 8: camera-ready critical — OCSD was submitted Aug 5, 2026 (10 days before "
            "AAAI deadline); not citing it alongside ADRS risks a reviewer noting the cluster "
            "was only partially acknowledged."
        ),
        "spec": {
            "script": "src/pipeline/ocsd_concurrent_citation_adrs_cluster_audit.py",
            "inputs": [
                "results/adrs_mera_differentiation_table.md",
                "results/adrs_concurrent_work_paragraph.md",
                "results/e2e_4cyc_gpt55/cycle_{0..3}/skillbook.json"
            ],
            "outputs": [
                "results/adrs_ocsd_concurrent_cluster_table.md",
                "results/sec2_privileged_training_signals_paragraph.md"
            ],
            "metrics": [
                "concurrent_papers_in_cluster",
                "mera_differentiators_from_cluster",
                "inference_time_prefix_collapse_summary"
            ],
            "comparison_dimensions": [
                "privileged_signal_type",
                "inference_time_conditioning",
                "routing_component",
                "task_horizon",
                "co_evolution_across_cycles"
            ],
            "known_values": {
                "ocsd": {
                    "arxiv": "2608.04788",
                    "submitted": "2026-08-05",
                    "privileged_signal": "future environment observations (observation-ablated gap)",
                    "inference_time_conditioning": False,
                    "routing": False,
                    "task_horizon": "multi-turn interactive agents",
                    "co_evolution": False
                },
                "adrs": {
                    "arxiv": "2608.03223",
                    "submitted": "2026-08-04",
                    "privileged_signal": "procedural skill prefix (TVA gate)",
                    "inference_time_conditioning": False,
                    "routing": False,
                    "task_horizon": "multi-turn interactive agents (3-7 steps)",
                    "co_evolution": False
                },
                "mera": {
                    "privileged_signal": "SkillBook procedure (GPT-5.5 distilled at cycle 0)",
                    "inference_time_conditioning": True,
                    "inference_prefix_effective_cycles": "cycle_0_only",
                    "inference_prefix_de_facto_empty": "cycles_1_3",
                    "routing": True,
                    "task_horizon": "single-turn code generation (HumanEval)",
                    "co_evolution": True
                }
            },
            "paper_sections": [
                "sec2_concurrent_adrs_ocsd_cluster_paragraph"
            ],
            "estimated_runtime_minutes": 15,
            "arxiv": "2608.04788",
            "immediately_runnable": True,
            "gpu_required": False,
            "deadline_critical": True,
            "target_revision": "camera_ready"
        }
    },
    {
        "id": "exp_2026_08_15_002_post_aaai_camera_ready_roadmap_2week_window",
        "priority": 6,
        "kind": "forgetting_eval",
        "gpu": "auto",
        "rationale": (
            "Today (2026-08-15) is the AAAI 2027 submission deadline. With the paper submitted, "
            "the camera-ready revision window opens (typically 2 weeks). This offline roadmap "
            "experiment compiles the complete camera-ready action plan from the experiment queue "
            "and local data, prioritized by impact."
            "\n\n"
            "Three categories of camera-ready work: "
            "\n"
            "(A) GPU EXPERIMENTS — must run if A800 connectivity is restored in the camera-ready "
            "window. Priority order: "
            "(1) EXP-099 (grpo_multi_seed_staircase): Staircase GRPO curriculum across 4 cycles "
            "— the §8 future work experiment that arxiv:2606.22317 predicts should push the "
            "small model ceiling beyond 75.61%. Estimated: ~6h on one A800 80GB. "
            "(2) EXP-144 (grpo_continual): GRSD intra-cycle skill distillation integration — "
            "adds §8 empirical evidence for the group-reflective self-distillation future work "
            "direction. Estimated: ~4h. "
            "(3) EXP-139 (grpo_curriculum_continual): Frontier teacher disagreement curriculum — "
            "selects the 20-70% pass@k band per arxiv:2606.22317 theory. Estimated: ~3h."
            "\n"
            "(B) OFFLINE CITATION AUDITS — runnable immediately on A800 without GPU: "
            "Priority order (from existing queue): "
            "(1) EXP-160: CoPES concurrent differentiation (§2) — CRITICAL. "
            "(2) EXP-162: DA-GRPO concurrent differentiation (§2) — CRITICAL. "
            "(3) EXP-164: ADRS concurrent differentiation (§2) — CRITICAL. "
            "(4) EXP-166 (today): OCSD concurrent cluster expansion (§2) — HIGH. "
            "(5) EXP-161: RFT forgetting external validation (§5.2) — HIGH. "
            "(6) EXP-163: SkillBook collapse audit (§3 limitation) — HIGH. "
            "(7) EXP-154: Router calibration UCCI audit (§4 Table 11) — MEDIUM. "
            "(8) EXP-165: Curriculum RL DAPO grounding (§3 footnote + §8) — MEDIUM. "
            "(9) EXP-158: Geometry Conflict Index trajectory (§5.1 Fig 3b) — MEDIUM."
            "\n"
            "(C) POST-SUBMISSION CONCURRENT PAPER MONITORING — check arxiv daily (Aug 16–30) "
            "for papers citing ADRS (2608.03223), OCSD (2608.04788), or DA-GRPO (2602.00166) "
            "that may require §2 updates in camera-ready. Key search terms: "
            "'agentic RL privileged skill', 'LLM router co-evolution', 'GRPO continual learning "
            "code generation', 'procedural skill distillation continual'."
            "\n\n"
            "EXP-167 compiles this roadmap into a single markdown file with a daily checklist, "
            "GPU experiment priority table, and concurrent paper monitoring plan. "
            "Offline, 0h GPU, ~10 lines Python + markdown, 15 minutes. "
            "Priority 6: useful for planning; not AAAI-critical for initial submission but "
            "essential for camera-ready organization."
        ),
        "spec": {
            "script": "src/pipeline/post_aaai_camera_ready_roadmap.py",
            "inputs": [
                "auto_research/reports/ideas-2026-08-14.md",
                "auto_research/reports/ideas-2026-08-15.md",
                "results/e2e_4cyc_gpt55/cycle_3/e2e_ablation_summary.json"
            ],
            "outputs": [
                "results/camera_ready_roadmap_2026_08_15.md"
            ],
            "sections": [
                "gpu_experiments_priority_table",
                "offline_citation_audits_checklist",
                "post_submission_concurrent_paper_monitoring",
                "camera_ready_submission_checklist"
            ],
            "gpu_experiment_priority": [
                {"exp": "EXP-099", "kind": "grpo_multi_seed_staircase",
                 "gpu_hours": 6, "paper_impact": "§8 empirical staircase curriculum"},
                {"exp": "EXP-144", "kind": "grpo_continual",
                 "gpu_hours": 4, "paper_impact": "§8 GRSD intra-cycle skill distillation"},
                {"exp": "EXP-139", "kind": "grpo_curriculum_continual",
                 "gpu_hours": 3, "paper_impact": "§3 Phase 3b curriculum empirical validation"}
            ],
            "offline_citation_priority": [
                "EXP-160 (CoPES §2)", "EXP-162 (DA-GRPO §2)", "EXP-164 (ADRS §2)",
                "EXP-166 (OCSD §2 cluster)", "EXP-161 (RFT forgetting §5.2)",
                "EXP-163 (SkillBook collapse §3)", "EXP-154 (router calibration §4)",
                "EXP-165 (curriculum RL §3+§8)", "EXP-158 (GCI trajectory §5.1)"
            ],
            "monitoring_search_terms": [
                "agentic RL privileged skill code generation",
                "LLM router co-evolution GRPO continual",
                "procedural skill distillation HumanEval routing",
                "MERA multi-component evolution LLM"
            ],
            "estimated_runtime_minutes": 15,
            "arxiv": None,
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
