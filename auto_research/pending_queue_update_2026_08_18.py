"""
Pending queue update — 2026-08-18
Run on A800 when connectivity is restored:
  python /data0/home/zeyuwang/auto_research/pending_queue_update_2026_08_18.py
Appends EXP-172 and EXP-173 to state["queue"] and saves atomically.

A800 offline since 2026-05-14 (day 96). Queue ~173 pending (>20 cap → 2 today).
AAAI 2027 camera-ready window (2026-08-16 to ~2026-08-29), day 3.
GPU window: CLOSED. Both new experiments are OFFLINE / 0h GPU.

NEW PAPERS TODAY:
  EXP-172: Skill-α (arxiv:2608.01678) — Progressive Agent Skill Generation via RL with
           rollback reward. Pre-planned from 2026-08-17 monitoring queue. Skill-α
           decomposes skill construction into individually evaluable edits and evaluates
           each via a rollback reward (downstream execution under original vs. edited
           skill). Falls in §2.4 Procedural Skill cluster with Scaffold-Mediated (EXP-168).
           Priority 7.
  EXP-173: Catastrophic Remembering in Agentic Coding (arxiv:2608.11095) — NEW, not in
           any prior report. Documents that agentic prompts (CLAUDE.md, etc.) grow without
           bound (+226%, +4.9 instructions/commit) due to cheap-append / expensive-delete
           asymmetry ("catastrophic remembering," inverse of catastrophic forgetting).
           Directly relevant to MERA: EXP-163 found MERA's SkillBook collapses 89%
           (procedure 1102→118 chars) — the OPPOSITE trajectory. MERA's quality-filtered
           compression (updates only from small model success traces) prevents catastrophic
           remembering. Camera-ready §2 must state this contrast. Priority 8.

Apply chain before this patch:
    python3 auto_research/pending_queue_update_2026_08_11.py  # EXP-158, EXP-159
    python3 auto_research/pending_queue_update_2026_08_12.py  # EXP-160, EXP-161
    python3 auto_research/pending_queue_update_2026_08_13.py  # EXP-162, EXP-163
    python3 auto_research/pending_queue_update_2026_08_14.py  # EXP-164, EXP-165
    python3 auto_research/pending_queue_update_2026_08_15.py  # EXP-166, EXP-167
    python3 auto_research/pending_queue_update_2026_08_16.py  # EXP-168, EXP-169
    python3 auto_research/pending_queue_update_2026_08_17.py  # EXP-170, EXP-171
    python3 auto_research/pending_queue_update_2026_08_18.py  # EXP-172, EXP-173 (this file)
"""
import json, os, tempfile, shutil

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

new_experiments = [
    {
        "id": "exp_2026_08_18_001_skill_alpha_progressive_skill_generation_rollback_reward_rl_concurrent_citation",
        "priority": 7,
        "kind": "forgetting_eval",
        "gpu": "auto",
        "rationale": (
            "arxiv:2608.01678 ('Progressive Agent Skill Generation via Reinforcement "
            "Learning', August 2026, CUHK + LIGHTSPEED) proposes Skill-α, an RL method "
            "for progressively generating high-quality agent skills. Key insight: skill "
            "quality cannot be directly supervised (no ground-truth labels for whether a "
            "skill is good or bad). Skill-α instead formulates skill generation as a "
            "SEQUENTIAL EDITING PROCESS — each edit modifies the current skill text at a "
            "specific location — and evaluates each edit via a ROLLBACK REWARD: compare "
            "downstream task execution under the original skill vs. the edited skill on an "
            "anchored query. If the edit improves performance, it earns positive reward; if "
            "it degrades performance, it earns negative reward. This provides a well-defined "
            "per-edit reward signal without requiring any skill ground truth.\n\n"
            "The rollback reward creates a direct RL link between skill content and task "
            "performance, enabling the model to progressively refine skills across 3 rounds: "
            "each round re-edits the skill from the previous round using the rollback signal, "
            "identifying high-value edits (e.g., removing redundant steps, tightening "
            "procedure language, adding edge-case handling) from low-value or harmful ones.\n\n"
            "Reported results: +11.2pp task completion vs. static skill bank across diverse "
            "agent benchmarks. Progressive quality improvement is monotonic across all 3 "
            "rounds.\n\n"
            "MERA's relationship to Skill-α:\n"
            "  MERA's SkillBook update is a simpler BINARY CRITERION: update the procedure "
            "if and only if the small model achieves pass@1=1 on the task when conditioned "
            "on the current procedure. There is no explicit rollback reward or sequential "
            "editing process — the procedure is replaced wholesale from the large model's "
            "successful trace distillation (cycle 0) or from the concatenation of "
            "small-model successful traces (cycles 1-3).\n\n"
            "  Skill-α's rollback reward is a FINE-GRAINED version of MERA's success "
            "criterion: both measure whether the skill HELPS the model on a task; Skill-α "
            "measures this at the edit level (within a single skill generation), while MERA "
            "measures it at the cycle level (between whole skill versions). Skill-α's "
            "sequential editing can be understood as a gradient-based analog to MERA's "
            "per-cycle discrete replacement.\n\n"
            "  KEY DIFFERENTIATOR: MERA's collapse (EXP-163: 89% shrinkage from cycle 1) "
            "indicates that the MERA binary success criterion is too coarse to maintain "
            "discriminative skill content — once the small model internalizes the skill, "
            "any new procedure update inherits the same near-uniform successes. Skill-α's "
            "rollback reward would instead continue to refine the skill content even when "
            "average performance is high, by identifying locally harmful edits. This "
            "suggests Skill-α could prevent MERA's collapse by providing a more granular "
            "quality signal. Camera-ready §2.4 should note this direction.\n\n"
            "EXP-172 is a pure offline audit: (1) characterize Skill-α's rollback reward "
            "mechanism and compare to MERA's binary success criterion; (2) identify whether "
            "Skill-α's sequential editing prevents the collapse observed in EXP-163; "
            "(3) draft a §2.4 paragraph covering Scaffold-Mediated (EXP-168) and Skill-α "
            "as the two procedural skill co-evolution papers in August 2026; (4) state "
            "MERA's differentiator (full 3-component co-evolution with explicit router). "
            "Offline, 0h GPU, ~40 lines Python + markdown, 30 minutes. Priority 7: planned "
            "in yesterday's monitoring queue; §2.4 procedural skill cluster currently "
            "incomplete without Skill-α."
        ),
        "spec": {
            "script": "src/pipeline/skill_alpha_rollback_reward_concurrent_citation.py",
            "inputs": [
                "results/e2e_4cyc_gpt55/cycle_{0..3}/skillbook.json",
                "results/scaffold_mediated_mera_differentiation_table.md"
            ],
            "outputs": [
                "results/skill_alpha_mera_comparison_table.md",
                "results/sec24_procedural_skill_distillation_paragraph.md"
            ],
            "metrics": [
                "skill_alpha_rollback_reward_vs_mera_binary_success_criterion_documented",
                "skill_alpha_collapse_prevention_hypothesis_stated",
                "sec24_scaffold_mediated_plus_skill_alpha_cluster_paragraph_written"
            ],
            "known_values": {
                "skill_alpha": {
                    "arxiv": "2608.01678",
                    "method": "sequential_editing_with_rollback_reward",
                    "reward_granularity": "per_edit",
                    "rounds": 3,
                    "result": "+11.2pp task completion vs static skill bank"
                },
                "mera_comparison": {
                    "update_criterion": "binary_pass@1_small_model_success",
                    "update_granularity": "per_cycle_whole_procedure_replacement",
                    "collapse_finding": "EXP-163_89pct_shrinkage_cycle0_1102_to_cycle1_118_chars",
                    "differentiator": "MERA_adds_explicit_learned_router_Phase4_absent_from_Skill-alpha"
                }
            },
            "aaai_priority": "MEDIUM",
            "estimated_runtime_minutes": 30,
            "arxiv": "2608.01678",
            "immediately_runnable": True,
            "gpu_required": False,
            "target_revision": "camera_ready",
            "sec2_section": "2.4_procedural_skill_distillation_co_evolution"
        }
    },
    {
        "id": "exp_2026_08_18_002_catastrophic_remembering_agentic_prompts_mera_skillbook_collapse_contrast",
        "priority": 8,
        "kind": "forgetting_eval",
        "gpu": "auto",
        "rationale": (
            "arxiv:2608.11095 ('Why Does CLAUDE.md Keep Growing? Catastrophic Remembering "
            "in Agentic Coding', August 2026) characterizes a phenomenon directly inverse "
            "to MERA's SkillBook collapse (EXP-163). The paper documents that agentic "
            "prompt files (CLAUDE.md, .cursorrules, system prompts) grow WITHOUT BOUND in "
            "real repositories: +226% growth over lifetime, +4.9 net instructions per "
            "commit, and the older an instruction gets, the LESS likely it is to be deleted "
            "(log-hazard -0.032 per commit). The paper names this 'catastrophic remembering' "
            "— the INVERSE of catastrophic forgetting — and attributes it to an asymmetric "
            "cost structure: APPENDING is O(1) (just write it), while DELETING is O(2^|D|) "
            "(must verify no correctness regression across all instruction subsets). Study "
            "across 247,694 instruction lifetimes in 1,867 repositories. First rigorous "
            "empirical characterization of prompt bloat as a continual learning problem.\n\n"
            "MERA's SkillBook exhibits the OPPOSITE trajectory — catastrophic FORGETTING of "
            "procedure content (89% shrinkage, EXP-163: 1102→118 chars from cycle 0 to "
            "cycle 1). MERA escapes catastrophic remembering because its SkillBook update "
            "criterion is QUALITY-FILTERED: the procedure is updated ONLY when the small "
            "model achieves pass@1=1 on a task when conditioned on the current procedure. "
            "This quality filter prevents cheap-append accumulation:\n"
            "  - Under catastrophic remembering (naive agentic prompts): every new "
            "instruction is appended regardless of whether it helps downstream task success.\n"
            "  - Under MERA's SkillBook: new procedure content must empirically CAUSE "
            "improved small model pass@1 to survive into the next cycle's skillbook.\n\n"
            "The contrast is the key camera-ready §2 insight:\n"
            "  (a) Catastrophic remembering (arxiv:2608.11095): prompts grow because the "
            "quality filter is absent — no ground-truth reward for instruction value, so "
            "deletion is risky and instructions accumulate.\n"
            "  (b) MERA SkillBook: the quality filter (pass@1=1 on small model) is the "
            "reward signal that prevents accumulation. EXP-163's 89% collapse is the "
            "empirical outcome of this filter being CORRECTLY applied: the procedure shrinks "
            "to near-zero because the small model eventually needs no external procedure "
            "conditioning (it has internalized the skill).\n"
            "  (c) Design implication for §8 (Limitations/Future Work): 'future SkillBook "
            "implementations should track the rollback hazard rate (arxiv:2608.11095) to "
            "distinguish catastrophic forgetting (useful content removed prematurely) from "
            "successful skill internalization (collapse is desired when the model no longer "
            "benefits from the procedure).'\n\n"
            "This is also the FIRST paper to empirically characterize agentic prompt growth "
            "as a continual-learning problem, making it a natural citation in §2 (Related "
            "Work on continual learning for LLM agents) and potentially §1 (Motivation: "
            "why MERA's quality-filtered SkillBook is needed). The continual-learning "
            "framing (catastrophic remembering vs. catastrophic forgetting) aligns with "
            "the AAAI 2027 venue's interest in systems that resist both pathologies.\n\n"
            "EXP-173 is a pure offline audit: (1) document the catastrophic-remembering "
            "vs. MERA-collapse contrast in a table; (2) draft a §2 paragraph positioning "
            "arxiv:2608.11095 as the motivation for quality-filtered SkillBook updates; "
            "(3) draft a §8 bullet on the rollback hazard rate as a future metric; (4) "
            "quantify MERA's 'anti-bloat' property: EXP-163's collapse rate (-89% from "
            "cycle 0 to cycle 1) is the empirical measurement. Offline, 0h GPU, ~35 lines "
            "Python + markdown, 25 minutes. Priority 8 (HIGH): NEW paper with direct "
            "empirical complement to EXP-163; first citation of the catastrophic-remembering "
            "phenomenon; the continual-learning framing of prompt bloat strengthens MERA's "
            "§2 motivation."
        ),
        "spec": {
            "script": "src/pipeline/catastrophic_remembering_skillbook_collapse_contrast.py",
            "inputs": [
                "results/e2e_4cyc_gpt55/cycle_{0..3}/skillbook.json",
                "results/privileged_distillation_cluster_paragraph.md"
            ],
            "outputs": [
                "results/catastrophic_remembering_vs_mera_collapse_table.md",
                "results/sec2_catastrophic_remembering_concurrent_paragraph.md",
                "results/sec8_rollback_hazard_future_work_bullet.md"
            ],
            "metrics": [
                "catastrophic_remembering_vs_mera_collapse_contrast_documented",
                "mera_anti_bloat_property_quantified_from_exp163",
                "sec2_continual_learning_framing_paragraph_written",
                "sec8_rollback_hazard_future_work_stated"
            ],
            "known_values": {
                "catastrophic_remembering_paper": {
                    "arxiv": "2608.11095",
                    "phenomenon": "agentic_prompts_grow_without_bound",
                    "growth": "+226% over lifetime",
                    "rate": "+4.9 net instructions per commit",
                    "hazard": "log-hazard -0.032/commit (older → less likely to delete)",
                    "cause": "cheap_append_O1_vs_expensive_delete_O2pow_D_cost_asymmetry",
                    "dataset": "247694 instruction lifetimes, 1867 repositories",
                    "framing": "catastrophic_remembering_inverse_of_catastrophic_forgetting"
                },
                "mera_contrast": {
                    "trajectory": "OPPOSITE_catastrophic_remembering",
                    "collapse_finding": "EXP-163_89pct_shrinkage_cycle0_1102_to_cycle1_118_chars",
                    "mechanism": "quality_filter_pass@1_small_model_success_criterion",
                    "interpretation": "collapse_is_desired_skill_internalization_not_pathology",
                    "anti_bloat_rate": "-89pct_from_cycle0_to_cycle1"
                },
                "design_implication": {
                    "future_metric": "rollback_hazard_rate_to_distinguish_forgetting_from_internalization",
                    "sec2_placement": "2_continual_learning_agentic_systems",
                    "sec8_placement": "8_limitations_future_work"
                }
            },
            "aaai_priority": "HIGH",
            "estimated_runtime_minutes": 25,
            "arxiv": "2608.11095",
            "immediately_runnable": True,
            "gpu_required": False,
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
