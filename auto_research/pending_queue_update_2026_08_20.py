"""
Pending queue update — 2026-08-20
Run on A800 when connectivity is restored:
  python /data0/home/zeyuwang/auto_research/pending_queue_update_2026_08_20.py
Appends EXP-174 and EXP-175 to state["queue"] and saves atomically.

A800 offline since 2026-05-14 (day 98). Queue ~178 pending (>20 cap → 2 today).
AAAI 2027 camera-ready window (2026-08-16 to ~2026-08-29), day 5.
GPU window: CLOSED. EXP-174 is OFFLINE / 0h GPU. EXP-175 is for when GPU returns.

NEW PAPERS TODAY:
  EXP-174: Self-Evolving Coding Agents survey (arxiv:2608.03392) × CODESKILL
           (arxiv:2605.25430) — taxonomy audit for camera-ready §2. The survey's
           five-axis taxonomy (memory, skills, tools, weights, collaboration) maps
           directly onto MERA's three-component co-evolution; auditing it yields
           a coherent §2 sentence covering what MERA evolves and what it doesn't.
           Priority 9 (CRITICAL — camera-ready §2 completeness, AAAI window day 5).
  EXP-175: G²RPO-A adaptive guided GRPO (arxiv:2508.13023) — inject SkillBook
           procedure as adaptive guidance during GRPO rollout generation; guidance
           weight annealed by current-cycle pass@1. Distinct from 2026-08-19's
           replay and hard-first curriculum experiments. GPU experiment, queued for
           when A800 comes online. Priority 7.

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
    python3 auto_research/pending_queue_update_2026_08_20.py  # EXP-179, EXP-180 (this file)
"""
import json, os, tempfile, shutil

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

new_experiments = [
    {
        "id": "exp_2026_08_20_001_self_evolving_coding_agents_survey_taxonomy_camera_ready_audit",
        "priority": 9,
        "kind": "forgetting_eval",
        "gpu": "auto",
        "rationale": (
            "arxiv:2608.03392 ('Self-Evolving Coding Agents', August 2026) presents a structured "
            "taxonomy and survey of coding agents that update their own behavior from repository "
            "interactions. The taxonomy organizes what evolves (memory, skills, tools, model "
            "weights, collaboration structure), how evolution is triggered (external feedback, "
            "in-context reflection, on-policy reward), and how updates are stored and applied "
            "across invocations. It distinguishes static post-deployment agents from agents that "
            "continuously refine capabilities.\n\n"
            "MERA's three-component co-evolution maps onto this taxonomy at every axis:\n"
            "  - 'skills' axis: SkillBook (procedure text updated each cycle via LLM distillation)\n"
            "  - 'model weights' axis: SFT → GRPO checkpoint sequence per cycle\n"
            "  - 'collaboration structure' axis: TF-IDF + LogReg router updated each cycle\n"
            "  - Evolution trigger: on-policy reward (pass@1 via pytest execution)\n"
            "  - Update storage: joblib (router), JSONB (skillbook), LoRA checkpoint (model)\n\n"
            "What MERA does NOT evolve per the taxonomy: tool set (fixed Python subprocess), "
            "memory (no persistent per-task episodic memory across cycles), agent communication "
            "structure (single model, no multi-agent negotiation). These gaps can be turned into "
            "future-work bullets in camera-ready §6.\n\n"
            "Complementary paper: arxiv:2605.25430 ('CODESKILL: Learning Self-Evolving Skills "
            "for Coding Agents', May 2026) reformulates skill extraction and maintenance as a "
            "learnable management policy trained with RL (dense rubric-based skill quality + "
            "sparse execution reward). CODESKILL's RL-trained management policy is the closest "
            "external analog to MERA's inline Python distillation loop. The key differentiator: "
            "MERA does NOT train the distiller — it uses a fixed GPT-5.5 distiller with a "
            "binary success filter. CODESKILL's approach is more adaptive but requires an "
            "additional RL training stage.\n\n"
            "EXP-174 is a pure offline audit (0h GPU, ~30min, ~50 lines Python + markdown): "
            "(1) Map MERA onto the Self-Evolving Coding Agents five-axis taxonomy; (2) identify "
            "gaps (tool, memory, multi-agent axes) as future work; (3) contrast MERA's distiller "
            "with CODESKILL's RL-trained management policy; (4) draft a §2 paragraph covering "
            "both papers as 'Self-Evolving Code Agent' related work; (5) verify no contradiction "
            "with the existing §2 privileged-distillation and Scaffold-Mediated paragraphs. "
            "Priority 9 (CRITICAL): camera-ready §2 currently lacks a 'self-evolving agent' "
            "cluster; this is the highest-overlap paper of the week with MERA's framing."
        ),
        "spec": {
            "script": "src/pipeline/self_evolving_survey_taxonomy_camera_ready_audit.py",
            "inputs": [
                "auto_research/reports/paper_draft.md",
                "results/e2e_4cyc_gpt55/",
            ],
            "papers": [
                {"arxiv_id": "2608.03392", "role": "taxonomy_anchor"},
                {"arxiv_id": "2605.25430", "role": "skill_management_comparison"},
            ],
            "outputs": [
                "auto_research/reports/taxonomy_audit_2026_08_20.md",
                "auto_research/reports/sec2_self_evolving_paragraph_draft.md",
            ],
            "gpu_hours": 0,
        },
    },
    {
        "id": "exp_2026_08_20_002_grpo_adaptive_guided_procedure_annealing_g2rpo_a",
        "priority": 7,
        "kind": "grpo_continual",
        "gpu": "auto",
        "rationale": (
            "arxiv:2508.13023 ('G²RPO-A: Guided Group Relative Policy Optimization with "
            "Adaptive Guidance', August 2025, Guo et al., CUHK-SZ) identifies a fundamental "
            "failure mode of vanilla GRPO on small models: if the model cannot generate any "
            "correct rollout for a problem, the reward signal is uniformly zero, the advantage "
            "is zero, and no gradient flows. G²RPO-A injects ground-truth reasoning steps as "
            "'guidance' into GRPO rollouts and uses a learned adaptive controller to anneal "
            "guidance strength as training progresses.\n\n"
            "MERA's connection: our SkillBook procedure is already injected at inference and SFT "
            "time (format: `procedure\\n\\n---\\n\\nproblem`). However, during GRPO rollout "
            "generation (Phase 3b), the procedure prefix is present but STATIC — the same "
            "procedure appears in every rollout regardless of whether the model is struggling. "
            "G²RPO-A suggests a complementary adaptive angle: use cycle N's pass@1 as a proxy "
            "for guidance need, and increase procedure injection strength (e.g., repeat the "
            "procedure prefix or append a more detailed step-by-step scaffold) for problems "
            "where pass@1 < threshold, while using the standard prefix for problems where "
            "pass@1 >= threshold.\n\n"
            "This is DISTINCT from exp_2026_08_19_002 (curriculum hard_first), which reorders "
            "training batches by difficulty. G²RPO-A-style adaptive guidance changes the "
            "CONTENT of the prompt for hard problems, not just their scheduling order.\n\n"
            "Proposed spec: continue from cycle-3 GRPO adapter; split HumanEval problems into "
            "hard (cycle-3 pass@1 < 0.5) and easy (>=0.5); hard problems receive a double "
            "procedure prefix ('read the following procedure twice, then solve') or an augmented "
            "scaffold with one worked example from the SkillBook's successful traces; easy "
            "problems use standard procedure prefix; guidance annealing: cycle 5 reduces hard "
            "scaffold to 1.5× (single prefix + first 2 steps only), cycle 6 uses standard "
            "prefix for all. Expected: higher pass@1 on tail problems without harming easy ones; "
            "compare against exp_2026_08_19_001 (replay) and exp_2026_08_19_002 (hard_first)."
        ),
        "spec": {
            "base_model": "05_qwen3_5_4b_273",
            "from_checkpoint": "results/e2e_4cyc_gpt55/cycle_3/grpo_adapter",
            "bench": "humaneval",
            "n_cycles": 3,
            "grpo_temperature": 0.9,
            "grpo_batch_size": 1,
            "n_generations": 8,
            "algo": "dapo",
            "adaptive_guidance": True,
            "guidance_threshold_pass1": 0.5,
            "guidance_anneal_schedule": {
                "cycle_4": {"hard_prefix_repeat": 2, "hard_scaffold_steps": "all"},
                "cycle_5": {"hard_prefix_repeat": 1, "hard_scaffold_steps": 2},
                "cycle_6": {"hard_prefix_repeat": 1, "hard_scaffold_steps": 0},
            },
            "cost_target": 0.30,
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
