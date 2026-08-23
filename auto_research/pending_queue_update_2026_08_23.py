"""
Pending queue update — 2026-08-23
Run on A800 when connectivity is restored:
  python /data0/home/zeyuwang/auto_research/pending_queue_update_2026_08_23.py
Appends EXP-185 and EXP-186 to state["queue"] and saves atomically.

A800 offline since 2026-05-14 (day ~101). Queue ~184 pending (>20 cap → 2 today).
AAAI 2027 camera-ready window (2026-08-16 to ~2026-08-29), day 8. 6 days remain.
GPU window: CLOSED. EXP-186 needs GPU (queued for restoration).

NEW PAPERS TODAY:
  EXP-185: P2Skill — Prompt-Based Skill Distillation for Cloud-Local Routing (arxiv:2608.14094)
           Validates MERA's procedure-prefix approach; audit structured decompose-route-reconstruct
           skill prompt vs. MERA's single procedure on HumanEval hard tasks. Offline, priority 8.
  EXP-186: CVPO — Value-Variance Advantage Adjustment for GRPO (arxiv:2608.03068)
           Variance-aware advantage scaling + dynamic curriculum; addresses ACR=52.4% zero-variance
           groups in MERA's Phase 3b. GPU required. Priority 7.

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
    python3 auto_research/pending_queue_update_2026_08_22.py  # EXP-183, EXP-184
    python3 auto_research/pending_queue_update_2026_08_23.py  # EXP-185, EXP-186 (this file)
"""
import json, os, tempfile, shutil

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

new_experiments = [
    {
        "id": "exp_2026_08_23_001_p2skill_prompt_skill_distillation_audit",
        "priority": 8,
        "kind": "offline_analysis",
        "gpu": "none",
        "rationale": (
            "arxiv:2608.14094 ('P2Skill: Privacy Preserving Skill Distillation for Cloud-Local "
            "LLM Inference Systems'). Validates MERA's procedure-prefix approach from a different "
            "angle: structured skill prompts (decompose → PII-route → paraphrase → reconstruct) "
            "vs. MERA's single `coding` procedure prefix. Audit on HumanEval hard tasks (small "
            "model fails without procedure): compare pass@1 across skill prompt schemas. "
            "Offline analysis, no GPU. Supports §3 Method in camera-ready."
        ),
        "config": {
            "bench": "humaneval",
            "skill_schema": "p2skill_decompose_reconstruct",
            "baseline_schema": "mera_single_procedure",
            "eval_subset": "hard_tasks_only",
            "model": "Qwen2.5-Coder-1.5B",
        },
        "paper": "arxiv:2608.14094",
        "aaai_relevance": "§3 Method — skill prompt design validation",
    },
    {
        "id": "exp_2026_08_23_002_cvpo_value_variance_grpo_advantage",
        "priority": 7,
        "kind": "grpo_training",
        "gpu": "required",
        "rationale": (
            "arxiv:2608.03068 ('CVPO: Enhancing LLM Reinforcement Learning Reasoning via "
            "Value-Variance Adaptation and Dynamic Curriculum Learning'). Adds value-variance "
            "term to GRPO advantage and dynamic curriculum weighting by problem difficulty. "
            "Addresses MERA Phase 3b ACR=52.4% zero-variance groups (same issue as EXP-184 but "
            "complementary approach: variance-scaling vs. latent credit). Implement as patch to "
            "src/pipeline/grpo_train_simple.py. Run GRPO_BATCH_SIZE=1. Compare pass@1 vs. "
            "standard GRPO. GPU required — queued for GPU window restoration."
        ),
        "config": {
            "bench": "humaneval",
            "grpo_variant": "cvpo_value_variance",
            "GRPO_BATCH_SIZE": 1,
            "GRPO_TEMPERATURE": 1.0,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        "paper": "arxiv:2608.03068",
        "aaai_relevance": "§4 GRPO ablation — positions ACR issue in literature",
    },
]


def main():
    if not os.path.exists(STATE_PATH):
        print(f"ERROR: {STATE_PATH} not found. Run on A800.")
        return

    with open(STATE_PATH) as f:
        state = json.load(f)

    existing_ids = {e["id"] for e in state.get("queue", [])}
    existing_ids |= {e["id"] for e in state.get("running", [])}
    existing_ids |= {e.get("id", "") for e in state.get("history", [])}

    added = 0
    for exp in new_experiments:
        if exp["id"] in existing_ids:
            print(f"SKIP (already present): {exp['id']}")
            continue
        state.setdefault("queue", []).append(exp)
        print(f"QUEUED: {exp['id']} (priority={exp['priority']})")
        added += 1

    if added == 0:
        print("No new experiments added.")
        return

    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    shutil.move(tmp, STATE_PATH)
    print(f"state.json updated: {added} experiments added.")


if __name__ == "__main__":
    main()
