"""
Pending queue update — 2026-08-08
Run on A800 when connectivity is restored:
  python /data0/home/zeyuwang/auto_research/pending_queue_update_2026_08_08.py
Appends EXP-152 and EXP-153 to state["queue"] and saves atomically.
"""
import json, os, tempfile, shutil

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

new_experiments = [
    {
        "id": "exp_2026_08_08_001_sgsd_multiskill_complexity_cluster_arm_advantage_counterfactual",
        "priority": 7,
        "kind": "forgetting_eval",
        "gpu": "auto",
        "rationale": (
            "arxiv:2605.28791 (SGSD, May 2026) shows that task-complexity stratification within "
            "a single skill domain reveals differential arm advantages: RL (GRPO) outperforms "
            "distillation on medium-difficulty tasks (pass@1 in [0.2, 0.8]) but underperforms on "
            "easy tasks (forgetting) and hard tasks (zero-variance collapse). Our pipeline uses "
            "ONE global skill ('coding') per design decision. EXP-152 audits this post-hoc: "
            "partition 82 HumanEval tasks by cycle-0 pass@1 into easy/medium/hard tiers; compute "
            "per-tier pass@1 for all 4 ablation arms across cycles. Provides §2 SGSD citation "
            "anchor + §8 Future Work multi-skill tiering paragraph. Offline, 0h GPU."
        ),
        "spec": {
            "script": "src/pipeline/sgsd_complexity_counterfactual.py",
            "inputs": [
                "results/e2e_4cyc_gpt55/cycle_{0..3}/e2e_ablation_summary.json"
            ],
            "outputs": [
                "results/sgsd_complexity_stratified_arm_advantage.csv",
                "results/sgsd_complexity_arm_advantage_table.md"
            ],
            "easy_threshold": 0.8,
            "hard_threshold": 0.2,
            "arms": ["large", "skills", "router", "full"],
            "cycles": 4,
            "estimated_runtime_minutes": 2,
            "paper_sections": ["sec2_related_work_sgsd_cite", "sec8_future_work_multiskill"],
            "arxiv": "2605.28791",
            "immediately_runnable": True,
            "gpu_required": False
        }
    },
    {
        "id": "exp_2026_08_08_002_cort_token_advantage_surrogate_teacher_logprob_feasibility",
        "priority": 6,
        "kind": "forgetting_eval",
        "gpu": "auto",
        "rationale": (
            "arxiv:2607.25659 (CoRT, Jul 28 2026) redistributes GRPO response-level advantage to "
            "token level via teacher-student log-likelihood contrast, gaining +4.4pp. Full CoRT "
            "requires teacher log-probs; GPT-5.5 does not expose these. However, cycle-3 "
            "llm_adapter/checkpoint-best has absorbed 4 cycles of GPT-5.5 SFT distillation and "
            "approximates the teacher distribution on in-distribution tasks. EXP-153 uses this "
            "surrogate: for 20 randomly sampled (task, student-rollout) pairs from cycle-0 "
            "traces.jsonl, compute per-token logprobs under the cycle-3 checkpoint (frozen, "
            "CPU, no gradient). Categorize by token type (keyword/operator/identifier/literal/"
            "whitespace). Key question: CoRT weight variance > 2x across categories? "
            "If yes: token-level redistribution is feasible → §8 Future Work paragraph. "
            "Offline, ~5min CPU, no GPU."
        ),
        "spec": {
            "script": "src/pipeline/cort_token_feasibility.py",
            "inputs": [
                "results/e2e_4cyc_gpt55/cycle_0/traces.jsonl",
                "results/e2e_4cyc_gpt55/cycle_3/llm_adapter/checkpoint-best"
            ],
            "outputs": [
                "results/cort_token_logprob_distribution.csv",
                "results/cort_token_feasibility_summary.md"
            ],
            "n_sample_pairs": 20,
            "device": "cpu",
            "torch_dtype": "float16",
            "token_categories": ["keyword", "operator", "identifier", "literal", "whitespace"],
            "feasibility_threshold_variance_ratio": 2.0,
            "estimated_runtime_minutes": 5,
            "paper_sections": ["sec2_related_work_cort_cite", "sec8_future_work_token_credit"],
            "arxiv": "2607.25659",
            "immediately_runnable": True,
            "gpu_required": False
        }
    }
]

# Duplicate-check helper
def already_exists(state, exp_id):
    for item in state.get("queue", []) + state.get("history", []):
        if item.get("id") == exp_id:
            return True
    return False

# Load, append, save atomically
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
    print("No new experiments added (all already present).")
