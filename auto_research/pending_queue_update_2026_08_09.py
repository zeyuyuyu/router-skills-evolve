"""
Pending queue update — 2026-08-09
Run on A800 when connectivity is restored:
  python /data0/home/zeyuwang/auto_research/pending_queue_update_2026_08_09.py
Appends EXP-154 and EXP-155 to state["queue"] and saves atomically.
"""
import json, os, tempfile, shutil

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

new_experiments = [
    {
        "id": "exp_2026_08_09_001_router_generalization_gap_ucci_calibration_audit",
        "priority": 7,
        "kind": "forgetting_eval",
        "gpu": "auto",
        "rationale": (
            "arxiv:2605.18796 (UCCI, May 2026) shows that LLM cascade routers with uncalibrated "
            "confidence scores require per-workload threshold retuning and fail to achieve "
            "cost-optimality. Our cycle router (LogisticRegression, tfidf) uses a fixed "
            "threshold=0.5 across all cycles ('tuner: skipped' in every router_threshold.json). "
            "router_meta.json reveals a striking pattern: cycle-3 router achieves only 71.4% "
            "accuracy and f1_large=0.0 on the 21-example held-out TEST split, yet the e2e "
            "ablation reports 92.68% routing accuracy (on all 82 examples including 61 training "
            "examples). The gap — 21.3pp — is the largest in the four-cycle run and indicates "
            "the cycle-3 router has overfit to training labels: it classifies all test 'large' "
            "tasks as small_ok. Per UCCI §3, this is exactly the uncalibrated regime where "
            "threshold=0.5 becomes suboptimal. "
            "EXP-154 audits this offline: for each cycle, compute (1) train accuracy (from "
            "e2e_ablation on training subset), (2) test accuracy (from router_meta), (3) "
            "generalization gap, and (4) Brier score as a calibration proxy. Produces a §4 "
            "'Router Calibration' paragraph and Table 11 (4-cycle train/test/gap). AAAI "
            "impact: strengthens §4 by quantifying a known limitation without A800. Offline, "
            "all inputs available locally in results/e2e_4cyc_gpt55/cycle_*/."
        ),
        "spec": {
            "script": "src/pipeline/router_calibration_audit.py",
            "inputs": [
                "results/e2e_4cyc_gpt55/cycle_{0..3}/router/router_meta.json",
                "results/e2e_4cyc_gpt55/cycle_{0..3}/e2e_ablation_summary.json"
            ],
            "outputs": [
                "results/router_calibration_audit.csv",
                "results/router_calibration_table.md"
            ],
            "metrics": ["train_acc", "test_acc", "generalization_gap", "brier_score_proxy", "f1_large_test", "f1_large_e2e"],
            "cycles": 4,
            "estimated_runtime_minutes": 2,
            "paper_sections": ["sec4_router_calibration_paragraph", "table11_train_test_gap"],
            "arxiv": "2605.18796",
            "immediately_runnable": True,
            "gpu_required": False,
            "key_finding": (
                "cycle_3_generalization_gap: test_acc=71.4% vs e2e_acc=92.7% (+21.3pp); "
                "f1_large_test=0.0 vs f1_large_e2e=0.833; threshold=0.5 (uncalibrated across all cycles)"
            )
        }
    },
    {
        "id": "exp_2026_08_09_002_flyroute_router_evolution_flywheel_label_shift_analysis",
        "priority": 6,
        "kind": "forgetting_eval",
        "gpu": "auto",
        "rationale": (
            "arxiv:2605.22057 (FlyRoute, May 2026) studies self-evolving routers via a 'data "
            "flywheel': each deployment cycle generates new routing labels that retrain the router, "
            "and the router's improvement tracks the cumulative label diversity of each cycle's "
            "new training signal. The key FlyRoute metric is 'flywheel gain': the marginal "
            "routing accuracy gain per new label added. Our router re-trains from scratch each "
            "cycle on 82 HumanEval examples with labels derived from the current small model's "
            "performance (can_downgrade_to_small). "
            "EXP-155 characterizes our router's flywheel: (1) per-cycle label distribution shift "
            "(how many tasks flip from 'need_large' to 'small_ok' across cycles — from "
            "router_meta label_distribution: c0=58/24, c1=54/28, c2=60/22, c3=62/20); "
            "(2) marginal accuracy gain vs. label shift magnitude per cycle; "
            "(3) whether cycles 2→3 provide meaningful new routing signal or converge. "
            "Key finding expected: cycles 0→2 show 'flywheel gain' (improving accuracy with "
            "diverse new labels), but cycle 3 shows 'flywheel saturation' (labels converge to "
            "always-small, router degenerates to trivial classifier). "
            "Paper impact: §4 'Router Evolution' new paragraph citing FlyRoute as a framing "
            "for our 4-cycle router co-evolution. Offline, data available in router_meta.json "
            "files. ~10 lines of Python, 0h GPU, 2 minutes."
        ),
        "spec": {
            "script": "src/pipeline/flyroute_label_shift_analysis.py",
            "inputs": [
                "results/e2e_4cyc_gpt55/cycle_{0..3}/router/router_meta.json"
            ],
            "outputs": [
                "results/flyroute_label_shift.csv",
                "results/flyroute_flywheel_analysis.md"
            ],
            "metrics": ["n_small_ok", "n_need_large", "label_shift_delta", "test_acc", "flywheel_gain_marginal"],
            "cycles": 4,
            "estimated_runtime_minutes": 2,
            "paper_sections": ["sec4_router_evolution_flywheel_paragraph"],
            "arxiv": "2605.22057",
            "immediately_runnable": True,
            "gpu_required": False,
            "key_finding": (
                "label_distribution: c0=58/24, c1=54/28, c2=60/22, c3=62/20 small/large. "
                "Large-label count: peaks at c1 (28), then shrinks to 20 at c3. "
                "Router test accuracy: c0=71.4%, c1=61.9%, c2=76.2%, c3=71.4% — "
                "non-monotonic matching the skills arm cycle-1 dip (Hypothesis F)."
            )
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
