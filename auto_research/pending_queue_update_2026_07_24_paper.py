#!/usr/bin/env python3
"""
Weekly paper pipeline queue patch — 2026-07-24 (EXP-132, EXP-133).

A800 connectivity: offline since 2026-05-14 (day ~71). Apply when restored.

Prior patches in chain: see pending_queue_update_2026_07_24.py header.
Queue was ~141 pending on 2026-07-24 (after EXP-130/131 from daily patch).

New papers motivating this run:
  arxiv:2606.21090 — Rise-and-Collapse failure mode in GRPO code training
  arxiv:2607.04364 — CPO "RL Forgets!": GRPO forgets SFT gains

AAAI 2027 deadline: 2026-08-15 (22 days from today 2026-07-24).
"""
import json
import os

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

new_experiments = [
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-132: Rise-and-Collapse Diagnostic — Best-Epoch Checkpoint Selection
    #          Tests Hypothesis E: "GRPO adapter is a collapsed checkpoint"
    #          (arXiv:2606.21090, June 2026)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_24_paper_001_rise_collapse_diagnostic_bestepoch_humaneval",
        "priority": 8,
        "gpu": "auto",
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2606.21090 — 'Self-Improvement Can Self-Regress: The Rise-and-Collapse "
            "Failure Mode in GRPO Code Training' (June 2026). Documents that pass@1 peaks "
            "at step 10-20 of GRPO training and then collapses toward baseline, as "
            "probability mass concentrates on the top-1 rollout. KL constraints and EWC do "
            "not prevent it. Hypothesis E: our cycle-3 GRPO adapter (and all prior cycle "
            "adapters) may be saved at the collapsed final checkpoint rather than the "
            "peak-step checkpoint, making Full = Router even if GRPO achieves real "
            "intermediate gains. If confirmed, the fix is trivial: save the best eval "
            "checkpoint within the GRPO run rather than the final epoch checkpoint. This "
            "is distinct from all other queued experiments: EXP-130 (CPO-PMP) addresses "
            "parameter-space forgetting; EXP-126 (GEPO) and others address group-level "
            "advantage computation. EXP-132 tests whether the problem is simply checkpoint "
            "timing — a zero-algorithm-change fix. Expected: if best-epoch Full arm "
            "> 92.68%, rise-and-collapse is confirmed and best-epoch saving closes the "
            "Full=Router gap without any GRPO algorithm change. Paper impact: adds "
            "Hypothesis E to §5.4 with quantitative support, and if confirmed, the fix is "
            "a 2-line code change and a new Table 9 row with best-epoch results."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 4,
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "grpo_eval_every_n_steps": 20,
            "grpo_save_every_n_steps": 20,
            "grpo_use_best_checkpoint": True,
            "grpo_best_checkpoint_metric": "task_pass_at_1",
            "grpo_temperature": 1.0,
            "n_generations": 8,
            "scaling_force_both": True,
            "sft_include_success": True,
            "analysis": [
                "grpo_pass_at_1_vs_step_per_cycle",
                "peak_step_vs_final_step_pass_at_1",
                "best_epoch_vs_final_epoch_full_arm_task_pass",
                "acr_fraction_per_step_per_cycle",
                "checkpoint_selection_full_vs_router_gap",
            ],
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (baseline): full=router=92.68% (final checkpoint)",
                "arxiv:2606.21090: peak at step 10-20, collapse within same run",
                "EXP-130 (CPO-PMP): parameter-space forgetting fix (orthogonal hypothesis)",
            ],
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
            ],
            "implementation_note": (
                "In grpo_train_simple.py, add per-step eval logging and best-checkpoint saving: "
                "1. Add env var GRPO_EVAL_EVERY_N_STEPS (default 0 = disabled; set to 20 for EXP-132). "
                "2. In training loop, after every N steps: "
                "   eval_pass = evaluate_pass_at_1(model, eval_tasks)  # 82 HE tasks "
                "   if eval_pass > best_eval_pass: "
                "     best_eval_pass = eval_pass "
                "     model.save_pretrained(save_path + '/checkpoint-best-epoch') "
                "     log(f'best checkpoint at step {step}: {eval_pass:.4f}') "
                "3. At end of GRPO: if GRPO_USE_BEST_CHECKPOINT: "
                "   copy checkpoint-best-epoch to grpo_adapter/ instead of final checkpoint. "
                "4. Log pass@1 per step to grpo_pass_at_1_log.csv for diagnostic analysis. "
                "Overhead: eval every 20 steps on 82 tasks x 1 pass = ~0.5 min per eval. "
                "With ~20 total GRPO steps, total overhead ~10 min per cycle (negligible). "
                "The pass@1 vs step curve (grpo_pass_at_1_log.csv) is the primary diagnostic: "
                "if it shows rise-then-fall, we have Hypothesis E confirmation. "
                "IMPORTANT: this must be zero-interference with existing GRPO runs — "
                "disabled by default (GRPO_EVAL_EVERY_N_STEPS=0)."
            ),
            "arxiv_ref": "2606.21090",
            "estimated_gpu_hours": 3.0,
        },
    },
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-133: Per-Cycle ACR-vs-Improvement Correlation Analysis (Offline)
    #          Uses existing grpo_info.json data from cycles 0-3
    #          Zero-GPU offline analysis → produces Table X for §5.3
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_24_paper_002_acr_improvement_correlation_offline",
        "priority": 7,
        "gpu": "none",
        "kind": "analysis_offline",
        "rationale": (
            "From grpo_info.json across 4 cycles (already locally available), the per-cycle "
            "ACR and skills arm data enable a quantitative correlation analysis: does lower "
            "ACR (more informative groups) predict larger skills arm improvement in the "
            "next cycle? Cycle data: (cycle, ACR%, skills_arm%) = "
            "(0, 51.2%, 70.73%), (1, 47.6%, 65.85%), (2, 46.3%, 73.17%), (3, 52.4%, 75.61%). "
            "Hypothesis: if ACR drives improvement, cycle 2 (lowest ACR 46.3%) should "
            "produce the largest cycle 2→3 improvement. Observation: cycle 2→3 skills arm "
            "+2.44pp (73.17%→75.61%), cycle 0→1 skills arm -4.88pp (70.73%→65.85%) despite "
            "ACR dropping from 51.2%→47.6%. This is ANTI-CORRELATED with the ACR hypothesis. "
            "Cycle 1 (ACR=47.6%, 2nd lowest) produces the WORST skills arm (65.85%). "
            "The ACR-driven explanation predicts monotonic improvement as ACR decreases; "
            "the actual data shows a non-monotonic pattern. This zero-GPU analysis: "
            "(a) produces the formal per-cycle ACR correlation table for §5.3, "
            "(b) provides quantitative support for Hypothesis D (GRPO forgetting) as an "
            "additional explanation beyond ACR, "
            "(c) requires no GPU — all data is in results/e2e_4cyc_gpt55/cycle_N/grpo_adapter/grpo_info.json."
        ),
        "spec": {
            "data_source": "results/e2e_4cyc_gpt55/cycle_N/grpo_adapter/grpo_info.json",
            "cycles": [0, 1, 2, 3],
            "metrics": ["n_dropped_zero_variance", "n_informative_groups", "n_rollouts_kept"],
            "output": [
                "acr_skills_correlation_table.md",
                "acr_vs_skills_improvement_plot.png",
            ],
            "analysis": [
                "pearson_r(ACR_cycle_N, skills_arm_gain_cycle_N_to_N+1)",
                "pearson_r(n_informative_groups, skills_arm_absolute)",
                "hypothesis_test: ACR-explanation vs forgetting-explanation",
            ],
            "expected_finding": (
                "Anti-correlation between ACR and immediate skills arm change "
                "(cycle 1 has lower ACR than cycle 0, but lower skills arm). "
                "Supports Hypothesis D: forgetting is the dominant factor in cycle 1 dip, "
                "not ACR. ACR reduction helps only when forgetting is controlled."
            ),
            "implementation": (
                "python3 auto_research/analyze_acr_correlation.py "
                "--cycles 0 1 2 3 "
                "--data-dir results/e2e_4cyc_gpt55 "
                "--output results/acr_correlation/"
            ),
            "estimated_gpu_hours": 0.0,
            "estimated_wall_hours": 0.1,
        },
    },
]


def apply_patch():
    if not os.path.exists(STATE_PATH):
        print(f"ERROR: {STATE_PATH} not found — apply on A800 when restored.")
        print("Experiments to queue:")
        for e in new_experiments:
            print(f"  {e['id']} (priority={e['priority']}, gpu={e['spec'].get('estimated_gpu_hours',0)}h)")
        return

    with open(STATE_PATH) as f:
        state = json.load(f)

    queue = state.setdefault("experiment_queue", [])
    existing_ids = {e["id"] for e in queue}
    added = []
    for exp in new_experiments:
        if exp["id"] not in existing_ids:
            queue.append(exp)
            added.append(exp["id"])
            print(f"  + queued {exp['id']}")
        else:
            print(f"  = already queued {exp['id']}")

    state["queue_size"] = len(queue)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)
    print(f"Done. Queue size: {len(queue)} (+{len(added)} new).")


if __name__ == "__main__":
    apply_patch()
