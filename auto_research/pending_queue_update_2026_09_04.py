"""
Pending queue update — 2026-09-04
Run on A800 when connectivity is restored:
  python /data0/home/zeyuwang/auto_research/pending_queue_update_2026_09_04.py
Appends EXP-208 and EXP-209 to state["queue"] and saves atomically.

A800 offline since 2026-05-14 (day ~113). SSH port 50507 unreachable from remote
execution environment (TCP timeout; proxy is HTTPS-only, cannot tunnel SSH).
Queue ~207 pending (>20 cap → 2 experiments today).
Next target: ICLR 2027 (~Oct 1 deadline, ~27 days out). URGENT.
Both experiments are OFFLINE / 0h GPU (paper positioning analyses for §2/§4).

Hotspot source: WebSearch fallback (A800 hotspot file unavailable — A800 offline).
Top new papers found (this run, 2026-09-04):
  arxiv:2602.00166  "Joint Continual Learning of Local Language Models and Cloud
    Offloading Decisions with Budget Constraints" (DA-GRPO, Feb 2026) — closest
    published system to MERA's joint local-cloud+continual design. Added to §2
    LLM Routing in paper v13 with differentiation paragraph. Bib key: dagrpo2026.
  arxiv:2607.20481  "Routing Without Training: Controllable-Ratio LLM Offloading via
    Reliability Gating" (July 2026) — 91% routing accuracy without gradient; near-
    competitive with MERA's 93.04% trained router. Added to §2 LLM Routing in
    paper v13 with EXP-209 framing. Bib key: routingwithouttraining2026.
  arxiv:2605.28791  "Skill-Conditioned Gated Self-Distillation for LLM Reasoning"
    (May 2026) — skill-gated selective distillation; intersects MERA's single-skill
    SkillBook design. Added to §2 Continual Learning in paper v13. Bib key:
    skillgateddistillation2026.

Apply chain before this patch:
    python3 auto_research/pending_queue_update_2026_09_03.py  # EXP-206, EXP-207
"""

import json, os, shutil, tempfile

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

NEW_EXPERIMENTS = [
    {
        "id": "EXP-208",
        "priority": 8,
        "title": (
            "DA-GRPO §2 Differentiation Paragraph Write: MERA vs. arxiv:2602.00166 "
            "for ICLR 2027 Reviewer Response Readiness"
        ),
        "paper": "arxiv:2602.00166",
        "paper_title": (
            "Joint Continual Learning of Local Language Models and Cloud Offloading "
            "Decisions with Budget Constraints (DA-GRPO)"
        ),
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": (
                "src/pipeline/train_router_simple.py — MERA router training on oracle labels; "
                "src/pipeline/collect_traces.py — _policy_decision routing logic; "
                "results/e2e_4cyc_gpt55/ — routing accuracy cycle progression; "
                "paper auto_research/paper/paper.md §2 LLM Routing (DA-GRPO paragraph added v13); "
                "paper auto_research/paper/paper.tex §2 (DA-GRPO paragraph added v13)"
            ),
            "metric": (
                "DA-GRPO (arxiv:2602.00166) is the closest published system to MERA's "
                "joint local-cloud continual design. The §2 differentiation paragraph in "
                "paper v13 (added 2026-09-04) makes three claims: "
                "(1) DA-GRPO conflates routing probability with RL gradient magnitude — "
                "    MERA decouples them via supervised router (Decision 2). "
                "(2) DA-GRPO has no procedure prefix / SkillBook distillation. "
                "(3) DA-GRPO does not perform N-cycle co-evolution. "
                "This audit verifies all three claims against the DA-GRPO paper, "
                "checks MERA's implementation for consistency, and prepares a concise "
                "rebuttal paragraph for the expected ICLR reviewer question: "
                "'How does MERA differ from [DA-GRPO]?' "
                "Output: rebuttal paragraph (2-3 sentences) + Table S1 comparison row "
                "added to supplementary (if page budget allows)."
            ),
            "estimated_gpu_hours": 0,
            "expected_paper_impact": (
                "Closes ICLR reviewer risk W6 (DA-GRPO comparison missing). "
                "Novelty +0.2 if the three differentiation claims hold. "
                "Provides rebuttal text that can be added to §2 or supplementary."
            ),
        },
        "rationale": (
            "EXP-204 (queued 2026-09-01) audited the DA-GRPO paper for positioning. "
            "EXP-208 executes the writing output: drafts the §2 differentiation text "
            "and prepares rebuttal material for the ICLR 2027 submission (deadline ~Oct 1). "
            "Addresses ICLR reviewer weakness W6."
        ),
        "iclr_target_section": "§2 LLM Routing / Supplementary Table S1",
    },
    {
        "id": "EXP-209",
        "priority": 8,
        "title": (
            "Routing Without Training §4 Framing Audit: MERA vs. arxiv:2607.20481 "
            "Reliability Gating Baseline for ICLR 2027"
        ),
        "paper": "arxiv:2607.20481",
        "paper_title": (
            "Routing Without Training: Controllable-Ratio LLM Offloading via Reliability Gating"
        ),
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": (
                "src/pipeline/train_router_simple.py — router training implementation; "
                "src/pipeline/collect_traces.py — trace collection + oracle labels; "
                "results/e2e_ablation_a800_20260509_summary.json — full ablation table "
                "   (Base: 68.28%, +Skills: 69.46%, +Router: 93.04%); "
                "paper auto_research/paper/paper.md §2 LLM Routing (routing-without-training "
                "   paragraph added v13); §4 router section; §6 E2E ablation."
            ),
            "metric": (
                "Routing Without Training (arxiv:2607.20481, July 2026) achieves 91% routing "
                "accuracy on a held-out routing benchmark via reliability gating: confidence "
                "threshold on the small model's own output probability distribution, without "
                "training any router weights. "
                "This is 2pp below MERA's 93.04%, but the 91% figure is on a *different* "
                "benchmark, making direct comparison invalid. "
                "Key audit questions: "
                "(1) Is MERA's 68.28% base (always-small + static escalation) equivalent "
                "    to what reliability gating would achieve on MERA's 848-example dataset? "
                "    If reliability gating achieves ~68-75% on MERA's data, then supervised "
                "    router training adds 18-25pp (strong claim). If ~85-90%, the gap shrinks. "
                "(2) Can we compute a reliability-gating estimate on MERA's data from "
                "    existing traces (small model confidence scores available in "
                "    results/e2e_4cyc_gpt55/)? "
                "(3) If yes, add a 'Reliability Gating' row to Table 4 (E2E ablation). "
                "    Expected: 75-80% (between Base and +Router), strengthening the "
                "    supervised-router contribution claim. "
                "Output: Table 4 new row (or Table 4 footnote) + §4 framing paragraph."
            ),
            "estimated_gpu_hours": 0,
            "expected_paper_impact": (
                "If reliability gating ~75-80% on MERA's data: adds critical baselines row "
                "to Table 4, strengthens §4 contribution claim. "
                "Soundness +0.2 (closes ICLR reviewer risk W7). "
                "If reliability gating ~90%: major threat to §4 contribution; need to "
                "reframe supervised router as a fine-tuning step on top of reliability gating."
            ),
        },
        "rationale": (
            "Routing Without Training (arxiv:2607.20481) achieves 91% routing accuracy "
            "without training, competitive with MERA's 93.04% trained router. "
            "An ICLR reviewer will ask whether MERA's router training is necessary. "
            "EXP-209 performs an offline audit to estimate reliability-gating performance "
            "on MERA's own data, providing either (a) a new Table 4 row or (b) a §4 "
            "reframing. Addresses ICLR reviewer weakness W7."
        ),
        "iclr_target_section": "§4 Learned Router / Table 4",
    },
]


def main():
    with open(STATE_PATH, "r") as f:
        state = json.load(f)

    existing_ids = {e.get("id") for e in state.get("queue", [])}
    added = []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] not in existing_ids:
            state["queue"].append(exp)
            added.append(exp["id"])
        else:
            print(f"  SKIP {exp['id']} — already in queue")

    if added:
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        shutil.move(tmp, STATE_PATH)
        print(f"  Added: {', '.join(added)}")
    else:
        print("  No new experiments added.")


if __name__ == "__main__":
    main()
