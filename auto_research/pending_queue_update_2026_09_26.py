"""
Pending queue update — 2026-09-26
Run on A800 when connectivity is restored:
  python /data0/home/zeyuwang/auto_research/pending_queue_update_2026_09_26.py
Appends EXP-218 and EXP-219 to state["queue"] and saves atomically.

A800 offline since 2026-05-14 (day ~135). SSH port 50507 unreachable from remote
execution environment (TCP timeout; proxy is HTTPS-only and cannot tunnel raw SSH).
Auto-mode SSH classifier also blocks sshpass with embedded credentials.
Queue ~213+ pending (>20 cap → 2 experiments today).
Next target: ICLR 2027 (~Oct 1 deadline, ~5 days out). FINAL SPRINT.
Both experiments are OFFLINE / 0h GPU (paper consistency + routing baseline analyses).

Context (2026-09-26):
  ICLR deadline: ~Oct 1 (5 days). Paper v13 composite 7.0/10.
  Binding blockers: W1 single-seed (needs A800), CERA-MoA differentiation (EXP-216, queued
  2026-09-21, offline), Abstract novelty claims may still say "first" (pre-CERA-MoA language).
  Recent paper additions since v13: EXP-210 (GRPO-PRM §3), EXP-211 (DSR §2), EXP-212/213
  (queued 2026-09-12), EXP-214/215 (pytest β=0 / CAUC, 2026-09-16), EXP-216/217
  (CERA-MoA / ExecuCritic, 2026-09-21).

Hotspot source: Local analysis (A800 hotspot file unavailable — A800 offline).
Top angles identified (this run, 2026-09-26):
  1. Abstract/Conclusion consistency risk: v13 Abstract and Conclusion were written before
     CERA-MoA (arxiv:2609.18779) and the Sep 2026 cluster of co-evolutionary routing papers.
     "First co-evolutionary" language must be softened to avoid reviewer rejection on novelty
     grounds (CERA-MoA is concurrent, Sep 16). → EXP-218: 5-Day Pre-Submission Abstract/
     Conclusion Consistency Audit.
  2. Chain-of-thought token budget as zero-training router proxy: a trend in late-Sep 2026
     is using the model's own generation budget (CoT token count at inference time) as a
     difficulty signal — long CoT → escalate. MERA's traces already contain CoT lengths;
     estimating whether CoT-length routing achieves >75% accuracy on MERA's 848-example
     router dataset would add a §4 baseline column alongside Routing Without Training
     (EXP-209). → EXP-219: CoT-Length Zero-Training Router Proxy Baseline Estimate.

Apply chain before this patch:
    python3 auto_research/pending_queue_update_2026_09_11.py  # EXP-210, EXP-211
    python3 auto_research/pending_queue_update_2026_09_12.py  # EXP-212, EXP-213
    python3 auto_research/pending_queue_update_2026_09_16.py  # EXP-214, EXP-215
    python3 auto_research/pending_queue_update_2026_09_21.py  # EXP-216, EXP-217
"""

import json
import os
import shutil

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

NEW_EXPERIMENTS = [
    {
        "id": "EXP-218",
        "priority": 9,
        "title": (
            "ICLR 5-Day Pre-Submission Abstract/Conclusion Consistency Audit — "
            "Concurrent-Work Softening & §2 Integration Check (CERA-MoA cluster)"
        ),
        "paper": "arxiv:2609.18779",
        "paper_title": (
            "CERA-MoA: Co-Evolving Routing Mechanisms with Continually Learning LLM Agents"
        ),
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": (
                "auto_research/paper/paper.md — Abstract, §1 Introduction, §7 Discussion / "
                "   Conclusion: look for 'first', 'novel', 'to our knowledge', 'unique', or "
                "   'no prior work' claims about co-evolutionary router+skill design. "
                "auto_research/paper/paper.tex — identical audit of \\Abstract and §7. "
                "auto_research/paper/references.bib — verify bib keys exist: cera2026, "
                "   execucritic2026 (from EXP-216/217), dagrpo2026, routingwithouttraining2026. "
                "auto_research/pending_queue_update_2026_09_21.py — EXP-216 draft for "
                "   CERA-MoA differentiation sentence and intro softening. "
                "results/e2e_4cyc_gpt55/ — routing accuracy cycle-0 ~80% → cycle-3 ~93.04%: "
                "   the co-evolutionary evidence; used in any rewritten novelty claim. "
                "CLAUDE.md design decisions 1-4: MERA's distinguishing properties for the "
                "   softened novelty sentence. "
                "auto_research/paper/reviews/review-2026-09-04.md — §2 additions list, "
                "   EXP-210 through EXP-217 integration status."
            ),
            "metric": (
                "ICLR novelty risk audit. CERA-MoA (arxiv:2609.18779, Sep 16, 2026, Jiang et al.) "
                "co-evolves a router and agent policies via iterative RL cycles — sharing MERA's "
                "paradigm. It was posted 15 days before the ICLR deadline. EXP-216 (queued "
                "2026-09-21) drafted a differentiation sentence and intro softening. This experiment "
                "executes EXP-216's §1/§2 tasks and also performs a cross-document consistency check "
                "covering EXP-210 through EXP-217 (7 positioning additions since v13): "
                ""
                "(A) Abstract + §1 novelty claim audit: "
                "  Does the Abstract claim MERA is 'the first to co-evolve router + skills + LLM'? "
                "  If so: rewrite as 'to our knowledge, the first to combine skill distillation "
                "  (written procedure prefix), multi-cycle SFT+GRPO, and supervised router "
                "  co-training in a single iterative pipeline' — a narrower, defensible claim that "
                "  CERA-MoA does not invalidate (CERA-MoA has no SkillBook, no SFT, no teacher "
                "  traces). "
                ""
                "(B) §2 Related Work integration check: "
                "  Confirm the following citations are present in paper.md §2: "
                "  - EXP-210: GRPO-PRM equivalence (arxiv:2509.21154) in §2 RL for Code or §3 "
                "  - EXP-211: DSR diversity-aware skill routing (arxiv:2609.05824) in §2 Self-Evolving "
                "  - EXP-214: Cheap Verifiers blind spot (arxiv:2609.01345) in §4 or §7 "
                "  - EXP-215: CAUC calibration-aware cascades (arxiv:2609.11446) in §2 LLM Routing "
                "  - EXP-216: CERA-MoA (arxiv:2609.18779) in §2 LLM Routing or concurrent-work note "
                "  - EXP-217: ExecuCritic (arxiv:2609.16604) in §2 RL for Code + §3 future-work note "
                "  For any missing: draft the 1-2 sentence addition and bib entry. "
                ""
                "(C) Conclusion consistency: "
                "  Conclusion must state the 3-cycle co-evolutionary gain (routing 80% → 93.04%). "
                "  CERA-MoA validates the paradigm rather than invalidating MERA — consider adding "
                "  'Concurrent work CERA-MoA [cera2026] validates the co-evolutionary paradigm; "
                "  MERA is distinguished by skill distillation from teacher traces and SFT.' "
                ""
                "(D) References.bib: confirm cera2026, execucritic2026, grpoprm2026, dsr2026, "
                "  cheapverifiers2026, cauc2026 bib entries are present or draft them. "
                ""
                "Output: list of (section, old text, replacement text) patches; updated bib entries. "
                "If all checks pass: write 'ICLR pre-flight: PASS — paper v13 consistent with "
                "EXP-210..217 integrations, Abstract softened for CERA-MoA.' "
            ),
            "estimated_gpu_hours": 0,
            "expected_paper_impact": (
                "CRITICAL (5-day deadline). "
                "If Abstract still contains pre-CERA-MoA 'first co-evolutionary' language: "
                "fixing it adds Novelty +0.1 (preempts reviewer objection, turns CERA-MoA "
                "into paradigm validation). If §2 integrations from EXP-210-217 are missing "
                "from the paper (only queued, not yet written): each completed addition adds "
                "Related-Work coverage +0.05. If all 7 integrations are already in the paper "
                "(best case): the audit confirms ICLR-readiness with zero score impact. "
                "This experiment has the highest expected value in the 5-day window because "
                "it closes the 'concurrent work blindspot' risk that could cause desk rejection "
                "on novelty grounds, and confirms the §2 coverage that reviewers will audit."
            ),
        },
        "gpu": "auto",
    },
    {
        "id": "EXP-219",
        "priority": 7,
        "title": (
            "CoT-Length Zero-Training Router Proxy: Baseline Accuracy Estimate "
            "on MERA's 848-Example Router Dataset for ICLR 2027 §4 (Frugal Thinking trend)"
        ),
        "paper": "arxiv:2609.11446",
        "paper_title": (
            "Calibration-Aware Uncertainty Cascades for Efficient Heterogeneous "
            "Model Collaboration"
        ),
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": (
                "results/e2e_4cyc_gpt55/cycle_3/ — router dataset (848 routing examples). "
                "   Each example has: raw_prompt, small_model_output, large_model_output, "
                "   oracle_route_label (small_correct → label=0, else label=1), "
                "   small_output_length (token count of small model's generation). "
                "src/pipeline/collect_traces.py — trace collection logic; oracle label source. "
                "src/pipeline/train_router_simple.py — feature extraction from raw prompts; "
                "   TF-IDF+LR achieves 93.04% on the 848-example dataset. "
                "auto_research/pending_queue_update_2026_09_16.py — EXP-215 which already "
                "   estimates a post-hoc calibration baseline (CAUC). "
                "auto_research/pending_queue_update_2026_09_04.py — EXP-209 which estimates "
                "   reliability-gating (Routing Without Training) baseline. "
                "CLAUDE.md design decision 3: Router trains on raw prompt, no procedure prefix."
            ),
            "metric": (
                "A September 2026 trend in efficient inference (related to 'frugal thinking' and "
                "budget-constrained generation papers, e.g., CAUC arxiv:2609.11446 and related "
                "work on cost-calibrated cascade routing) is to use the model's own output length "
                "as a proxy for task difficulty: a problem that requires a longer CoT is harder "
                "and should escalate. MERA's traces already contain small_model_output_length "
                "for every routing example. "
                ""
                "Baseline: route_to_large = (small_output_length > threshold_T). "
                "Sweep T from 10th to 90th percentile of small_output_length distribution. "
                "Report: (1) best-threshold accuracy on 848 examples; (2) precision/recall; "
                "(3) correlation between small_output_length and oracle_route_label (Pearson r). "
                ""
                "Comparison to MERA's existing §4 baselines: "
                "  - Routing Without Training (EXP-209): reliability-gating via confidence scores "
                "  - Random baseline: 50% "
                "  - TF-IDF+LR (MERA): 93.04% "
                ""
                "Expected outcome: CoT-length proxy likely achieves 55-70% routing accuracy "
                "(better than random but well below TF-IDF+LR), providing a natural §4 baseline "
                "row: 'Generation-Length Proxy: X% (zero-training, no feature extraction)'. "
                "If CoT-length achieves >85%: significant finding that simplifies the routing "
                "architecture and needs §4 framing revision. "
                ""
                "Implementation: Python on any machine with the router dataset. "
                "  import numpy as np; from sklearn.metrics import accuracy_score "
                "  lengths = [len(ex['small_output'].split()) for ex in traces] "
                "  labels = [ex['oracle_label'] for ex in traces] "
                "  for T in np.percentile(lengths, range(10, 91, 10)): "
                "      preds = [1 if l > T else 0 for l in lengths] "
                "      print(T, accuracy_score(labels, preds)) "
                ""
                "Output: Table row for §4 + correlation analysis + recommendation for §4 text."
            ),
            "estimated_gpu_hours": 0,
            "expected_paper_impact": (
                "§4 Router gains a new zero-training baseline row that complements EXP-209 "
                "(reliability gating) — distinguishing between 'no feature extraction needed' "
                "vs. 'no training needed'. If CoT-length accuracy ~60-70%: confirms MERA's "
                "93.04% learned router represents a substantial +20-33pp gain over all zero-training "
                "approaches, strengthening the §4 supervised-learning contribution. "
                "Soundness +0; Significance +0.05 (new baseline column); Related-Work 0. "
                "Fast to run (~5 minutes on local machine with the router dataset). "
                "Recommended as the second experiment to complete in the 5-day ICLR sprint "
                "(after EXP-218 abstract audit), if the router dataset JSON is accessible locally."
            ),
        },
        "gpu": "auto",
    },
]


def main():
    if not os.path.exists(STATE_PATH):
        print(f"ERROR: state.json not found at {STATE_PATH}")
        return

    tmp_path = STATE_PATH + ".tmp_0926"
    with open(STATE_PATH, "r") as f:
        state = json.load(f)

    existing_ids = set()
    for item in state.get("queue", []):
        existing_ids.add(item.get("id"))
    for item in state.get("history", []):
        existing_ids.add(item.get("id"))

    added = []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] in existing_ids:
            print(f"SKIP {exp['id']} — already in queue or history")
        else:
            state.setdefault("queue", []).append(exp)
            added.append(exp["id"])
            print(f"ADDED {exp['id']} (priority={exp['priority']}): {exp['title'][:60]}...")

    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    shutil.move(tmp_path, STATE_PATH)

    print(f"\nDone. Added {len(added)} experiments: {added}")
    print(f"Queue length: {len(state.get('queue', []))}")


if __name__ == "__main__":
    main()
