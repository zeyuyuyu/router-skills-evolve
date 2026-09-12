"""
Pending queue update — 2026-09-12
Run on A800 when connectivity is restored:
  python /data0/home/zeyuwang/auto_research/pending_queue_update_2026_09_12.py
Appends EXP-212 and EXP-213 to state["queue"] and saves atomically.

A800 offline since 2026-05-14 (day ~121). SSH port 50507 unreachable from remote
execution environment (TCP timeout; proxy is HTTPS-only, cannot tunnel SSH).
Queue ~209 pending (>20 cap → 2 experiments today).
Next target: ICLR 2027 (~Oct 1 deadline, ~19 days out). URGENT.
Both experiments are OFFLINE / 0h GPU (paper positioning analyses for §2/§3).

Hotspot source: WebSearch fallback (A800 hotspot file unavailable — A800 offline).
Top new papers found (this run, 2026-09-12):
  arxiv:2609.07786  "Signed Rescue Routing: Harm-Aware Cascades for Efficient LLM
    Inference" (Sep 7, 2026, Wang et al.) — Bayes-optimal LLM cascade routing via
    signed conditional gain (benefit of escalation minus harm). Directly in MERA's
    §2 LLM Routing space; not yet cited. MERA's oracle label already implements
    a coarser version of this signal. → EXP-212: SRR §2 Positioning Audit.
  arxiv:2609.04172  "Rethinking On-Policy Distillation of Large Language Models II:
    One Training Example" (Sep 4, 2026) — 1-query OPD reaches 71.5% state coverage;
    16 semantically distinct queries reach 98.9%, matching full-data OPD. Explains
    MERA's SFT collapse at 19 hard pairs (low coverage) vs. stability at 77 pairs
    (near-saturation). → EXP-213: OPD State Coverage §3 Framing Audit.
  arxiv:2609.05198  "What Matters in On-Policy Distillation?" (Sep 4, 2026) —
    harder examples with longer CoT paths give better gains than short easy examples;
    validates MERA's hard-task focus but complicates the Qwen3 no-CoT decision.
    Companion to EXP-213 (included in that audit's scope).

Apply chain before this patch:
    python3 auto_research/pending_queue_update_2026_09_11.py  # EXP-210, EXP-211
"""

import json, os, shutil

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

NEW_EXPERIMENTS = [
    {
        "id": "EXP-212",
        "priority": 8,
        "title": (
            "Signed Rescue Routing §2 LLM Routing Positioning: MERA vs. arxiv:2609.07786 "
            "Bayes-Optimal Cascade Routing for ICLR 2027"
        ),
        "paper": "arxiv:2609.07786",
        "paper_title": (
            "Signed Rescue Routing: Harm-Aware Cascades for Efficient LLM Inference"
        ),
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": (
                "src/pipeline/collect_traces.py — _policy_decision routing + oracle label "
                "   construction (lines ~90-160): label=1 (route-large) when small model "
                "   fails; label=0 (route-small) when small succeeds. This is effectively "
                "   the 'benefit of escalation' signal from the large oracle run. "
                "src/pipeline/train_router_simple.py — router training on oracle labels; "
                "   lines ~1-80: SVM/logistic regression on raw prompt features. "
                "paper auto_research/paper/paper.md §2 LLM Routing (references "
                "   arxiv:2607.20481 routing-without-training and arxiv:2602.00166 DA-GRPO; "
                "   SRR arxiv:2609.07786 not yet cited). "
                "results/e2e_4cyc_gpt55/ — routing accuracy 93.04% cycle-3; oracle label "
                "   distribution: ~25% large (hard tasks), ~75% small (easy tasks)."
            ),
            "metric": (
                "Signed Rescue Routing (SRR, arxiv:2609.07786, Sep 7 2026) proposes a "
                "Bayes-optimal LLM cascade routing score: the signed conditional gain "
                "P(large correct | small wrong) − P(large incorrect | small correct). "
                "Escalation helps when this score is positive (large rescues a failure) "
                "and hurts when it is negative (large degrades a correct answer). "
                "Connection to MERA: "
                "(1) MERA's oracle label (1 iff small_model fails the task) approximates "
                "    the first term P(large correct | small wrong) ≈ 1 for hard coding tasks "
                "    (the run-both oracle sets label=1 exactly when the small model fails AND "
                "    the large model succeeds, which is the beneficial-escalation case). "
                "    This is a coarser version of SRR's signed gain: MERA ignores the harm "
                "    term P(large incorrect | small correct) because gpt-5.5 almost never "
                "    regresses on HumanEval tasks where Qwen3-4B already passes. "
                "(2) SRR trains a static routing model; MERA's router co-evolves with the "
                "    student model across N cycles — as the student improves, what counts as "
                "    'easy' shifts, and MERA's oracle labels re-calibrate automatically. "
                "    SRR has no co-evolution mechanism. "
                "Audit tasks: "
                "  (a) Read SRR (arxiv:2609.07786) §3 routing model and verify that MERA's "
                "      oracle label is a special case of SRR's signed gain under the "
                "      near-zero harm assumption (P(large wrong | small right) ≈ 0 for gpt-5.5 "
                "      on HumanEval). Determine if this holds empirically in MERA's traces. "
                "  (b) Draft §2 LLM Routing positioning paragraph (3–4 sentences): "
                "      SRR is Bayes-optimal but static; MERA's oracle label is a zero-harm "
                "      approximation of SRR's signal that re-calibrates across training cycles. "
                "  (c) Check whether SRR's harm-aware framing could strengthen §4.2 "
                "      (Router Training): the routing accuracy claim (93.04%) is valid under "
                "      the zero-harm assumption; add a one-sentence caveat if harm is non-zero. "
                "  (d) Add bib entry signedrescuerouting2026. "
                "Output: §2 LLM Routing paragraph draft + §4.2 caveat sentence + bib entry."
            ),
            "estimated_gpu_hours": 0,
            "expected_paper_impact": (
                "Closes ICLR reviewer risk: Sep 7 paper directly at intersection of LLM "
                "cascades + routing that an LLM-routing reviewer would expect to see cited. "
                "Novelty: framing MERA's oracle label as a zero-harm approximation of the "
                "Bayes-optimal signed gain adds theoretical grounding to §4.2. "
                "Novelty +0.1 if the zero-harm assumption is verified empirically. "
                "§2 LLM Routing gains a new Sep 2026 citation, reducing the gap between "
                "submission date and camera-ready."
            ),
        },
        "rationale": (
            "arxiv:2609.07786 (Signed Rescue Routing, Sep 7 2026) is a brand-new paper on "
            "LLM cascade routing that defines the Bayes-optimal escalation score as a signed "
            "conditional gain. MERA's oracle label already implements the dominant term of "
            "this score (benefit of escalation), making the connection natural and strong. "
            "The paper is not yet cited in MERA's §2 LLM Routing section; with ICLR 2027 "
            "due in 19 days, any Sep 2026 LLM-routing paper uncited is a reviewer risk. "
            "Offline / 0h GPU; estimated 45 min to read and draft the paragraph."
        ),
        "iclr_target_section": "§2 LLM Routing / §4.2 Router Training",
    },
    {
        "id": "EXP-213",
        "priority": 7,
        "title": (
            "OPD State Coverage §3 SFT Framing: arxiv:2609.04172 + 2609.05198 — "
            "Explaining MERA SFT Collapse at 19 Pairs via Coverage Theory for ICLR 2027"
        ),
        "paper": "arxiv:2609.04172",
        "paper_title": (
            "Rethinking On-Policy Distillation of Large Language Models II: "
            "One Training Example"
        ),
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": (
                "src/pipeline/traces_to_sft.py — SFT data construction from oracle traces; "
                "   hard-tasks-only produces ~19 teacher pairs; SFT_INCLUDE_SUCCESS=1 "
                "   gives ~77 pairs (both documented in CLAUDE.md gotchas). "
                "src/pipeline/train_small_model.py — SFT training loop; "
                "   nan-grad steps occur at ~19 pairs with gradient accumulation. "
                "paper auto_research/paper/paper.md §3 SFT Training (currently cites "
                "   behavior cloning rationale for SFT_INCLUDE_SUCCESS=1 but lacks a "
                "   mechanistic explanation for why 19 pairs collapses). "
                "config/humaneval_dapo_gpt.yaml — SFT_INCLUDE_SUCCESS=1, "
                "   SCALING_FORCE_BOTH=1 flags documented as canonical run settings. "
                "results/e2e_4cyc_gpt55/ — SFT loss curves; confirm nan-grad at 19 pairs."
            ),
            "metric": (
                "Two companion Sep 2026 OPD papers provide a mechanistic account of MERA's "
                "SFT collapse: "
                "(1) arxiv:2609.04172 ('One Training Example', Sep 4): "
                "    State coverage is the fraction of rollout states visited by the "
                "    full-data training set that a subset also visits. A single query "
                "    reaches 71.5%; 16 semantically distinct queries reach 98.9% saturation. "
                "    MERA's ~19 hard-task pairs (after filtering for small-fail + large-success) "
                "    represent a SEMANTICALLY REDUNDANT subset: all are HumanEval coding "
                "    problems where Qwen3-4B fails. State coverage is likely near 71% "
                "    (single-query level), not 98.9%. This creates sparse gradient signal, "
                "    consistent with nan-grad steps under gradient accumulation with a small "
                "    effective batch. With SFT_INCLUDE_SUCCESS=1 (~77 pairs), the easier "
                "    problems add semantically distinct rollout states → coverage toward the "
                "    saturation regime → stable training. "
                "(2) arxiv:2609.05198 ('What Matters in OPD', Sep 4): "
                "    Harder examples yield better gains when they produce LONGER CoT paths. "
                "    MERA uses enable_thinking=False (Qwen3 no-CoT; CLAUDE.md gotcha: tau2 "
                "    corpus has no CoT). Hard MERA teacher traces have short outputs, not "
                "    long reasoning chains — so the 'hard examples are better' finding does "
                "    NOT apply to MERA in its current form. This limits the per-example gain "
                "    even when coverage is adequate. "
                "Audit tasks: "
                "  (a) Verify the coverage argument: count unique HumanEval problem IDs in "
                "      MERA's hard-task-only traces vs. success+fail traces. If hard-only "
                "      set covers <20 distinct HumanEval problems (out of 164), the semantic "
                "      redundancy / low-coverage claim holds. "
                "  (b) Draft §3 SFT Training footnote (3–4 sentences): 'State coverage "
                "      perspective on SFT stability: arxiv:2609.04172 shows 16 semantically "
                "      distinct OPD queries reach 98.9% state coverage; MERA's ~19 hard-task "
                "      traces likely cover fewer distinct states (restricted to failures of "
                "      Qwen3-4B on HumanEval). SFT_INCLUDE_SUCCESS=1 expands to ~77 traces "
                "      from both easy and hard problems, approaching saturation; this explains "
                "      the nan-grad collapse at 19 pairs and stability at 77.' "
                "  (c) Check §3: does MERA's paper currently acknowledge the CoT limitation "
                "      from arxiv:2609.05198? If not, add a one-sentence observation that "
                "      longer reasoning traces could further improve distillation quality — "
                "      a footnote (not a full contribution claim). "
                "  (d) Add bib entries opdonexample2026 and opdmatters2026. "
                "Output: §3 footnote draft + §3 CoT observation sentence + 2 bib entries."
            ),
            "estimated_gpu_hours": 0,
            "expected_paper_impact": (
                "§3 SFT Training gains a theoretically grounded explanation for the "
                "SFT_INCLUDE_SUCCESS=1 design decision, turning a pragmatic fix (avoids "
                "nan-grad) into a principled state-coverage argument. "
                "Soundness +0.1 (closes a gap reviewers may flag: 'Why does hard-only SFT "
                "collapse? Is it a data quality issue or a quantity issue?'). "
                "The CoT observation adds a forward-looking limitation sentence that "
                "could head off a reviewer asking 'Why not use chain-of-thought?'. "
                "Two new Sep 2026 citations fill the ICLR related-work gap."
            ),
        },
        "rationale": (
            "CLAUDE.md documents a known gotcha: 'SFT on hard-tasks-only collapses the "
            "model. With ~19 teacher pairs, grad_accum produces nan-grad steps and pass@1 "
            "drops to ~0. Always run with SFT_INCLUDE_SUCCESS=1 (~77 pairs, stable).' The "
            "paper §3 acknowledges this empirically but lacks a mechanistic explanation. "
            "Two new Sep 4 2026 papers (arxiv:2609.04172, 2609.05198) provide exactly this: "
            "state coverage theory explains why 19 semantically similar hard-task traces "
            "collapse (too-low coverage → sparse gradients), while 77 mixed traces stabilize "
            "(coverage near saturation). Both papers are Sep 2026, not yet in MERA. "
            "With ICLR 2027 in 19 days, a mechanistic §3 footnote adds soundness at low "
            "cost. Offline / 0h GPU; estimated 60 min."
        ),
        "iclr_target_section": "§3 SFT Training",
    },
]


def main():
    with open(STATE_PATH, "r") as f:
        state = json.load(f)

    existing_ids = {e.get("id") for e in state.get("queue", [])}
    existing_ids |= {e.get("id") for e in state.get("history", [])}
    added = []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] not in existing_ids:
            state["queue"].append(exp)
            added.append(exp["id"])
        else:
            print(f"  SKIP {exp['id']} — already in queue or history")

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
