"""
Pending queue update — 2026-09-13
Run on A800 when connectivity is restored:
  python /data0/home/zeyuwang/auto_research/pending_queue_update_2026_09_13.py
Appends EXP-214 and EXP-215 to state["queue"] and saves atomically.

A800 offline since 2026-05-14 (day ~122). SSH port 50507 unreachable from remote
execution environment (TCP timeout; proxy is HTTPS-only, cannot tunnel SSH).
Queue ~211 pending (>20 cap → 2 experiments today).
Next target: ICLR 2027 (~Oct 1 deadline, ~18 days out). URGENT.
Both experiments are OFFLINE / 0h GPU (paper §2/§3 positioning for new Sep 2026 papers).

Hotspot source: WebSearch fallback (A800 hotspot file unavailable — A800 offline).
Top new papers found (this run, 2026-09-13):
  arxiv:2609.11446  "Calibration-Aware Uncertainty Cascades for Efficient Heterogeneous
    Model Collaboration" (Sep 10, 2026, Zhang et al.) — CAUC replaces trained routers
    with post-hoc calibrated confidence cascades. Complementary positioning to SRR
    (EXP-212, arxiv:2609.07786). Was flagged as ★★ background cite candidate in
    2026-09-12 hotspot but not queued. → EXP-214: CAUC §2 LLM Routing Positioning Audit.
  arxiv:2609.07255  "SkillAlign: Aligning Skill Interfaces for LLM-based Agents"
    (Sep 2026, EMNLP 2026, Ren et al.) — exposure interface (full, hint, compressed,
    workflow) shifts success from 47.9% to 72.1% on ALFWorld. MERA's procedure format
    is a fixed exposure interface. Was deferred from 2026-09-11 as lower priority;
    EMNLP 2026 acceptance raises its citation authority. → EXP-215: SkillAlign §3
    SkillBook Interface Positioning Audit.

Apply chain before this patch:
    python3 auto_research/pending_queue_update_2026_09_11.py  # EXP-210, EXP-211
    python3 auto_research/pending_queue_update_2026_09_12.py  # EXP-212, EXP-213
"""

import json, os, shutil

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

NEW_EXPERIMENTS = [
    {
        "id": "EXP-214",
        "priority": 7,
        "title": (
            "CAUC §2 LLM Routing Cascade Positioning: arxiv:2609.11446 "
            "Calibration-Aware Uncertainty Cascades vs. MERA's Learned Co-Evolving Router "
            "for ICLR 2027"
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
                "src/pipeline/train_router_simple.py — trained router (logistic regression "
                "   on raw prompt features); routing accuracy 93.04% at cycle-3; "
                "   CLAUDE.md Design Decision #3: router trains on RAW prompt only. "
                "src/pipeline/collect_traces.py — oracle label construction: label=1 "
                "   when small model fails; label=0 when small succeeds. Trained labels, "
                "   not confidence-threshold labels. "
                "paper auto_research/paper/paper.md §2 LLM Routing — currently cites "
                "   arxiv:2609.07786 (EXP-212, SRR, Bayes-optimal signed gain), "
                "   arxiv:2607.20481 (routing-without-training). "
                "   CAUC (2609.11446) not yet cited. "
                "results/e2e_4cyc_gpt55/ — routing accuracy 93.04% cycle-3; "
                "   router update schedule (one trained router per cycle)."
            ),
            "metric": (
                "Zhang et al. (arxiv:2609.11446, Sep 10 2026) propose Calibration-Aware "
                "Uncertainty Cascades (CAUC), a post-hoc framework that calibrates each "
                "model's confidence independently and selects cascade thresholds on a common "
                "reliability scale using validation data. This eliminates the need for a "
                "trained router altogether — thresholds are derived from confidence scores "
                "alone, decoupling routing from any particular model pool or budget. "
                "Connection to MERA: "
                "(1) MERA's router IS a trained model (logistic regression on raw prompt "
                "    features, oracle labels from binary pass/fail). CAUC is post-hoc "
                "    (no training, no oracle labels); MERA's approach requires GPU runs "
                "    to generate oracle labels, while CAUC needs only calibration data. "
                "    The key MERA advantage: the trained router captures prompt-level "
                "    features (complexity, problem structure) that raw model confidence "
                "    does not, especially at cycle 0 when the small model is unspecialized. "
                "(2) MERA's router co-evolves with the student across N cycles. CAUC's "
                "    thresholds are static per deployment; there is no mechanism to "
                "    re-calibrate as the small model improves. MERA's cycle-by-cycle "
                "    re-training is the key differentiator. "
                "(3) SRR (EXP-212, 2609.07786) and CAUC (2609.11446) are companion papers "
                "    in the §2 cascade-routing space: SRR defines the optimal escalation "
                "    score, CAUC defines how to set thresholds without training. MERA's "
                "    router uses a trained score (oracle labels) with trained thresholds — "
                "    neither purely SRR nor purely CAUC, but extending both with co-evolution. "
                "Audit tasks: "
                "  (a) Read CAUC §3 threshold selection procedure; verify that MERA's "
                "      binary labels (small_pass/small_fail) can be viewed as a calibration "
                "      proxy for the CAUC framework. If so, MERA's approach can be framed "
                "      as a trained extension of CAUC with co-evolutionary threshold update. "
                "  (b) Draft §2 LLM Routing one-paragraph addition (3-4 sentences): "
                "      'CAUC (arxiv:2609.11446) calibrates cascades post-hoc on a common "
                "      reliability scale, eliminating the need for oracle label collection. "
                "      MERA's trained router extends this by incorporating prompt-structural "
                "      features and re-calibrating across training cycles as the student "
                "      model evolves — a capability CAUC's static thresholds lack.' "
                "  (c) Check whether §4.2 Router Training should add a one-sentence comparison "
                "      to CAUC's post-hoc approach (cost of oracle label collection vs. "
                "      routing accuracy gain at cycle 0). "
                "  (d) Add bib entry cauccascade2026. "
                "Output: §2 paragraph draft + §4.2 comparison sentence + bib entry."
            ),
            "estimated_gpu_hours": 0,
            "expected_paper_impact": (
                "§2 LLM Routing gains a second Sep 2026 cascade-routing citation alongside "
                "SRR (EXP-212). The CAUC–SRR–MERA triangle positions MERA uniquely: SRR "
                "defines the optimal routing signal, CAUC provides post-hoc calibration, "
                "and MERA trains a co-evolving router using oracle labels. "
                "Novelty +0.05 (closes a gap: two Sep 2026 papers in the cascade space, "
                "neither currently cited). "
                "Soundness +0.05 (§4.2 comparison shows MERA trades label-collection cost "
                "for routing accuracy + co-evolution, which is a principled tradeoff)."
            ),
        },
        "rationale": (
            "arxiv:2609.11446 (CAUC, Sep 10 2026) was flagged as a ★★ 'background cite "
            "candidate' in the 2026-09-12 hotspot analysis but not queued (EXP-212 and "
            "EXP-213 were higher priority that day). With ICLR 2027 in 18 days, an uncited "
            "Sep 10 paper directly in the §2 LLM cascade-routing space is a reviewer risk. "
            "CAUC and SRR (EXP-212, already queued) are companion papers that together define "
            "the no-training (CAUC) and optimal-signal (SRR) ends of the cascade-routing "
            "spectrum; MERA sits at the trained + co-evolving end. Positioning all three "
            "closes §2's coverage of the Sep 2026 routing literature. "
            "Offline / 0h GPU; estimated 45 min."
        ),
        "iclr_target_section": "§2 LLM Routing / §4.2 Router Training",
    },
    {
        "id": "EXP-215",
        "priority": 6,
        "title": (
            "SkillAlign §3 SkillBook Interface Positioning: arxiv:2609.07255 "
            "Exposure Interface Variance 47.9%→72.1% vs. MERA Fixed Procedure Format "
            "for ICLR 2027"
        ),
        "paper": "arxiv:2609.07255",
        "paper_title": "SkillAlign: Aligning Skill Interfaces for LLM-based Agents",
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": (
                "src/skills.py — SkillBook skill storage and procedure format: "
                "   f'{procedure}\\n\\n---\\n\\n{problem}' is MERA's fixed exposure "
                "   interface (CLAUDE.md Design Decision #4: format is shared across "
                "   SFT/GRPO/inference; changing it in one place creates a train/inference "
                "   mismatch). "
                "src/pipeline/collect_traces.py — procedure prefix prepended to the raw "
                "   prompt for small-model runs (procedure-augmented inference). "
                "paper auto_research/paper/paper.md §3 Skills — SkillBook description; "
                "   procedure format documentation. "
                "   SkillAlign (2609.07255) not yet cited. "
                "   SkillProx (2608.07449) not yet cited (deferred; lower priority). "
                "results/e2e_4cyc_gpt55/ — ablation arm 'skills' (always-small + procedure) "
                "   vs. 'full' (router + procedure): skills arm pass@1 used to quantify "
                "   the value of the procedure prefix."
            ),
            "metric": (
                "Ren et al. (arxiv:2609.07255, EMNLP 2026) introduce SkillAlign, a "
                "provider-agnostic framework that represents skills as multi-view procedural "
                "cards and renders them through alternative exposure interfaces: full "
                "instructions, hints, compressed summaries, workflows, or no exposure. "
                "The same skill changes ALFWorld success from 47.9% (no exposure) to 72.1% "
                "(optimal interface), a 24.2pp delta driven purely by how the skill is "
                "exposed — not what the skill contains. "
                "Connection to MERA: "
                "(1) MERA's procedure format (f'{procedure}\\n\\n---\\n\\n{problem}') is "
                "    a fixed 'full instructions' exposure interface (Design Decision #4). "
                "    SkillAlign would predict that alternative interfaces (e.g., compressed "
                "    hint, workflow-only) might improve or degrade the procedure-augmented "
                "    small model pass@1, with up to ~24pp variance. MERA does not explore "
                "    this axis — the format is fixed for train/inference consistency. "
                "(2) The MERA ablation 'skills' arm (always-small + procedure) measures "
                "    the value of the full-instructions interface vs. no interface. The "
                "    SkillAlign framing would position this as one point in the interface "
                "    space rather than the only option, opening a future-work direction. "
                "(3) Design Decision #4 (fixed format) is a principled constraint: changing "
                "    the interface at inference creates a train/inference mismatch. SkillAlign "
                "    highlights this as a known limitation worth acknowledging. "
                "Audit tasks: "
                "  (a) Read SkillAlign §4 exposure interface taxonomy and verify which "
                "      category MERA's procedure format falls into (full instructions vs. "
                "      compressed workflow — MERA's skillbook produces ~2-paragraph procedures). "
                "  (b) Draft §3 SkillBook one-paragraph discussion (3-4 sentences): "
                "      'Interface exposure matters: SkillAlign (arxiv:2609.07255) shows "
                "      that the same skill produces 47.9%–72.1% success depending on the "
                "      exposure interface. MERA uses a fixed full-instructions format "
                "      (Design Decision #4) for train/inference consistency; exploring "
                "      interface variants is left as future work.' "
                "  (c) Check §3 SkillBook footnote on multi-skill extension: should it also "
                "      reference SkillAlign's multi-view interface idea? "
                "  (d) Add bib entry skillalign2026. "
                "Output: §3 paragraph draft + footnote check + bib entry."
            ),
            "estimated_gpu_hours": 0,
            "expected_paper_impact": (
                "§3 SkillBook gains a concrete SkillAlign citation (EMNLP 2026, Sep 2026) "
                "that quantifies how much the exposure interface matters — providing "
                "independent evidence for why MERA's fixed format (Design Decision #4) "
                "is a meaningful constraint, not an arbitrary choice. "
                "Novelty +0.05 (adds a Sep 2026 EMNLP citation to §3). "
                "Soundness +0.05 (the 24.2pp interface variance number gives reviewers "
                "a concrete stake: MERA's fixed-format constraint may leave up to ~24pp "
                "uncaptured, which is acknowledged as a limitation)."
            ),
        },
        "rationale": (
            "arxiv:2609.07255 (SkillAlign, EMNLP 2026) was deferred from the 2026-09-11 "
            "queue update as lower priority vs. EXP-210/211 and subsequently from 2026-09-12 "
            "vs. EXP-212/213. Now that EXP-210 through EXP-213 are queued, SkillAlign "
            "is the next-highest priority uncited Sep 2026 paper in MERA's §3 skill space. "
            "Its EMNLP 2026 acceptance (the highest-ranked NLP venue) raises its citation "
            "authority above the earlier deferral priority. The 47.9%→72.1% ALFWorld result "
            "is a memorable concrete number that positions MERA's fixed-format constraint "
            "as a principled tradeoff rather than an oversight. "
            "Offline / 0h GPU; estimated 45 min."
        ),
        "iclr_target_section": "§3 Skills / §3 SkillBook",
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
