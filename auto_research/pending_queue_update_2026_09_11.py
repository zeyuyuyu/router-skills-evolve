"""
Pending queue update — 2026-09-11
Run on A800 when connectivity is restored:
  python /data0/home/zeyuwang/auto_research/pending_queue_update_2026_09_11.py
Appends EXP-210 and EXP-211 to state["queue"] and saves atomically.

A800 offline since 2026-05-14 (day ~120). SSH port 50507 unreachable from remote
execution environment (TCP timeout; proxy is HTTPS-only, cannot tunnel SSH).
Queue ~209 pending (>20 cap → 2 experiments today).
Next target: ICLR 2027 (~Oct 1 deadline, ~20 days out). URGENT.
Both experiments are OFFLINE / 0h GPU (paper §2/§3 positioning for new Sep 2026 papers).

Hotspot source: WebSearch fallback (A800 hotspot file unavailable — A800 offline).
Top new papers found (this run, 2026-09-11):
  arxiv:2509.21154  "GRPO is Secretly a Process Reward Model" (Sullivan & Koller,
    Sep 2025 / revised May 2026) — proves GRPO ≡ implicit PRM when rollouts share
    prefixes; proposes λ-GRPO fix for step imbalance. Directly intersects MERA's
    52.4% zero-variance groups finding (arxiv:2507.05386). Not yet in paper.
    → EXP-210: GRPO-PRM Equivalence §3 Framing for ICLR 2027.
  arxiv:2609.05824  "Beyond Top-k Skill Retrieval: Diversity-Aware Skill Routing
    for LLM Agents" (Sep 2026) — DSR uses Determinantal Point Process to select
    complementary skills, improving recall on multi-skill queries. New Sep 2026 paper
    directly at the intersection of MERA's SkillBook design and LLM routing.
    → EXP-211: DSR §2 Positioning Audit for ICLR 2027.
  arxiv:2609.07255  "SkillAlign: Aligning Skill Interfaces for LLM-based Agents"
    (Sep 2026) — skill interface alignment for heterogeneous skill registries;
    background citation candidate for §3 SkillBook (not queued today — lower priority).

Apply chain before this patch:
    python3 auto_research/pending_queue_update_2026_09_04.py  # EXP-208, EXP-209
"""

import json, os, shutil, tempfile

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

NEW_EXPERIMENTS = [
    {
        "id": "EXP-210",
        "priority": 8,
        "title": (
            "GRPO-PRM Equivalence §3 Framing Audit: arxiv:2509.21154 — "
            "Connecting MERA's Zero-Variance Groups to Implicit Process Credit for ICLR 2027"
        ),
        "paper": "arxiv:2509.21154",
        "paper_title": "GRPO is Secretly a Process Reward Model",
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": (
                "src/pipeline/grpo_train_simple.py — GRPO rollout logic, group sampling, "
                "   advantage computation; MERA uses 8 rollouts per prompt (G=8); "
                "src/pipeline/collect_traces.py — oracle labels inform which tasks get "
                "   zero-variance groups (all G pass or all G fail); "
                "paper auto_research/paper/paper.md §3 GRPO Training (zero-variance groups "
                "   52.4% finding, arxiv:2507.05386 citation already present); "
                "paper auto_research/paper/paper.md §7 Remedies (Dr.GRPO, DAPO, λ-GRPO "
                "   not yet mentioned); "
                "results/e2e_4cyc_gpt55/ — GRPO reward traces showing group composition."
            ),
            "metric": (
                "Sullivan & Koller (arxiv:2509.21154, revised May 2026) prove that GRPO "
                "with an outcome reward model is equivalent to a PRM-aware RL objective with "
                "a Monte-Carlo-based implicit process reward, provided that rollout subsets "
                "share identical prefixes. This has two implications for MERA: "
                "(1) MERA's zero-variance groups (52.4% of DAPO groups receive zero gradient, "
                "    arxiv:2507.05386 / CLAUDE.md §8) are equivalently described as groups "
                "    where the implicit PRM assigns uniform step credit — connecting the "
                "    zero-variance failure mode to the PRM framing provides a more principled "
                "    explanation for §3. "
                "(2) λ-GRPO, the paper's proposed fix for step-imbalance, is a sibling of "
                "    Dr.GRPO (EXP-120, blocked on A800) and DAPO (the variant already run "
                "    in MERA). §7 Remedies currently lists Dr.GRPO and DAPO but not λ-GRPO. "
                "    Adding a one-sentence comparison clarifies the design space. "
                "Audit tasks: "
                "  (a) Does the arxiv:2509.21154 PRM-equivalence proof apply to MERA's "
                "      rollout structure (G=8 independent completions, no shared prefix)? "
                "      If rollouts are fully independent (no branching), the shared-prefix "
                "      condition may NOT hold — in which case the framing is approximate. "
                "      Determine correct framing. "
                "  (b) Draft one-paragraph §3 addendum: 'GRPO-as-PRM perspective on "
                "      zero-variance groups' — adds mechanistic depth beyond the raw count. "
                "  (c) Add λ-GRPO to §7 Remedies as a one-sentence sibling of Dr.GRPO. "
                "  (d) Add bib entry grpoprmequiv2026. "
                "Output: §3 addendum draft + §7 one-liner + bib entry."
            ),
            "estimated_gpu_hours": 0,
            "expected_paper_impact": (
                "§3 zero-variance groups analysis gains a principled PRM-framing, "
                "making the failure-mode explanation more theoretically grounded. "
                "Soundness +0.1 (closes a gap reviewers may flag). "
                "§7 Remedies λ-GRPO addition rounds out the design-space coverage. "
                "Addresses potential ICLR reviewer question: 'How does zero-variance "
                "relate to known GRPO failure modes in the literature?'"
            ),
        },
        "rationale": (
            "MERA §3 already cites arxiv:2507.05386 for the 52.4% zero-variance groups "
            "finding and connects it to the implicit regularization that prevents forgetting. "
            "arxiv:2509.21154 (revised May 2026, not yet in paper) provides a complementary "
            "lens: GRPO implicitly computes a Monte-Carlo PRM, and zero-variance groups are "
            "exactly the groups where that implicit PRM assigns identical credit to all steps. "
            "Adding this framing deepens §3's mechanistic explanation. The paper also "
            "proposes λ-GRPO, a simple fix sibling to MERA's Dr.GRPO (EXP-120) — worth "
            "one sentence in §7 Remedies for completeness. Both additions are offline/0h GPU. "
            "With 20 days to ICLR deadline (Oct 1), each §3 strengthening point matters."
        ),
        "iclr_target_section": "§3 GRPO Training / §7 Remedies",
    },
    {
        "id": "EXP-211",
        "priority": 7,
        "title": (
            "DSR §2 Self-Evolving Agents Positioning Audit: MERA vs. arxiv:2609.05824 "
            "Diversity-Aware Skill Routing for ICLR 2027"
        ),
        "paper": "arxiv:2609.05824",
        "paper_title": (
            "Beyond Top-k Skill Retrieval: Diversity-Aware Skill Routing for LLM Agents"
        ),
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": (
                "src/skills.py — SkillBook.extract_signature always returns 'coding' "
                "   (CLAUDE.md Decision #1: single global skill, no per-cluster retrieval); "
                "src/pipeline/collect_traces.py — procedure prefix construction "
                "   (Decision #2: router owns routing, skill diagnostic is read-only); "
                "paper auto_research/paper/paper.md §2 Self-Evolving Agentic Systems "
                "   (already cites SkillForge arxiv:2608.24747, SKILL-KD arxiv:2607.28048, "
                "   Skill-Gated Distillation arxiv:2605.28791, Skill-R1 arxiv:2605.09359); "
                "paper auto_research/paper/paper.md §3 Skills (SkillBook design, procedure "
                "   format, single-bucket rationale)."
            ),
            "metric": (
                "DSR (arxiv:2609.05824, Sep 2026) proposes Diverse Skill Routing for LLM "
                "agents: a Determinantal Point Process (DPP) reranking framework over a large "
                "skill registry, balancing relevance and non-redundancy via a query-residual "
                "diversity kernel. On the SkillRouter benchmark, DSR improves recall and "
                "full coverage over pointwise reranking, with larger gains on multi-skill queries. "
                "This paper is directly relevant to MERA's SkillBook design in two ways: "
                "(1) MERA's single 'coding' skill (CLAUDE.md Decision #1) sidesteps the "
                "    multi-skill retrieval problem entirely — DSR addresses redundancy among "
                "    many skills, while MERA has exactly one. This is a deliberate design "
                "    trade-off: MERA sacrifices skill specificity for routing simplicity "
                "    (single global procedure prefix, no retrieval overhead). "
                "(2) DSR's query-residual kernel penalizes redundant skills — conceptually, "
                "    MERA's SkillBook already implements a weaker form of this by collapsing "
                "    all coding tasks into one bucket, effectively doing DPP with k=1. "
                "Audit tasks: "
                "  (a) Write §2 Self-Evolving Agentic Systems positioning paragraph "
                "      (3-4 sentences): DSR optimizes the skill-set selection problem; "
                "      MERA orthogonally optimizes which model executes a single skill — "
                "      framing them as complementary, not competing. "
                "  (b) Check whether §3 SkillBook should add a footnote on multi-skill "
                "      extension (currently EXP-188 is queued for this; reference it). "
                "  (c) Add bib entry dsr2026. "
                "Output: §2 paragraph draft + §3 footnote candidate + bib entry."
            ),
            "estimated_gpu_hours": 0,
            "expected_paper_impact": (
                "Closes a gap in §2 Related Work: Sep 2026 paper directly at the intersection "
                "of MERA's SkillBook and LLM routing not yet cited. ICLR reviewers scanning "
                "for September 2026 related work would flag this omission. "
                "Novelty: positioning DSR as a complementary axis (skill-set diversity) vs. "
                "MERA (model-selection routing) strengthens MERA's contribution story. "
                "Novelty +0.1 if framing is clean."
            ),
        },
        "rationale": (
            "arxiv:2609.05824 (Sep 2026) is a brand-new paper on diversity-aware skill "
            "routing that directly intersects MERA's SkillBook design. It was published "
            "after MERA's current §2 was written (last updated Sep 4). With the ICLR 2027 "
            "deadline ~20 days away, any Sep 2026 paper at the intersection of skill "
            "routing + LLM agents that is NOT cited will draw reviewer attention. "
            "The positioning is straightforward: DSR optimizes *which skills* to combine "
            "(diversity within a set), while MERA optimizes *which model* executes a skill "
            "(routing decision). They are complementary, not competing. "
            "EXP-211 produces a §2 paragraph, §3 footnote, and bib entry. Offline, 0h GPU."
        ),
        "iclr_target_section": "§2 Self-Evolving Agentic Systems / §3 Skills",
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
