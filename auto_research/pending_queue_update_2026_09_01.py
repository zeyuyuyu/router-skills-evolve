"""
Pending queue update — 2026-09-01
Run on A800 when connectivity is restored:
  python /data0/home/zeyuwang/auto_research/pending_queue_update_2026_09_01.py
Appends EXP-204 and EXP-205 to state["queue"] and saves atomically.

A800 offline since 2026-05-14 (day ~110). SSH port 50507 unreachable from remote
execution environment (TCP timeout; proxy is HTTPS-only, cannot tunnel SSH).
Queue ~201 pending (>20 cap → 2 experiments today).
Next target: ICLR 2027 (~Oct 2026 deadline, ~4 weeks out).
Both experiments are OFFLINE / 0h GPU (paper positioning analyses for §2/§3).

Hotspot source: WebSearch fallback (A800 hotspot file unavailable — A800 offline).
Top new papers found:
  arxiv:2602.00166  "Joint Continual Learning of Local Language Models and Cloud
    Offloading Decisions with Budget Constraints" (Feb 2026 — DA-GRPO: the closest
    published system to MERA's local+cloud+routing+continual design; key for §2
    differentiation in ICLR 2027)
  arxiv:2605.28791  "Skill-Conditioned Gated Self-Distillation for LLM Reasoning"
    (May 2026 — skill-gated selective distillation; intersects MERA's single-skill
    SkillBook design, useful for §3 skills contribution positioning)
  arxiv:2607.20481  "Routing Without Training: Controllable-Ratio LLM Offloading via
    Reliability Gating" (July 2026 — zero-training router via reliability gating;
    contrasts with MERA's learned router, useful for §4 router ablation framing)

NEW EXPERIMENTS TODAY:
  EXP-204 (Priority 8): DA-GRPO Local-Cloud Joint Learning Positioning Audit for
           ICLR 2027 §2 Differentiation. Maps MERA against the closest published
           near-neighbour (DA-GRPO, arxiv:2602.00166): joint advantage-based routing
           vs. MERA's separate supervised router; no procedure prefix in DA-GRPO vs.
           MERA's SkillBook distillation; non-iterative vs. MERA's N-cycle co-evolution.
           Drafts §2 differentiation paragraph. Cites arxiv:2602.00166. 0h GPU.

  EXP-205 (Priority 7): Skill-Gated Distillation Audit — MERA vs. arxiv:2605.28791
           for ICLR 2027 §3 Skills Contribution Framing. Maps MERA's single-global
           "coding" SkillBook (CLAUDE.md Decision #1) against skill-conditioned gated
           self-distillation (arxiv:2605.28791). Checks whether MERA's diagnostic
           can_downgrade_to_small verdict implements an implicit skill gate, and whether
           gated per-skill distillation would violate CLAUDE.md Decision #2 (router
           owns routing). Drafts §3 footnote or future-work citation. 0h GPU.

Apply chain before this patch:
    python3 auto_research/pending_queue_update_2026_08_31.py  # EXP-202, EXP-203
"""

import json, os, shutil, tempfile

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

NEW_EXPERIMENTS = [
    {
        "id": "EXP-204",
        "priority": 8,
        "title": (
            "DA-GRPO Local-Cloud Joint Learning Positioning Audit: MERA vs. "
            "arxiv:2602.00166 for ICLR 2027 §2 Differentiation"
        ),
        "paper": "arxiv:2602.00166",
        "paper_title": (
            "Joint Continual Learning of Local Language Models and Cloud Offloading "
            "Decisions with Budget Constraints"
        ),
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": (
                "src/pipeline/train_router_simple.py — MERA's router training on "
                "   raw-prompt oracle labels (CLAUDE.md Decision #3); "
                "src/pipeline/collect_traces.py — _policy_decision routing logic, "
                "   router-owns-routing design (CLAUDE.md Decision #2); "
                "src/skills.py — SkillBook.extract_signature single-bucket design "
                "   (CLAUDE.md Decision #1); "
                "results/e2e_4cyc_gpt55/ — routing accuracy cycle-0..3 progression; "
                "paper auto_research/paper/paper.md §2 Related Work + §4 Router"
            ),
            "metric": (
                "DA-GRPO (arxiv:2602.00166, Feb 2026) is the closest published system "
                "to MERA's core design — a local small model + cloud large model pair "
                "with routing decisions that are learned jointly during continual "
                "post-training. Key architectural differences to characterise: "
                "(1) Routing signal source: DA-GRPO encodes cloud-usage as a dual "
                "    advantage signal inside the GRPO objective (routing = part of "
                "    reward shaping), while MERA trains a separate supervised router "
                "    (train_router_simple.py) on oracle labels from collect_traces. "
                "    MERA's advantage: the router and the LLM update independently — "
                "    no reward entanglement, cleaner ablation (CLAUDE.md Decision #2). "
                "(2) Skill distillation: DA-GRPO has no procedure prefix — the local "
                "    model sees only raw prompts and learns routing/solving jointly. "
                "    MERA's SkillBook provides a procedure prefix that augments the "
                "    small model's prompt (CLAUDE.md Decision #4 format). "
                "    Ablation arm 'skills' in MERA's four-arm design (CLAUDE.md "
                "    Decision #5) directly measures the procedure-prefix contribution. "
                "(3) Iterative co-evolution: DA-GRPO runs a single post-training pass; "
                "    MERA iterates N cycles with checkpoint inheritance "
                "    (CLAUDE.md Decision #6). "
                "(4) Budget constraint: DA-GRPO optimizes an explicit cloud-call budget "
                "    (fraction of queries to offload); MERA's router is unconstrained "
                "    and learns purely from pass-rate signal. "
                "Analysis steps: "
                "(1) Read DA-GRPO paper fully (arxiv:2602.00166) and extract: "
                "    routing mechanism, skill/procedure component (or absence), "
                "    iterative update schedule, budget formulation. "
                "(2) Map each axis to MERA's design choices in CLAUDE.md. "
                "(3) Identify the one claim DA-GRPO makes that most threatens MERA "
                "    §2 novelty (likely: 'joint routing + LLM post-training from RL'). "
                "    Write the differentiator sentence that survives a reviewer who "
                "    cites DA-GRPO as prior art. "
                "(4) Draft §2 paragraph (~120 words) positioning MERA as the "
                "    'three-component co-evolving' system that DA-GRPO's "
                "    two-component (local+cloud) design lacks. "
                "Expected output: a §2 paragraph draft + a 2-row comparison table "
                "(MERA vs. DA-GRPO across the 4 axes) for inclusion in ICLR 2027."
            ),
            "iclr_2027_target": "§2 Related Work — nearest-neighbour differentiation",
        },
        "rationale": (
            "DA-GRPO (arxiv:2602.00166, Feb 2026) is a Dual-Advantage GRPO extension "
            "that jointly trains a local LLM and its cloud-offloading decisions under a "
            "budget constraint — the closest published design to MERA's local+cloud+routing "
            "architecture. An ICLR 2027 reviewer familiar with DA-GRPO could challenge "
            "MERA's §2 novelty claim. This audit produces a formal §2 differentiation "
            "paragraph mapping the four key axes (routing signal, skill distillation, "
            "iterative cycles, budget constraint) where MERA and DA-GRPO diverge, "
            "converting the near-neighbour risk into a strengthened novelty narrative. "
            "Priority 8: 4 weeks from ICLR 2027 deadline; §2 drafting is critical-path."
        ),
    },
    {
        "id": "EXP-205",
        "priority": 7,
        "title": (
            "Skill-Gated Distillation Audit: MERA single-global SkillBook vs. "
            "arxiv:2605.28791 for ICLR 2027 §3 Skills Contribution Framing"
        ),
        "paper": "arxiv:2605.28791",
        "paper_title": (
            "Skill-Conditioned Gated Self-Distillation for LLM Reasoning"
        ),
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": (
                "src/skills.py — SkillBook.extract_signature (always returns 'coding'), "
                "   can_downgrade_to_small (diagnostic verdict, CLAUDE.md Decision #2); "
                "src/pipeline/collect_traces.py — _policy_decision routing, "
                "   policy_skill_verdict field (diagnostic only, never overrides route); "
                "src/pipeline/run_e2e_ablation_simple.py — 'skills' arm "
                "   (always-small + procedure prefix, CLAUDE.md Decision #5); "
                "paper auto_research/paper/paper.md §3 SkillBook design"
            ),
            "metric": (
                "arxiv:2605.28791 (May 2026) proposes Skill-Conditioned Gated "
                "Self-Distillation: for each training sample, a skill classifier "
                "identifies which skills the student lacks, and a gating mechanism "
                "selectively activates teacher distillation only for the skill-gap "
                "tokens. This is a stronger distillation strategy than MERA's global "
                "procedure-prefix approach. Key questions to answer: "
                "(1) Does MERA's single-skill 'coding' SkillBook (CLAUDE.md Decision #1) "
                "    already implement an implicit skill gate? "
                "    The can_downgrade_to_small verdict in SkillBook compares small "
                "    vs. large pass rates — this is effectively a task-level skill-gate "
                "    (route to large when the skill gap is too large). It was demoted "
                "    to diagnostic because with ONE skill it yields a constant verdict "
                "    per cycle. With multiple skills this would be a real gate. "
                "(2) Would implementing per-skill gated distillation require breaking "
                "    CLAUDE.md Decision #2 (router owns routing) or Decision #1 "
                "    (single global skill)? Map the dependency. "
                "    Answer expected: yes — per-skill gating requires per-skill routing "
                "    verdicts, reintroducing the multi-cluster design that Decision #1 "
                "    removed. The current design correctly avoids this for simplicity. "
                "(3) Is the single-skill design a limitation or a strength for §3? "
                "    Framing: MERA shows that a single procedure is sufficient to "
                "    improve small model performance over the no-procedure baseline "
                "    (the 'skills' ablation arm). Skill-gating would further improve "
                "    the skills arm but at the cost of pipeline complexity. Frame as "
                "    an open extension rather than a gap. "
                "Analysis steps: "
                "(1) Read arxiv:2605.28791, extract: skill identification method, "
                "    gating mechanism, distillation target (tokens vs. logits), "
                "    benchmark results vs. standard self-distillation. "
                "(2) Map MERA's can_downgrade_to_small logic to the gating concept. "
                "(3) Enumerate what would need to change in src/skills.py to support "
                "    per-skill gating (multi-cluster signatures, per-cluster stats). "
                "(4) Draft §3 footnote (~80 words): 'Our single-skill design is "
                "    sufficient for the co-evolution signal; extending to skill-gated "
                "    distillation (arxiv:2605.28791) is a natural future direction.' "
                "Expected output: §3 footnote draft + CLAUDE.md Decision #1 rationale "
                "enrichment linking to arxiv:2605.28791 as the motivation for why "
                "single-skill simplicity was the right call."
            ),
            "iclr_2027_target": "§3 SkillBook — single-skill design justification footnote",
        },
        "rationale": (
            "Skill-Conditioned Gated Self-Distillation (arxiv:2605.28791, May 2026) is "
            "a stronger distillation baseline than MERA's global procedure-prefix approach: "
            "it gates teacher distillation on per-sample skill gaps rather than always "
            "prepending the same procedure text. An ICLR 2027 reviewer could ask why MERA "
            "doesn't use per-skill gating. This audit maps the dependency (per-skill gating "
            "would require reintroducing multi-cluster signatures, violating CLAUDE.md "
            "Decision #1 + #2) and drafts a §3 footnote converting the potential objection "
            "into a forward-looking citation. Cites arxiv:2605.28791. Priority 7: §3 "
            "contribution framing for ICLR 2027, secondary to §2 DA-GRPO differentiation."
        ),
    },
]


def main():
    with open(STATE_PATH, "r") as f:
        state = json.load(f)

    existing_ids = {e.get("id") for e in state.get("queue", [])}
    existing_ids |= {e.get("id") for e in state.get("history", [])}

    added = []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] in existing_ids:
            print(f"  SKIP {exp['id']} — already in queue/history")
            continue
        state.setdefault("queue", []).append(exp)
        added.append(exp["id"])
        print(f"  ADD  {exp['id']} — {exp['title'][:60]}")

    # Atomic write
    dirpath = os.path.dirname(STATE_PATH)
    with tempfile.NamedTemporaryFile("w", dir=dirpath, delete=False, suffix=".tmp") as tf:
        json.dump(state, tf, indent=2)
        tmp_path = tf.name
    shutil.move(tmp_path, STATE_PATH)

    print(f"\nDone. Added {len(added)} experiment(s): {added}")


if __name__ == "__main__":
    main()
