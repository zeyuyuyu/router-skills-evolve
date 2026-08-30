"""
Pending queue update — 2026-08-30
Run on A800 when connectivity is restored:
  python /data0/home/zeyuwang/auto_research/pending_queue_update_2026_08_30.py
Appends EXP-200 and EXP-201 to state["queue"] and saves atomically.

A800 offline since 2026-05-14 (day ~108). SSH port 50507 unreachable from remote
execution environment (TCP timeout; proxy is HTTPS-only, cannot tunnel SSH).
Queue ~197 pending (>20 cap → 2 experiments today).
AAAI 2027 camera-ready SUBMITTED (2026-08-29). Next target: ICLR 2027 (~Oct 2026).
Both experiments are OFFLINE / 0h GPU.

Hotspot source: WebSearch fallback (A800 hotspot file unavailable).
Top new papers found:
  arxiv:2603.22455 "SkillRouter: Skill Routing for LLM Agents at Scale"
    (March 2026 — multi-skill expert routing vs. MERA's binary large/small routing)
  arxiv:2605.12652 "Multi-Rollout On-Policy Distillation via Peer Successes and Failures"
    (May 2026 — peer success/failure signals in on-policy distillation, directly
    maps to MERA's run-both oracle + teacher-pass/student-fail trace split)
  arxiv:2601.09692 "Routing with Generated Data: Annotation-Free LLM Skill Estimation"
    (January 2026 — annotation-free routing via generated data vs. MERA's oracle labels)

NEW EXPERIMENTS TODAY:
  EXP-200 (Priority 8): SkillRouter Positioning Audit
           Maps SkillRouter's multi-skill expert routing against MERA's single-global-
           skill binary routing design. Drafts §2 LLM Routing paragraph that defends
           MERA's single-skill design choice (CLAUDE.md Decision #1) for ICLR 2027.
           Cites arxiv:2603.22455. 0h GPU.

  EXP-201 (Priority 7): Peer-Signal On-Policy Distillation Alignment Audit
           Quantifies how MERA's run-both oracle implicitly implements the peer
           success/failure signal from arxiv:2605.12652. Drafts §2b distillation
           paragraph + §3 trace-collection design-choice footnote for ICLR 2027.
           Cites arxiv:2605.12652. 0h GPU.

Apply chain before this patch:
    python3 auto_research/pending_queue_update_2026_08_29.py  # EXP-198, EXP-199
"""

import json, os, shutil, tempfile

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

NEW_EXPERIMENTS = [
    {
        "id": "EXP-200",
        "priority": 8,
        "title": (
            "SkillRouter Positioning Audit: MERA single-global-skill binary routing "
            "vs. multi-skill expert routing for ICLR 2027 §2 (arxiv:2603.22455)"
        ),
        "paper": "arxiv:2603.22455",
        "paper_title": "SkillRouter: Skill Routing for LLM Agents at Scale",
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": (
                "results/e2e_4cyc_gpt55/final_ablation_table.md; "
                "src/skills.py (extract_signature always returns 'coding'); "
                "src/pipeline/train_router_simple.py (TF-IDF binary router); "
                "paper auto_research/paper/paper.md §2 LLM Routing section"
            ),
            "metric": (
                "Contrast SkillRouter's design with MERA's: "
                "(1) SkillRouter: discrete routing to N skill-expert models (N>2), "
                "    skills are capability clusters, routing is one-to-many. "
                "    MERA: binary routing to {small, large}, one global skill bucket, "
                "    routing is learned on oracle labels from run-both oracle. "
                "(2) Identify the ICLR reviewers' natural objection: why not multi-skill? "
                "    Answer: MERA's single-global-skill design is a deliberate choice "
                "    (CLAUDE.md Decision #1): per-cluster skills destabilise routing "
                "    because the procedure prefix is constant within each cluster → "
                "    zero discriminative signal for the router. "
                "(3) Draft §2 paragraph (3 sentences): "
                "    'SkillRouter [arxiv:2603.22455] routes queries across N skill-expert "
                "    models using capability cluster membership. MERA's router is binary "
                "    (small vs. large) and deliberatley uses a single global skill: with "
                "    one skill bucket, the procedure prefix is constant, so routing "
                "    discriminability must come from the raw query surface alone — the "
                "    TF-IDF router learns exactly this, achieving 93.04% accuracy without "
                "    per-skill routing overheads. SkillRouter's multi-skill design is "
                "    complementary; MERA could adopt it if future work finds multiple "
                "    stable skill clusters on larger benchmarks.' "
                "(4) Check: does paper.md §2 already cite SkillRouter? If not, add bib "
                "    entry: skillrouter2026. If yes, strengthen the existing paragraph."
            ),
            "expected_output": (
                "3-sentence §2 LLM Routing paragraph draft; "
                "1-row comparison table: (SkillRouter vs. MERA routing, # skills, "
                "   routing accuracy, notes); "
                "new bib entry: skillrouter2026 if not already present; "
                "confirmation that CLAUDE.md Decision #1 is correctly motivated in §2"
            ),
            "estimated_time": "0h GPU, ~25min analysis + writing",
            "iclr_2027_priority": (
                "high — defends single-global-skill design against a predictable "
                "reviewer objection; directly cites comparable 2026 routing work"
            ),
        },
        "rationale": (
            "SkillRouter (arxiv:2603.22455, March 2026) proposes multi-skill expert "
            "routing for LLM agents at scale. MERA's §2 LLM Routing section for ICLR "
            "2027 must pre-empt the obvious reviewer question: 'why not multi-skill "
            "routing?' CLAUDE.md Decision #1 answers this — single global skill is "
            "deliberate because per-cluster procedure prefixes are constant → zero "
            "discriminative value for the TF-IDF router. The SkillRouter paper is the "
            "natural foil: their multi-skill approach assumes N stable capability "
            "clusters, while MERA's binary routing is stable by design and achieves "
            "93.04% accuracy. Citing SkillRouter in §2 frames MERA's single-skill "
            "choice as a considered engineering tradeoff, not an oversight."
        ),
        "venue_target": "ICLR 2027",
    },
    {
        "id": "EXP-201",
        "priority": 7,
        "title": (
            "Peer-Signal On-Policy Distillation Alignment Audit: MERA run-both oracle "
            "as implicit peer success/failure signal (arxiv:2605.12652)"
        ),
        "paper": "arxiv:2605.12652",
        "paper_title": (
            "Multi-Rollout On-Policy Distillation via Peer Successes and Failures"
        ),
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": (
                "results/e2e_4cyc_gpt55/cycle_{0,1,2,3}/traces.jsonl — "
                "fields: small_pass, large_pass (teacher_pass), task_id; "
                "src/pipeline/collect_traces.py — run-both oracle, _policy_decision; "
                "src/pipeline/traces_to_sft.py — SFT data selection logic; "
                "paper auto_research/paper/paper.md §2b distillation + §3 trace collection"
            ),
            "metric": (
                "arxiv:2605.12652 defines four peer-signal categories per rollout: "
                "  (A) peer_success: teacher passes, student fails → hard positive example "
                "  (B) peer_failure: both fail → confirms hard problem, not used for SFT "
                "  (C) self_success: student passes, teacher passes → routine positive "
                "  (D) self_failure: student fails, teacher fails → beyond current skill "
                "Map MERA's run-both oracle to these categories: "
                "  (A) → teacher_pass=True, small_pass=False [~20.7% C0; track C0-C3] "
                "  (B) → teacher_pass=False, small_pass=False "
                "  (C) → teacher_pass=True, small_pass=True "
                "  (D) → teacher_pass=False, small_pass=True (rare; small beats teacher) "
                "Compute per-cycle breakdown [C0..C3] of all four categories. "
                "Expected: category (A) shrinks → consistent with student improving; "
                "category (C) grows → correct. "
                "Draft deliverables: "
                "(1) 4×4 table: cycle × category → count/fraction. "
                "(2) §2b distillation addition (2 sentences): "
                "    'Multi-rollout distillation methods [arxiv:2605.12652] categorise "
                "    traces by peer success/failure to identify maximally informative "
                "    training signals. MERA's run-both oracle implicitly implements this "
                "    categorisation: the ~20.7% category-(A) traces (teacher passes, "
                "    student fails) are the primary SFT signal (SCALING_FORCE_BOTH=1 "
                "    ensures full coverage), while category-(B)/(D) traces are excluded.' "
                "(3) §3 trace-collection footnote: "
                "    'With SCALING_FORCE_BOTH=1, MERA's oracle produces all four peer- "
                "    signal categories; without it only categories (C) and (A) are "
                "    guaranteed (cost-saving fallback: large model only on small fails).' "
                "New bib entry: peerrollout2026."
            ),
            "expected_output": (
                "4×4 table (cycle × peer-signal category → fraction); "
                "§2b 2-sentence paragraph draft + citation; "
                "§3 footnote draft; "
                "new bib entry: peerrollout2026"
            ),
            "estimated_time": "0h GPU, ~40min analysis + writing",
            "iclr_2027_priority": (
                "medium-high — connects MERA's run-both oracle to a 2026 on-policy "
                "distillation framework; provides a principled name for what MERA does; "
                "SCALING_FORCE_BOTH=1 gotcha gets a proper §3 footnote"
            ),
        },
        "rationale": (
            "arxiv:2605.12652 (May 2026) introduces peer success/failure categorisation "
            "for multi-rollout on-policy distillation. MERA's run-both oracle "
            "(SCALING_FORCE_BOTH=1 path) already produces exactly these four categories "
            "for every task: teacher_pass × small_pass gives a 2×2 grid. The paper "
            "provides a principled framework to name and motivate what MERA's collect_traces "
            "does. For ICLR 2027: (a) citing arxiv:2605.12652 in §2b strengthens the "
            "distillation related-work coverage with a 2026 paper; (b) the 4×4 per-cycle "
            "breakdown quantifies how the peer signal evolves as the student improves "
            "(complements EXP-199's teacher-student gap analysis); (c) the §3 footnote "
            "explaining SCALING_FORCE_BOTH=1 addresses an anticipated implementation "
            "question from reviewers. No GPU or API calls needed — all data is in "
            "results/e2e_4cyc_gpt55/cycle_*/traces.jsonl."
        ),
        "venue_target": "ICLR 2027",
    },
]


def main():
    if not os.path.exists(STATE_PATH):
        print(f"ERROR: {STATE_PATH} not found — queue this file to run when A800 returns")
        return 1

    with open(STATE_PATH) as f:
        state = json.load(f)

    existing_ids = {e.get("id") for e in state.get("queue", [])}
    existing_ids |= {e.get("id") for e in state.get("history", [])}

    added = []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] in existing_ids:
            print(f"SKIP {exp['id']} — already in queue or history")
            continue
        state.setdefault("queue", []).append(exp)
        added.append(exp["id"])
        print(f"QUEUED {exp['id']} (priority={exp['priority']}) {exp['title'][:80]}")

    # Atomic write
    dir_ = os.path.dirname(STATE_PATH)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        shutil.move(tmp, STATE_PATH)
    except Exception:
        os.unlink(tmp)
        raise

    print(f"\nDONE: queued {len(added)} new experiments: {', '.join(added)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
