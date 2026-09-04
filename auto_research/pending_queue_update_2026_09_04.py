"""
Pending queue update — 2026-09-04
Run on A800 when connectivity is restored:
  python /data0/home/zeyuwang/auto_research/pending_queue_update_2026_09_04.py
Appends EXP-208 and EXP-209 to state["queue"] and saves atomically.

A800 offline since 2026-05-14 (day ~114). SSH port 50507 unreachable from remote
execution environment (TCP timeout; proxy is HTTPS-only, cannot tunnel SSH).
Queue ~205 pending (>20 cap → 2 experiments today).
Next target: ICLR 2027 (~Oct 2026 deadline, ~4 weeks out).
Both experiments are OFFLINE / 0h GPU (paper positioning analyses for §2/§4).

Hotspot source: WebSearch fallback (A800 hotspot file unavailable — A800 offline).
Top new papers found:
  arxiv:2607.07847  "When Does Continual Learning Require Learning" (July 2026 —
    mechanism-agnostic benchmark comparing GRPO, SFT, SDFT, SDPO, and prompt methods
    for continual learning across space and time axes; MERA's 3-way co-evolution is
    not covered by their protocol; needs §2 positioning paragraph for ICLR 2027)
  arxiv:2603.22455  "SkillRouter: Skill routing for LLM agents at scale" (March 2026 —
    routes between agent capabilities/skills at inference time; differs fundamentally
    from MERA's inter-model routing (small vs large); needs §4 footnote to prevent
    reviewer conflation of skill-routing and model-scale routing)

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
            "'When Does CL Require Learning?' Benchmarking Protocol Audit: "
            "MERA §2 Positioning vs. arxiv:2607.07847"
        ),
        "paper": "arxiv:2607.07847",
        "paper_title": "When Does Continual Learning Require Learning",
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": (
                "src/pipeline/collect_traces.py — Phase 1 trace collection (SFT + GRPO "
                "   labels produced jointly each cycle); "
                "src/pipeline/train_small_model.py — SFT phase; "
                "src/pipeline/grpo_train_simple.py — GRPO phase; "
                "src/skills.py — SkillBook distillation (third co-evolution axis "
                "   absent from arxiv:2607.07847 benchmark); "
                "src/pipeline/train_router_simple.py — router co-evolution (fourth axis); "
                "results/e2e_4cyc_gpt55/ — 4-cycle trajectory: router + skills + SFT + GRPO "
                "   all improve jointly; "
                "paper auto_research/paper/paper.md §2 Related Work, §1 Introduction"
            ),
            "metric": (
                "arxiv:2607.07847 (July 2026) introduces a mechanism-agnostic protocol "
                "to benchmark continual learning methods for LLMs along two axes: "
                "space (new domains) and time (drifting data under fixed task). "
                "The protocol covers prompt-based methods (GEPA, ACE), supervised "
                "learning (SFT, SDFT), reinforcement learning (GRPO, SDPO), and context "
                "compression (Cartridges, In-place TTT). "
                ""
                "A naive reader of MERA could classify it as a combination of SFT + GRPO, "
                "both studied in the arxiv:2607.07847 Table 2 protocol. An ICLR 2027 "
                "reviewer familiar with the paper might claim: 'MERA simply combines SFT "
                "and GRPO — the combination is already in the benchmark, with no new "
                "conceptual contribution beyond interleaving two known methods.' "
                ""
                "This audit proves the claim is incorrect by mapping the four dimensions "
                "where MERA differs structurally from the paper's GRPO+SFT baseline: "
                "(1) Co-evolution third axis: SkillBook distillation. "
                "    arxiv:2607.07847 studies GRPO and SFT as independent strategies. "
                "    MERA adds a skill distillation loop (SkillBook) that produces a "
                "    procedure prefix fed to the small model each cycle. No such "
                "    teacher-to-procedure distillation component appears in the benchmark. "
                "(2) Co-evolution fourth axis: Learned router. "
                "    arxiv:2607.07847 has no routing component — every method applies "
                "    to all tasks identically. MERA's router learns which tasks to send "
                "    to the small vs large model, so the 'continual' challenge is not "
                "    just forgetting but routing BIAS DRIFT as the small model improves. "
                "(3) N-cycle co-evolution structure. "
                "    arxiv:2607.07847 studies single-episode or short-horizon continual "
                "    learning. MERA's N alternating SFT→GRPO cycles create a multi-epoch "
                "    co-evolution trajectory where the router's training distribution "
                "    shifts each cycle (Decision #6 checkpoint boundary). "
                "(4) Oracle-grade teacher traces. "
                "    MERA's SFT phase uses teacher traces from a strictly stronger model "
                "    (GPT-5.5 run-both oracle). arxiv:2607.07847's SFT uses self-supervised "
                "    or in-domain data, not a stronger external oracle. "
                ""
                "Analysis steps: "
                "(1) Read arxiv:2607.07847 §3 benchmark protocol and Table 2. Extract the "
                "    exact methods compared and their architectures. "
                "(2) Draft a 4-row positioning table: "
                "    Axis | arxiv:2607.07847 best | MERA "
                "    (SkillBook distillation, learned routing, multi-cycle co-evolution, "
                "    oracle teacher traces). "
                "(3) Draft §2 paragraph (~120 words) positioning MERA relative to "
                "    the benchmark: 'The most comprehensive benchmark for LLM continual "
                "    learning, arxiv:2607.07847, evaluates ... across space and time axes. "
                "    MERA differs along four dimensions not covered by any single method "
                "    in their protocol: ...' "
                "(4) Confirm that none of the four axes appear in the paper's Table 2 "
                "    method list (cross-check with paper §4 method descriptions). "
                "Expected output: §2 paragraph (~120 words) + 4-row positioning table. "
                "Commit as auto_research/paper/sec2_cl_benchmark_positioning.md."
            ),
            "iclr_2027_target": "§2 Related Work — positioning vs. comprehensive CL benchmark",
        },
        "rationale": (
            "arxiv:2607.07847 (July 2026) introduces the most comprehensive LLM continual "
            "learning benchmark to date, comparing GRPO, SFT, SDFT, SDPO, and prompt methods "
            "along space and time axes. A naive ICLR 2027 reviewer may classify MERA as 'just "
            "SFT+GRPO interleaved — already in Table 2.' This audit formally shows that MERA "
            "differs along four structural axes absent from the benchmark: (1) SkillBook "
            "procedure distillation as a third co-evolution component, (2) a learned inter-model "
            "router as a fourth component, (3) N-cycle checkpoint-boundary co-evolution structure, "
            "and (4) oracle-grade teacher traces from a strictly stronger model. Output: §2 "
            "positioning paragraph (~120 words) + 4-row table. Priority 8: §2 differentiation "
            "is critical for ICLR 2027 acceptance with ~4 weeks to deadline."
        ),
    },
    {
        "id": "EXP-209",
        "priority": 7,
        "title": (
            "SkillRouter vs. MERA Inter-Model Router: §4 Router Contribution Differentiation "
            "Audit (arxiv:2603.22455)"
        ),
        "paper": "arxiv:2603.22455",
        "paper_title": "SkillRouter: Skill routing for LLM agents at scale",
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": (
                "src/pipeline/train_router_simple.py — MERA's router: logistic regression on "
                "   raw prompt embeddings → binary small/large decision (CLAUDE.md Decision #3); "
                "src/pipeline/collect_traces.py — _policy_decision: router is the SOLE "
                "   routing authority (CLAUDE.md Decision #2); "
                "src/skills.py — extract_signature always returns 'coding' single bucket "
                "   (CLAUDE.md Decision #1) — no per-skill routing; "
                "results/e2e_4cyc_gpt55/ — router accuracy cycle progression; "
                "paper auto_research/paper/paper.md §4 Router section"
            ),
            "metric": (
                "SkillRouter (arxiv:2603.22455, March 2026) is a system for routing LLM "
                "agent requests to appropriate skills (capabilities, tools, sub-agents) at "
                "scale. It learns which skill to invoke for a given task, optimizing "
                "capability coverage and latency. "
                ""
                "An ICLR 2027 reviewer could conflate SkillRouter with MERA's router, "
                "especially given MERA's SkillBook (which produces 'skills') and its "
                "learned router. The conflation risk: 'MERA's router simply selects among "
                "skills, like SkillRouter — not a novel contribution.' "
                ""
                "This audit documents the four key distinctions: "
                "(1) Routing objective. "
                "    SkillRouter: select WHICH skill (capability/tool/sub-agent) to invoke "
                "    for a task — same model is always used, routing chooses the function. "
                "    MERA: select WHICH MODEL (small Qwen3-4B vs. large GPT-5.5) to invoke "
                "    for a task — same task is always the target, routing chooses model scale. "
                "    Orthogonal dimensions: skill selection vs. model-scale selection. "
                "(2) Training signal. "
                "    SkillRouter: trained on skill-invocation success/failure across tasks. "
                "    MERA: trained on oracle labels (did small model pass the task?) from "
                "    the run-both oracle (CLAUDE.md Decision #3 + Gotcha SCALING_FORCE_BOTH). "
                "(3) Routing scope. "
                "    SkillRouter: operates within a single model's inference, routing to "
                "    sub-capabilities — intra-model routing. "
                "    MERA: operates across two fundamentally different models with different "
                "    cost/quality tradeoffs — inter-model routing. "
                "(4) Continual re-training. "
                "    SkillRouter: static or lightly updated routing policy. "
                "    MERA: router is retrained from scratch each cycle (train_router_simple.py) "
                "    on updated oracle labels, so routing tracks the improving small model. "
                ""
                "Analysis steps: "
                "(1) Read arxiv:2603.22455 §2 system design and routing mechanism. "
                "    Extract: what does SkillRouter route between? What is its training signal? "
                "(2) Map each of the 4 axes above to MERA's implementation. "
                "(3) Draft §4 footnote or sidebar (~80 words): 'SkillRouter (arxiv:2603.22455) "
                "    routes between agent capabilities within a single model. MERA's router "
                "    addresses the orthogonal problem of inter-model scale selection — which "
                "    model to invoke — and uniquely retrains each cycle to track the small "
                "    model's improving competence.' "
                "(4) Write 2-row comparison table (MERA vs. SkillRouter across 4 axes). "
                "Expected output: §4 footnote draft (~80 words) + 2-row table + bib entry "
                "`skillrouter2026`. Commit as auto_research/paper/sec4_router_differentiation.md."
            ),
            "iclr_2027_target": "§4 Router — differentiation from skill-selection routing systems",
        },
        "rationale": (
            "SkillRouter (arxiv:2603.22455, March 2026) routes LLM agent requests to "
            "appropriate skills/capabilities within a single model — intra-model skill "
            "selection. MERA's router solves the orthogonal problem of inter-model scale "
            "selection (small vs. large), retrained each cycle to track the small model's "
            "improving competence. An ICLR 2027 reviewer familiar with SkillRouter could "
            "conflate the two, especially given MERA's SkillBook and learned router component. "
            "This audit drafts the §4 footnote (~80 words) and 2-row comparison table that "
            "preempt the conflation: SkillRouter routes to capabilities, MERA routes to model "
            "scale. Priority 7: §4 Router is a key contribution claim; the distinction is "
            "critical but secondary to the §2/§7 analyses already queued at priority 8."
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
