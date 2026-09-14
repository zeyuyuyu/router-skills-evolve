"""
Pending queue update — 2026-09-14
Run on A800 when connectivity is restored:
  python /data0/home/zeyuwang/auto_research/pending_queue_update_2026_09_14.py
Appends EXP-214 and EXP-215 to state["queue"] and saves atomically.

A800 offline since 2026-05-14 (day ~123). SSH port 50507 unreachable from remote
execution environment (TCP timeout; proxy is HTTPS-only, cannot tunnel SSH).
Queue ~209 pending (>20 cap → 2 experiments today).
Next target: ICLR 2027 (~Oct 1 deadline, ~17 days out). URGENT.
Both experiments are OFFLINE / 0h GPU (paper §2/§5 positioning for new Sep 2026 papers).

Hotspot source: WebSearch fallback (A800 hotspot file unavailable — A800 offline).
Top new papers found (this run, 2026-09-14):
  arxiv:2609.12578  "From Collaboration to Capability: Internalizing Routed LLM
    Experts into Compact Reasoners" (Sep 11, 2026, Nie et al.) — RIVET method: compact
    1.7B/4B model internalizes routing to strong experts via RL + SFT two-stage framework;
    Stage I=expert-augmented RL, Stage II=verified trajectory SFT (+6.49pp after expert
    removal on competition math). MERA's Phase 3b (GRPO) + Phase 3a (SFT) implement the
    same expert internalization loop. Closest published comparator to MERA's §5 evolution
    mechanism. Not yet cited. → EXP-214: RIVET §2/§5 Positioning Audit.
  arxiv:2609.09153  "Procedural Graphs: Self-Evolving Execution Structures for LLM
    Agents" (Sep 8, 2026, Lu et al.) — Procedural Graph organizes procedural knowledge as
    (procedure, relation, procedure) triplets; agents localize the active node and follow
    situational guidance without dictating actions. Direct structural comparator to MERA's
    SkillBook (flat per-signature skill text). Not yet cited in §3 SkillBook.
    → EXP-215: Procedural Graphs §3 SkillBook Positioning Audit.
  arxiv:2609.07786  "Signed Rescue Routing" (Sep 7, 2026) — already queued as EXP-212.
  arxiv:2609.04172 / 2609.05198  OPD coverage papers (Sep 4, 2026) — already queued as EXP-213.

Apply chain before this patch:
    python3 auto_research/pending_queue_update_2026_09_12.py  # EXP-212, EXP-213
"""

import json, os, shutil

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

NEW_EXPERIMENTS = [
    {
        "id": "EXP-214",
        "priority": 8,
        "title": (
            "RIVET §2/§5 Expert Internalization Positioning: MERA vs. arxiv:2609.12578 "
            "'From Collaboration to Capability' for ICLR 2027"
        ),
        "paper": "arxiv:2609.12578",
        "paper_title": (
            "From Collaboration to Capability: Internalizing Routed LLM Experts "
            "into Compact Reasoners"
        ),
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": (
                "src/pipeline/train_small_model.py — Phase 3a SFT on teacher traces; "
                "   trains Qwen3-4B on (procedure + problem, solution) pairs from oracle "
                "   GPT-5.5 traces where the small model previously failed. "
                "src/pipeline/grpo_train_simple.py — Phase 3b GRPO: RL with verifiable "
                "   binary code-execution rewards on HumanEval (164 tasks, G=8 rollouts). "
                "paper auto_research/paper/paper.md §5 LLM Evolution via GRPO (Phase 3a+3b "
                "   described as joint SFT-then-GRPO pipeline; no existing comparator to "
                "   RIVET's two-stage RL+SFT framework cited). "
                "paper auto_research/paper/paper.md §2 Related Work → RL for Code "
                "   Generation (ends at arxiv:2607.08255; RIVET Sep 11 not yet added). "
                "results/e2e_4cyc_gpt55/ — 4-cycle MERA evolution; skills arm 75.61% at "
                "   cycle-3 (GRPO phase) vs. RIVET-4B 44.16% accuracy after expert removal "
                "   on competition math (different domain; direct numeric compare not valid)."
            ),
            "metric": (
                "Nie et al. (arxiv:2609.12578, Sep 11 2026) propose RIVET (Routing to "
                "Internalizing via Verified Expert Traces), a two-stage expert internalization "
                "framework: "
                "(Stage I) expert-augmented reinforcement learning: a compact controller "
                "   model (1.7B or 4B) queries a stronger expert LLM and applies a shared "
                "   outcome reward signal to BOTH the controller's routing decisions AND the "
                "   expert's returned solution spans. This is analogous to MERA's Phase 3b "
                "   (GRPO), where the small student model collects rollouts with binary "
                "   pass/fail reward. Key difference: RIVET's Stage I augments the RL "
                "   gradient with the expert's return, whereas MERA's GRPO trains only "
                "   on the student's own rollouts (the oracle trace is used only in Phase 3a). "
                "(Stage II) verified trajectory internalization: successful complete "
                "   interaction traces (controller + expert) are used as SFT targets, "
                "   consolidating the expert's reasoning into the controller's weights. This "
                "   is analogous to MERA's Phase 3a (SFT from teacher traces), but MERA "
                "   runs SFT BEFORE GRPO (SLR schedule: Skill → LLM → Router), whereas "
                "   RIVET runs RL first, then SFT from the RL-produced successful traces. "
                "Key RIVET result: Stage II improves RIVET-4B by 6.49pp on competition-math "
                "   benchmarks after full expert removal. RIVET-4B reaches 44.16% mean "
                "   accuracy across 7 competition-math benchmarks. "
                "Connection to MERA (for §2 and §5 positioning): "
                "  (1) MERA and RIVET share the same high-level loop: "
                "      route-to-expert → collect verified traces → train student on traces "
                "      → student eventually handles tasks without expert. "
                "  (2) MERA's key distinctions over RIVET: "
                "      (a) MULTI-CYCLE co-evolution: MERA iterates 4 cycles with a "
                "          co-evolving router; RIVET is single-stage expert internalization. "
                "      (b) ROUTER co-evolution: MERA trains a binary classifier (Phase 4) "
                "          from execution traces each cycle; RIVET's routing is implicit in "
                "          the controller model's action selection (no separate router). "
                "      (c) TASK DOMAIN: MERA covers HumanEval (code generation) and "
                "          tau2-bench (multi-domain agentic tool use); RIVET covers "
                "          competition math with Python execution. "
                "      (d) GRPO ORDER: MERA runs SFT→GRPO (SLR); RIVET runs RL→SFT. "
                "          This ordering matters because SFT-first provides a better RL "
                "          initialization (less token-exploration needed), while RL-first "
                "          collects harder traces for SFT. "
                "  (3) MERA's SkillBook (§3) provides additional procedural conditioning "
                "      not present in RIVET: the procedure prefix f'{procedure}\\n\\n---\\n\\n{problem}' "
                "      guides the student with extracted solving strategies, acting as a "
                "      lightweight expert proxy even after expert removal. RIVET's Stage II "
                "      internalizes all expert reasoning into weights; MERA's SkillBook "
                "      externalizes part of it into reusable text procedures. "
                "Audit tasks: "
                "  (a) Read RIVET §3–4 (method) and §5 (results). Confirm Stage I = RL "
                "      + expert augmentation, Stage II = SFT from RL-verified traces. "
                "  (b) Identify whether RIVET's binary reward (correct/incorrect on math) "
                "      is comparable to MERA's binary pass/fail on code execution. "
                "  (c) Draft §2 RL for Code Generation addition (4–5 sentences): "
                "      'RIVET [Nie et al., 2026; arxiv:2609.12578] introduces expert "
                "      internalization as a two-stage RL+SFT framework: a compact model "
                "      first trains with expert-augmented RL, then consolidates successful "
                "      interaction traces via SFT (+6.49pp on competition math after expert "
                "      removal). MERA similarly routes to a GPT-5.5 oracle for verified "
                "      traces and trains the small model via SFT (Phase 3a) then GRPO "
                "      (Phase 3b), but iterates this loop across N co-evolution cycles with "
                "      a joint router and SkillBook that dynamically re-calibrate as the "
                "      student improves. Unlike RIVET's single-stage internalization, MERA's "
                "      router and SkillBook statistics shift each cycle, progressively "
                "      expanding the small model's autonomous task coverage.' "
                "  (d) Draft §5 LLM Evolution addition (2–3 sentences, end of §5 intro): "
                "      Cross-reference RIVET as the closest published two-stage comparator "
                "      and state the ordering difference (SFT→GRPO vs RL→SFT). "
                "  (e) Add bib entry rivetinternalize2026. "
                "Output: §2 addition + §5 cross-reference + bib entry."
            ),
            "estimated_gpu_hours": 0,
            "expected_paper_impact": (
                "RIVET (Sep 11 2026) is the closest published system to MERA's Phase 3a+3b "
                "evolution mechanism: an ICLR 2027 reviewer on LLM distillation or code RL "
                "who reads RIVET will ask why MERA doesn't cite it. Adding a clear comparison "
                "closes the most likely novelty-vs-RIVET objection. "
                "Positioning value: MERA's multi-cycle co-evolution and joint router are "
                "the key differentiators — this audit makes them explicit in §2 and §5. "
                "Soundness +0.1 (closes a gap between MERA's SFT-first ordering and "
                "RIVET's RL-first ordering; the ordering rationale can now be stated). "
                "One new Sep 2026 citation at the most directly comparable method. "
                "Estimated 60 min to read and draft."
            ),
        },
        "rationale": (
            "arxiv:2609.12578 (RIVET, Sep 11 2026) is a new paper that proposes expert "
            "internalization via a two-stage RL+SFT framework — the same high-level loop "
            "as MERA's Phase 3b (GRPO) + Phase 3a (SFT). It is the closest published "
            "comparator to MERA's model evolution mechanism that has appeared in the 2609 "
            "series (Sep 2026 submissions). The paper is not yet cited in MERA; with ICLR "
            "2027 in 17 days, an uncited Sep 11 paper at the same core problem is a reviewer "
            "risk. The audit surfaces the key differentiators (multi-cycle, co-evolving "
            "router, SFT-first ordering, SkillBook procedural conditioning) that make MERA "
            "novel relative to RIVET. Offline / 0h GPU; estimated 60 min."
        ),
        "iclr_target_section": "§2 RL for Code Generation / §5 LLM Evolution via GRPO",
    },
    {
        "id": "EXP-215",
        "priority": 7,
        "title": (
            "Procedural Graphs §3 SkillBook Positioning: MERA vs. arxiv:2609.09153 "
            "'Self-Evolving Execution Structures' for ICLR 2027"
        ),
        "paper": "arxiv:2609.09153",
        "paper_title": (
            "Procedural Graphs: Self-Evolving Execution Structures for LLM Agents"
        ),
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": (
                "src/skills.py — SkillBook class and extract_signature(); single global "
                "   skill 'coding'; skill text accumulated from teacher traces; "
                "   can_downgrade_to_small(): Laplace-smoothed confidence estimate per "
                "   (signature, model_role). Flat per-signature text, no graph structure. "
                "src/pipeline/collect_traces.py — skill distillation inline: GPT-5.5 traces "
                "   → skillbook.json update each cycle. The procedure is a text paragraph "
                "   summarizing solving strategies, not a graph of sub-procedures. "
                "paper auto_research/paper/paper.md §3 SkillBook: Signature-Based Routing "
                "   Evolution (describes SkillBook as online statistics + skill text; "
                "   design decision §1 states single global skill). "
                "paper auto_research/paper/paper.md §2 Related Work (no 'Procedural Memory "
                "   for Agents' subsection currently; Managing Procedural Memory in LLM "
                "   Agents [arxiv:2606.23127] is not yet cited)."
            ),
            "metric": (
                "Lu et al. (arxiv:2609.09153, Sep 8 2026) propose Procedural Graphs, which "
                "organize procedural knowledge for LLM agents as a graph of "
                "(procedure, relation, procedure) triplets — analogous to how knowledge "
                "graphs organize facts as (entity, relation, entity). At each decision step, "
                "the agent localizes its active node in the graph and receives step-level "
                "situational guidance derived from the surrounding subgraph, biasing the "
                "solver's next action without dictating it. "
                "Connection to MERA's SkillBook (for §3 positioning): "
                "  (1) Both Procedural Graphs and MERA's SkillBook address the same problem: "
                "      encoding procedural knowledge (what to do, in what order) to guide an "
                "      LLM agent's execution without weight updates. "
                "  (2) Procedural Graph uses structured graph topology: procedures are "
                "      linked by typed relations (e.g., 'precondition', 'subtask', 'retry'); "
                "      the graph grows via self-evolution from agent interaction traces. "
                "      MERA's SkillBook uses FLAT TEXT: the skill is a paragraph of free-form "
                "      solving strategies extracted by the distiller from oracle traces. "
                "      This is an intentional simplification: CLAUDE.md §1 'Single global "
                "      skill' — extract_signature() always returns 'coding' to avoid "
                "      per-cluster routing complexity. A graph structure would require "
                "      per-task node localization, which adds inference overhead and "
                "      is incompatible with the single-skill design. "
                "  (3) MERA's SkillBook includes ROUTING STATISTICS (Laplace-smoothed "
                "      success rates per model role) that Procedural Graph does not: the "
                "      SkillBook is simultaneously a procedural guide and a routing signal, "
                "      dual-purpose design justified by the single-signature constraint. "
                "  (4) Self-evolution mechanism: Procedural Graph grows its graph from "
                "      interaction traces (adding nodes/edges for new sub-procedures). "
                "      MERA's SkillBook text is re-synthesized by the distiller (GPT-5.5) "
                "      each cycle from fresh oracle traces — text replacement, not graph "
                "      growth. Simpler but cycle-level granularity only (no within-cycle "
                "      evolution). "
                "Audit tasks: "
                "  (a) Read Procedural Graphs §3 (method) and §4 (experiments). Confirm "
                "      the graph representation and self-evolution mechanism. Determine "
                "      whether 'self-evolving' matches MERA's across-cycle skill distillation. "
                "  (b) Check if arxiv:2606.23127 ('Managing Procedural Memory in LLM "
                "      Agents') is already cited in the paper. If not, determine if it "
                "      should be added alongside Procedural Graphs as a pair. "
                "  (c) Draft §3 SkillBook design rationale addition (3–4 sentences): "
                "      'Recent work structures procedural knowledge as graphs with typed "
                "      relations [Procedural Graphs, arxiv:2609.09153] or explicit memory "
                "      control mechanisms [arxiv:2606.23127]. MERA's SkillBook uses a flat "
                "      text representation — the entire coding domain is a single skill "
                "      (extract_signature() returns \"coding\") — intentionally avoiding "
                "      graph topology to minimize inference overhead and maintain a single "
                "      routing signal. The SkillBook is dual-purpose: the same per-signature "
                "      entry stores both the distilled procedure text (fed to the small model) "
                "      and the Laplace-smoothed success statistics (used by the router), a "
                "      co-location not present in graph-structured alternatives.' "
                "  (d) Add bib entries proceduralgraphs2026 and (if not present) "
                "      proceduralmemory2026 (for arxiv:2606.23127). "
                "Output: §3 design rationale addition + 1-2 bib entries."
            ),
            "estimated_gpu_hours": 0,
            "expected_paper_impact": (
                "§3 SkillBook currently lacks a comparison to structured procedural memory "
                "approaches. Adding 3–4 sentences that cite Procedural Graphs and justify "
                "MERA's flat-text design choice (single-skill, dual-purpose, no graph "
                "overhead) converts the simplicity from a potential weakness into a stated "
                "design decision. "
                "Clarity +0.1: the flat-vs-graph contrast makes the §1 'single global skill' "
                "decision legible to a reviewer familiar with graph-structured skill systems. "
                "Two new Sep 2026 citations (2609.09153 + possibly 2606.23127). "
                "Estimated 45 min to read and draft."
            ),
        },
        "rationale": (
            "arxiv:2609.09153 (Procedural Graphs, Sep 8 2026) proposes graph-structured "
            "procedural memory as the way to organize LLM agent skills — the structural "
            "alternative to MERA's flat SkillBook text representation. MERA's §3 SkillBook "
            "describes the flat design but does not explain why it is preferred over graph "
            "structures. An ICLR 2027 reviewer familiar with Procedural Graphs may ask why "
            "MERA doesn't use a richer representation. The audit adds a 3–4 sentence "
            "rationale citing the graph alternative and explicitly justifying MERA's choice "
            "(single-skill, dual-purpose, zero graph overhead). Offline / 0h GPU; ~45 min."
        ),
        "iclr_target_section": "§3 SkillBook: Signature-Based Routing Evolution",
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
