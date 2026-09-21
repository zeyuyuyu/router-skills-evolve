"""
Pending queue update — 2026-09-21
Run on A800 when connectivity is restored:
  python /data0/home/zeyuwang/auto_research/pending_queue_update_2026_09_21.py
Appends EXP-216 and EXP-217 to state["queue"] and saves atomically.

A800 offline since 2026-05-14 (day ~130). SSH port 50507 unreachable from remote
execution environment (TCP timeout; proxy is HTTPS-only, cannot tunnel SSH).
Queue ~213 pending (>20 cap → 2 experiments today).
Next target: ICLR 2027 (~Oct 1 deadline, ~10 days out). CRITICAL SPRINT.
Both experiments are OFFLINE / 0h GPU (paper positioning analyses for §2).

Hotspot source: WebSearch fallback (A800 hotspot file unavailable — A800 offline).
Top new papers found (this run, 2026-09-21):
  arxiv:2609.18779  "CERA-MoA: Co-Evolving Routing Mechanisms with Continually
    Learning LLM Agents" (Sep 16, 2026, Jiang et al.) — a direct co-evolutionary
    routing+continual-learning framework that shares MERA's core paradigm. The paper
    co-trains a dynamic router and independent agent policies via continual RL. Uses
    hidden-state familiarity estimators for routing; no explicit skill distillation.
    Most critical new paper for ICLR 2027 reviewers to find — must be differentiated
    in §2. → EXP-216: CERA-MoA §2 Co-Evolutionary Routing Positioning Audit.
  arxiv:2609.16604  "ExecuCritic: Calibrated Critic Shaping for Code Generation
    with Verifiable Rewards" (Sep 2026, Liu et al.) — GRPO variant with a joint
    critic that predicts pass/fail and gives diagnostic feedback on failing rollouts,
    producing denser credit signal than binary pass/fail. Directly targets the
    same problem as MERA's 52.4% ACR zero-variance groups. → EXP-217: ExecuCritic
    §2c/§3 GRPO Credit Assignment Positioning Audit.

Apply chain before this patch:
    python3 auto_research/pending_queue_update_2026_09_11.py  # EXP-210, EXP-211
    python3 auto_research/pending_queue_update_2026_09_12.py  # EXP-212, EXP-213
    python3 auto_research/pending_queue_update_2026_09_16.py  # EXP-214, EXP-215
"""

import json
import os
import shutil

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

NEW_EXPERIMENTS = [
    {
        "id": "EXP-216",
        "priority": 9,
        "title": (
            "CERA-MoA §2 Co-Evolutionary Routing Positioning: MERA vs. "
            "arxiv:2609.18779 Co-Evolving Router + Continual-RL Agents for ICLR 2027"
        ),
        "paper": "arxiv:2609.18779",
        "paper_title": (
            "CERA-MoA: Co-Evolving Routing Mechanisms with Continually Learning "
            "LLM Agents"
        ),
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": (
                "paper auto_research/paper/paper.md — §1 Introduction (co-evolutionary "
                "   framing of Router+Skills+LLM), §2 Related Work (LLM Routing + "
                "   Continual Learning subsections), §3 Pipeline Overview, abstract; "
                "src/pipeline/train_router_simple.py — router training on oracle labels; "
                "   iterative retraining each cycle as small model capability shifts. "
                "src/skills.py — SkillBook: single-skill procedure extraction from traces; "
                "   extract_signature() → 'coding' global skill. "
                "src/pipeline/collect_traces.py — oracle label construction, run-both design. "
                "CLAUDE.md — design decisions 1–4 (single global skill, router owns routing, "
                "   procedure format shared across SFT/GRPO/inference). "
                "results/e2e_4cyc_gpt55/ — routing accuracy cycle-0: ~80% → cycle-3: ~93.04%; "
                "   co-evolutionary improvement quantification across 4 cycles."
            ),
            "metric": (
                "CERA-MoA (arxiv:2609.18779, Sep 16 2026, Jiang, He, Fang) introduces a "
                "co-evolutionary iterative RL framework where the dynamic router and "
                "independent agent policies evolve together. A predictive familiarity "
                "estimator (mid-layer hidden states) evaluates semantic competence among "
                "agents without requiring full rollouts. A cumulative-threshold adaptive "
                "routing mechanism activates a tailored minimal agent subset. Agents are "
                "fine-tuned with continual RL on routed queries; the router is updated "
                "after each RL cycle to reflect the new capability distribution. "
                "MERA intersection / differentiation: "
                "(1) SHARED PARADIGM: Both MERA and CERA-MoA co-evolve a router and "
                "    model policies through iterative cycles. This validates MERA's core "
                "    design premise. Reviewers will notice this overlap immediately. "
                "    A clear differentiation sentence in §2 is essential. "
                "(2) AGENT POOL DESIGN: CERA-MoA uses N homogeneous agents of the same "
                "    size class (~4B each), forming a Mixture-of-Agents; MERA uses a "
                "    heterogeneous two-arm design (small: Qwen3-4B, large: frontier GPT-5.5) "
                "    with a strict cost hierarchy — 'escalate or not' routing vs. 'which "
                "    agent' routing. MERA's design explicitly exploits the capability gap "
                "    between a cheap local model and an expensive frontier API. "
                "(3) ROUTING SIGNAL: CERA-MoA routes via mid-layer hidden-state familiarity "
                "    embeddings (runtime overhead: a forward pass to the mid-layer for every "
                "    query). MERA routes via logistic regression on raw prompt text features "
                "    (zero runtime overhead beyond tokenization). MERA's router is also "
                "    trained on oracle binary labels (deterministic pass/fail) vs. CERA-MoA's "
                "    RL reward signal (possibly stochastic). "
                "(4) SKILL DISTILLATION: CERA-MoA has NO explicit skill distillation — agents "
                "    learn from RL on routed queries only. MERA's SkillBook extracts a written "
                "    procedure (from teacher traces) and prepends it to every prompt, giving "
                "    the small model a structured solving scaffold that generalizes beyond "
                "    seen examples. This is MERA's unique mechanism absent from CERA-MoA. "
                "(5) KNOWLEDGE SOURCE: CERA-MoA's agents learn from RL rollouts (on-policy, "
                "    self-reward). MERA's small model learns from frontier teacher traces "
                "    (SFT on GPT-5.5 solutions + GRPO on-policy rollouts), giving it access "
                "    to teacher knowledge unreachable by self-play alone. "
                "Audit tasks: "
                "  (a) Check §2 Related Work. Is there a 'co-evolutionary' or 'joint "
                "      routing+fine-tuning' cluster? If not, locate the best paragraph for "
                "      a CERA-MoA cite (likely §2 LLM Routing or a joint §2 para). "
                "  (b) Draft a positioning sentence: "
                "      'Concurrent work CERA-MoA [cera2026] also co-evolves a router and "
                "      agent policies via iterative RL cycles; unlike MERA, it operates on a "
                "      pool of homogeneous agents and routes via mid-layer hidden states, with "
                "      no explicit skill distillation or teacher knowledge injection.' "
                "  (c) Check §1 Introduction: does the abstract/intro claim MERA is the first "
                "      co-evolutionary design? If so, soften to 'to our knowledge, the first "
                "      to combine skill distillation + SFT + GRPO + router co-training.' "
                "  (d) Add bib entry cera2026 (arxiv:2609.18779). "
                "  (e) Check if the simultaneous-submission / concurrent-work convention "
                "      applies (CERA-MoA Sep 16, MERA submitted ~Oct 1 → concurrent; "
                "      use 'Concurrent work' phrasing). "
                "Output: §2 positioning sentence + bib entry + intro softening if needed."
            ),
            "estimated_gpu_hours": 0,
            "expected_paper_impact": (
                "CRITICAL: This is the highest-risk new paper for MERA's ICLR 2027 submission. "
                "CERA-MoA shares the 'co-evolving router + continual learning' paradigm and was "
                "posted Sep 16 — 15 days before the ICLR deadline. An ICLR reviewer discovering "
                "CERA-MoA during review (not finding it in §2) could issue a rejection: 'closely "
                "related concurrent work not cited, novelty unclear.' Addressing it proactively "
                "— with a clear differentiation on skill distillation, teacher-student gap, and "
                "routing mechanism — turns the threat into a validation of the paradigm. "
                "Priority: 9 (highest urgency); soundness +0.1; novelty defense essential."
            ),
        },
        "gpu": "auto",
    },
    {
        "id": "EXP-217",
        "priority": 7,
        "title": (
            "ExecuCritic §2c/§3 GRPO Credit Assignment Positioning: MERA's 52.4% ACR "
            "vs. Joint-Critic Diagnostic Shaping for ICLR 2027 (arxiv:2609.16604)"
        ),
        "paper": "arxiv:2609.16604",
        "paper_title": (
            "ExecuCritic: Calibrated Critic Shaping for Code Generation with Verifiable Rewards"
        ),
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": (
                "src/pipeline/grpo_train_simple.py — GRPO rollout logic, G=8 rollouts per "
                "   prompt, binary pass/fail reward, advantage = (reward − group_mean) / std. "
                "   Zero-variance groups (all pass or all fail) produce zero advantage. "
                "paper auto_research/paper/paper.md — §3 GRPO Training (ACR=52.4% finding, "
                "   arxiv:2507.05386 citation); §2c RL for Code (Cue-GRPO, Dr.GRPO, DAPO, "
                "   λ-GRPO referenced in prior runs; ExecuCritic arxiv:2609.16604 NOT present). "
                "results/e2e_4cyc_gpt55/ — GRPO reward traces showing ACR per cycle; "
                "   trajectory composition (hard-fail vs. partial-pass groups)."
            ),
            "metric": (
                "ExecuCritic (arxiv:2609.16604, Sep 2026, Liu et al.) addresses the sparse "
                "binary reward problem in GRPO for code generation: a unit test suite reduces "
                "each code rollout to a single pass/fail bit, leaving the policy gradient to "
                "solve a hard credit assignment problem from an impoverished signal. "
                "ExecuCritic proposes training a joint critic alongside the coder. The critic "
                "is updated on the same execution rollouts, predicting pass/fail outcomes and "
                "generating short diagnostic feedback (e.g., which test case fails and why). "
                "The coder uses the critic's feedback ONLY when the critic agrees with the "
                "executor (calibrated usage), avoiding noisy guidance. Across 8 code benchmarks "
                "and 2 recent open backbones, ExecuCritic improves over GRPO without a critic, "
                "prompted reviewer systems, and scalar reward model baselines, with fewer "
                "policy gradient steps and fewer sandbox executions. "
                "MERA intersection: "
                "(1) The 52.4% ACR (all-fail zero-variance group rate) documented in MERA's "
                "    §3 is exactly the problem ExecuCritic targets: failing rollouts contribute "
                "    zero advantage under GRPO but could receive diagnostic feedback from a "
                "    joint critic, allowing the policy to improve even on hard-fail groups. "
                "    ExecuCritic is a complementary remedy to CVPO (EXP-186) and Dr.GRPO (EXP-120). "
                "(2) MERA scope restriction: ExecuCritic adds a critic model at training time. "
                "    In MERA's pipeline, the critic would be an additional module in "
                "    src/pipeline/grpo_train_simple.py (Phase 3b). MERA does not currently "
                "    use a critic. This is a future-work cite, not a design flaw. "
                "(3) §2c RL for Code positioning: ExecuCritic is the most recent (Sep 2026) "
                "    execution-feedback GRPO variant. The cluster of GRPO variants in MERA's "
                "    §2c now includes: Dr.GRPO (masked zero-variance), DAPO (clip-higher), "
                "    λ-GRPO (step-imbalance), CVPO (variance-aware curriculum), Cue-GRPO "
                "    (rarity-aware credit), G²RPO-A (adaptive guided), and now ExecuCritic "
                "    (joint critic + diagnostic feedback). ExecuCritic fills the 'critic-based "
                "    credit assignment' niche not yet represented in §2c. "
                "Audit tasks: "
                "  (a) Check §2c RL for Code in paper.md. Is there a sentence about critic-based "
                "      or diagnostic-feedback GRPO variants? If not, add ExecuCritic as the "
                "      representative: 'ExecuCritic [execucritic2026] trains a joint critic "
                "      on the same execution rollouts, providing calibrated diagnostic feedback "
                "      to reduce the credit assignment burden of binary pass/fail rewards; "
                "      MERA currently uses vanilla GRPO and reports a 52.4% all-fail group "
                "      rate as a known limitation.' "
                "  (b) Check §3 GRPO Training discussion of ACR=52.4%: is there a "
                "      'future work' pointer to critic-based remedies? If not, add a forward "
                "      reference: 'Critic-based shaping (e.g., ExecuCritic [execucritic2026]) "
                "      could provide denser signal on failing groups; we leave this to future work.' "
                "  (c) Add bib entry execucritic2026 (arxiv:2609.16604). "
                "Output: §2c positioning sentence + §3 future-work pointer + bib entry."
            ),
            "estimated_gpu_hours": 0,
            "expected_paper_impact": (
                "§2c RL for Code gains a Sep 2026 critic-based GRPO citation (ExecuCritic), "
                "filling the 'credit assignment' niche in the GRPO variant landscape. "
                "§3 GRPO Training gains a future-work forward reference that preemptively "
                "addresses reviewer questions about the 52.4% ACR: 'why not use a critic?' "
                "The answer ('ExecuCritic is complementary and left to future work') is both "
                "honest and technically grounded. "
                "Soundness 0; Related-Work coverage +0.1; 1 new Sep 2026 citation."
            ),
        },
        "gpu": "auto",
    },
]


def main():
    if not os.path.exists(STATE_PATH):
        print(f"ERROR: state.json not found at {STATE_PATH}")
        return

    tmp_path = STATE_PATH + ".tmp_0921"
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
