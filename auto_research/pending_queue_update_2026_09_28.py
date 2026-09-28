"""
Pending queue update — 2026-09-28
Run on A800 when connectivity is restored:
  python /data0/home/zeyuwang/auto_research/pending_queue_update_2026_09_28.py
Appends EXP-220 and EXP-221 to state["queue"] and saves atomically.

A800 offline since 2026-05-14 (day ~137). SSH port 50507 unreachable from remote
execution environment (TCP timeout; proxy is HTTPS-only and cannot tunnel raw SSH).
sshpass not installed in remote env; auto-mode classifier also blocks SSH attempts.
Queue ~215+ pending (>20 cap → 2 experiments today).
Next target: ICLR 2027 (~Oct 1 deadline, ~3 days out). FINAL 72-HOUR PUSH.
Both experiments are OFFLINE / 0h GPU (paper positioning + metadata audits).

Context (2026-09-28):
  ICLR deadline: ~Oct 1 (3 days). Paper v13 composite 7.0/10.
  Key discovery today: Co-Skill (arxiv:2609.16008, revised Sep 17) shares MERA's cloud->edge
  skill transfer paradigm and is absent from paper.md. This is the SkillBook-space counterpart
  to CERA-MoA (EXP-216) in the routing space — a concurrent paper that must be cited.
  MERA is already on arxiv as 2608.10333 (submitted Aug 11, 2026).
  Binding blockers: W1 single-seed (EXP-099, needs A800), Co-Skill uncited (EXP-220, offline).

Hotspot source: WebSearch fallback (A800 hotspot file unavailable — A800 offline).
Top angles identified (this run, 2026-09-28):
  1. Co-Skill (arxiv:2609.16008, Sep 17): cloud-edge collaborative skill evolution, paradigm
     overlap with MERA's SkillBook. Revised 14 days before ICLR deadline, currently absent
     from §2. MERA differentiators: LoRA vs. prefix injection; single global skill vs.
     per-trajectory trie; co-trained router absent in Co-Skill; code+pytest vs. agent tasks.
     -> EXP-220: Co-Skill §2 Skill Distillation Positioning Audit.
  2. ICLR 2027 reproducibility/metadata: EXP-218 covered abstract/§2 text; remaining gaps
     are bib key completeness (10+ new keys since v13), GRPO hyperparameter table, §7
     Limitations Co-Skill pointer, MERA arxiv self-cite (2608.10333), figure caption accuracy.
     -> EXP-221: Final ICLR Reproducibility & Metadata Completeness Audit.

Apply chain before this patch:
    python3 auto_research/pending_queue_update_2026_09_11.py  # EXP-210, EXP-211
    python3 auto_research/pending_queue_update_2026_09_12.py  # EXP-212, EXP-213
    python3 auto_research/pending_queue_update_2026_09_16.py  # EXP-214, EXP-215
    python3 auto_research/pending_queue_update_2026_09_21.py  # EXP-216, EXP-217
    python3 auto_research/pending_queue_update_2026_09_26.py  # EXP-218, EXP-219
"""

import json
import os
import shutil

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

NEW_EXPERIMENTS = [
    {
        "id": "EXP-220",
        "priority": 9,
        "title": (
            "Co-Skill §2 Skill Distillation Positioning Audit — "
            "Cloud->Edge Skill Transfer Paradigm Differentiation (arxiv:2609.16008)"
        ),
        "paper": "arxiv:2609.16008",
        "paper_title": (
            "Co-Skill: A Collaborative Communication Framework for Skill Evolution"
        ),
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": (
                "auto_research/paper/paper.md — §2 Skill Distillation / Self-Evolving Agentic "
                "  Systems: locate SkillForge (skillforge2026) and SKILL-KD (skillkd2026) entries. "
                "auto_research/paper/references.bib — confirm skillforge2026, skillkd2026 exist; "
                "  add coskill2026 entry. "
                "auto_research/pending_queue_update_2026_09_21.py — EXP-216 (CERA-MoA) and "
                "  EXP-217 (ExecuCritic) differentiation sentences for style reference. "
                "CLAUDE.md design decision 1: single global skill — key MERA property "
                "  that distinguishes from Co-Skill's per-trajectory trie. "
                "CLAUDE.md design decision 2: Router owns routing; Skills only distill procedure. "
                "  Co-Skill has no separate router — the cloud is always invoked for synthesis. "
                "results/e2e_4cyc_gpt55/cycle_3/ — routing accuracy 93.04%, zero inference "
                "  overhead after training (vs. Co-Skill's per-query prefix injection overhead). "
            ),
            "metric": (
                "Co-Skill (arxiv:2609.16008, revised Sep 17, 2026): "
                "cloud LLM synthesizes per-trajectory skill trie; edge SLM absorbs via RL "
                "in-context prefix injection. Results: 25.8-76.4% success improvement, "
                "15.6-41.9% token reduction on ALFWorld/WebShop. "
                ""
                "MERA vs. Co-Skill key differentiators: "
                "1. Skill: Co-Skill per-trajectory trie; MERA single global paragraph (decision 1). "
                "2. Internalization: Co-Skill in-context prefix (runtime overhead); "
                "   MERA LoRA SFT+GRPO (zero inference overhead after training). "
                "3. Router: Co-Skill none (cloud always invoked); MERA 93.04% accuracy LR. "
                "4. Oracle: Co-Skill agent success/fail; MERA pytest beta=0 deterministic. "
                "5. Domain: agent nav/shopping vs. code generation + formal verifier. "
                ""
                "Audit tasks: "
                "(a) §2 Skill Distillation: insert 2-sentence Co-Skill differentiation after "
                "    SkillForge/SKILL-KD entries. Draft sentence provided in rationale. "
                "(b) Add bib entry coskill2026 to references.bib. "
                "(c) Optionally add COBRA-Skills (arxiv:2609.11682) and SkillAA (arxiv:2609.20455) "
                "    as background cites if §2 has space. "
                "(d) Confirm skillforge2026 and skillkd2026 bib keys are present. "
                ""
                "Output: 'Co-Skill cite added to §2 (coskill2026). §2 Skill Distillation "
                "now covers: SkillForge, SKILL-KD, Co-Skill, DSR, Skill-Gated Distillation.'"
            ),
            "estimated_gpu_hours": 0,
            "expected_paper_impact": (
                "Novelty defense: Co-Skill is the SkillBook-space counterpart to CERA-MoA. "
                "An ICLR reviewer in agent-skill space will find it (Sep 17, 2026). "
                "Without EXP-220: SkillBook novelty risk comparable to pre-EXP-216 CERA-MoA. "
                "With EXP-220: §2 Skill Distillation complete for Sep 2026 cluster. "
                "Novelty +0.1. GPU cost 0. Time ~20 minutes."
            ),
        },
        "gpu": "auto",
    },
    {
        "id": "EXP-221",
        "priority": 8,
        "title": (
            "Final ICLR 2027 Reproducibility & Metadata Completeness Audit — "
            "Bib Keys, GRPO Hyperparameters, §7 Limitations, arxiv Self-Cite"
        ),
        "paper": "arxiv:2608.10333",
        "paper_title": (
            "MERA: Model Evolution and Routing with Skill Adaptation for Agentic Systems "
            "at Scale (MERA's own arxiv preprint — self-cite audit)"
        ),
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": (
                "auto_research/paper/references.bib — full bib file for key-existence check. "
                "auto_research/paper/paper.md — §3 Training (GRPO hyperparameter table), "
                "  §7 Limitations, §A Appendix (reproducibility statement). "
                "auto_research/paper/meta.json — paper_version, scores, bib keys added per EXP. "
                "auto_research/pending_queue_update_2026_09_26.py — EXP-218/219 bib list. "
                "auto_research/pending_queue_update_2026_09_28.py — EXP-220 coskill2026 key. "
                "CLAUDE.md §design decision 8: GRPO_TEMPERATURE must be >0 (0.7-1.0 training). "
                "results/e2e_4cyc_gpt55/cycle_3/router/router_meta.json — 93.04% routing accuracy."
            ),
            "metric": (
                "ICLR 2027 submission final validation: "
                ""
                "(A) Bib key completeness: check references.bib for "
                "    grpoprm2026, dsr2026, srr2026, opd2026, cheapverifiers2026, "
                "    cauc2026, cera2026, execucritic2026, coskill2026, intercascade2025. "
                "    For each missing: draft @misc entry. "
                ""
                "(B) GRPO hyperparameter table (§3 or §A): "
                "    GRPO_TEMPERATURE=1.0, GRPO_BATCH_SIZE=1, K=8, GRPO_MAX_LEN=4096, "
                "    SFT_INCLUDE_SUCCESS=1, SCALING_FORCE_BOTH=1. "
                ""
                "(C) §7 Limitations Co-Skill pointer: "
                "    'MERA's single global skill sacrifices per-task granularity; richer "
                "    per-trajectory skill structures (Co-Skill [coskill2026], SkillAA) "
                "    are a natural extension.' "
                ""
                "(D) MERA arxiv self-cite (2608.10333): "
                "    Verify double-blind compliance — remove preprint cite if deanonymizing. "
                ""
                "(E) Figure/Table: curve.png caption = '4-cycle'; ablation = large/skills/router/full. "
                ""
                "(F) Reproducibility checklist: dataset availability, compute hours, code URL. "
                ""
                "Output: 'ICLR-ready: PASS/FAIL — N bib keys missing, arxiv self-cite safe/removed.'"
            ),
            "estimated_gpu_hours": 0,
            "expected_paper_impact": (
                "Missing bib keys -> compile errors -> desk rejection. "
                "Missing hyperparameter table -> ICLR reproducibility checklist violation. "
                "§7 Co-Skill pointer closes single-global-skill justification gap. "
                "Arxiv self-cite deanonymization: low probability, catastrophic risk. "
                "~30 minutes. Zero GPU. Highest submission-safety value per minute."
            ),
        },
        "gpu": "auto",
    },
]


def main():
    if not os.path.exists(STATE_PATH):
        print(f"ERROR: state.json not found at {STATE_PATH}")
        return

    tmp_path = STATE_PATH + ".tmp_0928"
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
