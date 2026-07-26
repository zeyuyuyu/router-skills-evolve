# MERA: Model Evolution and Routing with Skill Adaptation for Agentic Systems at Scale
<!-- paper_draft.md v8 — auto-updated 2026-07-26 by weekly paper pipeline -->
<!-- A800 offline day 72; 0 new A800 results; 8 experiments queued (EXP-126–EXP-133) -->
<!-- Snapshot: paper_draft-2026-07-26.md -->

**Zeyu Wang**  
0G.ai / Institute of Artificial Intelligence  
zeyu.wang@0g.ai

---

## Abstract

Large language model (LLM) serving faces a fundamental cost-quality trade-off: powerful
frontier models are expensive while cheaper models often fail on hard tasks. We present
**MERA** (Model Evolution and Routing with Skill Adaptation), a self-improving agentic
serving system that jointly evolves routing decisions and model capabilities via
reinforcement learning and supervised fine-tuning. MERA maintains a *SkillBook* of
per-signature routing statistics and a *learned router* trained from execution traces.
On HumanEval (164 code tasks), MERA achieves **99% task accuracy at 83% lower cost** than
always routing to the frontier model. A BERT-based router achieves **93.04% routing
accuracy** with a **2.12% fallback rate**. Over 4 end-to-end evolution cycles on HumanEval
with DAPO multi-turn repair (G=8), the skills arm follows a non-monotonic trajectory
(70.73%→65.85%→73.17%→75.61%), improving overall yet dipping at cycle 1—anti-correlated
with ACR (lowest ACR yet worst skills arm)—suggesting two additional failure modes beyond
zero-variance collapse: GRPO-induced forgetting of SFT gains (Hypothesis D) and
within-cycle rise-and-collapse (Hypothesis E). A mechanistic analysis reveals that
**52.4% of training groups produce zero gradient** even with DAPO dynamic sampling,
identifying zero-variance collapse as the binding improvement bottleneck. A standalone
GRPO pass on MBPP yields +2pp pass@1 for Qwen2.5-Coder-1.5B (n=1 seed). The
Group-Standard-Deviation Identity [groupsd2026] provides formal theoretical grounding:
GRPO, Dr. GRPO, and DAPO differ only in their treatment of within-group σ; our 52.4%
zero-variance finding corresponds precisely to the σ=0 regime. Extended to three-domain
agentic tau2-bench tasks with a Qwen3.6-35B-A3B adapter, MERA achieves **89.19% task
pass at 22.16% of always-large cost** at peak; a held-out evaluation shows the
domain-specialized 35B model (80%) surpassing the frontier GPT-5.4 (71%), revealing that
agent specialization can render frontier escalation counterproductive. Code and datasets
are released publicly.

*(Full paper content identical to auto_research/reports/paper_draft.md v8 — see that file for complete text.)*
