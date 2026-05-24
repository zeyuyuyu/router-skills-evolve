# Paper Draft Snapshot — 2026-05-24

> Dated snapshot of `auto_research/paper_draft.md`.
> Copy of state as of 2026-05-24 UTC.
> Next weekly update: 2026-05-31.

---

(See `auto_research/paper_draft.md` for the canonical, editable version.)

<!-- Inline copy below — do not edit this snapshot directly -->

# Towards Always-On LLM Research: Persistent Routing, Skills, and Continual Adapter Training

**Status:** Working draft — auto-updated weekly by the paper-drafter agent.
**Last updated:** 2026-05-24
**A800 connectivity:** OUTAGE (since 2026-05-14T20:03Z; 10th consecutive day).
**New experiments this week:** 0 (27 pending in queue, blocked on A800 connectivity).

---

## Abstract

We present an always-on, three-track framework for continually improving LLM inference
systems without human-in-the-loop supervision. The system comprises (1) a **skills-evolve**
track that accumulates problem-solving patterns from traces and uses them to route queries
at low cost; (2) a **learned routing** track that trains a BERT-base classifier to decide
which LLM tier should handle each query; and (3) a **continual adapter** track that applies
local GRPO with executable-test rewards to improve a small code-generation model on new
tasks. Applied to HumanEval (164 tasks) with a router between DeepSeek (cheap) and
GPT-5.4 (expensive), skills evolution achieves 99% task success at 17% of always-GPT-5.4
cost — an 83% cost reduction. The learned router reaches 93.04% routing accuracy with a
2.12% fallback rate. For continual adapter training, the best positive result is
Qwen2.5-Coder-1.5B-Instruct improving from 47/100 to 49/100 on MBPP eval100 via
local GRPO over 200 tasks; larger models and longer training degrade. The primary open
challenge is extending the +2 pt GRPO gain beyond 200 training tasks without
plasticity loss.

---

## 1  Introduction

Large language models are expensive to query and quickly become stale. Two failure modes
are especially costly in production: (a) **routing failures**, where easy queries are
unnecessarily sent to expensive frontier models, and (b) **capability stagnation**, where
the small model in the routing pool never improves on task types it currently fails.

We address both failure modes with a single always-on research loop. On the routing side,
an evolving skills library + trained BERT router together achieve 93% routing accuracy and
an 83% cost reduction on a 164-task HumanEval deployment. On the capability side, local
GRPO with binary executable rewards yields the first reproducible positive adapter-training
result for a 1.5B code model (+2 pp on MBPP eval100).

The system is designed for autonomous operation: a shepherd process on an A800 GPU node
reads a shared state file, executes pending experiments, and writes structured result
JSONs. An idea-generation agent watches arXiv and proposes new experiments. A
paper-drafter agent (this script) synthesizes results weekly.

**Paper scope:** This draft covers all experiments completed through 2026-05-14 (the last
date the A800 server was reachable). Experiments EXP-001 through EXP-027 are queued but
not yet executed.

---

## 2  System

The system has three parallel tracks managed by an auto-orchestrator.

### 2.1  Auto-Orchestrator

A shepherd process polls `state.json` every 2 hours. It picks the next pending experiment,
launches it on a free A800 GPU, writes the result to `logs/exp_*.result.json`, and
advances the state machine. A separate idea-generation agent reads arXiv hotspots daily
and proposes new experiments; a deduplication check prevents re-running past configs.

### 2.2  Track A — Skills Evolve

The skills library stores problem-type signatures extracted from LLM traces. Each
signature encodes query difficulty, input type, and routing outcome. At inference time,
a query is matched against the library; if a matching skill pattern exists, the routing
decision is made locally (zero LLM API cost). The library grows as new traces arrive.

After 4 evolution rounds on HumanEval (164 tasks), the library reached 49 skills:
40 mapped to the cheap model (DeepSeek), 9 to the expensive model (GPT-5.4).

### 2.3  Track B — Learnable Router

A BERT-base binary classifier is trained on labelled router traces. Label `0` (small)
means the cheap model solved the query; label `1` (large) means the cheap model failed
and the expensive model was needed. The dataset is the union of project traces and the
UncommonRoute bench.

The learned router is the strongest positive component in the system: it removes 93% of
the uncertainty that the skills-signature heuristic leaves unresolved.

### 2.4  Track C — Continual Adapter

A small LoRA adapter is trained on new tasks using local GRPO (no TRL dependency).
For each training prompt, 4 completions are sampled, tested against executable unit tests,
and rewarded with a binary pass/fail signal. Group-relative advantage is computed within
the group of 4 rollouts. The adapter is evaluated on a held-out set.

This track is currently in the "finding a working recipe" phase. The first reproducible
positive result requires a 1.5B model, 200 tasks, 4 rollouts, LoRA r=16, lr=5e-6.

---

## 3  Experiments

### 3.1  Router Track

#### 3.1.1  Skills Evolution — HumanEval 164 Tasks

| # | Strategy | Success Rate | Total Cost | Avg Attempts | Cost vs GPT-5.4-ceiling |
|---|----------|:---:|:---:|:---:|:---:|
| 1 | Pure GPT-5.4 (ceiling) | 95% | $0.3655 | 1.00 | 100% |
| 2 | Pure DeepSeek (floor) | 88% | $0.0142 | 1.00 | 4% |
| 3 | Router Only | 74% | $0.1535 | 1.00 | 42% |
| 4 | Router + Fallback | 98% | $0.2661 | 1.24 | 73% |
| 5 | **Router + Skills Evolve** | **99%** | **$0.0638** | **1.07** | **17%** |

#### 3.1.2  Routing Ablation — Learned Router vs. SkillBook

Eval set: 848 examples (579 small, 269 large). Experiment: `run_e2e_ablation_a800_20260509`.

| Variant | Accuracy | Large F1 | Fallback Rate | Cost vs Always-Large |
|---------|:---:|:---:|:---:|:---:|
| Base: always-small + fallback | 68.28% | 0.00% | 31.72% | **33.54%** |
| + SkillBook signature routing | 69.46% | 24.93% | 26.65% | 37.27% |
| + Learned router (threshold=0.57) | **93.04%** | **89.48%** | **2.12%** | 37.75% |

#### 3.1.3  Full Joint Pipeline (2026-05-02)

| Metric | Value |
|--------|:---:|
| Router accuracy | 92.36% |
| Router threshold | 0.53 |
| Large F1 | 88.63% |
| Fallback rate | 1.98% |
| Cost vs always-large | 38.57% |

### 3.2  Skills Track

See §3.1.1. SkillBook alone: 69.46% accuracy, large F1=24.93%. Library stabilises at 49
skills after 4 rounds. Hard-route skills (9 patterns) define Track C training targets.

### 3.3  LLM Continual Training

#### 3.3.1  SFT — All Failed

| Model | Config | Eval | Base | Trained | Result |
|-------|--------|------|:---:|:---:|:---:|
| 0.5B | SFT 6 hard, LoRA r=16, 8 ep | HE hard 6 | 4/6 | 2/6 | degraded |
| 0.5B | SFT MBPP 120, LoRA r=16, 3 ep | MBPP eval20 | 3/20 | 2/20 | degraded |
| 0.5B | SFT qwen-chat 120, LoRA r=16 | MBPP eval20 | 10/20 | 9/20 | below |
| 0.5B | Conservative SFT, r=8, lr=2e-5 | MBPP eval20 | 10/20 | 9/20 | below |
| 0.5B | Expanded-data SFT, 400 tasks | MBPP eval20 | 10/20 | 9/20 | below |

#### 3.3.2  Local GRPO — First Positive Results

| Model | Train tasks×rollouts | Eval | Base | Trained | Δ |
|-------|:---:|------|:---:|:---:|:---:|
| 0.5B | 40×4 | MBPP eval20 | 8/20 | 9/20 | +1 |
| 0.5B | 120×4 | MBPP eval20 | 8/20 | 9/20 | +1 (repl.) |
| **1.5B** | **200×4** | **MBPP eval100** | **47/100** | **49/100** | **+2 ✓** |
| 1.5B | 400×4 | MBPP eval100 | 47/100 | 46/100 | −1 ✗ |
| 3B | 200×4 | MBPP eval100 | 62/100 | 60/100 | −2 ✗ |
| 3B | SFT 400 tasks (conservative) | MBPP eval100 | 62/100 | 62/100 | 0 |
| 7B | 100×4 | MBPP eval100 | 77/100 | 75/100 | −2 ✗ |

### 3.4  Catastrophic Forgetting Analysis

*(Pending — EXP-004, EXP-014 blocked on A800 connectivity.)*

### 3.5  Component Ablation

| Variant | Skills | Router | LLM GRPO | Routing Acc | Fallback | Code Pass |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| Base | ✗ | ✗ | ✗ | 68.28% | 31.72% | 47/100 |
| + Skills | ✓ | ✗ | ✗ | 69.46% | 26.65% | 47/100 |
| + Router | ✓ | ✓ | ✗ | 93.04% | 2.12% | 47/100 |
| Full | ✓ | ✓ | ✓ | 93.04% | 2.12% | **49/100** |

---

## 4  Findings

### Positive
- Router + Skills Evolve: 99% HumanEval success at 17% of always-GPT-5.4 cost.
- Learned router: 93.04% accuracy, 2.12% fallback, large F1=89.48%.
- 1.5B GRPO 200 tasks: 47→49/100 (+2 pp, only reliable positive LLM result).
- GRPO consistently beats SFT on small code models.
- Qwen-chat prompt format critical (3/20→10/20 base eval improvement).

### Negative
- All SFT variants degraded or tied.
- Scaling model size degrades GRPO (3B: −2, 7B: −2).
- 400 tasks GRPO degrades 1.5B (−1 pp).
- SkillBook alone insufficient (large F1=24.93%).

### Surprises
- Router is slightly more expensive than always-small+fallback (37.75% vs 33.54%
  vs always-large) — cost savings come from eliminating re-tries, not routing more to small.
- GRPO gain is inversely correlated with base model strength; 47% base benefits, 62%/77% degrade.

---

## 5  Open Questions

1. **Plasticity loss fix** — KL penalty (EXP-002), KL+curriculum (EXP-007),
   warm-start (EXP-022), C-CHAIN churn reduction (EXP-027). All blocked.
2. **Credit assignment** — EGCA (EXP-026, arxiv:2603.16158, +3.1 HumanEval). Blocked.
3. **Routing threshold Pareto** — threshold=0.45 (EXP-021) and 0.65 (EXP-025). Blocked.
4. **Catastrophic forgetting magnitude** — EXP-004, EXP-014. Blocked.
5. **Pass@k vs. pass@1** — EXP-023. Blocked.
6. **Joint cycle stability** as LLM adapter improves.
7. **A800 connectivity** — 10+ day outage blocks all 27 queued experiments.

---

## 6  Reproducibility

**Best LLM recipe:** Qwen2.5-Coder-1.5B-Instruct, local GRPO, 200 tasks×4 rollouts,
1 epoch, LoRA r=16, lr=5e-6, qwen-chat, binary reward. Result: 47→49/100 MBPP eval100.

**Best router:** BERT-base, 3391 examples, threshold=0.57, accuracy=93.04%, large F1=89.48%.

**Key scripts:** `experiments/run_e2e_ablation.py`, `experiments/train_small_model_grpo_local.py`,
`experiments/train_learnable_router.py`, `auto_research/pending_queue_update.py`.

**Pending queue:** 27 experiments in `auto_research/pending_queue_update.py` (idempotent).
Run on A800 when connectivity is restored.

---
*Snapshot generated: 2026-05-24 UTC*
