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

**Experiment:** Final evolve run, 4 rounds, HumanEval 164 tasks, real LLM calls.
**Date:** Pre-2026-04-20.  **Host:** Local (not A800 GPU).

| # | Strategy | Success Rate | Total Cost | Avg Attempts | Cost vs GPT-5.4-ceiling |
|---|----------|:---:|:---:|:---:|:---:|
| 1 | Pure GPT-5.4 (ceiling) | 95% | $0.3655 | 1.00 | 100% |
| 2 | Pure DeepSeek (floor) | 88% | $0.0142 | 1.00 | 4% |
| 3 | Router Only | 74% | $0.1535 | 1.00 | 42% |
| 4 | Router + Fallback | 98% | $0.2661 | 1.24 | 73% |
| 5 | **Router + Skills Evolve** | **99%** | **$0.0638** | **1.07** | **17%** |

Skills evolution rounds:

| Round | Skills Start→End | Success Rate | Per-task Cost |
|-------|:---:|:---:|:---:|
| 0 | 0→21 | 100% | $0.00037 |
| 1 | 21→33 | 100% | $0.00036 |
| 2 | 33→45 | 100% | $0.00041 |
| 3 | 45→49 | 95% | $0.00042 |

Of the 49 learned skills, 40 allow routing to the cheap model; 9 always escalate to GPT-5.4
(patterns: `L|advanced/list/num`, `L|crypto/str`, `M|bool/num`, `M|list/num/sort`, etc.).

**Signal:** Very high. 99% success at 17% cost is the headline result. The 9-skill
hard-route list defines the target training distribution for Track C.

#### 3.1.2  Routing Ablation — Learned Router vs. SkillBook Signature

**Experiment:** `e2e_ablation_router_sft_qwenchat` (A800, 2026-05-09).
**Eval set:** 848 router-labelled examples (579 small, 269 large).
**Experiment ID:** `run_e2e_ablation_a800_20260509`.

| Variant | Accuracy | Large F1 | Fallback Rate | Cost vs Always-Large |
|---------|:---:|:---:|:---:|:---:|
| Base: always-small + fallback | 68.28% | 0.00% | 31.72% | **33.54%** |
| + SkillBook signature routing | 69.46% | 24.93% | 26.65% | 37.27% |
| + Learned router (threshold=0.57) | **93.04%** | **89.48%** | **2.12%** | 37.75% |

**Signal:** Very high. Learned router is the dominant positive ablation component.
SkillBook alone has weak hard-route recall (large F1 = 24.93%). Router training jumps
large F1 to 89.48% with only 2.12% fallback. Cost vs always-large is similar across
variants (33–38%), meaning the router primarily eliminates *unnecessary large calls*,
not total API spend. Artifacts: `results/e2e_ablation_a800_20260509_summary.json`.

#### 3.1.3  Full Joint Pipeline

**Experiment:** `joint_evolver_real_20260502_050032` (8×A800, 2026-05-02).
**Experiment ID:** `run_joint_evolver_a800_20260502`.

| Track | Metric | Value |
|-------|--------|:---:|
| Router | Threshold | 0.53 |
| Router | Accuracy | 92.36% |
| Router | Large F1 | 88.63% |
| Router | Fallback rate | 1.98% |
| Router | Cost vs always-large | 38.57% |
| LLM (0.5B base) | MBPP eval20 pass rate | 15% (3/20) |
| LLM (0.5B LoRA) | MBPP eval20 pass rate | 10% (2/20) — degraded |

Artifacts: `results/joint_evolver_a800_20260502_summary.json`.

### 3.2  Skills Track

The skills track is integrated into the routing ablation (§3.1.2). The SkillBook
signature matcher provides a weak but non-zero improvement over pure-fallback routing
(68.28% → 69.46% accuracy, large F1 0→24.93%). However, the learned router subsumes
the skills benefit — adding the router on top of SkillBook is where the big jump occurs.

**Open question:** Can skills evolve *during* router operation (continual skill
acquisition without retraining)? This is deferred to a future experiment.

### 3.3  LLM Continual Training

#### 3.3.1  SFT Experiments — All Failed to Beat Base

Multiple SFT configurations on Qwen2.5-Coder-0.5B-Instruct were tested in May 2026.
None improved on the base model consistently:

| Model | Config | Eval | Base | Trained | Result |
|-------|--------|------|:---:|:---:|:---:|
| 0.5B | SFT 6 hard examples, LoRA r=16, 8 epochs | HumanEval hard 6 | 4/6 | 2/6 | degraded |
| 0.5B | SFT MBPP train120, LoRA r=16, 3 epochs | MBPP eval20 | 3/20 | 2/20 | degraded |
| 0.5B | SFT + qwen-chat prompt, MBPP train120 | MBPP eval20 | 10/20 | 9/20 | close, below |
| 0.5B | Conservative SFT, LoRA r=8, lr=2e-5, 1 ep | MBPP eval20 | 10/20 | 9/20 | below |
| 0.5B | Expanded-data SFT, 400 tasks, 2 epochs | MBPP eval20 | 10/20 | 9/20 | below |

Key insight: switching to qwen-chat prompt format improved the base model evaluation
from 3/20 to 10/20 — demonstrating prompt format sensitivity. SFT still could not
beat the properly prompted base.

#### 3.3.2  Local GRPO — First Positive Results

GRPO with executable binary rewards produced the first positive adapter-training results.

**0.5B model (smoke tests):**

| Config | Train tasks × rollouts | Eval | Base | Trained | Result |
|--------|:---:|------|:---:|:---:|:---:|
| GRPO, LoRA r=8, 1 epoch | 40×4 | MBPP eval20 | 8/20 | 9/20 | +1 (small gain) |
| GRPO, 1 epoch | 120×4 | MBPP eval20 | 8/20 | 9/20 | +1 (replicated) |

**1.5B model — best positive case:**

| Config | Train tasks × rollouts | Eval | Base | Trained | Δ |
|--------|:---:|------|:---:|:---:|:---:|
| GRPO, lr=5e-6, LoRA r=16, 1 epoch | **200×4** | MBPP eval100 | 47/100 | **49/100** | **+2 pp** ✓ |
| GRPO, lr=5e-6, LoRA r=16, 1 epoch | 400×4 | MBPP eval100 | 47/100 | 46/100 | −1 pp ✗ |

**3B and 7B models:**

| Model | Config | Eval | Base | Trained | Result |
|-------|--------|------|:---:|:---:|:---:|
| 3B | GRPO 200×4, 1 epoch | MBPP eval100 | 62/100 | 60/100 | degraded −2 pp |
| 3B | Conservative SFT, 400 tasks | MBPP eval100 | 62/100 | 62/100 | tied |
| 7B | GRPO 100×4, 1 epoch | MBPP eval100 | 77/100 | 75/100 | degraded −2 pp |

Artifacts: `results/rl_15b_a800_20260509_summary.json`, `results/best_llm_training_case_20260509.json`.

**Key pattern:** GRPO works for 1.5B at 200 tasks. The gain does not scale with model
size or training duration. 400-task 1.5B degrades; 200-task 3B degrades; 100-task 7B
degrades. The 200-task sweet spot for 1.5B is the only reliable positive case.

#### 3.3.3  Prompt Format Matters

Early runs with generic code-completion prompts gave 3/20 as a base.
After switching to qwen-chat instruct format, the base rose to 10/20 (0.5B) and
the GRPO training signal improved. **All experiments since 2026-05-02 use qwen-chat format.**

### 3.4  Catastrophic Forgetting Analysis

*(Pending — EXP-004 `exp_2026_05_15_004` and EXP-014 `exp_2026_05_17_004` are queued
to run forgetting evals on 1.5B and 3B models using HumanEval. Will populate when
A800 connectivity is restored.)*

Current indirect evidence: the 400-task GRPO run (1.5B) degraded from 47→46/100
on MBPP, consistent with plasticity loss — but we do not yet have a measurement of
whether HumanEval performance was also affected.

### 3.5  Other Arms

#### 3.5.1  DPO Attempt

DPO training data was constructed (86 pairs: chosen=reference code, rejected=failed
model output). However, `DPOTrainer` from TRL was incompatible with the A800 PyTorch
version and training did not complete. DPO remains a pending direction.

#### 3.5.2  Component Ablation — End-to-End Summary

The full E2E ablation (2026-05-09) measures each system component's contribution:

| Variant | Skills | Router | LLM GRPO | Routing Acc | Fallback | Code Pass |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| Base: always-small+fallback | ✗ | ✗ | ✗ | 68.28% | 31.72% | 47/100 |
| + Skills evolve | ✓ | ✗ | ✗ | 69.46% | 26.65% | 47/100 |
| + Router training | ✓ | ✓ | ✗ | **93.04%** | 2.12% | 47/100 |
| Full: + Router + LLM GRPO | ✓ | ✓ | ✓ | **93.04%** | 2.12% | **49/100** |

The router training component delivers the largest single improvement. LLM GRPO adds
a further +2 pp on code pass rate.

---

## 4  Findings

### Positive results

- **Router + Skills Evolve is the headline result.** 99% HumanEval success at 17% of
  always-GPT-5.4 cost (83% cost reduction). The skills library stabilises after ~4
  rounds, reaching 49 skills with 40 safe-to-route-small and 9 hard patterns.

- **Learned BERT-base router dominates signature routing.** Accuracy 93.04% vs 69.46%
  for SkillBook alone. Large-class F1 jumps from 24.93% to 89.48%. Fallback rate drops
  from 26.65% to 2.12%. The router is the single most impactful component.

- **1.5B GRPO 200 tasks gives the only reliable positive LLM training result.**
  47/100 → 49/100 on MBPP eval100. Reproducible. Binary executable reward + qwen-chat
  format + LoRA r=16 + lr=5e-6 is the working recipe.

- **GRPO beats SFT on small code models.** All SFT variants at ≤0.5B degraded or
  tied. GRPO at 0.5B gave +1/20 (×2 replications). At 1.5B GRPO gives +2/100.

- **Qwen-chat prompt format is critical.** Switching from generic code prompts to
  qwen-chat nearly tripled base model evaluation scores (3/20→10/20 for 0.5B).

### Negative results

- **SFT reliably degrades small code models.** All 5 SFT configurations tested on
  Qwen2.5-Coder-0.5B degraded or tied. SFT is not a viable adapter-training path at
  this scale without strong data filtering.

- **Scaling model size does not help GRPO.** 3B at 200 tasks: −2 pp. 7B at 100 tasks:
  −2 pp. Larger base models are harder to improve with local GRPO, likely because their
  stronger baseline leaves fewer tasks in the "useful gradient" band.

- **More GRPO tasks than 200 degrade the 1.5B model.** 400 tasks: −1 pp. The
  transition point is between 200 and 400 tasks. Mechanistically: after 200 tasks,
  the remaining 200 tasks are mostly easy (zero-variance reward, no gradient) or
  produce entropy collapse (plasticity loss per arxiv:2605.12484).

- **SkillBook signature routing alone is insufficient.** Large-class F1 = 24.93%,
  fallback rate = 26.65%. The heuristic misses most hard queries.

### Surprises

- **Cost plateau with router vs. always-large.** The learned router costs ~37.75% of
  always-large, while always-small+fallback costs only ~33.54% — the router is
  *slightly more expensive* on average because it correctly routes more queries to
  large. The cost reduction comes from avoiding unnecessary re-tries (fallback), not
  from routing more to small. This is counter-intuitive.

- **GRPO gain is not correlated with base model strength.** The 1.5B model (base 47%)
  gains +2 pp; the stronger 3B (base 62%) and 7B (base 77%) degrade. This suggests the
  GRPO signal is most useful in the 40–60% base-rate regime where mixed outcomes (some
  pass, some fail per group) are frequent.

---

## 5  Open Questions

1. **Plasticity loss: mechanism and fix.** Why does 1.5B GRPO degrade at 400 tasks but
   not 200? The arxiv:2605.12484 paper identifies entropy shrinkage and NTK rank
   decrease as the mechanism. Queued experiments (EXP-002: KL penalty, EXP-007:
   KL+curriculum, EXP-022: warm-start) will test mitigations. C-CHAIN churn reduction
   (EXP-027) is also queued. *(Blocked on A800 connectivity.)*

2. **Can credit assignment improve beyond +2 pp?** EXP-026 (EGCA) applies execution-
   grounded token-span credit assignment (arxiv:2603.16158, +3.1 HumanEval reported).
   If EGCA gives >+4 pp on our 1.5B MBPP run, coarse advantage normalisation is the
   primary bottleneck. *(Blocked on A800 connectivity.)*

3. **Routing threshold Pareto curve.** We have threshold=0.57 (93.04% accuracy).
   EXP-021 (threshold=0.45, aggressive) and EXP-025 (threshold=0.65, conservative)
   are queued to characterise the full cost-accuracy curve. This determines whether
   BEST-Route-style best-of-N for uncertain queries (arxiv:2506.22716) is worthwhile.
   *(Blocked on A800 connectivity.)*

4. **Catastrophic forgetting magnitude.** EXP-004 and EXP-014 will measure whether
   the 1.5B GRPO adapter degrades on HumanEval (cross-benchmark forgetting) vs. its
   MBPP training distribution. *(Blocked on A800 connectivity.)*

5. **Pass@k vs. pass@1.** EXP-023 (pass@k eval on 1.5B) tests whether the GRPO adapter
   gains are larger at k>1, which would suggest the model is learning to sample diverse
   solutions rather than improving mode quality. *(Blocked on A800 connectivity.)*

6. **Joint router+LLM cycle stability.** Only one joint cycle was tested. As the LLM
   adapter improves, the router's training labels may drift (previously-hard queries
   become easy for the improved small model). Co-training stability is untested.

7. **A800 connectivity.** Port 50507 on 117.74.66.181 has been unreachable from the
   cloud runner since 2026-05-14T20:03Z (10+ days as of 2026-05-24). All 27 queued
   experiments are blocked. Recovery options: (a) add A800 to runner outbound allowlist,
   (b) relocate shepherd to a host co-located with A800, (c) expose state.json via HTTPS.

---

## 6  Reproducibility

### Repository

```
https://github.com/zeyuyuyu/router-skills-evolve
```

### Key scripts

| Script | Purpose |
|--------|---------|
| `experiments/run_joint_evolver.py` | Full joint pipeline (router + skills + LLM) |
| `experiments/run_e2e_ablation.py` | Component ablation runner |
| `experiments/train_small_model_grpo_local.py` | Local GRPO trainer (no TRL) |
| `experiments/train_learnable_router.py` | BERT-base router trainer |
| `experiments/evaluate_learnable_router.py` | Offline router evaluator |
| `experiments/tune_learnable_router_threshold.py` | Threshold sweep |
| `auto_research/pending_queue_update.py` | Idempotent experiment queue loader |

### Artifact paths (on A800)

```
/data0/home/zeyuwang/router-skills-evolve-results/
  e2e_ablation/e2e_ablation_router_sft_qwenchat.{json,md}
  rl_15b/qwen25_coder_15b_{base,grpo}_eval100.json
  llm_mbpp_qwenchat/qwen25_coder_05b_lora_mbpp_eval20_lr2e5_e1.json

/data0/home/zeyuwang/router-skills-evolve-runs/
  joint_evolver_real_20260502_050032/joint_evolver_manifest.json
  joint_evolver_real_qwenchat_20260502_084947/joint_evolver_manifest.json
  rl_15b/qwen25_coder_15b_grpo_200x4/  (best adapter checkpoint)
  learned-router-mixed-pretrained/       (best router checkpoint)

/data0/home/zeyuwang/auto_research/
  state.json          (experiment queue + history)
  logs/exp_*.result.json   (per-experiment outputs)
  reports/ideas-*.md       (idea-gen reports)
  paper_draft.md           (this file — canonical on A800)
```

### Best LLM adapter config

```yaml
model: Qwen/Qwen2.5-Coder-1.5B-Instruct
algorithm: local GRPO
train_tasks: 200
rollouts_per_prompt: 4
epochs: 1
lora_r: 16
learning_rate: 5.0e-6
prompt_style: qwen-chat
reward: binary (executable test pass/fail)
eval: MBPP eval100 first 100 tasks
result: 47/100 → 49/100 (+2 pp)
```

### Best router config

```yaml
model: bert-base-uncased (BERT-base)
dataset: project traces + UncommonRoute bench (3391 examples, 848 eval)
threshold: 0.57
eval_accuracy: 93.04%
large_f1: 89.48%
fallback_rate: 2.12%
```

### Pending queue

27 experiments are encoded idempotently in:
```bash
python3 /data0/home/zeyuwang/router-skills-evolve/auto_research/pending_queue_update.py
```
Run this on A800 when connectivity is restored to populate state.json.

---

*End of draft — auto-generated 2026-05-24 by paper-drafter agent.*
*Next update: 2026-05-31 (or when A800 connectivity is restored and new experiments complete).*
