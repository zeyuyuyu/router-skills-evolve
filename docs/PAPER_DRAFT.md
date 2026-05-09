# Skill-Aware Model Routing and Self-Evolving LLM Systems

> Draft for discussion. Target venue style: EMNLP / Findings / workshop.

## Title

**Skill-Aware Model Routing: Toward Self-Evolving Cost-Quality Optimization for LLM Inference**

## Abstract

Large language model applications increasingly rely on heterogeneous model
pools, where small models are cheap but brittle and frontier models are strong
but expensive. Existing routing systems often use static heuristics or
black-box classifiers to select a model, but they rarely close the loop between
runtime outcomes, routing policy updates, and model improvement. We introduce a
prototype self-evolving routing framework that combines three signals:
(1) a lightweight SkillBook that tracks historical success rates by prompt
signature, (2) a learnable router trained from traces and weak supervision, and
(3) a model-evolution track that fine-tunes or reinforcement-trains smaller
models on hard examples. We construct an Evolver Dataset that unifies
executable code-generation tasks from HumanEval and MBPP, hard traces where a
large model succeeds after a small model fails, and router supervision from
UncommonRoute benchmarks. In offline experiments, the learnable router reaches
over 94% routing accuracy and substantially reduces estimated cost relative to
an always-large baseline. We further show that naive supervised fine-tuning of
a small code model does not reliably improve held-out pass rate, while a small
local GRPO-style reinforcement loop can produce a modest improvement over its
base model in a smoke setting. These results suggest that self-evolving LLM
systems need both robust routing data and careful model-improvement objectives,
and provide a reproducible starting point for studying joint router/model
evolution.

## 1. Introduction

Serving LLM applications increasingly involves choosing among many models with
different prices, latencies, context windows, and capability profiles. A simple
policy---always use the strongest available model---maximizes quality but is
often economically infeasible. Conversely, always using a cheap model causes
quality regressions on hard tasks. This creates a routing problem: for a given
prompt, select the cheapest model that is likely to satisfy the user request.

Existing routing approaches often rely on fixed heuristics, model-size
thresholds, or classifiers trained once on static labels. However, deployed
systems continuously observe new signals: which model was selected, whether a
fallback was needed, which tasks failed, how much the request cost, and which
prompt patterns are consistently easy or hard. This suggests that routing should
be treated as an evolving system rather than a static component.

We study this problem in the setting of code generation and LLM routing. Our
goal is not only to route prompts to small or large models, but also to use
routing traces to improve both the router and the small model. We ask:

1. Can runtime traces and weak supervision train a learnable router that
   improves cost-quality trade-offs over static routing?
2. Can hard examples discovered by the router improve the small model through
   supervised fine-tuning or reinforcement learning?
3. What dataset structure is needed to evaluate router evolution and model
   evolution jointly?

We make three contributions:

- **A joint evolver pipeline.** We implement an end-to-end runner that connects
  SkillBook statistics, learnable router training, threshold tuning, and small
  model training/evaluation.
- **A unified Evolver Dataset.** We build a dataset bundle with executable code
  tasks and router supervision, combining HumanEval, MBPP, hard traces, and
  UncommonRoute weak labels.
- **Empirical lessons.** We show that router learning is immediately useful,
  while model evolution requires more careful objectives: naive SFT can
  underperform the base model, whereas a small test-reward GRPO loop can
  slightly exceed base in a smoke setting.

## 2. Problem Setup

Let a request be a prompt \(x\). A routing policy \(\pi\) chooses an initial
model \(m \in \{m_s, m_l\}\), where \(m_s\) is a cheaper small model and \(m_l\)
is a stronger large model. If \(m_s\) fails, the system may fall back to
\(m_l\). For each request, the system observes:

- prompt text and optional metadata;
- selected model and fallback chain;
- success or failure under a task-specific verifier;
- token/cost estimates;
- a cluster or skill signature.

The objective is to minimize expected cost while keeping task success above a
target threshold. In offline evaluation, we approximate success with executable
tests for code-generation tasks and approximate routing quality with labels
derived from known tiers or observed small/large outcomes.

## 3. System Overview

Our evolver has three interacting tracks.

### 3.1 SkillBook statistics

The SkillBook maps prompts to coarse signatures such as `M|list/num` or
`L|crypto/str`. For each signature and model, it stores success counts and
estimates whether a small model can safely handle similar prompts. This provides
a transparent online memory over historical traces.

### 3.2 Learnable router

The learnable router is a BERT-style binary classifier that predicts whether a
prompt should be routed to a small or large model. It is trained on:

- real traces, where small success implies a `small` label and small failure
  followed by large success implies a `large` label;
- weak tier labels from UncommonRoute, where `SIMPLE/MEDIUM` map to `small` and
  `COMPLEX/REASONING` map to `large`.

The router outputs \(p(\text{large} \mid x)\). A threshold controls the
cost-quality trade-off: lower thresholds route more prompts to the large model,
reducing fallback risk but increasing cost.

### 3.3 Model evolution

The model-evolution track trains small code models using examples where the
small model is weak. We explored:

- supervised fine-tuning (SFT) on reference solutions;
- DPO data construction from reference solutions and base-model failures;
- a local GRPO/REINFORCE-style loop where executable tests provide binary
  rewards.

## 4. Evolver Dataset

We introduce a dataset bundle built by:

```bash
python3 experiments/build_evolver_dataset.py \
  --output-dir data/evolver_dataset \
  --mbpp-splits train validation test \
  --uncommonroute-files bench/data/all.jsonl
```

The bundle contains two task families.

### 4.1 Code tasks

Code tasks contain prompts, reference solutions, entry points, and executable
tests. They are used for SFT, DPO, GRPO, and held-out evaluation.

Sources:

| Source | Count | Tests | References |
| --- | ---: | --- | --- |
| HumanEval | 164 | yes | yes |
| Hard traces | 6 | yes | yes |
| MBPP | 420 | yes | yes |
| **Total** | **590** | **yes** | **yes** |

Split:

| Split | Count |
| --- | ---: |
| train | 472 |
| dev | 59 |
| test | 59 |

### 4.2 Router tasks

Router tasks contain prompts and small/large weak labels. They are used for
router training and threshold analysis.

| Source | Count |
| --- | ---: |
| UncommonRoute bench | 3,328 |

Label distribution:

| Label | Count |
| --- | ---: |
| small | 2,258 |
| large | 1,070 |

Split:

| Split | Count |
| --- | ---: |
| train | 2,662 |
| dev | 332 |
| test | 334 |

## 5. Experiments

### 5.1 Router learning

We train a tiny BERT router on mixed trace and UncommonRoute data and evaluate
offline routing decisions. We also sweep the large-routing threshold.

Representative result:

| Setting | Accuracy | Large F1 | Fallback Rate | Cost vs Always-Large |
| --- | ---: | ---: | ---: | ---: |
| A800 pretrained tiny-BERT, tuned threshold | 94.49% | 91.52% | 1.98% | 36.48% |
| Joint runner, threshold 0.53 | 92.36% | 88.63% | 1.98% | 38.57% |

These results show that router learning produces a strong cost-quality tradeoff
even with weak labels.

### 5.2 Naive SFT for model evolution

We evaluate Qwen2.5-Coder-0.5B-Instruct on an MBPP held-out slice and compare
LoRA SFT adapters.

| Training recipe | Eval pass rate |
| --- | ---: |
| Base, qwen-chat prompt | 10/20 |
| Naive LoRA SFT | 9/20 |
| Conservative LoRA SFT | 9/20 |
| 400-sample MBPP SFT | 9/20 |

SFT improved formatting relative to earlier prompts, but did not exceed the
base model. This suggests that small-data imitation can overfit or disrupt
instruction-tuned behavior.

### 5.3 Local GRPO-style RL

We implement a lightweight local GRPO/REINFORCE trainer that samples multiple
completions per prompt, runs executable tests, and updates LoRA adapters with
group-relative rewards.

On an 8x4090 server:

| Setting | Eval pass rate |
| --- | ---: |
| Base Qwen2.5-Coder-0.5B-Instruct | 8/20 |
| GRPO, 40 train tasks, 4 rollouts | 9/20 |
| GRPO, 120 train tasks, 4 rollouts | 9/20 |

This is a small but positive signal: RL with executable rewards can exceed the
measured base in a smoke setting. However, the margin is not yet large enough
for a strong claim.

## 6. Discussion

The experiments reveal an asymmetry between router evolution and model
evolution.

Router evolution is data-efficient: weak labels and traces are enough to learn
useful cost-quality boundaries. Model evolution is harder: SFT on limited code
tasks can fail to improve over an instruction-tuned base. RL with test rewards
is more promising, but requires stronger optimization recipes, larger rollouts,
and stability mechanisms such as KL regularization.

For a stronger paper, the next experiments should:

- expand code tasks with MBPP+, HumanEval+, APPS, or CodeContests;
- evaluate on larger held-out sets rather than 20-task smoke slices;
- compare SFT, DPO, and GRPO under matched budgets;
- report pass@1/pass@k, routing cost, fallback rate, and end-to-end success;
- include ablations over threshold tuning, SkillBook features, and trace
  supervision.

## 7. Limitations

This draft reports prototype-scale experiments. Several numbers are based on
small held-out slices and should be treated as smoke-test results. The
UncommonRoute labels are weak supervision rather than direct success/failure
labels. The current local GRPO implementation is intentionally lightweight and
lacks a reference-model KL penalty. Larger and more controlled experiments are
required before making strong claims.

## 8. Reproducibility

Key scripts:

```text
experiments/build_evolver_dataset.py
experiments/run_joint_evolver.py
experiments/train_learnable_router.py
experiments/tune_learnable_router_threshold.py
experiments/train_small_model.py
experiments/build_mbpp_dpo_data.py
experiments/train_small_model_grpo_local.py
experiments/evaluate_finetuned_model.py
```

Dataset artifacts:

```text
/data/djh/router-skills/datasets/evolver_dataset_v1
```

Selected result summaries:

```text
results/datasets/evolver_dataset_v1_stats.json
results/joint_evolver_a800_20260502_summary.json
results/llm_sft_qwenchat_a800_20260502_summary.json
```

## 9. Ethics and Broader Impact

The system is designed to reduce inference cost and energy by routing easier
requests to smaller models, while preserving quality through fallbacks. However,
routing errors can cause users to receive lower-quality outputs, and cost
optimization can conflict with reliability. Any deployment should include
monitoring, fallback policies, user-visible quality safeguards, and careful
evaluation across languages and task types.

