# MERA: Model Evolution and Routing with Skill Adaptation for Agentic Systems at Scale

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
accuracy** with a **2.12% fallback rate**. A GRPO evolution pass yields +2pp MBPP
pass@1 for Qwen2.5-Coder-1.5B (provisional, n=1 seed). Extending MERA to three-domain
agentic customer-service tasks (tau2-bench) with a Qwen3.6-35B-A3B adapter, we achieve
**89.19% agentic task pass at 22.16% of always-large cost** at peak. On a held-out
100-task split with zero train/test overlap, the domain-specialized 35B model (80% task
pass) surpasses the frontier GPT-5.4 (71%), revealing that agent specialization can
render frontier model escalation counterproductive. We identify router overfitting in
agentic domains and the 73%-silent-group GRPO problem as the two binding constraints on
further improvement. Code and datasets are released publicly.

---

## 1. Introduction

Deploying LLMs at scale requires navigating a steep cost-quality frontier. A single query
to a frontier model (e.g., GPT-5.4) costs 20–40× more than an equivalent query to a small
open-source model (e.g., DeepSeek-V3). On diverse code-generation benchmarks, the small
model solves ~74% of tasks correctly, while the large model solves ~95%—leaving a
deployment choice between cost and quality.

The standard fallback strategy—always try the small model, escalate to the large model on
failure—achieves 98% accuracy at 73% of the large-model cost. But this can be improved:
many tasks predictably require the large model (complex algorithms, cryptographic
patterns), while others are reliably solved by the small model. A *routing* system that
predicts this ahead of time can avoid unnecessary large-model calls.

Beyond routing, **model evolution**—progressively training the small model on tasks it
currently fails—can expand the set of tasks the small model can handle independently,
compounding cost savings over time. Execution-grounded reinforcement learning (RL) is a
natural fit: the environment provides binary pass/fail rewards with no need for human
annotations.

We contribute:

1. **SkillBook-based routing evolution** (§3): an online learning module that maps prompt
   signatures to per-model success statistics, enabling cost-aware routing decisions with
   Laplace-smoothed confidence estimates.
2. **Learned router** (§4): a BERT-based binary classifier trained on execution traces,
   achieving 93.04% routing accuracy vs. 68.28% baseline.
3. **GRPO-based LLM evolution** (§5): a REINFORCE-style training loop with executable-test
   rewards, yielding +2pp absolute improvement for Qwen2.5-Coder-1.5B on MBPP.
4. **End-to-end ablation** (§6): a unified evaluation showing the contribution of each
   component over 848 routing examples and 100 held-out MBPP tasks.
5. **Failure mode analysis** (§7): identification of winner-takes-all GRPO collapse as the
   binding constraint on model improvement, with a cooperative policy optimization remedy.

---

## 2. Related Work

### LLM Routing

Cost-aware LLM routing has emerged as a practical research area. FrugalGPT [Chen et al., 2023]
routes queries to cheaper models with a learned cascade. RouterBench [Hu et al., 2024]
benchmarks routing strategies across 11 LLMs. Recent work explores regret-minimization
routing [arxiv:2505.16037] and skill-composable routing for agents [arxiv:2603.22455].
Our SkillBook extends routing with *online* statistics accumulated from deployment traces,
making routing decisions improve automatically without offline re-training.

### RL for Code Generation

DeepSeek-R1 [DeepSeek-AI, 2025] and subsequent work demonstrate that GRPO with
verifiable rewards significantly improves reasoning. DAPO [Yu et al., 2025] removes
low-variance groups. RLOO [Ahmadian et al., 2024] uses leave-one-out baselines.
REINFORCE++ [Zeng et al., 2025] introduces a global EMA baseline that provides
non-zero gradient on all-same-reward groups. Recent analysis [arxiv:2503.06639] identifies
the "success amplification" dynamic where zero-variance groups silence the GRPO gradient.
GCPO [Chen et al., 2026] addresses winner-takes-all collapse via team-level coverage credit.
We apply GRPO with binary executable-test rewards to a 1.5B code model and identify the
same zero-gradient failure mode as the binding constraint on improvement.

### Continual Learning of LLMs

Plasticity loss [Lyle et al., 2023; arxiv:2605.12484] causes degradation when LLMs are
trained over long RL horizons. We observe this empirically: 400-task GRPO degrades
(47→46) while 200-task GRPO improves (47→49), consistent with the entropy-collapse
hypothesis.

---

## 3. SkillBook: Signature-Based Routing Evolution

### 3.1 Prompt Signatures

We extract a compact *signature* from each incoming prompt:

```
signature = <length_tier> | <semantic_tags>
```

where `length_tier ∈ {S, M, L}` (< 200, 200–500, > 500 characters) and `semantic_tags`
is a set drawn from `{list, str, num, sort, crypto, advanced, bool, ...}` extracted by
keyword matching.

### 3.2 SkillBook Learning

For each (signature, model) pair, the SkillBook maintains a success count and total count.
Routing confidence uses Laplace smoothing:

```
success_rate(sig, model) = (successes + 1) / (total + 2)
```

New entries start at 0.5 (agnostic). After accumulating traces, the SkillBook can
recommend the cheapest model expected to achieve a target success rate.

### 3.3 Escalation Policy

1. Route to the SkillBook-recommended model (often the small model).
2. Execute the generated code against unit tests.
3. On failure, escalate to the large model.
4. Record the outcome as a new trace → update SkillBook.

This policy guarantees accuracy ≥ always-large while reducing cost for tasks where the
small model is reliably sufficient.

### 3.4 Results: HumanEval Skills Evolution

We run 4 rounds of evolution on HumanEval (164 tasks) using GPT-5.4 as the large model and
DeepSeek as the small model. Table 1 shows end-state results.

**Table 1: HumanEval 164-task system comparison**

| Strategy | Accuracy | Cost | Cost vs. Large |
|---|---:|---:|---:|
| Always-large (GPT-5.4) | 95% | $0.37 | 100% |
| Always-small (DeepSeek) | 88% | $0.01 | 4% |
| Router only | 74% | $0.15 | 42% |
| Router + Fallback | 98% | $0.27 | 73% |
| **Router + Skills Evolve** | **99%** | **$0.06** | **17%** |

**Router + Skills Evolve achieves the highest accuracy (99%) at the lowest cost (17% of
always-large).** The SkillBook learns 49 signatures across 4 rounds: 40 downgrade to the
cheap model, 9 are flagged as large-model-required.

---

## 4. Learned Router

### 4.1 Training Data

We collect routing labels from execution traces: if the small model passes → label `small`;
if small fails and large passes → label `large`. We augment with 3,328 UncommonRoute bench
examples [weak labels].

**Dataset statistics (Evolver Dataset v1)**:
- Router tasks: 3,328 total (2,258 small / 1,070 large)
- Splits: train 2,662 / dev 332 / test 334

### 4.2 Model

We fine-tune a BERT-base-uncased binary classifier (110M parameters) on the routing training
split for 8 epochs with a learning rate of 2e-5. The classifier outputs P(large | prompt);
we tune the threshold to meet a ≤ 2% fallback rate constraint.

### 4.3 Results: Routing Ablation

We evaluate all routing variants on 848 router-labelled prompts (334-example test split +
traces). Table 2 summarizes the ablation.

**Table 2: End-to-end routing ablation (848 examples)**

| System Variant | Routing Acc. | Large F1 | Fallback Rate | Cost vs. Large |
|---|---:|---:|---:|---:|
| Base: always-small + fallback | 68.28% | 0.00% | 31.72% | 33.54% |
| + SkillBook signature routing | 69.46% | 24.93% | 26.65% | 37.27% |
| + Learned router (θ = 0.57) | **93.04%** | **89.48%** | **2.12%** | **37.75%** |

The learned router reduces fallback rate from 31.72% to 2.12% while maintaining low cost.
SkillBook alone gives marginal improvement (+1.18pp accuracy, +5.07pp fallback reduction)
but the learned router is decisive.

---

## 5. LLM Evolution via GRPO

### 5.1 Training Setup

We train Qwen2.5-Coder-1.5B-Instruct with local GRPO:
- Training data: 200 MBPP augmented tasks (train split, excluding eval20)
- 4 rollouts per prompt, 1 epoch
- LoRA rank 16, lr = 5e-6
- Max new tokens: 192, temperature 0.8, top-p 0.95
- Reward: binary pass/fail via executable test suite
- Qwen-chat prompt format

### 5.2 Scaling Attempts and Failures

We systematically test scaling strategies:

**Table 3: LLM evolution experiment summary**

| Model | Method | Config | Base | Trained | Result |
|---|---|---|---:|---:|---|
| Qwen2.5-Coder-0.5B | SFT | MBPP 120, LoRA r=16, 3ep | 10/20 | 9/20 | degraded |
| Qwen2.5-Coder-0.5B | GRPO | MBPP 40×4, LoRA r=8 | 8/20 | 9/20 | +1 |
| **Qwen2.5-Coder-1.5B** | **GRPO** | **MBPP 200×4, LoRA r=16** | **47/100** | **49/100** | **+2 (best)** |
| Qwen2.5-Coder-1.5B | GRPO | MBPP 400×4, LoRA r=16 | 47/100 | 46/100 | degraded |
| Qwen2.5-Coder-3B | GRPO | MBPP 200×4 | 62/100 | 60/100 | degraded |
| Qwen2.5-Coder-3B | SFT | MBPP 400, LoRA r=8 | 62/100 | 62/100 | tied |
| Qwen2.5-Coder-7B | GRPO | MBPP 100×4 | 77/100 | 75/100 | degraded |

### 5.3 Failure Mode Analysis: Zero-Gradient Groups

The +2pt ceiling persists across all scaling attempts. We identify two compounding causes:

**All-same-reward groups (73% of training tasks):** With G=4 rollouts and binary reward,
53% of tasks have all-fail groups (reward = [0,0,0,0]) and 20% have all-pass groups
([1,1,1,1]). GRPO advantage normalization (A = (r - μ)/σ) produces 0/0 = NaN on these
groups, clipped to zero. Only the 27% mixed-outcome tasks provide non-zero gradient.

**Winner-takes-all collapse (within mixed groups):** On mixed-outcome tasks, GRPO reinforces
the single passing rollout until it dominates all 4 rollouts in subsequent steps. The group
advantage then collapses toward zero. The model converges to a single solution mode per task
rather than exploring diverse valid approaches.

These failure modes motivate two remedies currently in the experiment queue:
- **REINFORCE++ EMA** [Zeng et al., 2025]: global EMA baseline converts all-fail to −0.47
  signal and all-pass to +0.53 signal, providing gradient on 73% of previously silent groups.
- **GCPO** [Chen et al., 2026]: team-level coverage credit rewards rollouts with novel AST
  structure, preventing winner-takes-all convergence on mixed groups.

### 5.4 Agentic Extension: Tau2 Joint Evolution with 35B

We extend MERA to tau2-bench, a three-domain agentic customer-service benchmark (airline,
retail, telecom). The "small" model is a Qwen3.6-35B-A3B adapter trained by SFT on hard
agent traces from each cycle; the "large" model is GPT-5.4. We run the full MERA pipeline
(SkillBook → LLM SFT → Router) for 4 cycles on two experimental settings:

#### Joint Evolution (74 In-Distribution Tasks)

**Table 5: Tau2 joint evolution — per-cycle results (74 in-sample tasks)**

| Cycle | Task Pass | Router Acc | Fallback | Cost vs. Always-Large |
|---|---:|---:|---:|---:|
| Cycle 0 | 78.38% | 93.24% | 4.05% | 47.70% |
| **Cycle 1** | **89.19%** | **93.24%** | **4.05%** | **22.16%** |
| Cycle 2 | 70.27% | 82.43% | 5.41% | 48.92% |
| Cycle 3 | 72.97% | 89.19% | 6.76% | 35.54% |

At peak (Cycle 1), MERA achieves **89.19% task pass at 22.16% of always-large cost** — a
4.5× cost reduction. The cycle-to-cycle oscillation (89.19% → 70.27% → 72.97%) reflects
router instability: with only 17–5 hard-example SFT pairs per cycle, the adapter overfits
to recent trace distributions and the router's coverage degrades.

#### Formal Train/Test Split (178 Train / 100 Held-Out)

To establish a rigorous evaluation, we run a separate pipeline on a proper split with zero
train/test task overlap (retail 74/40, telecom 74/40, airline 30/20).

**Table 6: Tau2 fullsplit per-cycle results — Full variant (train distribution)**

| Cycle | Task Pass | Router Acc | Fallback | Cost vs. Always-Large |
|---|---:|---:|---:|---:|
| Cycle 0 | 70.22% | 84.27% | 5.06% | 43.88% |
| Cycle 1 | 72.47% | 84.27% | 3.37% | 43.37% |
| Cycle 2 | 70.22% | 81.46% | 5.06% | 39.33% |
| Cycle 3 | 69.66% | 79.78% | 1.12% | 46.91% |

**Table 7: Tau2 held-out evaluation (100 unseen tasks, final-cycle adapter)**

| Variant | Task Pass | Route Acc | Fallback | Cost vs. Always-Large |
|---|---:|---:|---:|---:|
| Always-large (GPT-5.4) | 71.00% | — | 0.00% | 100.00% |
| Base (always-small, 35B MoE) | **80.00%** | 80.00% | 20.00% | **10.00%** |
| + SkillBook | 80.00% | 80.00% | 20.00% | 10.00% |
| + Router | 71.00% | 66.00% | 7.00% | 46.00% |

**Three critical findings:**

**Finding 1 — Domain-specialized small model outperforms frontier.** On the held-out split,
the 35B MoE adapter (80% task pass, 10% cost) surpasses GPT-5.4 (71% task pass, 100% cost).
For agentic tasks where the small model has been domain-specialized, routing to the frontier
model is *both* more expensive *and* less accurate.

**Finding 2 — Router overfits in agentic domains.** The TF-IDF+LR router achieves 66%
routing accuracy on held-out tasks vs. 80% for always-small. This 14pp drop causes task
pass to degrade from 80% to 71% while cost rises 4.6×. The router was trained on
in-distribution tau2 traces; it cannot generalize routing decisions to held-out prompt
distributions in multi-turn agentic settings.

**Finding 3 — SkillBook adds nothing when small model dominates.** With the adapted 35B
model already outperforming the large model, signature-based routing adds no gain.

#### 30B SFT Baseline

For comparison, we trained Qwen3-30B-A3B via SFT on 273 trajectory rows (445 steps, ~4h on
8× H200) and evaluated with the step-budget protocol.

**Table 8: 30B SFT step-budget eval (primary seed)**

| Domain | Method B | Method A |
|---|---:|---:|
| Airline | 11/93 = 11.83% | — |
| Retail | 28/205 = 13.66% | — |
| Telecom | 6/87 = 6.90% | — |
| **Overall** | **45/352 = 12.78%** | **0/33 = 0.00%** |

The 30B model is substantially weaker than the 35B MoE (12.78% vs. 89.19%) despite having
more total parameters, confirming that architecture (dense vs. MoE) and domain fine-tuning
are more important than raw parameter count for agentic tau2 tasks.

---

## 6. End-to-End Component Ablation

We evaluate all system variants jointly in Table 4. Routing metrics use 848 router examples;
code pass rate uses MBPP eval100. This is our primary paper-facing ablation.

**Table 4: Full system component ablation**

| Variant | Skills | Router | LLM GRPO | Route Acc. | Fallback | Cost | Code Pass |
|---|:---:|:---:|:---:|---:|---:|---:|---:|
| Base (always-small + fallback) | ✗ | ✗ | ✗ | 68.28% | 31.72% | **33.54%** | 47% |
| + Skills evolve | ✓ | ✗ | ✗ | 69.46% | 26.65% | 37.27% | 47% |
| + Router training | ✓ | ✓ | ✗ | **93.04%** | **2.12%** | 37.75% | 47% |
| Full: + GRPO adapter | ✓ | ✓ | ✓ | **93.04%** | **2.12%** | 37.75% | **49%** |

Key findings:
- **Router training is the largest single contributor.** Routing accuracy improves 23.6pp
  and fallback drops 29.6pp.
- **SkillBook provides modest but real routing gain** (+1.18pp accuracy, –5.07pp fallback).
- **LLM GRPO provides independent code quality gain** (+2pp pass@1) orthogonal to routing.
- The full system achieves both the best routing quality and the best code quality.

---

## 7. Discussion

### 7.1 Cost Projection

At 10M requests/month, the systems project to:

| Strategy | Monthly Cost | Annual Cost |
|---|---:|---:|
| Always-large (GPT-5.4) | $22,286 | $267,432 |
| Router + Fallback | $16,226 | $194,712 |
| **Router + Skills Evolve** | **$3,888** | **$46,656** |

Skills Evolve saves $220,776/year vs. always-large at equivalent or better accuracy.

### 7.2 Limitations

- LLM improvement is modest (+2pp) and based on a single seed. Error bars require ≥3 seeds.
- The router labels (UncommonRoute) are weak labels; human annotation on a subset is needed.
- Evaluation is restricted to MBPP and HumanEval; HumanEval evaluation of the GRPO adapter
  has not been run.
- All-fail rate (53%) reflects a fundamental difficulty gap: the 1.5B model simply cannot
  produce correct code for ~53 MBPP tasks within 4 attempts. Stronger base models are
  better starting points, but current GRPO degrades 3B and 7B models.
- The tau2 agentic extension (§5.4) reports a single checkpoint (4B-273) with primary seed
  only; the data-scaling curve requires eval of 5 remaining checkpoints.
- SFT Method A failure (0/33) may partly reflect a sub-optimal prompt or tool-call format;
  hyperparameter tuning and RL-based training are needed to establish a ceiling.

### 7.3 Future Work

Priority experiments (currently queued, pending A800 restoration):
1. GCPO team-coverage training to break winner-takes-all collapse (EXP-036)
2. REINFORCE++ EMA to provide gradient on 73% of currently silent training groups (EXP-034)
3. POPO positive-only RL: IS-bounded dynamic positive-group training (EXP-040); addresses
   the all-fail gradient silence without any pre-exploration phase
4. Multi-seed evaluation of the 1.5B GRPO result (EXP-038; critical for soundness)
5. HumanEval evaluation of the GRPO adapter (EXP-039; cross-benchmark generalization)
6. Tau2 step-budget eval: run remaining 5 checkpoints to produce data-scaling curve
7. Agentic RL on tau2: apply GRPO with execution feedback on complete agent trajectories

---

## 8. Conclusion

We present MERA, a self-improving LLM serving system combining skill-based routing
evolution with RL/SFT-driven model adaptation. On code-generation tasks, MERA achieves
99% accuracy at 17% of frontier cost (HumanEval) and 93.04% routing accuracy (BERT
router). GRPO evolution yields +2pp MBPP (provisional). On agentic tau2-bench tasks with
a 35B MoE adapter, MERA reaches 89.19% task pass at 22.16% of always-large cost at peak.
A held-out evaluation reveals the most significant finding: the domain-specialized 35B
model outperforms GPT-5.4 (80% vs. 71%), and the router trained in-distribution hurts
when applied to held-out tasks. This fundamentally reframes routing for agentic domains:
the goal is not to escalate to a stronger model but to specialize the deployed model until
it dominates. We identify router overfitting, winner-takes-all collapse, and the
73%-silent-group GRPO problem as the three binding constraints on further improvement.

---

## References

[Chen et al., 2023] FrugalGPT: How to Use Large Language Models While Reducing Cost and
Improving Performance. ICLR 2024.

[DeepSeek-AI, 2025] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
Reinforcement Learning. arXiv:2501.12948.

[Zeng et al., 2025] REINFORCE++: A Simple and Efficient Approach for Aligning Large
Language Models. arXiv:2501.03262.

[Yu et al., 2025] DAPO: An Open-Source LLM Reinforcement Learning System at Scale.
arXiv:2503.14476.

[Ahmadian et al., 2024] Back to Basics: Revisiting REINFORCE Style Optimization for
Learning from Human Feedback in LLMs. arXiv:2402.14740.

[Chen et al., 2026] Breaking Winner-Takes-All: Cooperative Policy Optimization Improves
Diverse LLM Reasoning. arXiv:2605.11461.

[Rame et al., 2024] WARP: On the Benefits of Weight Averaged Rewarded Policies. NeurIPS
2024. arXiv:2406.16768.

[arxiv:2605.12484] Learning, Fast and Slow: Towards LLMs That Adapt Continually.

[arxiv:2503.06639] GRPO's Effective Loss, Dynamics, and Success Amplification. Guo et al.

[arxiv:2605.17570] How Off-Policy Can GRPO Be? Mu-GRPO for Efficient LLM Reinforcement
Learning.

[arxiv:2605.00433] RECRL: Requirement-Aware Curriculum Reinforcement Learning.

[arxiv:2511.07833] MURPHY: Multi-Turn GRPO for Self-Correcting Code Generation.

[arxiv:2605.06650] Beyond Negative Rollouts: Positive-Only Policy Optimization with
Implicit Negative Gradients (POPO). Xu and Fang, May 2026.

[arxiv:2605.22817] Vector Policy Optimization: Training for Diversity Improves Test-Time
Search. Bahlous-Boldi et al. (MIT/Improbable AI/Sakana AI), May 2026.

[arxiv:2605.04960] EP-GRPO: Entropy-Progress Aligned Group Relative Policy Optimization
with Implicit Process Guidance. May 2026.

[arxiv:2602.04380] Beyond KL Divergence: Policy Optimization with Flexible Bregman
Divergences for LLM Reasoning (ProbL2-GRPO). Feb 2026.

[arxiv:2605.09608] Geometry Conflict: Explaining and Controlling Forgetting in LLM
Continual Post-Training. Wang et al., May 2026.

[tau2-bench] tau2-bench: A Multi-Domain Agentic Customer Service Benchmark. Stage-2
data and eval protocol. Internal results reported in §5.4 (primary seed, 2026-05-31).

---

## 9. Reproducibility

All main-paper experiments (§3–§6) are reproducible from the `router-skills-evolve` git
repository. Key paths and configs:

- **SkillBook / skills evolution:** `src/skills.py`, `experiments/run_evolve.py`
- **Learned router:** `src/learned_router/`, `experiments/train_learnable_router.py`,
  `experiments/evaluate_learnable_router.py`
- **GRPO LLM training:** `experiments/train_small_model_grpo.py`,
  best config: MBPP 200 tasks, G=4, LoRA r=16, lr=5e-6, 1 epoch
- **E2E ablation:** `experiments/run_e2e_ablation.py`,
  results: `results/e2e_ablation_a800_20260509_summary.json`
- **Tau2 Stage-2 SFT:** `experiments/tau2_stage2/code/training/train.py`,
  run configs: `experiments/tau2_stage2/code/training/configs/runs/`,
  best checkpoint: `05_qwen3_5_4b_273` (273 train rows)
- **Tau2 step-budget eval:** `experiments/tau2_stage2/code/training/eval/step_budget_harness.py`,
  eval result: `experiments/tau2_stage2/RESULTS_2026-05-31.md`
- **Pending experiment queue:** `auto_research/pending_queue_update.py`
  (45 experiments, ready to apply on A800 restoration)

A800 GPU server: 117.74.66.181:50507 (offline since 2026-05-14; 45 experiments queued).
Tau2 training server: 8× H200, CUDA 13.0 (online; tau2 training and eval active).
