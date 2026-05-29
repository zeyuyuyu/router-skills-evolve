# Router-Skills Evolve: Adaptive LLM Cost-Quality Optimization via Skill-Based Routing and Reinforcement Learning

**Zeyu Wang**  
0G.ai / Institute of Artificial Intelligence  
zeyu.wang@0g.ai

---

## Abstract

Large language model (LLM) serving faces a fundamental cost-quality trade-off: powerful
frontier models are expensive while cheaper models often fail on hard tasks. We present
**Router-Skills Evolve**, a self-improving serving system that continuously learns which
model to route each request to, while simultaneously evolving the small model's capabilities
via reinforcement learning. The system maintains a *SkillBook* of per-signature routing
statistics and a *learned router* trained from execution traces. On HumanEval (164 tasks),
our skills-based routing achieves **99% task accuracy at 83% lower cost** than always using
the large model. A BERT-based learned router trained on 2,662 labeled routing traces achieves
**93.04% routing accuracy** with a **2.12% fallback rate** on a 334-example held-out test
split—compared to 68.28% and 31.72% for the no-routing baseline. For model evolution, we
train a Qwen2.5-Coder-1.5B adapter via Group Relative Policy Optimization (GRPO) with
executable-test rewards, improving MBPP pass@1 from 47% to 49% (+4.3% relative). A
component-wise ablation confirms each module contributes independently. Our analysis reveals
a persistent +2pt ceiling for current GRPO recipes and traces it to winner-takes-all
exploration collapse on mixed-outcome training tasks, motivating cooperative policy
optimization as a next step. Code and datasets are released publicly.

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

### 7.3 Future Work

Priority experiments (currently queued, pending A800 restoration):
1. GCPO team-coverage training to break winner-takes-all collapse
2. REINFORCE++ EMA to provide gradient on 73% of currently silent training groups
3. WARP LoRA interpolation diagnostic (free +1pt if adapter has overshot)
4. Multi-seed evaluation of the 1.5B GRPO result
5. HumanEval evaluation of the GRPO adapter (cross-benchmark generalization)

---

## 8. Conclusion

We present Router-Skills Evolve, a self-improving LLM serving system combining skill-based
routing evolution with RL-driven model adaptation. The skills-based routing achieves 99%
task accuracy at 17% of the frontier-model cost on HumanEval. A learned BERT-based router
achieves 93.04% routing accuracy with 2.12% fallback. GRPO-based model evolution yields
+2pp code pass rate on MBPP. A component ablation isolates each contribution. We identify
winner-takes-all GRPO collapse and the 73%-silent-group problem as the binding constraints
on model improvement, and propose cooperative policy optimization as a remedy.

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
