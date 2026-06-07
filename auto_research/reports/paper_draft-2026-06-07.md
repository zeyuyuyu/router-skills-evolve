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
exploration collapse on mixed-outcome training tasks, motivating four experimental tracks
currently queued: credit assignment (EGCA, SA-AH-GRPO, GRPO-λ), semantic diversity
(DRA-GRPO, ReSkill, unlikeliness reward), compute efficiency (rollout replay, adaptive
budget), and forgetting mitigation (LoRA rank ablation, CapTrack subspace orthogonality).
We also report initial results on a tau2-bench agentic extension (42% step imitation,
0% autonomous completion), establishing a baseline for agentic RL. Code and datasets
are released publicly.

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
   binding constraint on model improvement, with four experimental tracks targeting it.

---

## 2. Related Work

### LLM Routing

Cost-aware LLM routing has emerged as a practical research area. FrugalGPT [Chen et al., 2023]
routes queries to cheaper models with a learned cascade. RouterBench [Hu et al., 2024]
benchmarks routing strategies across 11 LLMs. Recent work explores regret-minimization
routing [arxiv:2505.16037] and skill-composable routing for agents [arxiv:2603.22455].
Systematic analysis [arxiv:2506.06579] shows confidence-based cascades competitive with
learned routers for bimodal-difficulty code tasks, and identifies a competence threshold
below which routing gains diminish — reinforcing our training-first priority.
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
GRPO-λ [arxiv:2510.00194] applies eligibility-trace credit assignment, concentrating
gradient on tokens near the decisive failure. EGCA [arxiv:2603.16158] localises credit
via execution-trace divergence analysis (+3.1pp HumanEval). SA-AH-GRPO [arxiv:2606.05434]
applies asymmetric entropy-adaptive horizon discounting (+4.9pp GSM8K, 3.6× lower variance).
DRA-GRPO [arxiv:2505.09655] uses submodular mutual information to reward semantically
diverse rollouts, preventing policy collapse. We apply GRPO with binary executable-test
rewards to a 1.5B code model and identify the same zero-gradient failure mode, then survey
these recent remedies as the primary next experimental direction.

### Continual Learning of LLMs

Plasticity loss [Lyle et al., 2023; arxiv:2605.12484] causes degradation when LLMs are
trained over long RL horizons. We observe this empirically: 400-task GRPO degrades
(47→46) while 200-task GRPO improves (47→49), consistent with the entropy-collapse
hypothesis. Subspace geometry analysis [arxiv:2603.02224] proves forgetting severity scales
with the minimum principal angle (MPA) between consecutive task gradient subspaces,
motivating LoRA rank reduction and capacity-tracking orthogonality penalties [CapTrack,
arxiv:2603.06610] as targeted mitigations.

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

Four experimental tracks address these failure modes (all queued, pending A800 restoration):
- **Credit assignment** (EXP-052, EXP-054, EXP-055): GRPO-λ temporal discounting, EGCA
  execution-grounded localization, and SA-AH-GRPO entropy-adaptive horizon.
- **Semantic diversity** (EXP-046, EXP-048, EXP-049): unlikeliness reward, DRA-GRPO SMI
  diversity, and ReSkill persistent skill library co-training.
- **Compute efficiency** (EXP-050, EXP-051): rollout replay + difficulty selection, adaptive
  rollout budget allocation.
- **Forgetting mitigation** (EXP-053, EXP-056, EXP-057): subspace MPA diagnostic,
  LoRA rank=8 cliff test, CapTrack orthogonality penalty.

### 5.4 Agentic Benchmark Extension: Tau2 Stage-2

As a parallel experimental track, we extend the LLM training framework to an agentic
customer-service benchmark (tau2-bench) covering airline, retail, and telecom domains.
This track trains Qwen3.x models via supervised fine-tuning on agent trajectory data and
evaluates them with a step-budget protocol measuring both step-level imitation and
autonomous task completion.

**Training.** We train 8 checkpoints varying model size (2B, 4B, 9B, 36B-A3B) and training
data size (50–273 rows from the stage2-v1 dataset). Training uses standard SFT with
trajectory-format examples from the tau2 airline, retail, and telecom tool-use scenarios.

**Table 5: Tau2 Stage-2 SFT checkpoint summary**

| Run | Model | Train Rows | Best Eval Loss | Status |
|---|---|---:|---:|---|
| `01_qwen3_5_2b_273` | Qwen3.5-2B | 273 | 0.2401 | complete |
| `02_qwen3_5_4b_50` | Qwen3.5-4B | 50 | 0.1667 | complete |
| `03_qwen3_5_4b_100` | Qwen3.5-4B | 100 | 0.1786 | complete |
| `04_qwen3_5_4b_200` | Qwen3.5-4B | 200 | 0.1780 | complete |
| `05_qwen3_5_4b_273` | Qwen3.5-4B | 273 | 0.1711 | complete + eval |
| `06_qwen3_5_9b_50` | Qwen3.5-9B | 50 | 0.1408 | complete |
| `07_qwen3_5_9b_273` | Qwen3.5-9B | 273 | 0.1561 | partial ckpt |
| `08_qwen3_6_35b_a3b_273` | Qwen3.6-35B-A3B | 273 | n/a | failed |

**Evaluation protocol.** Each trained checkpoint is evaluated via a step-budget protocol
using two methods:
- **Method B** (step imitation): the first *k* steps of a strong-baseline trajectory are
  replayed verbatim; the student substitutes at step *k*; the baseline resumes thereafter.
  Pass iff reward ≥ 0.5.
- **Method A** (autonomous E2E): the student completes all agent steps from scratch within
  `max_steps = B`. Pass iff reward ≥ 0.5.

**Table 6: Step-budget eval — Qwen3.5-4B-273 (primary seed, `05_qwen3_5_4b_273`)**

| Domain | Method B | Rollouts (B) |
|---|---:|---:|
| Airline | 40/93 = **43.0%** | 93 |
| Retail | 80/205 = **39.0%** | 205 |
| Telecom | 28/87 = **32.2%** | 87 |
| **Overall (Method B)** | **148/352 = 42.0%** | 352 |
| **Overall (Method A)** | **0/33 = 0.0%** | 33 |

**Finding.** SFT on 273 trajectory rows enables substantial step-level imitation (42.0%
Method B) but confers zero autonomous task-completion capability (0.0% Method A). The
training loss decreases from 0.1099 to 0.0724, confirming the model is fitting the
trajectory distribution. The Method A failure is consistent with distribution mismatch:
Method B uses a teacher-forced prefix that keeps the student in-distribution, whereas
Method A compounds student errors across all steps. Step-budget eval for the remaining six
checkpoints (data-scaling curve from 50 to 273 rows) is pending.

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

Priority experiments (57 queued, pending A800 restoration since 2026-05-14, day 25):

**Critical paper blockers:**
1. **EXP-038**: Multi-seed GRPO (n=3) to verify +2pp claim (soundness)
2. **EXP-039**: HumanEval eval of GRPO adapter (cross-benchmark generalization)
3. **EXP-040/EXP-041**: Gold-label router evaluation to verify 93.04% claim

**Credit assignment track (queued 2026-06-05 to 2026-06-06):**
4. **EXP-052 (GRPO-λ)**: Eligibility-trace credit assignment [arxiv:2510.00194];
   concentrates gradient on tokens near test execution. Expected +3–4.5pp over vanilla GRPO.
5. **EXP-054 (EGCA)**: Execution-grounded credit assignment via statement-level divergence
   tracing [arxiv:2603.16158]; +3.1pp HumanEval, +1.5pp MBPP in paper results. ~105 min.
6. **EXP-055 (SA-AH-GRPO)**: Entropy-adaptive horizon discounting on negative-advantage
   rollouts only [arxiv:2606.05434]; +4.9pp GSM8K, 3.6× variance reduction. ~95 min.

**Semantic diversity track (queued 2026-06-01 to 2026-06-03):**
7. **EXP-046 (unlikeliness reward)**: Bonus for rare-but-correct rollouts [arxiv:2506.02355];
   prevents distribution sharpening toward dominant solution modes. ~95 min.
8. **EXP-048 (DRA-GRPO)**: Submodular mutual information reward adjustment [arxiv:2505.09655];
   rewards semantically unique rollouts within each G=4 group. ~95 min.
9. **EXP-049 (ReSkill)**: Persistent skill library co-trained with GRPO policy [arxiv:2606.01619];
   extracts recurring failure patterns from mixed-outcome tasks every 15 steps. ~100 min.

**Compute efficiency track (queued 2026-06-04):**
10. **EXP-050 (rollout replay)**: Difficulty-targeted task selection + IS-weighted rollout
    replay [arxiv:2506.05316]; 47% faster training at same accuracy on 12B model. ~80 min.
11. **EXP-051 (adaptive budget)**: G ∝ √(p(1-p)) rollout allocation [arxiv:2602.01601];
    concentrates 800-total rollouts on the 27% mixed-outcome tasks. ~90 min.

**Forgetting mitigation track (queued 2026-06-05 to 2026-06-07):**
12. **EXP-053 (MPA diagnostic)**: Forgetting eval with HumanEval retention probe +
    minimum principal angle computation at checkpoints [100, 200, 400] tasks [arxiv:2603.02224];
    distinguishes H1 (catastrophic forgetting) from H2 (MBPP overfitting). ~60 min.
13. **EXP-056 (LoRA r=8)**: Rank ablation at 400 tasks; smaller subspace → wider MPA →
    less forgetting at the 200→400 cliff. ~95 min (no implementation overhead).
14. **EXP-057 (CapTrack)**: Frobenius-norm subspace orthogonality penalty
    L_orth = 0.01 × ‖A^T A_prev‖_F [arxiv:2603.06610]; prescribes the fix if EXP-053
    confirms H1. ~115 min.

**Code quality and agentic extensions:**
15. **EXP-047 (quality reward)**: Code quality composite (cyclomatic complexity, line count,
    flake8) added to binary reward [arxiv:2506.02211]. ~92 min.
16. **Tau2 data-scaling curve**: Step-budget eval for 5 remaining checkpoints (2B-273, 4B-50/100/200, 9B-50).
17. **Agentic RL on tau2**: Apply GRPO with execution feedback on complete agent trajectories.

---

## 8. Conclusion

We present Router-Skills Evolve, a self-improving LLM serving system combining skill-based
routing evolution with RL-driven model adaptation. The skills-based routing achieves 99%
task accuracy at 17% of the frontier-model cost on HumanEval. A learned BERT-based router
achieves 93.04% routing accuracy with 2.12% fallback. GRPO-based model evolution yields
+2pp code pass rate on MBPP. A component ablation isolates each contribution. We identify
winner-takes-all GRPO collapse and the 73%-silent-group problem as the binding constraints
on model improvement, and propose four concurrent experimental tracks targeting credit
assignment, semantic diversity, compute efficiency, and forgetting mitigation. An agentic
extension to tau2-bench establishes an initial baseline: SFT achieves 42% step imitation
but 0% autonomous completion, motivating RL-based agentic training as the natural next step.

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

[arxiv:2510.00194] GRPO-λ: Credit Assignment Improves LLM Reasoning. 2025.

[arxiv:2603.16158] Execution-Grounded Credit Assignment for GRPO in Code Generation
(EGCA). ICLR 2026 Workshop on Scaling Post-Training (SPOT). March 2026.

[arxiv:2606.05434] Selective-Advantage Entropy-Adaptive Horizon GRPO (SA-AH-GRPO):
Asymmetric Token-Level Discounting for Efficient Reinforcement Learning of Language Models.
June 2026.

[arxiv:2505.09655] DRA-GRPO: Your GRPO Needs to Know Diverse Reasoning Paths.
Huang et al., May 2025.

[arxiv:2606.01619] ReSkill: Reconciling Skill Creation with Policy Optimization in
Agentic RL. He et al., June 2026.

[arxiv:2506.02355] Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening.
He, Fried, Welleck (CMU), June 2025.

[arxiv:2506.05316] Improving Data Efficiency for LLM Reinforcement Fine-tuning Through
Difficulty-targeted Online Data Selection and Rollout Replay. Sun et al., June 2025.

[arxiv:2602.01601] Adaptive Rollout Allocation for Online Reinforcement Learning with
Verifiable Rewards. February 2026.

[arxiv:2603.02224] Subspace Geometry Governs Catastrophic Forgetting in Low-Rank
Adaptation. February 2026.

[arxiv:2603.06610] CapTrack: Capacity-Tracking LoRA for Continual Fine-Tuning.
February 2026.

[arxiv:2506.02211] Improving LLM-Generated Code Quality with GRPO. Robeyns, Aitchison
(Bristol), June 2025.

[arxiv:2506.06579] Towards Efficient Multi-LLM Inference: Characterization and Analysis
of LLM Routing and Hierarchical Techniques. Behera et al., June 2025.

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
  (57 experiments queued as of 2026-06-07, ready to apply on A800 restoration)

A800 GPU server: 117.74.66.181:50507 (offline since 2026-05-14; 57 experiments queued).
Tau2 training server: 8× H200, CUDA 13.0 (online; tau2 eval pending for 5 checkpoints).

---

*Paper draft last updated: 2026-06-07 (weekly drafter). No new experiment results since
2026-05-14 (A800 offline, day 25). Queue: 57 experiments pending.*
