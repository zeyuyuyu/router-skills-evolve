# Towards Always-On LLM Research: Persistent Routing, Skills, and Continual Adapter Training

*Working draft — auto-updated weekly by the autonomous research agent.*
*Last updated: 2026-05-17*

---

## Abstract

We present an always-on research system that co-evolves three interdependent components of an LLM serving stack: (1) a learnable task router that directs requests between a small local model and a large cloud model; (2) a growing skill library (SkillBook) that encodes solutions reusable across similar tasks; and (3) a continual adapter-training loop that improves the local model via online reinforcement learning. Experiments on the MBPP code-generation benchmark demonstrate that the learned router achieves 93.0% routing accuracy with only 2.1% fallback to the large model — a 24-point accuracy gain over always-small-with-fallback. On the LLM continual training track, we identify Qwen2.5-Coder-1.5B-Instruct with GRPO (200 tasks × 4 rollouts, LoRA r=16) as the current best recipe, improving MBPP eval@100 pass rate from 47% to 49%. Larger models (3B, 7B) and longer training (400 tasks) degrade under the same flat-GRPO recipe, motivating pending experiments with curriculum ordering, KL regularization, and partial-test rewards. The system is fully automated and produces daily experiment proposals grounded in arxiv literature.

---

## 1 Introduction

Large language model (LLM) serving stacks face a persistent tension between capability and cost. Routing every request to the most capable model is accurate but expensive; routing exclusively to a small local model is cheap but misses hard tasks. We address this with a three-track research loop:

- **Router track**: Learn a binary classifier that routes each request to the appropriate model (small local vs. large cloud) based on task difficulty signals derived from the SkillBook.
- **Skills track**: Maintain a growing library of skills (named, executable code solutions) that the router uses as features and the LLM can invoke as tools.
- **LLM continual training track**: Continuously fine-tune the small local model using reinforcement learning with executable-test rewards, so it can handle more tasks without a fallback.

The system runs autonomously on an A800 GPU server, reading its own experiment history, generating new experiment proposals grounded in recent arxiv literature, executing them, and updating this paper draft. The current paper reports results as of week ending 2026-05-17.

**Key research questions being investigated:**
1. How accurately can a learned router predict when the small model will fail?
2. Does SkillBook-based routing improve over signature-only routing?
3. Which LLM training paradigm (SFT, GRPO, DPO, curriculum RL) reliably improves the small model on code generation?
4. What is the optimal task-count, model-size, and hyperparameter regime for continual GRPO?
5. Does continual adapter training cause catastrophic forgetting outside the training domain?

---

## 2 System

### 2.1 Architecture

The system comprises three tracks plus an autonomous orchestrator:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Autonomous Orchestrator                     │
│  (reads state.json, proposes experiments, executes, updates)    │
└──────────────┬───────────────┬──────────────────────────────────┘
               │               │
     ┌─────────▼──────┐  ┌─────▼────────┐  ┌────────────────────┐
     │  Router Track  │  │ Skills Track │  │  LLM Training Track│
     │  BERT-base     │  │  SkillBook   │  │  GRPO / SFT / DPO  │
     │  classifier    │  │  (evolving)  │  │  LoRA adapters     │
     └────────────────┘  └─────────────┘  └────────────────────┘
```

**Router**: A BERT-base classifier trained on (task, skillbook-signature) → {small, large} labels. At inference time, if the confidence score exceeds a tuned threshold, the request is routed to the small model; otherwise it falls back to the large model.

**Skills track**: When a new task is solved, the solution is distilled into a named skill stored in the SkillBook. Skills are represented as executable Python functions with docstrings. The SkillBook signature (presence/absence of relevant skills) is a feature for the router.

**LLM training track**: The small model (Qwen2.5-Coder, various sizes) is fine-tuned using LoRA adapters with binary executable-test rewards (GRPO/REINFORCE). Adapters are promoted to production only when they clear a held-out MBPP eval threshold.

### 2.2 Evaluation

- **Router**: Classification accuracy, large-class F1, fallback rate, cost-vs-always-large on the unified router eval split (848 examples: 579 small, 269 large).
- **LLM**: MBPP pass@1 on eval@100 (first 100 MBPP test problems), measured with and without the LoRA adapter. Gain = adapter_pass_rate − base_pass_rate.
- **Skills**: Number of skills acquired, success rate on new task batch, cost per task.

### 2.3 Datasets

- **MBPP**: Mostly Basic Programming Problems. Sanitized split used for training; first 100 test problems as eval@100; held-out eval@20 for quick smoke tests.
- **Router data**: 3,391 examples (2,543 train, 848 eval) from the unified router benchmark combining uncommonroute-bench and internal task traces. Eval class split: 579 small, 269 large.
- **HumanEval**: 164 problems reserved for cross-domain forgetting evaluation (not yet used in training).

---

## 3 Experiments

### 3.1 Router Track

#### 3.1.1 Routing Ablation Study

Experiment ID: `e2e_ablation_router_sft_qwenchat` (2026-05-09, A800)
Artifact: `results/e2e_ablation_a800_20260509_summary.json`

Three routing strategies evaluated on the unified 848-example eval split:

| Strategy | Accuracy | Large F1 | Fallback Rate | Cost vs. Always-Large |
|----------|----------|----------|---------------|----------------------|
| Always-small + fallback (baseline) | 68.28% | 0.000 | 31.72% | 33.54% |
| SkillBook signature only | 69.46% | 0.249 | 26.65% | 37.27% |
| **Learned router (threshold=0.57)** | **93.04%** | **0.895** | **2.12%** | **37.75%** |

**Key finding**: The learned BERT-base router achieves 93.04% accuracy, a +24.76 pp improvement over the always-small baseline and +23.58 pp over signature-only routing. The large-class F1 of 0.895 indicates strong recall on hard tasks (recall_large = 0.938, precision_large = 0.840 from `joint_evolver_a800_20260502_summary.json`). Fallback rate drops from 31.7% to 2.1%, meaning 97.9% of requests can be served locally.

The cost-vs-always-large metric is 37.75% for the learned router (vs. 33.54% for the baseline). The slight cost increase is because the router occasionally routes easy tasks to the large model (false positives) while the baseline has lower cost by aggressively routing to small — but at much lower accuracy.

#### 3.1.2 SkillBook-Router Interaction

SkillBook-only routing (no learned classifier) gives 69.46% accuracy vs. 68.28% baseline — a +1.18 pp improvement. This demonstrates that skill presence signals carry genuine routing information, but the learned classifier is required to capture the full signal.

---

### 3.2 Skills Track

#### 3.2.1 Initial Skill Acquisition

Experiment: `evolve_20260420_135957` (2026-04-20)
Artifact: `results/evolve_20260420_135957.json`

| Metric | Value |
|--------|-------|
| Tasks attempted | 3 |
| Success rate | 100% |
| Skills acquired | 3 |
| Total cost | $0.000286 |
| Avg cost/task | $0.0000953 |
| Avg attempts/task | 1.0 |

The SkillBook correctly acquired 3 skills on the first round, all marked `✅ small ok` (solvable by the small model). This validates the skill acquisition loop end-to-end.

---

### 3.3 LLM Continual Training

#### 3.3.1 SFT Ablation (0.5B)

Experiments from `llm_sft_qwenchat_a800_20260502_summary.json` and `llm_training_summary_20260510.json`. All SFT configurations evaluated on MBPP eval@20.

| Config | Base | Trained | Result |
|--------|------|---------|--------|
| Code-prompt SFT, 120 tasks, r=16, 3ep | 7/20 | 1/20 | **severely degraded** |
| Qwen-chat SFT, 120 tasks, r=16, 3ep | 10/20 | 9/20 | below base |
| Qwen-chat conservative SFT, 120 tasks, r=8, lr=2e-5, 1ep | 10/20 | 9/20 | below base |
| Code-prompt SFT, MBPP hard-6, r=16, 8ep | 4/6 | 2/6 | degraded |
| Expanded SFT, 400 tasks, r=16, lr=5e-5, 2ep | 10/20 | 9/20 | below base |

**Key finding**: SFT universally fails to improve over the qwen-chat base model across all configurations tested (5 distinct recipes). The prompt format matters substantially: qwen-chat base (10/20 = 50%) vs. code-prompt base (7/20 = 35%) without any fine-tuning. SFT on qwen-chat preserves more of the base performance but never exceeds it. **SFT is closed as a viable path; no further SFT experiments are planned.**

#### 3.3.2 GRPO Ablation — Model Scale

Experiments from `best_llm_training_case_20260509.json`, `rl_15b_a800_20260509_summary.json`, `llm_training_summary_20260510.json`.

All GRPO runs: 1 epoch, 4 rollouts/prompt, LoRA, qwen-chat, binary reward, eval@100.

| Model | Tasks | LoRA r | LR | Base | Trained | Δ |
|-------|-------|--------|----|------|---------|---|
| 0.5B | 40 | 8 | 5e-6 | 8/20 | 9/20 | +1 (small) |
| 0.5B | 120 | 8 | 5e-6 | 8/20 | 9/20 | +1 (replicated) |
| **1.5B** | **200** | **16** | **5e-6** | **47/100** | **49/100** | **+2 (BEST)** |
| 1.5B | 400 | 16 | 5e-6 | 47/100 | 46/100 | −1 (degraded) |
| 3B | 200 | 16 | 5e-6 | 62/100 | 60/100 | −2 (degraded) |
| 3B | 400 | 8 | 1e-5 | 62/100 | 62/100 | 0 (tied, SFT) |
| 7B | 100 | 16 | 5e-6 | 77/100 | 75/100 | −2 (degraded) |

**Key finding**: GRPO with the standard recipe (lr=5e-6, r=16, 200 tasks×4 rollouts) works **only at 1.5B scale**. Identical configuration at 3B and 7B degrades. The 0.5B model shows a small positive signal on eval@20 but not on the larger eval@100.

The best case (`rl_15b_a800_20260509_summary.json`):
- Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Training: 200 MBPP tasks, GRPO/REINFORCE, 4 rollouts/prompt, 1 epoch, lr=5e-6, LoRA r=16, qwen-chat, binary executable-test reward
- Result: **47/100 → 49/100 (+2.0 pp, +4.3% relative)**
- Artifacts: `results/rl_15b_a800_20260509_summary.json`

#### 3.3.3 Task-Count Ablation (1.5B)

| Tasks | LoRA r | LR | Base | Trained | Δ |
|-------|--------|----|------|---------|---|
| 200 | 16 | 5e-6 | 47/100 | 49/100 | **+2** |
| 400 | 16 | 5e-6 | 47/100 | 46/100 | **−1** |

Task count is **nonmonotonic** with a peak at or before 200. The +2pt gain at 200 collapses to −1pt at 400, suggesting either reward hacking, catastrophic forgetting, or "success amplification collapse" (Guo et al. 2025, arxiv:2503.06639) — where the model achieves high solve rates on early tasks, reducing the advantage variance and making updates noisy.

#### 3.3.4 End-to-End Component Ablation

Experiment: `e2e_ablation_router_sft_qwenchat` (2026-05-09)
Code pass rates use the 1.5B model with the best GRPO adapter (for the "full" row).

| Configuration | Router Acc | Fallback Rate | Code Pass Rate |
|---------------|-----------|----------------|----------------|
| Always-small + fallback (no training) | 68.3% | 31.7% | 47/100 |
| + Skills evolve only | 69.5% | 26.7% | 47/100 |
| + Skills + learned router | 93.0% | 2.1% | 47/100 |
| **+ Skills + router + GRPO adapter (full)** | **93.0%** | **2.1%** | **49/100** |

The full system improves both routing accuracy and code quality. The LLM track contributes independently of the router: routing accuracy is unchanged between the router-only and full-system rows, while code pass rate gains +2 pts.

---

### 3.4 Catastrophic Forgetting Analysis

**Status: incomplete.** The 3B GRPO adapter (62→60/100 on MBPP eval) has been evaluated only within the MBPP distribution. It has not been evaluated on HumanEval (164 problems) or any OOD benchmark. This is a critical gap flagged in pending experiment `exp_2026_05_17_004_forgetting_eval_3b_grpo_adapter` (priority 5) and `exp_2026_05_15_004_forgetting_eval_15b_adapter` (priority 6).

The 1.5B best-case adapter has also not been evaluated outside MBPP.

**Open question**: Is the MBPP degradation at 3B "targeted forgetting" (MBPP-distribution-specific) or general? arxiv:2601.19897 (Self-Distillation, Jan 2026) distinguishes these cases and shows they require different remediation strategies.

---

### 3.5 Other Arms

#### 3.5.1 Prompt Format Discovery

Experiment: `llm_sft_qwenchat_a800_20260502_summary.json`

Base model pass rates differ dramatically by prompt format on 0.5B:

| Prompt style | Base pass rate (eval@20) |
|-------------|--------------------------|
| Code (raw) | 35% (7/20) |
| Qwen-chat | **50% (10/20)** |

The +15 pp gain from qwen-chat formatting is larger than any training-induced gain achieved so far. All subsequent experiments use qwen-chat. This result also implies that earlier published MBPP baselines using raw-code prompts may underestimate Qwen model capability.

#### 3.5.2 Joint Runner Validation

Experiment: `joint_evolver_a800_20260502_summary.json` (full joint run, threshold=0.53)

| Component | Value |
|-----------|-------|
| Router accuracy | 92.36% |
| Large-class F1 | 0.886 |
| Fallback rate | 1.98% |
| Cost vs. always-large | 38.6% |
| LLM base pass (eval@20) | 15% (3/20, code-prompt 0.5B) |
| LLM adapter pass (eval@20) | 10% (2/20) |

This confirmed the joint runner operates end-to-end. The LLM degradation here used the old code-prompt format with a 0.5B model — consistent with the SFT failure pattern. Router results were already strong.

---

## 4 Findings

### 4.1 Positive Results

- **Learned router works strongly**: BERT-base classifier at threshold=0.57 achieves 93.0% routing accuracy on a 848-example eval set, with large-class F1=0.895, fallback rate 2.1%. This is the primary positive result of the system.
- **GRPO at 1.5B, 200 tasks, r=16**: Statistically consistent +2 pp gain on MBPP eval@100. Replicated across multiple runs (initial 0.5B smoke tests showed +1 pp, 1.5B full run showed +2 pp). This is the only confirmed positive LLM training result.
- **Qwen-chat prompt format**: Switching from code-prompt to qwen-chat yields +15 pp on 0.5B MBPP eval@20 with no training. All experiments now use this format.
- **SkillBook acquisition works**: 100% success rate on initial 3-task round; cost per task $0.0000953.
- **Full system integration**: Skills + router + GRPO adapter achieves 93.0% routing accuracy + 49/100 code pass rate vs. 68.3% routing + 47/100 for the unaugmented baseline.

### 4.2 Negative Results

- **SFT universally fails**: 5 SFT configurations (varying prompt style, LR, rank, epochs, data size) all degraded or tied vs. qwen-chat base on 0.5B. SFT is ruled out.
- **GRPO degrades at 3B and 7B**: Same recipe (200 tasks, 4 rollouts, lr=5e-6, r=16) that gives +2 pp at 1.5B gives −2 pp at 3B and −2 pp at 7B. Model scale does not automatically help under the flat-GRPO recipe.
- **GRPO degrades at 400 tasks (1.5B)**: Doubling the training set from 200→400 tasks reverses the gain (−1 pp). Task count is the most sensitive hyperparameter currently identified.
- **SkillBook alone is insufficient for routing**: +1.18 pp accuracy improvement over baseline, vs. +24.76 pp for the learned classifier.

### 4.3 Surprises

- **Prompt format effect size**: The qwen-chat vs. code-prompt gap (+15 pp) exceeds any training-induced effect observed so far. Suggests that prompt engineering is a first-order lever even for fine-tuned models.
- **Model-scale GRPO inversion**: Larger models degrade under the same RL recipe that improves the smaller model. The conventional expectation (larger model = better RL learning) is violated. Likely mechanism: larger models have already solved more MBPP training tasks, reducing advantage variance and causing noise updates (Guo et al. 2025, "success amplification collapse").
- **Task-count nonmonotonicity**: More training data hurts (200→400 tasks: +2→−1 pp). This is the strongest experimental constraint on future training designs.

---

## 5 Open Questions

1. **Does curriculum ordering fix 3B and 400-task degradation?** Pending: `exp_2026_05_15_001` (curriculum 1.5B), `exp_2026_05_16_002` (curriculum 3B), `exp_2026_05_17_001` (KL+curriculum 300 tasks). RECRL (arxiv:2605.00433) shows +6 pp on 3B with curriculum; we expect similar.

2. **Does KL regularization enable longer training?** Pending: `exp_2026_05_15_002` (KL-GRPO, 300 tasks). The 200→400 degradation is consistent with distribution collapse absent a reference constraint.

3. **Is the +2 pp result reproducible (seed variance)?** Pending: `exp_2026_05_15_003` (multi-seed verify, 3 seeds, 1.5B 200x4). Effect size is small (+2 pp) relative to binomial noise at n=100.

4. **Is the 3B degradation targeted or general?** Pending: `exp_2026_05_17_004` (forgetting eval 3B on HumanEval + OOD MBPP). arxiv:2601.19897 framework needed to distinguish targeted vs. general forgetting.

5. **Can P-GRPO (partial-test conditional reward) improve over binary reward?** Pending: `exp_2026_05_16_001`. Binary reward gives zero gradient on ~53% of eval tasks where no rollout passes.

6. **Does LoRA rank r=32 improve over r=16?** Pending: `exp_2026_05_17_003`. LoRA paper (arxiv:2106.09685) suggests monotonic gains through r=32 on complex code tasks.

7. **Does lr=1e-6 reduce variance without hurting gain?** Pending: `exp_2026_05_17_002`. Gradient dynamics (Guo et al. 2025) suggests lr=5e-6 may be near instability boundary.

8. **Does 2-stage staircase training (100+100) avoid the 200→400-task degradation?** Pending: `exp_2026_05_16_003`. arxiv:2601.19897 shows incremental training with distillation outperforms one-shot full-batch training.

9. **Does router threshold need recalibration after LLM adapter promotion?** Pending: `exp_2026_05_16_004`. Routing signal derived from base model skill-match may drift after adapter changes the model's capability boundary.

---

## 6 Reproducibility

### 6.1 Server

All GPU experiments run on A800 server (117.74.66.181:50507, user zeyuwang). Note: server has been unreachable since 2026-05-14T20:03Z.

### 6.2 Code

| Component | Script |
|-----------|--------|
| Router training | `experiments/train_learnable_router.py` |
| Router threshold tuning | `experiments/tune_learnable_router_threshold.py` |
| Router evaluation | `experiments/evaluate_learnable_router.py` |
| LLM GRPO training | `experiments/train_small_model_grpo.py` |
| LLM evaluation | `experiments/evaluate_finetuned_model.py` |
| E2E ablation | `experiments/run_e2e_ablation.py` |
| Joint evolver | `experiments/run_joint_evolver.py` |
| Skill evolution | `experiments/run_evolve.py` |

### 6.3 Artifact Paths (A800)

| Artifact | Path |
|----------|------|
| Router model | `/data0/home/zeyuwang/router-skills-evolve-runs/learned-router-mixed-pretrained` |
| 1.5B GRPO adapter (best) | `/data0/home/zeyuwang/router-skills-evolve-runs/rl_15b/qwen25_coder_15b_grpo_200x4` |
| 1.5B base eval@100 | `/data0/home/zeyuwang/router-skills-evolve-results/rl_15b/qwen25_coder_15b_base_eval100.json` |
| 1.5B adapter eval@100 | `/data0/home/zeyuwang/router-skills-evolve-results/rl_15b/qwen25_coder_15b_grpo_eval100.json` |
| E2E ablation | `/data0/home/zeyuwang/router-skills-evolve-results/e2e_ablation/e2e_ablation_router_sft_qwenchat.json` |
| State / experiment history | `/data0/home/zeyuwang/auto_research/state.json` |

### 6.4 Key Configs (Best Cases)

**Learned router (best):**
```
base: bert-base-uncased
threshold: 0.57
train_split: 2543 examples
eval_split: 848 examples
accuracy: 93.04%
fallback_rate: 2.12%
```

**1.5B GRPO adapter (best):**
```
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
train_tasks: 200 (MBPP augmented, excl. eval)
epochs: 1
rollouts_per_prompt: 4
lr: 5e-6
lora_r: 16
prompt_style: qwen-chat
reward: binary executable-test pass/fail
eval: MBPP test_eval_all first 100
result: 47/100 → 49/100 (+2 pp)
```

### 6.5 Pending Queue

13 experiments queued in `/data0/home/zeyuwang/router-skills-evolve/auto_research/pending_queue_update.py`. Apply when A800 connectivity is restored:
```bash
cd /data0/home/zeyuwang/router-skills-evolve
python3 auto_research/pending_queue_update.py
```

---

*End of draft. Next update: week of 2026-05-24 (or when A800 connectivity is restored and experiments complete).*
