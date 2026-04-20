# 训练方法对比: SFT vs RL

> 回应白的问题："这不是强化学习框架吧"
> 
> 确实不是。当前是 SFT。下面解释两种方案，你可以选。

---

## TL;DR

| 方法 | 当前脚本 | 数据需求 | 复杂度 | 推荐场景 |
|------|---------|---------|--------|---------|
| **SFT** (当前) | `train_small_model.py` | 需要**正确代码**做标签 | 简单 | 数据少 < 1000 条, 快速 PoC |
| **DPO** | (可加) | 需要**正/负对比** | 中等 | 已有 small 失败的代码 + large 成功的代码 |
| **GRPO** | (可加) | 只需要 **test case** (reward) | 复杂 | 数据多, 要真正提升 reasoning |

**推荐第一轮用 SFT**（快、简单、数据够用），**验证通路**后看效果决定是否升级到 RL。

---

## 1. SFT (当前方案) - 标签学习

### 原理

```
数据:
  instruction: "写一个多项式求根函数"
  output:      大模型的正确代码  ← 监督信号

损失:
  Loss = -log P(output | instruction)
  # 让模型尽量复现大模型的答案
```

### 优点

- ✅ **简单**: 就是 next-token prediction
- ✅ **快**: 1 epoch 几百条数据 30 分钟跑完
- ✅ **稳定**: 不会崩 (RL 经常崩)
- ✅ **数据少也能用**: 10 条就能微调
- ✅ **LoRA 兼容好**: 显存友好

### 缺点

- ❌ **学风格 > 学能力**: 模型学会"像大模型一样写"而非"真正理解"
- ❌ **受限于大模型**: 超不过 teacher
- ❌ **过拟合风险**: 数据少时易记硬答案

### 适合场景

→ **我们现在的场景: 小数据 + 快速验证**

---

## 2. DPO (Direct Preference Optimization) - 偏好学习

### 原理

```
数据:
  instruction: "写一个多项式求根函数"
  chosen:      大模型的正确代码    ← 偏好
  rejected:    小模型的错误代码    ← 避免
  
损失:
  最大化 log(σ(β * (logP(chosen) - logP(rejected))))
  # 让模型偏向正确、远离错误
```

### 优点

- ✅ **更明确的信号**: 知道"对"和"错"
- ✅ **相对简单**: 不需要 reward model
- ✅ **对比学习**: 效率比 SFT 高
- ✅ **我们有现成的数据!**: 每个 hard task 都有
  - small 失败的代码 → rejected
  - large 成功的代码 → chosen

### 缺点

- ⚠️ 需要成对数据 (我们恰好有!)
- ⚠️ β 参数调起来有讲究

### 适合场景

→ **已有 traces 的完美方案**: 我们的 traces 天然是 (成功/失败) 对比

---

## 3. GRPO (Group Relative Policy Optimization) - RL

DeepSeek-R1 用的方法。

### 原理

```
对每个 prompt:
  1. Sample N 个回答 (roll out)
  2. 每个回答跑 test case 得 reward (0/1)
  3. 归一化: advantage = (r - mean) / std
  4. PPO 式更新

损失:
  L = E[min(r·A, clip(r, 1-ε, 1+ε)·A) - β·KL(π||π_old)]
```

### 优点

- ✅ **真正的 RL**: 学会 "解决任务"
- ✅ **不需要 ground truth 代码**: 只需要能判断对错
- ✅ **强**: R1 用这个超越 o1
- ✅ **可用数据**: HumanEval 有 test cases → 可直接当 reward

### 缺点

- ❌ **复杂**: 需要 roll-out + reward model + PPO loop
- ❌ **慢**: 每步生成 N 个 sample, 再跑 test
- ❌ **显存大**: 需要 online policy + ref policy
- ❌ **易崩**: hyperparameters 敏感

### 适合场景

→ **高级方案**: 数据多 + 资源够 + 想真正提升能力

---

## 🎯 推荐路径 (Roadmap)

### Phase 1 (本周): SFT with LoRA ← **当前**

```bash
python3 experiments/train_small_model.py \
    --data data/training_data.jsonl \
    --base-model "MiniMaxAI/MiniMax-M2" \
    --use-4bit
```

**目标**: 跑通 pipeline, 验证能提升 5-10% success rate

### Phase 2 (下周): DPO with LoRA ← **推荐升级路径**

利用我们已有的 (small失败, large成功) 对比数据。

```bash
# 提取 DPO 格式 (待加)
python3 experiments/extract_dpo_data.py

# DPO 训练 (待加)  
python3 experiments/train_small_model_dpo.py \
    --data data/dpo_data.jsonl \
    --base-model "MiniMaxAI/MiniMax-M2" \
    --use-4bit
```

### Phase 3 (有空): GRPO / RL

用 HumanEval test cases 直接做 reward。

```bash
python3 experiments/train_small_model_grpo.py \
    --tasks data/HumanEval.jsonl \
    --base-model "MiniMaxAI/MiniMax-M2" \
    --n-rollout 8
```

---

## 📊 三种方法数据对比 (基于 HumanEval 30-90 实验)

| 训练数据类型 | 样本数 | SFT 可用? | DPO 可用? | GRPO 可用? |
|-------------|-------|----------|----------|-----------|
| 只知道 large 正确代码 | 6 | ✅ | ⚠️ (无 rejected) | ✅ (用 test) |
| (small失败代码, large正确代码) 对 | 6 | ✅ | ✅ | ✅ |
| HumanEval test cases | 164 | ❌ | ❌ | ✅ |

**本项目现状**:
- SFT 数据: ✅ 已生成 6 条 (`data/training_data.jsonl`)
- DPO 数据: 🔜 可加 (需要补 small 的失败代码)
- GRPO: 🔜 可加 (用 HumanEval 的 test cases)

---

## 为什么 **先 SFT** 再谈 RL?

1. **Pipeline 通验证**: SFT 简单, 先证明整个链路能跑通
2. **Teacher-student 已经 work**: 大模型 → 小模型的知识蒸馏是有成熟 recipe 的
3. **数据效率**: 几条样本 SFT 就有效, RL 需要大量 rollout
4. **LoRA 友好**: SFT 对 LoRA 非常友好, RL 对 LoRA 适配要小心
5. **风险低**: SFT 不会崩, RL 各种 stability issue

**但也不排斥 RL**: 如果白想直接上 GRPO / DPO, 我可以写这两个脚本。

---

## 🤝 要我加 RL 脚本吗?

如果白觉得"要做就做 RL 的"，我可以这周就加上：

### A. DPO 脚本 (3 小时写完, 比 SFT 好, 复杂度适中)

```python
# 核心代码 (用 trl 库的 DPOTrainer)
from trl import DPOTrainer, DPOConfig

trainer = DPOTrainer(
    model=model,
    args=DPOConfig(...),
    train_dataset=dpo_dataset,  # (prompt, chosen, rejected) 格式
    tokenizer=tokenizer,
    peft_config=lora_config,
    beta=0.1,
)
trainer.train()
```

### B. GRPO 脚本 (8 小时写, 最强, 最复杂)

```python
# 核心: 每步 sample N 个, 跑 test case 算 reward
from trl import GRPOTrainer, GRPOConfig

def reward_fn(prompts, completions, **kwargs):
    # 跑 HumanEval test
    return [1.0 if run_test(c) else 0.0 for c in completions]

trainer = GRPOTrainer(
    model=model,
    args=GRPOConfig(num_generations=8, ...),
    train_dataset=humaneval_tasks,
    reward_funcs=[reward_fn],
    peft_config=lora_config,
)
trainer.train()
```

**白选哪个?**
