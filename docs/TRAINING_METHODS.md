# 训练方法：SFT + GRPO/DAPO

> 现状（已实现并接入 pipeline）：**Phase 3a = SFT**，**Phase 3b = GRPO/DAPO on-policy RL**。
> 早期讨论过的 **DPO 已废弃并从代码移除**——下面保留它只为说明为什么不选它。

---

## TL;DR

| 方法 | pipeline 位置 | 数据需求 | 状态 |
|------|--------------|---------|------|
| **SFT** | Phase 3a (`train_small_model.py` / `tau2_train_wrapper.sh`) | 正确代码做标签（teacher + self-repair） | ✅ 默认开 |
| **GRPO** | Phase 3b (`grpo_train_simple.py` / `grpo_tau2_train.py`) | 只需可验证 reward（test / env passed） | ✅ HumanEval 默认开；tau2 opt-in |
| **DAPO** | Phase 3b，`GRPO_ALGO=dapo` | 同 GRPO | ✅ 一键切换 |
| ~~DPO~~ | — | 正/负对比对 | ❌ 已移除 |

**流程**：先 SFT warm-start（稳、数据省），再 GRPO/DAPO 用真实 reward 拔高通过率。

---

## 1. SFT（Phase 3a）— 标签学习

```
instruction: procedure + "---" + 问题
output:      大模型正确代码（teacher）/ 小模型自修正确代码（self-repair）
Loss = -log P(output | instruction)
```

- ✅ 简单、快、稳、数据少也能用、LoRA 友好
- ❌ 学风格 > 学能力，受限于 teacher，数据少时易过拟合
- 用途：每轮的 warm-start 基础

---

## 2. GRPO（Phase 3b）— on-policy RL

DeepSeekMath 的 group-relative estimator，无需 value network。

```
对每个 prompt：
  1. 采样 K 个回答（rollout，temperature > 0）
  2. 每个回答跑可验证 reward：HumanEval pytest / tau2 env passed
  3. 组内归一化 advantage = (r - mean) / (std + eps)
  4. PPO 式 clip 更新 + KL(π‖π_ref) 惩罚（β）
```

- ✅ 直接优化通过率、不需 ground-truth 代码、在 SFT 之上 warm-start
- ❌ 需要 rollout（慢）、显存大（policy + ref）、超参敏感
- 多轮（tau2）要点：reward 取轨迹末端 env 结果（最好是 graded）、只对 agent token 算梯度
  （tool/user token mask 掉）、rollout 温度必须 >0 否则 K 条坍缩。详见
  `src/pipeline/grpo_tau2_train.py`。

---

## 3. DAPO（Phase 3b，`GRPO_ALGO=dapo`）— GRPO 的改良

DeepSeek/ByteDance 2025。基于 TRL 原生 `GRPOConfig`，无自定义 Trainer。

| | `grpo`（默认） | `dapo` |
|--|--|--|
| PPO clip | 对称 `clip(r, 1-ε, 1+ε)` | 非对称 `clip(r, 1-ε_low, 1+ε_high)` |
| loss 归一化 | 序列级 mean | token 级全局（多轮长度偏置更小） |
| 动态采样 | 无 | 丢掉 K 条 reward 全同的零方差组（零梯度，省算力） |

**为什么 DAPO**：编程/agentic 里大量简单题（必过）和极难题（必败）让整组 advantage=0；
动态采样跳过它们，非对称 clip 的高上界（ε_high）防 entropy collapse。

切换：`GRPO_ALGO=dapo DAPO_CLIP_HIGH=0.5`。每轮 `cycle_N/grpo_adapter/grpo_info.json`
记录 `algo` / `trl_loss_type` / `epsilon_low` / `epsilon_high` / `dapo_dynamic_sampling` 可对比。

---

## 4. 为什么不用 DPO

- DPO 需要成对 (chosen, rejected) 偏好数据，且偏 offline；
- 我们已有可验证 reward（test / env），GRPO/DAPO 能直接 on-policy 优化通过率，信号更强；
- 维护两套 RL 路径无必要 → DPO 脚本/数据已从仓库移除。

---

## 5. 配套超参

见 [TRAINING.md §5](TRAINING.md)（`GRPO_N_GENERATIONS` / `GRPO_LR` / `GRPO_BETA` /
`GRPO_ALGO` / `GRPO_TEMPERATURE` / `DAPO_CLIP_LOW` / `DAPO_CLIP_HIGH`）。
