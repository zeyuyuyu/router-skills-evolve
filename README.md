# Router + Skills Evolve

> **一句话**: Router 把请求路由到合适的 LLM，Skills 通过历史 traces 学习"哪类题该用哪个模型"，自主 evolve 变聪明。

已验证：**99% 准确率 + 省 83% 成本**（HumanEval 164 题真实实验）

---

## 🚀 5 分钟上手

### 1. 环境准备

```bash
# Python 3.10+
pip install openai requests

# 拿 CommonStack API key (团队群问)
export COMMONSTACK_API_KEY="ak-f53d..."
```

### 2. 启动 UncommonRoute proxy (本地)

```bash
# 克隆 UncommonRoute (如果没装)
git clone https://github.com/CommonstackAI/UncommonRoute.git
cd UncommonRoute

# 启动 proxy
UNCOMMON_ROUTE_API_KEY=$COMMONSTACK_API_KEY \
UNCOMMON_ROUTE_UPSTREAM="https://api.commonstack.ai/v1" \
python3 -m uncommon_route.cli serve --port 8403 &
```

### 3. 跑基线实验（最简单）

```bash
cd router_skills_evolve

# 在 HumanEval 30 题上跑 Router+Skills (约 5 分钟)
python3 experiments/run_evolve.py --n 30 --rounds 3
```

### 4. 收集训练数据（给白）

```bash
# 从 traces 提取训练集 (小模型失败、大模型成功的题)
python3 experiments/extract_training_data.py \
    --traces data/traces/traces.jsonl \
    --output data/training_data.jsonl
```

### 5. 训练小模型

```bash
# 需要 GPU
python3 experiments/train_small_model.py \
    --base_model deepseek-ai/deepseek-v3.2 \
    --data data/training_data.jsonl \
    --output outputs/deepseek-finetuned
```

---

## 📂 项目结构

```
router_skills_evolve/
├── README.md                 ← 本文件
├── src/                      ← 核心代码
│   ├── config.py             ← 配置 (API, 模型, 价格)
│   ├── models.py             ← LLM 调用封装
│   ├── skills.py             ← Skills 数据结构 + 学习
│   ├── router.py             ← Router + Skills 路由决策
│   └── evaluator.py          ← 代码评估 (跑 pytest)
├── experiments/              ← 运行脚本
│   ├── run_baseline.py       ← 5 种策略 baseline 对比
│   ├── run_evolve.py         ← Evolve 主实验
│   ├── extract_training_data.py  ← 从 traces 提取训练集
│   └── train_small_model.py  ← LoRA fine-tuning
├── data/
│   ├── traces/               ← 收集的 traces (prompt+model+success)
│   └── skills/               ← 学到的 skills (JSON)
├── results/                  ← 实验结果
└── docs/
    ├── ARCHITECTURE.md       ← 架构详解
    ├── TRAINING.md           ← 训练指南 (白看这个)
    └── EXPERIMENTS.md        ← 过往实验结果
```

---

## 🎯 核心概念

### Pipeline

```
用户 Prompt
    ↓
  ┌─────────────┐
  │   Router    │  ← UncommonRoute 分类器 (本地, 无 LLM 调用)
  └─────────────┘
    ↓ tier: SIMPLE/MEDIUM/COMPLEX
  ┌─────────────┐
  │   Skills    │  ← 查 "这类题用啥模型最便宜"
  └─────────────┘
    ↓ 推荐具体 model
  ┌─────────────┐
  │    LLM      │  ← 真实调用 (这里才产生成本)
  └─────────────┘
    ↓
  ┌─────────────┐
  │  Test code  │  ← pytest 验证代码是否通过
  └─────────────┘
    ↓ 收集 trace: (prompt, model, success, cost)
  ┌─────────────┐
  │ Skills 学习  │  ← 下次同类题决策更优
  └─────────────┘
```

### Evolve 的 3 个阶段

1. **Skills Evolve** (已验证): 收集 traces → 学出 skills → 降级决策更准
2. **Model Evolve** (下一步): traces → 训练 SMALL model → 模型变强 → 更多降级
3. **持续循环**: 越用越省钱，准确率不降

---

## 📊 已验证结果（HumanEval 164 题真实实验）

| 策略 | 成功率 | 成本 | vs GPT-5.4 |
|------|--------|------|-----------|
| Pure GPT-5.4 | 95% | $0.37 | 100% |
| Router Only | 74% | $0.15 | 42% |
| Router + Fallback | 98% | $0.27 | 73% |
| **Router + Skills Evolve** | **99%** | **$0.06** | **17%** |

**Evolve 赢在**: 同时最高准确率 + 最低成本。

---

## 🔧 谁负责什么

| 模块 | 负责人 | 代码位置 |
|------|--------|---------|
| Router (UncommonRoute) | 已有 | `UncommonRoute` repo |
| Skills 数据结构 + 学习 | Franklin | `src/skills.py` |
| 路由决策 | Franklin | `src/router.py` |
| 代码评估 | Franklin | `src/evaluator.py` |
| **模型训练** | 白 | `experiments/train_small_model.py` |
| Inference 部署 | 平台团队 | TBD |

---

## 📚 进一步阅读

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - 架构详解
- [TRAINING.md](docs/TRAINING.md) - **训练指南（白看这个）**
- [EXPERIMENTS.md](docs/EXPERIMENTS.md) - 实验结果汇总
- [原始 HumanEval 164 报告](../uncommonroute_skill_experiment/FULL_HUMANEVAL_FINAL_REPORT.md)

---

## ❓ FAQ

**Q: 为什么实验只用 2 个模型（SMALL + LARGE）？**  
A: 控制变量简化对比。架构支持 N 个，生产 UncommonRoute 已支持 48 个。

**Q: 训练数据怎么来？**  
A: 自动从 traces 提取 "小模型失败+大模型成功" 的题，大模型代码做 ground truth。

**Q: 为什么不用闭源模型训练？**  
A: 闭源（GPT-5.4, Claude, Gemini）不开权重。开源可训：deepseek, qwen, glm, minimax 等（24 个）。

**Q: 需要什么 GPU？**  
A: 看模型大小。小（minimax-m2 ~20B）用 2×A100；大（deepseek-v3.2 37B activate MoE）用 8×A100。
