# 训练指南（白看这个）

> **给白**: 这份文档解释清楚**在训什么**、**为什么要训**、**怎么训**。  
> **预计时间**: 读懂 10 分钟，跑通 PoC 半天。

---

## 1. 我们在训什么？为什么要训？

### 背景（一句话）

> 我们有个 **Router + Skills** 系统：自动把简单题路由到便宜模型（deepseek），难题路由到贵模型（gpt-5.4）。

已经跑了 HumanEval 164 题实验，结果很好：**99% 准确率 + 省 83% 成本**。

### 但还有改进空间

Skills 分析后发现：**有 9 类题（signature）**小模型做不到，必须升级到大模型。
比如：
- 多项式计算（poly）
- 密码学（decode_cyclic, decode_shift）
- 布尔字符串（bool/str）

### 训练目标

**让小模型学会这些"难题"**！训完后：
- 原本要用 gpt-5.4 的题 → 小模型也能做
- 省更多钱（从省 83% → 省 90%+）
- 准确率保持 99%+

### 怎么训？

**用大模型的正确答案当老师**，教小模型做这些题：

```
训练数据:
  - Instruction: "Write a function to decode_cyclic..."
  - Output:      大模型生成的正确代码  <- 老师的答案
  
训练后的小模型:
  学会了这类题的解法 → 自己也能做对
```

这叫**知识蒸馏 (Knowledge Distillation) + 监督微调 (SFT)**。

---

## 2. 具体步骤（走一遍）

### Step 1: 收集 traces (已完成)

之前的 `run_evolve.py` 已经产生了 traces，存在：
```
router_skills_evolve/data/traces/traces_*.jsonl
```

每条 trace 记录了：
- 哪道题
- 小模型 / 大模型 分别 成功还是失败
- 花了多少钱

### Step 2: 提取训练数据

跑一个脚本，筛"小模型做不到但大模型能做"的题：

```bash
cd router_skills_evolve

python3 experiments/extract_training_data.py \
    --traces "data/traces/traces_*.jsonl" \
    --output data/training_data.jsonl \
    --include-test-cases
```

这会：
1. 扫描所有 trace 文件
2. 筛 hard tasks（小模型失败 + 大模型成功）
3. **重跑一次大模型**获取完整代码（ground truth）
4. 输出 JSONL 格式（Alpaca 风格）

输出示例：
```json
{
  "task_id": "HumanEval/32",
  "instruction": "def poly(xs, x): ... (完整 prompt)",
  "input": "",
  "output": "def poly(xs, x):\n    return sum(...)\n\ndef find_zero(xs):\n    ...",
  "signature": "L|advanced/list/num",
  "source_model": "openai/gpt-5.4-2026-03-05"
}
```

### Step 3: 准备训练环境

**推荐第一次用 MiniMax-M2**（小模型，容易跑通）：

```bash
# 系统要求:
#   GPU: 1-2 × A100 80GB (MiniMax-M2 ~20B)
#   Disk: 50GB+
#   CUDA: 12.1+

# 装依赖:
pip install torch transformers peft datasets accelerate bitsandbytes trl

# 可选: 装 flash-attention 加速 (需要 CUDA)
pip install flash-attn --no-build-isolation
```

### Step 4: 训练！

```bash
python3 experiments/train_small_model.py \
    --data data/training_data.jsonl \
    --base-model "MiniMaxAI/MiniMax-M2" \
    --output outputs/minimax-m2-finetuned \
    --lora-r 16 \
    --epochs 3 \
    --batch-size 4 \
    --use-4bit
```

**解释**:
- `--base-model`: 要微调的底模（MiniMax-M2 = 20B, 适合 PoC）
- `--lora-r 16`: LoRA 秩，越大越强但越慢
- `--epochs 3`: 训 3 轮（数据少时多训几轮）
- `--batch-size 4`: 每次处理 4 条
- `--use-4bit`: 4-bit 量化，省显存（QLoRA）

**预计时间**: 30-60 分钟（~100 条样本）

### Step 5: 评估训练效果

训完后，用 finetuned 模型跑一遍 HumanEval 看效果：

```bash
# TODO (Franklin 会加这个脚本):
python3 experiments/evaluate_finetuned.py \
    --model outputs/minimax-m2-finetuned \
    --base-model "MiniMaxAI/MiniMax-M2"
```

看指标：
- Small 模型 HumanEval success rate: 88% → ? (目标 >= 92%)
- Skills 中"必须升级"的数: 9 → ? (目标 <= 5)

---

## 3. 不同模型的资源需求

| 模型 | 参数量 | 最小 GPU | 训练时长 (~100 样本) |
|------|-------|---------|---------------------|
| **MiniMaxAI/MiniMax-M2** ⭐ | ~20B | 1 × A100 80G (用 4bit) | ~30 分钟 |
| zai-org/GLM-4.5-Air | ~12B MoE | 1 × A100 40G | ~20 分钟 |
| xiaomi/mimo-v2-omni | ~10B | 1 × A100 40G | ~15 分钟 |
| MiniMaxAI/MiniMax-M2.5 | ~20B | 1 × A100 80G | ~30 分钟 |
| Qwen/Qwen3-Coder-480B | 35B activate | 8 × A100 80G | ~4 小时 |
| deepseek-ai/DeepSeek-V3.2 | 37B activate | 8 × A100 80G | ~6 小时 |

**强烈推荐先用 MiniMax-M2 跑 PoC**，验证 pipeline 通了再扩大。

---

## 4. 具体参数说明

### `--lora-r` (LoRA rank)

| 值 | 效果 | 建议 |
|----|------|------|
| 8 | 参数最少，学习能力弱 | 数据量 <100 条 |
| 16 | **平衡** | **默认推荐** |
| 32 | 更强学习力 | 数据 >500 条时用 |
| 64 | 接近全量微调 | 数据很多时用 |

### `--epochs`

- 数据少（<100）: **3-5 epochs**
- 数据中（100-1000）: 2-3 epochs  
- 数据多（>1000）: 1-2 epochs

**警告**: 过多 epochs 会过拟合（模型只记住训练集，泛化差）。

### `--batch-size + --grad-accum`

**有效 batch size = batch_size × grad_accum**
- 小 GPU: `--batch-size 2 --grad-accum 8` (effective 16)
- 大 GPU: `--batch-size 4 --grad-accum 4` (effective 16)

### `--use-4bit` (QLoRA)

**打开** = 节省 4 倍显存，但速度慢 20%。单卡时必开。

---

## 5. 常见问题

### Q: 训练数据太少怎么办？

我们 HumanEval 164 题里只有 ~20 条 hard samples 不够。

**扩充方法**:
```bash
# 1. 用更多 benchmark 产生 traces:
python3 experiments/run_evolve.py --n 164 --rounds 4  # HumanEval

# 2. (TODO Franklin) 加 MBPP 支持:
python3 experiments/run_evolve_mbpp.py --n 500

# 3. 混入公开 code dataset:
# - CodeAlpaca-20k
# - CodeContests
# - APPS
```

### Q: GPU OOM (out of memory)

```bash
# 方案 1: 用 4bit 量化
--use-4bit

# 方案 2: 减小 batch
--batch-size 1 --grad-accum 16

# 方案 3: 减少 max seq length
--max-seq-len 1024

# 方案 4: 用更小的模型
--base-model "MiniMaxAI/MiniMax-M2"  # 不用 DeepSeek-V3.2
```

### Q: 训完模型怎么用？

训完只有 **LoRA adapter**（几百 MB），需要：
1. 上传到 HuggingFace Hub 或本地
2. 部署到 inference 服务（vLLM, TGI）
3. 注册到 CommonStack（需要平台团队帮忙）
4. 在 UncommonRoute 配置里加这个新模型

### Q: 如何知道训得好不好？

**客观指标**:
- Training loss 持续下降 → 正在学
- Eval loss 先降后升 → 过拟合了，早停

**业务指标**:
- Small 模型在 HumanEval 上 success rate ↑
- Skills 中"必须升级"的 cluster 数 ↓
- Pipeline 总成本 ↓

---

## 6. 完整快速跑通 (30 分钟)

```bash
# 0. 进目录
cd router_skills_evolve

# 1. (如果 traces 不够, 先跑 evolve 产生数据)
python3 experiments/run_evolve.py --n 60 --rounds 4

# 2. 提取训练数据
python3 experiments/extract_training_data.py \
    --traces "data/traces/*.jsonl" \
    --output data/training_data.jsonl

# 3. 检查数据
head -1 data/training_data.jsonl | python3 -m json.tool

# 4. 训练 (需要 GPU)
python3 experiments/train_small_model.py \
    --data data/training_data.jsonl \
    --base-model "MiniMaxAI/MiniMax-M2" \
    --output outputs/minimax-m2-v1 \
    --lora-r 16 --epochs 3 --use-4bit

# 5. 查看 output
ls outputs/minimax-m2-v1/
```

---

## 7. 联系人

- **数据准备 / Pipeline**: Franklin
- **GPU 资源 / 部署**: （待分配）
- **模型训练（这份文档）**: 白

有问题随时问。**先跑通 MiniMax-M2 的 PoC**，再考虑大模型。
