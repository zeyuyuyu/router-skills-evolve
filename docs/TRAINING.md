# LLM 训练指南（Phase 3）

> 这份文档讲 pipeline 里**小模型怎么训**：训什么、数据哪来、怎么跑、怎么调。
> LLM 训练是 `scripts/run_full_pipeline.sh` 的 **Phase 3a（SFT）+ Phase 3b（GRPO/DAPO）**，
> 不是独立脚本——下面命令都通过主入口跑。

---

## 1. 训什么、为什么

每个 cycle 里小模型（Student）从大模型（Teacher）和自己的成功轨迹学习，目标是把
原本必须升级到大模型的难题也学会，让 router 能更多地走小模型 → 省钱、保质量。

- **Phase 3a SFT**：模仿正确答案（teacher 蒸馏 + 自修复）。快、稳、数据少也能用。
- **Phase 3b GRPO/DAPO**：on-policy RL，用**可验证 reward**（HumanEval 跑 pytest / tau2 看 env passed）
  直接优化通过率。在 SFT checkpoint 之上 warm-start。

两阶段的 prompt 都前置 procedure：`f"{procedure}\n\n---\n\n{problem}"`，和推理时一致。

---

## 2. 数据从哪来（无需手动准备）

Phase 1 收的 `traces.jsonl` → `traces_to_sft.py` 自动产出两类 SFT 样本：

| 类型 | 选择规则 | target |
|------|---------|--------|
| teacher pairs | 小模型全轮失败 **且** 大模型成功 | 大模型的正确代码 |
| self-repair pairs | 小模型第 1 轮失败、后续轮自修成功 | 多轮"错误→修正"对话链 |

GRPO（Phase 3b）不需要标签代码，只需可验证 reward，所以直接在 bench 任务上 rollout。

---

## 3. 怎么跑

```bash
# 完整：Phase 3a SFT + Phase 3b GRPO（需要 GPU）
bash scripts/run_full_pipeline.sh --bench humaneval --n-cycles 4

# 只 SFT、跳过 GRPO
SKIP_GRPO=1 bash scripts/run_full_pipeline.sh --bench humaneval

# 完全跳过 LLM 训练（只跑 Skills + Router，无需 GPU）
SKIP_LLM=1 SKIP_GRPO=1 bash scripts/run_full_pipeline.sh --bench humaneval

# tau2（SFT 走 tau2_train_wrapper.sh / FSDP2+FA2；GRPO 默认关，SKIP_GRPO=0 开启）
bash scripts/run_full_pipeline.sh --bench tau2_bench --n-cycles 4
```

产物：`cycle_N/llm_adapter/checkpoint-best`（SFT）、`cycle_N/grpo_adapter/`（GRPO）。
下一轮 Phase 1 的小模型按 **`grpo_adapter/` > `llm_adapter/checkpoint-best` > 基座** 优先级加载。

---

## 4. 默认模型与资源

| bench | small（被训） | large（teacher） | 训练 GPU |
|-------|--------------|------------------|---------|
| humaneval | Qwen2.5-Coder-1.5B-Instruct | Qwen2.5-Coder-3B-Instruct | 1.5B：1–2×A100/A800 |
| tau2_bench | Qwen3 系列（见 `MODEL_SWEEP` YAML） | gpt-5.4（API） | 4B：2×；9B：4×；35B-A3B：8× |

只跑 trace 收集 + Router 训练（`SKIP_LLM=1 SKIP_GRPO=1`）不需要 GPU。

---

## 5. 关键超参（env var）

| 变量 | 默认 | 说明 |
|------|------|------|
| `SCALING_NUM_TRAIN_EPOCHS` | 2 | SFT epochs（数据少→3–5；多→1–2，防过拟合） |
| `GRPO_N_GENERATIONS` | 8 | GRPO 每 prompt 采样数 K（组内算 advantage） |
| `GRPO_EPOCHS` | 1 | GRPO epochs |
| `GRPO_LR` | 5e-6 | GRPO 学习率 |
| `GRPO_BETA` | 0.04 | KL 惩罚系数 |
| `GRPO_ALGO` | grpo | `grpo` \| `dapo`（见 TRAINING_METHODS.md） |
| `GRPO_TEMPERATURE` | 1.0 | **rollout 温度，必须 >0**（贪心会让 K 条坍缩、零梯度）；训练用 0.7–1.0，eval 用贪心 |

LoRA：SFT/GRPO 都用 LoRA（默认 r=16）。单卡显存紧→开 4-bit / 减 batch / 减 `max_seq_length`。

---

## 6. 常见问题

**数据太少？** 多跑几个 cycle（闭环会持续积累 hard task），或多挑几个 tau2 domain（`TAU2_DOMAINS=retail,telecom,airline`）。

**GPU OOM？** 减 `per_device_train_batch_size` + 加 grad-accum；开 4-bit；降 `max_seq_length`；flash-attn 编译用 `MAX_JOBS=4`。

**怎么看训得好不好？** 看 `results/<exp>/curve.png`（`aggregate_cycles.py` 生成）的 task_pass 跨 cycle 走势，以及 `cycle_N/grpo_adapter/grpo_info.json`（algo/K/lr/beta 记录）。

**训完怎么用？** 产出是 LoRA adapter；闭环里由 `vllm_serve.sh` 起成 OpenAI-compatible server（cycle≥1 的 Phase 1 自动 launch/kill），以 `openai/evol-llm-student` 作为小模型。

---

## 7. 历史结论

Qwen2.5-Coder-1.5B + GRPO：47/100 → 49/100（+2pt）。3B/7B GRPO 出现退化，**单纯放大模型不够**——
增益主要来自 Skills + Router 联合迭代，LLM 训练在小模型上提供边际收益。详见
[E2E_ABLATION_RESULTS.md](E2E_ABLATION_RESULTS.md)。
