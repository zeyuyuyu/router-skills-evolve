# CWY：30B 正式运行计划

## 目标

只跑 30B 大模型的 tau2 正式实验。

不跑 2B / 4B / 9B。
不把 35B-A3B 当成 30B。
如果当前仓库没有精确 30B 配置，先补齐 30B 配置，再正式启动。

## 训练方式

不是从零预训练。

这次是拿现成的 30B 模型，在 `tau2_stage2` 框架里做 SFT / 微调，然后进入 scaling pipeline：

1. 收集 tau2 traces
2. 生成 / 更新 SkillBook
3. 训练 30B SFT checkpoint
4. 训练 router
5. 跑 E2E ablation
6. 保存结果到 `results/`

## 代码位置

最新代码：

```bash
/root/cwy/projects/evol/router-skills-evolve
```

主流程：

```bash
scaling/run_full_pipeline.sh
```

训练包装：

```bash
experiments/scaling/tau2_train_wrapper.sh
```

模型配置目录：

```bash
experiments/tau2_stage2/code/training/configs/runs/
```

## 先确认 30B 配置

已确认 CEPHFS 上有 30B 模型：

```text
/mnt/cephfs/home/dan.qiao/models/Qwen3-30B-A3B-Base
```

检查结果：

- 大小：57G
- safetensors：16 个
- `config.json`：存在
- `tokenizer.json`：存在
- `tokenizer_config.json`：存在
- `generation_config.json`：存在
- `model.safetensors.index.json`：存在

结论：不需要下载模型，直接用 CEPHFS 上的 30B 模型。

检查是否已有 30B run config：

```bash
cd /root/cwy/projects/evol/router-skills-evolve
ls experiments/tau2_stage2/code/training/configs/runs/
rg -n "30B|30b" experiments/tau2_stage2/code/training/configs/runs/
```

如果没有 30B 配置，需要新增一个 30B run config，例如：

```text
XX_<30b_model_name>_273.yaml
```

配置里必须明确：

- 30B 模型名
- revision
- output_dir
- FSDP / tensor parallel 配置
- max_seq_length
- batch size
- checkpoint 保存策略

## GPU 机器检查

已检查机器：

```text
10.100.0.53:27525
```

状态：

- 8 张 NVIDIA H200
- GPU 显存占用约 1MB
- GPU 利用率 0%
- 没有训练 / eval / vLLM 进程占卡
- CEPHFS 已挂载：`/mnt/cephfs`

正式跑前确认：

```bash
nvidia-smi
df -h
pgrep -af "python|torchrun|accelerate|vllm|train|eval"
```

要求：

- 8 张 GPU 可用
- 没有旧训练 / 旧 eval 占卡
- 磁盘足够放 30B 权重和 checkpoint
- Hugging Face 能下载 30B 模型
- CommonStack API 可用

## mock smoke

先验证 pipeline 调度，不用 GPU，不花 API：

```bash
cd /root/cwy/projects/evol/router-skills-evolve
MODEL_SWEEP=<30B_RUN_CONFIG> \
N_CYCLES=1 \
bash scaling/run_full_pipeline.sh --mock --skip-llm
```

这里 `<30B_RUN_CONFIG>` 必须换成真实 30B 配置名。

## tau2 adapter sanity

正式训练前先跑 3 条真实 tau2，确认 adapter 没坏：

```bash
cd /root/cwy/projects/evol/router-skills-evolve
SCALING_MOCK=0 \
TAU2_DOMAIN=retail \
python experiments/scaling/collect_traces.py \
  --bench tau2_bench \
  --n-tasks 3 \
  --small-model <30B_MODEL_OR_ENDPOINT> \
  --large-model openai/gpt-5.4-2026-03-05 \
  --out /tmp/cwy_tau2_30b_sanity.jsonl
```

通过标准：

- 有 3 行结果
- `total_cost` 非 0
- `final_model` 非空
- `large_completion` 非空
- 不是全 0

## 正式跑 30B

```bash
cd /root/cwy/projects/evol/router-skills-evolve
MODEL_SWEEP=<30B_RUN_CONFIG> \
N_CYCLES=2 \
bash scaling/run_full_pipeline.sh
```

## 输出物

需要保存：

- 30B checkpoint
- traces
- SkillBook
- router
- E2E ablation summary
- cost 记录
- wall-clock 时间
- 最终结果目录

## 当前阻塞点

无。

## 当前状态

30B SFT 正式训练已完成，训练后 tau2 / step-budget 正式评测也已完成。

机器：

```text
10.100.0.53:27525
```

训练日志：

```text
/root/cwy/projects/evol/router-skills-evolve/experiments/tau2_stage2/train_outputs/11_qwen3_30b_a3b_273/cwy_train_20260605_225120_restart2.log
```

训练结果：

- 状态：`done`
- 总步数：`445/445`
- 训练耗时：约 `3小时58分钟`
- 最终 `train_loss`：`0.702`
- 最终 `eval_loss`：`0.3652`
- 最终 `eval_mean_token_accuracy`：`0.9023`
- 最优 checkpoint：`train_outputs/11_qwen3_30b_a3b_273/checkpoint-best -> checkpoint-356`
- 当前 GPU：8 张 H200 已空闲

评测计划：

- 只评测已训练好的 30B：`11_qwen3_30b_a3b_273`
- checkpoint：`train_outputs/11_qwen3_30b_a3b_273/checkpoint-best`
- 使用 tau2 step-budget eval 的单 target harness
- seed：`primary`
- Method B：`352` 个 step 替换 rollout
- Method A：`33` 个端到端 rollout
- 总评测 rollout：`385`
- 可断点续跑：已完成的 rollout 会跳过
- baseline 轨迹：已从旧 GPU 同步到 H200，数量 `63`

baseline 含义：评测里的“基准/对照组”，用于和训练后的 30B 模型结果对比，不是要训练的新模型。

评测最终状态：

- 评测已跑完
- 远端 H200 进程：`10.100.0.53:27525`
- vLLM：评测完成后已停止，GPU 已释放
- vLLM 模型服务名：`evol-llm-student`
- vLLM 上下文：`131072`
- CommonStack：H200 无公网，已通过 CPU 机器 SSH 反向隧道转发
- H200 API 地址：`https://api.commonstack.ai:18443/v1`
- 第一个 rollout 已完成：`airline__0__seed300__methodB_step0`
- 最后完成 rollout：`telecom__svc_H_003__seed300__methodA`
- 最终进度：`385/385`
- 最终通过：`45/385`
- 总通过率：`11.69%`
- 评测成本：约 `$10.44`（按 raw eval 单条 cost 汇总）
- LLM calls：`1732`
- prompt tokens：`9,672,472`
- completion tokens：`578,751`
- 评测耗时：约 `3小时36分钟`
- 当前 GPU：8 张 H200 已释放，显存约 `4MiB/卡`
- 输出目录：`experiments/tau2_stage2/step_budget_outputs/11_qwen3_30b_a3b_273/`

评测拆分：

| 维度 | 数量 | 通过 | 通过率 |
| --- | ---: | ---: | ---: |
| 总计 | 385 | 45 | 11.69% |
| Method B | 352 | 45 | 12.78% |
| Method A | 33 | 0 | 0.00% |
| airline | 93 | 11 | 11.83% |
| retail | 205 | 28 | 13.66% |
| telecom | 87 | 6 | 6.90% |

结论：

- 训练本身正常收敛，`eval_mean_token_accuracy=0.9023`。
- tau2 正式评测结果偏低，尤其 Method A 端到端 rollout 是 `0/33`。
- Method B 有少量成功，说明模型在局部 step 替换场景里能工作一部分，但完整任务完成能力还不稳定。
- 日志里看到多次工具调用格式混乱、自然语言和工具结果混在一起、重复符号/乱码、token limit 截断，这些会直接拉低 tau2 pass rate。
- 本次结果可以作为训练后 30B 的正式 baseline/result，不是 mock，不是 smoke。

修复记录：

1. 新增 30B run config：`experiments/tau2_stage2/code/training/configs/runs/11_qwen3_30b_a3b_273.yaml`
2. 新增 30B FSDP 配置：`experiments/tau2_stage2/code/training/configs/accelerate_fsdp2_qwen3_moe.yaml`
3. 30B 实际模型类是 `Qwen3Moe*`，不是原 35B 配置里的 `Qwen3_5Moe*`
4. 使用实际存在的 wrap 类：

```text
Qwen3MoeAttention,Qwen3MoeExperts,Qwen3MoeSparseMoeBlock,Qwen3MoeDecoderLayer
```
