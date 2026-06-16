# Router + Skills Evolve

> **一句话**: Router 把请求路由到合适的 LLM，Skills 通过历史 traces 学习"哪类题该用哪个模型"，三件套（Skills + LLM + Router）联合迭代自主进化，越用越省钱。

已验证：**99% 准确率 + 省 83% 成本**（HumanEval 164 题真实实验）

---

## 核心 Pipeline（`scaling/run_full_pipeline.sh`）

```
Cycle N:

  Phase 1: Trace 收集（闭环入口）
    → 跑所有 bench 任务，记 (task, small_pass, large_pass, cost)
    → Cycle ≥ 1：用上一轮产物闭环：
        small_model = cycle_{N-1}/llm_adapter/checkpoint-best（vLLM serve）
        --router    = cycle_{N-1}/router/router.joblib
        --skillbook = cycle_{N-1}/skillbook.json

  Phase 2: Skills Evolve
    → carry over 上一轮 SkillBook，从当前 traces 增量更新
    → 统计 key 用规范化角色名 "small"/"large"（不是原始 model ID，否则每轮都变）
    → 对每个 signature cluster 提炼 procedure（成功轨迹 → 可复用代码片段/tool序列）

  Phase 3: LLM 训练（SFT）
    → traces_to_sft.py 从当前 traces 提难题（small 失败 + large 成功）→ training_data.jsonl
    → 若 skillbook 已有，把 distilled procedure 前置到 SFT prompt（procedure → 小模型）
    → tau2_train_wrapper.sh（FSDP2 + FA2）→ cycle_N/llm_adapter/checkpoint-best

  Phase 4: Router 训练
    → train_router_simple.py：当前 traces → TF-IDF + LightGBM 二分类器
    → cycle_N/router/router.joblib

  Phase 5: E2E Ablation 评估
    → 4 路对比：Base / +Skills / +Router / Full（+LLM）

  → 下一轮 Phase 1 直接读以上三件产物（真闭环）
```

**顺序 Skills → LLM → Router（SLR）经 8-cycle 实验验证为最优**（MERA 论文 + main 分支 5/20 实验）。

> **注意**：`experiments/run_joint_evolver.py` 是早期原型，每 cycle 重置 SkillBook、不加载上一轮 router/LLM、LLM 训的是固定 MBPP 数据，**不是真闭环**，不用于正式实验。

---

## 5 分钟上手

### 1. 环境准备

```bash
# Python 3.10+
pip install openai requests

# 拿 CommonStack API key (团队群问)
export COMMONSTACK_API_KEY="ak-f53d..."

# 中国大陆: HuggingFace 镜像 (下模型权重要用)
export HF_ENDPOINT=https://hf-mirror.com
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

---

## HumanEval 跑法

HumanEval 现已支持端到端闭环迭代，可直接用 `scaling/run_full_pipeline.sh`（见下节"HumanEval 端到端跑法"）。下面是手动分步跑法，便于 debug 或单次实验：

```bash
# 收集 traces（内部 rounds 间 SkillBook 有效果，但不跨 cycle 闭环）
python3 experiments/run_evolve.py --n 164 --rounds 4

# 从 traces 提 SFT 数据（HumanEval 专用）
python3 experiments/extract_training_data.py \
    --traces "data/traces/*.jsonl" \
    --output data/training_data.jsonl

# 训练小模型 —— 方案 A: SFT
python3 experiments/train_small_model.py \
    --data data/training_data.jsonl \
    --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --epochs 3 --use-4bit

# 训练小模型 —— 方案 B: GRPO（main 分支验证 +2pt）
python3 experiments/train_small_model_grpo.py \
    --tasks data/HumanEval.jsonl \
    --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --n-generations 4 --use-4bit

# 训练 Router（HumanEval-coupled）
python3 experiments/train_learnable_router.py \
    --traces "data/traces/*.jsonl" \
    --out src/learned_router/router.joblib

# 4 路 ablation 评估
python3 experiments/run_e2e_ablation.py \
    --traces "data/traces/*.jsonl" \
    --router src/learned_router/router.joblib
```

**多轮闭环迭代请用 `scaling/run_full_pipeline.sh`**（见下节"HumanEval 端到端跑法"）。HumanEval adapter（`experiments/scaling/benches/humaneval/adapter.py`）已实现，可直接接入闭环 pipeline。

---

## Tau-2 端到端跑法（主文件）

**`scaling/run_full_pipeline.sh` 是正式实验的主入口**，支持真闭环多轮迭代。Tau-2 bench 是 multi-turn agentic 场景（客服 agent）。

### 前置需求

| 资源 | 要求 |
|------|------|
| 训练 GPU | 8× H200 或 8× A800 80GB |
| CUDA | 12.x（A800）或 13.0（H200） |
| 磁盘 | ≥ 500 GB |
| Python | 3.12（conda） |
| 额外依赖 | `scikit-learn lightgbm sentence-transformers`（router/skills） + `experiments/tau2_stage2/code/training/requirements.txt`（LLM 训练） |

### 环境变量

```bash
export OPENAI_API_KEY=sk-...          # Phase 1 large model + tau2 eval judge
export HF_TOKEN=hf_...                # Qwen3 系列下载（35B-A3B 必须）
export BUNDLE_ROOT=$(pwd)/experiments/tau2_stage2
export EXPERIMENT_NAME=scaling_$(date -u +%Y%m%d_%H%M%S)
export BENCH=tau2_bench
export MODEL_SWEEP=05_qwen3_5_4b_273  # tau2_stage2/runs/ 下的 YAML 名（去掉 .yaml）
export N_CYCLES=4                      # MERA 推荐 4，main 分支跑过 8
```

### Smoke 测试（无 GPU，~10 分钟）

```bash
bash scaling/run_full_pipeline.sh --smoke --mock
```

验证标准：`results/scaling_*/cycle_0/` 下有 `traces.jsonl`、`skillbook.json`、`router/router.joblib`、`e2e_ablation_summary.json`，且 `full.routing_acc > base.routing_acc`。

### 真实跑（4 cycle，~24–48 小时）

```bash
bash scaling/run_full_pipeline.sh --n-cycles 4
```

常用 flag：

| Flag | 默认 | 说明 |
|------|------|------|
| `--smoke` | off | 30 任务 × 1 cycle × smoke_2b 模型 |
| `--mock` | off | 不调 API/GPU，全用 mock 数据验证 orchestration |
| `--skip-llm` | off | 跳过 Phase 3 LLM 训练（只跑 Skills + Router） |
| `--n-cycles N` | `$N_CYCLES` | 迭代轮数 |
| `--schedule SLR` | `SLR` | Skills→LLM→Router 顺序（ablation 时换） |
| `--resume <cycle>` | none | 从某个 cycle 断点继续 |
| `--dry-run` | off | 只打印计划，不真跑 |

### 输出结构

```
results/$EXPERIMENT_NAME/
├── MANIFEST.json                  # 各 phase 耗时 + 状态
├── config_snapshot.json           # 复现用环境变量快照
├── cycle_0/
│   ├── traces.jsonl               # Phase 1: oracle small+large 结果
│   ├── skillbook.json             # Phase 2: SkillBook
│   ├── llm_adapter/               # Phase 3: LoRA 权重
│   ├── router/router.joblib       # Phase 4: 分类器
│   └── e2e_ablation_summary.json  # Phase 5: 4 路指标
├── cycle_1/ ... cycle_N/
├── curve.png                      # 多轮迭代曲线（routing acc / code pass / cost）
└── final_ablation_table.md        # 论文用汇总表
```

### 分步手动跑 Tau-2（便于 debug）

```bash
OUT=results/$EXPERIMENT_NAME/cycle_0

# Phase 1: 收 traces（cycle 0，直接用默认 small/large model）
python3 experiments/scaling/collect_traces.py \
    --bench tau2_bench \
    --n-tasks 848 \
    --small-model deepseek/deepseek-v3.2 \
    --large-model openai/gpt-5.4-2026-03-05 \
    --cycle 0 --split train \
    --out $OUT/traces.jsonl

# Phase 1 cycle ≥ 1（闭环）：小模型换成上一轮 LoRA，传入 router + skillbook
# 上一轮 LLM 需要先用 vllm_serve.sh 起起来，再用 --small-model openai/evol-llm-student
python3 experiments/scaling/collect_traces.py \
    --bench tau2_bench --n-tasks 848 --cycle 1 --split train \
    --small-model openai/evol-llm-student \
    --large-model openai/gpt-5.4-2026-03-05 \
    --router results/$EXPERIMENT_NAME/cycle_0/router/router.joblib \
    --skillbook results/$EXPERIMENT_NAME/cycle_0/skillbook.json \
    --out results/$EXPERIMENT_NAME/cycle_1/traces.jsonl

# Phase 2: SkillBook（内嵌在 run_full_pipeline.sh 里的 Python 片段，无独立脚本）
# 直接跑 shell 的 Phase 2：统计 key 用 "small"/"large" 规范角色名，
# 并对每个 cluster 提炼 procedure（heuristic，从成功 completion 抽代码片段）

# Phase 3: traces → SFT 数据，再训 LLM
python3 experiments/scaling/traces_to_sft.py \
    --traces $OUT/traces.jsonl \
    --output $OUT/training_data.jsonl \
    --skillbook $OUT/skillbook.json     # procedure 前置到 SFT prompt

TRAINING_DATA=$OUT/training_data.jsonl \
TRAIN_OUTPUT_DIR=$OUT/llm_adapter \
RUN_CONFIG=$MODEL_SWEEP \
BUNDLE_ROOT=$BUNDLE_ROOT \
MODE=scaling_traces \
    bash experiments/scaling/tau2_train_wrapper.sh

# Phase 4: Router 训练（bench-agnostic，直接读 trace 里的 prompt 字段）
python3 experiments/scaling/train_router_simple.py \
    --traces $OUT/traces.jsonl \
    --output-dir $OUT/router

# Phase 5: E2E Ablation（4 路：Base / +Skills / +Router / Full）
python3 experiments/scaling/run_e2e_ablation_simple.py \
    --traces $OUT/traces.jsonl \
    --skillbook $OUT/skillbook.json \
    --router-dir $OUT/router \
    --router-threshold 0.5 \
    --output $OUT/e2e_ablation_summary.json

# Phase 6: 汇总多 cycle 曲线
python3 experiments/scaling/aggregate_cycles.py \
    --experiment-dir results/$EXPERIMENT_NAME \
    --n-cycles 4 \
    --output-md results/$EXPERIMENT_NAME/final_ablation_table.md \
    --output-png results/$EXPERIMENT_NAME/curve.png
```

---

## HumanEval 端到端跑法（闭环）

HumanEval adapter 已实现（`experiments/scaling/benches/humaneval/adapter.py`），可直接用主 pipeline 跑多轮闭环迭代。

### 环境变量

```bash
export OPENAI_API_KEY=sk-...          # large model API key
export EXPERIMENT_NAME=humaneval_$(date -u +%Y%m%d_%H%M%S)
export BENCH=humaneval
export N_CYCLES=4
```

### Smoke 测试（无 GPU，~5 分钟）

```bash
bash scaling/run_full_pipeline.sh --bench humaneval --smoke --mock
```

### 真实跑（4 cycle）

```bash
bash scaling/run_full_pipeline.sh --bench humaneval --n-cycles 4
```

> `--skip-llm` 可跳过 Phase 3 只跑 Skills + Router，无需 GPU。

---

## 踩坑预警

1. **Phase 2 统计 key 必须是规范角色名**：SkillBook 的 `stats` 用 `"small"/"large"` 做 key，不能用原始 model ID（每轮 model ID 会变，会导致 `can_downgrade_to_small` 永远查不到数据）。`run_full_pipeline.sh` 已经处理好了，手动跑时注意。

2. **Cycle ≥ 1 的 Phase 1 需要先 serve 上一轮 LLM**：`checkpoint-best` 要用 `vllm_serve.sh` 起成 OpenAI-compatible server（默认端口 8050），再以 `openai/evol-llm-student` 为 small_model。`run_full_pipeline.sh` 会自动 launch/kill vLLM 进程，手动跑时要手动管理。

3. **Qwen3 思考模式**: `enable_thinking=False`——tau2_stage2 corpus 无 CoT，开了会 OOD

4. **flash-attn 编译**: `MAX_JOBS=4`，不要更高，否则 12–15 GB/job 会 OOM

5. **A800 vs H200**: A800 → `max_seq_length=16384`；H200 → 可上 `max_seq_length=32768`

6. **shepherd 文件污染**: `run_full_pipeline.sh` 的 preflight 会自动清理；手动跑时先 `rm -rf shepherd* .shepherd*`

---

## 已验证结果

### HumanEval 164 题（main 分支，8×A800）

| 策略 | 成功率 | 成本 | vs GPT-5.4 |
|------|--------|------|-----------|
| Pure GPT-5.4 | 95% | $0.37 | 100% |
| Router Only | 74% | $0.15 | 42% |
| Router + Fallback | 98% | $0.27 | 73% |
| **Router + Skills Evolve** | **99%** | **$0.06** | **17%** |

### E2E Ablation（HumanEval，4 路对比）

| 系统 | Routing Acc | Large F1 | Fallback | Cost vs always-large | Code pass |
|------|---:|---:|---:|---:|---:|
| Base (always-small + fallback) | 68.28% | 0% | 31.72% | 33.54% | 47/100 |
| + Skills evolve | 69.46% | 24.93% | 26.65% | 37.27% | 47/100 |
| + Learned router | **93.04%** | **89.48%** | **2.12%** | 37.75% | 47/100 |
| **Full (+ LLM GRPO)** | **93.04%** | **89.48%** | **2.12%** | 37.75% | **49/100** |

LLM 训练结论：Qwen2.5-Coder-1.5B + GRPO: 47/100 → 49/100（+2pt）。3B/7B GRPO 退化，**scaling alone 不够**。

---

## 项目结构

```
router-skills-evolve/
├── src/                          # 核心代码
│   ├── config.py                 # API、模型池（48 个）、价格
│   ├── models.py                 # LLM 调用封装
│   ├── skills.py                 # SkillBook 数据结构 + 学习
│   ├── router.py                 # Router + Skills 路由决策
│   └── learned_router/           # Learnable router（BERT/TF-IDF + LightGBM）
│       ├── data.py               # 从 traces 构建训练集
│       ├── model.py              # BERT 路由模型
│       ├── policy.py             # 路由决策逻辑
│       └── router.py             # Router 封装
│
├── experiments/                  # 运行脚本
│   ├── run_evolve.py             # HumanEval: Skills evolve 主实验
│   ├── run_joint_evolver.py      # HumanEval: Skills + Router + LLM 联合迭代
│   ├── run_e2e_ablation.py       # HumanEval: 4 路 ablation（HumanEval 版）
│   ├── extract_training_data.py  # traces → SFT 数据
│   ├── extract_dpo_data.py       # traces → DPO 数据
│   ├── train_small_model.py      # SFT + LoRA
│   ├── train_small_model_dpo.py  # DPO + LoRA
│   ├── train_small_model_grpo.py # GRPO（真 RL）
│   ├── train_learnable_router.py # Router 训练（HumanEval 版）
│   ├── evaluate_finetuned_model.py
│   │
│   ├── scaling/                  # Scaling pipeline（bench-agnostic）
│   │   ├── collect_traces.py     # Phase 1: 并行收 traces
│   │   ├── traces_to_sft.py      # Phase 3 helper: traces → LLM 训练数据
│   │   ├── tau2_train_wrapper.sh # Phase 3: 调 tau2_stage2 SFT 框架
│   │   ├── train_router_simple.py # Phase 4: bench-agnostic Router 训练
│   │   ├── run_e2e_ablation_simple.py # Phase 5: bench-agnostic 4 路评估
│   │   ├── aggregate_cycles.py   # Phase 6: 多 cycle 曲线 + 汇总表
│   │   └── benches/
│   │       ├── tau2_bench/adapter.py  # τ²-bench 接口（已接通）
│   │       ├── humaneval/adapter.py   # HumanEval 接口（已接通）
│   │       └── swe_bench/adapter.py   # SWE-Bench 接口（stub，待实现）
│   │
│   └── tau2_stage2/              # Colleague 的 LLM SFT 框架（FSDP2 + FA2）
│       ├── code/training/        # TRL SFTTrainer，75 个单测全过
│       ├── data_processed/       # τ²-bench 语料（6413 train + 394 val）
│       └── docs/                 # 设计文档
│
├── scaling/
│   ├── run_full_pipeline.sh      # Tau-2 E2E 入口：6 phases × N cycles
│   └── README.md                 # Scaling 详细文档
│
├── data/
│   ├── HumanEval.jsonl           # 164 题
│   ├── training_data.jsonl       # 提取好的 SFT 数据
│   ├── traces/                   # 收集的 traces
│   └── skills/                   # 学到的 SkillBook（JSON）
│
├── results/                      # 实验结果
├── docs/
│   ├── ARCHITECTURE.md           # 架构详解
│   ├── TRAINING.md               # LLM 训练指南
│   ├── TRAINING_METHODS.md       # SFT vs DPO vs GRPO 对比
│   ├── LEARNABLE_ROUTER.md       # Router 训练详解
│   ├── E2E_ABLATION_RESULTS.md   # 主分支实验数字
│   └── JOINT_EVOLVER.md          # run_joint_evolver 文档
└── requirements.txt
```

---

## 谁负责什么

| 模块 | 负责人 | 代码位置 |
|------|--------|---------|
| Router (UncommonRoute) | 已有 | `UncommonRoute` repo |
| Skills 数据结构 + SkillBook | Franklin/Zeyu | `src/skills.py` |
| 路由决策 | Franklin/Zeyu | `src/router.py` |
| Learnable Router 训练 | Zeyu | `experiments/scaling/train_router_simple.py` |
| Scaling Pipeline 编排 | Zeyu | `scaling/run_full_pipeline.sh` |
| τ²-bench Adapter | Zeyu | `experiments/scaling/benches/tau2_bench/` |
| HumanEval Adapter | Zeyu | `experiments/scaling/benches/humaneval/` |
| **LLM SFT 框架** | 白（colleague） | `experiments/tau2_stage2/` |
| SWE-Bench Adapter | Teammate（待做） | `experiments/scaling/benches/swe_bench/` |

---

## 进一步阅读

- [scaling/README.md](scaling/README.md) - Tau-2 / SWE-Bench scaling 完整文档
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - 架构详解
- [docs/TRAINING.md](docs/TRAINING.md) - LLM 训练指南（白看这个）
- [docs/TRAINING_METHODS.md](docs/TRAINING_METHODS.md) - SFT vs DPO vs GRPO 对比
- [docs/E2E_ABLATION_RESULTS.md](docs/E2E_ABLATION_RESULTS.md) - 主分支实验数字

---

## FAQ

**Q: HumanEval 有没有完整的闭环 pipeline？**  
A: 有。`experiments/scaling/benches/humaneval/adapter.py` 已实现 `load_tasks()` 和 `run_task_pair()`，可直接用 `bash scaling/run_full_pipeline.sh --bench humaneval --n-cycles 4` 跑端到端闭环迭代。

**Q: 不训 LLM 只跑 Skills + Router 可以吗？**  
A: 可以，加 `--skip-llm` 或设 `SKIP_LLM=1` 跳过 Phase 3。Router 准确率 93%，cost 节省也显著（见上方 ablation 表）。

**Q: 为什么实验只用 2 个模型？**  
A: 控制变量。架构支持 N 个，生产 UncommonRoute 已支持 48 个。

**Q: 需要什么 GPU？**  
A: LLM 训练：1.5B–4B 用 2×A100；9B 用 4×A100；35B-A3B 用 8×A100/A800。只跑 trace 收集 + Router 训练不需要 GPU。

**Q: 多轮迭代收益递减怎么看？**  
A: 看 `curve.png`（aggregate_cycles.py 生成）。main 分支 8-cycle 实验 routing_acc 从 cycle 3 起基本平台，Skills+Router 联合迭代是主要增益来源，LLM 训练在小模型（1.5B）上有增益，大模型（3B/7B）GRPO 有退化。
