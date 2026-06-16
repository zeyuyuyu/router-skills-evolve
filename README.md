# Router + Skills Evolve

> **一句话**: Router 决定每道题用大模型还是小模型，Skills 从历史 traces 提炼可复用解题 procedure（喂给小模型），LLM 用这些数据持续微调；三件套（Skills + LLM + Router）联合迭代自主进化，越用越省钱。

> **职责划分（重要）**: **Router 独占路由**（per-prompt 决定 small/large）；**Skills 只负责提炼 procedure**（拼进小模型 prompt，不参与路由）。SkillBook 用单一全局 skill（`extract_signature` 恒返回 `"coding"`），不再按 cluster 细分。

已验证：**99% 准确率 + 省 83% 成本**（HumanEval 164 题真实实验）

---

## 系统设计（升级版）

本系统是一个带 Router 做课程采样、带 Skills 做 scaffold 增强的 **on-policy iterative distillation** 系统。大模型是 Teacher，小模型是 Student，每个 cycle 更新三件套（Skills + LLM + Router），越迭代越省钱。

### 核心 Pipeline（`scaling/run_full_pipeline.sh`）

```
Cycle N:

  Phase 1: Trace 收集（闭环入口）
    → 小模型：多轮 ReAct 修复（最多 HE_MAX_REPAIR_TURNS 次，error feedback loop）
              procedure prefix 注入（与 SFT 训练格式对齐）
    → 大模型：同样多轮修复，更高 oracle 质量 → 更好的 teacher 数据
    → Cycle ≥ 1：用上一轮产物闭环：
        small_model = cycle_{N-1}/grpo_adapter（优先）或 llm_adapter/checkpoint-best
        --router    = cycle_{N-1}/router/router.joblib
        --skillbook = cycle_{N-1}/skillbook.json

  Phase 2: Skills Evolve
    → carry over 上一轮 SkillBook，从当前 traces 增量更新
    → 单一全局 skill "coding"（统计 key 用规范化角色名 "small"/"large"）
    → LLM distiller（DISTILLER_MODEL）从所有成功轨迹提炼一份解题 procedure：
        problem type / key algorithm / step-by-step template / reusable snippet / pitfalls
    → procedure 是 Skills 的唯一产物，路由完全交给 Phase 4 的 router

  Phase 3a: LLM SFT
    → traces_to_sft.py 提取两类训练数据：
        teacher pairs: small 全轮失败 + large 成功 → 以大模型解为 target
        self-repair pairs: small 第1轮失败、后续轮自修成功 → 多轮对话链为 target
    → procedure 前置到 SFT prompt（推理格式与训练格式对齐）
    → HumanEval: train_small_model.py（LoRA SFT）
      Tau-2: tau2_train_wrapper.sh（FSDP2 + FA2）
    → cycle_N/llm_adapter/checkpoint-best

  Phase 3b: on-policy RL（HumanEval 专用，SKIP_GRPO=0）
    → 从 SFT checkpoint warm-start
    → K=GRPO_N_GENERATIONS 个 completion/prompt，test 执行 → 二值 reward（pass=1/fail=0）
    → advantage = (r - group_mean) / (group_std + ε)，无需 value network
    → 算法 GRPO_ALGO 可切：grpo（对称 clip）| dapo（非对称 clip + 动态采样）
    → cycle_N/grpo_adapter/（Phase 1 下一 cycle 优先读这个）

  Phase 4: Router 训练（独占路由）
    → train_router_simple.py：当前 traces 的【原始 prompt】→ TF-IDF + LogReg 二分类器
    → 特征用原始 prompt 不拼 procedure（单 skill 下 procedure 是常数前缀，零区分力；
      且 label small_success 已是带 procedure 跑出的结果，增益已隐式烘进 label）
    → cycle_N/router/router.joblib

  Phase 5: E2E Ablation（四臂）
    → large（always-large）/ skills（always-small + procedure）/ router / full（+LLM）
    → skills 臂不路由（单 skill），等价 always-small + procedure，是小模型基线；
      路由对比看 router / full 臂

  → 下一轮 Phase 1 读 grpo_adapter + router + skillbook（真闭环）
```

**顺序 Skills → LLM → Router（SLR）经 8-cycle 实验验证为最优**。

### 训练数据来源（on-policy distillation）

| 数据类型 | 来源 | 训练目标 |
|---------|------|---------|
| teacher pairs | small 全败，large 成功 | 学大模型解法 |
| self-repair pairs | small 第N轮自修成功（N≥2） | 学 error→fix 对话链 |
| GRPO on-policy | 实时生成 K 条，test 打分 | 直接优化 pass rate |

### 关键设计决策

- **单一全局 skill**：`extract_signature` 恒返回 `"coding"`。早期按"长度桶+关键词"分 20-30 cluster，但每簇样本太少、统计不可靠；合成一个 bucket 后样本量最大，procedure 从全部成功轨迹提炼，覆盖最广。
- **Router 独占路由，Skills 只管 procedure**：单 skill 下 `can_downgrade_to_small` 对所有题返回同一 verdict，会碾压 router 的 per-prompt 信号，故 `collect_traces._policy_decision` 不再用 skill verdict 覆盖路由（verdict 仅留作诊断字段 `policy_skill_verdict`）。路由全权交给学到的 router。
- **大小模型都走多轮修复**：减少不必要的大模型调用；大模型多轮也提升 oracle 数据质量
- **procedure 推理/训练格式对齐**：SFT、GRPO 和推理都前置 procedure（`procedure\n\n---\n\n{problem}`），消除 train/inference mismatch
- **RL 在 SFT 之后**：SFT warm-start 避免 RL 从随机策略开始的不稳定性
- **checkpoint 优先级**：grpo_adapter > llm_adapter/checkpoint-best > base model

### Phase 3b 算法：GRPO vs DAPO

Phase 3b 支持两种 on-policy RL 算法，用 `GRPO_ALGO` 一键切换。两者都基于 TRL 1.6 原生 `GRPOConfig`，无自定义 Trainer。

| | `GRPO_ALGO=grpo`（默认） | `GRPO_ALGO=dapo` |
|--|--|--|
| TRL `loss_type` | `grpo` | `dapo` |
| PPO clip | 对称 `clip(r, 1-ε, 1+ε)` | 非对称 `clip(r, 1-ε_low, 1+ε_high)` |
| loss normalization | 序列级 mean | token 级全局（梯度信号更稠密） |
| 动态采样 | 无 | 有：K 条 reward 全同的 group 置零（零方差→零梯度） |

**为什么 DAPO**（DeepSeek/ByteDance 2025）：HumanEval 里大量简单题（小模型必过）和极难题（必失败）会让整个 group 的 advantage=0，梯度浪费——动态采样跳过它们；非对称 clip 的高上界（ε_high）防止 entropy collapse，让模型能放大正确解的概率而不被小上界惩罚。

切换参数（env var）：

| 变量 | 默认 | 说明 |
|------|------|------|
| `GRPO_ALGO` | `grpo` | `grpo` \| `dapo` |
| `DAPO_CLIP_LOW` | `0.2` | PPO ε 下界（两算法通用） |
| `DAPO_CLIP_HIGH` | `0.5` | DAPO ε 上界（仅 dapo 生效；grpo 时强制 = 下界） |

每次跑的 `cycle_N/grpo_adapter/grpo_info.json` 记录 `algo` / `trl_loss_type` / `epsilon_low` / `epsilon_high` / `dapo_dynamic_sampling`，可直接对比。

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

HumanEval 与 tau2 共用同一个入口 `scaling/run_full_pipeline.sh --bench humaneval`（见下节"HumanEval 端到端跑法"）。HumanEval adapter（`experiments/scaling/benches/humaneval/adapter.py`）跑本地 code 模型 + pytest，无需 API key。

> 早期那套独立分步脚本（`run_evolve.py` / `extract_training_data.py` / `train_small_model_grpo.py` / `train_learnable_router.py` / `run_e2e_ablation.py` 等）已随单一 e2e pipeline 收敛而移除——统一走 `run_full_pipeline.sh`。

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
│   ├── traces.jsonl               # Phase 1: oracle small+large 多轮 traces
│   ├── skillbook.json             # Phase 2: SkillBook（含 LLM-distilled procedure）
│   ├── training_data.jsonl        # Phase 3a: SFT 数据（teacher + self-repair pairs）
│   ├── llm_adapter/               # Phase 3a: SFT LoRA 权重
│   ├── grpo_adapter/              # Phase 3b: GRPO LoRA 权重（优先于 llm_adapter）
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

### 真实跑（4 cycle，SFT + GRPO）

```bash
# 完整跑：SFT + GRPO（需 GPU，~4–8 小时/cycle）
bash scaling/run_full_pipeline.sh --bench humaneval --n-cycles 4

# 只跑 Skills + Router（无需 GPU）
SKIP_LLM=1 SKIP_GRPO=1 bash scaling/run_full_pipeline.sh --bench humaneval --n-cycles 4

# 只跑 SFT，跳过 GRPO
SKIP_GRPO=1 bash scaling/run_full_pipeline.sh --bench humaneval --n-cycles 4

# 调整 GRPO 超参
GRPO_N_GENERATIONS=4 GRPO_EPOCHS=2 bash scaling/run_full_pipeline.sh --bench humaneval

# 切换 RL 算法：GRPO（默认）vs DAPO
GRPO_ALGO=grpo bash scaling/run_full_pipeline.sh --bench humaneval --n-cycles 1
GRPO_ALGO=dapo DAPO_CLIP_HIGH=0.5 bash scaling/run_full_pipeline.sh --bench humaneval --n-cycles 1
# 对比：diff 两次跑的 cycle_*/grpo_adapter/grpo_info.json，
#       看 phase3b_grpo.log 里 [dapo] dynamic_sampling 行过滤了多少零方差 group
```

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

> ⚠️ 下表是**旧设计**（多 cluster signature + Skills 参与路由 + 独立 Base 臂）的历史数字。
> 当前代码已改为**单 skill + Router 独占路由**，四臂变为 `large / skills / router / full`，
> 其中 `skills`（always-small + procedure）取代了原来的独立 `Base` 臂。新设计的数字需重跑实验后更新。

| 系统（旧设计） | Routing Acc | Large F1 | Fallback | Cost vs always-large | Code pass |
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
├── src/                          # 核心代码（e2e pipeline 用到的）
│   ├── config.py                 # API、模型池、价格
│   ├── models.py                 # LLM 调用 + extract_code / run_humaneval_test
│   ├── skills.py                 # SkillBook 数据结构 + 单一全局 skill
│   └── train_plots.py            # 训练曲线绘制（SFT/GRPO 用）
│
├── experiments/
│   ├── train_small_model.py      # Phase 3a (HumanEval): SFT + LoRA；也提供 format_prompt
│   │
│   ├── scaling/                  # Scaling e2e pipeline（bench-agnostic）
│   │   ├── collect_traces.py     # Phase 1: 多轮 ReAct 收 traces（router 路由）
│   │   ├── traces_to_sft.py      # Phase 3a helper: traces → teacher + self-repair SFT 数据
│   │   ├── tau2_train_wrapper.sh # Phase 3a: tau2 SFT 框架（FSDP2+FA2）
│   │   ├── grpo_train_simple.py  # Phase 3b (HumanEval): on-policy RL，test-exec reward
│   │   ├── grpo_tau2_train.py    # Phase 3b (tau2): 轨迹级 GRPO/DAPO，reward = env passed
│   │   ├── grpo_core.py          # GRPO/DAPO advantage + update（两个 phase3b 共用）
│   │   ├── train_router_simple.py # Phase 4: 独占路由，原始 prompt → TF-IDF+LogReg
│   │   ├── run_e2e_ablation_simple.py # Phase 5: 四臂评估 large/skills/router/full
│   │   ├── aggregate_cycles.py   # Phase 6: 多 cycle 曲线 + 汇总表
│   │   ├── eval_checkpoints.py   # 工具: 对 pipeline 产出的 checkpoint 做评估
│   │   └── benches/
│   │       ├── tau2_bench/adapter.py  # τ²-bench 接口（remote agent API）
│   │       ├── humaneval/adapter.py   # HumanEval 接口（本地 code 模型 + pytest）
│   │       └── swe_bench/adapter.py   # SWE-Bench 接口（stub，待实现）
│   │
│   └── tau2_stage2/              # Colleague 的 LLM SFT 框架（FSDP2 + FA2）
│       ├── code/training/        # TRL SFTTrainer
│       ├── data_processed/       # τ²-bench 语料
│       └── docs/                 # 设计文档
│
├── scaling/
│   ├── run_full_pipeline.sh      # E2E 主入口：6 phases × N cycles（tau2 / humaneval）
│   └── README.md                 # Scaling 详细文档
│
├── notebooks/
│   └── router_skills_walkthrough.ipynb  # 动手拆解（无需 GPU，模拟 trace）
│
├── data/                         # HumanEval.jsonl、traces、skills
├── results/                      # 实验结果（gitignored）
├── docs/
│   ├── ARCHITECTURE.md           # 架构详解
│   ├── TRAINING.md               # LLM 训练指南
│   └── E2E_ABLATION_RESULTS.md   # 主分支实验数字
├── CLAUDE.md                     # Claude Code 操作速查 + 设计不变量
└── requirements.txt
```

---

## 谁负责什么

| 模块 | 负责人 | 代码位置 |
|------|--------|---------|
| Router (UncommonRoute) | 已有 | `UncommonRoute` repo |
| Skills 数据结构 + SkillBook | Franklin/Zeyu | `src/skills.py` |
| Router 训练（独占路由） | Zeyu | `experiments/scaling/train_router_simple.py` |
| Scaling Pipeline 编排 | Zeyu | `scaling/run_full_pipeline.sh` |
| τ²-bench Adapter | Zeyu | `experiments/scaling/benches/tau2_bench/` |
| HumanEval Adapter | Zeyu | `experiments/scaling/benches/humaneval/` |
| **LLM SFT 框架** | 白（colleague） | `experiments/tau2_stage2/` |
| SWE-Bench Adapter | Teammate（待做） | `experiments/scaling/benches/swe_bench/` |

---

## 进一步阅读

- [notebooks/router_skills_walkthrough.ipynb](notebooks/router_skills_walkthrough.ipynb) - **动手拆解**：signature/skills/SFT/GRPO/router 逐步跑通（无需 GPU，模拟 trace）
- [scaling/README.md](scaling/README.md) - Tau-2 / SWE-Bench scaling 完整文档
- [CLAUDE.md](CLAUDE.md) - Claude Code 操作速查 + 设计不变量
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - 架构详解
- [docs/TRAINING.md](docs/TRAINING.md) - LLM 训练指南
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
