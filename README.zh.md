# router-skills-evolve

一个**自我进化**的 LLM 服务系统，从生产 trace 同时优化三个正交组件：

1. **Router（路由器）** — 一个可训练的 BERT-style 分类器，把每条 prompt 路由到便宜或贵的 LLM
2. **SkillBook（技能簿）** — 在线频率表，按 prompt 签名追踪每个模型在每类题上的成功率
3. **LLM adapter（小模型微调）** — 在便宜 LLM 上持续训练的 LoRA，专攻路由器目前还在送给贵模型的题

三者在一个 cycle 里互相喂养：

```
    SkillBook (在线统计)
           │
           ▼
    LLM continual GRPO  ── traces ──── Learnable router (BERT-tiny)
           │                                   │
           ▼                                   ▼
       per-task pass/fail               routing accuracy / fallback
                  │       │       │
                  ▼       ▼       ▼
                Joint evaluation
```

单次运行用 `experiments/run_joint_evolver.py`；7×24 自主运行用 `auto_research/` 框架。

## 快速上手

```bash
# 环境
pip install -r requirements.txt
export COMMONSTACK_API_KEY="..."        # 调 LLM 用
export HF_ENDPOINT=https://hf-mirror.com  # 国内镜像（可选）

# clone + smoke run
git clone https://github.com/zeyuyuyu/router-skills-evolve
cd router-skills-evolve
python3 experiments/run_evolve.py --n 30 --rounds 3
```

## 目录结构

```
router-skills-evolve/
├── src/                    # 核心：SkillBook、RouterWithSkills、models.py
├── experiments/            # 单次训练 + 评估脚本
│   ├── run_evolve.py                       # SkillBook + trace 收集循环
│   ├── run_joint_evolver.py                # 三条线 1 cycle joint
│   ├── run_e2e_ablation.py                 # 离线 ablation 报告
│   ├── train_learnable_router.py           # BERT-tiny router 训练
│   ├── train_small_model_grpo_local.py     # LoRA + GRPO + K3 KL
│   ├── train_small_model_dpo.py            # LoRA + DPO
│   └── evaluate_finetuned_model.py
├── scripts/                # 多 cycle ordering / iterated 驱动脚本
│   ├── run_iterated_skill_llm_router.sh    # N-cycle joint loop
│   ├── run_ordering_seq.sh                 # skill_first / llm_first
│   ├── run_ordering_exp.sh                 # serial / xiaojie / user / parallel
│   └── run_ordering_parallel.sh            # router ‖ LLM 双 GPU 并行
├── auto_research/          # 7×24 自主 orchestrator
│   ├── orchestrator.py                     # cron tick：收 finished + 起 next
│   ├── runner.py                           # 按 kind 分发实验
│   ├── idea_gen_local.py                   # 规则 fallback 想法生成器
│   ├── arxiv_scrape.py                     # 每日 LLM 关键词 arxiv 抓取
│   ├── cron_entry.sh                       # 每 30 min 跑一次
│   └── README.md                           # 自动机文档
├── data/
│   ├── HumanEval.jsonl                     # 164 道代码题
│   ├── training_data.jsonl                 # 抽出来的 SFT pair
│   ├── traces/                             # 累积的路由 trace
│   └── skills/                             # SkillBook snapshot
├── results/                # 每个 run 的 JSON 摘要
│   ├── iterated_skill_llm_router_8cycles/  # 8-cycle joint 完整 eval
│   └── *.json                              # 历史 headline 数字
├── docs/
│   ├── HIGHLIGHTS.md                       # ⭐ 当前最佳结果
│   ├── E2E_ABLATION_RESULTS.md             # 完整 ablation 表
│   ├── ARCHITECTURE.md, JOINT_EVOLVER.md, ...
└── requirements.txt
```

## 三种跑法

### 1. 单次 joint evolver

```bash
python3 experiments/run_joint_evolver.py \
  --cycles 1 \
  --traces "data/traces/*.jsonl" \
  --router-base-model google/bert_uncased_L-2_H-128_A-2 \
  --llm-train-data data/training_data.jsonl \
  --llm-base-model Qwen/Qwen2.5-Coder-1.5B-Instruct
```

输出 `joint_evolver_manifest.json`，包含每一步命令、metric、产物路径。

### 2. 多 cycle 迭代 (Skill → LLM → Router, 重复 N 次)

```bash
NUM_CYCLES=8 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_iterated_skill_llm_router.sh
```

每个 cycle：SkillBook 在线累积 → LoRA 续训新一个 MBPP chunk → BERT router 在累积数据上重训。

### 3. 7×24 自主模式（cron 驱动）

```bash
# 在一台有 A800 级显卡的机器上：
bash auto_research/cron_entry.sh   # smoke 测试
crontab -e  # 加：*/30 * * * * /path/to/auto_research/cron_entry.sh
```

Orchestrator 每 30 min：`git pull` 拉远程更新 → 应用 cloud agent 提交的待办实验 → 在空闲 GPU 上起最高优先级的实验 → 收已完成实验的 metric。配合 cloud Claude routine（每日 idea-gen、每 2h shepherd 等）就是完全自主的研究循环。

详见 [`auto_research/README.md`](auto_research/README.md)。

## 当前最佳结果

详见 [`docs/HIGHLIGHTS.md`](docs/HIGHLIGHTS.md)。截至 2026-05-21：

- **Router**：冷启动 57.6% → cycle 3 峰 87.8% → 稳定在 82-88%（8-cycle iterated，BERT-tiny，832 题 held-out）
- **Skills**：单调累积，8-cycle 跑出 34 个 signature
- **LLM 1.5B GRPO**：MBPP eval200 上 ~47%（对 ordering 和 cycle 数都不敏感；有 4 个 candidate 配置在 eval100 上 50-51%，正在 eval200 上复核）
- **LLM 3B GRPO**：MBPP eval200 上 61.0%
- **流水线**：只有 `parallel` ordering 真省时间（router ‖ LLM 双 GPU 每 cycle 省 ~4 min）

完整表见 [`docs/E2E_ABLATION_RESULTS.md`](docs/E2E_ABLATION_RESULTS.md)。

## 三个组件核心概念

### Router 路由器

默认是 `google/bert_uncased_L-2_H-128_A-2`，2 层 128 维的小 BERT。二分类输出"是否走贵模型"。训练数据 = UncommonRoute 弱标签 + 累积 trace。按业务的 cost/quality 目标调阈值。

### SkillBook 技能簿

签名 → 频率表。签名是手写的 prompt 折叠：

```
长度桶 × {list, str, num, sort, theory, crypto, advanced, bool, ...}
```

每个签名下，按模型 ID 累积 `(successes, total)`。路由决策时查 Laplace-平滑后的胜率。

### Joint cycle 联合 cycle

cycle k 顺序：

1. SkillBook ← 灌入新 trace（在线，亚秒级）
2. LLM adapter ← 在新 task chunk 上 continual GRPO 一步；LoRA **接力** cycle k-1
3. Router ← 在累积的 router 监督数据上重训 BERT

评估：LLM 用 MBPP eval200；router 用 832 题 held-out。

## 语言 / Languages

- **中文（本文件）**：README.zh.md
- **English**：[README.md](README.md)

## 引用

如果用到这套代码或数据，请引用 GitHub release 页：
https://github.com/zeyuyuyu/router-skills-evolve

## License

MIT.
