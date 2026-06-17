# 架构说明

本仓库是一个 **on-policy 迭代蒸馏**研究系统：大模型当 Teacher，小模型当 Student，
每个 cycle 联合更新三件套（Skills + LLM + Router），越迭代越省钱。唯一入口是
`scripts/run_full_pipeline.sh`（bench：`tau2_bench` 默认 / `humaneval`）。

## 训练 Pipeline（每个 cycle，默认 schedule SLR）

```
Cycle N
  Phase 1  collect_traces.py
           小模型(带 procedure) + 大模型，run-both 记 traces.jsonl
           路由 = 上一轮 router；skillbook 只供 procedure 前缀（不路由）
              │
  Phase 2  SkillBook.update() × N + distill_all()        ← 单一全局 "coding" skill
           累积 small/large 成功率统计 + 提炼一份解题 procedure → skillbook.json
              │
  Phase 3a traces_to_sft.py → SFT（train_small_model.py / tau2_train_wrapper.sh）
           选 small_fail & large_ok 的 hard task + self-repair 链，procedure 前置
              │                                           → llm_adapter/checkpoint-best
  Phase 3b grpo_train_simple.py (HumanEval) / grpo_tau2_train.py (tau2)
           on-policy RL：K 采样 + 可验证 reward + 组内 advantage（GRPO/DAPO 可切）
              │                                           → grpo_adapter/
  Phase 4  train_router_simple.py                         ← 独占路由
           原始 prompt → TF-IDF + LogReg，label = 带 procedure 的小模型是否失败
              │                                           → router/router.joblib
  Phase 5  run_e2e_ablation_simple.py
           四臂：large / skills(全 small+procedure) / router / full
              │
  下一轮 Phase 1 读 grpo_adapter（small）+ router.joblib（路由）+ skillbook.json（procedure）
```

## 推理时的职责划分

```
prompt
  │
  ├─[Router]  router.joblib：P(需要大模型) ≥ threshold ? → large : small   （独占路由）
  │
  └─[SkillBook]  get_procedure(prompt) → 把 procedure 前缀拼到小模型 prompt 上  （不参与路由）
        │
        ▼
   被选中的模型生成 → （HumanEval）pytest / （tau2）env 判定 → 成功/失败回流为下一轮 trace
```

**关键**：Router 决定走小还是大；Skills 只产出 procedure 喂给小模型。两者职责不重叠。

## 核心模块

| 模块 | 作用 |
|------|------|
| `src/config.py` | API key、模型池、价格 |
| `src/models.py` | LLM 调用 + `extract_code` / `run_humaneval_test` |
| `src/skills.py` | `Skill` / `SkillBook`：单一全局 skill 的统计 + procedure 蒸馏 |
| `src/train_plots.py` | SFT/GRPO 训练曲线 |
| `src/pipeline/*` | 6 个 phase 的 bench-agnostic 实现 + `benches/` 适配器 |
| `src/pipeline/train_small_model.py` | HumanEval SFT（LoRA）+ `format_prompt` |
| `tau2_stage2/` | 同事的 tau2 SFT 框架（FSDP2 + FA2） |

## 关键设计

### 单一全局 skill
`extract_signature(prompt)` 恒返回 `"coding"`。早期按"长度桶+关键词"分 20-30 cluster，
但每簇样本太少、统计不可靠；合成一个 bucket 后样本量最大，procedure 从全部成功轨迹提炼，
覆盖最广。路由不再依赖 signature。

### Router 独占路由
`collect_traces._policy_decision` 只用学到的 router 做路由；SkillBook 的
`can_downgrade_to_small` verdict 仅作诊断字段 `policy_skill_verdict`，**不覆盖路由**
（单 skill 下它对所有题相同，会碾压 per-prompt 信号）。

### Procedure 格式对齐
SFT、GRPO、推理统一用 `f"{procedure}\n\n---\n\n{problem}"`，消除 train/inference mismatch。
Router 则训练在**原始 prompt** 上（procedure 是常数前缀，对线性模型零区分力）。

### Laplace 平滑
`success_rate = (successes + 1) / (total + 2)`，新 skill 初始 rate = 0.5。

### Checkpoint 优先级（下一轮 small 模型）
`grpo_adapter/` > `llm_adapter/checkpoint-best` > 原始 `SMALL_MODEL`。

## 成本说明

Router（TF-IDF+LogReg，~ms）和 SkillBook 查询（dict，<1ms）本身 0 LLM 成本——
只有被选中的模型那一次生成才花钱。省钱来自**用训练后的小模型替代大模型**，
而路由把真正需要大模型的难题挑出来兜底。
