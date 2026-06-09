# CWY：35B Skills + LLM + Router 联合演进记录

## 2026-06-09 最新交付状态

目标：在约 1 天窗口内完成 35B tau2 `Skills -> LLM -> Router`
全流程，不把单模型 SFT 当成最终交付。

当前状态：

- 代码修复已提交并推送。
- 文档已脱敏：只保留相对路径和泛化 worker 描述，不记录连接信息。
- 正式 run 已在 shared 8-GPU worker 上启动。
- Cycle 1 已完成：trace skip、SkillBook、bounded SFT、router、E2E ablation。
- Cycle 1 SFT 使用 bounded replay：
  `512` base replay rows + `4` hard trace rows repeated `16` times。
- Cycle 1 E2E 结果：base/skills task `85.14%`，router/full task `89.19%`。
- Cycle 2 已干净重启：旧的失败 trace 已归档，新的
  `results/cwy_35b_joint_20260606_165203/cycle_2/traces.jsonl`
  已开始写入。
- worker 运行环境已补 vLLM Qwen3.5 text-only 兼容：
  language-model-only 跳过视觉塔、纯文本 mrope 返回普通 position ids，
  probe 生成请求通过且服务健康。
- Cycle 2 进一步修正 vLLM serving 参数：
  `MAX_MODEL_LEN=32768`，`REASONING_PARSER=`。原因是 8K context 会在
  tau2 多轮任务中溢出，且 reasoning parser 会把 35B 输出放进
  `reasoning` 字段，导致 trace 中 `small_completion` 为空。
- 修正后已重新归档不干净 trace，并干净重启 Cycle 2 Phase 1。
  当前前两条正式 trace 均为 35B student 成功，`small_completion`
  非空，vLLM health OK。
- tau2 adapter 已进一步修正：evaluation 阶段异常不再丢弃已完成的
  trajectory。这样 retail evaluator / golden action 的 JSON parse 异常
  只会让 reward 记为失败，不会写出空 completion。
- 修正后已再次归档旧 trace 并干净重启 Cycle 2 Phase 1。当前前四条
  trace 均保留了 35B student completion；其中 evaluator 异常任务也
  有非空 `small_completion` / `large_completion`，vLLM health OK。
- 最新复查：Cycle 2 trace 已继续增长到 `6` 行；抽查无
  `small_completion` / `large_completion` 同时为空的行，vLLM health OK。
- 最新复查：Cycle 2 trace 已继续增长到 `8` 行；train split task id
  本身会跳号，已确认不是漏写。最新行 `small_completion` 和
  `large_completion` 均非空，empty-both completion count 仍为 `0`。
- 最新复查：Cycle 2 trace 已继续增长到 `10` 行；`small_empty=0`，
  `empty_both=0`，最新行 `small_completion` / `large_completion` 均非空，
  vLLM health OK。
- 最新复查：Cycle 2 trace 已继续增长到 `12` 行；`small_empty=0`，
  `empty_both=0`，最新行仍为 35B student / large 双 completion 非空。
- 最新复查：Cycle 2 trace 已继续增长到 `14` 行；`small_empty=0`，
  `empty_both=0`。最新行 35B student 失败但 completion 已保留，
  large 成功，`final_success=True`，vLLM health OK。
- 后续复查：Cycle 2 trace 已到 `18/74` 行。虽然 full-pipeline 命令
  保留 tau2 上限参数，但当前 train split 实际加载 `74` 个任务，不是
  要跑满 `848` 条 trace；当前耗时主要来自逐任务多轮 tau2 对话采集。
  最新质量检查仍为 `small_empty=0`、`large_empty=0`、`empty_both=0`。
  最新行 `final_success=True`。
- 当前正在执行 Cycle 2 Phase 1 trace collection；之后会继续进入
  SkillBook、LLM SFT、router train、E2E ablation 和 cycle 汇总。

相对路径：

```text
repo: router-skills-evolve/
run: results/cwy_35b_joint_20260606_165203/
pipeline log: results/cwy_35b_joint_20260606_165203/cwy_corrected_bounded_replay.log
train log: results/cwy_35b_joint_20260606_165203/cycle_1/llm_adapter/train_stdout.log
cycle 2 traces: results/cwy_35b_joint_20260606_165203/cycle_2/traces.jsonl
```

## 2026-06-09 配方修正：bounded replay 35B 正式续跑

结论：旧 tau2 port 的 LLM track 不是代码崩坏，但训练策略过重，偏离 MERA 论文里的 hard-example adaptation 口径。

旧实现：

```text
current-cycle hard SFT rows + full stage2_v1 train corpus
= 4 scaling rows + 6413 base rows
```

然后按 35B 配置跑 5 epochs，cycle_1 实测约 `222s/step`，会把大部分时间花在反复 replay 历史 stage2 corpus 上，且本轮 hard examples 信号被稀释。

已改为 bounded replay：

```text
SCALING_BASE_REPLAY_ROWS=512
SCALING_TRACE_REPEAT=16
SCALING_NUM_TRAIN_EPOCHS=1
```

含义：

- 本轮 hard examples 全部使用。
- 当前 cycle hard rows 默认重复 16 次，避免被 replay 淹没。
- base corpus 只做 deterministic domain-balanced replay，默认 512 行。
- wrapper 会写 `llm_adapter/scaling_sft/replay_mix_meta.json` 记录 replay 组成。
- train_all scaling config 支持用环境变量覆盖 `num_train_epochs` / `max_seq_length` / `max_steps` 等训练参数。

重要恢复说明：

- 旧 `checkpoint-89` 是按 `6413+4` 全量 replay、5 epochs 配方产生的。
- 新短训配方的总 step 数小于 89，不能安全从旧 Trainer optimizer/scheduler state auto-resume。
- 旧 checkpoint 继续作为共享存储备份/审计保留。
- 新正式 run 启动时必须设置 `EVOL_DISABLE_AUTO_RESUME=1`，避免 Trainer 把旧 checkpoint-89 当成兼容续训点。
- 如果需要从旧权重继续而不是从 base 模型开始，应另开独立 init-from-checkpoint 配方；本次为了 1 天内完成全流程，按 corrected hard-example SFT recipe 重训 cycle_1 LLM track。

新 8 卡运行环境：

```text
worker: current private 8-GPU worker
gpu: 8 x H200-class
shared storage: ready
repo path: router-skills-evolve/
experiment path: results/cwy_35b_joint_20260606_165203/
```

## 结论先写

当前要跑的是仓库正式 35B 联合演进任务，不是单模型 eval。

必须完整包含：

1. `collect_traces`
2. `skills evolve`
3. `LLM train`
4. `router train`
5. `E2E ablation`
6. 每轮 cycle 的 accuracy / cost / fallback / task pass

## 论文和项目口径

## 已读项目后的统一理解

我已按本次任务重新读了仓库主入口和 scaling/tau2 关键代码；2026-06-07 08:20 CST 又按新加论文要求复核了一遍项目和论文全文：

```text
README.md
scaling/README.md
scaling/run_full_pipeline.sh
src/skills.py
experiments/scaling/collect_traces.py
experiments/scaling/traces_to_sft.py
experiments/scaling/tau2_train_wrapper.sh
experiments/scaling/train_router_simple.py
experiments/scaling/run_e2e_ablation_simple.py
experiments/scaling/aggregate_cycles.py
experiments/scaling/benches/tau2_bench/adapter.py
experiments/tau2_stage2/code/training/configs/runs/08_qwen3_6_35b_a3b_273.yaml
```

本仓库正式做的不是“单个模型 eval”，而是一个闭环系统：

```text
Phase 1: collect_traces
  用 small model 和 large model 跑 tau2，得到 small_success / large_success / cost / completion。

Phase 2: skills evolve
  从 traces 里按 prompt signature 统计 small/large 成功率，并从成功轨迹蒸馏 procedure。

Phase 3: LLM train
  从 small 失败且 large 成功的 hard traces 提 SFT pair，训练本次指定的 35B adapter。

Phase 4: router train
  用 trace prompt -> 是否 need_large 的标签训练 learnable router。

Phase 5: E2E ablation
  报 Base / +Skills / +Router / Full 四个系统变体的 routing_acc、fallback、cost_vs_large、task_pass。

Phase 6: aggregate
  汇总每轮 cycle 曲线和最终表。
```

当前仓库代码里，本次 35B 的 LLM 更新方式是 SFT：

```text
hard traces -> traces_to_sft.py -> tau2_train_wrapper.sh -> tau2_stage2 SFT
```

不是 RL，也不是论文旧实验里的 1.5B GRPO。Router 是 `TF-IDF + LogisticRegression`，SkillBook 是 `signature stats + heuristic procedure distillation`。

当前实现细节必须如实报告：

- `SkillBook` 不是一次 prompt 让大模型凭空生成，而是按 prompt signature 聚类，记录 small / large 成功统计，并从成功 completion 里用无 API heuristic 蒸馏 procedure。
- `LLM train` 使用 `MODE=scaling_traces`，把本轮 hard traces 转为 SFT pairs，再和 tau2_stage2 原 corpus 合并训练；当前 cycle 0 是 17 条 hard pairs。
- `Router` 标签是 `small_success=false => need_large`，模型是 TF-IDF + LogisticRegression。
- `E2E ablation` 的 Base / Skills / Router 会直接从 trace + skillbook + router 产物算；但 `Full` 若未额外传入 LLM task pass，默认只等同 router routing 指标。35B checkpoint 出来后必须另补 35B adapter 的 tau2 任务级 eval，不能把默认 `full == router` 当成 LLM 效果。

已读取论文：

```text
../routerevolving-2.pdf
```

PDF 元信息：

```text
title: MERA: Model Evolution and Routing with Skill Adaptation for Agentic Systems at Scale
pages: 7
created: 2026-06-07 00:07:27 CST
read_method: pdfinfo + pdftotext
```

论文标题：

```text
MERA: Model Evolution and Routing with Skill Adaptation for Agentic Systems at Scale
```

论文主张：

- MERA 是 trace-driven 的三轨联合演进框架。
- 三条轨道是 `SkillBook`、`LLM adapter`、`learned router`。
- 更新单位不是整条用户请求，而是 agent trace 里的单次 invocation / step slice。
- trace 同时提供三类证据：SkillBook 统计、LLM hard-example SFT 数据、router cheap/strong 标签。
- 新的 SkillBook / LLM adapter / router 不能单独看指标，要通过 joint replay / E2E ablation 一起准入。
- 正式评价不能只看单模型 SFT，也不能只看单模型 eval。
- 要看联合 replay / E2E ablation 后的系统效果。

论文主指标：

| 指标 | 含义 |
| --- | --- |
| Router accuracy | router 是否正确判断 cheap / strong |
| Fallback | 便宜模型失败后回退强模型比例，越低越好 |
| Cost vs always-large | 相比全用大模型的成本，越低越好 |
| LLM pass / Task pass | 任务最终通过率 |

论文主 schedule：

```text
SLR = Skill -> LLM -> Router
```

论文四轮主结果口径：

```text
Skill -> LLM -> Router: Router acc 87.3%, fallback 4.4%, cost 51.8%
```

注意：上面 87.3% / 4.4% / 51.8% 是论文 code 域旧实验结果，不是本次 tau2 + 35B 的结果。
本次项目要按同一口径汇报 tau2 实测结果，不再只报单模型 tau2 pass。

论文和当前仓库的对应关系：

| MERA 论文组件 | 当前仓库实现 | 本次必须记录的指标 |
| --- | --- | --- |
| SkillBook | `src/skills.py` + `phase2_skills_evolve` | `skills` 变体 accuracy / fallback / cost_vs_large |
| LLM adapter | `traces_to_sft.py` + `tau2_train_wrapper.sh` + tau2_stage2 SFT | 35B adapter task accuracy；训练 token accuracy 只作过程日志 |
| Learned router | `train_router_simple.py`，TF-IDF + LogisticRegression | router accuracy / fallback / cost_vs_large |
| Joint replay / admission | `run_e2e_ablation_simple.py` + `aggregate_cycles.py` | Full = Skills + LLM + Router 的 E2E accuracy / cost |

本次报告口径：

| 名称 | 本次含义 |
| --- | --- |
| 初始 accuracy | cycle 0 训练前 small model 在真实 tau2 trace 上的通过率 / always-small router label accuracy |
| fallback 后 task pass | small 失败后调用 large，最终任务通过率 |
| router accuracy | Phase 4/5 里 router 对 `small_ok` vs `need_large` 标签的准确率 |
| cost | trace 实际 API cost；E2E 表另报 `cost_vs_large` |
| full | Skills + 35B SFT adapter + Router 三者一起后的 E2E ablation 结果 |

## 本次必须汇报的 accuracy / cost 表

最开始的 accuracy 已按 cycle 0 原始 traces 计算：

```text
initial_small_accuracy = small_success / tasks = 42 / 74 = 56.76%
initial_fallback_needed = small_failed / tasks = 32 / 74 = 43.24%
initial_task_pass_after_fallback = final_success / tasks = 59 / 74 = 79.73%
initial_observed_api_cost = $3.49623640
```

每轮 cycle 训练完后按下面表补齐，不能用论文旧数字顶替：

| Cycle | Base accuracy | Skills accuracy | LLM/SFT accuracy | Router accuracy | Full accuracy | Fallback | Cost vs Large | 实际 API cost | 备注 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 56.76% | pending E2E | pending 35B checkpoint | pending router | pending E2E | 43.24% initial | pending E2E | $3.49623640 | Phase 3 SFT running |
| 1 | pending | pending | pending | pending | pending | pending | pending | pending | 等 cycle 0 产物闭环后开始 |
| 2 | pending | pending | pending | pending | pending | pending | pending | pending | 等 cycle 1 产物闭环后开始 |
| 3 | pending | pending | pending | pending | pending | pending | pending | pending | 最终汇总 |

说明：

- `Base accuracy` 当前按 small model 是否成功理解，即 always-small 的初始能力。
- `Skills accuracy / Router accuracy / Full accuracy` 必须等 Phase 5 `e2e_ablation_summary.json` 出来后填。
- `LLM/SFT accuracy` 必须等 35B adapter checkpoint 产出并进入后续评测后填；训练日志里的 `mean_token_accuracy` 只是 token 级训练指标，不等于任务 accuracy。
- 每轮训练完必须同步记录两类 cost：
  - `actual API cost`：本轮 collect_traces 真实调用 small / large / fallback 的美元费用。
  - `cost_vs_large`：E2E ablation 中相对 always-large 的模拟服务成本。

## 本次正式模型

按仓库正式 35B 配置跑。

```text
run_id: 08_qwen3_6_35b_a3b_273
model: Qwen/Qwen3.6-35B-A3B
revision: 995ad96eacd98c81ed38be0c5b274b04031597b0
config: experiments/tau2_stage2/code/training/configs/runs/08_qwen3_6_35b_a3b_273.yaml
```

注意：

- 不跑 `11_qwen3_30b_a3b_273`。
- 不把 30B 单模型 eval 当正式结果。
- 不新增别的模型替代仓库模型。

## 正式运行设置

```text
bench: tau2_bench
model_sweep: 08_qwen3_6_35b_a3b_273
schedule: SLR
n_cycles: 4
mock: false
skip_llm: false
```

每轮必须产出：

```text
results/<experiment>/cycle_N/traces.jsonl
results/<experiment>/cycle_N/skillbook.json
results/<experiment>/cycle_N/training_data.jsonl
results/<experiment>/cycle_N/llm_adapter/checkpoint-best
results/<experiment>/cycle_N/router/router.joblib
results/<experiment>/cycle_N/e2e_ablation_summary.json
results/<experiment>/cycle_N/e2e_ablation_summary.md
```

最终必须产出：

```text
results/<experiment>/final_ablation_table.md
results/<experiment>/curve.png
```

## 每轮记录表

运行后逐轮填写。当前 cycle 0 已完成 trace / SkillBook / 35B SFT / router / E2E ablation；cycle 1 正在启动本地 35B adapter vLLM 继续收集 traces。

| Cycle | Phase | 状态 | Accuracy | Task Pass | Fallback | Cost vs Large | 实际 Cost | 产物/日志 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | Base / trace | done | 56.76% | 79.73% | 43.24% | pending | $3.496236 | `cycle_0/traces.jsonl` |
| 0 | Skills | done | 56.76% | 56.76% | 43.24% | 10.00% | no extra API | `cycle_0/skillbook.json`，15 skills，14 procedures；cycle0 ablation 中 skills 与 base 相同 |
| 0 | LLM SFT | done | pending task-level eval | pending task-level eval | pending | pending | no API；GPU time done | 17 hard tasks -> 17 SFT pairs；35B SFT 445/445 done；`checkpoint-best -> checkpoint-89`；train_loss 0.03668；eval_loss 0.1352；eval token acc 0.883 |
| 0 | Router | done | 93.24% E2E routing / 73.68% held-out router | 78.38% | 4.05% | 47.70% | no extra API | `cycle_0/router/router.joblib`；router_meta acc 0.7368，f1_large 0.6667 |
| 0 | Full | done | 93.24% | 78.38% | 4.05% | 47.70% | no extra API | `cycle_0/e2e_ablation_summary.json`；当前 simple ablation 中 full == router，35B task-level eval 仍需单独跑 |
| 1 | Base / trace | paused | pending | 39/74 clean traces kept | pending | pending | CommonStack key cap hit | `cycle_1/traces.jsonl` 已裁回干净 39 行；完整坏现场归档到 `debug_archive/api_cap_bad_traces_20260608_1004/`；等待 API 额度恢复后用 resume 继续 |
| 1 | Full | pending |  |  |  |  |  |  |
| 2 | Full | pending |  |  |  |  |  |  |
| 3 | Full | pending |  |  |  |  |  |  |

Cycle 0 初始数字明细：

```text
tasks: 74
small_success: 42/74 = 56.76%
small_failed / fallback_needed: 32/74 = 43.24%
large_real_success_on_small_failed: 17/32 = 53.13%
final_success_after_fallback: 59/74 = 79.73%
hard_sft_candidates: 17
observed_total_api_cost: $3.49623640
small_cost_sum: $0.99030330
large_cost_sum: $2.50593310
```

## 当前检查

- 本地代码：`main`
- 本地 commit：`769b39c`
- GPU worker：当前私有 8 卡环境
- GPU：当前 8 张 H200-class 正在跑 35B SFT，约 `120GB+ / 卡`
- 远端代码：已从本地同步最新代码和本文档；保留远端 `.env`、venv、vendor、数据、历史结果
- 35B checkpoint：尚未发现已有产物，需要正式训练
- 35B模型缓存：CPU 本机已下载完整 HF cache，已同步到 H200
- H200 模型验证：已用 `local_files_only=True` 验证 `AutoConfig` 和 `AutoTokenizer` 可加载
- API / HF token：`.env` 中已确认存在，不在文档里记录密钥
- CommonStack：H200 通过 CPU 反向隧道 `<api-base-from-env>` 测试通过，HTTP 200

## 当前进度

2026-06-07 00:52 CST：

- 已启动修复后的正式有效运行。
- H200 tmux session：

```text
cwy35b_20260606_165203
```

- 实验目录：

```text
results/cwy_35b_joint_20260606_165203
```

- 已确认进入 Phase 1：

```text
mock=False
loaded 74 tasks
SMALL_MODEL=deepseek/deepseek-v3.2
LARGE_MODEL=openai/openai/gpt-5.4-2026-03-05
```

- 第一条 trace 已落盘并验证有效：

```text
cycle: 0
phase: collect_traces
progress: 1/74
task_id: 0
prompt_len: 460
signature: You received your order #W2378156 ...
small_success: false
large_success: false
small_cost: 0.01106875
large_cost: 0.07611075
total_cost: 0.08717950
```

- 当前还没有可报告的 cycle accuracy；要等 Phase 1 收完 74 条并完成 Phase 5 E2E ablation。

- Phase 1 早期进度：

```text
progress: 74/74
progress_pct: 100.00%
prompt_empty: 0
large_404_like: 0
small_success: 42/74
large_success: 17/32
final_success: 59/74
hard_sft_candidates: 17
observed_cost_sum: 3.49623640
small_cost_sum: 0.99030330
large_cost_sum: 2.50593310
last_task_id_done: 113
skillbook_size: 15
procedures_distilled: 14
sft_pairs: 17
```

2026-06-07 03:37 CST：

- Cycle 0 Phase 1 `collect_traces` 已完成：74/74。
- Cycle 0 Phase 2 `skills evolve` 已完成：`SkillBook size=15`，`procedures_distilled=14`。
- Cycle 0 Phase 3 `traces_to_sft` 已完成：17 个 hard tasks -> 17 个 SFT pairs，全部带 procedure。
- Phase 3 wrapper 修复：
  - 补 `_p.domain`，解决 `convert_to_prompt_completion.py` 的 `KeyError: 'domain'`。
  - 补 `_p.run_dir` 和同类 provenance，解决 `training.train` 的 `KeyError: 'run_dir'`。
  - H200 HF cache 补 `refs/main=995ad96eacd98c81ed38be0c5b274b04031597b0`，解决离线 tokenizer validation 找不到默认 revision。
- 当前状态：35B SFT 已通过 chat-template validation，正在 tokenizing / training 准备阶段。

2026-06-07 03:52 CST：

- H200 tmux session 仍在运行：

```text
cwy35b_resume3_cwy_35b_joint_20260606_165203_193736
```

- 35B SFT 状态：

```text
STATUS=running
progress: 1/445 steps
GPU: 8 张 H200 全部接近满载，显存约 120GB+/卡
```

- 说明：
  - 当前还没有 cycle 0 的 router / full E2E accuracy。
  - 训练结束后会自动进入 Phase 4 router training、Phase 5 E2E ablation，再继续 cycle 1-3。
  - 每轮完成后按本文“每轮记录表”补齐 `accuracy / task pass / fallback / cost`。

2026-06-07 03:58 CST：

- 35B SFT 仍在正常前进：

```text
STATUS=running
progress: 2/445 steps
observed_step_time: about 218s/step
rough_cycle0_sft_eta_if_speed_holds: about 26-27 hours
GPU: 8 张 H200 高利用，显存约 124GB/卡
```

- 日志提示当前仓库正式配置的 `loss_type=chunked_nll` + FSDP2 `reshard_after_forward=true` 会慢。
- 现在不停止训练、不改模型、不切配置；先按仓库正式 35B 配置继续跑，避免浪费已启动训练。
- 若后续 step 速度明显变化，以最新 step 速度修正 ETA。

2026-06-07 03:59 CST：

- 35B SFT 继续前进：

```text
STATUS=running
progress: 3/445 steps
observed_step_time: about 218s/step
GPU: 8 张 H200 高利用，显存约 124GB/卡
```

- 当前还没有 `checkpoint-best`，因此 Phase 4 router training 和 Phase 5 E2E ablation 尚未开始。
- 已确认当前 tmux 后续命令链完整：

```text
cycle0 phase3 train
-> cycle0 phase4 router train
-> cycle0 phase5 e2e ablation
-> bash scaling/run_full_pipeline.sh --resume 1
```

- 所以训练完成后会自动进入 router / E2E，并继续 cycle 1-3；当前不需要重新启动。

2026-06-07 04:03 CST：

- 35B SFT 继续前进，速度稳定：

```text
STATUS=running
progress: 4/445 steps
observed_step_time: about 217.6s/step
rough_cycle0_sft_eta_if_speed_holds: about 26-27 hours total
GPU: 8 张 H200 基本 99-100% 利用，显存约 124GB/卡
```

- 尚未生成：

```text
train_outputs/08_qwen3_6_35b_a3b_273/checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
```

2026-06-07 04:08 CST：

- 35B SFT 继续正常前进：

```text
STATUS=running
progress: 5/445 steps
observed_step_time: 216-218s/step
GPU: 8 张 H200 仍高利用，显存约 124GB/卡
```

- `checkpoint-best`、`cycle_0/llm_adapter/checkpoint-best`、`router.joblib`、`e2e_ablation_summary.json` 仍未生成。
- 结论：当前仍处于 cycle 0 Phase 3 训练阶段；Phase 4/5 等 checkpoint 产出后自动继续。

2026-06-07 04:12 CST：

- 35B SFT 继续正常前进：

```text
STATUS=running
progress: 6/445 steps
observed_step_time: about 216s/step
GPU: 8 张 H200 100% 左右利用，显存约 124GB/卡
```

- 尚未生成 `checkpoint-best` / router / E2E 产物，仍处于 cycle 0 Phase 3。

2026-06-07 04:13 CST：

- 35B SFT 继续正常前进：

```text
STATUS=running
progress: 7/445 steps
observed_step_time: about 216-218s/step
GPU: 8 张 H200 高利用，显存约 124GB/卡
```

- 尚未生成 `checkpoint-best` / `cycle_0/llm_adapter/checkpoint-best` / `router.joblib` / `e2e_ablation_summary.json`。
- 当前没有训练报错；只看到 FSDP2 `chunked_nll + reshard_after_forward=true` 的慢速提示。

2026-06-07 04:18 CST：

- 35B SFT 继续正常前进：

```text
STATUS=running
progress: 8/445 steps
observed_step_time: about 217s/step
GPU: 8 张 H200 显存约 124GB/卡，利用率有波动但仍在训练
```

- 尚未生成 `checkpoint-best` / router / E2E 产物。

2026-06-07 04:22 CST：

- 35B SFT 继续正常前进：

```text
STATUS=running
progress: 9/445 steps
observed_step_time: about 217s/step
GPU: 8 张 H200 高利用，显存约 124GB/卡
```

- 尚未生成 `checkpoint-best` / `cycle_0/llm_adapter/checkpoint-best` / `router.joblib` / `e2e_ablation_summary.json`。
- 仍处于 cycle 0 Phase 3，后续 Phase 4/5 等 checkpoint 出来后自动继续。

2026-06-07 04:26 CST：

- 35B SFT 继续正常前进，并出现第一条训练指标：

```text
STATUS=running
progress: 10/445 steps
observed_step_time: about 217.7s/step
loss: 0.1289
grad_norm: 1.938
learning_rate: 3.913e-06
mean_token_accuracy: 0.9588
epoch: 0.1125
GPU: 8 张 H200 高利用，显存约 124GB/卡
```

- 尚未生成 `checkpoint-best` / `cycle_0/llm_adapter/checkpoint-best` / `router.joblib` / `e2e_ablation_summary.json`。

2026-06-07 04:30 CST：

- 35B SFT 继续正常前进：

```text
STATUS=running
progress: 11/445 steps
observed_step_time: about 218s/step
last_logged_train_loss: 0.1289 at step 10
last_logged_mean_token_accuracy: 0.9588 at step 10
GPU: 8 张 H200 显存约 124GB/卡，利用率采样有波动但训练仍在推进
```

- 尚未生成 `checkpoint-best` / `cycle_0/llm_adapter/checkpoint-best` / `router.joblib` / `e2e_ablation_summary.json`。

2026-06-07 04:31 CST：

- 35B SFT 继续正常前进：

```text
STATUS=running
progress: 12/445 steps
observed_step_time: about 217s/step
last_logged_train_loss: 0.1289 at step 10
last_logged_mean_token_accuracy: 0.9588 at step 10
GPU: 8 张 H200 高利用，显存约 124GB/卡
```

- 尚未生成 `checkpoint-best` / `cycle_0/llm_adapter/checkpoint-best` / `router.joblib` / `e2e_ablation_summary.json`。

2026-06-07 04:35 CST：

- 35B SFT 继续正常前进：

```text
STATUS=running
progress: 13/445 steps
observed_step_time: about 216-217s/step
last_logged_train_loss: 0.1289 at step 10
last_logged_mean_token_accuracy: 0.9588 at step 10
GPU: 8 张 H200 显存约 124GB/卡，利用率采样有波动但训练仍在推进
```

- 尚未生成 `checkpoint-best` / `cycle_0/llm_adapter/checkpoint-best` / `router.joblib` / `e2e_ablation_summary.json`。

2026-06-07 04:40 CST：

- 35B SFT 继续正常前进：

```text
STATUS=running
progress: 14/445 steps
observed_step_time: about 217s/step
last_logged_train_loss: 0.1289 at step 10
last_logged_mean_token_accuracy: 0.9588 at step 10
GPU: 8 张 H200 高利用，显存约 124GB/卡
```

- 尚未生成 `checkpoint-best` / `cycle_0/llm_adapter/checkpoint-best` / `router.joblib` / `e2e_ablation_summary.json`。

2026-06-07 04:44 CST：

- 35B SFT 继续正常前进：

```text
STATUS=running
progress: 15/445 steps
observed_step_time: about 217s/step
last_logged_train_loss: 0.1289 at step 10
last_logged_mean_token_accuracy: 0.9588 at step 10
GPU: 8 张 H200 高利用，显存约 124GB/卡
```

- 尚未生成 `checkpoint-best` / `cycle_0/llm_adapter/checkpoint-best` / `router.joblib` / `e2e_ablation_summary.json`。

2026-06-07 04:48 CST：

- 35B SFT 继续正常前进：

```text
STATUS=running
progress: 16/445 steps
observed_step_time: about 216.9s/step
last_logged_train_loss: 0.1289 at step 10
last_logged_mean_token_accuracy: 0.9588 at step 10
GPU: 8 张 H200 显存约 124GB/卡，利用率采样有波动但训练仍在推进
```

- 尚未生成 `checkpoint-best` / `cycle_0/llm_adapter/checkpoint-best` / `router.joblib` / `e2e_ablation_summary.json`。

2026-06-07 04:50 CST：

- 35B SFT 继续正常前进：

```text
STATUS=running
progress: 17/445 steps
observed_step_time: about 218s/step
last_logged_train_loss: 0.1289 at step 10
last_logged_mean_token_accuracy: 0.9588 at step 10
GPU: 8 张 H200 显存约 124GB/卡，利用率采样有波动但训练仍在推进
```

- 尚未生成 `checkpoint-best` / `cycle_0/llm_adapter/checkpoint-best` / `router.joblib` / `e2e_ablation_summary.json`。

2026-06-07 04:54 CST：

- 35B SFT 继续正常前进：

```text
STATUS=running
progress: 18/445 steps
observed_step_time: about 217s/step
last_logged_train_loss: 0.1289 at step 10
last_logged_mean_token_accuracy: 0.9588 at step 10
GPU: 8 张 H200 显存约 124GB/卡，利用率采样有波动但训练仍在推进
```

- 尚未生成 `checkpoint-best` / `cycle_0/llm_adapter/checkpoint-best` / `router.joblib` / `e2e_ablation_summary.json`。

2026-06-07 04:58 CST：

- 35B SFT 继续正常前进：

```text
STATUS=running
progress: 19/445 steps
observed_step_time: about 215-217s/step
last_logged_train_loss: 0.1289 at step 10
last_logged_mean_token_accuracy: 0.9588 at step 10
GPU: 8 张 H200 显存约 124GB/卡，利用率采样有波动但训练仍在推进
```

- 尚未生成 `checkpoint-best` / `cycle_0/llm_adapter/checkpoint-best` / `router.joblib` / `e2e_ablation_summary.json`。

2026-06-07 05:02 CST：

- 已按新增论文要求重新对齐项目和论文口径：本次结果必须报告最开始 accuracy，以及每轮 Skills / LLM / Router / Full 的 accuracy、fallback、cost。
- 35B SFT 继续正常前进：

```text
STATUS=running
progress: 20/445 steps
observed_step_time: about 216s/step
latest_train_loss: 0.09616 at step 20
latest_mean_token_accuracy: 0.9639 at step 20
GPU: 8 张 H200 显存约 124GB/卡，训练仍在推进
```

- 仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
final_ablation_table.md
```

- 因此当前还没有 cycle 0 的 Skills / Router / Full E2E accuracy；正式结果必须等 SFT checkpoint 后自动进入 Phase 4/5 才能填。

2026-06-07 05:04 CST：

- 35B SFT 继续正常前进：

```text
STATUS=running
progress: 21/445 steps
observed_step_time: about 215s/step
latest_train_loss: 0.09616 at step 20
latest_mean_token_accuracy: 0.9639 at step 20
```

- 仍未生成 checkpoint / router / E2E 产物。

2026-06-07 05:05 CST：

- 远端 H200 权威检查：

```text
worker: current private 8-GPU worker
repo: router-skills-evolve/
remote_git: 0cce7aa
tmux: cwy35b_resume3_cwy_35b_joint_20260606_165203_193736
STATUS=running
progress: 21/445 steps
GPU: 8 张 H200 均占用约 124GB 显存
```

- 当前产物状态：

```text
checkpoint-best: missing
cycle_0/llm_adapter/checkpoint-best: missing
cycle_0/router/router.joblib: missing
cycle_0/router/router_meta.json: missing
cycle_0/e2e_ablation_summary.json: missing
cycle_1/traces.jsonl: missing
final_ablation_table.md: missing
curve.png: missing
```

- 判断：仍在 cycle 0 Phase 3 SFT。没有失败日志；后续 router/E2E 必须等 35B checkpoint 产出后自动继续。

2026-06-07 05:10 CST：

- 等待一个 step 周期后再次检查，确认训练没有卡住：

```text
STATUS=running
progress: 22/445 steps
elapsed_train_time: 1:19:26
observed_step_time: about 214.7s/step
latest_train_loss: 0.09616 at step 20
latest_mean_token_accuracy: 0.9639 at step 20
```

- 当前仍无 checkpoint / router / E2E 产物。SFT 继续跑，后续 phase 等 checkpoint 产出后自动接上。

2026-06-07 05:11 CST：

- 远端 H200 再次检查：

```text
STATUS=running
progress: 23/445 steps
elapsed_train_time: 1:23:08
observed_step_time: about 217.1s/step
train_log_mtime: 2026-06-07 05:10:26
GPU: 8 张 H200 持续占用约 124GB 显存
```

- 当前依然没有以下产物：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：正式 35B SFT 仍健康推进；cycle 0 Router / E2E 尚未开始。

2026-06-07 05:12 CST：

- 已验证当前 tmux 里的命令链不是单独 SFT，而是完整恢复链：

```text
cycle0 phase3 extract retry
-> cycle0 phase3 train retry
-> cycle0 phase4 router train
-> cycle0 phase5 e2e ablation
-> resume pipeline from cycle1
```

- 正在运行的训练命令仍是正式 35B：

```text
accelerate launch ... -m training.train
--run-config training/configs/runs/08_qwen3_6_35b_a3b_273.yaml
--plan-config .../plan_c_prime.yaml
```

2026-06-07 05:16 CST：

- 等待一个短窗口后检查到新 step，训练仍在推进：

```text
STATUS=running
progress: 24/445 steps
elapsed_train_time: 1:26:46
observed_step_time: about 217.4s/step
train_log_mtime: 2026-06-07 05:14:05
GPU: 8 张 H200 仍在训练，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
```

- 判断：还在 cycle 0 Phase 3 SFT；Phase 4 Router 和 Phase 5 E2E 尚未开始。

2026-06-07 05:19 CST：

- 再次等待短窗口后检查到新 step：

```text
STATUS=running
progress: 25/445 steps
elapsed_train_time: 1:30:20
observed_step_time: about 216.3s/step
train_log_mtime: 2026-06-07 05:17:38
GPU: 8 张 H200 仍在训练，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
```

- 判断：cycle 0 Phase 3 SFT 继续健康推进；Router / E2E 仍需等待 checkpoint。

2026-06-07 05:22 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 26/445 steps
elapsed_train_time: 1:34:03
observed_step_time: about 218.4s/step
train_log_mtime: 2026-06-07 05:21:21
GPU: 8 张 H200 仍在训练，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
```

- 判断：仍处于 cycle 0 Phase 3 SFT；后续 Router / E2E 等 checkpoint 出来后自动接上。

2026-06-07 05:26 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 27/445 steps
elapsed_train_time: 1:37:33
observed_step_time: about 215.9s/step
train_log_mtime: 2026-06-07 05:24:52
GPU: 8 张 H200 仍在训练，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
```

- 判断：cycle 0 Phase 3 SFT 继续健康推进；后续 Router / E2E 仍等待 checkpoint。

2026-06-07 06:06 CST：

- 按用户新增要求重新读取项目和论文，并明确最终报告必须包含：

```text
initial accuracy
Skills accuracy / cost
LLM adapter task accuracy / cost
Router accuracy / cost
Full Skills+LLM+Router accuracy / cost
每轮 cycle 的 fallback、task pass、cost_vs_large、实际 API cost
```

- 再次检查 H200，35B SFT 正常推进：

```text
STATUS=running
progress: 38/445 steps
elapsed_train_time: 2:17:26
observed_step_time: about 217.8s/step
train_log_mtime: 2026-06-07 06:04:44
latest_train_loss: 0.06071 at step 30
latest_mean_token_accuracy: 0.976 at step 30
GPU: 8 张 H200 仍在训练，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
```

- 判断：仍处于 cycle 0 Phase 3 SFT。Router / E2E / 35B adapter 任务级 eval 都必须等 checkpoint 产出后继续。

2026-06-07 06:09 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 39/445 steps
elapsed_train_time: 2:21:06
observed_step_time: about 218.5s/step
train_log_mtime: 2026-06-07 06:08:24
latest_train_loss: 0.06071 at step 30
latest_mean_token_accuracy: 0.976 at step 30
GPU: 8 张 H200 96-100% 利用，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
```

- 判断：仍在 cycle 0 Phase 3 SFT；下一条训练指标预计 step 40 左右出现。

2026-06-07 06:13 CST：

- 再次检查到新 step，并出现 step 40 训练指标：

```text
STATUS=running
progress: 40/445 steps
elapsed_train_time: 2:24:46
observed_step_time: about 218.9s/step
train_log_mtime: 2026-06-07 06:12:04
loss: 0.06273
grad_norm: 0.7969
learning_rate: 9.968e-06
mean_token_accuracy: 0.9765
epoch: 0.4501
GPU: 8 张 H200 显存约 124GB/卡，训练仍在推进
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
```

- 注意：step 40 的 `mean_token_accuracy=0.9765` 仍然只是训练 token 指标，不等于 tau2 task accuracy。正式 task accuracy 要等 checkpoint 后补 35B adapter eval。

2026-06-07 06:16 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 41/445 steps
elapsed_train_time: 2:28:27
observed_step_time: about 219.7s/step
train_log_mtime: 2026-06-07 06:15:45
latest_train_loss: 0.06273 at step 40
latest_mean_token_accuracy: 0.9765 at step 40
GPU: 8 张 H200 仍在训练，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

- 判断：仍在 cycle 0 Phase 3 SFT；Phase 4/5 尚未开始。

2026-06-07 07:35 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 63/445 steps
elapsed_train_time: 3:48:11
observed_step_time: about 215.8s/step
train_log_mtime: 2026-06-07 07:35:29
latest_train_loss: 0.07143 at step 60
latest_mean_token_accuracy: 0.9717 at step 60
GPU: 8 张 H200 高利用，显存约 124GB/卡，训练继续推进
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

- 判断：仍在 cycle 0 Phase 3 SFT；Phase 4/5 尚未开始。

2026-06-07 07:38 CST：

- 按新增论文要求再次复核项目和论文，并确认最终报告表必须按下面口径填写：

```text
initial accuracy / initial cost
cycle_N skills accuracy / cost
cycle_N LLM adapter task accuracy / cost
cycle_N router accuracy / cost
cycle_N full Skills+LLM+Router accuracy / cost
```

- H200 状态：

```text
STATUS=running
progress: last logged 63/445 steps
train_log_mtime: 2026-06-07 07:35:29
latest_train_loss: 0.07143 at step 60
latest_mean_token_accuracy: 0.9717 at step 60
GPU: 8 张 H200 全部 99%-100% 利用，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：机器已经在跑正式 35B SFT；还没有任何一轮训练后的 Skills / Router / Full accuracy 可填。等 cycle 0 checkpoint 出来后，pipeline 会进入 Phase 4 Router、Phase 5 E2E ablation，再补第一轮完整 accuracy/cost。

2026-06-07 07:41 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 64/445 steps = 14.38%
elapsed_train_time: 3:51:48
observed_step_time: about 216.0s/step
rough_remaining_for_cycle0_sft_if_speed_holds: about 22.9 hours
train_log_mtime: 2026-06-07 07:39:06
latest_train_loss: 0.07143 at step 60
latest_mean_token_accuracy: 0.9717 at step 60
GPU: 8 张 H200 继续训练，显存约 124GB/卡，利用率大多 100%
```

- 当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：仍处于 cycle 0 Phase 3 SFT。还不能报告每轮训练后的 Skills / LLM / Router / Full accuracy；正式结果必须等 35B checkpoint 后接 Phase 4/5。

2026-06-07 07:42 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 65/445 steps = 14.61%
elapsed_train_time: 3:55:27
observed_step_time: about 217.0s/step
rough_remaining_for_cycle0_sft_if_speed_holds: about 22.9 hours
train_log_mtime: 2026-06-07 07:42:45
latest_train_loss: 0.07143 at step 60
latest_mean_token_accuracy: 0.9717 at step 60
GPU: 8 张 H200 基本 99%-100% 利用，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：正式 35B SFT 继续健康推进。当前仍没有 cycle 0 训练后的 Skills / Router / Full E2E accuracy，不能提前填结果。

2026-06-07 07:44 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 65/445 steps
train_log_mtime: 2026-06-07 07:42:45
GPU: 8 张 H200 仍在训练，基本 99%-100% 利用，显存约 124GB/卡
latest_metric: loss 0.07143 / mean_token_accuracy 0.9717 at step 60
```

- 当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：仍在 cycle 0 Phase 3 SFT；无错误日志，无新 checkpoint。继续等待 Phase 3 完成后自动进入 Phase 4/5。

2026-06-07 07:45 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 65/445 steps
train_log_mtime: 2026-06-07 07:42:45
GPU: 8 张 H200 仍占用约 124GB/卡，训练 session 仍存在
latest_metric: loss 0.07143 / mean_token_accuracy 0.9717 at step 60
```

- 当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：本次检查没有新 step，也没有 checkpoint；仍在 cycle 0 Phase 3 SFT。训练 token accuracy 仍不能当 tau2 task accuracy。

2026-06-07 07:48 CST：

- 再次检查到新 step，确认训练没有卡住：

```text
STATUS=running
progress: 66/445 steps = 14.83%
elapsed_train_time: 3:59:10
observed_step_time: about 218.8s/step
rough_remaining_for_cycle0_sft_if_speed_holds: about 23.0 hours
train_log_mtime: 2026-06-07 07:46:28
latest_train_loss: 0.07143 at step 60
latest_mean_token_accuracy: 0.9717 at step 60
GPU: 8 张 H200 继续训练，基本 99%-100% 利用，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：仍在 cycle 0 Phase 3 SFT；Phase 4 Router / Phase 5 E2E 尚未开始。正式每轮 Skills / LLM / Router / Full accuracy 仍需等待 checkpoint 和 E2E 产物。

2026-06-07 07:50 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 67/445 steps = 15.06%
elapsed_train_time: 4:02:51
observed_step_time: about 219.5s/step
rough_remaining_for_cycle0_sft_if_speed_holds: about 23.0 hours
train_log_mtime: 2026-06-07 07:50:09
latest_train_loss: 0.07143 at step 60
latest_mean_token_accuracy: 0.9717 at step 60
GPU: 8 张 H200 继续训练，100% 左右利用，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：cycle 0 Phase 3 SFT 继续推进；Router / E2E / final aggregation 尚未开始。

2026-06-07 07:54 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 68/445 steps = 15.28%
elapsed_train_time: 4:06:26
observed_step_time: about 218.3s/step
rough_remaining_for_cycle0_sft_if_speed_holds: about 22.9 hours
train_log_mtime: 2026-06-07 07:53:45
latest_train_loss: 0.07143 at step 60
latest_mean_token_accuracy: 0.9717 at step 60
GPU: 8 张 H200 继续训练，约 99%-100% 利用，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：cycle 0 Phase 3 SFT 正常推进；还没到 Phase 4/5。

2026-06-07 07:55 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 68/445 steps
train_log_mtime: 2026-06-07 07:53:45
processes: tau2_train_wrapper + accelerate launch 仍存在
GPU: 8 张 H200 仍占用约 124GB/卡，利用率约 99%-100%
latest_metric: loss 0.07143 / mean_token_accuracy 0.9717 at step 60
```

- 当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：本次检查没有新 step，也没有 checkpoint；仍在 cycle 0 Phase 3 SFT。

2026-06-07 07:58 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 69/445 steps = 15.51%
elapsed_train_time: 4:10:02
observed_step_time: about 217.5s/step
rough_remaining_for_cycle0_sft_if_speed_holds: about 22.7 hours
train_log_mtime: 2026-06-07 07:57:20
latest_train_loss: 0.07143 at step 60
latest_mean_token_accuracy: 0.9717 at step 60
GPU: 8 张 H200 继续训练，约 99%-100% 利用，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：cycle 0 Phase 3 SFT 继续推进；还没到 Router / E2E。

2026-06-07 07:59 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 69/445 steps
train_log_mtime: 2026-06-07 07:57:20
processes: tau2_train_wrapper + accelerate launch 仍存在
GPU: 8 张 H200 仍占用约 124GB/卡，利用率有波动但训练 session 正常
latest_metric: loss 0.07143 / mean_token_accuracy 0.9717 at step 60
```

- 当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：本次检查没有新 step，也没有 checkpoint；仍在 cycle 0 Phase 3 SFT。

2026-06-07 08:03 CST：

- 再次检查到新 step，并出现 step 70 训练指标：

```text
STATUS=running
progress: 70/445 steps = 15.73%
elapsed_train_time: 4:13:47
observed_step_time: about 219.7s/step
rough_remaining_for_cycle0_sft_if_speed_holds: about 22.9 hours
train_log_mtime: 2026-06-07 08:01:05
loss: 0.05096
grad_norm: 0.7109
learning_rate: 9.739e-06
mean_token_accuracy: 0.9764
epoch: 0.7876
GPU: 8 张 H200 继续训练，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 注意：`mean_token_accuracy=0.9764` 是训练 token 指标，不是 tau2 task accuracy。正式 Skills / LLM / Router / Full accuracy 仍要等 checkpoint、router 和 E2E 产物。

2026-06-07 08:04 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 70/445 steps
train_log_mtime: 2026-06-07 08:01:05
processes: tau2_train_wrapper + accelerate launch 仍存在
GPU: 8 张 H200 仍占用约 124GB/卡，训练 session 正常
latest_metric: loss 0.05096 / mean_token_accuracy 0.9764 at step 70
```

- 当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：本次检查没有新 step，也没有 checkpoint；仍在 cycle 0 Phase 3 SFT。

2026-06-07 08:05 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 71/445 steps = 15.96%
elapsed_train_time: 4:17:29
observed_step_time: about 220.4s/step
rough_remaining_for_cycle0_sft_if_speed_holds: about 22.9 hours
train_log_mtime: 2026-06-07 08:04:47
latest_train_loss: 0.05096 at step 70
latest_mean_token_accuracy: 0.9764 at step 70
GPU: 8 张 H200 继续训练，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：cycle 0 Phase 3 SFT 继续推进；Phase 4 Router / Phase 5 E2E 尚未开始。

2026-06-07 08:09 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 72/445 steps = 16.18%
elapsed_train_time: 4:21:12
observed_step_time: about 221.3s/step
rough_remaining_for_cycle0_sft_if_speed_holds: about 22.9 hours
train_log_mtime: 2026-06-07 08:08:30
latest_train_loss: 0.05096 at step 70
latest_mean_token_accuracy: 0.9764 at step 70
GPU: 8 张 H200 继续训练，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：cycle 0 Phase 3 SFT 正常推进；还没到 Router / E2E。

2026-06-07 08:10 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 72/445 steps
train_log_mtime: 2026-06-07 08:08:30
processes: tau2_train_wrapper + accelerate launch 仍存在
GPU: 8 张 H200 仍占用约 124GB/卡，训练 session 正常
latest_metric: loss 0.05096 / mean_token_accuracy 0.9764 at step 70
```

- 当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：本次检查没有新 step，也没有 checkpoint；仍在 cycle 0 Phase 3 SFT。

2026-06-07 08:14 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 73/445 steps = 16.40%
elapsed_train_time: 4:24:53
observed_step_time: about 221.3s/step
rough_remaining_for_cycle0_sft_if_speed_holds: about 22.9 hours
train_log_mtime: 2026-06-07 08:12:12
latest_train_loss: 0.05096 at step 70
latest_mean_token_accuracy: 0.9764 at step 70
GPU: 8 张 H200 继续训练，约 99%-100% 利用，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：cycle 0 Phase 3 SFT 正常推进；还没到 Router / E2E。

2026-06-07 08:15 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 73/445 steps
train_log_mtime: 2026-06-07 08:12:12
processes: tau2_train_wrapper + accelerate launch 仍存在
GPU: 8 张 H200 仍占用约 124GB/卡，训练 session 正常
latest_metric: loss 0.05096 / mean_token_accuracy 0.9764 at step 70
```

- 当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：本次检查没有新 step，也没有 checkpoint；仍在 cycle 0 Phase 3 SFT。

2026-06-07 08:16 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 74/445 steps = 16.63%
elapsed_train_time: 4:28:27
observed_step_time: about 218.9s/step
rough_remaining_for_cycle0_sft_if_speed_holds: about 22.6 hours
train_log_mtime: 2026-06-07 08:15:45
latest_train_loss: 0.05096 at step 70
latest_mean_token_accuracy: 0.9764 at step 70
GPU: 8 张 H200 继续训练，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：cycle 0 Phase 3 SFT 继续推进；后续 Router / E2E 仍等待 checkpoint。

2026-06-07 08:20 CST：

- 按新加入论文再次复核项目和指标口径：
  - 已读 `../routerevolving-2.pdf` 全文。
  - 已重读 `scaling/run_full_pipeline.sh`、`collect_traces.py`、`traces_to_sft.py`、`tau2_train_wrapper.sh`、`train_router_simple.py`、`run_e2e_ablation_simple.py`、`aggregate_cycles.py`、`src/skills.py`。
  - 本次必须按 MERA 三轨口径报告：初始 small accuracy、Skills 变体、LLM adapter 任务级 accuracy、Router accuracy、Full=Skills+LLM+Router 的 E2E task pass / fallback / cost。

- 最新 35B SFT 状态：

```text
STATUS=running
progress: 75/445 steps = 16.85%
elapsed_train_time: 4:31:59
observed_step_time: about 217.0s/step
rough_remaining_for_cycle0_sft_if_speed_holds: about 22.3 hours
train_log_mtime: 2026-06-07 08:19:17
latest_train_loss: 0.05096 at step 70
latest_mean_token_accuracy: 0.9764 at step 70
GPU: 8 张 H200 继续训练，显存约 124GB/卡，利用率约 99%-100%
```

- 当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：
  - 现在仍是 cycle 0 Phase 3 SFT，没有到 Router / Full 结果。
  - 训练日志里的 `mean_token_accuracy=0.9764` 只作为过程指标，不能写成 tau2 task accuracy。
  - 初始 accuracy/cost 已固定：`small_success=42/74=56.76%`，`final_success=59/74=79.73%`，`observed_api_cost=$3.49623640`。
  - 等 checkpoint 出来后，自动进入 Router train 和 E2E ablation；每轮结果从 `e2e_ablation_summary.json` 和后续 35B adapter tau2 task eval 填表。

2026-06-07 08:23 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 76/445 steps = 17.08%
elapsed_train_time: 4:35:35
observed_step_time: about 216.5s/step
rough_remaining_for_cycle0_sft_if_speed_holds: about 22.2 hours
train_log_mtime: 2026-06-07 08:22:53
latest_train_loss: 0.05096 at step 70
latest_mean_token_accuracy: 0.9764 at step 70
GPU: 8 张 H200 继续训练，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：仍在 cycle 0 Phase 3 SFT；Router / Full accuracy 还不能填。

2026-06-07 08:26 CST：

- 发现并修复一个会影响后续闭环的必要问题：
  - 原 pipeline 在 `cycle > 0` 时会把上一轮 `llm_adapter/checkpoint-best` 作为 `small_model` 传给 tau2。
  - 但 tau2 scaling adapter 原来只会把 agent model 发到 CommonStack；本地 checkpoint 路径不能直接作为 CommonStack 模型调用。
  - 如果不修，cycle1 起会出现“看起来用了训练后模型，实际没有正确服务本地 35B adapter”的风险。

- 已做最小修复：
  - `experiments/scaling/benches/tau2_bench/adapter.py`：仅当 agent model 是 `openai/evol-llm-student` / `evol-llm-student` 时，使用 `TAU2_LOCAL_API_BASE` 的本地 vLLM；large model 继续使用 CommonStack。
  - `scaling/run_full_pipeline.sh`：cycle1 起检测上一轮 `llm_adapter/checkpoint-best` 后，先用 `experiments/tau2_stage2/code/training/eval/vllm_serve.sh` 以 `TP_SIZE=8` 启动本地 35B vLLM，再跑 trace collection，Phase 1 结束后停止 vLLM，释放 8 卡给下一轮 SFT。

- 已验证：

```text
python3 -m py_compile experiments/scaling/benches/tau2_bench/adapter.py: pass
bash -n scaling/run_full_pipeline.sh: pass
local student alias args: openai/evol-llm-student -> http://127.0.0.1:8050/v1
remote eval deps: vllm 0.20.2 / tau2 / openai import OK
eval_tasks: 35
heldout_tasks: 15
```

- 这个修复是为了保证后续 cycle 真正使用训练后的 35B adapter 参与 `collect_traces`，不是替换模型或降级任务。

2026-06-07 08:27 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 77/445 steps = 17.30%
elapsed_train_time: 4:39:10
observed_step_time: about 216.0s/step
rough_remaining_for_cycle0_sft_if_speed_holds: about 22.1 hours
train_log_mtime: 2026-06-07 08:26:28
latest_train_loss: 0.05096 at step 70
latest_mean_token_accuracy: 0.9764 at step 70
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

- 判断：cycle 0 Phase 3 SFT 正常推进；刚同步的本地 vLLM 修复会在 checkpoint 后、cycle1 `collect_traces` 时生效。

2026-06-07 08:31 CST：

- 继续检查后续多轮训练链路，发现并修复第二个必要问题：
  - tau2_stage2 原生训练输出默认写到全局 `experiments/tau2_stage2/train_outputs/08_qwen3_6_35b_a3b_273`。
  - scaling 多轮模式如果每轮都复用同一个全局目录，后续 cycle 可能复用/覆盖 cycle0 checkpoint，导致每轮 LLM 产物不独立。

- 已做最小修复：
  - `experiments/scaling/tau2_train_wrapper.sh`：在 `MODE=scaling_traces` 时向训练后端传入 `SCALING_OUTPUT_DIR=$TRAIN_OUTPUT_DIR`，并要求 checkpoint 直接出现在本轮 `cycle_N/llm_adapter/`。
  - `experiments/tau2_stage2/code/training/orchestration/train_all.sh`：当 `SCALING_OUTPUT_DIR` 存在时，生成一份临时 run yaml，把 `training.output_dir` 改成本轮 cycle 的 `llm_adapter` 目录；日志、`STATUS`、`checkpoint-best` 都写入该 cycle 目录。

- 已验证：

```text
bash -n experiments/scaling/tau2_train_wrapper.sh: pass
bash -n experiments/tau2_stage2/code/training/orchestration/train_all.sh: pass
bash -n scaling/run_full_pipeline.sh: pass
python3 -m py_compile experiments/scaling/benches/tau2_bench/adapter.py: pass
```

- 说明：
  - 这不是换模型，也不是降级配置。
  - 这是为了保证后续 cycle 的 LLM adapter 真实独立产出，最终能按每轮填 `Skills + LLM + Router` 的结果。

- 同时检查训练进度：

```text
STATUS=running
progress: 78/445 steps = 17.53%
elapsed_train_time: 4:42:49
observed_step_time: about 217.0s/step
rough_remaining_for_cycle0_sft_if_speed_holds: about 22.1 hours
train_log_mtime: 2026-06-07 08:30:07
checkpoint-best: missing
cycle_0/e2e_ablation_summary.json: missing
cycle_1/traces.jsonl: missing
```

2026-06-07 08:32 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 78/445 steps
train_log_mtime: 2026-06-07 08:30:07
tmux: cwy35b_resume3_cwy_35b_joint_20260606_165203_193736 仍存在
processes: tau2_train_wrapper + accelerate launch 仍存在
GPU: 8 张 H200 仍占用约 124GB/卡，有训练负载
latest_train_loss: 0.05096 at step 70
latest_mean_token_accuracy: 0.9764 at step 70
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：本次检查无新 step，但训练进程和 GPU 负载正常；仍等待 cycle 0 SFT checkpoint。

2026-06-07 08:34 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 79/445 steps = 17.75%
elapsed_train_time: 4:46:24
observed_step_time: about 216.4s/step
rough_remaining_for_cycle0_sft_if_speed_holds: about 22.0 hours
train_log_mtime: 2026-06-07 08:33:42
latest_train_loss: 0.05096 at step 70
latest_mean_token_accuracy: 0.9764 at step 70
GPU: 8 张 H200 基本 99%-100% 利用，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：仍在 cycle 0 Phase 3 SFT；Router / E2E 还不能填结果。

2026-06-07 08:35 CST：

- 检查到一个正在运行进程的衔接风险并已补救：
  - cycle0 的 SFT 进程是在 `SCALING_OUTPUT_DIR` 修复前启动的，训练实际仍写到全局目录 `experiments/tau2_stage2/train_outputs/08_qwen3_6_35b_a3b_273`。
  - 但 wrapper 文件已同步成新版本，训练结束后可能会从 `cycle_0/llm_adapter/checkpoint-best` 检查产物。
  - 为避免训练完成后因为 checkpoint 路径不一致导致链路中断，已预先创建 symlink：

```text
results/cwy_35b_joint_20260606_165203/cycle_0/llm_adapter/checkpoint-best
-> router-skills-evolve/experiments/tau2_stage2/train_outputs/08_qwen3_6_35b_a3b_273/checkpoint-best
```

- 当前目标 checkpoint 还没生成，所以 symlink 暂时不 resolve；等全局 `checkpoint-best` 出来后，`cycle_0/llm_adapter/checkpoint-best` 会自动可用。
- 这只影响 cycle0 热修复衔接；cycle1 起已使用 `SCALING_OUTPUT_DIR`，会直接写到各自 `cycle_N/llm_adapter/`。

2026-06-07 08:37 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 79/445 steps
train_log_mtime: 2026-06-07 08:33:42
tmux: cwy35b_resume3_cwy_35b_joint_20260606_165203_193736 仍存在
processes: tau2_train_wrapper + accelerate launch 仍存在
GPU: 8 张 H200 仍占用约 124GB/卡，有训练负载
latest_train_loss: 0.05096 at step 70
latest_mean_token_accuracy: 0.9764 at step 70
```

- checkpoint 衔接状态：

```text
global checkpoint-best: missing
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
symlink_target: experiments/tau2_stage2/train_outputs/08_qwen3_6_35b_a3b_273/checkpoint-best
cycle_0/router/router.joblib: missing
cycle_0/e2e_ablation_summary.json: missing
```

- 判断：无新 step，但训练仍正常；继续等待 cycle 0 SFT checkpoint。

2026-06-07 08:38 CST：

- 检查到新 step 和新训练指标：

```text
STATUS=running
progress: 80/445 steps = 17.98%
elapsed_train_time: 4:50:07
observed_step_time: about 218.6s/step
rough_remaining_for_cycle0_sft_if_speed_holds: about 22.2 hours
train_log_mtime: 2026-06-07 08:37:26
latest_train_loss: 0.07003 at step 80
latest_grad_norm: 0.9258
latest_learning_rate: 9.615e-06
latest_mean_token_accuracy: 0.9726 at step 80
latest_epoch: 0.9001
```

- 进程树确认：

```text
tau2_train_wrapper.sh
-> train_pipeline.sh
-> train_all.sh
-> accelerate launch
-> 8 x training.train rank process
```

- 当前仍未生成：

```text
global checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：cycle 0 Phase 3 SFT 正常推进；训练指标仍只是过程 token 指标，不是 tau2 task accuracy。

2026-06-07 08:39 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 80/445 steps
train_log_mtime: 2026-06-07 08:37:26
tmux: cwy35b_resume3_cwy_35b_joint_20260606_165203_193736 仍存在
processes: tau2_train_wrapper -> train_pipeline -> train_all -> accelerate 仍存在
GPU: 8 张 H200 仍占用约 124GB/卡，有训练负载
latest_train_loss: 0.07003 at step 80
latest_mean_token_accuracy: 0.9726 at step 80
```

- 当前仍未生成：

```text
global checkpoint-best
global checkpoint-final
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
```

- checkpoint symlink 衔接仍在：

```text
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
target: experiments/tau2_stage2/train_outputs/08_qwen3_6_35b_a3b_273/checkpoint-best
```

- 判断：本次检查没有新 step；训练仍正常，继续等待 cycle 0 SFT checkpoint。

2026-06-07 08:40 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 80/445 steps
train_log_mtime: 2026-06-07 08:37:26
processes: tau2_train_wrapper -> train_pipeline -> train_all -> accelerate 仍存在
GPU: 8 张 H200 仍占用约 124GB/卡，有训练负载
checkpoint-best: missing
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
cycle_0/e2e_ablation_summary.json: missing
```

- 判断：无新 step；仍在 cycle 0 Phase 3 SFT。

2026-06-07 08:42 CST：

- 检查到新 step：

```text
STATUS=running
progress: 81/445 steps = 18.20%
elapsed_train_time: 4:53:42
observed_step_time: about 217.4s/step
rough_remaining_for_cycle0_sft_if_speed_holds: about 22.0 hours
train_log_mtime: 2026-06-07 08:41:00
latest_train_loss: 0.07003 at step 80
latest_mean_token_accuracy: 0.9726 at step 80
GPU: 8 张 H200 仍占用约 124GB/卡，有训练负载
```

- 当前仍未生成：

```text
global checkpoint-best
global checkpoint-final
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- checkpoint symlink 衔接仍在：

```text
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
target: experiments/tau2_stage2/train_outputs/08_qwen3_6_35b_a3b_273/checkpoint-best
```

- 判断：cycle 0 Phase 3 SFT 继续正常推进；还没到 Router / E2E。

2026-06-07 08:43 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 81/445 steps
train_log_mtime: 2026-06-07 08:41:00
tmux: cwy35b_resume3_cwy_35b_joint_20260606_165203_193736 仍存在
processes: tau2_train_wrapper -> train_pipeline -> train_all -> accelerate 仍存在
GPU: 8 张 H200 仍占用约 124GB/卡，有训练负载
checkpoint-best: missing
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
cycle_0/e2e_ablation_summary.json: missing
```

- 判断：无新 step；仍在 cycle 0 Phase 3 SFT。

2026-06-07 08:45 CST：

- 按用户新要求再次确认：已经先读项目主线，再读论文 `routerevolving-2.pdf`；本文档的汇报口径固定为 `initial accuracy/cost` + 每轮 `Skills / LLM / Router / Full` 的 `accuracy / fallback / cost`。
- H200 当前仍在正式 35B cycle 0 Phase 3 SFT：

```text
STATUS=running
progress: 82/445 steps = 18.43%
elapsed_train_time: 4:57:23
observed_step_time: about 218.4s/step
rough_remaining_for_cycle0_sft_if_speed_holds: about 22.0 hours
train_log_mtime: 2026-06-07 08:44:41
latest_train_loss: 0.07003 at step 80
latest_mean_token_accuracy: 0.9726 at step 80
GPU: 8 张 H200 约 99%-100% 利用，显存约 124GB/卡
```

- 当前仍未生成：

```text
global checkpoint-best
global checkpoint-final
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 已有可报告的真实初始数：

```text
initial_small_accuracy = 42/74 = 56.76%
initial_fallback_needed = 32/74 = 43.24%
initial_task_pass_after_fallback = 59/74 = 79.73%
initial_observed_api_cost = $3.49623640
cycle0_skillbook = 15 skills / 14 procedures
cycle0_sft_pairs = 17 hard tasks -> 17 pairs
```

- 还不能填每轮训练后的 `Skills / LLM / Router / Full` accuracy：
  - Router / E2E 要等 35B checkpoint 产出后自动进入 Phase 4/5。
  - `mean_token_accuracy=0.9726` 只是 SFT token 训练指标，不是 tau2 task accuracy。

2026-06-07 08:49 CST：

- 再次检查 phase3 训练日志，确认训练继续推进：

```text
STATUS=running
progress: 83/445 steps = 18.65%
elapsed_train_time: 5:01:05
observed_step_time: about 219.6s/step
train_log_mtime: 2026-06-07 08:48
latest_train_loss: 0.07003 at step 80
latest_mean_token_accuracy: 0.9726 at step 80
```

- 当前仍未生成：

```text
global checkpoint-best
global checkpoint-final
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：仍在 cycle 0 Phase 3 SFT；Phase 4 Router / Phase 5 E2E / cycle1 还没开始。

2026-06-07 08:50 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 83/445 steps = 18.65%
train_log_mtime: 2026-06-07 08:48:24
latest_train_loss: 0.07003 at step 80
latest_mean_token_accuracy: 0.9726 at step 80
GPU: 8 张 H200 仍占用约 124GB/卡，有训练负载
```

- 当前仍未生成：

```text
global checkpoint-best
global checkpoint-final
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：本次检查没有新 step，也没有 checkpoint；仍在 cycle 0 Phase 3 SFT，继续等待 35B checkpoint 后自动接 Phase 4/5。

2026-06-07 08:51 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 83/445 steps = 18.65%
train_log_mtime: 2026-06-07 08:48:24
latest_train_loss: 0.07003 at step 80
latest_mean_token_accuracy: 0.9726 at step 80
GPU: 8 张 H200 仍占用约 124GB/卡，有训练负载
```

- 当前仍未生成：

```text
global checkpoint-best
global checkpoint-final
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：无新 checkpoint / router / E2E 产物；仍在 cycle 0 Phase 3 SFT。距离上一条 step 还短，GPU 仍有负载，继续等待下一条训练日志。

2026-06-07 08:52 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 84/445 steps = 18.88%
elapsed_train_time: 5:04:44
observed_step_time: about 219.4s/step
rough_remaining_for_cycle0_sft_if_speed_holds: about 22.0 hours
train_log_mtime: 2026-06-07 08:52:02
latest_train_loss: 0.07003 at step 80
latest_mean_token_accuracy: 0.9726 at step 80
GPU: 8 张 H200 约 99%-100% 利用，显存约 124GB/卡
```

- 当前仍未生成：

```text
global checkpoint-best
global checkpoint-final
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：cycle 0 Phase 3 SFT 继续推进；还没到 Router / E2E。

2026-06-07 08:53 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 84/445 steps = 18.88%
train_log_mtime: 2026-06-07 08:52:02
latest_train_loss: 0.07003 at step 80
latest_mean_token_accuracy: 0.9726 at step 80
processes: tau2_train_wrapper -> train_pipeline -> train_all -> accelerate -> training.train 仍存在
GPU: 8 张 H200 仍占用约 124GB/卡，有训练负载
```

- 当前仍未生成：

```text
global checkpoint-best
global checkpoint-final
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：本次检查没有新 step，但训练进程树完整、GPU 有负载；仍在 cycle 0 Phase 3 SFT。

2026-06-07 08:54 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 84/445 steps = 18.88%
train_log_mtime: 2026-06-07 08:52:02
latest_train_loss: 0.07003 at step 80
latest_mean_token_accuracy: 0.9726 at step 80
GPU: 8 张 H200 仍占用约 124GB/卡，有训练负载
```

- 当前仍未生成：

```text
global checkpoint-best
global checkpoint-final
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：无新 checkpoint / router / E2E；仍在 cycle 0 Phase 3 SFT。

2026-06-07 08:55 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 84/445 steps = 18.88%
train_log_mtime: 2026-06-07 08:52:02
latest_train_loss: 0.07003 at step 80
latest_mean_token_accuracy: 0.9726 at step 80
processes: tau2_train_wrapper -> train_pipeline -> train_all -> accelerate -> training.train 仍存在
GPU: 8 张 H200 仍占用约 124GB/卡，有训练负载
```

- 当前仍未生成：

```text
global checkpoint-best
global checkpoint-final
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：本次检查没有新 step；训练进程和 GPU 负载正常，仍在 cycle 0 Phase 3 SFT。

2026-06-07 08:56 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 85/445 steps = 19.10%
elapsed_train_time: 5:08:19
observed_step_time: about 218.1s/step
rough_remaining_for_cycle0_sft_if_speed_holds: about 21.8 hours
train_log_mtime: 2026-06-07 08:55:37
latest_train_loss: 0.07003 at step 80
latest_mean_token_accuracy: 0.9726 at step 80
GPU: 8 张 H200 仍占用约 124GB/卡，有训练负载
```

- 当前仍未生成：

```text
global checkpoint-best
global checkpoint-final
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：cycle 0 Phase 3 SFT 继续推进；Router / E2E 尚未开始。

2026-06-07 08:57 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 85/445 steps = 19.10%
train_log_mtime: 2026-06-07 08:55:37
latest_train_loss: 0.07003 at step 80
latest_mean_token_accuracy: 0.9726 at step 80
GPU: 8 张 H200 仍占用约 124GB/卡，训练负载约 82%-100%
```

- 当前仍未生成：

```text
global checkpoint-best
global checkpoint-final
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：本次检查没有新 checkpoint；仍在 cycle 0 Phase 3 SFT，Router / E2E 未开始。

2026-06-07 08:58 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 85/445 steps = 19.10%
train_log_mtime: 2026-06-07 08:55:37
latest_train_loss: 0.07003 at step 80
latest_mean_token_accuracy: 0.9726 at step 80
GPU: 8 张 H200 仍占用约 124GB/卡，训练负载约 100%
```

- 当前仍未生成：

```text
global checkpoint-best
global checkpoint-final
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：本次检查没有新 step / checkpoint；训练仍在 cycle 0 Phase 3 SFT。

2026-06-07 08:59 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 86/445 steps = 19.33%
elapsed_train_time: 5:11:57
observed_step_time: about 218.0s/step
rough_remaining_for_cycle0_sft_if_speed_holds: about 21.7 hours
train_log_mtime: 2026-06-07 08:59:15
latest_train_loss: 0.07003 at step 80
latest_mean_token_accuracy: 0.9726 at step 80
GPU: 8 张 H200 约 99%-100% 利用，显存约 124GB/卡
```

- 当前仍未生成：

```text
global checkpoint-best
global checkpoint-final
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：cycle 0 Phase 3 SFT 继续推进；Router / E2E 尚未开始。

2026-06-07 09:01 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 86/445 steps = 19.33%
train_log_mtime: 2026-06-07 08:59:15
latest_train_loss: 0.07003 at step 80
latest_mean_token_accuracy: 0.9726 at step 80
rough_remaining_for_cycle0_sft_if_speed_holds: about 21.7 hours
GPU: 8 张 H200 仍占用约 124GB/卡，训练负载约 99%-100%
```

- 当前仍未生成：

```text
global checkpoint-best
global checkpoint-final
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：本次检查没有新 step / checkpoint；仍在 cycle 0 Phase 3 SFT。

2026-06-07 09:02 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 86/445 steps = 19.33%
train_log_mtime: 2026-06-07 08:59:15
latest_train_loss: 0.07003 at step 80
latest_mean_token_accuracy: 0.9726 at step 80
GPU: 8 张 H200 仍占用约 124GB/卡，有训练负载
```

- 当前仍未生成：

```text
global checkpoint-best
global checkpoint-final
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：本次检查没有新 step；仍在 cycle 0 Phase 3 SFT，Router / E2E 未开始。

2026-06-07 09:03 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 87/445 steps = 19.55%
elapsed_train_time: 5:15:35
observed_step_time: about 218.1s/step
rough_remaining_for_cycle0_sft_if_speed_holds: about 21.7 hours
train_log_mtime: 2026-06-07 09:02:54
latest_train_loss: 0.07003 at step 80
latest_mean_token_accuracy: 0.9726 at step 80
GPU: 8 张 H200 仍占用约 124GB/卡，有训练负载
```

- 当前仍未生成：

```text
global checkpoint-best
global checkpoint-final
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：cycle 0 Phase 3 SFT 继续推进；Router / E2E 尚未开始。

2026-06-07 09:04 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 87/445 steps = 19.55%
train_log_mtime: 2026-06-07 09:02:54
latest_train_loss: 0.07003 at step 80
latest_mean_token_accuracy: 0.9726 at step 80
GPU: 8 张 H200 仍占用约 124GB/卡，有训练负载
```

- 当前仍未生成：

```text
global checkpoint-best
global checkpoint-final
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：本次检查没有新 step / checkpoint；仍在 cycle 0 Phase 3 SFT。

2026-06-07 09:05 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 87/445 steps = 19.55%
train_log_mtime: 2026-06-07 09:02:54
latest_train_loss: 0.07003 at step 80
latest_mean_token_accuracy: 0.9726 at step 80
GPU: 8 张 H200 仍占用约 124GB/卡，训练负载仍在；单卡利用率瞬时有波动
```

- 当前仍未生成：

```text
global checkpoint-best
global checkpoint-final
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：本次检查没有新 step / checkpoint；仍在 cycle 0 Phase 3 SFT。

2026-06-07 09:06 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 87/445 steps = 19.55%
train_log_mtime: 2026-06-07 09:02:54
latest_train_loss: 0.07003 at step 80
latest_mean_token_accuracy: 0.9726 at step 80
GPU: 8 张 H200 仍占用约 124GB/卡，有训练负载
```

- 当前仍未生成：

```text
global checkpoint-best
global checkpoint-final
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：本次检查没有新 step / checkpoint；仍在 cycle 0 Phase 3 SFT。

2026-06-07 09:07 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 88/445 steps = 19.78%
elapsed_train_time: 5:19:12
observed_step_time: about 217.6s/step
rough_remaining_for_cycle0_sft_if_speed_holds: about 21.6 hours
train_log_mtime: 2026-06-07 09:06:30
latest_train_loss: 0.07003 at step 80
latest_mean_token_accuracy: 0.9726 at step 80
GPU: 8 张 H200 仍占用约 124GB/卡，有训练负载
```

- 当前仍未生成：

```text
global checkpoint-best
global checkpoint-final
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：cycle 0 Phase 3 SFT 继续推进；Router / E2E 尚未开始。

2026-06-07 09:08 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 88/445 steps = 19.78%
train_log_mtime: 2026-06-07 09:06:30
latest_train_loss: 0.07003 at step 80
latest_mean_token_accuracy: 0.9726 at step 80
GPU: 8 张 H200 仍占用约 124GB/卡，有训练负载
```

- 当前仍未生成：

```text
global checkpoint-best
global checkpoint-final
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：本次检查没有新 step / checkpoint；仍在 cycle 0 Phase 3 SFT。

2026-06-07 09:09 CST：

- 健康检查：

```text
STATUS=running
progress: last logged 88/445 steps = 19.78%
train_log_mtime: 2026-06-07 09:06:30
latest_train_loss: 0.07003 at step 80
latest_mean_token_accuracy: 0.9726 at step 80
GPU: 8 张 H200 仍占用约 124GB/卡，训练负载约 99%-100%
```

- 当前仍未生成：

```text
global checkpoint-best
global checkpoint-final
cycle_0/llm_adapter/checkpoint-best: symlink exists, currently unresolved
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：本次检查没有新 step / checkpoint；仍在 cycle 0 Phase 3 SFT。

2026-06-07 07:34 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 62/445 steps
elapsed_train_time: 3:44:35
observed_step_time: about 215.7s/step
train_log_mtime: 2026-06-07 07:31:53
latest_train_loss: 0.07143 at step 60
latest_mean_token_accuracy: 0.9717 at step 60
GPU: 8 张 H200 仍占用约 124GB/卡，训练继续推进
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

- 判断：仍在 cycle 0 Phase 3 SFT；Phase 4/5 尚未开始。

2026-06-07 07:24 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 59/445 steps
elapsed_train_time: 3:33:49
observed_step_time: about 216.5s/step
train_log_mtime: 2026-06-07 07:21:07
latest_train_loss: 0.05547 at step 50
latest_mean_token_accuracy: 0.9788 at step 50
GPU: 8 张 H200 仍占用约 124GB/卡，训练继续推进
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

- 判断：仍在 cycle 0 Phase 3 SFT；下一条训练指标预计 step 60 左右出现。

2026-06-07 07:25 CST：

- 再次检查到新 step，并出现 step 60 训练指标：

```text
STATUS=running
progress: 60/445 steps
elapsed_train_time: 3:37:27
observed_step_time: about 217.1s/step
train_log_mtime: 2026-06-07 07:24:46
loss: 0.07143
grad_norm: 0.6953
learning_rate: 9.839e-06
mean_token_accuracy: 0.9717
epoch: 0.6751
GPU: 8 张 H200 高利用，显存约 124GB/卡，训练继续推进
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

- 注意：`mean_token_accuracy=0.9717` 是训练 token 指标，不等于 tau2 task accuracy；正式 task accuracy 仍要等 checkpoint 后补 35B adapter tau2 eval。

2026-06-07 07:29 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 61/445 steps
elapsed_train_time: 3:40:58
observed_step_time: about 215.2s/step
train_log_mtime: 2026-06-07 07:28:16
latest_train_loss: 0.07143 at step 60
latest_mean_token_accuracy: 0.9717 at step 60
GPU: 8 张 H200 高利用，显存约 124GB/卡，训练继续推进
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

- 判断：仍在 cycle 0 Phase 3 SFT；Phase 4/5 尚未开始。

2026-06-07 07:19 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 58/445 steps
elapsed_train_time: 3:30:14
observed_step_time: about 217.0s/step
train_log_mtime: 2026-06-07 07:17:32
latest_train_loss: 0.05547 at step 50
latest_mean_token_accuracy: 0.9788 at step 50
GPU: 8 张 H200 高利用，显存约 124GB/卡，训练继续推进
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

- 判断：仍在 cycle 0 Phase 3 SFT；Phase 4/5 尚未开始。

2026-06-07 07:14 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 57/445 steps
elapsed_train_time: 3:26:40
observed_step_time: about 218.3s/step
train_log_mtime: 2026-06-07 07:13:58
latest_train_loss: 0.05547 at step 50
latest_mean_token_accuracy: 0.9788 at step 50
GPU: 8 张 H200 高利用，显存约 124GB/卡，训练继续推进
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

- 判断：仍在 cycle 0 Phase 3 SFT；Phase 4/5 尚未开始。

2026-06-07 07:13 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 56/445 steps
elapsed_train_time: 3:23:05
observed_step_time: about 220.1s/step
train_log_mtime: 2026-06-07 07:10:24
latest_train_loss: 0.05547 at step 50
latest_mean_token_accuracy: 0.9788 at step 50
GPU: 8 张 H200 高利用，显存约 124GB/卡，训练继续推进
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

- 判断：仍在 cycle 0 Phase 3 SFT；Phase 4/5 尚未开始。

2026-06-07 07:03 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 54/445 steps
elapsed_train_time: 3:15:42
observed_step_time: about 218.2s/step
train_log_mtime: 2026-06-07 07:03:00
latest_train_loss: 0.05547 at step 50
latest_mean_token_accuracy: 0.9788 at step 50
GPU: 8 张 H200 高利用，显存约 124GB/卡，训练继续推进
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

- 判断：仍在 cycle 0 Phase 3 SFT；Phase 4 Router / Phase 5 E2E 尚未开始。

2026-06-07 07:08 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 55/445 steps
elapsed_train_time: 3:19:23
observed_step_time: about 219.0s/step
train_log_mtime: 2026-06-07 07:06:41
latest_train_loss: 0.05547 at step 50
latest_mean_token_accuracy: 0.9788 at step 50
GPU: 8 张 H200 仍占用约 124GB/卡，训练继续推进
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

- 判断：仍在 cycle 0 Phase 3 SFT；Phase 4/5 尚未开始。

2026-06-07 06:21 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 42/445 steps
elapsed_train_time: 2:32:07
observed_step_time: about 219.6s/step
train_log_mtime: 2026-06-07 06:19:25
latest_train_loss: 0.06273 at step 40
latest_mean_token_accuracy: 0.9765 at step 40
GPU: 8 张 H200 仍在训练，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

- 判断：cycle 0 Phase 3 SFT 继续推进；按当前速度，第一轮 SFT 仍是约 24 小时级别。

2026-06-07 06:26 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 43/445 steps
elapsed_train_time: 2:35:45
observed_step_time: about 219.1s/step
train_log_mtime: 2026-06-07 06:23:03
latest_train_loss: 0.06273 at step 40
latest_mean_token_accuracy: 0.9765 at step 40
GPU: 8 张 H200 仍占用约 124GB/卡，训练继续推进
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

- 判断：仍在 cycle 0 Phase 3 SFT；Router / E2E / 35B adapter task eval 都还没到执行点。

2026-06-07 06:30 CST：

- 再次检查到新 step，这轮从 43 继续到 45：

```text
STATUS=running
progress: 45/445 steps
elapsed_train_time: 2:43:07
observed_step_time: about 220.4s/step
train_log_mtime: 2026-06-07 06:30:25
latest_train_loss: 0.06273 at step 40
latest_mean_token_accuracy: 0.9765 at step 40
GPU: 8 张 H200 仍占用约 124GB/卡，训练继续推进
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

- 判断：cycle 0 Phase 3 SFT 正常推进；Phase 4 Router / Phase 5 E2E 尚未开始。

2026-06-07 06:35 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 46/445 steps
elapsed_train_time: 2:46:46
observed_step_time: about 220.1s/step
train_log_mtime: 2026-06-07 06:34:05
latest_train_loss: 0.06273 at step 40
latest_mean_token_accuracy: 0.9765 at step 40
GPU: 8 张 H200 仍占用约 124GB/卡，训练继续推进
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

- 判断：仍在 cycle 0 Phase 3 SFT；正式 Skills / Router / Full accuracy 还不能填。

2026-06-07 06:39 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 47/445 steps
elapsed_train_time: 2:50:23
observed_step_time: about 219.0s/step
train_log_mtime: 2026-06-07 06:37:41
latest_train_loss: 0.06273 at step 40
latest_mean_token_accuracy: 0.9765 at step 40
GPU: 8 张 H200 仍占用约 124GB/卡，训练继续推进
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

- 判断：仍在 cycle 0 Phase 3 SFT；Phase 4/5 还未开始。

2026-06-07 06:44 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 48/445 steps
elapsed_train_time: 2:53:59
observed_step_time: about 218.3s/step
train_log_mtime: 2026-06-07 06:41:18
latest_train_loss: 0.06273 at step 40
latest_mean_token_accuracy: 0.9765 at step 40
GPU: 8 张 H200 高利用，显存约 124GB/卡，训练继续推进
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

- 判断：仍在 cycle 0 Phase 3 SFT；下一条训练指标预计 step 50 左右出现。

2026-06-07 06:52 CST：

- 再次检查到新 step，并出现 step 50 训练指标：

```text
STATUS=running
progress: 51/445 steps
elapsed_train_time: 3:04:45
observed_step_time: about 216.1s/step
train_log_mtime: 2026-06-07 06:52:03
loss: 0.05547
grad_norm: 0.5156
learning_rate: 9.916e-06
mean_token_accuracy: 0.9788
epoch: 0.5626
GPU: 8 张 H200 仍占用约 124GB/卡，训练继续推进
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

- 注意：`mean_token_accuracy=0.9788` 仍只是训练 token 指标，不等于 tau2 task accuracy。正式 task accuracy 必须等 checkpoint 后跑 35B adapter tau2 eval。

2026-06-07 06:57 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 52/445 steps
elapsed_train_time: 3:08:23
observed_step_time: about 216.7s/step
train_log_mtime: 2026-06-07 06:55:41
latest_train_loss: 0.05547 at step 50
latest_mean_token_accuracy: 0.9788 at step 50
GPU: 8 张 H200 仍占用约 124GB/卡，训练继续推进
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

- 判断：仍在 cycle 0 Phase 3 SFT；Phase 4 Router 和 Phase 5 E2E 尚未开始。

2026-06-07 07:02 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 53/445 steps
elapsed_train_time: 3:12:02
observed_step_time: about 217.3s/step
train_log_mtime: 2026-06-07 06:59:20
latest_train_loss: 0.05547 at step 50
latest_mean_token_accuracy: 0.9788 at step 50
GPU: 8 张 H200 仍占用约 124GB/卡，训练继续推进
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

- 判断：仍在 cycle 0 Phase 3 SFT；Phase 4/5 尚未开始。

2026-06-07 05:29 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 28/445 steps
elapsed_train_time: 1:41:10
observed_step_time: about 216.2s/step
train_log_mtime: 2026-06-07 05:28:28
GPU: 8 张 H200 仍在训练，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
```

- 判断：仍处于 cycle 0 Phase 3 SFT；后续 Router / E2E 等 checkpoint 出来后自动接上。

2026-06-07 05:33 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 29/445 steps
elapsed_train_time: 1:44:50
observed_step_time: about 217.2s/step
train_log_mtime: 2026-06-07 05:32:08
GPU: 8 张 H200 仍在训练，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
```

- 判断：cycle 0 Phase 3 SFT 继续健康推进；后续 Router / E2E 仍等待 checkpoint。

2026-06-07 05:36 CST：

- 再次检查到新 step，并出现第三条训练指标：

```text
STATUS=running
progress: 30/445 steps
elapsed_train_time: 1:48:23
observed_step_time: about 215.9s/step
train_log_mtime: 2026-06-07 05:35:41
loss: 0.06071
grad_norm: 0.6602
learning_rate: 9.996e-06
mean_token_accuracy: 0.976
epoch: 0.3376
GPU: 8 张 H200 仍在训练，显存约 124GB/卡
```

- 注意：`mean_token_accuracy=0.976` 是训练 token 指标，不等于 tau2 task accuracy；正式 task / router / full accuracy 必须等 checkpoint 后 Phase 4/5 产物。
- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
```

- 判断：cycle 0 Phase 3 SFT 继续健康推进；后续 Router / E2E 仍等待 checkpoint。

2026-06-07 05:40 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 31/445 steps
elapsed_train_time: 1:52:01
observed_step_time: about 216.6s/step
train_log_mtime: 2026-06-07 05:39:19
latest_train_loss: 0.06071 at step 30
latest_mean_token_accuracy: 0.976 at step 30
GPU: 8 张 H200 仍在训练，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
```

- 判断：cycle 0 Phase 3 SFT 继续健康推进；后续 Router / E2E 仍等待 checkpoint。

2026-06-07 05:44 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 32/445 steps
elapsed_train_time: 1:55:41
observed_step_time: about 217.7s/step
train_log_mtime: 2026-06-07 05:42:59
latest_train_loss: 0.06071 at step 30
latest_mean_token_accuracy: 0.976 at step 30
GPU: 8 张 H200 仍在训练，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
```

- 判断：cycle 0 Phase 3 SFT 继续健康推进；后续 Router / E2E 仍等待 checkpoint。

2026-06-07 05:47 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 33/445 steps
elapsed_train_time: 1:59:18
observed_step_time: about 217.5s/step
train_log_mtime: 2026-06-07 05:46:36
latest_train_loss: 0.06071 at step 30
latest_mean_token_accuracy: 0.976 at step 30
GPU: 8 张 H200 仍在训练，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
```

- 判断：cycle 0 Phase 3 SFT 继续健康推进；后续 Router / E2E 仍等待 checkpoint。

2026-06-07 05:51 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 34/445 steps
elapsed_train_time: 2:02:57
observed_step_time: about 217.8s/step
train_log_mtime: 2026-06-07 05:50:15
latest_train_loss: 0.06071 at step 30
latest_mean_token_accuracy: 0.976 at step 30
GPU: 8 张 H200 仍在训练，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
```

- 判断：cycle 0 Phase 3 SFT 继续健康推进；后续 Router / E2E 仍等待 checkpoint。

2026-06-07 05:55 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 35/445 steps
elapsed_train_time: 2:06:34
observed_step_time: about 217.8s/step
train_log_mtime: 2026-06-07 05:53:52
latest_train_loss: 0.06071 at step 30
latest_mean_token_accuracy: 0.976 at step 30
GPU: 8 张 H200 仍在训练，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
```

- 判断：cycle 0 Phase 3 SFT 继续健康推进；后续 Router / E2E 仍等待 checkpoint。

2026-06-07 05:59 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 36/445 steps
elapsed_train_time: 2:10:07
observed_step_time: about 216.3s/step
train_log_mtime: 2026-06-07 05:57:25
latest_train_loss: 0.06071 at step 30
latest_mean_token_accuracy: 0.976 at step 30
GPU: 8 张 H200 仍在训练，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
```

- 判断：cycle 0 Phase 3 SFT 继续健康推进；后续 Router / E2E 仍等待 checkpoint。

2026-06-07 06:03 CST：

- 再次检查到新 step：

```text
STATUS=running
progress: 37/445 steps
elapsed_train_time: 2:13:47
observed_step_time: about 217.3s/step
train_log_mtime: 2026-06-07 06:01:05
latest_train_loss: 0.06071 at step 30
latest_mean_token_accuracy: 0.976 at step 30
GPU: 8 张 H200 仍在训练，显存约 124GB/卡
```

- 当前仍未生成：

```text
checkpoint-best
cycle_0/llm_adapter/checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
```

- 判断：cycle 0 Phase 3 SFT 继续健康推进；后续 Router / E2E 仍等待 checkpoint。

2026-06-07 00:49 CST：

- 已启动新的正式有效候选运行。
- H200 tmux session：

```text
cwy35b_20260606_164919
```

- 实验目录：

```text
results/cwy_35b_joint_20260606_164919
```

- 本次关键模型设置：

```text
SMALL_MODEL=deepseek/deepseek-v3.2
LARGE_MODEL=openai/openai/gpt-5.4-2026-03-05
TAU2_USER_MODEL=openai/openai/gpt-5.2
MODEL_SWEEP=08_qwen3_6_35b_a3b_273
```

- 已确认：

```text
mock=False
loaded 74 tasks
```

- 该次运行在写入 trace 前被主动停止，不计入正式结果。
- 原因：adapter 之前没有从 tau2 的 `user_scenario.instructions` 提取用户任务，导致 prompt/signature 为空或不稳定。
- 已修复：
  - `load_tasks`：按 `TAU2_DOMAIN=retail` 加载真实任务，再按 `split_tasks.json` 过滤 train/eval
  - `prompt`：使用 `reason_for_call / known_info / unknown_info / task_instructions`
  - `signature`：现在从具体任务内容开头生成，不再从空字符串或泛化 persona 生成
- H200 验证结果：前 5 个 train task 的 `prompt_len > 0`，signature 已包含具体任务描述。

2026-06-07 00:45 CST：

- 已重新启动正式 35B SLR 全流程。
- H200 tmux session：

```text
cwy35b_20260606_164508
```

- 实验目录：

```text
results/cwy_35b_joint_20260606_164508
```

- 已确认 Phase 1 是真实运行：

```text
mock=False
loaded 74 tasks
TAU2_DOMAIN=retail
```

- 该次运行随后被主动停止，不计入正式结果。
- 原因：repo 默认 `LARGE_MODEL=openai/gpt-5.4-2026-03-05` 被 LiteLLM 当作 provider 前缀处理，实际发到 CommonStack 变成 `gpt-5.4-2026-03-05`，返回 404。
- 已确认 CommonStack 支持的真实模型名是 `openai/gpt-5.4-2026-03-05`。
- 正式运行需要把 LiteLLM model 写成：

```text
LARGE_MODEL=openai/openai/gpt-5.4-2026-03-05
```

- 已做最小 API 连通性验证，返回 `OK`。

2026-06-07 00:43 CST：

- 第一次正式启动进入 Phase 1 前失败，未开始训练、未产生有效 accuracy。
- 失败原因：scaling 层把 `split=train` 传给底层 tau2 adapter；底层 adapter 的参数实际是 domain，于是错误寻找 `domains/train/tasks.json`。
- 已修复 `experiments/scaling/benches/tau2_bench/adapter.py`：现在先加载 `TAU2_DOMAIN=retail` 的真实 `tasks.json`，再用 `split_tasks.json` 过滤 train/eval。
- 已在 H200 验证真实任务加载：
  - train split：74 个任务
  - eval/test split：40 个任务
  - domain：retail
- 下一次启动从 cycle 0 重新跑，前一次失败目录只作为错误日志，不计入结果。

2026-06-07 00:19 CST：

- 已读取项目主线代码、pipeline、skills、router、LLM train wrapper、E2E ablation 聚合代码。
- 已读取论文 `routerevolving-2.pdf`。
- 已确认正式模型是仓库里的 `Qwen/Qwen3.6-35B-A3B`。
- 已确认 H200 空闲。
- 已完成 CPU 本机下载 `Qwen/Qwen3.6-35B-A3B` 到 HF cache。
- 已完成 CPU 到 H200 的模型同步。
- 已完成 H200 离线加载验证。

下载记录：

```text
CPU download log: tmp/cwy_hf_35b_seq_download.log
CPU cache: <hf-cache>/models--Qwen--Qwen3.6-35B-A3B
download method: sequential hf download, max-workers=1
worker HF cache: <hf-cache>/models--Qwen--Qwen3.6-35B-A3B
H200 files: 40
H200 model_type: qwen3_5_moe
H200 architecture: Qwen3_5MoeForConditionalGeneration
H200 vocab_size: 248077
```

下一步：

1. 启动正式 `N_CYCLES=4` / `SLR` / `mock=false` / `skip_llm=false`。
2. 每轮结束后更新本文档的 accuracy / cost / fallback / task pass。
3. 最后汇总初始 accuracy、每轮 Full SLR accuracy/cost，以及总成本。

## 错误修正记录

之前错误跑了：

```text
11_qwen3_30b_a3b_273
```

这不是本次正式主线。后续汇报不得把 30B 单模型结果混入 35B joint evolution。

## 最新权威状态

2026-06-07 09:13 CST：

- 已按用户新增论文要求重新读取项目主线和论文，本文档前面的“论文和项目口径”是当前报告口径。
- H200 正在跑正式 35B 联合演进，不是 30B，不是 mock。
- 当前仍处于 `cycle 0 / Phase 3 / 35B SFT`，Router / E2E / cycle1 还没有开始。

训练状态：

```text
worker: current private 8-GPU worker
repo: router-skills-evolve/
experiment: results/cwy_35b_joint_20260606_165203
model_config: 08_qwen3_6_35b_a3b_273
model: Qwen/Qwen3.6-35B-A3B
schedule: SLR = Skills -> LLM -> Router
n_cycles: 4
status: running
progress: 89/445 steps = 20.00%
elapsed_train_time: 5:23:40
latest_train_metric: loss 0.07003 / mean_token_accuracy 0.9726 at step 80
epoch1_eval: eval_loss 0.1109 / eval_mean_token_accuracy 0.8822
checkpoint_saved: experiments/tau2_stage2/train_outputs/08_qwen3_6_35b_a3b_273/checkpoint-89
```

当前已有的真实初始结果：

```text
cycle0_tasks: 74
initial_small_accuracy: 42/74 = 56.76%
initial_fallback_needed: 32/74 = 43.24%
initial_task_pass_after_fallback: 59/74 = 79.73%
initial_observed_api_cost: $3.49623640
cycle0_skillbook: 15 skills / 14 procedures
cycle0_sft_pairs: 17 hard tasks -> 17 pairs
```

当前还没有可填写的每轮训练后结果：

```text
checkpoint-best: missing
checkpoint-final: missing
cycle_0/llm_adapter/checkpoint-best: symlink exists, waits for global checkpoint-best
cycle_0/router/router.joblib: missing
cycle_0/router/router_meta.json: missing
cycle_0/e2e_ablation_summary.json: missing
cycle_0/e2e_ablation_summary.md: missing
cycle_1/traces.jsonl: missing
final_ablation_table.md: missing
curve.png: missing
```

结论：

- `mean_token_accuracy` 和 `eval_mean_token_accuracy` 只是 SFT token 级过程指标，不是 tau2 task accuracy。
- 第一轮正式的 `Skills / LLM / Router / Full` accuracy 和 cost，要等 `checkpoint-best` 产出后自动进入 Phase 4 Router、Phase 5 E2E ablation，再补 35B adapter 的 tau2 任务级 eval。
- 后续最终报告必须按表格填：最开始 accuracy/cost、每轮 Skills accuracy/cost、LLM task accuracy/cost、Router accuracy/cost、Full Skills+LLM+Router accuracy/cost。

2026-06-07 09:21 CST：

- 复查 4 分钟 step 窗口后确认：训练没有卡死，已经从 epoch1 eval + checkpoint 保存阶段继续推进到 step 90。
- `checkpoint-89` 保存完整，目录大小约 `259G`：

```text
checkpoint-89/model-00001-of-00002.safetensors: 47G
checkpoint-89/model-00002-of-00002.safetensors: 19G
checkpoint-89/pytorch_model_fsdp.bin: 65G
checkpoint-89/optimizer.bin: 130G
checkpoint-89/trainer_state.json: written at 09:17:19
```

最新训练状态：

```text
status: running
progress: 90/445 steps = 20.22%
elapsed_train_time: 5:33:41
train_log_mtime: 2026-06-07 09:20:59
loss: 0.0647
grad_norm: 0.5859
learning_rate: 9.468e-06
mean_token_accuracy: 0.9579
epoch: 1.011
GPU: 8 张 H200 继续训练，采样时均为 100% 利用
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 SFT 正常继续；checkpoint-89 是周期性 checkpoint，不是最终 `checkpoint-best`。
- Phase 4 Router / Phase 5 E2E 仍未开始，所以还不能填第一轮训练后的 Skills / Router / Full accuracy。
- 训练后的 35B adapter task accuracy 仍要等 `checkpoint-best` 后用 `training.eval.harness` 跑 tau2 task-level eval，输出 `eval_results.json/pass_rate` 后再填。

2026-06-07 09:27 CST：

- 再等一个 step 窗口后复查，确认训练继续推进：

```text
status: running
progress: 91/445 steps = 20.45%
elapsed_train_time: 5:37:20
train_log_mtime: 2026-06-07 09:24:39
latest_logged_metric: loss 0.0647 / mean_token_accuracy 0.9579 at step 90
GPU: 8 张 H200 继续占用，采样时均为 100% 利用
```

- 当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
```

- 判断：
  - step 90 的耗时包含 epoch1 eval + checkpoint-89 保存，所以日志里的剩余 ETA 偏保守，不能直接当稳定 step time。
  - 当前仍处于 cycle 0 Phase 3 SFT；Router / E2E / Full accuracy 仍需等 `checkpoint-best` 后继续。

2026-06-07 09:29 CST：

- 再次复查，训练继续推进到 step 92：

```text
status: running
progress: 92/445 steps = 20.67%
elapsed_train_time: 5:41:07
train_log_mtime: 2026-06-07 09:28:25
latest_logged_metric: loss 0.0647 / mean_token_accuracy 0.9579 at step 90
GPU: 8 张 H200 继续训练，采样时 99%-100% 利用
```

- 当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：
  - cycle 0 Phase 3 SFT 正常推进。
  - 还不能手动启动 Router / E2E；必须等正式 `checkpoint-best` 产物，否则会污染第一轮 Full 结果。

2026-06-07 09:34 CST：

- 再等一个 step 窗口后复查，训练继续推进到 step 93：

```text
status: running
progress: 93/445 steps = 20.90%
elapsed_train_time: 5:44:42
train_log_mtime: 2026-06-07 09:32:01
latest_logged_metric: loss 0.0647 / mean_token_accuracy 0.9579 at step 90
GPU: 8 张 H200 继续训练，采样时 87%-100% 利用
```

- 当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

- 判断：
  - cycle 0 Phase 3 SFT 继续正常推进。
  - Router / E2E / 35B task-level eval 均仍等待 `checkpoint-best`。

2026-06-07 09:38 CST：

- 按新加论文要求，再次确认项目和论文已经读完，并且记录口径固定为：
  - 先报最开始的真实 accuracy / cost。
  - 再报每轮 `Skills + LLM + Router` 联合演进训练完后的 accuracy / fallback / cost。
  - LLM 训练日志里的 `mean_token_accuracy` 只作为过程指标，不当作 tau2 task accuracy。

最新远端状态：

```text
status: running
progress: 94/445 steps = 21.12%
elapsed_train_time: 5:48:18
train_log_mtime: 2026-06-07 09:35:36
latest_logged_metric: loss 0.0647 / mean_token_accuracy 0.9579 at step 90
GPU: 8 张 H200 继续训练，采样时 7 张 99%-100%，1 张短时 0% 但显存仍占用约 124GB
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 目前已有可报告真实初始结果：`42/74 = 56.76%` small accuracy，fallback 后 `59/74 = 79.73%`，实际 API cost `$3.49623640`。

2026-06-07 09:42 CST：

- 再次复查，训练继续推进到 step 96：

```text
status: running
progress: 96/445 steps = 21.57%
elapsed_train_time: 5:55:36
train_log_mtime: 2026-06-07 09:42:54
latest_logged_metric: loss 0.0647 / mean_token_accuracy 0.9579 at step 90
GPU: 8 张 H200 继续训练，采样时 5 张 100%，2 张短时 0%，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

判断：

- cycle 0 Phase 3 35B SFT 继续正常推进。
- 还未进入 Phase 4 Router / Phase 5 E2E。

2026-06-07 09:48 CST：

- 再次复查，训练继续推进到 step 97：

```text
status: running
progress: 97/445 steps = 21.80%
elapsed_train_time: 5:59:16
train_log_mtime: 2026-06-07 09:46:35
latest_logged_metric: loss 0.0647 / mean_token_accuracy 0.9579 at step 90
GPU: 8 张 H200 继续训练，采样时 7 张 100%，1 张短时 0%，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

判断：

- cycle 0 Phase 3 35B SFT 继续正常推进。
- 仍未进入 Phase 4 Router / Phase 5 E2E。

2026-06-07 09:54 CST：

- 再次复查，训练继续推进到 step 99：

```text
status: running
progress: 99/445 steps = 22.25%
elapsed_train_time: 6:06:37
train_log_mtime: 2026-06-07 09:53:55
latest_logged_metric: loss 0.0647 / mean_token_accuracy 0.9579 at step 90
GPU: 8 张 H200 继续训练，采样时 80%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `checkpoint-best` 未生成前，不能启动 Router / E2E，否则会变成未使用 35B adapter 的错误结果。

2026-06-07 09:59 CST：

- 训练推进到 step 100，并打印新的训练过程 metric：

```text
status: running
progress: 100/445 steps = 22.47%
elapsed_train_time: 6:10:20
train_log_mtime: 2026-06-07 09:57:38
loss: 0.03776
grad_norm: 0.9141
learning_rate: 9.299e-06
mean_token_accuracy: 0.9846
epoch: 1.124
GPU: 8 张 H200 继续训练，采样时 99%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
```

判断：

- 35B SFT 正常继续。
- `mean_token_accuracy=0.9846` 只能说明 SFT token 级训练状态，不是 tau2 task accuracy。
- 第一轮训练后的 Skills / LLM / Router / Full accuracy 仍必须等 `checkpoint-best` 后继续跑 Router、E2E 和 task-level eval。

2026-06-07 10:05 CST：

- 再次复查，训练继续推进到 step 102：

```text
status: running
progress: 102/445 steps = 22.92%
elapsed_train_time: 6:17:34
train_log_mtime: 2026-06-07 10:04:53
latest_logged_metric: loss 0.03776 / mean_token_accuracy 0.9846 at step 100
GPU: 8 张 H200 继续训练，采样时 99%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待正式 `checkpoint-best`。

2026-06-07 10:11 CST：

- 再次复查，训练继续推进到 step 103：

```text
status: running
progress: 103/445 steps = 23.15%
elapsed_train_time: 6:21:18
train_log_mtime: 2026-06-07 10:08:36
latest_logged_metric: loss 0.03776 / mean_token_accuracy 0.9846 at step 100
GPU: 8 张 H200 继续训练，采样时 99%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- 这段没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 10:12 CST：

- 再次复查，训练继续推进到 step 104：

```text
status: running
progress: 104/445 steps = 23.37%
elapsed_train_time: 6:25:01
train_log_mtime: 2026-06-07 10:12:20
latest_logged_metric: loss 0.03776 / mean_token_accuracy 0.9846 at step 100
GPU: 8 张 H200 继续训练，采样时 72%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `checkpoint-best` 没出现前，不能报第一轮训练后的 Skills / LLM / Router / Full accuracy。
- 这段没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 10:18 CST：

- 再次复查，训练继续推进到 step 105：

```text
status: running
progress: 105/445 steps = 23.60%
elapsed_train_time: 6:28:45
train_log_mtime: 2026-06-07 10:16:03
latest_logged_metric: loss 0.03776 / mean_token_accuracy 0.9846 at step 100
GPU: 8 张 H200 继续训练，采样时 99%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 10:19 CST：

- 再次复查，训练继续推进到 step 106：

```text
status: running
progress: 106/445 steps = 23.82%
elapsed_train_time: 6:32:23
train_log_mtime: 2026-06-07 10:19:41
latest_logged_metric: loss 0.03776 / mean_token_accuracy 0.9846 at step 100
GPU: 8 张 H200 继续训练，采样时 71%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `checkpoint-best` 没出现前，仍不能报第一轮训练后的 Skills / LLM / Router / Full accuracy。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 10:25 CST：

- 再次复查，训练继续推进到 step 107：

```text
status: running
progress: 107/445 steps = 24.04%
elapsed_train_time: 6:36:06
train_log_mtime: 2026-06-07 10:23:24
latest_logged_metric: loss 0.03776 / mean_token_accuracy 0.9846 at step 100
GPU: 8 张 H200 继续训练，采样时 100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 15:32 CST：

- 定时复查，训练继续推进到 step 190：

```text
status: running
progress: 190/445 steps = 42.70%
elapsed_train_time: 11:42:48
train_log_mtime: 2026-06-07 15:30:06
latest_logged_metric: loss 0.02503 / mean_token_accuracy 0.99 at step 190
GPU: 8 张 H200 继续训练，采样时 0%-99% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- `mean_token_accuracy=0.99` 仍是训练 token 指标，不是 tau2 task accuracy。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 16:09 CST：

- 定时复查，训练继续推进到 step 200：

```text
status: running
progress: 200/445 steps = 44.94%
elapsed_train_time: 12:20:06
train_log_mtime: 2026-06-07 16:07:24
latest_logged_metric: loss 0.02559 / mean_token_accuracy 0.9897 at step 200
GPU: 8 张 H200 继续训练，采样时 17%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- `mean_token_accuracy=0.9897` 仍是训练 token 指标，不是 tau2 task accuracy。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 16:44 CST：

- 定时复查，训练继续推进到 step 210：

```text
status: running
progress: 210/445 steps = 47.19%
elapsed_train_time: 12:57:16
train_log_mtime: 2026-06-07 16:44:35
latest_logged_metric: loss 0.02911 / mean_token_accuracy 0.9891 at step 210
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- `mean_token_accuracy=0.9891` 仍是训练 token 指标，不是 tau2 task accuracy。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 17:17 CST：

- 定时复查，训练继续推进到 step 218：

```text
status: running
progress: 218/445 steps = 48.99%
elapsed_train_time: 13:26:55
train_log_mtime: 2026-06-07 17:14:13
latest_logged_metric: loss 0.02911 / mean_token_accuracy 0.9891 at step 210
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- `mean_token_accuracy=0.9891` 仍是训练 token 指标，不是 tau2 task accuracy。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 17:28 CST：

- 定时复查，训练继续推进到 step 221：

```text
status: running
progress: 221/445 steps = 49.66%
elapsed_train_time: 13:37:59
train_log_mtime: 2026-06-07 17:25:18
latest_logged_metric: loss 0.02832 / mean_token_accuracy 0.9888 at step 221
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- `mean_token_accuracy=0.9888` 仍是训练 token 指标，不是 tau2 task accuracy。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 18:02 CST：

- 定时复查，训练继续推进到 step 231：

```text
status: running
progress: 231/445 steps = 51.91%
elapsed_train_time: 14:15:03
train_log_mtime: 2026-06-07 18:02:21
latest_logged_metric: loss 0.02897 / mean_token_accuracy 0.9886 at step 231
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- `mean_token_accuracy=0.9886` 仍是训练 token 指标，不是 tau2 task accuracy。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 18:15 CST：

- 定时复查，训练继续推进到 step 234：

```text
status: running
progress: 234/445 steps = 52.58%
elapsed_train_time: 14:26:18
train_log_mtime: 2026-06-07 18:13:36
latest_logged_metric: loss 0.02897 / mean_token_accuracy 0.9886 at step 231
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- `mean_token_accuracy=0.9886` 仍是训练 token 指标，不是 tau2 task accuracy。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 18:17 CST：

- 定时复查，训练继续推进到 step 235：

```text
status: running
progress: 235/445 steps = 52.81%
elapsed_train_time: 14:29:58
train_log_mtime: 2026-06-07 18:17:16
latest_logged_metric: loss 0.02897 / mean_token_accuracy 0.9886 at step 231
GPU: 8 张 H200 继续训练，采样时 100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- `mean_token_accuracy=0.9886` 仍是训练 token 指标，不是 tau2 task accuracy。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 18:23 CST：

- 定时复查，训练继续推进到 step 236：

```text
status: running
progress: 236/445 steps = 53.03%
elapsed_train_time: 14:33:44
train_log_mtime: 2026-06-07 18:21:03
latest_logged_metric: loss 0.02897 / mean_token_accuracy 0.9886 at step 231
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 18:28 CST：

- 定时复查，训练继续推进到 step 238：

```text
status: running
progress: 238/445 steps = 53.48%
elapsed_train_time: 14:41:02
train_log_mtime: 2026-06-07 18:28:20
latest_logged_metric: loss 0.02897 / mean_token_accuracy 0.9886 at step 231
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 18:33 CST：

- 定时复查，训练继续推进到 step 239：

```text
status: running
progress: 239/445 steps = 53.71%
elapsed_train_time: 14:44:47
train_log_mtime: 2026-06-07 18:32:05
latest_logged_metric: loss 0.02897 / mean_token_accuracy 0.9886 at step 231
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 18:38 CST：

- 定时复查，训练继续推进到 step 240，并打印新的训练过程 metric：

```text
status: running
progress: 240/445 steps = 53.93%
elapsed_train_time: 14:48:30
train_log_mtime: 2026-06-07 18:35:48
loss: 0.03198
grad_norm: 0.9922
learning_rate: 5.333e-06
mean_token_accuracy: 0.9875
epoch: 2.698
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9875` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 18:43 CST：

- 定时复查，训练继续推进到 step 242：

```text
status: running
progress: 242/445 steps = 54.38%
elapsed_train_time: 14:55:58
train_log_mtime: 2026-06-07 18:43:16
latest_logged_metric: loss 0.03198 / mean_token_accuracy 0.9875 at step 240
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 18:48 CST：

- 定时复查，训练继续推进到 step 243：

```text
status: running
progress: 243/445 steps = 54.61%
elapsed_train_time: 14:59:36
train_log_mtime: 2026-06-07 18:46:54
latest_logged_metric: loss 0.03198 / mean_token_accuracy 0.9875 at step 240
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 18:53 CST：

- 定时复查，训练继续推进到 step 244：

```text
status: running
progress: 244/445 steps = 54.83%
elapsed_train_time: 15:03:22
train_log_mtime: 2026-06-07 18:50:40
latest_logged_metric: loss 0.03198 / mean_token_accuracy 0.9875 at step 240
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 18:58 CST：

- 定时复查，训练继续推进到 step 246：

```text
status: running
progress: 246/445 steps = 55.28%
elapsed_train_time: 15:10:51
train_log_mtime: 2026-06-07 18:58:10
latest_logged_metric: loss 0.03198 / mean_token_accuracy 0.9875 at step 240
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 19:03 CST：

- 定时复查，训练继续推进到 step 247：

```text
status: running
progress: 247/445 steps = 55.51%
elapsed_train_time: 15:14:36
train_log_mtime: 2026-06-07 19:01:54
latest_logged_metric: loss 0.03198 / mean_token_accuracy 0.9875 at step 240
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 19:08 CST：

- 定时复查，训练继续推进到 step 248：

```text
status: running
progress: 248/445 steps = 55.73%
elapsed_train_time: 15:18:21
train_log_mtime: 2026-06-07 19:05:39
latest_logged_metric: loss 0.03198 / mean_token_accuracy 0.9875 at step 240
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 19:18 CST：

- 定时复查，训练继续推进到 step 251；step 250 打印了新的训练过程 metric：

```text
status: running
progress: 251/445 steps = 56.40%
elapsed_train_time: 15:29:25
train_log_mtime: 2026-06-07 19:16:43
loss: 0.03214
grad_norm: 0.8281
learning_rate: 4.999e-06
mean_token_accuracy: 0.9876
epoch: 2.81
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9876` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 19:23 CST：

- 定时复查，训练继续推进到 step 252：

```text
status: running
progress: 252/445 steps = 56.63%
elapsed_train_time: 15:33:13
train_log_mtime: 2026-06-07 19:20:31
latest_logged_metric: loss 0.03214 / mean_token_accuracy 0.9876 at step 250
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 19:28 CST：

- 定时复查，训练继续推进到 step 254：

```text
status: running
progress: 254/445 steps = 57.08%
elapsed_train_time: 15:40:33
train_log_mtime: 2026-06-07 19:27:51
latest_logged_metric: loss 0.03214 / mean_token_accuracy 0.9876 at step 250
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 19:33 CST：

- 定时复查，训练继续推进到 step 255：

```text
status: running
progress: 255/445 steps = 57.30%
elapsed_train_time: 15:44:16
train_log_mtime: 2026-06-07 19:31:34
latest_logged_metric: loss 0.03214 / mean_token_accuracy 0.9876 at step 250
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 19:38 CST：

- 定时复查，训练继续推进到 step 256：

```text
status: running
progress: 256/445 steps = 57.53%
elapsed_train_time: 15:47:58
train_log_mtime: 2026-06-07 19:35:17
latest_logged_metric: loss 0.03214 / mean_token_accuracy 0.9876 at step 250
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 19:43 CST：

- 定时复查，训练继续推进到 step 258：

```text
status: running
progress: 258/445 steps = 57.98%
elapsed_train_time: 15:55:33
train_log_mtime: 2026-06-07 19:42:51
latest_logged_metric: loss 0.03214 / mean_token_accuracy 0.9876 at step 250
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 19:48 CST：

- 定时复查，训练继续推进到 step 259：

```text
status: running
progress: 259/445 steps = 58.20%
elapsed_train_time: 15:59:20
train_log_mtime: 2026-06-07 19:46:38
latest_logged_metric: loss 0.03214 / mean_token_accuracy 0.9876 at step 250
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 19:53 CST：

- 定时复查，训练继续推进到 step 260，并打印新的训练过程 metric：

```text
status: running
progress: 260/445 steps = 58.43%
elapsed_train_time: 16:03:11
train_log_mtime: 2026-06-07 19:50:29
loss: 0.02413
grad_norm: 0.7773
learning_rate: 4.667e-06
mean_token_accuracy: 0.9885
epoch: 2.923
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9885` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 19:58 CST：

- 定时复查，训练继续推进到 step 262：

```text
status: running
progress: 262/445 steps = 58.88%
elapsed_train_time: 16:10:41
train_log_mtime: 2026-06-07 19:57:59
latest_logged_metric: loss 0.02413 / mean_token_accuracy 0.9885 at step 260
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 20:03 CST：

- 定时复查，训练继续推进到 step 263：

```text
status: running
progress: 263/445 steps = 59.10%
elapsed_train_time: 16:14:22
train_log_mtime: 2026-06-07 20:01:40
latest_logged_metric: loss 0.02413 / mean_token_accuracy 0.9885 at step 260
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 20:08 CST：

- 定时复查，训练继续推进到 step 264：

```text
status: running
progress: 264/445 steps = 59.33%
elapsed_train_time: 16:18:07
train_log_mtime: 2026-06-07 20:05:26
latest_logged_metric: loss 0.02413 / mean_token_accuracy 0.9885 at step 260
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 20:13 CST：

- 定时复查，训练继续推进到 step 266：

```text
status: running
progress: 266/445 steps = 59.78%
elapsed_train_time: 16:25:38
train_log_mtime: 2026-06-07 20:12:57
latest_logged_metric: loss 0.02413 / mean_token_accuracy 0.9885 at step 260
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- 即将到 epoch 3 eval 附近；稳定 `checkpoint-best` 仍需等待训练发布。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 20:24 CST：

- 定时复查，训练继续推进到 step 268，并确认 epoch 3 eval 已完成：

```text
status: running
progress: 268/445 steps = 60.22%
elapsed_train_time: 16:34:04
train_log_mtime: 2026-06-07 20:21:22
eval_epoch_1: eval_loss 0.1109 / eval_mean_token_accuracy 0.8822
eval_epoch_2: eval_loss 0.1188 / eval_mean_token_accuracy 0.8813
eval_epoch_3: eval_loss 0.1244 / eval_mean_token_accuracy 0.8837
checkpoint dirs: checkpoint-89 only
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- epoch 3 eval 已完成，但 best numbered checkpoint 仍是 epoch 1 的 `checkpoint-89`。
- 稳定 `checkpoint-best` 仍等待 `publish_best_checkpoint()` 在训练完成后发布。
- `eval_mean_token_accuracy` 仍是 token 级验证指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 20:33 CST：

- 定时复查，训练继续推进到 step 271；step 270 打印了新的训练过程 metric：

```text
status: running
progress: 271/445 steps = 60.90%
elapsed_train_time: 16:45:12
train_log_mtime: 2026-06-07 20:32:31
loss: 0.02796
grad_norm: 0.6953
learning_rate: 4.341e-06
mean_token_accuracy: 0.9871
epoch: 3.034
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9871` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 21:08 CST：

- 定时复查，训练继续推进到 step 280，并打印新的训练过程 metric：

```text
status: running
progress: 280/445 steps = 62.92%
elapsed_train_time: 17:18:49
train_log_mtime: 2026-06-07 21:06:07
loss: 0.02068
grad_norm: 0.918
learning_rate: 4.021e-06
mean_token_accuracy: 0.9919
epoch: 3.146
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9919` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 21:48 CST：

- 定时复查，训练继续推进到 step 291；step 290 打印了新的训练过程 metric：

```text
status: running
progress: 291/445 steps = 65.39%
elapsed_train_time: 18:00:17
train_log_mtime: 2026-06-07 21:47:35
loss: 0.01789
grad_norm: 0.7461
learning_rate: 3.709e-06
mean_token_accuracy: 0.9924
epoch: 3.259
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9924` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 22:23 CST：

- 定时复查，训练继续推进到 step 300，并打印新的训练过程 metric：

```text
status: running
progress: 300/445 steps = 67.42%
elapsed_train_time: 18:34:22
train_log_mtime: 2026-06-07 22:21:40
loss: 0.02385
grad_norm: 0.918
learning_rate: 3.406e-06
mean_token_accuracy: 0.9901
epoch: 3.371
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9901` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 22:58 CST：

- 定时复查，训练继续推进到 step 310，并打印新的训练过程 metric：

```text
status: running
progress: 310/445 steps = 69.66%
elapsed_train_time: 19:11:32
train_log_mtime: 2026-06-07 22:58:50
loss: 0.02517
grad_norm: 0.4473
learning_rate: 3.116e-06
mean_token_accuracy: 0.9901
epoch: 3.484
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9901` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 23:38 CST：

- 定时复查，训练继续推进到 step 320，并打印新的训练过程 metric：

```text
status: running
progress: 320/445 steps = 71.91%
elapsed_train_time: 19:49:13
train_log_mtime: 2026-06-07 23:36:31
loss: 0.0209
grad_norm: 0.9102
learning_rate: 2.839e-06
mean_token_accuracy: 0.9908
epoch: 3.596
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9908` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-08 00:19 CST：

- 定时复查，训练继续推进到 step 331；step 330 打印了新的训练过程 metric：

```text
status: running
progress: 331/445 steps = 74.38%
elapsed_train_time: 20:30:28
train_log_mtime: 2026-06-08 00:17:47
loss: 0.01791
grad_norm: 0.6719
learning_rate: 2.576e-06
mean_token_accuracy: 0.9922
epoch: 3.709
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9922` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-08 00:54 CST：

- 定时复查，训练继续推进到 step 340，并打印新的训练过程 metric：

```text
status: running
progress: 340/445 steps = 76.40%
elapsed_train_time: 21:04:20
train_log_mtime: 2026-06-08 00:51:38
loss: 0.01968
grad_norm: 0.5469
learning_rate: 2.33e-06
mean_token_accuracy: 0.9921
epoch: 3.821
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9921` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-08 01:14 CST：

- 定时复查，训练继续推进到 step 346：

```text
status: running
progress: 346/445 steps = 77.75%
elapsed_train_time: 21:26:42
train_log_mtime: 2026-06-08 01:14:00
latest_logged_metric: loss 0.01968 / mean_token_accuracy 0.9921 at step 340
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-08 01:29 CST：

- 定时复查，训练继续推进到 step 350，并打印新的训练过程 metric：

```text
status: running
progress: 350/445 steps = 78.65%
elapsed_train_time: 21:41:21
train_log_mtime: 2026-06-08 01:28:40
loss: 0.02023
grad_norm: 0.7773
learning_rate: 2.101e-06
mean_token_accuracy: 0.9931
epoch: 3.934
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9931` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-08 02:09 CST：

- 定时复查，训练继续推进到 step 360，并打印新的训练过程 metric：

```text
status: running
progress: 360/445 steps = 80.90%
elapsed_train_time: 22:19:50
train_log_mtime: 2026-06-08 02:07:08
loss: 0.01787
grad_norm: 0.8398
learning_rate: 1.891e-06
mean_token_accuracy: 0.9911
epoch: 4.045
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9911` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-08 02:49 CST：

- 定时复查，训练继续推进到 step 371，并打印新的训练过程 metric：

```text
status: running
progress: 371/445 steps = 83.37%
elapsed_train_time: 23:01:06
train_log_mtime: 2026-06-08 02:48:24
loss: 0.01952
grad_norm: 0.6992
learning_rate: 1.701e-06
mean_token_accuracy: 0.9923
epoch: 4.158
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9923` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-08 03:24 CST：

- 定时复查，训练继续推进到 step 380，并打印新的训练过程 metric：

```text
status: running
progress: 380/445 steps = 85.39%
elapsed_train_time: 23:34:50
train_log_mtime: 2026-06-08 03:22:09
loss: 0.01493
grad_norm: 0.7578
learning_rate: 1.532e-06
mean_token_accuracy: 0.9942
epoch: 4.27
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9942` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-08 04:14 CST：

- 定时复查，训练继续推进到 step 393，并在 step 391 附近打印新的训练过程 metric：

```text
status: running
progress: 393/445 steps = 88.31%
elapsed_train_time: 24:23:58
train_log_mtime: 2026-06-08 04:11:16
latest_logged_metric: loss 0.01906 / mean_token_accuracy 0.9924 at epoch 4.383
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9924` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-08 04:39 CST：

- 定时复查，训练继续推进到 step 400，并打印新的训练过程 metric：

```text
status: running
progress: 400/445 steps = 89.89%
elapsed_train_time: 24:50:16
train_log_mtime: 2026-06-08 04:37:35
loss: 0.0151
grad_norm: 0.7305
learning_rate: 1.261e-06
mean_token_accuracy: 0.9945
epoch: 4.495
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9945` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-08 05:29 CST：

- 定时复查，训练继续推进到 step 413，并在 step 411 附近打印新的训练过程 metric：

```text
status: running
progress: 413/445 steps = 92.81%
elapsed_train_time: 25:39:16
train_log_mtime: 2026-06-08 05:26:34
latest_logged_metric: loss 0.01724 / mean_token_accuracy 0.9942 at epoch 4.608
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9942` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-08 05:54 CST：

- 定时复查，训练继续推进到 step 420，并打印新的训练过程 metric：

```text
status: running
progress: 420/445 steps = 94.38%
elapsed_train_time: 26:05:45
train_log_mtime: 2026-06-08 05:53:03
loss: 0.01666
grad_norm: 1.102
learning_rate: 1.084e-06
mean_token_accuracy: 0.9936
epoch: 4.72
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9936` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-08 06:34 CST：

- 定时复查，训练继续推进到 step 431，并打印新的训练过程 metric：

```text
status: running
progress: 431/445 steps = 96.85%
elapsed_train_time: 26:47:12
train_log_mtime: 2026-06-08 06:34:30
loss: 0.01585
grad_norm: 0.793
learning_rate: 1.032e-06
mean_token_accuracy: 0.9936
epoch: 4.833
checkpoint-best: missing
router/e2e: not started
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9936` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-08 07:30 CST：

- cycle 0 Phase 3 / 4 / 5 已跑完，正式 `checkpoint-best` 已发布并自动接上 router 与 E2E ablation。

```text
35B SFT:
  status: done
  progress: 445/445 steps = 100%
  train_runtime: 99650s = 27:40:51
  train_loss: 0.03668
  final_eval_loss: 0.1352
  final_eval_mean_token_accuracy: 0.883
  checkpoint-best: experiments/tau2_stage2/train_outputs/08_qwen3_6_35b_a3b_273/checkpoint-best -> checkpoint-89

Router:
  n_examples_total: 74
  label_distribution: small_ok=42 / need_large=32
  heldout_accuracy: 0.7368421052631579
  heldout_f1_large: 0.6666666666666666

E2E ablation on cycle0 traces:
  base:   routing_acc=56.76%  task_pass=56.76%  fallback=43.24%  cost_vs_large=10.00%
  skills: routing_acc=56.76%  task_pass=56.76%  fallback=43.24%  cost_vs_large=10.00%
  router: routing_acc=93.24%  task_pass=78.38%  fallback=4.05%   cost_vs_large=47.70%
  full:   routing_acc=93.24%  task_pass=78.38%  fallback=4.05%   cost_vs_large=47.70%
```

判断：

- cycle 0 的 skills + 35B LLM SFT + router + E2E ablation 已完成。
- 这里的 `final_eval_mean_token_accuracy=0.883` 仍是 SFT token 级 eval 指标，不是 tau2 task accuracy。
- 当前 `run_e2e_ablation_simple.py` 的 full 与 router 数字相同；35B adapter 的真实 tau2 task-level pass rate 仍需之后用 tau2 eval harness 单独跑。
- pipeline 已从 cycle 1 继续：正在用 cycle 0 的 `checkpoint-best` 启动本地 vLLM，准备收集下一轮 traces。
- 本段无新增外部 API 花费；已确认的真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-08 07:48 CST：

- cycle 1 首次启动后发现 `TAU2_USER_MODEL=openai/openai/gpt-5.2` 在 CommonStack/OpenAI 代理上不可用，日志报：

```text
litellm.NotFoundError: OpenAIException - The model `openai/gpt-5.2` does not exist.
```

处理：

```text
bad_partial_cycle1:
  observed_trace_rows: 5
  empty_completion_rows: 3
  archive: results/cwy_35b_joint_20260606_165203/debug_bad_cycle1_20260608_074239
  action: removed results/cwy_35b_joint_20260606_165203/cycle_1 and restarted from cycle 1

fixed_restart:
  tmux: cwy35b_cycle1_fixed_20260608_074255
  TAU2_USER_MODEL: openai/openai/gpt-5.4-2026-03-05
  LARGE_MODEL: openai/openai/gpt-5.4-2026-03-05
  vLLM: ready on port 8050
  clean_trace_rows_after_restart: 1
  empty_completion_rows_after_restart: 0
```

判断：

- 这是配置/API 模型名问题，不是 35B checkpoint 本身的问题。
- 坏的 cycle 1 局部 traces 没有继续用于训练，已归档后删除。
- 固定后 cycle 1 已重新开始收集干净 traces。
- `evol-llm-student` 的 cost mapping warning 仍会出现，但只影响本地学生模型成本估算，不影响 vLLM 生成和 trace 内容。

2026-06-08 08:24 CST：

- 继续排查 cycle 1 时又发现两层问题，并已修复后重启：

```text
problem_1:
  symptom: gpt-5.2 不再出现后，仍出现 openai/gpt-5.4-2026-03-05 model does not exist
  root_cause: NL judge 被 stage2 adapter 错误路由到本地 vLLM 8050，而不是 CommonStack
  fix: experiments/tau2_stage2/code/adapters/tau2_bench/adapter.py
       _route_nl_judge_through(dict(config.agent.args))
       -> _route_nl_judge_through(dict(config.user.args))

problem_2:
  symptom: 第一次 judge patch 后 cycle1 74/74 traces 全空，21.9s 结束
  root_cause: adapter.py 新增 os.environ 但漏了 import os，导致 NameError 被 collect_traces 吃掉为空结果
  fix: adapter.py 添加 import os

archives:
  debug_bad_cycle1_judge_20260608_075529
  debug_bad_cycle1_nameerror_20260608_080146
  debug_bad_cycle1_judgelocal_20260608_081850

clean_restart:
  tmux: cwy35b_cycle1_routefix_20260608_081908
  TAU2_USER_MODEL: openai/openai/gpt-5.4-2026-03-05
  TAU2_NL_JUDGE_MODEL: openai/openai/gpt-5.4-2026-03-05
  vLLM: ready on port 8050
  trace_rows_checked: 2
  empty_completion_rows: 0
  model_not_found_errors: 0
```

判断：

- cycle 1 坏 trace 均已归档并移出正式 `cycle_1`，不会进入后续 SkillBook/SFT/router。
- 现在 routefix 版本的 cycle 1 正在正常收集，当前已确认前 2 条 trace 非空。
- `evol-llm-student` cost mapping warning 仍为成本估算告警，不是生成失败；后续 cost 分析需注意本地学生模型成本可能低估。

2026-06-08 08:44 CST：

- cycle 1 routefix 版本继续正常收集：

```text
phase: cycle1 trace collection
progress: 10/74 traces
decisions:
  oracle:small_OK+large_run: 9
  probe:small_fail->large_OK: 1
empty_completion_rows: 0
model_not_found_errors: 0
vLLM: ready on port 8050
```

判断：

- cycle 1 已产生可用于下一轮 SFT 的 hard trace：`small_fail->large_OK`。
- 当前没有再出现 `gpt-5.2`、`model does not exist`、`NameError`。
- `evol-llm-student` cost mapping warning 继续存在，仍只影响本地学生模型 cost 估算。

2026-06-08 09:14 CST：

- cycle 1 routefix 版本继续正常收集：

```text
phase: cycle1 trace collection
progress: 21/74 traces
decisions:
  oracle:small_OK+large_run: 20
  probe:small_fail->large_OK: 1
empty_completion_rows: 0
model_not_found_errors: 0
vLLM: ready on port 8050
```

判断：

- 这轮正式 trace 仍在用 cycle 0 的 35B `checkpoint-best` 本地 vLLM 跑，不是 mock。
- 目前没有再出现空 completion、`gpt-5.2` 不存在、NL judge 路由到本地 vLLM 之类问题。
- 继续等待 74/74 trace 完成，然后进入 cycle 1 SkillBook 更新、35B SFT、router、E2E ablation。

2026-06-08 09:44 CST：

- cycle 1 trace collection 继续推进：

```text
phase: cycle1 trace collection
progress: 31/74 traces = 41.89%
decisions:
  oracle:small_OK+large_run: 28
  probe:small_fail->large_OK: 2
  probe:small_fail->large_fail: 1
empty_completion_rows: 0
vLLM: ready on port 8050
```

判断：

- 当前可用于下一轮 SFT 的 hard trace 是 2 条 `small_fail->large_OK`。
- 新增 1 条 `small_fail->large_fail`，表示 small 和 large 都失败；这不是环境错误，但通常不会作为正向 SFT 样本。
- 继续等待 trace 收满 74 条；后续会自动进入 SkillBook、35B SFT、router、E2E。

2026-06-08 10:06 CST：

- cycle 1 trace collection 被我主动停止并清理：

```text
原因: CommonStack API key 触发 429
429 message: Access key max cost limit exceeded (cap 100)
停止前文件: 71 rows
第一条坏 row: line 40, task_id 59, small_completion empty, large_completion empty
坏 rows: line 40-71, empty_completion_rows 32
保留干净前缀: line 1-39
正式 traces: results/cwy_35b_joint_20260606_165203/cycle_1/traces.jsonl = 39 rows
归档: results/cwy_35b_joint_20260606_165203/debug_archive/api_cap_bad_traces_20260608_1004/
```

干净 39 行分布：

```text
oracle:small_OK+large_run: 35
probe:small_fail->large_OK: 2
probe:small_fail->large_fail: 2
empty_completion_rows: 0
```

判断：

- 这是 API 额度硬上限，不是 35B 本地模型坏，也不是 tau2 adapter 再出错。
- 额度触顶后继续跑会继续写空 completion，污染后续 SkillBook/SFT/router，所以已经停掉 routefix tmux 和本地 vLLM。
- 已补 `collect_traces.py --resume` 和 `SCALING_TRACE_RESUME=1`，额度恢复后可从已有 39 条继续 append，避免从 0 重跑。
- 暂停原因只剩 API key 额度；需要提升当前 CommonStack key cap 或换一个有余额/额度的 key，再继续正式流程。

2026-06-08 10:18 CST：

- 做了恢复尝试和防污染保护：

```text
preflight:
  <api-base-from-env>        -> connection refused
  <api-base-from-env>  -> minimal chat 200 OK
resume attempt:
  SCALING_TRACE_RESUME=1
  --resume 1
  existing clean task_ids loaded: 39
  vLLM local student: ready on port 8050
failure:
  task_id: 59
  error: 429 Access key max cost limit exceeded (cap 100)
  guard: FatalTraceCollectionError on empty completions
result:
  no poisoned rows appended
  cycle_1/traces.jsonl remains 39 rows
  empty_completion_rows remains 0
```

同时改动：

- `experiments/scaling/collect_traces.py`：
  - 支持 `--resume`，可跳过已存在 task_id 后继续 append；
  - 遇到 429 / rate limit / cap exceeded 直接中止；
  - small 和 large completion 都为空时直接中止，不写入 trace。
- `scaling/run_full_pipeline.sh`：
  - 支持 `SCALING_TRACE_RESUME=1` 传入 `--resume`。

判断：

- 当前 35B 本地 adapter、cycle0 router、cycle0 SkillBook 都能正常加载。
- 当前无法继续的唯一原因仍是 CommonStack key 的 `cap 100`；即使 endpoint 可连，正式 tau2 user/judge 调用会触发额度上限。
- 需要提高这个 key 的 cap，或换一个有足够额度的 CommonStack key；恢复命令已经具备断点续跑能力。

2026-06-08 10:34 CST：

- 再次检查并重试：

```text
CommonStack probe:
  tiny ping: 200 OK
  tau2_like_probe: 200 OK
formal resume:
  session: cwy35b_cycle1_resume39_guard3_20260608_1027
  existing clean task_ids loaded: 39
  local vLLM: ready on port 8050
  failing task: task_id 59
  failure: 429 Access key max cost limit exceeded (cap 100)
  guard: FatalTraceCollectionError, no poisoned trace written
final trace state:
  rows: 39
  empty_completion_rows: 0
  decisions:
    oracle:small_OK+large_run: 35
    probe:small_fail->large_OK: 2
    probe:small_fail->large_fail: 2
process state:
  no active vLLM / run_full_pipeline / collect_traces process
```

结论：

- 同一个外部额度限制已经连续阻断正式恢复；继续反复启动只会浪费启动 vLLM 的时间。
- 当前目标不能在不提高 CommonStack key cap 或不换 key 的情况下继续完成。
- 代码和数据已经停在可恢复状态：换 key/提 cap 后用 `SCALING_TRACE_RESUME=1` + `--resume 1` 从 39/74 接着跑。

2026-06-07 17:52 CST：

- 快照复查，训练继续推进到 step 228：

```text
status: running
progress: 228/445 steps = 51.24%
elapsed_train_time: 14:03:57
train_log_mtime: 2026-06-07 17:51:15
latest_logged_metric: loss 0.02832 / mean_token_accuracy 0.9888 at step 221
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- `mean_token_accuracy=0.9888` 仍是训练 token 指标，不是 tau2 task accuracy。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 17:40 CST：

- 定时复查，训练继续推进到 step 225：

```text
status: running
progress: 225/445 steps = 50.56%
elapsed_train_time: 13:52:55
train_log_mtime: 2026-06-07 17:40:13
latest_logged_metric: loss 0.02832 / mean_token_accuracy 0.9888 at step 221
GPU: 8 张 H200 继续训练，采样时 1%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- `mean_token_accuracy=0.9888` 仍是训练 token 指标，不是 tau2 task accuracy。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 17:50 CST：

- 定时复查，训练继续推进到 step 227：

```text
status: running
progress: 227/445 steps = 51.01%
elapsed_train_time: 14:00:15
train_log_mtime: 2026-06-07 17:47:33
latest_logged_metric: loss 0.02832 / mean_token_accuracy 0.9888 at step 221
GPU: 8 张 H200 继续训练，采样时 68%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- `mean_token_accuracy=0.9888` 仍是训练 token 指标，不是 tau2 task accuracy。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 17:30 CST：

- 快照复查，训练继续推进到 step 222：

```text
status: running
progress: 222/445 steps = 49.89%
elapsed_train_time: 13:41:44
train_log_mtime: 2026-06-07 17:29:02
latest_logged_metric: loss 0.02832 / mean_token_accuracy 0.9888 at step 221
GPU: 8 张 H200 继续训练，采样时 99%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- `mean_token_accuracy=0.9888` 仍是训练 token 指标，不是 tau2 task accuracy。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 17:18 CST：

- 快照复查，训练继续推进到 step 219：

```text
status: running
progress: 219/445 steps = 49.21%
elapsed_train_time: 13:30:38
train_log_mtime: 2026-06-07 17:17:56
latest_logged_metric: loss 0.02911 / mean_token_accuracy 0.9891 at step 210
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- `mean_token_accuracy=0.9891` 仍是训练 token 指标，不是 tau2 task accuracy。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 17:06 CST：

- 定时复查，训练继续推进到 step 216：

```text
status: running
progress: 216/445 steps = 48.54%
elapsed_train_time: 13:19:32
train_log_mtime: 2026-06-07 17:06:51
latest_logged_metric: loss 0.02911 / mean_token_accuracy 0.9891 at step 210
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- `mean_token_accuracy=0.9891` 仍是训练 token 指标，不是 tau2 task accuracy。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 16:56 CST：

- 快照复查，训练继续推进到 step 213：

```text
status: running
progress: 213/445 steps = 47.87%
elapsed_train_time: 13:08:27
train_log_mtime: 2026-06-07 16:55:45
latest_logged_metric: loss 0.02911 / mean_token_accuracy 0.9891 at step 210
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- `mean_token_accuracy=0.9891` 仍是训练 token 指标，不是 tau2 task accuracy。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 16:54 CST：

- 定时复查，训练继续推进到 step 212：

```text
status: running
progress: 212/445 steps = 47.64%
elapsed_train_time: 13:04:42
train_log_mtime: 2026-06-07 16:52:00
latest_logged_metric: loss 0.02911 / mean_token_accuracy 0.9891 at step 210
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- `mean_token_accuracy=0.9891` 仍是训练 token 指标，不是 tau2 task accuracy。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 16:34 CST：

- 定时复查，训练继续推进到 step 207：

```text
status: running
progress: 207/445 steps = 46.52%
elapsed_train_time: 12:46:03
train_log_mtime: 2026-06-07 16:33:21
latest_logged_metric: loss 0.02559 / mean_token_accuracy 0.9897 at step 200
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- `mean_token_accuracy=0.9897` 仍是训练 token 指标，不是 tau2 task accuracy。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 16:22 CST：

- 定时复查，训练继续推进到 step 204：

```text
status: running
progress: 204/445 steps = 45.84%
elapsed_train_time: 12:34:58
train_log_mtime: 2026-06-07 16:22:16
latest_logged_metric: loss 0.02559 / mean_token_accuracy 0.9897 at step 200
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- `mean_token_accuracy=0.9897` 仍是训练 token 指标，不是 tau2 task accuracy。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 16:11 CST：

- 快照复查，训练继续推进到 step 201：

```text
status: running
progress: 201/445 steps = 45.17%
elapsed_train_time: 12:23:54
train_log_mtime: 2026-06-07 16:11:12
latest_logged_metric: loss 0.02559 / mean_token_accuracy 0.9897 at step 200
GPU: 8 张 H200 继续训练，采样时 58%-86% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- `mean_token_accuracy=0.9897` 仍是训练 token 指标，不是 tau2 task accuracy。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 15:57 CST：

- 定时复查，训练继续推进到 step 197：

```text
status: running
progress: 197/445 steps = 44.27%
elapsed_train_time: 12:08:55
train_log_mtime: 2026-06-07 15:56:13
latest_logged_metric: loss 0.02503 / mean_token_accuracy 0.99 at step 190
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- `mean_token_accuracy=0.99` 仍是训练 token 指标，不是 tau2 task accuracy。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 15:41 CST：

- 定时复查，训练继续推进到 step 193：

```text
status: running
progress: 193/445 steps = 43.37%
elapsed_train_time: 11:53:57
train_log_mtime: 2026-06-07 15:41:16
latest_logged_metric: loss 0.02503 / mean_token_accuracy 0.99 at step 190
GPU: 8 张 H200 继续训练，采样时 65%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- `mean_token_accuracy=0.99` 仍是训练 token 指标，不是 tau2 task accuracy。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 15:27 CST：

- 快照复查，训练继续推进到 step 189：

```text
status: running
progress: 189/445 steps = 42.47%
elapsed_train_time: 11:39:09
train_log_mtime: 2026-06-07 15:26:27
latest_logged_metric: loss 0.04725 / mean_token_accuracy 0.9648 at step 180
GPU: 8 张 H200 继续训练，采样时 70%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 15:26 CST：

- 定时复查，训练继续推进到 step 188：

```text
status: running
progress: 188/445 steps = 42.25%
elapsed_train_time: 11:35:24
train_log_mtime: 2026-06-07 15:22:42
latest_logged_metric: loss 0.04725 / mean_token_accuracy 0.9648 at step 180
GPU: 8 张 H200 继续训练，采样时 99%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 15:20 CST：

- 定时复查，训练继续推进到 step 187：

```text
status: running
progress: 187/445 steps = 42.02%
elapsed_train_time: 11:31:50
train_log_mtime: 2026-06-07 15:19:08
latest_logged_metric: loss 0.04725 / mean_token_accuracy 0.9648 at step 180
GPU: 8 张 H200 继续训练，采样时 99%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 15:14 CST：

- 再次复查，训练继续推进到 step 185：

```text
status: running
progress: 185/445 steps = 41.57%
elapsed_train_time: 11:24:23
train_log_mtime: 2026-06-07 15:11:41
latest_logged_metric: loss 0.04725 / mean_token_accuracy 0.9648 at step 180
GPU: 8 张 H200 继续训练，采样时 66%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 15:11 CST：

- 再次复查，训练继续推进到 step 184：

```text
status: running
progress: 184/445 steps = 41.35%
elapsed_train_time: 11:20:40
train_log_mtime: 2026-06-07 15:07:58
latest_logged_metric: loss 0.04725 / mean_token_accuracy 0.9648 at step 180
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 15:06 CST：

- 再次复查，训练继续推进到 step 183：

```text
status: running
progress: 183/445 steps = 41.12%
elapsed_train_time: 11:17:05
train_log_mtime: 2026-06-07 15:04:23
latest_logged_metric: loss 0.04725 / mean_token_accuracy 0.9648 at step 180
GPU: 8 张 H200 继续训练，采样时 99%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 15:01 CST：

- 再次复查，训练继续推进到 step 182：

```text
status: running
progress: 182/445 steps = 40.90%
elapsed_train_time: 11:13:25
train_log_mtime: 2026-06-07 15:00:44
latest_logged_metric: loss 0.04725 / mean_token_accuracy 0.9648 at step 180
GPU: 8 张 H200 继续训练，采样时 87%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 14:54 CST：

- 再次复查，训练继续推进到 step 180，并打印新的训练过程 metric：

```text
status: running
progress: 180/445 steps = 40.45%
elapsed_train_time: 11:05:57
train_log_mtime: 2026-06-07 14:53:15
loss: 0.04725
grad_norm: 0.7617
learning_rate: 7.291e-06
mean_token_accuracy: 0.9648
epoch: 2.023
GPU: 8 张 H200 继续训练，采样时 99%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9648` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 14:59 CST：

- 再次复查，训练继续推进到 step 181：

```text
status: running
progress: 181/445 steps = 40.67%
elapsed_train_time: 11:09:40
train_log_mtime: 2026-06-07 14:56:58
latest_logged_metric: loss 0.04725 / mean_token_accuracy 0.9648 at step 180
GPU: 8 张 H200 继续训练，采样时 84%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 14:49 CST：

- 再次复查，训练继续推进到 step 179：

```text
status: running
progress: 179/445 steps = 40.22%
elapsed_train_time: 11:02:13
train_log_mtime: 2026-06-07 14:49:31
latest_logged_metric: loss 0.03866 / mean_token_accuracy 0.9838 at step 170
GPU: 8 张 H200 继续训练，采样时 99%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 14:45 CST：

- 再次复查，训练继续推进到 step 178：

```text
status: running
progress: 178/445 steps = 40.00%
elapsed_train_time: 10:57:06
train_log_mtime: 2026-06-07 14:45:36
latest_logged_metric: loss 0.03866 / mean_token_accuracy 0.9838 at step 170
GPU: 8 张 H200 继续训练，采样时 91%-96% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 14:44 CST：

- 再次复查，训练继续推进到 step 177：

```text
status: running
progress: 177/445 steps = 39.78%
elapsed_train_time: 10:53:55
train_log_mtime: 2026-06-07 14:41:13
latest_logged_metric: loss 0.03866 / mean_token_accuracy 0.9838 at step 170
GPU: 8 张 H200 继续训练，采样时 79%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 14:39 CST：

- 再次复查，训练继续推进到 step 176：

```text
status: running
progress: 176/445 steps = 39.55%
elapsed_train_time: 10:50:09
train_log_mtime: 2026-06-07 14:37:27
latest_logged_metric: loss 0.03866 / mean_token_accuracy 0.9838 at step 170
GPU: 8 张 H200 继续训练，采样时 50%-79% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 14:32 CST：

- 再次复查，训练继续推进到 step 174：

```text
status: running
progress: 174/445 steps = 39.10%
elapsed_train_time: 10:42:46
train_log_mtime: 2026-06-07 14:30:04
latest_logged_metric: loss 0.03866 / mean_token_accuracy 0.9838 at step 170
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- 训练过程 metric 仍不能当作 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 14:34 CST：

- 再次复查，训练继续推进到 step 175：

```text
status: running
progress: 175/445 steps = 39.33%
elapsed_train_time: 10:46:26
train_log_mtime: 2026-06-07 14:33:44
latest_logged_metric: loss 0.03866 / mean_token_accuracy 0.9838 at step 170
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 14:22 CST：

- 再次复查，训练继续推进到 step 172：

```text
status: running
progress: 172/445 steps = 38.65%
elapsed_train_time: 10:35:23
train_log_mtime: 2026-06-07 14:22:41
latest_logged_metric: loss 0.03866 / mean_token_accuracy 0.9838 at step 170
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9838` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 14:27 CST：

- 再次复查，训练继续推进到 step 173：

```text
status: running
progress: 173/445 steps = 38.88%
elapsed_train_time: 10:39:02
train_log_mtime: 2026-06-07 14:26:21
latest_logged_metric: loss 0.03866 / mean_token_accuracy 0.9838 at step 170
GPU: 8 张 H200 继续训练，采样时 99%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 14:16 CST：

- 再次复查，训练继续推进到 step 170，并打印新的训练过程 metric：

```text
status: running
progress: 170/445 steps = 38.20%
elapsed_train_time: 10:28:11
train_log_mtime: 2026-06-07 14:15:30
loss: 0.03866
grad_norm: 0.957
learning_rate: 7.594e-06
mean_token_accuracy: 0.9838
epoch: 1.911
GPU: 8 张 H200 继续训练，采样时 68%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9838` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 14:21 CST：

- 再次复查，训练继续推进到 step 171：

```text
status: running
progress: 171/445 steps = 38.43%
elapsed_train_time: 10:31:49
train_log_mtime: 2026-06-07 14:19:07
latest_logged_metric: loss 0.03866 / mean_token_accuracy 0.9838 at step 170
GPU: 8 张 H200 继续训练，采样时 67%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 14:11 CST：

- 再次复查，训练继续推进到 step 169：

```text
status: running
progress: 169/445 steps = 37.98%
elapsed_train_time: 10:24:25
train_log_mtime: 2026-06-07 14:11:43
latest_logged_metric: loss 0.03373 / mean_token_accuracy 0.9857 at step 160
GPU: 8 张 H200 继续训练，采样时 65%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 14:06 CST：

- 再次复查，训练继续推进到 step 167：

```text
status: running
progress: 167/445 steps = 37.53%
elapsed_train_time: 10:16:57
train_log_mtime: 2026-06-07 14:04:15
latest_logged_metric: loss 0.03373 / mean_token_accuracy 0.9857 at step 160
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 14:01 CST：

- 再次复查，训练继续推进到 step 166：

```text
status: running
progress: 166/445 steps = 37.30%
elapsed_train_time: 10:13:21
train_log_mtime: 2026-06-07 14:00:39
latest_logged_metric: loss 0.03373 / mean_token_accuracy 0.9857 at step 160
GPU: 8 张 H200 继续训练，采样时 67%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 13:56 CST：

- 再次复查，训练继续推进到 step 165：

```text
status: running
progress: 165/445 steps = 37.08%
elapsed_train_time: 10:09:35
train_log_mtime: 2026-06-07 13:56:53
latest_logged_metric: loss 0.03373 / mean_token_accuracy 0.9857 at step 160
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 13:53 CST：

- 再次复查，训练继续推进到 step 164：

```text
status: running
progress: 164/445 steps = 36.85%
elapsed_train_time: 10:05:53
train_log_mtime: 2026-06-07 13:53:11
latest_logged_metric: loss 0.03373 / mean_token_accuracy 0.9857 at step 160
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 11:34 CST：

- 再次复查，训练继续推进到 step 126：

```text
status: running
progress: 126/445 steps = 28.31%
elapsed_train_time: 7:46:02
train_log_mtime: 2026-06-07 11:33:20
latest_logged_metric: loss 0.04278 / mean_token_accuracy 0.9828 at step 120
GPU: 8 张 H200 继续训练，采样时 99%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 11:48 CST：

- 再次复查，训练继续推进到 step 130，并打印新的训练 metric：

```text
status: running
progress: 130/445 steps = 29.21%
elapsed_train_time: 8:00:41
train_log_mtime: 2026-06-07 11:48:00
loss: 0.05111
grad_norm: 0.6406
learning_rate: 8.67e-06
mean_token_accuracy: 0.9822
epoch: 1.461
GPU: 8 张 H200 继续训练，采样时 99%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9822` 是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 11:52 CST：

- 再次复查，训练继续推进到 step 131：

```text
status: running
progress: 131/445 steps = 29.44%
elapsed_train_time: 8:04:21
train_log_mtime: 2026-06-07 11:51:39
latest_logged_metric: loss 0.05111 / mean_token_accuracy 0.9822 at step 130
GPU: 8 张 H200 继续训练，采样时 15%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 11:57 CST：

- 再次复查，训练继续推进到 step 132：

```text
status: running
progress: 132/445 steps = 29.66%
elapsed_train_time: 8:08:00
train_log_mtime: 2026-06-07 11:55:18
latest_logged_metric: loss 0.05111 / mean_token_accuracy 0.9822 at step 130
GPU: 8 张 H200 继续训练，采样时 99%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 12:02 CST：

- 再次复查，训练继续推进到 step 134：

```text
status: running
progress: 134/445 steps = 30.11%
elapsed_train_time: 8:15:25
train_log_mtime: 2026-06-07 12:02:43
latest_logged_metric: loss 0.05111 / mean_token_accuracy 0.9822 at step 130
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 12:04 CST：

- 复核 35B run 配置和 checkpoint 机制：

```text
eval_strategy: epoch
save_strategy: best
save_total_limit: 2
load_best_model_at_end: false
current_numbered_checkpoint: checkpoint-89
checkpoint-best: train.py 在 trainer.train() 完整结束后调用 publish_best_checkpoint() 发布
```

判断：

- 现在只有 `checkpoint-89`、没有 `checkpoint-best` 是预期状态，不是卡住。
- 原 pipeline 后续 `cycle_0/llm_adapter/checkpoint-best` 指向全局 `checkpoint-best`，所以必须等本次 SFT 完整结束后才能进入 router / E2E。
- 不提前把 `checkpoint-89` 当正式 LLM adapter 使用；按仓库正式流程等待 `checkpoint-best`。

2026-06-07 12:08 CST：

- 再次复查，训练继续推进到 step 135：

```text
status: running
progress: 135/445 steps = 30.34%
elapsed_train_time: 8:19:06
train_log_mtime: 2026-06-07 12:06:24
latest_logged_metric: loss 0.05111 / mean_token_accuracy 0.9822 at step 130
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 12:13 CST：

- 再次复查，训练继续推进到 step 137：

```text
status: running
progress: 137/445 steps = 30.79%
elapsed_train_time: 8:26:30
train_log_mtime: 2026-06-07 12:13:48
latest_logged_metric: loss 0.05111 / mean_token_accuracy 0.9822 at step 130
GPU: 8 张 H200 继续训练，采样时 1%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 12:18 CST：

- 再次复查，训练继续推进到 step 138：

```text
status: running
progress: 138/445 steps = 31.01%
elapsed_train_time: 8:30:15
train_log_mtime: 2026-06-07 12:17:33
latest_logged_metric: loss 0.05111 / mean_token_accuracy 0.9822 at step 130
GPU: 8 张 H200 继续训练，采样时 76%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 12:23 CST：

- 再次复查，训练继续推进到 step 139：

```text
status: running
progress: 139/445 steps = 31.24%
elapsed_train_time: 8:33:53
train_log_mtime: 2026-06-07 12:21:11
latest_logged_metric: loss 0.05111 / mean_token_accuracy 0.9822 at step 130
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 12:27 CST：

- 再次复查，训练继续推进到 step 140，并打印新的训练 metric：

```text
status: running
progress: 140/445 steps = 31.46%
elapsed_train_time: 8:37:23
train_log_mtime: 2026-06-07 12:24:41
loss: 0.04297
grad_norm: 0.6719
learning_rate: 8.424e-06
mean_token_accuracy: 0.983
epoch: 1.574
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.983` 是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 12:32 CST：

- 再次复查，训练继续推进到 step 142：

```text
status: running
progress: 142/445 steps = 31.91%
elapsed_train_time: 8:44:44
train_log_mtime: 2026-06-07 12:32:02
latest_logged_metric: loss 0.04297 / mean_token_accuracy 0.983 at step 140
GPU: 8 张 H200 继续训练，采样时 99%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 12:36 CST：

- 再次复查，训练继续推进到 step 143：

```text
status: running
progress: 143/445 steps = 32.13%
elapsed_train_time: 8:48:23
train_log_mtime: 2026-06-07 12:35:41
latest_logged_metric: loss 0.04297 / mean_token_accuracy 0.983 at step 140
GPU: 8 张 H200 继续训练，采样时 71%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 12:41 CST：

- 再次复查，训练继续推进到 step 144：

```text
status: running
progress: 144/445 steps = 32.36%
elapsed_train_time: 8:51:58
train_log_mtime: 2026-06-07 12:39:16
latest_logged_metric: loss 0.04297 / mean_token_accuracy 0.983 at step 140
GPU: 8 张 H200 继续训练，采样时 64%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 12:46 CST：

- 再次复查，训练继续推进到 step 145：

```text
status: running
progress: 145/445 steps = 32.58%
elapsed_train_time: 8:55:40
train_log_mtime: 2026-06-07 12:42:58
latest_logged_metric: loss 0.04297 / mean_token_accuracy 0.983 at step 140
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 12:50 CST：

- 再次复查，训练继续推进到 step 147：

```text
status: running
progress: 147/445 steps = 33.03%
elapsed_train_time: 9:02:58
train_log_mtime: 2026-06-07 12:50:16
latest_logged_metric: loss 0.04297 / mean_token_accuracy 0.983 at step 140
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 12:56 CST：

- 再次复查，训练继续推进到 step 148：

```text
status: running
progress: 148/445 steps = 33.26%
elapsed_train_time: 9:06:43
train_log_mtime: 2026-06-07 12:54:01
latest_logged_metric: loss 0.04297 / mean_token_accuracy 0.983 at step 140
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 13:01 CST：

- 再次复查，训练继续推进到 step 149：

```text
status: running
progress: 149/445 steps = 33.48%
elapsed_train_time: 9:10:27
train_log_mtime: 2026-06-07 12:57:45
latest_logged_metric: loss 0.04297 / mean_token_accuracy 0.983 at step 140
GPU: 8 张 H200 继续训练，采样时 59%-86% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 13:05 CST：

- 再次复查，训练继续推进到 step 151，并打印新的训练 metric：

```text
status: running
progress: 151/445 steps = 33.93%
elapsed_train_time: 9:17:54
train_log_mtime: 2026-06-07 13:05:12
loss: 0.05193
grad_norm: 0.6602
learning_rate: 8.161e-06
mean_token_accuracy: 0.98
epoch: 1.686
GPU: 8 张 H200 继续训练，采样时 66%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.98` 是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 13:10 CST：

- 再次复查，训练继续推进到 step 152：

```text
status: running
progress: 152/445 steps = 34.16%
elapsed_train_time: 9:21:36
train_log_mtime: 2026-06-07 13:08:54
latest_logged_metric: loss 0.05193 / mean_token_accuracy 0.98 at step 150
GPU: 8 张 H200 继续训练，采样时 68%-84% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 13:15 CST：

- 再次复查，训练继续推进到 step 153：

```text
status: running
progress: 153/445 steps = 34.38%
elapsed_train_time: 9:25:20
train_log_mtime: 2026-06-07 13:12:38
latest_logged_metric: loss 0.05193 / mean_token_accuracy 0.98 at step 150
GPU: 8 张 H200 继续训练，采样时 65%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 13:17 CST：

- 再次复查，训练继续推进到 step 154：

```text
status: running
progress: 154/445 steps = 34.61%
elapsed_train_time: 9:29:06
train_log_mtime: 2026-06-07 13:16:24
latest_logged_metric: loss 0.05193 / mean_token_accuracy 0.98 at step 150
GPU: 8 张 H200 继续训练，采样时 62%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 13:22 CST：

- 再次复查，训练继续推进到 step 155：

```text
status: running
progress: 155/445 steps = 34.83%
elapsed_train_time: 9:32:40
train_log_mtime: 2026-06-07 13:19:59
latest_logged_metric: loss 0.05193 / mean_token_accuracy 0.98 at step 150
GPU: 8 张 H200 继续训练，采样时 89%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 13:27 CST：

- 再次复查，训练继续推进到 step 157：

```text
status: running
progress: 157/445 steps = 35.28%
elapsed_train_time: 9:39:53
train_log_mtime: 2026-06-07 13:27:11
latest_logged_metric: loss 0.05193 / mean_token_accuracy 0.98 at step 150
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 13:32 CST：

- 再次复查，训练继续推进到 step 158：

```text
status: running
progress: 158/445 steps = 35.51%
elapsed_train_time: 9:43:35
train_log_mtime: 2026-06-07 13:30:53
latest_logged_metric: loss 0.05193 / mean_token_accuracy 0.98 at step 150
GPU: 8 张 H200 继续训练，采样时 3%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 13:38 CST：

- 再次复查，训练继续推进到 step 160，并打印新的训练 metric：

```text
status: running
progress: 160/445 steps = 35.96%
elapsed_train_time: 9:51:03
train_log_mtime: 2026-06-07 13:38:21
loss: 0.03373
grad_norm: 0.8281
learning_rate: 7.884e-06
mean_token_accuracy: 0.9857
epoch: 1.799
GPU: 8 张 H200 继续训练，采样时 36%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9857` 是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 13:44 CST：

- 再次复查，训练继续推进到 step 161：

```text
status: running
progress: 161/445 steps = 36.18%
elapsed_train_time: 9:54:46
train_log_mtime: 2026-06-07 13:42:04
latest_logged_metric: loss 0.03373 / mean_token_accuracy 0.9857 at step 160
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 13:49 CST：

- 再次复查，训练继续推进到 step 163：

```text
status: running
progress: 163/445 steps = 36.63%
elapsed_train_time: 10:02:06
train_log_mtime: 2026-06-07 13:49:24
latest_logged_metric: loss 0.03373 / mean_token_accuracy 0.9857 at step 160
GPU: 8 张 H200 继续训练，采样时 70%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 11:43 CST：

- 再次复查，训练继续推进到 step 128：

```text
status: running
progress: 128/445 steps = 28.76%
elapsed_train_time: 7:53:25
train_log_mtime: 2026-06-07 11:40:43
latest_logged_metric: loss 0.04278 / mean_token_accuracy 0.9828 at step 120
GPU: 8 张 H200 继续训练，采样时 46%-83% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 11:38 CST：

- 再次复查，训练继续推进到 step 127：

```text
status: running
progress: 127/445 steps = 28.54%
elapsed_train_time: 7:49:46
train_log_mtime: 2026-06-07 11:37:04
latest_logged_metric: loss 0.04278 / mean_token_accuracy 0.9828 at step 120
GPU: 8 张 H200 继续训练，采样时 70%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 11:33 CST：

- 再次复查，训练继续推进到 step 125：

```text
status: running
progress: 125/445 steps = 28.09%
elapsed_train_time: 7:42:16
train_log_mtime: 2026-06-07 11:29:35
latest_logged_metric: loss 0.04278 / mean_token_accuracy 0.9828 at step 120
GPU: 8 张 H200 继续训练，采样时 99%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 11:26 CST：

- 再次复查，训练继续推进到 step 124：

```text
status: running
progress: 124/445 steps = 27.87%
elapsed_train_time: 7:38:34
train_log_mtime: 2026-06-07 11:25:52
latest_logged_metric: loss 0.04278 / mean_token_accuracy 0.9828 at step 120
GPU: 8 张 H200 继续训练，采样时 70%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 11:25 CST：

- 再次复查，训练继续推进到 step 123：

```text
status: running
progress: 123/445 steps = 27.64%
elapsed_train_time: 7:34:49
train_log_mtime: 2026-06-07 11:22:07
latest_logged_metric: loss 0.04278 / mean_token_accuracy 0.9828 at step 120
GPU: 8 张 H200 继续训练，采样时 99%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 11:19 CST：

- 再次复查，训练继续推进到 step 122：

```text
status: running
progress: 122/445 steps = 27.42%
elapsed_train_time: 7:31:14
train_log_mtime: 2026-06-07 11:18:32
latest_logged_metric: loss 0.04278 / mean_token_accuracy 0.9828 at step 120
GPU: 8 张 H200 继续训练，采样时 99%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 11:17 CST：

- 再次复查，训练继续推进到 step 121：

```text
status: running
progress: 121/445 steps = 27.19%
elapsed_train_time: 7:27:30
train_log_mtime: 2026-06-07 11:14:49
latest_logged_metric: loss 0.04278 / mean_token_accuracy 0.9828 at step 120
GPU: 8 张 H200 继续训练，采样时 65%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 11:11 CST：

- 再次复查，训练继续推进到 step 120，并打印新的训练过程 metric：

```text
status: running
progress: 120/445 steps = 26.97%
elapsed_train_time: 7:23:53
train_log_mtime: 2026-06-07 11:11:11
loss: 0.04278
grad_norm: 0.6641
learning_rate: 8.899e-06
mean_token_accuracy: 0.9828
epoch: 1.349
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.9828` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 11:05 CST：

- 再次复查，训练继续推进到 step 118：

```text
status: running
progress: 118/445 steps = 26.52%
elapsed_train_time: 7:16:21
train_log_mtime: 2026-06-07 11:03:39
latest_logged_metric: loss 0.04472 / mean_token_accuracy 0.982 at step 110
GPU: 8 张 H200 继续训练，采样时 67%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 11:00 CST：

- 再次复查，训练继续推进到 step 117：

```text
status: running
progress: 117/445 steps = 26.29%
elapsed_train_time: 7:12:48
train_log_mtime: 2026-06-07 11:00:06
latest_logged_metric: loss 0.04472 / mean_token_accuracy 0.982 at step 110
GPU: 8 张 H200 继续训练，采样时部分卡短时 0%，其余 100%，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 10:54 CST：

- 再次复查，训练继续推进到 step 115：

```text
status: running
progress: 115/445 steps = 25.84%
elapsed_train_time: 7:05:27
train_log_mtime: 2026-06-07 10:52:45
latest_logged_metric: loss 0.04472 / mean_token_accuracy 0.982 at step 110
GPU: 8 张 H200 继续训练，采样时 72%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 10:52 CST：

- 再次复查，训练继续推进到 step 114：

```text
status: running
progress: 114/445 steps = 25.62%
elapsed_train_time: 7:01:46
train_log_mtime: 2026-06-07 10:49:04
latest_logged_metric: loss 0.04472 / mean_token_accuracy 0.982 at step 110
GPU: 8 张 H200 继续训练，采样时 96%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 10:46 CST：

- 再次复查，训练继续推进到 step 113：

```text
status: running
progress: 113/445 steps = 25.39%
elapsed_train_time: 6:58:05
train_log_mtime: 2026-06-07 10:45:23
latest_logged_metric: loss 0.04472 / mean_token_accuracy 0.982 at step 110
GPU: 8 张 H200 继续训练，采样时 52%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/router/router_meta.json
cycle_0/e2e_ablation_summary.json
cycle_0/e2e_ablation_summary.md
cycle_1/traces.jsonl
final_ablation_table.md
curve.png
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 10:45 CST：

- 再次复查，训练继续推进到 step 112：

```text
status: running
progress: 112/445 steps = 25.17%
elapsed_train_time: 6:54:21
train_log_mtime: 2026-06-07 10:41:39
latest_logged_metric: loss 0.04472 / mean_token_accuracy 0.982 at step 110
GPU: 8 张 H200 继续训练，采样时 81%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 10:38 CST：

- 再次复查，训练继续推进到 step 111，并打印新的训练过程 metric：

```text
status: running
progress: 111/445 steps = 24.94%
elapsed_train_time: 6:50:40
train_log_mtime: 2026-06-07 10:37:59
loss: 0.04472
grad_norm: 0.5977
learning_rate: 9.109e-06
mean_token_accuracy: 0.982
epoch: 1.236
GPU: 8 张 H200 继续训练，采样时 0%-100% 利用波动，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- `mean_token_accuracy=0.982` 仍是 SFT token 级过程指标，不是 tau2 task accuracy。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。

2026-06-07 10:31 CST：

- 再次复查，训练继续推进到 step 109：

```text
status: running
progress: 109/445 steps = 24.49%
elapsed_train_time: 6:43:20
train_log_mtime: 2026-06-07 10:30:38
latest_logged_metric: loss 0.03776 / mean_token_accuracy 0.9846 at step 100
GPU: 8 张 H200 继续训练，采样时 83%-100% 利用，显存仍约 124GB/card
```

当前仍未生成：

```text
checkpoint-best
checkpoint-final
eval_results.json
cycle_0/llm_adapter/checkpoint-best: symlink exists, still waits for global checkpoint-best
cycle_0/router/router.joblib
cycle_0/e2e_ablation_summary.json
cycle_1/traces.jsonl
```

判断：

- cycle 0 Phase 3 35B SFT 正常推进。
- Router / E2E / 35B task-level eval 仍等待 `checkpoint-best`。
- 这段仍没有新增 API 花费；真实 API cost 仍是 cycle 0 trace 的 `$3.49623640`。
