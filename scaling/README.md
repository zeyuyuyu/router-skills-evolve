# Scaling Experiment — Router + Skills + LLM Joint Evolve on Larger Models & New Benches

> **目标 (Goal)**: 把主分支已验证的三件套（Skills evolve + Learnable Router + LLM training）从 HumanEval/MBPP × 1.5B 推广到**更大模型** + **更复杂 bench**（默认 τ²-bench，可切 SWE-Bench），跑出**完整 E2E ablation 表**和**多轮迭代曲线**。
>
> **Goal**: Scale the validated three-track pipeline (Skills evolve + Learnable Router + LLM training) from HumanEval/MBPP × 1.5B to **larger models** + **harder benches** (τ²-bench by default, SWE-Bench as stretch). Produce **full E2E ablation table** + **multi-cycle iteration curves**.

---

## ⚠ Status & TODO (READ FIRST)

| 组件 / Component | 状态 / Status | Owner |
|---|---|---|
| `scaling/run_full_pipeline.sh` | ✅ Ready | Zeyu |
| `scaling/README.md` (this file) | ✅ Ready | Zeyu |
| `experiments/scaling/collect_traces.py` (Phase 1) | ✅ Ready | Zeyu |
| `experiments/scaling/benches/tau2_bench/adapter.py` | ✅ **Wrapper fixed and real 1-task tau2 sanity verified** — see §11 TODO #1 | **Teammate** |
| `experiments/scaling/benches/swe_bench/adapter.py` | ❌ **Stub only** — fill in if SWE-Bench is selected | **Teammate** (2–3 days) |
| `experiments/scaling/aggregate_cycles.py` (Phase 6) | ✅ Ready | Zeyu |
| `experiments/scaling/train_router_simple.py` (Phase 4, bench-agnostic) | ✅ Ready | Zeyu |
| `experiments/scaling/run_e2e_ablation_simple.py` (Phase 5, bench-agnostic) | ✅ Ready | Zeyu |
| `experiments/scaling/tau2_train_wrapper.sh` (Phase 3) | 🟡 **`MODE=colleague_corpus` works; `MODE=scaling_traces` needs data injection wired** — see §11 TODO #3 | **Teammate** |
| `experiments/tau2_stage2/` (colleague's SFT framework) | ✅ **Merged from `codex/tau2-stage2-training-eval` into `main`**; teammate's prior work preserved | — |
| `experiments/run_evolve.py` (HumanEval-coupled, **not** used) | n/a | — |
| `experiments/train_learnable_router.py` (HumanEval-coupled, **not** used) | n/a | — |
| `experiments/run_e2e_ablation.py` (HumanEval-coupled, **not** used) | n/a | — |

**Why we wrote new `*_simple.py` versions** for Phases 4 and 5: the main-branch `train_learnable_router.py` and `run_e2e_ablation.py` both hard-code `data/HumanEval.jsonl` for the prompt source (they look up prompts by `task_id` from HumanEval). On tau2-bench / SWE-Bench tasks that lookup always misses → "No supervised router examples could be built from traces". The `*_simple` versions read `prompt` from the trace row directly, so they work for any bench whose adapter follows our trace schema.

**Translation**: pipeline is wired end-to-end and **smoke-tested working** (`bash scaling/run_full_pipeline.sh --smoke --mock` passes all 5 phases on a no-GPU laptop). The tau2 adapter signature and domain task loading have also been real-sanity verified on GPU via a one-task run. Before any large spend, still decide §10 and optionally #3 (scaling-trace-driven LLM training).

---

## 0. 之前实验的硬数字（基线对照）/ Baseline numbers to beat

主分支已经在 8×A800 上跑出的真实结果（见 `docs/E2E_ABLATION_RESULTS.md`、`docs/HANDOFF.md`）：

| 系统 / System | Routing Acc | Large F1 | Fallback | Cost vs always-large | Code pass |
|---|---:|---:|---:|---:|---:|
| Base (always-small + fallback) | 68.28% | 0% | 31.72% | 33.54% | 47/100 |
| + Skills evolve (SkillBook) | 69.46% | 24.93% | 26.65% | 37.27% | 47/100 |
| + Learned router | **93.04%** | **89.48%** | **2.12%** | 37.75% | 47/100 |
| **Full (+ LLM GRPO)** | **93.04%** | **89.48%** | **2.12%** | 37.75% | **49/100** |

LLM 训练侧已有的结论：
- Qwen2.5-Coder-1.5B + local GRPO: **47/100 → 49/100** (MBPP eval100, +2pt)
- **Scaling alone is not enough**：3B GRPO 62→60、7B GRPO 77→75 都退化了

**本次实验要回答的核心问题**：
1. 大模型（4B / 9B / 35B-A3B）+ 更大数据集时，**Router 准确率还能保持 93%+ 吗**？
2. 在 multi-turn agentic bench（τ²-bench）/ 真实 code repo bench（SWE-Bench）上，**Skills evolve 的边际增益**会不会比 HumanEval 上更明显？
3. **`Skills → LLM → Router` 多轮迭代**（main 分支 5/20 的 8-cycle 实验）在大模型上是否仍然 dominant？

---

## 1. Pipeline 总览 / Pipeline overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 0:  Env setup + base model fetch + bench data import          │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 1:  Trace collection                                          │
│           Run bench tasks with BOTH small model + large model.       │
│           Log (task, small_pass, large_pass, small_cost, large_cost) │
│           → produces labelled router examples + DPO chosen/rejected  │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 2:  Skills evolve  (SkillBook)                                │
│           Cluster recurring prompt signatures across successful      │
│           traces → SkillBook with (signature, recommended model,     │
│           success stats).                                            │
│           Output: src/skills/skillbook_v{cycle}.json                 │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 3:  LLM training  (SFT → DPO → GRPO)                          │
│           On the slice where small model failed but learning could   │
│           help. Use tau2_stage2 framework (FSDP2 + FA2 + packing).   │
│           Sweep model sizes per `runs/01..10`.                       │
│           Output: train_outputs/{run_id}/checkpoint-best/            │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 4:  Router training  (Learnable router)                       │
│           Train classifier on labelled examples from Phase 1.        │
│           Includes the trained LLM adapter as "small_model" so       │
│           router learns the NEW boundary.                            │
│           Output: src/learned_router/router_v{cycle}.pkl             │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 5:  E2E ablation                                              │
│           Evaluate all four variants on held-out bench split:        │
│           Base / +Skills / +Router / Full                            │
│           Produces the paper-facing table.                           │
│           Output: results/e2e_ablation_cycle{N}_summary.json         │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
                  Iterate (Cycle N+1) — re-run Phase 1
                  using the new LLM + Router as inner loop
```

**为什么是这个顺序 (Skills → LLM → Router)？**
不是随便排的——MERA 论文（OpenReview 6oyBiDMCHs）和 main 分支 5/20 的 8-cycle 迭代实验都验证过这个 schedule 在 code 域是最优的。本次实验之一就是在新 bench 上验证它是否仍然最优（对照 `LLM → Skills → Router` 等 schedule）。

---

## 2. Prereq / 环境要求

### 硬件 Hardware
- **训练**: 8× H200 (~141 GB/卡) 或 8× A800 80GB（main 分支已验证）
- **CUDA**: 13.0（colleague 的 tau2_stage2 框架要求；A800 上也可用 12.x）
- **磁盘**: ≥500 GB（多个 model checkpoint + bench data + traces）

### 软件 Software
- Python 3.12 (conda or an existing no-conda venv; current GPU sanity used the no-conda tau2_stage2 `.venv`)
- 见 `experiments/tau2_stage2/code/training/requirements.txt`
- 额外 router/skills 需要：`scikit-learn`, `sentence-transformers`, `lightgbm`

### Secrets
- `OPENAI_API_KEY` — Phase 1 用 GPT-5.x 当 large model + eval judge；τ²-bench eval 必需
- `COMMONSTACK_API_KEY` — 如果用 UncommonRoute proxy 收集 traces
- `HF_TOKEN` — Qwen3 系列下载提速；35B-A3B 必需

### Bench 数据 Bench data
- **τ²-bench** (默认 default): 通过 `setup_env_server.sh` 自动 clone `sierra-research/tau2-bench@17e07b1d`
- **SWE-Bench** (可选 optional): 见 `docs/SWE_BENCH_ADAPTER.md`（TODO，下面 §6 有 stub）

---

## 3. Quick start — 一条命令跑通 / one-command run

```bash
# 进 8×H200/A800 服务器
ssh -p 50507 zeyuwang@117.74.66.181   # main 分支 A800
cd ~/router-skills-evolve

# (1) Pull 最新代码 + 切到 main，merge tau2_stage2 框架（详见 §4）
git checkout main && git pull
git merge origin/codex/tau2-stage2-training-eval --no-edit

# (2) 配置环境变量
export OPENAI_API_KEY=sk-...
export HF_TOKEN=hf_...
export BUNDLE_ROOT=$(pwd)/experiments/tau2_stage2
export EXPERIMENT_NAME=scaling_$(date -u +%Y%m%d_%H%M%S)
export BENCH=tau2_bench            # 或 swe_bench (需要先做 §6 SWE adapter)
export MODEL_SWEEP=05_qwen3_5_4b_273   # 选一个 run config；smoke 选 smoke_2b
export N_CYCLES=4                   # MERA 默认 4，main 分支 5/20 跑过 8

# (3) Smoke test (1 cycle, 30 mock tasks, no API/GPU)
bash scaling/run_full_pipeline.sh --smoke --mock

# (4) 真正跑：完整 N-cycle iterated pipeline
bash scaling/run_full_pipeline.sh

# (5) 结果在
# results/scaling_$STAMP/cycle{0..N}/e2e_ablation_summary.json
# results/scaling_$STAMP/curve.png   ← 多轮迭代的 routing acc / code pass 曲线
```

---

## 4. 跟 colleague 的 tau2_stage2 怎么合 / How to merge with colleague's branch

### 合并策略 Merge strategy

**colleague 的 branch 是好东西，要保留；但要补回 Router + Skills 两个 track。**

具体做法：
1. **复用** `experiments/tau2_stage2/code/training/` 整个目录当 LLM track 的训练后端
2. **新增** `experiments/scaling/skills/`（从 main 分支 `src/skills.py` 复制 + 扩展支持多 domain）
3. **新增** `experiments/scaling/router/`（从 main 分支 `src/learned_router/` 复制）
4. **新增** `scaling/run_full_pipeline.sh`（即下面 §5 的 shell）
5. **修改** `experiments/tau2_stage2/code/training/orchestration/train_pipeline.sh` 让它接受 `--training-data` 参数，可以喂 Skills+Router 阶段产生的过滤数据集，不只是 stage2_v1 corpus

### 不要 fork 走的代价 Cost of forking
不要把 colleague 的 branch 当成 dead-end 重写——他做的 FSDP2 + FA2 + packing 的 SFT 框架是干净的、有 75 个单测全过的、bug 都摸出来了。**重写一次就是浪费 2 周。**

### 跟 colleague 沟通的话术 What to tell colleague
（建议直接发给他）

> 你 tau2_stage2 那个 SFT 框架做得很扎实，FSDP2 + FA2 + 75 个单测全过这套我直接复用了。但 scaling 实验需求是把 main 分支的 Skills + Router + LLM 三件套（不只是 LLM SFT）整体推到大模型上，所以我加了 `experiments/scaling/` 把 Skills evolve 和 Router training 接回来，你的 training/ 仍然是 LLM track 的后端。如果方便可以一起看一下我加的 Phase 4 router training 的输入和你 SFT 输出对接的接口，确保 cycle 之间数据流通的。

---

## 5. 同名 shell：`run_full_pipeline.sh`

文件在 `run_full_pipeline.sh`（本目录），主要 flag：

| Flag | Default | 说明 |
|---|---|---|
| `--smoke` | off | 跑 30 个任务、1 cycle、smoke 2B 模型 — 应该在 30 分钟内出结果 |
| `--bench {tau2_bench,swe_bench}` | `$BENCH` | 选 bench；swe_bench 需要先做 §6 adapter |
| `--model-config <name>` | `$MODEL_SWEEP` | tau2_stage2 的 run YAML 名（去掉 `.yaml`） |
| `--n-tasks N` | `$N_TASKS` | real run 的 Phase 1 task 数；默认 tau2 全量 848 |
| `--n-cycles N` | `$N_CYCLES` | 几轮迭代；MERA 推荐 4，main 分支跑过 8 |
| `--schedule {SLR,LSR,LRS,SR-L,...}` | `SLR` | Skills → LLM → Router 顺序；做 ablation 时换 |
| `--resume <cycle>` | none | 从指定 cycle 继续，跳过前面 cycle；Phase 1 trace collection 会跳过已有 `task_id` |
| `--dry-run` | off | 只打印计划，不真跑 |

Cost guard:

```bash
SCALING_MAX_COST_USD=2 bash scaling/run_full_pipeline.sh --n-tasks 30 --skip-llm
```

Phase 1 stops before starting another task once accumulated `total_cost`
reaches the cap. Already-written `traces.jsonl` rows are preserved and skipped
on resume.

If the GPU cannot reach CommonStack directly, run
`python scaling/commonstack_proxy.py --port 18082` on a networked host and
expose it with `ssh -N -R 18082:127.0.0.1:18082 <gpu-host>`, then set
`OPENAI_API_BASE=http://127.0.0.1:18082/v1` on the GPU.

跑完看 `results/$EXPERIMENT_NAME/MANIFEST.json`，里面有每个 phase 的耗时、artifact 路径、success/failure。

---

## 6. 切到 SWE-Bench / Switching to SWE-Bench

τ²-bench 是 multi-turn customer service，SWE-Bench 是 real GitHub issue resolution。切 bench 需要做一个 adapter：

### Adapter contract

```python
# experiments/scaling/benches/{bench_name}/adapter.py
class BenchAdapter:
    def load_tasks(self, split: str) -> list[Task]: ...
    def run_task(self, task: Task, model: str, prompt_signature: dict) -> TaskResult: ...
    def cost_per_task(self, task: Task, model: str, result: TaskResult) -> float: ...
    def judge(self, task: Task, result: TaskResult) -> bool: ...
```

### SWE-Bench 已知 gotcha
- 需要 Docker（每个 task 一个 sandbox）
- SWE-Bench Lite 是 300 tasks，Verified 是 500——推荐先用 Lite smoke
- patch 形式的 result，需要 `git apply` 验证 + `pytest` 验证
- per-task cost 比 τ²-bench 高一个数量级（complex repo context）

### SWE-Bench 接入清单
- [ ] `experiments/scaling/benches/swe_bench/` 目录
- [ ] `adapter.py` 实现上面 4 个 method
- [ ] `tasks_train.jsonl` / `tasks_eval.jsonl` （SWE-Bench Lite split）
- [ ] Docker harness（可参考 SWE-Bench 官方 `evaluation/` 目录）
- [ ] 修改 `run_full_pipeline.sh` Phase 1 让 trace collection 调 Docker

**预估**: 一个熟练工程师 2-3 天能接上 SWE-Bench Lite；接 Verified 多一天做 split 校准。

---

## 7. Output schema / 输出格式

`results/$EXPERIMENT_NAME/` 下结构：

```
$EXPERIMENT_NAME/
├── MANIFEST.json                 # 整个 run 的元数据 + 各 phase 状态
├── config_snapshot.json          # 复现用的所有 env vars + flag
├── cycle_0/
│   ├── traces.jsonl              # Phase 1 输出
│   ├── skillbook.json            # Phase 2 输出
│   ├── llm_adapter/              # Phase 3 输出 (LoRA / full SFT)
│   ├── router.pkl                # Phase 4 输出
│   └── e2e_ablation_summary.json # Phase 5 输出
├── cycle_1/  ...
├── cycle_N/  ...
├── curve.png                     # 多 cycle 的 routing acc / code pass 曲线
└── final_ablation_table.md       # paper-facing 表，对照 §0 baseline
```

`e2e_ablation_summary.json` schema（兼容 main 分支现有格式）：

```json
{
  "cycle": 0,
  "bench": "tau2_bench",
  "model_config": "05_qwen3_5_4b_273",
  "n_eval": 848,
  "variants": {
    "base": {"routing_acc": 0.6828, "large_f1": 0.0, "fallback": 0.3172, "cost_vs_large": 0.3354, "task_pass": 0.47},
    "skills": {"routing_acc": ..., ...},
    "router": {"routing_acc": ..., ...},
    "full": {"routing_acc": ..., ...}
  },
  "schedule": "SLR",
  "git_sha": "abc1234",
  "started_at": "2026-06-01T08:00:00Z",
  "duration_s": 14400
}
```

---

## 8. Common pitfalls / 踩坑预警

1. **`enable_thinking=False`**: Qwen3.5/3.6 thinking 模式必须关——tau2_stage2 corpus 是无 CoT 的 (见 design doc §2.2)，开了会 OOD
2. **flash-attn build**: `MAX_JOBS=4` 不要更高，否则 12-15GB/job 会 OOM
3. **A800 vs H200**: A800 → packing 用 `bfd` + `max_seq_length=16384`；H200 → 可以 `max_seq_length=32768`
4. **多轮迭代不要 stale**: 每个 cycle 的 Phase 1 traces 必须用**上一轮**训出来的 LLM 和 Router 当 "small_model"，否则 Router 学不到新边界（main 分支 5/20 实验掉过坑）
5. **shepherd 自动化**: 仓库根目录的 `shepherd*` 文件是 cloud cron 写的、跟训练无关；运行前 `rm -rf shepherd* .shepherd*` 一遍，否则会污染 trace 收集
6. **SWE-Bench Docker timeout**: per-task 默认 5min 不够；改 `SWE_BENCH_TASK_TIMEOUT=1800`
7. **学习率 schedule**: tau2_stage2 默认 cosine + 5% warmup；大模型 (>9B) 改 `1e-5`，小模型 (≤4B) 用默认 `2e-5`

---

## 9. Smoke test 通过标准 / Smoke acceptance criteria

`bash scaling/run_full_pipeline.sh --smoke --mock` 跑完后，下面几件事都满足才算 mock smoke 过：

- [ ] `cycle_0/traces.jsonl` 行数 ≥ 30
- [ ] `cycle_0/skillbook.json` 存在且包含至少 1 个 signature（mock prompts intentionally cluster tightly）
- [ ] `cycle_0/llm_adapter/STATUS` 为 `skipped`（smoke 默认跳过 LLM 训练）
- [ ] `cycle_0/router/router.joblib` 存在
- [ ] `cycle_0/e2e_ablation_summary.json` 中 `variants.full.routing_acc > variants.base.routing_acc` （任意 margin，证明 router 起作用了）
- [ ] `final_ablation_table.md` 和 `curve.png` 生成

mock smoke 应在 CPU 上几分钟内跑完，不需要 API key 或 GPU。

---

## 10. 提问 / Open questions

发同事之前希望 Zeyu 确认：

1. **Bench 是 τ²-bench 还是 SWE-Bench？** 默认走 τ²（colleague 已经把环境跑通），SWE 多 2-3 天接 adapter
2. **模型 sweep 范围？** 默认从 colleague 的 `runs/01..10` 里挑哪几个跑？建议起手 `01_qwen3_5_2b_273`（small floor）+ `05_qwen3_5_4b_273`（main candidate）+ `07_qwen3_5_9b_273`（high capacity） 三个，35B-A3B 等 small/medium 出曲线后再决定
3. **N cycles**：4 还是 8？8 cycle 时间约 2× 但能更稳验证 schedule
4. **要不要做 schedule ablation**？(SLR vs LSR vs LRS)；要做的话训练时间 ×3

---

## 11. Teammate TODO (with effort estimates)

### TODO #1 — Verify tau2 adapter `run_task` signature  (status: done for current tau2_stage2 bundle)

**Where**: `experiments/scaling/benches/tau2_bench/adapter.py:_run_one`

`_run_one` now matches the colleague adapter signature:
`run_task(task, config: RunTaskConfig, *, domain=None)`. It constructs a
`RunTaskConfig` with `agent`, `user`, `seed`, `max_steps`, and `max_errors`
before calling tau2. The wrapper also loads tau2 tasks by domain
(`retail`/`airline`/`telecom`) and extracts prompts from nested
`user_scenario` rows, which are both required for the real tau2 bundle.

Verified on 2026-05-31 with:

- local mock orchestration: `bash scaling/run_full_pipeline.sh --smoke --mock`
- GPU real sanity: `1` retail task, `deepseek/deepseek-v3.2` small model,
  `openai/gpt-5.4-2026-03-05` large model, nonzero recorded cost

To re-verify after upstream tau2_stage2 changes:

```bash
# Activate the tau2_stage2 venv/environment first
cd experiments/tau2_stage2

# Inspect the real signature
PYTHONPATH=code python -c "
from adapters.tau2_bench.adapter import Tau2BenchAdapter
import inspect
print(inspect.signature(Tau2BenchAdapter.run_task))
"
```

If this signature changes upstream, update `_run_one` in the scaling adapter.

**Smoke test the fix**:
```bash
SCALING_MOCK=0 OPENAI_API_KEY=$YOUR_KEY \
  python experiments/scaling/collect_traces.py \
    --bench tau2_bench --n-tasks 5 \
    --small-model deepseek/deepseek-v3.2 \
    --large-model openai/gpt-5.4-2026-03-05 \
    --out /tmp/test_traces.jsonl
# Expect: 5 rows in /tmp/test_traces.jsonl with non-empty final_model + non-zero total_cost
```

### TODO #2 — Run full smoke (mock) on your machine  (effort: ~5 min)

Just to verify the pipeline orchestration is wired correctly before doing any GPU work:

```bash
bash scaling/run_full_pipeline.sh --smoke --mock
```

Should produce `results/scaling_*/cycle_0/` with `traces.jsonl`, `skillbook.json`, `router/`, `e2e_ablation_summary.json`, and `final_ablation_table.md` — all using synthetic data, no API/GPU. Confirms shell + Python imports are clean on your box.

### TODO #3 — (Optional) Wire scaling traces into tau2_stage2 SFT  (effort: 2–3 days)

**Where**: `experiments/scaling/tau2_train_wrapper.sh` — currently defaults to `MODE=colleague_corpus` (trains on the colleague's pre-existing `data_processed/stage2_v1/` corpus, ignoring our Phase 3 extracted slice).

**Why optional**: with `MODE=colleague_corpus`, the LLM track trains on the same data every cycle (so cycle-N LLM ≈ cycle-0 LLM). Skills + Router still iterate properly, so we still get a valid 3-track ablation, just with a flat LLM line.

**To wire scaling traces** (Phase 3 traces → LLM training data):
1. Write `experiments/scaling/convert_traces_to_stage2.py` that turns `training_data.jsonl` (output of `extract_training_data.py`) into colleague's prompt-completion format. See `experiments/tau2_stage2/code/training/data/convert_to_prompt_completion.py` for the target schema.
2. Drop into `experiments/tau2_stage2/data_processed/stage2_v1/train.jsonl` (overwrite or new split).
3. Update `_build_meta.json` so the validator passes (colleague has `data_audit.py` that checks).
4. Change `tau2_train_wrapper.sh:MODE=scaling_traces` branch to call the converter then invoke `train_pipeline.sh`.

For the first scaling run, **leave as `colleague_corpus`** — get the system working end-to-end first, then wire trace-driven LLM training as a v2.

### TODO #4 — Decide Open Questions (§10)  (effort: 15 min discussion with Zeyu)

The four open questions in §10 (bench / model sweep / N cycles / schedule ablation) need to be locked before kicking off the real run. Ping Zeyu on WeChat when you're at this step.

### TODO #5 — (If SWE-Bench is chosen) Implement SWE-Bench adapter  (effort: 2–3 days)

See `experiments/scaling/benches/swe_bench/adapter.py` module docstring for the full implementation TODO. Only do this if §10 Q1 lands on SWE-Bench.

---

## 12. 一句话给同事的话术 / One-liner for handoff

> Pipeline 接好了，`scaling/` 在 main 上。直接 `bash scaling/run_full_pipeline.sh --smoke --mock` 试一下 orchestration，过了之后看 `scaling/README.md §11` 的 5 个 TODO；TODO #1 是 30 分钟的 tau2 adapter 签名 fix，#2 是 mock 验证，剩下三个按情况做。卡住任何一步直接微信我。

---

**Maintainer**: Zeyu Wang (zeyu.wang@0g.ai)
**Last updated**: 2026-05-31
