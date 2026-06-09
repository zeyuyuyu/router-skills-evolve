# Next Experiment TODO

## 2026-06-09 immediate action

User-approved runtime target: finish the full 35B tau2 skills + LLM + router
closed-loop run within about one day if possible.

Current delivery status:

- Code fixes were committed and pushed before the corrected run.
- The corrected run is active on a shared 8-GPU worker.
- Cycle 1 is complete: trace skip, SkillBook, bounded SFT, router, and E2E ablation.
- Cycle 1 Phase 3 trained with bounded replay:
  `512` base replay rows + `4` hard trace rows repeated `16` times.
- Cycle 1 E2E result: base/skills task `85.14%`, router/full task `89.19%`.
- Cycle 2 Phase 1 has been cleanly restarted after a vLLM Qwen3.5 text-only
  compatibility fix; the new trace file has started writing:
  `results/cwy_35b_joint_20260606_165203/cycle_2/traces.jsonl`.
- Cycle 2 was restarted again after confirming the previous rows had empty
  `small_completion`. Serving now uses `MAX_MODEL_LEN=32768` and an empty
  `REASONING_PARSER` so tau2 receives normal chat `content` instead of
  reasoning-only responses.
- Current verification: the first two clean Cycle 2 traces have non-empty
  35B student completions and no context-window / empty-assistant errors.
- The tau2 adapter now keeps completed trajectories when evaluation fails.
  Retail evaluator JSON/golden-action exceptions mark reward as failed, but
  no longer discard `small_completion` / `large_completion`.
- Current verification after the adapter patch: Cycle 2 was restarted cleanly
  again, and the first four trace rows all retain non-empty completions.
- Latest watch: Cycle 2 trace collection has reached `6` rows; sampled rows
  have no empty-both completion records, and the local 35B student service is
  healthy.
- Latest watch: Cycle 2 trace collection has reached `8` rows. The tau2 train
  split skips some numeric task ids by design; this was checked against the
  loaded task list. The latest row has non-empty student and large completions.
- Latest watch: Cycle 2 trace collection has reached `10` rows. Current
  sampled quality check: `small_empty=0`, `empty_both=0`, service healthy.
- Latest watch: Cycle 2 trace collection has reached `12` rows. Current
  sampled quality check remains `small_empty=0`, `empty_both=0`.
- Latest watch: Cycle 2 trace collection has reached `14` rows. Current
  quality check remains `small_empty=0`, `empty_both=0`; the latest row keeps
  the 35B student completion even when the student fails, and the large-model
  fallback succeeds with `final_success=True`.
- Latest watch: Cycle 2 trace collection has reached `49/74` rows. The
  full-pipeline command keeps the tau2 cap parameter, but the current train
  split actually loaded `74` tasks, not `848` traces. Current quality check
  remains `small_empty=0`, `large_empty=0`, `empty_both=0`; current
  `final_success=38/49`. The latest row has `final_success=True` and keeps
  non-empty student and large completions, so collection quality remains
  clean.
- The student model cost-mapping warning still appears in logs, but it is
  currently non-blocking and trace rows continue to be written.
- Runtime environment fixes applied on the worker:
  torch `2.11.0`, flash-attn `2.8.3`, torchvision `0.26.0`, all inside the run venv.
- Runtime vLLM patch applied on the worker: language-model-only skips vision
  tower and uses plain text mrope positions for text-only requests.
- Current watch item: let Cycle 2 trace collection finish, then verify the
  downstream SkillBook, LLM SFT, router training, and E2E ablation stages.

Code change before rerun:

- `experiments/scaling/tau2_train_wrapper.sh`
  - scaling SFT now defaults to bounded replay instead of full 6413-row replay.
  - default `SCALING_BASE_REPLAY_ROWS=512`.
  - default `SCALING_TRACE_REPEAT=16`.
  - writes `scaling_sft/replay_mix_meta.json`.
- `experiments/tau2_stage2/code/training/orchestration/train_all.sh`
  - scaling-generated run configs accept env overrides such as
    `SCALING_NUM_TRAIN_EPOCHS=1`.

Launch policy for the corrected run:

```bash
export SCALING_BASE_REPLAY_ROWS=512
export SCALING_TRACE_REPEAT=16
export SCALING_BASE_REPLAY_SEED=1234
export SCALING_NUM_TRAIN_EPOCHS=1
export SCALING_ALLOW_MISSING_VAL=1
export SCALING_EVAL_STRATEGY=no
export SCALING_SAVE_STRATEGY=epoch
export SCALING_LOAD_BEST_MODEL_AT_END=false
export EVOL_DISABLE_AUTO_RESUME=1
```

Reason for `EVOL_DISABLE_AUTO_RESUME=1`: old `checkpoint-89` belongs to the
previous full-replay 5-epoch recipe. The corrected recipe has far fewer steps,
so auto-resuming Trainer optimizer/scheduler state from step 89 is incompatible.
Keep the checkpoint in shared storage for audit/backup, but do not auto-resume
it for the short hard-example adaptation run.

Current worker:

```text
shared 8-GPU worker
repo path: router-skills-evolve/
experiment path: results/cwy_35b_joint_20260606_165203/
```

Before experiment launch:

- commit and push code.
- sync repo to the private worker.
- restore experiment dir from shared storage, but protect/archive old cycle_1
  `llm_adapter` state or disable auto-resume as above.
- if the portable bundle lacks `stage2_v1/train.jsonl`, restore the 6413 base
  rows from the previous `train_augmented_stage2.jsonl` by excluding
  `_p.source == scaling_traces`; for this corrected run, val can be skipped via
  `SCALING_ALLOW_MISSING_VAL=1` with `SCALING_EVAL_STRATEGY=no`.
- start tmux and verify logs show bounded replay and `EVOL_DISABLE_AUTO_RESUME=1`.

## 当前唯一正式任务：35B Skills + LLM + Router 联合演进

状态：准备正式启动。

不要再把单模型 SFT / 单模型 eval 当成最终任务。当前任务必须完整跑：

1. 真实 tau2 traces 收集，不用 mock。
2. Skills evolve，生成 / 更新 `skillbook.json`。
3. LLM train，用仓库正式 35B 配置训练 SFT checkpoint。
4. Router train，生成 `router.joblib`。
5. E2E ablation，报告 `base / skills / router / full` 四组。
6. 每轮 cycle 都记录 accuracy、task pass、fallback、cost、训练产物和日志。

仓库正式 35B 配置：

```text
run_id: 08_qwen3_6_35b_a3b_273
model: Qwen/Qwen3.6-35B-A3B
revision: 995ad96eacd98c81ed38be0c5b274b04031597b0
config: experiments/tau2_stage2/code/training/configs/runs/08_qwen3_6_35b_a3b_273.yaml
```

注意：用户提到过 `Qwen/Qwen3.5-35B-A3B`，但当前仓库已有正式 35B 配置是
`Qwen/Qwen3.6-35B-A3B`。本次按仓库正式配置跑，不新增、不替换成其他模型。

论文 `routerevolving-2.pdf` 的报告口径：

- `Router accuracy`
- `Fallback`
- `Cost vs always-large`
- `LLM pass / Task pass`
- 每轮 `Full = Skills + LLM + Router` 的 progression

本次先跑论文主设置：

```text
schedule: SLR = Skills -> LLM -> Router
n_cycles: 4
bench: tau2_bench
model_sweep: 08_qwen3_6_35b_a3b_273
mock: false
skip_llm: false
```

正式记录文档：

```text
CWY_35B_JOINT_EVOLUTION.md
```

## Goal

Run a larger-model / harder-benchmark version of the current scaling pipeline.

Targets to add:

- 1.5B baseline / rerun for continuity with the HumanEval/MBPP result
- 4B tau2 closed-loop run as the first real scaling target
- `Qwen/Qwen3.5-35B-A3B`
- `Qwen/Qwen3.5-122B-A10B`
- SWE-Bench support, starting with SWE-Bench Lite

Keep `tofix.md` as the pipeline correctness / bug-fix checklist. This file tracks the next experiment expansion.

## Small / Dense Baseline Runs

Run these before the big MoE jobs so we have a cheap sanity curve and a
comparison point against the earlier 1.5B MBPP/GRPO result.

### 1. 1.5B continuity run

Goal: rerun the best 1.5B setting on the current codebase and record whether the
previous positive signal still holds.

Suggested starting point:

```text
Qwen/Qwen2.5-Coder-1.5B-Instruct
```

TODO:

- confirm whether this is a legacy HumanEval/MBPP-only run or should be wired
  into the tau2 scaling pipeline as a new run config
- if tau2: add a dedicated run YAML under
  `experiments/tau2_stage2/code/training/configs/runs/`
- run a 1-cycle smoke first, then a 2-cycle closed-loop run
- save base vs adapter eval and note whether SFT/GRPO improves or regresses

Candidate commands:

```bash
# Legacy MBPP/HumanEval continuity path, if using the old 1.5B code setup:
# use the existing qwen25_coder_15b GRPO/SFT scripts documented in docs/HANDOFF.md

# Tau2 path, after adding a run config:
MODEL_SWEEP=13_qwen2_5_coder_1_5b_273 N_CYCLES=1 bash scaling/run_full_pipeline.sh --mock --skip-llm
MODEL_SWEEP=13_qwen2_5_coder_1_5b_273 N_CYCLES=2 bash scaling/run_full_pipeline.sh
```

### 2. 4B tau2 closed-loop run

Goal: make `05_qwen3_5_4b_273` the first real end-to-end tau2 run after the
pipeline correctness fixes.

Use the existing config:

```text
experiments/tau2_stage2/code/training/configs/runs/05_qwen3_5_4b_273.yaml
```

Recommended sequence:

```bash
# no API/GPU smoke
MODEL_SWEEP=05_qwen3_5_4b_273 N_CYCLES=2 bash scaling/run_full_pipeline.sh --mock --skip-llm

# short real pilot on server
MODEL_SWEEP=05_qwen3_5_4b_273 N_CYCLES=1 MAX_STEPS_OVERRIDE=20 bash scaling/run_full_pipeline.sh

# first real closed-loop run
MODEL_SWEEP=05_qwen3_5_4b_273 N_CYCLES=2 bash scaling/run_full_pipeline.sh
```

Before treating the 4B result as valid, verify the open item in `tofix.md`:
`large_completion` must be non-empty for hard tasks so Phase 3 actually receives
per-cycle SFT data.

## Big MoE Runs

### 1. Verify model availability and local/offline paths

Before adding configs, verify the exact Hugging Face repos and revisions exist:

```bash
huggingface-cli repo show Qwen/Qwen3.5-35B-A3B
huggingface-cli repo show Qwen/Qwen3.5-122B-A10B
```

Then add local model resolution aliases if needed in:

```text
experiments/tau2_stage2/code/training/model_resolution.py
```

Also update offline/download scripts if these models should be available on the server:

```text
experiments/tau2_stage2/code/training/orchestration/prepare_offline_artifacts.sh
experiments/tau2_stage2/code/training/eval/fetch_base_models.py
```

### 2. Add run configs

Create new YAMLs under:

```text
experiments/tau2_stage2/code/training/configs/runs/
```

Suggested names:

```text
11_qwen3_5_35b_a3b_273.yaml
12_qwen3_5_122b_a10b_273.yaml
```

Use the existing MoE config as the starting point:

```text
08_qwen3_6_35b_a3b_273.yaml
```

Check/update:

- `model.name`
- `run_id`
- `training.output_dir`
- learning rate
- batch size / grad accumulation
- max sequence length
- FSDP / MoE accelerate config
- checkpoint save cadence

### 3. Smoke test before full training

Run a pipeline-only mock smoke for each MoE before full cycles:

```bash
MODEL_SWEEP=11_qwen3_5_35b_a3b_273 bash scaling/run_full_pipeline.sh --mock --n-cycles 1 --skip-llm
MODEL_SWEEP=12_qwen3_5_122b_a10b_273 bash scaling/run_full_pipeline.sh --mock --n-cycles 1 --skip-llm
```

Then run a real 20-100 step training smoke on the server with GPUs.

Watch for:

- FSDP all-gather OOM
- expert routing / MoE aux loss issues
- tokenizer/chat-template validation failures
- context length clipping
- vLLM serving failure for eval

### 4. Add eval scheduling for large MoE

Update eval scripts so 35B/122B use safe tensor parallelism and do not run alongside smaller models:

```text
experiments/tau2_stage2/code/training/orchestration/eval_all.sh
experiments/tau2_stage2/code/training/orchestration/eval_all_step_budget.sh
```

Likely policy:

- 35B-A3B: TP=8, sequential
- 122B-A10B: verify memory first; may require TP=8 plus stricter max model length or external serving

### 5. Add plotting/summary family names

Update family detection and plotting labels:

```text
experiments/tau2_stage2/code/training/orchestration/plotting.py
experiments/tau2_stage2/code/training/orchestration/plotting_step_budget.py
experiments/tau2_stage2/code/training/orchestration/summarize_step_budget.py
```

Make sure active-parameter counts are reported separately from total parameter counts.

## SWE-Bench

### 1. Implement the adapter

Current file is a stub:

```text
experiments/scaling/benches/swe_bench/adapter.py
```

Implement:

- `load_tasks(n, split)`
- `run_task_pair(task, small_model, large_model, cycle, force_both=False)`

Start with SWE-Bench Lite before Verified.

### 2. Add task data and splits

Prepare:

```text
experiments/scaling/benches/swe_bench/tasks_train.jsonl
experiments/scaling/benches/swe_bench/tasks_eval.jsonl
```

Each row should include at least:

- `task_id`
- `prompt` / issue statement
- repo metadata
- base commit
- test command or official harness metadata

### 3. Wire Docker/harness execution

SWE-Bench requires isolated repo sandboxes.

TODO:

- reuse official SWE-Bench Docker images or official evaluation harness
- set per-task timeout, likely `SWE_BENCH_TASK_TIMEOUT=1800`
- capture generated patch/diff as completion text
- record pass/fail from official tests
- make runs resume-safe at task level

### 4. Preserve the same trace schema

The adapter must emit the schema consumed by the scaling pipeline:

- `task_id`
- `signature`
- `decision`
- `attempts`
- `attempts_count`
- `final_success`
- `final_model`
- `total_cost`
- `round`
- `small_success`
- `large_success`
- `small_cost`
- `large_cost`
- `prompt`
- `small_completion`
- `large_completion`

For SWE-Bench, `small_completion` / `large_completion` should probably be the generated patch or structured trajectory that produced the patch.

### 5. Cost and runtime guardrails

SWE-Bench is much more expensive than tau2.

Add:

- `--n-tasks` override for small pilot runs
- cost cap / max tasks per cycle
- timeout defaults
- per-task failure rows instead of hard aborts
- resume logic

Recommended first pilots:

```bash
BENCH=swe_bench N_CYCLES=1 N_TASKS=20 MODEL_SWEEP=05_qwen3_5_4b_273 bash scaling/run_full_pipeline.sh
BENCH=swe_bench N_CYCLES=2 N_TASKS=50 MODEL_SWEEP=05_qwen3_5_4b_273 bash scaling/run_full_pipeline.sh
```

## Additional TODOs

### 1. Verify correctness checklist status

Before trusting new big runs, verify or explicitly accept the open item in:

```text
tofix.md
```

Current priority: confirm live tau2 `large_completion` extraction works for hard
tasks. If it is empty, patch the scaling tau2 adapter before running 4B/35B/122B.

### 2. Add per-cycle provenance

Every cycle should record:

- exact model repo/revision or local checkpoint path
- train data sha256
- trace file sha256
- router data sha256
- SkillBook sha256
- git commit
- schedule
- bench
- domain/split

This should go into each cycle directory as `cycle_manifest.json`.

### 3. Add sanity tests for the closed loop

Add tests that assert:

- cycle 1 trace collection loads cycle 0 router and SkillBook when present
- phase3 refuses empty SFT data instead of silently training fixed corpus
- router training uses this cycle's traces
- SkillBook carries over from the previous cycle
- ablation parses current SkillBook schema

### 4. Add explicit large-model budget gates

For 35B/122B runs, fail fast unless the operator confirms:

- enough free disk
- enough GPUs
- enough GPU memory
- expected wallclock
- expected API/eval spend

### 5. Re-run step-budget eval after each promoted checkpoint

Once a big MoE checkpoint trains successfully, run step-budget eval and save:

```text
experiments/tau2_stage2/step_budget_outputs/SUMMARY_step_budget.csv
experiments/tau2_stage2/step_budget_outputs/SUMMARY_step_budget.json
experiments/tau2_stage2/step_budget_outputs/plots/
```

Then update:

```text
experiments/tau2_stage2/RESULTS_2026-05-31.md
```

or create a new dated results file.
