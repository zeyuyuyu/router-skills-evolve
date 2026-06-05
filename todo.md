# Next Experiment TODO

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
MODEL_SWEEP=13_qwen2_5_coder_1_5b_273 N_CYCLES=1 bash scaling/run_full_pipeline.sh --mock
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

Run a short smoke for each MoE before full cycles:

```bash
MODEL_SWEEP=11_qwen3_5_35b_a3b_273 MAX_STEPS_OVERRIDE=20 bash scaling/run_full_pipeline.sh --mock --n-cycles 1
MODEL_SWEEP=12_qwen3_5_122b_a10b_273 MAX_STEPS_OVERRIDE=20 bash scaling/run_full_pipeline.sh --mock --n-cycles 1
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
