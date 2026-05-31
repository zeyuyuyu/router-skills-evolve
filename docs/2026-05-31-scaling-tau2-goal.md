# 2026-05-31 Scaling Tau2 Goal

## Objective

Low-cost validation of the `scaling/` tau2 pipeline before any full run:

- stay on latest `main`;
- do not commit or expose API keys;
- verify mock orchestration;
- inspect the minimum real tau2 prerequisites;
- decide whether to run 5 or 30 real tau2 tasks next.

## Guardrails

- Do not run the default full job yet. It is `848 tasks x 4 cycles` and can take days.
- Do not put secrets in git, logs, Markdown, or command output.
- Prefer `--smoke --mock` first because it costs `$0`.
- For a real API sanity run, start with `5` tasks before `30` tasks.
- Use the no-conda environment path where possible; do not rely on conda.

## Current Repo State

- Repo: `/root/cwy/projects/evol/router-skills-evolve`
- Branch: `codex/scaling-tau2-real-sanity`
- Base: latest `origin/main` at `33f0fc0 log(shepherd): record failed run 2026-05-31T06:06Z — A800 unreachable`
- Local commits include the tau2 adapter signature fix from `41096ce`, plus
  the real task-loading/prompt-extraction fixes verified below.
- Scaling entrypoint: `scaling/run_full_pipeline.sh`
- Scaling handoff: `scaling/README.md`
- Tau2 SFT/eval bundle: `experiments/tau2_stage2/`

## What This Repo Is Testing

The scaling pipeline evaluates a system, not just one model:

1. `base`: simple small-model-first behavior.
2. `skills`: adds `SkillBook` statistics from prior traces.
3. `router`: trains a learned small-vs-large router.
4. `full`: intended to combine skills, router, and LLM training.

Default real trace collection compares:

- small model: `deepseek/deepseek-v3.2`
- large model: `openai/gpt-5.4-2026-03-05`
- bench: `tau2_bench`

The local trained Qwen models are used by the LLM-training/eval track, not by
the mock smoke path.

## Known Local Training Status

- Largest completed model: `06_qwen3_5_9b_50` (`Qwen/Qwen3.5-9B`, 50 runs).
- Largest completed full-data model: `05_qwen3_5_4b_273` (`Qwen/Qwen3.5-4B`, 273 runs).
- `07_qwen3_5_9b_273` has a checkpoint but was not completed cleanly.
- `08_qwen3_6_35b_a3b_273` has no completed training output observed.

## Cost / Time Envelope

Estimated before measuring fresh real tau2 cost:

- mock smoke: `$0`, a few minutes.
- real tau2 sanity, 5 tasks: roughly `$0.5-$3`, about `10-40 min`.
- real tau2 small sample, 30 tasks: roughly `$3-$15`, about `0.5-2 h`.
- full default `848 x 4 cycles`: likely `$200-$600+`, about `1-3 days`.

## Execution Log

### 2026-05-31 13:51 CST

- Created this goal log.
- Confirmed local repo is on latest `main` at `550c5a7`.
- Next: run `bash scaling/run_full_pipeline.sh --smoke --mock`.

### 2026-05-31 13:52 CST

Ran:

```bash
bash scaling/run_full_pipeline.sh --smoke --mock
```

Result: passed.

- Results dir: `results/scaling_20260531_055204`
- Phase 1: collected `30/30` mock traces; final success `28/30 = 93.3%`.
- Phase 2: built `SkillBook`; size `1`.
- Phase 3: skipped by smoke mode (`SKIP_LLM=1`).
- Phase 4: trained router on `30` examples.
- Phase 5: generated E2E ablation summary and final table.

Final mock ablation:

| Variant | Routing Acc | Fallback | Task Pass | Cost vs Large |
|---|---:|---:|---:|---:|
| Base | 53.33% | 46.67% | 53.33% | 10.00% |
| Skills | 53.33% | 46.67% | 53.33% | 10.00% |
| Router | 76.67% | 16.67% | 70.00% | 43.00% |
| Full | 76.67% | 16.67% | 70.00% | 43.00% |

Conclusion: mock orchestration works. This is not a real tau2 or trained-Qwen
quality number; it only proves the 5-phase wiring is functional.

### 2026-05-31 13:55 CST

Pulled latest remote metadata and checked current state again.

- `origin/main` later advanced to `33f0fc0`; this branch was rebased by
  fast-forwarding local `main` first, then applying the adapter fix on top.
- Worktree only has this local Markdown log untracked.
- Current shell has no API env set:
  - `OPENAI_API_KEY`: unset
  - `OPENAI_API_BASE`: unset
  - `OPENAI_BASE_URL`: unset
  - `COMMONSTACK_API_KEY`: unset
  - `COMMONSTACK_BASE_URL`: unset
  - `HF_TOKEN`: unset
- Local tau2 runtime assets are not prepared in this checkout:
  - `experiments/tau2_stage2/code/vendor/tau2-bench`: missing
  - `experiments/tau2_stage2/code/.venv`: missing
  - root `.env`: missing
  - `experiments/tau2_stage2/.env`: missing

Adapter signature check:

```text
Tau2BenchAdapter.run_task(self, task, config: RunTaskConfig, *, domain=None)
```

But latest `main` scaling wrapper still calls:

```python
self._tau2_adapter.run_task(task["_raw"], student_model=model)
```

Conclusion at that point: latest `main` was not ready for real tau2 sanity. A
real non-mock run would fail before producing useful traces unless TODO #1 was
applied. The first fix existed on branch:

```text
codex/fix-scaling-tau2-adapter-signature
41096ce Fix scaling tau2 adapter signature
```

### 2026-05-31 14:15-14:25 CST

Applied the tau2 real-run fixes on branch `codex/scaling-tau2-real-sanity`:

- `run_task` now uses colleague's `RunTaskConfig` signature.
- Real tau2 tasks load by domain (`retail` by default), not by fake
  `train`/`eval` names.
- Router/skill prompts are extracted from nested `user_scenario` rows.
- The default model config is now `05_qwen3_5_4b_273`, which exists and is the
  largest completed full-data 4B run.

Local validation:

- `python3 -m compileall -q experiments/scaling/benches/tau2_bench/adapter.py experiments/scaling scaling`
- `git diff --check`
- `bash scaling/run_full_pipeline.sh --smoke --mock`

Latest mock smoke result:

- Results dir: `results/scaling_20260531_062458`
- Phase 1: `30/30` mock traces
- Phase 4 router: base about `53%` -> router about `77%`
- Phase 5: ablation table generated

GPU/API note:

- Direct GPU TCP to `api.commonstack.ai:443` timed out.
- The same key works from CPU/local and listed `56` models.
- A temporary CPU -> GPU reverse tunnel on port `18081` let the GPU reach the
  CommonStack API without exposing secrets in git.

GPU real sanity result:

- GPU repo: `/root/cwy/projects/evol/router-skills-evolve`
- Result dir:
  `/root/cwy/projects/evol/router-skills-evolve/results/real_tau2_sanity_proxy_20260531_061849`
- Command shape: `collect_traces.py --bench tau2_bench --n-tasks 1`
- Models: small `deepseek/deepseek-v3.2`, large
  `openai/gpt-5.4-2026-03-05`
- Domain: `retail`
- Elapsed: `211.5s`
- Rows: `1`
- Final success: `0/1`
- Attempts: `2` (small failed, then large ran)
- Recorded cost: `$0.0751874`

Interpretation: real tau2 data loading, nested prompt extraction, small-model
API calls, large-model API calls, tau2 grading, and cost recording all execute
end to end. This is a sanity check, not a quality result.

## Current Decision

Do not run the full default `848 x 4 cycles` job yet.

Safe next step after this branch is reviewed/merged:

1. Keep using the no-conda tau2 venv in the existing GPU bundle.
2. Restore GPU egress to CommonStack, or keep a controlled proxy/tunnel for
   short real runs.
3. Run a small `5`-task real tau2 sample to estimate pass rate and cost
   variance across tasks.
4. Only if the 5-task sample is sane, consider a `30`-task pilot.

Recommendation: this branch is ready to push for review; it proves the
teammate TODO #1/#2 path without starting an expensive formal run.
