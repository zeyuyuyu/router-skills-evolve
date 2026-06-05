# Scaling Pipeline Fix List

## Current Flow

`scaling/run_full_pipeline.sh` is the main closed-loop scaling pipeline.

Each cycle starts with `phase1_collect_traces`.

- Cycle 0 uses the base small model with no previous router or SkillBook.
- Cycles `k >= 1` use the latest artifacts from cycle `k-1`:
  - small model: `cycle_{k-1}/llm_adapter/checkpoint-best`
  - router: `cycle_{k-1}/router/router.joblib`
  - skills: `cycle_{k-1}/skillbook.json`

After trace collection, the default schedule is `SCHEDULE=SLR`:

1. Skills: update/carry over SkillBook from this cycle's `traces.jsonl`, then write `skillbook.json`.
2. LLM: extract hard tasks from this cycle's traces, using `large_completion` as SFT targets, then train this cycle's adapter.
3. Router: train a new router from this cycle's `prompt + small_success` data, then write `router.joblib`.
4. Ablation: summarize this cycle's base/skills/router/full variants.

The artifacts trained in cycle `k` affect inference in cycle `k+1`, not the already-collected traces from cycle `k`.

## Fixed

All five issues fixed. Verified by 2-cycle mock + `tests/test_scaling_smoke.py` (6 passing).

1. **LLM training skipped after first cycle** — `train_all.sh` now force-retrains
   (clears stale `STATUS`) whenever `SCALING_TRAIN_FILE_STAGE2` is set, so each
   scaling cycle retrains on its fresh per-cycle data. Unset => original
   STATUS=done idempotence preserved.

2. **SkillBook stats broken by adapter-path model ids** — Phase 2 now records
   stats under **canonical roles `small` / `large`**, not raw adapter paths.
   `collect_traces._policy_decision` queries `can_downgrade_to_small("small")`.
   Stats survive across cycles regardless of adapter checkpoint path.

3. **Procedures distilled but unused at inference** — `traces_to_sft.py` gains
   `--skillbook`; when given, the matched cluster's distilled procedure is
   prepended to each SFT prompt (the small model now trains WITH the reusable
   scaffold). Phase 3 passes the cycle's `skillbook.json`. Live-inference
   injection remains an operator hook (`SkillBook.get_procedure`).

4. **SkillBook learned policy outcome, not true capability** — Phase 2 now
   ingests **both oracle outcomes** (`small_success` -> role `small`,
   `large_success` -> role `large`, skipping `large_skipped` placeholders),
   building the real per-signature capability table. Policy/deployment outcome
   stays in the trace as metadata.

5. **Ablation read the wrong SkillBook schema** — `run_e2e_ablation_simple.py`
   `predict_skills` now parses the real `{"skills":[{signature, stats:{role:
   [s,t]}}]}` schema (was doing `sb_data.get(sig)` on a dict whose only key is
   `"skills"`, silently degrading the skills variant to always-small). Handles
   canonical roles + legacy model-name keys.

## Open

### 1. Live tau2 completion extraction may still be empty

`experiments/scaling/benches/tau2_bench/adapter.py::_run_one` now calls the real
`Tau2BenchAdapter.run_task(task, config, domain=...)` signature, but the
best-effort completion extraction should be verified on a live tau2 run.

Current risk: `TaskRunResult.messages` is empty unless the tau2 adapter populates
it, and agent content in `TaskRunResult.steps` lives under
`StepData.response["content"]`, not a top-level `StepData.content`.

If `large_completion` is empty on hard tasks, `traces_to_sft.py` will skip them
and Phase 3 will fail or train with no per-cycle additions.

Fix direction: extract completion from `step.response["content"]` and, when the
assistant used tools, serialize `step.response["tool_calls"]` into the SFT target.
