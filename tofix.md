# Scaling Pipeline TODO

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

## Issues To Fix

### 1. LLM training may be skipped after the first cycle

`tau2_train_wrapper.sh` calls the tau2 training pipeline with the same `RUN_CONFIG` each cycle. The real training output still lands under:

```text
experiments/tau2_stage2/train_outputs/$RUN_CONFIG
```

`train_all.sh` skips runs whose `STATUS=done`, so later cycles may not actually retrain even when `cycle_k/training_data.jsonl` changes.

Fix direction: make per-cycle tau2 training output unique, or force retraining when `SCALING_TRAIN_FILE_STAGE2` changes.

### 2. SkillBook small-model stats may not survive adapter path changes

SkillBook stats are keyed by exact `model_id`. Cycle 0 uses the base small model name, while later cycles use adapter paths like:

```text
cycle_0/llm_adapter/checkpoint-best
cycle_1/llm_adapter/checkpoint-best
```

`can_downgrade_to_small(small_model)` may fail to match historical small-model stats and return insufficient data.

Fix direction: normalize model ids into roles such as `small` and `large`, or canonicalize all adapter paths to a stable small-model key.

### 3. Skills have procedures, but inference does not use them yet

`Skill` now stores exemplars and distilled procedures, so it is not only a counter. However, `collect_traces.py` currently uses SkillBook mainly for routing override, not for injecting the procedure into model prompts.

Fix direction: decide whether procedural skills should be fed into small-model inference, SFT data construction, or both.

### 4. Skills update deployment outcomes, not full oracle outcomes

Closed-loop traces contain both `small_success` and `large_success`, but phase2 updates SkillBook mostly from the policy/final outcome.

This is useful as deployment history, but it may not fully learn the real capability table for small vs large on each signature.

Fix direction: consider recording both oracle outcomes into stats while keeping policy outcome as separate deployment metadata.

### 5. Skills ablation may read the wrong SkillBook schema

`run_e2e_ablation_simple.py` appears to expect an older flat skillbook schema, while `SkillBook.save()` writes:

```json
{"skills": [...]}
```

This can make the `skills` ablation variant silently degrade toward always-small.

Fix direction: update `predict_skills` to parse the current `SkillBook.save()` schema.

---

## Resolution (2026-06-05)

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
