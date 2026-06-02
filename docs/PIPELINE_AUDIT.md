# Scaling pipeline audit (2026-05-21)

Triggered by review feedback: *"Phase 1 currently uses only the previous
round's updated LLM adapter, not the previous round's router or SkillBook.
Is this by design? Phase 1 (`experiments/scaling/collect_traces.py`) should
include the latest skills / LLM / router."*

## What was checked

`scaling/run_full_pipeline.sh` runs each cycle as five phases under a
configurable schedule (default `SLR` = Skills → LLM → Router):

```
phase1_collect_traces  → traces.jsonl
phase2_skills_evolve   → skillbook.json   (carries over from cycle k-1 ✓)
phase3_llm_train       → llm_adapter/     (resumes from cycle k-1 ✓)
phase4_router_train    → router/router.joblib
phase5_e2e_ablation    → e2e_ablation_summary.json
```

## Finding (confirmed bug)

Cycle k's **Phase 1 trace collection** was only closing the loop on **one** of
the three evolvers:

| Evolver | Carried into Phase 1 of cycle k? | Mechanism |
| --- | --- | --- |
| LLM adapter | ✅ yes | `--small-model = cycle_{k-1}/llm_adapter/checkpoint-best` |
| SkillBook | ❌ **no** | not passed to `collect_traces.py` |
| Router | ❌ **no** | not passed to `collect_traces.py` |

`collect_traces.py` ran a fixed `small → (fallback) large` heuristic baked into
each bench adapter (`run_task_pair`), ignoring the trained router and the
carried-over SkillBook entirely. The evolved routing policy therefore never
influenced the traces that trained the *next* cycle's router/skills — the loop
was open on two of three tracks.

## Fix

`collect_traces.py` now accepts optional `--router cycle_{k-1}/router/router.joblib`
and `--skillbook cycle_{k-1}/skillbook.json`. When present (cycle ≥ 1) each
trace row gains the decision the **current evolved policy** would make:

```
policy_route          "small" | "large"
policy_router_prob     P(needs large) from the trained router
policy_skill_verdict   True/False/None from the SkillBook signature
policy_final_model     model the policy would have billed
policy_total_cost      cost under the policy decision
policy_decision        "policy:small_ok" | "policy:small_fail->large" | "policy:route_large"
```

The SkillBook verdict (grounded in observed per-cluster success counts)
overrides the router when confident; otherwise the router's probability decides.

**Both** oracle model outcomes (`small_success`, `large_success`) are still
recorded so the next cycle's router trainer keeps clean labels — the `policy_*`
fields close the loop *without* introducing the label bias you'd get from only
ever observing the routed model. `run_full_pipeline.sh phase1_collect_traces`
now passes the two artifacts automatically for cycle ≥ 1.

Verified via 2-cycle mock run:

```
Cycle 0:  closed_loop=False                       (no prior artifacts — correct)
Cycle 1:  loaded router from cycle_0/router/router.joblib
          loaded SkillBook (… signatures) from cycle_0/skillbook.json
          closed_loop=True
          policy_route: small=… large=…
```

If `--router`/`--skillbook` are omitted (cycle 0, or one-off ablation runs) the
behaviour is byte-for-byte identical to before.

## Still open — richer skills (separate review item)

Second review request: *"the skill is currently only success-rate statistics;
it should learn how to solve the task / tool-use steps / domain policy /
reusable code snippets / agent sub-workflows."*

This is a larger design change to `src/skills.py`, moving the SkillBook from a
**frequency table** (signature → per-model success counts) to a **procedural
skill library** (signature → distilled solution scaffold). Proposed shape:

```
Skill {
    signature: str
    stats:      {model_id: (succ, total)}      # keep — used by routing
    procedure:  str   # distilled "how to solve this cluster" (tool-use steps,
                      #             domain policy, reusable snippet)
    exemplars:  [trace_id, ...]                # successful trajectories
    embedding:  vector                          # for nearest-signature recall
}
```

Generation path (agent sub-workflow, mirrors Voyager-style skill induction):
on a successful large-model trajectory, summarise the solving procedure into
`procedure` and store the snippet. At inference, the small model is prompted
with the retrieved `procedure` for the matched signature.

This needs: (a) a distillation call per new successful cluster, (b) a retrieval
step in the router/serve path, (c) an eval that isolates the procedure's
contribution. Estimated ~400 LoC + a held-out "does the procedure help the
small model" ablation. **Not implemented in this change** — flagged here so the
next experiment cycle can scope it.

## Files touched

- `experiments/scaling/collect_traces.py` — closed-loop routing
- `scaling/run_full_pipeline.sh` — pass `--router` / `--skillbook` at cycle ≥ 1

---

# Scaling pipeline audit — round 2 (2026-05-21, LLM track)

Second review finding: *"New traces feed SkillBook and router training and
generate `training_data.jsonl`, but they never enter LLM SFT. And
`experiments/extract_training_data.py` is HumanEval-era code — it loads
HumanEval.jsonl, finds small-fail/large-OK hard tasks, and re-generates code
with the large model. It is NOT a tau2-trajectory → prompt/completion
converter. So each cycle's new traces never train the model."*

## Finding (confirmed — two coupled bugs)

1. **`extract_training_data.py` is HumanEval-coupled.** It calls
   `load_humaneval_tasks()` and looks tasks up by `task_id` in
   `data/HumanEval.jsonl`, then re-queries the large model via
   `src.solve_task`. On tau2/SWE traces the lookup misses → it produces nothing
   usable. It is a HumanEval regenerator, not a trace converter.

2. **`tau2_train_wrapper.sh` ignored the per-cycle data.** Phase 3 invoked the
   wrapper with `MODE=colleague_corpus` (the default), which trains on the
   colleague's fixed `stage2_v1` corpus and **explicitly ignores
   `TRAINING_DATA`**. The `scaling_traces` mode was a stub that *silently fell
   back* to `colleague_corpus`. Net effect: every cycle trained the LLM on the
   same fixed corpus; the evolve loop never reached the LLM track, and the
   silent fallback made it invisible.

## Fix

- **New converter `experiments/scaling/traces_to_sft.py`** (replaces the
  HumanEval-coupled `extract_training_data.py` for the scaling pipeline). It
  reads the trace schema directly, keeps small-fail/large-OK rows, and emits
  `{prompt, completion}` SFT pairs sourced from the large model's completion
  already in the trace — no HumanEval lookup, no extra model calls, any bench.

- **Adapters now record completions.** `run_task_pair` writes
  `small_completion` / `large_completion`; the tau2 `_run_one` extracts the
  completion best-effort from the colleague's `run_task` result (explicit
  key list + trajectory-join fallback); the mock adapter emits synthetic
  completions so the data flow is smoke-testable.

- **Phase 3 rewired.** `run_full_pipeline.sh` Phase 3 now calls
  `traces_to_sft.py` (not `extract_training_data.py`) and defaults to
  `MODE=scaling_traces` (was `colleague_corpus`). Set
  `TAU2_TRAIN_MODE=colleague_corpus` to restore the fixed-corpus behaviour.

- **`scaling_traces` no longer silently falls back.** It now (a) requires
  non-empty `TRAINING_DATA`, (b) converts `{prompt,completion}` → the
  colleague stage-2 row format → runs the colleague's
  `convert_to_prompt_completion.py` to TRL format, and (c) **fails loudly** with
  the staged data path + the exact one-line teammate hook if the colleague
  pipeline does not yet honour an external train file (`SCALING_TRAIN_FILE`).
  Failing loudly is deliberate: silently training on the wrong corpus is the
  bug we just removed.

Verified in mock: 30 traces → 12 hard tasks → 12 SFT pairs, completions
sourced from `large_completion`. The 2-phase smoke (phases 1,2,4,5 + closed
loop) still passes.

## Remaining teammate hook

The colleague's `train_pipeline.sh` must accept an external train file
(`SCALING_TRAIN_FILE` → `data_processed/stage2_v1/train.jsonl`, then regen the
build-meta the chat-template validator checks). Until that one line lands,
`scaling_traces` stages the converted TRL data and exits non-zero with
instructions rather than training on the wrong data.

## Files touched (round 2)

- `experiments/scaling/traces_to_sft.py` — NEW bench-agnostic converter
- `experiments/scaling/benches/tau2_bench/adapter.py` — record completions
- `experiments/scaling/tau2_train_wrapper.sh` — real `scaling_traces` mode, no silent fallback
- `scaling/run_full_pipeline.sh` — Phase 3 uses traces_to_sft.py + scaling_traces default
