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

---

# Scaling pipeline audit — round 3 (2026-05-21, closing the loop properly)

Round-2 review caught that the round-1 "closed loop" was only a *policy
annotation* on the trace, not a fully-closed loop:

1. **The "oracle double-run" claim was false.** The tau2 adapter
   (`run_task_pair`) skipped the large model whenever small succeeded
   (`large_res = {success:True, cost:0, skipped:True}`); the mock set
   `large_success=False` on small-OK rows. So if the policy routed to large on
   a small-OK task, the billed `large_*` was a fake placeholder, not a real
   outcome.

2. **Phase 2 SkillBook still ingested `final_*`, not `policy_*`.** It read
   `final_model` / `final_success` (the adapter's original small-first
   decision), so the SkillBook learned from the heuristic routing, not from the
   evolved policy's actual decisions.

## Round-3 fix

- **Real oracle in closed-loop collection.** `run_task_pair` gains
  `force_both` (default False = deployment cost-control). `collect_traces.py`
  passes `force_both=closed_loop`, so in cycle ≥ 1 BOTH models always run and
  `large_success` / `large_cost` / `large_completion` are real. A new
  `large_skipped` flag marks placeholder rows when force_both is off.

- **Honest unknowns.** If a trace ever has `large_skipped=True` and the policy
  routes large, `_apply_policy` sets `policy_final_success=None` /
  `policy_total_cost=None` (decision `policy:route_large(unknown:large_skipped)`)
  rather than fabricating an outcome.

- **SkillBook ingests policy outcomes.** Phase 2 now uses
  `policy_final_model` / `policy_final_success` when present and not None,
  falling back to `final_*` only for cycle-0 / non-closed-loop traces or
  unknown rows.

Verified (2-cycle mock): cycle 1 has `large_skipped=False` on all rows (true
oracle), small-OK tasks carry a real `large_completion` and a non-null
`policy_final_success`, and Phase 2 ingests the policy outcomes.

## Honest status of the loop now

| Loop element | Closed? |
| --- | --- |
| Phase 1 routes with cycle-(k-1) router + SkillBook | ✅ |
| Phase 1 small model = cycle-(k-1) adapter | ✅ |
| Oracle both-model outcomes in closed-loop collection | ✅ (force_both) |
| SkillBook learns from policy outcomes | ✅ |
| Router trains on clean both-model labels | ✅ (oracle preserved) |
| LLM SFT consumes per-cycle traces | ✅ data path; ⏳ teammate hook (`SCALING_TRAIN_FILE`) |
| Richer procedural skills (review item 2) | ❌ design only — `src/skills.py` unchanged |

So: the evolve loop is now genuinely closed on routing, oracle outcomes, and
SkillBook learning. The only remaining gaps are the one-line colleague
`SCALING_TRAIN_FILE` hook for LLM SFT, and the (separately-scoped) richer-skills
redesign of `src/skills.py`.
