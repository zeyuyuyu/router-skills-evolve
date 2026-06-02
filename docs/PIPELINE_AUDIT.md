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
