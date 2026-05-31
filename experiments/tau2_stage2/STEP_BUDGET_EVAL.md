# Step-budget eval — server deployment guide

This document describes the step-budget eval add-on layered on top of the
training+eval bundle that's already running on the 8× H200 / CUDA 13.0
server. It lists every file that is new or changed relative to the previous
bundle you deployed, and gives the exact commands to install missing data,
activate the env, and run the full eval end-to-end.

---

## 1. What changed in this bundle

Relative to the bundle previously deployed on the server (the portable
bundle stamped 2026-05-13 — i.e., the one your trained checkpoints in
`train_outputs/` were produced under), inside `stage2_ship/`:

### Modified files (2)

| File | Why it changed |
|---|---|
| `code/adapters/tau2_bench/adapter.py` | **Critical substitution-condition fix.** Method-B at any `step_idx > 0` was silently a no-op under tau2 1.0.0's trajectory semantics — the swap loop's count never matched the target step, so every "student-substituted" rollout was actually a baseline replay. The patched version invokes the student at the correct step. |
| `code/training/eval/vllm_serve.sh` | Three changes: (a) `--max-model-len` default 32768 → **131072** (4× SFT context, well inside Qwen3.5/3.6's native 256K) so the longest Method-B rollouts don't get clipped at the wire. (b) `CUDA_VISIBLE_DEVICES` honors the `$GPU` positional arg when `TP_SIZE > 1` (used to hardcode `0..TP-1`); required for the new TP=2 parallel layout where 4 vLLM instances claim disjoint GPU pairs. (c) Range check fails fast if `GPU + TP_SIZE > NUM_GPUS` (default 8). |

### New files (11)

| File | Purpose |
|---|---|
| `.env` | Secrets at bundle root: `OPENAI_API_KEY`, `OPENAI_API_BASE`, `OPENAI_BASE_URL` (CommonStack) and `HF_TOKEN`. |
| `STEP_BUDGET_EVAL.md` | This file. |
| `code/training/eval/_judge_patch.py` | Monkey-patches tau2's NL-assertion judge to route through CommonStack (`openai/openai/gpt-5.2`). tau2 hardcodes `gpt-4.1-2025-04-14`, which CommonStack doesn't host. |
| `code/training/eval/fetch_baselines.py` | Pulls the 63 `phase0_baseline.json` files from the private HF dataset into `eval_baselines/`. |
| `code/training/eval/fetch_base_models.py` | Snapshots the 4 untrained Qwen base models at pinned SHAs into `base_models/`. ~100 GB. |
| `code/training/eval/step_budget_harness.py` | Main eval driver. Method B (per-step substitution) + Method A (E2E supplement). Per-rollout atomic writes for resume. |
| `code/training/orchestration/setup_step_budget_eval.sh` | One-shot prep: preflight CommonStack reachability + fetch base models + fetch baselines. Idempotent. |
| `code/training/orchestration/eval_all_step_budget.sh` | Group-aware scheduler (4B→9B→2B→35B). TP=2 wave-batching for small models (up to 4 vLLM instances in parallel per wave); TP=8 sequential for the 35B MoE. |
| `code/training/orchestration/eval_pipeline_step_budget.sh` | Top-level eval driver (eval all targets → summarize → plot). |
| `code/training/orchestration/run_step_budget_eval.sh` | **Single end-to-end entry point**: verifies env, sources `.env`, calls `setup_step_budget_eval.sh`, then calls `eval_pipeline_step_budget.sh`. |
| `code/training/orchestration/summarize_step_budget.py` | Aggregates per-target rollouts into `SUMMARY_step_budget.csv` / `.json`. |
| `code/training/orchestration/plotting_step_budget.py` | Scaling-curve PNGs: `rep_rate`, `closure_ratio`, `pass_rate_e2e` vs `n_train_rows`. |

### Nothing else in the bundle was touched

The original training scripts (`train.py`, `train_all.sh`, `train_pipeline.sh`),
the original eval flow (`eval_all.sh`, `eval_pipeline.sh`, `harness.py`,
`summarize.py`, `plotting.py`), the data converter, the test suite, the
pipeline phases, etc., are bit-for-bit unchanged. The two patched files
above are also read by the original eval flow, but the changes are
backward-compatible with that flow's TP=1/TP=8 invocation patterns.

---

## 2. Server-side prerequisites (already satisfied)

The conda env `tau2-stage2` set up by `code/training/orchestration/setup_env_server.sh`
(when you ran training) already contains every Python dependency the eval
needs: `torch`, `vllm`, `litellm`, `openai`, `huggingface_hub`, `hf_transfer`,
`transformers`, `tau2-bench`, `matplotlib`, `pandas`. **No new pip installs
are required.**

The only things that need downloading are the HF assets — `setup_step_budget_eval.sh`
handles that. See §3.

---

## 3. End-to-end run (one command)

```bash
# 1. Activate the conda env you used for training
conda activate tau2-stage2

# 2. Position at the bundle root (where this MD lives)
export BUNDLE_ROOT=/path/to/stage2_ship      # adjust to your actual path
cd $BUNDLE_ROOT

# 3. Run the full eval — single entry point
bash code/training/orchestration/run_step_budget_eval.sh
```

That's it. The script does:

1. Verifies the conda env is `tau2-stage2`, errors clearly if not.
2. Sources `$BUNDLE_ROOT/.env` to export CommonStack + HF tokens.
3. Runs `setup_step_budget_eval.sh` to fetch the 63 baselines + 4 base
   models from HF (~100 GB, 10–30 min depending on bandwidth). Idempotent —
   skipped if already present.
4. Runs `eval_pipeline_step_budget.sh` which:
   - Evaluates all 12 targets (4 untrained base models + 8 trained
     checkpoints) at the primary seed.
   - Aggregates results into `SUMMARY_step_budget.csv` / `.json`.
   - Produces scaling-curve PNGs in `step_budget_outputs/plots/`.

Wallclock: ~6–9 hours on 8× H200 (CommonStack-bandwidth-bound, not
GPU-bound). The 35B family contributes ~2–3 hours of that, sequential.

---

## 4. Useful variants

```bash
# Both seeds back-to-back (Phase 1 + Phase 2 for noise estimation)
bash code/training/orchestration/run_step_budget_eval.sh --seed-policy both

# Only the secondary seed (after a clean primary-seed run)
bash code/training/orchestration/run_step_budget_eval.sh --seed-policy secondary

# Only one model family
bash code/training/orchestration/run_step_budget_eval.sh --plan 4b

# Only one specific checkpoint
bash code/training/orchestration/run_step_budget_eval.sh --only-target 05_qwen3_5_4b_273

# Only Method A (skip the expensive Method B substitution rollouts)
bash code/training/orchestration/run_step_budget_eval.sh --method A
```

---

## 5. Resume + recovery

The eval is fully resume-safe at the per-rollout level (atomic file writes).
If anything dies mid-run, just rerun the same command:

```bash
bash code/training/orchestration/run_step_budget_eval.sh
```

Finished rollouts are skipped on the next pass.

| Symptom | Action |
|---|---|
| Exit 42 with `STOP_REASON.json` saying "credit_exhausted" | Top up CommonStack, rerun the same command. |
| One target's `harness_stdout.*.log` shows vLLM OOM/crash | Rerun — only that target re-spins its vLLM. |
| A specific rollout failed and you want to force-retry it | Delete its JSON under `step_budget_outputs/<target>/raw_eval/*.json` and rerun. |

---

## 6. Outputs

```
$BUNDLE_ROOT/step_budget_outputs/
├── <target>/
│   ├── raw_eval/<dom>__<task_id>__seed<seed>__methodA.json
│   ├── raw_eval/<dom>__<task_id>__seed<seed>__methodB_step<i>.json
│   ├── llm_calls/<rollout_id>.jsonl
│   ├── progress.json
│   └── harness_stdout.<seed_policy>.log
├── SUMMARY_step_budget.csv       # one row per target — the headline table
├── SUMMARY_step_budget.json
└── plots/
    ├── rep_rate_scaling.png          # HEADLINE: student_rep_rate vs n_train_rows
    ├── closure_ratio_scaling.png     # rep_rate / rep_rate_pinned
    └── pass_rate_e2e_scaling.png     # Method A (autonomy)
```

---

## 7. Checkpoint-path mapping (verified — no action required)

I verified that what training saved and what the eval looks for line up,
so no manual renaming is needed on the server side.

- **Trained checkpoints**: `train.py:602` writes to `train_outputs/<run_id>/checkpoint-final/`.
  `eval_all_step_budget.sh`'s `resolve_ckpt()` prefers `checkpoint-best` and
  falls back to `checkpoint-final` — the fallback always fires because
  `train.py` never creates a `checkpoint-best` directory. This matches the
  same fallback convention used by the existing `eval_all.sh` (the original
  eval flow already in production), so it's a known-good pattern.
- **Trained run-ids**: the 8 trained targets in `ALL_TARGETS` (rows `01_*` …
  `08_*`) exactly match the 8 eval-eligible runs in `plan_c_prime.yaml`
  (rows `09_*` and `10_*` are marked `skip_eval: true` and correctly omitted
  from the eval target list).
- **Base models**: `fetch_base_models.py:179` writes to `base_models/<target_name>/`
  with names that exactly match the four `base_*` entries in `ALL_TARGETS`
  (`base_Qwen3.5-2B`, `base_Qwen3.5-4B`, `base_Qwen3.5-9B`, `base_Qwen3.6-35B-A3B`).

---

## 8. First-run sanity checks

After the eval kicks off, glance at the first few minutes of stdout
(`step_budget_outputs/<target>/harness_stdout.<seed>.log`):

1. Scheduler announces:
   ```
   group=4b tp=2 parallelism=4 targets=5 waves=2
   ```
   If you see `tp=1`, the patched `eval_all_step_budget.sh` didn't replace
   correctly.

2. Per-target vLLM log (`<checkpoint>/vllm_serve.log`) should show:
   ```
   --tensor-parallel-size 2 --max-model-len 131072 --gpu-memory-utilization 0.92
   ```
   for small models; `--tensor-parallel-size 8` for the 35B.
   `CUDA_VISIBLE_DEVICES` should be a contiguous pair (`0,1`, `2,3`, …) for
   small models, `0,1,2,3,4,5,6,7` for the 35B.

3. The first Method-A rollout JSON
   (`step_budget_outputs/<target>/raw_eval/*__methodA.json`) should NOT have
   `student_outcome.error_type = "ContextLengthExceededError"`. If it does,
   the patched `vllm_serve.sh` didn't replace correctly — verify
   `--max-model-len 131072` (not 32768).

---

## 9. Cost expectations

Based on per-call CS pricing for `openai/gpt-5.2`, `qwen/qwen3.5-397b-a17b`,
`anthropic/claude-opus-4-7`.

| Phase | Per checkpoint | × 12 targets |
|---|---|---|
| Method B (~28 substitution rollouts each, sum of B across the chosen set) | ~$24 | ~$290 |
| Method A supplement (33 E2E rollouts) | ~$2.5 | ~$30 |
| **Phase 1 (primary seed) total** | **~$27** | **~$320** |
| Phase 2 (secondary seed; ~22 task-seed pairs) | ~$17 | ~$200 |

No cost cap is enforced. Cumulative spend per target is tracked in
`step_budget_outputs/<target>/progress.json`; per-call costs are in
`step_budget_outputs/<target>/llm_calls/<rollout_id>.jsonl`.

---

## 10. Methodology recap (one paragraph)

For each (task, seed) pair where the original strong baseline
(`qwen/qwen3.5-397b-a17b`) passed (33 of 35 unique tasks at the primary
seed; `retail/5` and `retail/8` excluded because the baseline failed both):
let `B` = number of agent steps in the passing baseline. **Method B** runs
`B` independent rollouts — one per `step_idx ∈ {0..B-1}` — where the first
`step_idx` agent messages are replayed verbatim from the baseline, the
student substitutes at `step_idx`, and the original baseline resumes for
the tail. **Method A** runs one E2E rollout where the student handles all
`B` steps from scratch with `max_steps = B`. Pass iff `reward ≥ 0.5`. The
NL-assertion judge is `openai/gpt-5.2` via CommonStack. Headline metric is
`student_rep_rate = (# passing Method-B rollouts) / B` aggregated per
target, plotted vs `n_train_rows` to give a scaling curve from base
(`n_train_rows=0`) to the most-trained checkpoint.
