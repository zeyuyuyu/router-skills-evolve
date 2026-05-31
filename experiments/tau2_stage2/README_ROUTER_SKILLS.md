# Tau2 Stage-2 Training And Step-Budget Eval

This directory contains the tau2 Stage-2 SFT and evaluation code used for the
router-skills-evolve handoff.

## What Is Included

- `code/training/`: SFT training code, run configs, orchestration scripts, tests.
- `code/training/eval/step_budget_harness.py`: step-budget eval harness.
- `code/training/orchestration/*step_budget*`: setup, scheduling, summary, and plotting for Method A/B eval.
- `code/adapters/`, `code/core/`, `code/pipeline/`: shared tau2 adapter and pipeline support code.
- `data_processed/stage2_v1/eval_tasks.jsonl`: eval task manifest.
- `data_processed/stage2_v1/domain_assets/`: lightweight domain assets.
- `STEP_BUDGET_EVAL.md`: server deployment and run guide.

## What Is Not Included

The following are intentionally not committed:

- model weights and checkpoints: `train_outputs/`, `base_models/`
- eval raw outputs and logs: `step_budget_outputs/`, `vllm_logs/`
- private/semi-large baseline cache: `eval_baselines/`
- virtualenv/conda/offline wheel caches
- `.env` and all API/HF secrets
- full `train.jsonl` / `val.jsonl` prompt-completion corpora

To reproduce training, stage `train.jsonl` and `val.jsonl` under
`data_processed/stage2_v1/`. To reproduce step-budget eval, run
`setup_step_budget_eval.sh` to fetch base models and baselines, then run the
step-budget eval entry point.

## Current Important Fixes

- Qwen3.5 vLLM serving is handled via the server-side vLLM compatibility patch.
- Student generation is capped via `STUDENT_MAX_TOKENS` to prevent runaway local outputs.
- Step-budget eval is per-rollout resumable through existing `raw_eval/*.json` files.
- Telecom short eval IDs are mapped to the real tau2 task IDs from baseline files, so telecom tasks resume correctly.

## Typical Commands

```bash
export BUNDLE_ROOT=/path/to/tau2_stage2
cd "$BUNDLE_ROOT"

# Training
bash code/training/orchestration/train_pipeline.sh

# Full step-budget eval
bash code/training/orchestration/run_step_budget_eval.sh --seed-policy primary --method both

# One target only
cd "$BUNDLE_ROOT/code"
python -m training.eval.step_budget_harness \
  --target 05_qwen3_5_4b_273 \
  --checkpoint "$BUNDLE_ROOT/train_outputs/05_qwen3_5_4b_273/checkpoint-best" \
  --bundle-root "$BUNDLE_ROOT" \
  --baselines-root "$BUNDLE_ROOT/eval_baselines" \
  --eval-tasks "$BUNDLE_ROOT/data_processed/stage2_v1/eval_tasks.jsonl" \
  --output-dir "$BUNDLE_ROOT/step_budget_outputs/05_qwen3_5_4b_273" \
  --port 8100 --gpu 0 --tp 8 \
  --seed-policy primary --method both
```
