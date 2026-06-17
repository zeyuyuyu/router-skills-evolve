#!/usr/bin/env bash
# code/training/orchestration/run_step_budget_eval.sh
#
# Single entry point for the step-budget eval pipeline. Calls:
#   1. setup_step_budget_eval.sh  — fetch baselines + base models (idempotent)
#   2. eval_pipeline_step_budget.sh — eval all 12 targets + summarize + plot
#
# Assumes the conda env `tau2-stage2` is active (the same env training used).
# The env already has every Python dependency the eval needs — no new pip
# installs are required.
#
# Usage:
#   conda activate tau2-stage2
#   cd $BUNDLE_ROOT     # path to the unpacked stage2_ship/ bundle
#   bash code/training/orchestration/run_step_budget_eval.sh \
#        [--plan full|<group>] \
#        [--seed-policy primary|secondary|both] \
#        [--only-target <name>] \
#        [--method both|A|B]
#
# CLI args are passed through to eval_pipeline_step_budget.sh.
#
# Wallclock: ~6-9 hours for the full 12-target sweep on 8× H200 (Phase 1,
# primary seed). The eval is CommonStack-bandwidth-bound, not GPU-bound.
#
# Resume: if anything dies mid-run (credit exhausted → exit 42, vLLM OOM,
# kill -9), just rerun this script. Per-rollout atomic writes mean finished
# rollouts are skipped on the next pass.
set -euo pipefail

# --- Locate bundle root --------------------------------------------------- #
# If BUNDLE_ROOT isn't already set, infer from this script's location:
#   code/training/orchestration/run_step_budget_eval.sh
#   <- BUNDLE_ROOT is three levels up
if [[ -z "${BUNDLE_ROOT:-}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    BUNDLE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi
export BUNDLE_ROOT
cd "$BUNDLE_ROOT"

# --- Sanity: conda env active? ------------------------------------------- #
# `tau2-stage2` is the conda env name used by setup_env_server.sh (training).
# We don't auto-activate here — that depends on the operator's shell config.
# A clear error is more useful than a guessing-game.
if [[ "${CONDA_DEFAULT_ENV:-}" != "tau2-stage2" ]]; then
    echo "ERROR: conda env 'tau2-stage2' is not active." >&2
    echo "       Run:  conda activate tau2-stage2  before invoking this script." >&2
    exit 1
fi

# --- Source .env (CommonStack + HF tokens) ------------------------------- #
if [[ ! -f "$BUNDLE_ROOT/.env" ]]; then
    echo "ERROR: $BUNDLE_ROOT/.env not found." >&2
    echo "       The eval needs OPENAI_API_KEY / OPENAI_API_BASE (CommonStack)" >&2
    echo "       and HF_TOKEN (to fetch baselines + base models from Hugging Face)." >&2
    exit 2
fi
set -a
# shellcheck disable=SC1091
source "$BUNDLE_ROOT/.env"
set +a

# Validate required env vars are now set.
: "${OPENAI_API_KEY:?OPENAI_API_KEY missing from .env}"
: "${HF_TOKEN:?HF_TOKEN missing from .env}"
: "${OPENAI_API_BASE:=${OPENAI_BASE_URL:-https://api.commonstack.ai/v1}}"
: "${OPENAI_BASE_URL:=$OPENAI_API_BASE}"
export OPENAI_API_BASE OPENAI_BASE_URL

# --- Step 1: setup (idempotent — skips already-complete snapshots) ------- #
echo ""
echo "============================================================"
echo "  STEP 1/2: setup (fetch baselines + base models from HF)"
echo "============================================================"
bash "$BUNDLE_ROOT/code/training/orchestration/setup_step_budget_eval.sh"

# --- Step 2: eval pipeline (eval → summarize → plot) --------------------- #
# Pass through any CLI args (--plan / --seed-policy / --only-target / --method).
echo ""
echo "============================================================"
echo "  STEP 2/2: eval pipeline (12 targets, ~6-9 h on 8× H200)"
echo "============================================================"
bash "$BUNDLE_ROOT/code/training/orchestration/eval_pipeline_step_budget.sh" "$@"
PIPELINE_RC=$?

echo ""
echo "============================================================"
if [[ $PIPELINE_RC -eq 42 ]]; then
    echo "  PAUSED: CommonStack credits exhausted."
    echo "  Top up your CS account, then rerun this exact command."
    echo "  Finished rollouts will be skipped (per-rollout atomic writes)."
    echo "============================================================"
    exit 42
elif [[ $PIPELINE_RC -ne 0 ]]; then
    echo "  PIPELINE EXITED $PIPELINE_RC — some targets failed."
    echo "  Inspect: \$BUNDLE_ROOT/step_budget_outputs/<target>/harness_stdout.*.log"
    echo "  Rerun this command to retry — per-rollout resume is automatic."
    echo "============================================================"
    exit "$PIPELINE_RC"
fi
echo "  DONE — full step-budget eval complete."
echo "  Summary: $BUNDLE_ROOT/step_budget_outputs/SUMMARY_step_budget.csv"
echo "  Plots:   $BUNDLE_ROOT/step_budget_outputs/plots/"
echo "============================================================"
