#!/usr/bin/env bash
# code/training/orchestration/eval_pipeline_step_budget.sh
#
# TOP-LEVEL DRIVER for the step-budget eval pipeline.
# Stages: preflight → eval_all_step_budget.sh → summarize → plot.
#
# Resume-safe at the per-rollout level. If a previous run was killed (credit
# exhausted, OOM, kill -9, etc.), re-running this script picks up where it
# left off.
#
# Usage:
#   conda activate tau2-stage2
#   export BUNDLE_ROOT=/path/to/bundle
#   bash code/training/orchestration/eval_pipeline_step_budget.sh \
#       [--plan full|<group>] [--seed-policy primary|secondary|both] \
#       [--only-target <name>] [--method both|A|B]
#
#   Examples:
#     # Full pipeline, primary seed (default), all 12 targets:
#     bash code/training/orchestration/eval_pipeline_step_budget.sh
#
#     # Re-run after credit top-up:
#     bash code/training/orchestration/eval_pipeline_step_budget.sh
#
#     # Single group, single seed:
#     bash code/training/orchestration/eval_pipeline_step_budget.sh --plan 9b
#
#     # Phase 2 (second seed) after Phase 1 finishes:
#     bash code/training/orchestration/eval_pipeline_step_budget.sh --seed-policy secondary
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:?set BUNDLE_ROOT}"
cd "$BUNDLE_ROOT"

if [[ -f "$BUNDLE_ROOT/.env" ]]; then
    set -a; source "$BUNDLE_ROOT/.env"; set +a
fi

# --- arg parsing ---------------------------------------------------------- #
PLAN="full"
SEED_POLICY="primary"
ONLY_TARGET=""
METHOD="both"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --plan)         PLAN="$2"; shift 2 ;;
        --seed-policy)  SEED_POLICY="$2"; shift 2 ;;
        --only-target)  ONLY_TARGET="$2"; shift 2 ;;
        --method)       METHOD="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

VENV="${VENV:-${CONDA_PREFIX:+$CONDA_PREFIX/bin}}"
VENV="${VENV:-$BUNDLE_ROOT/code/.venv/bin}"

# --- preflight ------------------------------------------------------------ #
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set. Either source $BUNDLE_ROOT/.env or export it." >&2
    exit 2
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN is not set. Required only if base_models/ or eval_baselines/ are missing." >&2
fi
: "${OPENAI_API_BASE:=${OPENAI_BASE_URL:-https://api.commonstack.ai/v1}}"
: "${OPENAI_BASE_URL:=$OPENAI_API_BASE}"
export OPENAI_API_KEY OPENAI_API_BASE OPENAI_BASE_URL

if [[ ! -d "$BUNDLE_ROOT/eval_baselines" ]]; then
    echo "ERROR: $BUNDLE_ROOT/eval_baselines/ missing. Run setup_step_budget_eval.sh first." >&2
    exit 3
fi
if [[ ! -d "$BUNDLE_ROOT/base_models" ]]; then
    echo "ERROR: $BUNDLE_ROOT/base_models/ missing. Run setup_step_budget_eval.sh first." >&2
    exit 3
fi

# --- seed-policy expansion ------------------------------------------------ #
# "both" means: run primary, then if it succeeded, run secondary.
SEED_POLICIES=()
case "$SEED_POLICY" in
    primary)   SEED_POLICIES=(primary) ;;
    secondary) SEED_POLICIES=(secondary) ;;
    both)      SEED_POLICIES=(primary secondary) ;;
    *) echo "ERROR: --seed-policy must be primary|secondary|both" >&2; exit 2 ;;
esac

# --- group selection ------------------------------------------------------ #
GROUP_ARG="$PLAN"
if [[ "$PLAN" == "full" ]]; then GROUP_ARG="all"; fi

# --- Run Phase 2 (eval) --------------------------------------------------- #
ANY_CREDIT=0
ANY_FAIL=0
for sp in "${SEED_POLICIES[@]}"; do
    echo ""
    echo "#######################################################################"
    echo "###  PHASE 2 — eval (seed_policy=$sp, group=$GROUP_ARG, method=$METHOD)"
    echo "#######################################################################"

    rc=0
    if [[ -n "$ONLY_TARGET" ]]; then
        SEED_POLICY="$sp" GROUP="$GROUP_ARG" ONLY_TARGET="$ONLY_TARGET" METHOD="$METHOD" \
            bash "$BUNDLE_ROOT/code/training/orchestration/eval_all_step_budget.sh" || rc=$?
    else
        SEED_POLICY="$sp" GROUP="$GROUP_ARG" METHOD="$METHOD" \
            bash "$BUNDLE_ROOT/code/training/orchestration/eval_all_step_budget.sh" || rc=$?
    fi

    if [[ $rc -eq 42 ]]; then
        echo ""
        echo "###  CREDITS EXHAUSTED. Top up and re-run this exact command." >&2
        ANY_CREDIT=1
        break
    elif [[ $rc -ne 0 ]]; then
        echo "WARNING: eval_all_step_budget.sh exited $rc — some targets failed." >&2
        ANY_FAIL=1
        # Still proceed to summarize whatever we have, but mark.
    fi
done

# --- Phase 3: summarize + plot (always runs over whatever exists) --------- #
echo ""
echo "#######################################################################"
echo "###  PHASE 3 — summarize + plot"
echo "#######################################################################"
cd "$BUNDLE_ROOT/code"
"$VENV/python" -m training.orchestration.summarize_step_budget \
    --bundle-root "$BUNDLE_ROOT" \
    || echo "WARNING: summarize_step_budget failed (probably incomplete data)" >&2
"$VENV/python" -m training.orchestration.plotting_step_budget \
    --bundle-root "$BUNDLE_ROOT" \
    || echo "WARNING: plotting_step_budget failed" >&2
cd "$BUNDLE_ROOT"

echo ""
echo "======================================================================="
if [[ "$ANY_CREDIT" -eq 1 ]]; then
    echo "===  PIPELINE PAUSED (credit exhausted). Re-run to resume.            ==="
    exit 42
elif [[ "$ANY_FAIL" -eq 1 ]]; then
    echo "===  PIPELINE COMPLETED with some failed targets — see logs.          ==="
    echo "===  Re-run this command to retry; per-rollout resume is automatic.  ==="
    exit 1
else
    echo "===  PIPELINE COMPLETE                                                ==="
    echo "===  Summary: $BUNDLE_ROOT/step_budget_outputs/SUMMARY_step_budget.csv"
    echo "===  Plots:   $BUNDLE_ROOT/step_budget_outputs/plots/                ==="
    exit 0
fi
