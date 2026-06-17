#!/usr/bin/env bash
# code/training/orchestration/setup_step_budget_eval.sh
#
# One-shot prep for the step-budget eval:
#   1. Source $BUNDLE_ROOT/.env (CommonStack creds + HF token).
#   2. Preflight: validate CommonStack reachable, gpt-5.2 callable.
#   3. Fetch the 63 phase0_baseline.json files (~760 MB) into eval_baselines/.
#   4. Fetch the 4 untrained Qwen base models (~100 GB) into base_models/.
#
# Re-runnable: every step is idempotent.
#
# Usage:
#   conda activate tau2-stage2
#   export BUNDLE_ROOT=/path/to/bundle
#   bash code/training/orchestration/setup_step_budget_eval.sh
#
# Env vars (loaded from $BUNDLE_ROOT/.env):
#   OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_BASE_URL  — CommonStack
#   HF_TOKEN                                          — private HF datasets
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:?set BUNDLE_ROOT to the path of the unpacked bundle}"
cd "$BUNDLE_ROOT"

if [[ -f "$BUNDLE_ROOT/.env" ]]; then
    echo "[setup_step_budget_eval] sourcing $BUNDLE_ROOT/.env"
    set -a
    # shellcheck disable=SC1091
    source "$BUNDLE_ROOT/.env"
    set +a
else
    echo "[setup_step_budget_eval] WARNING: $BUNDLE_ROOT/.env not found; relying on already-exported env vars" >&2
fi

# --- preflight: required env vars ----------------------------------------- #
require_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: $name is not set (need it from .env or shell export)" >&2
        exit 2
    fi
}
require_env OPENAI_API_KEY
require_env HF_TOKEN
: "${OPENAI_API_BASE:=${OPENAI_BASE_URL:-https://api.commonstack.ai/v1}}"
: "${OPENAI_BASE_URL:=$OPENAI_API_BASE}"
export OPENAI_API_BASE OPENAI_BASE_URL

VENV="${VENV:-${CONDA_PREFIX:+$CONDA_PREFIX/bin}}"
VENV="${VENV:-$BUNDLE_ROOT/code/.venv/bin}"
if [[ ! -x "$VENV/python" ]]; then
    echo "ERROR: no python at $VENV/python; activate the conda env first." >&2
    exit 2
fi

echo "[setup_step_budget_eval] python=$VENV/python"
echo "[setup_step_budget_eval] OPENAI_API_BASE=$OPENAI_API_BASE"

# --- Preflight: CommonStack reachable + gpt-5.2 callable ------------------ #
echo "[setup_step_budget_eval] preflight: sanity ping CommonStack..."
"$VENV/python" - <<'PYEOF'
import os, sys
try:
    from openai import OpenAI
except ImportError:
    print("openai SDK not installed in this env; "
          "tau2-bench[knowledge,voice] should have brought it in transitively. Continuing.",
          file=sys.stderr)
    sys.exit(0)
try:
    c = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_API_BASE"])
    r = c.chat.completions.create(
        model="openai/gpt-5.2",
        messages=[{"role":"user","content":"reply exactly: ok"}],
        max_completion_tokens=10,
        temperature=0,
    )
    print(f"[preflight] CommonStack OK; got: {r.choices[0].message.content!r}")
except Exception as e:
    print(f"[preflight] CommonStack ping FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(3)
PYEOF

# --- Fetch baselines from HF ---------------------------------------------- #
echo ""
echo "[setup_step_budget_eval] fetching baselines from HF (eval_baselines/)..."
cd "$BUNDLE_ROOT/code"
"$VENV/python" -m training.eval.fetch_baselines \
    --bundle-root "$BUNDLE_ROOT" \
    --hf-token "$HF_TOKEN"

# --- Fetch base models (the big one — uses hf_transfer) ------------------- #
echo ""
echo "[setup_step_budget_eval] fetching 4 base models from HF (base_models/, ~100 GB)..."
echo "[setup_step_budget_eval] this can take 10-30 minutes depending on bandwidth."
cd "$BUNDLE_ROOT/code"
"$VENV/python" -m training.eval.fetch_base_models \
    --bundle-root "$BUNDLE_ROOT" \
    --hf-token "$HF_TOKEN"

echo ""
echo "=========================================================================="
echo "[setup_step_budget_eval] DONE"
echo "  eval_baselines/   — phase0_baseline.json per (task, seed)"
echo "  base_models/      — 4 untrained Qwen snapshots at pinned SHAs"
echo ""
echo "Next: bash code/training/orchestration/eval_pipeline_step_budget.sh"
echo "=========================================================================="
