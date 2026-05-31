#!/usr/bin/env bash
# code/training/orchestration/eval_pipeline.sh
# PHASE 2 + 3 ONLY — eval every trained checkpoint, then aggregate + plot.
# Requires OPENAI_API_KEY (tau2 NL-judge + user simulator).
#
# Pairs with train_pipeline.sh. Expects every run in plan_c_prime.yaml to
# already have train_outputs/<run_id>/checkpoint-final/ (or checkpoint-best/)
# from a prior successful train_pipeline.sh.
#
# Usage:
#   export OPENAI_API_KEY=sk-...
#   BUNDLE_ROOT=/path/to/bundle bash code/training/orchestration/eval_pipeline.sh
#
# Idempotent: skips runs whose eval_results.json already exists.
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:?set BUNDLE_ROOT}"
VENV="${VENV:-${CONDA_PREFIX:+$CONDA_PREFIX/bin}}"
VENV="${VENV:-$BUNDLE_ROOT/code/.venv/bin}"

# Preflight 1: API key. Catch this BEFORE we touch any GPU, kick off vLLM,
# or burn judge tokens.
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set — required for tau2's NL-judge + user simulator." >&2
    echo "" >&2
    echo "Export it first:" >&2
    echo "    export OPENAI_API_KEY=sk-..." >&2
    echo "    BUNDLE_ROOT=$BUNDLE_ROOT bash code/training/orchestration/eval_pipeline.sh" >&2
    exit 1
fi

cd "$BUNDLE_ROOT"

# Preflight 2: sanity-check that training actually finished. Without this,
# eval_all.sh would silently skip every run (no checkpoints → "SKIP (no
# checkpoint)" per run) and the operator would only notice 20 min later
# when SUMMARY.csv is empty. The check is plan-aware: it only counts
# runs that don't carry `skip_eval: true` in plan_c_prime.yaml (runs 09/10
# are LR-ablations and never eval), so finishing only those would
# falsely pass a count-only check.
PLAN="${PLAN:-code/training/configs/plan_c_prime.yaml}"
N_DONE_ELIGIBLE=$(
    "$VENV/python" - "$PLAN" <<'PYEOF'
import sys, yaml
from pathlib import Path
plan_path = sys.argv[1]
plan = yaml.safe_load(open(plan_path))
n = 0
for r in plan["runs"]:
    if r.get("skip_eval", False):
        continue
    status_file = Path("train_outputs") / r["id"] / "STATUS"
    if status_file.exists() and status_file.read_text().strip() == "done":
        n += 1
print(n)
PYEOF
)
if [[ "$N_DONE_ELIGIBLE" -eq 0 ]]; then
    echo "ERROR: No eval-eligible run has STATUS=done." >&2
    echo "       (skip_eval runs alone don't count — they're LR-ablations.)" >&2
    echo "       Run train_pipeline.sh first, or wait for it to finish." >&2
    exit 1
fi
echo "Found $N_DONE_ELIGIBLE eval-eligible fully-trained run(s) ready to evaluate."

echo ""
echo "=== Phase 2: Eval ==="
bash code/training/orchestration/eval_all.sh

echo ""
echo "=== Phase 3: Aggregate ==="
cd "$BUNDLE_ROOT/code"
"$VENV/python" -m training.orchestration.summarize --root "$BUNDLE_ROOT/train_outputs"
"$VENV/python" -m training.orchestration.plotting --root "$BUNDLE_ROOT/train_outputs"
cd "$BUNDLE_ROOT"

echo ""
echo "================================================================"
echo "=== Eval pipeline complete.                                  ==="
echo "================================================================"
echo "Summary: $BUNDLE_ROOT/train_outputs/SUMMARY.csv"
echo "Plots:   $BUNDLE_ROOT/train_outputs/plots/"
echo "================================================================"
