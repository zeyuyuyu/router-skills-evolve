#!/usr/bin/env bash
# code/training/orchestration/run_pipeline.sh
# Top-level end-to-end pipeline: train_all → eval_all → summarize → plotting.
# Requires OPENAI_API_KEY upfront (eval phase). For a train-now-eval-later
# split (training without the API key, eval later), use the two-script flow:
#     bash code/training/orchestration/train_pipeline.sh   # no key needed
#     # ... export OPENAI_API_KEY ...
#     bash code/training/orchestration/eval_pipeline.sh    # key required
#
# Usage:
#   export OPENAI_API_KEY=sk-...
#   BUNDLE_ROOT=/path/to/bundle bash code/training/orchestration/run_pipeline.sh
#
# FAIL-FAST POLICY (Phase-O iter 2): train_all.sh and eval_all.sh both emit
# sweep summaries and exit non-zero on ANY failure. Combined with `set -e`
# below, this means a single failed train run (or single failed parallel
# eval) ABORTS the pipeline before downstream steps run. This is INTENTIONAL —
# the alternative silently produces incomplete SUMMARY.csv that downstream
# analysis treats as complete. To proceed past a partial sweep:
#   1. Investigate the failed run (check train_outputs/<rid>/train_stdout.log
#      or .../eval_stdout.log).
#   2. STATUS=failed (auto-retried on next train_all.sh run — no manual rm needed).
#      STATUS=done (already-complete; `rm train_outputs/<rid>/STATUS` to force
#      re-run only if you intentionally want to redo a successful run).
#   3. To accept partial data and skip the rest of the pipeline, invoke
#      `python -m training.orchestration.summarize --root train_outputs`
#      and `python -m training.orchestration.plotting --root train_outputs`
#      manually after fixing or accepting whichever runs you intend to ship.
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:?set BUNDLE_ROOT}"
# Default VENV to the active conda env's bin if one is active (server case);
# fall back to the local-dev venv path otherwise.
VENV="${VENV:-${CONDA_PREFIX:+$CONDA_PREFIX/bin}}"
VENV="${VENV:-$BUNDLE_ROOT/code/.venv/bin}"

cd "$BUNDLE_ROOT"

echo "=== Phase 1: Training ==="
bash code/training/orchestration/train_all.sh

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
echo "=== Pipeline complete ==="
echo "Summary: $BUNDLE_ROOT/train_outputs/SUMMARY.csv"
echo "Plots:   $BUNDLE_ROOT/train_outputs/plots/"
