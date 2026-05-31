#!/usr/bin/env bash
# code/training/orchestration/train_pipeline.sh
# PHASE 1 ONLY — train all 10 SFT runs. Does NOT require OPENAI_API_KEY.
#
# Pairs with eval_pipeline.sh (the OpenAI-key-gated half) to give the operator
# a clean train-now-eval-later split, so training can start on the GPU node
# before the API key is provisioned.
#
# Usage:
#   BUNDLE_ROOT=/path/to/bundle bash code/training/orchestration/train_pipeline.sh
#
# After this finishes successfully, every run in plan_c_prime.yaml has
# train_outputs/<run_id>/checkpoint-final/ (or checkpoint-best/) + STATUS=done.
# To resume training-only (idempotent): re-run this script — completed runs
# (STATUS=done) are skipped automatically by train_all.sh.
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:?set BUNDLE_ROOT}"

cd "$BUNDLE_ROOT"

echo "=== Phase 1: Training (no OpenAI key required) ==="
bash code/training/orchestration/train_all.sh

echo ""
echo "================================================================"
echo "=== Training complete.                                       ==="
echo "================================================================"
echo "All trained checkpoints are under:"
echo "    $BUNDLE_ROOT/train_outputs/<run_id>/checkpoint-final/"
echo ""
echo "Next: evaluate them on real τ²-bench tasks (requires OpenAI API key)."
echo ""
echo "  1. Export your key in this shell:"
echo "       export OPENAI_API_KEY=sk-..."
echo ""
echo "  2. Run the eval-only pipeline:"
echo "       BUNDLE_ROOT=$BUNDLE_ROOT bash code/training/orchestration/eval_pipeline.sh"
echo ""
echo "  3. Results land at:"
echo "       $BUNDLE_ROOT/train_outputs/SUMMARY.csv"
echo "       $BUNDLE_ROOT/train_outputs/plots/"
echo "================================================================"
