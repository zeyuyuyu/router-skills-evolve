#!/usr/bin/env bash
# tau2_train_wrapper.sh — wrap colleague's tau2_stage2 train_pipeline.sh for the
# scaling pipeline. Two modes:
#
#   MODE=colleague_corpus   (default)  Train on colleague's stage2_v1 corpus.
#                                       Ignores TRAINING_DATA from Phase 3.
#                                       This is the safest first pass.
#
#   MODE=scaling_traces     (TODO)     Inject TRAINING_DATA (our extracted
#                                       slice) into stage2_v1 layout, then
#                                       train. NOT YET WIRED — see notes below.
#
# Required env vars:
#   BUNDLE_ROOT          /path/to/experiments/tau2_stage2
#   RUN_CONFIG           one of runs/<NN>_<model>_<data>.yaml (sans extension)
#   TRAIN_OUTPUT_DIR     where to land train_outputs/<run_id>/
#
# Optional:
#   TRAINING_DATA        path to scaling traces' extract (mode=scaling_traces)
#   MODE                 colleague_corpus | scaling_traces  (default: colleague_corpus)

set -euo pipefail

: "${BUNDLE_ROOT:?BUNDLE_ROOT must be set (path to experiments/tau2_stage2)}"
: "${RUN_CONFIG:?RUN_CONFIG must be set (e.g. 05_qwen3_5_4b_273)}"
: "${TRAIN_OUTPUT_DIR:?TRAIN_OUTPUT_DIR must be set}"
: "${MODE:=colleague_corpus}"

cd "$BUNDLE_ROOT"

case "$MODE" in
  colleague_corpus)
    echo "[tau2_train_wrapper] MODE=colleague_corpus  using stage2_v1 SFT corpus"
    # Pre-validate the run config exists
    if [[ ! -f "code/training/configs/runs/${RUN_CONFIG}.yaml" ]]; then
      echo "[tau2_train_wrapper] ERROR: code/training/configs/runs/${RUN_CONFIG}.yaml not found"
      echo "Available run configs:"
      ls -1 code/training/configs/runs/ | sed 's/^/  /'
      exit 2
    fi
    # Colleague's pipeline trains ALL 10 runs in plan_c_prime.yaml by default.
    # For the scaling pipeline we only want ONE. Filter via PLAN_RUN_FILTER.
    PLAN_RUN_FILTER="$RUN_CONFIG" \
      bash code/training/orchestration/train_pipeline.sh
    # Copy the result to TRAIN_OUTPUT_DIR so the rest of the scaling pipeline
    # has a stable path.
    src="$BUNDLE_ROOT/train_outputs/$RUN_CONFIG"
    mkdir -p "$TRAIN_OUTPUT_DIR"
    if [[ -d "$src/checkpoint-best" ]]; then
      ln -sfn "$src/checkpoint-best" "$TRAIN_OUTPUT_DIR/checkpoint-best"
      echo "[tau2_train_wrapper] linked $src/checkpoint-best -> $TRAIN_OUTPUT_DIR/checkpoint-best"
    else
      echo "[tau2_train_wrapper] WARN: $src/checkpoint-best missing; training may have failed"
      exit 3
    fi
    ;;

  scaling_traces)
    # TODO(teammate): wire TRAINING_DATA (jsonl with prompt + completion +
    # bench-specific tools) into colleague's data_processed/stage2_v1/ layout.
    # Specifically:
    #   1. Convert TRAINING_DATA jsonl rows to colleague's prompt-completion
    #      format (see code/training/data/convert_to_prompt_completion.py).
    #   2. Drop into data_processed/stage2_v1/train.jsonl.
    #   3. Either: (a) modify RUN_CONFIG's data.n_train_runs to match, or
    #      (b) regenerate _build_meta.json so the validator passes.
    # See docs/SCALING_TRAINING_DATA_INJECTION.md (also TODO) for the precise
    # format mapping.
    echo "[tau2_train_wrapper] MODE=scaling_traces not yet wired — see TODO in this script."
    echo "Falling back to colleague_corpus mode for this run."
    MODE=colleague_corpus exec "$0" "$@"
    ;;

  *)
    echo "[tau2_train_wrapper] Unknown MODE=$MODE (use colleague_corpus or scaling_traces)"
    exit 2
    ;;
esac
