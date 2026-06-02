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
    # Train on THIS cycle's extracted traces (closes the evolve loop on the LLM
    # track). Review 2026-05-21: the old stub silently fell back to
    # colleague_corpus, so per-cycle traces never reached SFT and the bug was
    # invisible. This mode now (a) requires non-empty TRAINING_DATA, (b) converts
    # it through the colleague's convert_to_prompt_completion.py, and (c) FAILS
    # LOUDLY if anything is missing — it never silently trains on the wrong data.
    : "${TRAINING_DATA:?MODE=scaling_traces requires TRAINING_DATA (set by phase3)}"
    if [[ ! -s "$TRAINING_DATA" ]]; then
      echo "[tau2_train_wrapper] FATAL: TRAINING_DATA=$TRAINING_DATA is missing or empty."
      echo "  This cycle produced no SFT pairs. Likely causes:"
      echo "   - no hard tasks (small-fail + large-OK) this cycle, or"
      echo "   - the bench adapter did not record 'large_completion' in traces."
      echo "  See docs/PIPELINE_AUDIT.md. Refusing to silently fall back to the"
      echo "  fixed colleague corpus (that is the bug we just fixed)."
      exit 4
    fi

    n_pairs=$(wc -l < "$TRAINING_DATA" | tr -d ' ')
    echo "[tau2_train_wrapper] MODE=scaling_traces  $n_pairs SFT pairs from $TRAINING_DATA"

    STAGE_DIR="$TRAIN_OUTPUT_DIR/scaling_sft"
    mkdir -p "$STAGE_DIR"
    STAGE2_ROWS="$STAGE_DIR/stage2_rows.jsonl"
    TRL_OUT="$STAGE_DIR/train_prompt_completion.jsonl"

    # 1. {prompt, completion} -> colleague stage-2 row format
    #    ({messages, _target_index, _p, domain}). _target_index points at the
    #    assistant turn so convert_to_prompt_completion masks the user prompt.
    "${PYTHON:-python3}" - "$TRAINING_DATA" "$STAGE2_ROWS" "${TAU2_DOMAIN:-retail}" <<'PY'
import json, sys
src, dst, domain = sys.argv[1], sys.argv[2], sys.argv[3]
n = 0
with open(src) as fh, open(dst, "w") as out:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        prompt = r.get("prompt") or r.get("instruction") or ""
        completion = r.get("completion") or r.get("output") or ""
        if not prompt or not completion:
            continue
        row = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            "_target_index": 1,
            "domain": domain,
            "_p": {"source": "scaling_traces", "task_id": r.get("task_id", "")},
        }
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
print(f"[scaling_traces] wrote {n} stage-2 rows -> {dst}", file=sys.stderr)
PY

    # 2. Run the colleague's converter to produce TRL prompt/completion + tools.
    CONVERT="$BUNDLE_ROOT/code/training/data/convert_to_prompt_completion.py"
    if [[ ! -f "$CONVERT" ]]; then
      echo "[tau2_train_wrapper] FATAL: colleague converter not found at $CONVERT"
      echo "  Merge codex/tau2-stage2-training-eval, or run with MODE=colleague_corpus."
      exit 4
    fi
    "${PYTHON:-python3}" "$CONVERT" --input "$STAGE2_ROWS" --output "$TRL_OUT" \
      2>&1 | tee "$STAGE_DIR/convert.log" || {
        echo "[tau2_train_wrapper] FATAL: convert_to_prompt_completion.py failed."
        echo "  Inspect $STAGE_DIR/convert.log. The colleague converter's --input/--output"
        echo "  flags or domain_assets path may differ; this is the teammate hook."
        exit 4
      }

    # 3. Hand off to the colleague pipeline pointed at our data.
    #    The pipeline must accept an external train file. If your run config does
    #    not yet support SCALING_TRAIN_FILE, this fails loudly (no silent corpus).
    if ! grep -rq "SCALING_TRAIN_FILE" "$BUNDLE_ROOT/code/training/orchestration/" 2>/dev/null; then
      echo "[tau2_train_wrapper] DATA READY but pipeline data-override not wired."
      echo "  Converted TRL data is at: $TRL_OUT  ($n_pairs pairs)"
      echo "  TEAMMATE HOOK (1 line): make train_pipeline.sh honour"
      echo "    SCALING_TRAIN_FILE=\$TRL_OUT  ->  data_processed/stage2_v1/train.jsonl"
      echo "  Until then this run is intentionally NOT falling back to the fixed"
      echo "  colleague corpus. Set TAU2_TRAIN_MODE=colleague_corpus to opt out."
      exit 5
    fi
    SCALING_TRAIN_FILE="$TRL_OUT" PLAN_RUN_FILTER="$RUN_CONFIG" \
      bash "$BUNDLE_ROOT/code/training/orchestration/train_pipeline.sh"
    src="$BUNDLE_ROOT/train_outputs/$RUN_CONFIG"
    if [[ -d "$src/checkpoint-best" ]]; then
      ln -sfn "$src/checkpoint-best" "$TRAIN_OUTPUT_DIR/checkpoint-best"
      echo "[tau2_train_wrapper] scaling_traces trained; linked $src/checkpoint-best"
    else
      echo "[tau2_train_wrapper] FATAL: training produced no checkpoint-best at $src"
      exit 3
    fi
    ;;

  *)
    echo "[tau2_train_wrapper] Unknown MODE=$MODE (use colleague_corpus or scaling_traces)"
    exit 2
    ;;
esac
