#!/usr/bin/env bash
# code/training/orchestration/eval_all.sh
# Phase 2 — eval every main run in parallel on 8 GPUs.
# Idempotent: skips runs whose eval_results.json exists.
#
# Usage:
#   BUNDLE_ROOT=/path/to/bundle bash code/training/orchestration/eval_all.sh
#
# Env vars:
#   BUNDLE_ROOT       (required)
#   PLAN              (default: code/training/configs/plan_c_prime.yaml)
#   VENV              (default: active conda env, falling back to local .venv)
#   ONLY_RUN          (optional) — eval only this run_id
#
# Held-out tasks: train_all.sh now produces both heldout_task_ids.json (the
# id list) and heldout_tasks.jsonl (the descriptor list this harness reads).
# The expansion is in training.data.heldout_split.expand_heldout_ids_to_descriptors.
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:?set BUNDLE_ROOT}"
PLAN="${PLAN:-code/training/configs/plan_c_prime.yaml}"
VENV="${VENV:-${CONDA_PREFIX:+$CONDA_PREFIX/bin}}"
VENV="${VENV:-$BUNDLE_ROOT/code/.venv/bin}"
EVAL_TASKS="$BUNDLE_ROOT/data_processed/stage2_v1/eval_tasks.jsonl"
HELDOUT_TASKS="$BUNDLE_ROOT/train_outputs/_data_cache/heldout_tasks.jsonl"

cd "$BUNDLE_ROOT"

# Preflight: tau2's NL-judge evaluator uses an OpenAI judge by default. Eval
# requires an OpenAI key (or the operator can edit tau2's NL-judge config).
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo "OPENAI_API_KEY not set (needed for tau2 NL-judge); set FORCE=1 to override"; [[ "${FORCE:-0}" = "1" ]] || exit 1; }

# Build a list of (run_id, gpu_idx, port) — skip runs marked skip_eval.
# Refuse to launch when >8 eval-eligible runs map to the same port modulo 8:
# port = 8000 + (i % 8) collides as soon as i==8 has not been skipped. The
# current plan_c_prime has 10 runs, 2 with skip_eval=true (09/10), so the
# 8 remaining all get distinct ports. Future expansion (e.g., grid search)
# must add port-allocation logic before crossing 8 eligible runs.
RUNS=$(
    "$VENV/python" -c "
import yaml, sys
plan = yaml.safe_load(open('$PLAN'))
eligible = [r for r in plan['runs'] if not r.get('skip_eval', False)]
if len(eligible) > 8:
    sys.stderr.write(
        f'FATAL: {len(eligible)} eval-eligible runs exceeds 8-GPU port pool '
        '(port = 8000 + (i %% 8) collides). Add port-allocation logic before '
        'expanding the plan.\n'
    )
    sys.exit(2)
for i, r in enumerate(plan['runs']):
    if r.get('skip_eval', False):
        continue
    print(r['id'], i % 8, 8000 + (i % 8))
"
)

echo "=== Phase 2 — parallel eval on 8 GPUs ==="

# Two-phase scheduling. The 35B-A3B run needs TP=8 (all 8 GPUs in one
# vLLM instance), so it cannot run in parallel with any TP=1 run that
# would claim a subset of the same GPUs. Iter-6 audit caught this:
# the old loop launched every run with `&` and the 35B's TP=8
# CUDA_VISIBLE_DEVICES=0..7 collided with every other run's GPU claim.
# Split into:
#   1) parallel phase — all TP=1 runs concurrently across GPUs 0..N-1
#   2) sequential phase — TP=8 runs (just 35B) after wait, claiming all GPUs
PIDS=()
SEQUENTIAL_RUNS=()
while IFS=' ' read -r RID GPU PORT; do
    [[ -z "$RID" ]] && continue
    if [[ -n "${ONLY_RUN:-}" && "$RID" != "$ONLY_RUN" ]]; then
        continue
    fi
    OUT_DIR="$BUNDLE_ROOT/train_outputs/$RID"
    CKPT="$OUT_DIR/checkpoint-best"
    [[ -d "$CKPT" ]] || CKPT="$OUT_DIR/checkpoint-final"
    if [[ ! -d "$CKPT" ]]; then
        echo "  SKIP (no checkpoint): $RID"
        continue
    fi
    # Don't eval against a 1-step smoke checkpoint. Phase-N had train.py
    # write STATUS=smoke_done under --smoke-test so train_all.sh wouldn't
    # skip the real run, but eval_all.sh had no equivalent gate — a smoke
    # run that produced a checkpoint-final would silently be evaluated as
    # if it were a fully-trained model. Require STATUS=done (or absent for
    # external runs that don't write STATUS).
    STATUS_FILE="$OUT_DIR/STATUS"
    if [[ -f "$STATUS_FILE" ]] && grep -q "^smoke_done$" "$STATUS_FILE"; then
        echo "  SKIP (smoke-trained checkpoint only — re-run full train before eval): $RID"
        continue
    fi
    if [[ -f "$OUT_DIR/eval_results.json" ]]; then
        echo "  SKIP (already evaluated): $RID"
        continue
    fi
    # 35B MoE doesn't fit on a single H200 alongside vLLM's KV reservation;
    # serve it across all 8 GPUs (tensor-parallel). Defer to sequential
    # phase so it doesn't race with parallel TP=1 evals for GPUs 0..6.
    case "$RID" in
        08_qwen3_6_35b_a3b_273)
            SEQUENTIAL_RUNS+=("$RID:$OUT_DIR:$CKPT")
            echo "  DEFER (TP=8, runs after parallel phase): $RID"
            continue
            ;;
    esac

    echo "  EVAL (parallel, TP=1): $RID on GPU $GPU port $PORT"
    cd "$BUNDLE_ROOT/code"
    TP_SIZE=1 "$VENV/python" -m training.eval.harness \
        --checkpoint "$CKPT" \
        --output-dir "$OUT_DIR" \
        --bundle-root "$BUNDLE_ROOT" \
        --eval-tasks "$EVAL_TASKS" \
        --heldout-tasks "$HELDOUT_TASKS" \
        --port "$PORT" --gpu "$GPU" --seeds 300 \
        > "$OUT_DIR/eval_stdout.log" 2>&1 &
    PIDS+=("$!")
    cd "$BUNDLE_ROOT"
done <<< "$RUNS"

# Wait for all parallel evals. Mirror train_all.sh's sweep summary semantics:
# count failures, refuse to proceed to winner-selection (and downstream
# SUMMARY.csv) on any failure so partial eval data isn't silently treated
# as complete. To override (e.g., to accept partial eval), the operator
# can rerun winner-selection and summarize.py manually.
EVAL_FAILED=0
EVAL_OK=0
for PID in "${PIDS[@]}"; do
    if wait "$PID"; then
        EVAL_OK=$((EVAL_OK + 1))
    else
        EVAL_FAILED=$((EVAL_FAILED + 1))
        echo "  ✗ eval PID $PID exited non-zero" >&2
    fi
done

echo ""
echo "=== Parallel eval summary ==="
echo "  ✓ ok:     $EVAL_OK"
echo "  ✗ failed: $EVAL_FAILED"
if [[ "$EVAL_FAILED" -gt 0 ]]; then
    echo "" >&2
    echo "WARNING: $EVAL_FAILED parallel evals failed; halting before winner-selection" >&2
    echo "         so SUMMARY.csv isn't silently produced from partial data." >&2
    echo "         Investigate train_outputs/<rid>/eval_stdout.log for the failures." >&2
    exit 1
fi

# Sequential phase — TP=8 runs (35B-A3B). Cannot run in parallel with
# the TP=1 runs above because TP=8 claims GPUs 0..7. Run AFTER `wait`
# so GPUs are free.
for ENTRY in "${SEQUENTIAL_RUNS[@]}"; do
    IFS=':' read -r RID OUT_DIR CKPT <<< "$ENTRY"
    echo ""
    echo "=== EVAL (sequential, TP=8): $RID ==="
    cd "$BUNDLE_ROOT/code"
    # TP_SIZE=8 makes vllm_serve.sh expand CUDA_VISIBLE_DEVICES to 0..7;
    # --port/--gpu are nominal for log labelling (vllm_serve.sh overrides
    # the positional $GPU when TP>1 — see vllm_serve.sh:24-30).
    if TP_SIZE=8 "$VENV/python" -m training.eval.harness \
            --checkpoint "$CKPT" \
            --output-dir "$OUT_DIR" \
            --bundle-root "$BUNDLE_ROOT" \
            --eval-tasks "$EVAL_TASKS" \
            --heldout-tasks "$HELDOUT_TASKS" \
            --port 8000 --gpu 0 --seeds 300 \
            > "$OUT_DIR/eval_stdout.log" 2>&1; then
        EVAL_OK=$((EVAL_OK + 1))
        echo "  ✓ $RID ok"
    else
        EVAL_FAILED=$((EVAL_FAILED + 1))
        echo "  ✗ $RID FAILED — see $OUT_DIR/eval_stdout.log" >&2
    fi
    cd "$BUNDLE_ROOT"
done

if [[ "$EVAL_FAILED" -gt 0 ]]; then
    echo "" >&2
    echo "WARNING: $EVAL_FAILED total evals failed (parallel + sequential); halting." >&2
    exit 1
fi

echo ""
echo "=== Identifying winning config for re-eval ==="
WINNER=$(
    "$VENV/python" - <<'PYEOF'
import json, glob, os
candidates = []
root = os.environ["BUNDLE_ROOT"]
for p in sorted(glob.glob(f"{root}/train_outputs/*/eval_results.json")):
    rid = os.path.basename(os.path.dirname(p))
    d = json.load(open(p))
    cost = d.get("total_task_cost_usd_mean", 1e9)
    pr = d.get("pass_rate", 0.0)
    if cost < 0.0212:
        candidates.append((pr, -cost, rid))
candidates.sort(reverse=True)
print(candidates[0][2] if candidates else "")
PYEOF
)

if [[ -n "$WINNER" ]]; then
    echo "  Winner: $WINNER"
    OUT_DIR="$BUNDLE_ROOT/train_outputs/$WINNER"
    CKPT="$OUT_DIR/checkpoint-best"
    [[ -d "$CKPT" ]] || CKPT="$OUT_DIR/checkpoint-final"
    if [[ ! -f "$OUT_DIR/eval_results_seed301.json" ]]; then
        echo "  Re-evaluating winner with seed 301..."
        # Move the existing seed-300 outputs aside before invoking the
        # harness — the harness short-circuits if eval_results.json already
        # exists, so without renaming it would silently no-op AND the final
        # `mv` below would mislabel the seed-300 results as seed-301.
        [[ -f "$OUT_DIR/eval_results.json" && ! -f "$OUT_DIR/eval_results_seed300.json" ]] \
            && mv "$OUT_DIR/eval_results.json" "$OUT_DIR/eval_results_seed300.json"
        [[ -f "$OUT_DIR/eval_rollouts.jsonl" && ! -f "$OUT_DIR/eval_rollouts_seed300.jsonl" ]] \
            && mv "$OUT_DIR/eval_rollouts.jsonl" "$OUT_DIR/eval_rollouts_seed300.jsonl"
        cd "$BUNDLE_ROOT/code"
        # 35B-A3B doesn't fit on a single H200 alongside vLLM KV reservation
        # — same constraint as the parallel phase. Match the per-run TP
        # decision when the winner happens to be the 35B run; iter-7 audit
        # caught this re-eval would have OOM'd otherwise.
        case "$WINNER" in
            08_qwen3_6_35b_a3b_273) REEVAL_TP_SIZE=8 ;;
            *) REEVAL_TP_SIZE=1 ;;
        esac
        TP_SIZE="$REEVAL_TP_SIZE" "$VENV/python" -m training.eval.harness \
            --checkpoint "$CKPT" \
            --output-dir "$OUT_DIR" \
            --bundle-root "$BUNDLE_ROOT" \
            --eval-tasks "$EVAL_TASKS" \
            --port 8000 --gpu 0 --seeds 301
        mv "$OUT_DIR/eval_results.json" "$OUT_DIR/eval_results_seed301.json"
        [[ -f "$OUT_DIR/eval_rollouts.jsonl" ]] \
            && mv "$OUT_DIR/eval_rollouts.jsonl" "$OUT_DIR/eval_rollouts_seed301.jsonl"
        cd "$BUNDLE_ROOT"
    fi
fi

echo ""
echo "=== Phase 2 complete. Run summarize.py next. ==="
