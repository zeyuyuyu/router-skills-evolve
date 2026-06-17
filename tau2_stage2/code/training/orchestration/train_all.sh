#!/usr/bin/env bash
# code/training/orchestration/train_all.sh
# Phase 1 — train all 10 runs in plan_c_prime.yaml.
# Each run is sequenced (FSDP runs use all 8 GPUs; small DDP runs share 8).
# Idempotent: skips runs whose STATUS=done.
#
# Usage:
#   BUNDLE_ROOT=/path/to/bundle bash code/training/orchestration/train_all.sh
#
# Env vars:
#   BUNDLE_ROOT       (required) — path to bundle root
#   PLAN              (default: code/training/configs/plan_c_prime.yaml)
#   VENV              (default: $BUNDLE_ROOT/code/.venv/bin)
#   ONLY_RUN          (optional) — if set, run only this run_id (for staged rollout)
#   MAX_STEPS_OVERRIDE (optional) — cap training to this many steps with FULL
#                                   data + FULL bs/accum. Passes through to
#                                   train.py --max-steps N. STATUS is written
#                                   as smoke_done so eval_all.sh skips the
#                                   partial checkpoint. Useful for the staged
#                                   rollout (Stage 2/3/4 in LOCAL-VS-CLOUD-
#                                   handoff.md) where the operator wants real
#                                   distributed config under N steps.
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:?set BUNDLE_ROOT}"
PLAN="${PLAN:-code/training/configs/plan_c_prime.yaml}"
VENV="${VENV:-${CONDA_PREFIX:+$CONDA_PREFIX/bin}}"
VENV="${VENV:-$BUNDLE_ROOT/code/.venv/bin}"
GIT_SHA=$(git -C "$BUNDLE_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")
export GIT_SHA BUNDLE_ROOT
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$BUNDLE_ROOT"

# Preflight checks (Spec §4 — operator prerequisites).
echo "=== Preflight ==="
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "H200" || { echo "Expected H200 GPUs (set FORCE=1 to override)"; [[ "${FORCE:-0}" = "1" ]] || exit 1; }
[[ -d "$BUNDLE_ROOT/data_processed/stage2_v1" ]] || { echo "Source data missing"; exit 1; }
# HF_TOKEN is OPTIONAL: Qwen3.5/3.6 are public, but unauthenticated downloads
# from HF Hub are rate-limited and slower. Warn rather than abort.
[[ -n "${HF_TOKEN:-}" ]] || echo "  warn: HF_TOKEN unset — model downloads will be rate-limited"

# Convert + validate (idempotent under a version stamp). When the converter
# behavior changes (e.g., iter-5 added a user-less-row filter), the cached
# jsonl on disk is stale; the bare exist-check would let it through. Bump
# EXPECTED_CACHE_VERSION here whenever convert_to_prompt_completion changes
# its output contract, and stale caches auto-regenerate on next run.
mkdir -p train_outputs/_data_cache
EXPECTED_CACHE_VERSION="v2-filter-user-less-rows"   # iter-5: drops msgs[:ti] with no user
CACHE_VERSION_FILE="train_outputs/_data_cache/.cache_version"
NEED_RECONVERT=true
if [[ -f train_outputs/_data_cache/train_prompt_completion.jsonl ]] \
   && [[ -f train_outputs/_data_cache/val_prompt_completion.jsonl ]] \
   && [[ -f "$CACHE_VERSION_FILE" ]] \
   && [[ "$(cat "$CACHE_VERSION_FILE")" == "$EXPECTED_CACHE_VERSION" ]]; then
    NEED_RECONVERT=false
fi
# Scaling-pipeline hook (review 2026-05-21): when SCALING_TRAIN_FILE_STAGE2 is
# set, train on the scaling pipeline's per-cycle traces (stage-2 row format)
# instead of the fixed stage2_v1 corpus. The scaling wrapper concatenates the
# per-cycle hard examples ONTO the stage2_v1 corpus, so this augments rather
# than replaces. Additive: unset => byte-for-byte original behaviour. Cache is
# keyed by version not source, so force a reconvert when the override is set.
TRAIN_SRC="$BUNDLE_ROOT/data_processed/stage2_v1/train.jsonl"
VAL_SRC="$BUNDLE_ROOT/data_processed/stage2_v1/val.jsonl"
if [[ -n "${SCALING_TRAIN_FILE_STAGE2:-}" ]]; then
    [[ -s "$SCALING_TRAIN_FILE_STAGE2" ]] || { echo "[train_all] FATAL: SCALING_TRAIN_FILE_STAGE2=$SCALING_TRAIN_FILE_STAGE2 missing/empty"; exit 1; }
    TRAIN_SRC="$SCALING_TRAIN_FILE_STAGE2"
    NEED_RECONVERT=true
    echo "=== SCALING hook: training on $TRAIN_SRC (scaling per-cycle traces) ==="
fi
if [[ -n "${SCALING_VAL_FILE_STAGE2:-}" ]]; then
    [[ -s "$SCALING_VAL_FILE_STAGE2" ]] || { echo "[train_all] FATAL: SCALING_VAL_FILE_STAGE2=$SCALING_VAL_FILE_STAGE2 missing/empty"; exit 1; }
    VAL_SRC="$SCALING_VAL_FILE_STAGE2"
fi
if [[ "$NEED_RECONVERT" == "true" ]]; then
    echo "=== Converting train + val to TRL prompt/completion format ($EXPECTED_CACHE_VERSION) ==="
    rm -f train_outputs/_data_cache/train_prompt_completion.jsonl
    rm -f train_outputs/_data_cache/val_prompt_completion.jsonl
    # Stale per-tokenizer validation reports must also be invalidated: they
    # were computed against the previous cache contents.
    rm -f train_outputs/_data_cache/validation_report_*.json
    cd "$BUNDLE_ROOT/code"
    "$VENV/python" -m training.data.convert_to_prompt_completion \
        --src "$TRAIN_SRC" \
        --dst "$BUNDLE_ROOT/train_outputs/_data_cache/train_prompt_completion.jsonl" \
        --domain-assets "$BUNDLE_ROOT/data_processed/stage2_v1/domain_assets" \
        --drop-log "$BUNDLE_ROOT/train_outputs/_data_cache/dropped_train.jsonl"
    if [[ -s "$VAL_SRC" ]]; then
        "$VENV/python" -m training.data.convert_to_prompt_completion \
            --src "$VAL_SRC" \
            --dst "$BUNDLE_ROOT/train_outputs/_data_cache/val_prompt_completion.jsonl" \
            --domain-assets "$BUNDLE_ROOT/data_processed/stage2_v1/domain_assets" \
            --drop-log "$BUNDLE_ROOT/train_outputs/_data_cache/dropped_val.jsonl"
    elif [[ -n "${SCALING_TRAIN_FILE_STAGE2:-}" && "${SCALING_ALLOW_MISSING_VAL:-0}" == "1" ]]; then
        echo "[train_all] WARN: VAL_SRC=$VAL_SRC missing; scaling run continues without val cache."
    else
        echo "[train_all] FATAL: VAL_SRC=$VAL_SRC missing/empty" >&2
        exit 1
    fi
    cd "$BUNDLE_ROOT"
    echo "$EXPECTED_CACHE_VERSION" > "$CACHE_VERSION_FILE"
fi

# Validate against the chat templates for the runs in scope. This keeps
# offline/local runs from trying to validate unrelated Hub models.
cd "$BUNDLE_ROOT/code"
TOKENIZERS=$("$VENV/python" - <<PY
import yaml
from pathlib import Path

plan = yaml.safe_load(open("$BUNDLE_ROOT/$PLAN"))
seen = set()
for run in plan["runs"]:
    if "${ONLY_RUN:-}" and run["id"] != "${ONLY_RUN:-}":
        continue
    cfg = yaml.safe_load(open(Path("$BUNDLE_ROOT/code/training") / run["config"]))
    tok = cfg["model"]["name"]
    if tok not in seen:
        print(tok)
        seen.add(tok)
PY
)
while IFS= read -r TOK; do
    [[ -z "$TOK" ]] && continue
    SLUG=$(echo "$TOK" | tr '/:' '__' | tr '[:upper:]' '[:lower:]')
    for SPLIT in train val; do
        REPORT="$BUNDLE_ROOT/train_outputs/_data_cache/validation_report_${SLUG}_${SPLIT}.json"
        SRC="$BUNDLE_ROOT/train_outputs/_data_cache/${SPLIT}_prompt_completion.jsonl"
        [[ -f "$SRC" ]] || continue   # val cache may legitimately be absent
        if [[ ! -f "$REPORT" ]]; then
            echo "=== Validating chat template for $TOK ($SPLIT) ==="
            "$VENV/python" -m training.data.validate_chat_template \
                --src "$SRC" \
                --tokenizer "$TOK" \
                --out "$REPORT"
        fi
        # Always inspect n_fail — under set -e a failed validate would have
        # exited above, but a cached report from a prior failed run (or
        # from data that has since changed) needs to be checked too.
        # Validate val too because runs have eval_strategy=steps and an
        # unrenderable val row would crash SFTTrainer's first eval pass
        # (not NaN, so NaNGuard can't catch it).
        N_FAIL=$("$VENV/python" -c "import json; print(json.load(open('$REPORT')).get('n_fail', 0))")
        if [[ "$N_FAIL" -gt 0 ]]; then
            echo "  ✗ $TOK ($SPLIT) chat-template validation has $N_FAIL failing rows; see $REPORT" >&2
            echo "  Delete the report file and re-run after investigating." >&2
            exit 1
        fi
    done
done <<< "$TOKENIZERS"
cd "$BUNDLE_ROOT"

# Build held-out task split + expand to eval-harness-ready descriptors.
if [[ ! -f train_outputs/_data_cache/heldout_task_ids.json ]] || \
   [[ ! -f train_outputs/_data_cache/heldout_tasks.jsonl ]]; then
    echo "=== Computing held-out task split ==="
    cd "$BUNDLE_ROOT/code"
    "$VENV/python" - <<'PYEOF'
import json
from pathlib import Path
import sys, os
sys.path.insert(0, ".")
from training.data.heldout_split import (
    select_heldout_task_ids,
    expand_heldout_ids_to_descriptors,
)
br = Path(os.environ["BUNDLE_ROOT"])
src = br / "data_processed/stage2_v1/train.jsonl"
rows = [json.loads(l) for l in src.read_text().splitlines()]
heldout = select_heldout_task_ids(rows, n_per_domain=5, seed=42)

ids_out = br / "train_outputs/_data_cache/heldout_task_ids.json"
ids_out.write_text(json.dumps({d: sorted(s) for d, s in heldout.items()}, indent=2))
print(f"Held-out task ids → {ids_out}")

# Expand to JSONL of descriptors (task_id/domain/max_steps/max_errors); this
# is what training.eval.harness consumes via --heldout-tasks.
descriptors = expand_heldout_ids_to_descriptors(
    {d: sorted(s) for d, s in heldout.items()}
)
desc_out = br / "train_outputs/_data_cache/heldout_tasks.jsonl"
desc_out.write_text("".join(json.dumps(d) + "\n" for d in descriptors))
print(f"Held-out task descriptors ({len(descriptors)} rows) → {desc_out}")
PYEOF
    cd "$BUNDLE_ROOT"
fi

# Per-run training.
RUNS=$(
    "$VENV/python" -c "
import yaml
plan = yaml.safe_load(open('$PLAN'))
for r in plan['runs']:
    print(r['id'], r['config'])
"
)

echo ""
echo "=== Training all runs ==="
while IFS=' ' read -r RID CFG; do
    [[ -z "$RID" ]] && continue
    if [[ -n "${ONLY_RUN:-}" && "$RID" != "$ONLY_RUN" ]]; then
        continue
    fi
    OUT_DIR="train_outputs/$RID"
    if [[ -n "${SCALING_TRAIN_FILE_STAGE2:-}" && -n "${SCALING_OUTPUT_DIR:-}" ]]; then
        OUT_DIR="$SCALING_OUTPUT_DIR"
    fi
    if [[ "$OUT_DIR" = /* ]]; then
        OUT_DIR_ABS="$OUT_DIR"
    else
        OUT_DIR_ABS="$BUNDLE_ROOT/$OUT_DIR"
    fi
    STATUS_FILE="$OUT_DIR_ABS/STATUS"
    # In the scaling closed loop the same RUN_CONFIG is retrained
    # every cycle on fresh per-cycle data (SCALING_TRAIN_FILE_STAGE2). The
    # STATUS=done idempotence skip would wrongly skip cycles k>=1. When the
    # scaling override is active, force a retrain (clear the stale STATUS).
    if [[ -n "${SCALING_TRAIN_FILE_STAGE2:-}" ]]; then
        [[ -f "$STATUS_FILE" ]] && { echo "  scaling: clearing stale STATUS for $RID (force retrain)"; rm -f "$STATUS_FILE"; }
    elif [[ -f "$STATUS_FILE" ]] && grep -q "^done$" "$STATUS_FILE"; then
        echo "  SKIP (done): $RID"
        continue
    fi
    mkdir -p "$OUT_DIR_ABS"
    echo ""
    echo "=== TRAIN: $RID ==="

    ACCEL_CFG=$("$VENV/python" -c "import yaml; print(yaml.safe_load(open('code/training/$CFG'))['distributed']['accelerate_config'])")
    RUN_CONFIG_ARG="training/$CFG"
    if [[ -n "${SCALING_TRAIN_FILE_STAGE2:-}" && -n "${SCALING_OUTPUT_DIR:-}" ]]; then
        CFG_DIR="$BUNDLE_ROOT/train_outputs/_scaling_configs"
        mkdir -p "$CFG_DIR"
        CFG_HASH=$(printf '%s' "$OUT_DIR_ABS" | sha1sum | awk '{print substr($1,1,12)}')
        GENERATED_CFG="$CFG_DIR/${RID}_${CFG_HASH}.yaml"
        "$VENV/python" - "$BUNDLE_ROOT/code/training/$CFG" "$GENERATED_CFG" "$OUT_DIR_ABS" <<'PY'
import os
import sys
from pathlib import Path
import yaml
src, dst, out_dir = map(Path, sys.argv[1:])
cfg = yaml.safe_load(src.read_text())
train = cfg.setdefault("training", {})
train["output_dir"] = str(out_dir)

overrides = {
    "SCALING_NUM_TRAIN_EPOCHS": ("num_train_epochs", float),
    "SCALING_MAX_SEQ_LENGTH": ("max_seq_length", int),
    "SCALING_GRADIENT_ACCUMULATION_STEPS": ("gradient_accumulation_steps", int),
    "SCALING_PER_DEVICE_TRAIN_BATCH_SIZE": ("per_device_train_batch_size", int),
    "SCALING_CONFIG_MAX_STEPS": ("max_steps", int),
}
string_overrides = {
    "SCALING_EVAL_STRATEGY": "eval_strategy",
    "SCALING_SAVE_STRATEGY": "save_strategy",
    "SCALING_ATTN_IMPLEMENTATION": "attn_implementation",
}
bool_overrides = {
    "SCALING_LOAD_BEST_MODEL_AT_END": "load_best_model_at_end",
    "SCALING_PACKING": "packing",
    "SCALING_PADDING_FREE": "padding_free",
}
applied = {}
for env_name, (cfg_key, caster) in overrides.items():
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        continue
    value = caster(raw)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    train[cfg_key] = value
    applied[cfg_key] = value

for env_name, cfg_key in string_overrides.items():
    raw = os.environ.get(env_name, "").strip()
    if raw:
        train[cfg_key] = raw
        applied[cfg_key] = raw

for env_name, cfg_key in bool_overrides.items():
    raw = os.environ.get(env_name, "").strip().lower()
    if raw:
        value = raw in {"1", "true", "yes", "on"}
        train[cfg_key] = value
        applied[cfg_key] = value

if applied:
    cfg.setdefault("scaling_overrides", {})["training"] = applied
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
        RUN_CONFIG_ARG="$GENERATED_CFG"
        echo "  scaling: using per-cycle output_dir=$OUT_DIR_ABS"
        if [[ -n "${SCALING_NUM_TRAIN_EPOCHS:-}" || -n "${SCALING_MAX_SEQ_LENGTH:-}" || -n "${SCALING_CONFIG_MAX_STEPS:-}" || -n "${SCALING_EVAL_STRATEGY:-}" || -n "${SCALING_SAVE_STRATEGY:-}" || -n "${SCALING_ATTN_IMPLEMENTATION:-}" || -n "${SCALING_PACKING:-}" || -n "${SCALING_PADDING_FREE:-}" ]]; then
            echo "  scaling: training overrides epochs=${SCALING_NUM_TRAIN_EPOCHS:-default} max_seq=${SCALING_MAX_SEQ_LENGTH:-default} max_steps=${SCALING_CONFIG_MAX_STEPS:-default} eval=${SCALING_EVAL_STRATEGY:-default} save=${SCALING_SAVE_STRATEGY:-default} attn=${SCALING_ATTN_IMPLEMENTATION:-default} packing=${SCALING_PACKING:-default} padding_free=${SCALING_PADDING_FREE:-default}"
        fi
    fi

    EXTRA_ARGS=()
    if [[ -n "${MAX_STEPS_OVERRIDE:-}" ]]; then
        # Pass through to train.py --max-steps; train.py caps steps with FULL
        # data + FULL batch/accum and writes STATUS=smoke_done (so eval_all.sh
        # skips the partial checkpoint).
        EXTRA_ARGS+=(--max-steps "$MAX_STEPS_OVERRIDE")
    fi

    cd "$BUNDLE_ROOT/code"
    # `set -e -o pipefail` is on; without `|| true` a non-zero accelerate
    # launch propagates through tee and aborts the entire 10-run sweep
    # before the failure-branch below can mark this run as failed and
    # continue to the next. We intentionally swallow it here and rely on
    # train.py writing STATUS=done to indicate success.
    ACCELERATE_LAUNCH=("$VENV/python" -m accelerate.commands.launch)
    if [[ -x "$VENV/accelerate" ]]; then
        ACCELERATE_LAUNCH=("$VENV/accelerate" launch)
    fi
    "${ACCELERATE_LAUNCH[@]}" \
        --config_file "training/$ACCEL_CFG" \
        -m training.train \
        --run-config "$RUN_CONFIG_ARG" \
        --plan-config "$BUNDLE_ROOT/$PLAN" \
        --bundle-root "$BUNDLE_ROOT" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$OUT_DIR_ABS/train_stdout.log" || true
    cd "$BUNDLE_ROOT"

    # Accept both "done" (full run) and "smoke_done" (--smoke-test from
    # MAX_STEPS_OVERRIDE) as success. The skip-check at line 136 only
    # matches "^done$" so smoke runs intentionally don't cause skip, but
    # the success check here must include smoke_done — without this, the
    # else-branch overwrites STATUS=smoke_done with "failed", and a real
    # green smoke shows up as red to the operator.
    if grep -qE "^(done|smoke_done)$" "$STATUS_FILE" 2>/dev/null; then
        if grep -q "^smoke_done$" "$STATUS_FILE" 2>/dev/null; then
            echo "  ✓ $RID smoke_done (1-step smoke; full run pending)"
        else
            echo "  ✓ $RID done"
        fi
    else
        echo "  ✗ $RID FAILED — see $OUT_DIR_ABS/train_stdout.log" >&2
        echo "failed" > "$STATUS_FILE"
    fi
done <<< "$RUNS"

# Sweep summary. Without this, after a 10-run sweep with N failures the
# operator only sees scattered ✗ lines mixed into multi-hour stdout; no
# terminal verdict. Count STATUS files post-loop and exit non-zero on any
# failure so downstream eval_all.sh / summarize.py don't silently process
# partial data.
N_DONE=0
N_SMOKE=0
N_FAILED=0
N_OTHER=0
N_TOTAL=0
while IFS=' ' read -r RID _; do
    [[ -z "$RID" ]] && continue
    if [[ -n "${ONLY_RUN:-}" && "$RID" != "$ONLY_RUN" ]]; then
        continue
    fi
    N_TOTAL=$((N_TOTAL + 1))
    if [[ -n "${SCALING_TRAIN_FILE_STAGE2:-}" && -n "${SCALING_OUTPUT_DIR:-}" ]]; then
        SF="$SCALING_OUTPUT_DIR/STATUS"
    else
        SF="train_outputs/$RID/STATUS"
    fi
    if [[ -f "$SF" ]] && grep -q "^done$" "$SF"; then
        N_DONE=$((N_DONE + 1))
    elif [[ -f "$SF" ]] && grep -q "^smoke_done$" "$SF"; then
        N_SMOKE=$((N_SMOKE + 1))
    elif [[ -f "$SF" ]] && grep -q "^failed$" "$SF"; then
        N_FAILED=$((N_FAILED + 1))
    else
        N_OTHER=$((N_OTHER + 1))
    fi
done <<< "$RUNS"

echo ""
echo "=== Phase 1 sweep summary ==="
echo "  Total runs in scope: $N_TOTAL"
echo "  ✓ done:        $N_DONE"
echo "  ✓ smoke_done:  $N_SMOKE  (1-step smoke; run full pipeline to upgrade)"
echo "  ✗ failed:      $N_FAILED"
echo "  ? other:       $N_OTHER  (STATUS missing, 'running', or unknown)"
echo ""

if [[ "$N_FAILED" -gt 0 || "$N_OTHER" -gt 0 ]]; then
    echo "WARNING: $((N_FAILED + N_OTHER)) of $N_TOTAL runs did not complete cleanly." >&2
    echo "         Subsequent eval_all.sh / summarize.py will operate on partial data." >&2
    exit 1
fi

echo "=== Phase 1 complete. Run eval_all.sh next. ==="
