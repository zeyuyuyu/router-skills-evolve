#!/usr/bin/env bash
# code/training/orchestration/eval_all_step_budget.sh
#
# Group-aware scheduler for step_budget_harness.py. On the 8× H200 server:
#   1. 4B family   (5 targets, TP=2 → 4 parallel + 1 leftover wave)
#   2. 9B family   (3 targets, TP=2 → 1 wave of 3 parallel; 2 GPUs idle)
#   3. 2B family   (2 targets, TP=2 → 1 wave of 2 parallel; 4 GPUs idle)
#   4. 35B family  (2 targets, TP=8, fully sequential — all 8 GPUs per eval)
#
# TP=2 for small models is the headline change over the original layout:
# pairing two H200s per vLLM gives each instance ~120 GB of KV cache budget,
# enough to serve --max-model-len=131072 (4× the SFT context) comfortably.
# The longest eval tasks (airline/8 with B=59) need ~75K tokens in Method B
# at max_steps=B+10; the previous TP=1 layout capped at 65K and would clip
# those rollouts.
#
# Within each group, batches of `8 / tp` targets run concurrently, each in
# its own subprocess serving vLLM on a distinct port + contiguous GPU pair.
# Wave 2 of the 4B family contains a single target and uses GPUs 0..1.
# The harness's per-rollout atomic writes mean an interrupted run resumes
# cleanly on rerun, regardless of which wave it was killed in.
#
# Usage:
#   BUNDLE_ROOT=/path/to/bundle SEED_POLICY=primary \
#       bash code/training/orchestration/eval_all_step_budget.sh [--group {4b,9b,2b,35b,all}]
#
# Env (loaded from $BUNDLE_ROOT/.env if present):
#   BUNDLE_ROOT (required)
#   OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_BASE_URL  (CommonStack)
#   HF_TOKEN
#   SEED_POLICY = primary | secondary  (default: primary)
#   GROUP = 4b | 9b | 2b | 35b | all   (default: all)
#   ONLY_TARGET = <target_name>        (optional; restricts to one target)
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:?set BUNDLE_ROOT}"
cd "$BUNDLE_ROOT"

# Source .env if present.
if [[ -f "$BUNDLE_ROOT/.env" ]]; then
    set -a; source "$BUNDLE_ROOT/.env"; set +a
fi

VENV="${VENV:-${CONDA_PREFIX:+$CONDA_PREFIX/bin}}"
VENV="${VENV:-$BUNDLE_ROOT/code/.venv/bin}"
EVAL_TASKS="$BUNDLE_ROOT/data_processed/stage2_v1/eval_tasks.jsonl"
BASELINES_ROOT="$BUNDLE_ROOT/eval_baselines"
OUTPUTS_ROOT="$BUNDLE_ROOT/step_budget_outputs"
mkdir -p "$OUTPUTS_ROOT"

SEED_POLICY="${SEED_POLICY:-primary}"
GROUP="${GROUP:-all}"
ONLY_TARGET="${ONLY_TARGET:-}"
METHOD="${METHOD:-both}"  # both | A | B

# --- parse --group flag (overrides $GROUP) -------------------------------- #
while [[ $# -gt 0 ]]; do
    case "$1" in
        --group)        GROUP="$2"; shift 2 ;;
        --seed-policy)  SEED_POLICY="$2"; shift 2 ;;
        --only-target)  ONLY_TARGET="$2"; shift 2 ;;
        --method)       METHOD="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ ! -f "$EVAL_TASKS" ]]; then
    echo "ERROR: eval tasks not found at $EVAL_TASKS" >&2; exit 2
fi
if [[ ! -d "$BASELINES_ROOT" ]]; then
    echo "ERROR: baselines not found at $BASELINES_ROOT — run setup_step_budget_eval.sh first." >&2
    exit 2
fi

# --- Target inventory ----------------------------------------------------- #
# Each row: <target_name> <checkpoint_path_relative_to_BUNDLE_ROOT> <tp> <group>
# Trained checkpoints: train_outputs/<run_id>/checkpoint-best (fallback: checkpoint-final)
# Base models:       base_models/<dirname>/
ALL_TARGETS=(
    # 4B family — TP=2 (5 targets → 4 parallel + 1 leftover)
    "base_Qwen3.5-4B          base_models/base_Qwen3.5-4B                   2  4b"
    "02_qwen3_5_4b_50         train_outputs/02_qwen3_5_4b_50                2  4b"
    "03_qwen3_5_4b_100        train_outputs/03_qwen3_5_4b_100               2  4b"
    "04_qwen3_5_4b_200        train_outputs/04_qwen3_5_4b_200               2  4b"
    "05_qwen3_5_4b_273        train_outputs/05_qwen3_5_4b_273               2  4b"
    # 9B family — TP=2 (3 targets in one wave)
    "base_Qwen3.5-9B          base_models/base_Qwen3.5-9B                   2  9b"
    "06_qwen3_5_9b_50         train_outputs/06_qwen3_5_9b_50                2  9b"
    "07_qwen3_5_9b_273        train_outputs/07_qwen3_5_9b_273               2  9b"
    # 2B family — TP=2 (2 targets in one wave)
    "base_Qwen3.5-2B          base_models/base_Qwen3.5-2B                   2  2b"
    "01_qwen3_5_2b_273        train_outputs/01_qwen3_5_2b_273               2  2b"
    # 35B family — TP=8, fully sequential (one model claims all 8 GPUs)
    "base_Qwen3.6-35B-A3B     base_models/base_Qwen3.6-35B-A3B              8  35b"
    "08_qwen3_6_35b_a3b_273   train_outputs/08_qwen3_6_35b_a3b_273          8  35b"
)

# Total physical GPUs on the server. Used to compute per-wave parallelism
# (NUM_GPUS / tp_size). Override via env var on non-standard machines.
NUM_GPUS="${NUM_GPUS:-8}"
PORT_BASE="${PORT_BASE:-8000}"

# Resolve checkpoint path: for trained targets, prefer checkpoint-best.
resolve_ckpt() {
    local rel="$1"
    local abs="$BUNDLE_ROOT/$rel"
    if [[ "$rel" == base_models/* ]]; then
        echo "$abs"
        return 0
    fi
    if [[ -d "$abs/checkpoint-best" ]]; then
        echo "$abs/checkpoint-best"; return 0
    fi
    if [[ -d "$abs/checkpoint-final" ]]; then
        echo "$abs/checkpoint-final"; return 0
    fi
    echo ""
}

# --- Launch one harness process (background-friendly) --------------------- #
launch_one() {
    local target="$1"
    local ckpt="$2"
    local tp="$3"
    local port="$4"
    local gpu="$5"

    local out_dir="$OUTPUTS_ROOT/$target"
    mkdir -p "$out_dir"
    echo "[$target] starting; gpu=$gpu port=$port tp=$tp ckpt=$ckpt"
    cd "$BUNDLE_ROOT/code"
    # Each subprocess inherits env vars; logs combined to per-target file.
    "$VENV/python" -m training.eval.step_budget_harness \
        --target "$target" \
        --checkpoint "$ckpt" \
        --bundle-root "$BUNDLE_ROOT" \
        --baselines-root "$BASELINES_ROOT" \
        --eval-tasks "$EVAL_TASKS" \
        --output-dir "$out_dir" \
        --port "$port" --gpu "$gpu" --tp "$tp" \
        --seed-policy "$SEED_POLICY" \
        --method "$METHOD" \
        > "$out_dir/harness_stdout.${SEED_POLICY}.log" 2>&1
    local rc=$?
    cd "$BUNDLE_ROOT"
    return $rc
}

# --- Group dispatcher ----------------------------------------------------- #
# For each group, batch targets into waves of (NUM_GPUS / tp) parallel
# instances. Within a wave, assign GPU base indices 0, tp, 2*tp, ... and
# distinct ports 8000, 8001, ... Wait for the wave to finish before launching
# the next. This produces:
#   4B (TP=2): wave1 of 4 targets on GPUs (0-1, 2-3, 4-5, 6-7),
#              wave2 of 1 target on GPUs (0-1). 2 waves.
#   9B (TP=2): wave1 of 3 targets on GPUs (0-1, 2-3, 4-5). 1 wave; 2 idle.
#   2B (TP=2): wave1 of 2 targets on GPUs (0-1, 2-3). 1 wave; 4 idle.
#   35B (TP=8): wave1 of 1 target on GPUs 0..7; wave2 of 1 target. 2 waves.
GROUP_ORDER=("4b" "9b" "2b" "35b")
[[ "$GROUP" != "all" ]] && GROUP_ORDER=("$GROUP")

OVERALL_FAILURES=0
ANY_CREDIT_EXHAUSTED=0

# Run one wave of targets in parallel. Args:
#   $1 = wave label (for logging)
#   $2 = port_base — first port to use for this wave (subsequent targets in
#        the wave get base+1, base+2, ...). Each wave should use a fresh
#        port range so a previous wave's TCP connections in TIME_WAIT can't
#        block a new vLLM bind.
#   $@ = list of "tname|ckpt|tp" entries (all entries MUST have the same tp).
# Returns 42 immediately if any child returns 42 (credit exhausted), else
# the count of failed children (accumulated into OVERALL_FAILURES by caller).
run_wave() {
    local wave_label="$1"; shift
    local port_base="$1"; shift
    local -a wave_entries=("$@")
    [[ ${#wave_entries[@]} -eq 0 ]] && return 0

    unset PIDS PID_TO_NAME
    declare -a PIDS=()
    declare -A PID_TO_NAME=()
    local local_gpu=0
    local local_port="$port_base"
    local wave_tp=""
    for entry in "${wave_entries[@]}"; do
        IFS='|' read -r tname ckpt ttp <<< "$entry"
        if [[ -z "$wave_tp" ]]; then wave_tp="$ttp"; fi
        if (( local_gpu + ttp > NUM_GPUS )); then
            echo "WAVE OVERFLOW: trying to place $tname at GPU=$local_gpu (tp=$ttp) on $NUM_GPUS GPUs" >&2
            return 1
        fi
        echo "  wave=$wave_label launch $tname gpu=$local_gpu..$((local_gpu+ttp-1)) port=$local_port"
        launch_one "$tname" "$ckpt" "$ttp" "$local_port" "$local_gpu" &
        pid=$!
        PIDS+=("$pid")
        PID_TO_NAME[$pid]="$tname"
        local_gpu=$((local_gpu + ttp))
        local_port=$((local_port + 1))
    done

    local local_rc=0
    local credit_seen=0
    for pid in "${PIDS[@]}"; do
        if wait "$pid"; then
            echo "[${PID_TO_NAME[$pid]}] completed OK"
        else
            rc=$?
            if [[ $rc -eq 42 ]]; then
                echo "[${PID_TO_NAME[$pid]}] credit exhausted (exit 42)" >&2
                credit_seen=1
            fi
            echo "[${PID_TO_NAME[$pid]}] FAILED with exit code $rc — see $OUTPUTS_ROOT/${PID_TO_NAME[$pid]}/harness_stdout.${SEED_POLICY}.log" >&2
            local_rc=$((local_rc + 1))
        fi
    done

    if [[ "$credit_seen" -eq 1 ]]; then
        return 42
    fi
    return $local_rc
}

for grp in "${GROUP_ORDER[@]}"; do
    echo ""
    echo "====================================================================="
    echo "=== GROUP: $grp ($SEED_POLICY seed)                                "
    echo "====================================================================="

    # Build target list for this group
    declare -a TGT_LIST=()
    GROUP_TP=""
    for row in "${ALL_TARGETS[@]}"; do
        # shellcheck disable=SC2086
        read -r tname tpath ttp tgrp <<< $row
        if [[ "$tgrp" != "$grp" ]]; then continue; fi
        if [[ -n "$ONLY_TARGET" && "$tname" != "$ONLY_TARGET" ]]; then continue; fi
        ckpt=$(resolve_ckpt "$tpath")
        if [[ -z "$ckpt" ]]; then
            echo "[$tname] SKIP — checkpoint not found at $BUNDLE_ROOT/$tpath"
            continue
        fi
        TGT_LIST+=("$tname|$ckpt|$ttp")
        # All targets in a group share the same TP — sanity-checked here so
        # an accidental ALL_TARGETS edit doesn't cause silent mis-scheduling.
        if [[ -z "$GROUP_TP" ]]; then
            GROUP_TP="$ttp"
        elif [[ "$GROUP_TP" != "$ttp" ]]; then
            echo "ERROR: group $grp mixes tp=$GROUP_TP and tp=$ttp (target $tname)" >&2
            exit 2
        fi
    done

    if [[ ${#TGT_LIST[@]} -eq 0 ]]; then
        echo "  (no eligible targets in this group)"
        continue
    fi

    PARALLELISM=$(( NUM_GPUS / GROUP_TP ))
    if (( PARALLELISM < 1 )); then PARALLELISM=1; fi
    N_WAVES=$(( (${#TGT_LIST[@]} + PARALLELISM - 1) / PARALLELISM ))
    echo "  group=$grp tp=$GROUP_TP parallelism=$PARALLELISM targets=${#TGT_LIST[@]} waves=$N_WAVES"

    wave_idx=0
    while (( wave_idx < N_WAVES )); do
        wave_start=$(( wave_idx * PARALLELISM ))
        wave_end=$(( wave_start + PARALLELISM ))
        (( wave_end > ${#TGT_LIST[@]} )) && wave_end=${#TGT_LIST[@]}
        declare -a WAVE_ENTRIES=()
        for (( i = wave_start; i < wave_end; i++ )); do
            WAVE_ENTRIES+=("${TGT_LIST[$i]}")
        done
        # Each wave uses a distinct port range so a stale vLLM connection
        # from the previous wave in TIME_WAIT can't EADDRINUSE the new bind.
        wave_port_base=$(( PORT_BASE + wave_idx * PARALLELISM ))
        run_wave "$grp/$((wave_idx + 1))/$N_WAVES" "$wave_port_base" "${WAVE_ENTRIES[@]}"
        wave_rc=$?
        if [[ $wave_rc -eq 42 ]]; then
            ANY_CREDIT_EXHAUSTED=1
            OVERALL_FAILURES=$((OVERALL_FAILURES + wave_rc))
            break
        elif (( wave_rc > 0 )); then
            OVERALL_FAILURES=$((OVERALL_FAILURES + wave_rc))
        fi
        wave_idx=$((wave_idx + 1))
    done

    if [[ "$ANY_CREDIT_EXHAUSTED" -eq 1 ]]; then
        echo "Stopping further groups: credit exhausted." >&2
        break
    fi
done

echo ""
echo "====================================================================="
echo "=== eval_all_step_budget.sh DONE (seed_policy=$SEED_POLICY, group=$GROUP)"
echo "  failures: $OVERALL_FAILURES; credit_exhausted: $ANY_CREDIT_EXHAUSTED"
echo "====================================================================="

if [[ "$ANY_CREDIT_EXHAUSTED" -eq 1 ]]; then
    exit 42
fi
if [[ "$OVERALL_FAILURES" -gt 0 ]]; then
    exit 1
fi
exit 0
