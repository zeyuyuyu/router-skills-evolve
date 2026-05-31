#!/usr/bin/env bash
# run_full_pipeline.sh
#
# End-to-end: Skills evolve + Router training + LLM training + E2E ablation
# multi-cycle iteration. Default bench: tau2-bench. Supports SWE-Bench via
# adapter (see README §6).
#
# Usage:
#   bash run_full_pipeline.sh                 # full run
#   bash run_full_pipeline.sh --smoke         # 30-task smoke, 1 cycle, ~30 min
#   bash run_full_pipeline.sh --dry-run       # print plan, no execution
#   bash run_full_pipeline.sh --resume 2      # restart from cycle 2
#
# Requires env vars:
#   OPENAI_API_KEY  (Phase 1 trace collection + tau2 judge)
#   HF_TOKEN        (Qwen3 download; required for 35B-A3B)
#   BUNDLE_ROOT     (default: $PWD/experiments/tau2_stage2)
#
# Optional env vars:
#   EXPERIMENT_NAME   (default: scaling_$(date -u +%Y%m%d_%H%M%S))
#   BENCH             (default: tau2_bench; alt: swe_bench)
#   MODEL_SWEEP       (default: 04_qwen3_5_4b_273)
#   N_CYCLES          (default: 4)
#   SCHEDULE          (default: SLR; alt: LSR LRS SRL RSL RLS)
#   SMALL_MODEL       (default: deepseek/deepseek-v3.2)
#   LARGE_MODEL       (default: openai/gpt-5.4-2026-03-05)

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 0. Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

SMOKE=false
DRY_RUN=false
RESUME_FROM=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --resume) RESUME_FROM="$2"; shift 2 ;;
    --bench) BENCH="$2"; shift 2 ;;
    --model-config) MODEL_SWEEP="$2"; shift 2 ;;
    --n-cycles) N_CYCLES="$2"; shift 2 ;;
    --schedule) SCHEDULE="$2"; shift 2 ;;
    -h|--help) sed -n '1,30p' "$0"; exit 0 ;;
    *) echo "[ERROR] Unknown flag: $1"; exit 2 ;;
  esac
done

# ─────────────────────────────────────────────────────────────────────────────
# 1. Config defaults
# ─────────────────────────────────────────────────────────────────────────────

: "${BUNDLE_ROOT:=$PWD/experiments/tau2_stage2}"
: "${EXPERIMENT_NAME:=scaling_$(date -u +%Y%m%d_%H%M%S)}"
: "${BENCH:=tau2_bench}"
: "${MODEL_SWEEP:=04_qwen3_5_4b_273}"
: "${N_CYCLES:=4}"
: "${SCHEDULE:=SLR}"
: "${SMALL_MODEL:=deepseek/deepseek-v3.2}"
: "${LARGE_MODEL:=openai/gpt-5.4-2026-03-05}"

if $SMOKE; then
  MODEL_SWEEP="smoke_2b"
  N_CYCLES=1
  N_TASKS=30
else
  N_TASKS=848   # tau2-bench eval split size; SWE-Bench Lite uses 300
fi

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$REPO_ROOT/results/$EXPERIMENT_NAME"
mkdir -p "$RESULTS_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Pre-flight checks
# ─────────────────────────────────────────────────────────────────────────────

preflight() {
  echo "═══ Pre-flight ═══"
  echo "  EXPERIMENT_NAME = $EXPERIMENT_NAME"
  echo "  BENCH           = $BENCH"
  echo "  MODEL_SWEEP     = $MODEL_SWEEP"
  echo "  N_CYCLES        = $N_CYCLES"
  echo "  SCHEDULE        = $SCHEDULE  (Skills→LLM→Router default)"
  echo "  RESULTS_DIR     = $RESULTS_DIR"
  echo "  SMOKE=$SMOKE DRY_RUN=$DRY_RUN RESUME_FROM=$RESUME_FROM"
  echo

  # secrets
  [[ -z "${OPENAI_API_KEY:-}" ]] && { echo "[FATAL] OPENAI_API_KEY not set"; exit 3; }
  [[ -z "${HF_TOKEN:-}" ]] && echo "[WARN] HF_TOKEN not set — 35B-A3B download will be slow/rate-limited"

  # disk
  free_gb=$(df -BG "$REPO_ROOT" | tail -1 | awk '{print $4}' | tr -d 'G')
  (( free_gb < 100 )) && { echo "[FATAL] Need ≥100 GB free; have ${free_gb}G"; exit 3; }

  # GPUs
  if command -v nvidia-smi >/dev/null; then
    n_gpu=$(nvidia-smi -L | wc -l)
    echo "  GPUs detected: $n_gpu"
    if ! $SMOKE && (( n_gpu < 8 )); then
      echo "[WARN] Full run designed for 8 GPUs; have $n_gpu"
    fi
  fi

  # shepherd junk cleanup (per README §8 pitfall 5)
  shopt -s nullglob
  junk=(shepherd.log shepherd_*.log .shepherd shepherd_logs shepherd_local_*)
  for f in "${junk[@]}"; do
    if [[ -e "$f" ]]; then
      echo "  Removing shepherd junk: $f"
      $DRY_RUN || rm -rf "$f"
    fi
  done

  # snapshot config
  $DRY_RUN || cat > "$RESULTS_DIR/config_snapshot.json" <<EOF
{
  "experiment_name": "$EXPERIMENT_NAME",
  "bench": "$BENCH",
  "model_sweep": "$MODEL_SWEEP",
  "n_cycles": $N_CYCLES,
  "schedule": "$SCHEDULE",
  "small_model": "$SMALL_MODEL",
  "large_model": "$LARGE_MODEL",
  "smoke": $SMOKE,
  "resume_from": $RESUME_FROM,
  "git_sha": "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)",
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "host": "$(hostname)"
}
EOF
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Phase implementations
# ─────────────────────────────────────────────────────────────────────────────
#
# Each phase reads its inputs from previous cycle's outputs and writes to
# results/$EXPERIMENT_NAME/cycle_$N/.
#
# All scripts are expected to exist in main branch's experiments/ directory.
# For tau2-bench LLM training we route through colleague's tau2_stage2 framework.

phase1_collect_traces() {
  local cycle=$1
  local out="$RESULTS_DIR/cycle_$cycle"
  mkdir -p "$out"
  echo "  [Phase 1] Trace collection — bench=$BENCH cycle=$cycle"

  # Inputs:
  #   - small model: cycle 0 → SMALL_MODEL; cycle ≥1 → previous adapter
  #   - large model: LARGE_MODEL (always)
  local small_arg="$SMALL_MODEL"
  if (( cycle > 0 )); then
    small_arg="$RESULTS_DIR/cycle_$((cycle-1))/llm_adapter/checkpoint-best"
  fi

  local cmd=(
    python experiments/scaling/collect_traces.py
    --bench "$BENCH"
    --n-tasks "$N_TASKS"
    --small-model "$small_arg"
    --large-model "$LARGE_MODEL"
    --out "$out/traces.jsonl"
  )
  $DRY_RUN && { echo "  DRY: ${cmd[*]}"; return; }
  "${cmd[@]}" 2>&1 | tee "$out/phase1.log"
}

phase2_skills_evolve() {
  local cycle=$1
  local out="$RESULTS_DIR/cycle_$cycle"
  echo "  [Phase 2] Skills evolve (SkillBook) — cycle=$cycle"

  local prev_skillbook=""
  if (( cycle > 0 )); then
    prev_skillbook="--prev-skillbook $RESULTS_DIR/cycle_$((cycle-1))/skillbook.json"
  fi

  local cmd=(
    python experiments/run_evolve.py
    --traces "$out/traces.jsonl"
    $prev_skillbook
    --out "$out/skillbook.json"
  )
  $DRY_RUN && { echo "  DRY: ${cmd[*]}"; return; }
  "${cmd[@]}" 2>&1 | tee "$out/phase2.log"
}

phase3_llm_train() {
  local cycle=$1
  local out="$RESULTS_DIR/cycle_$cycle"
  echo "  [Phase 3] LLM training (via tau2_stage2 framework) — cycle=$cycle model=$MODEL_SWEEP"

  # Extract training slice: where small model failed but is "learnable"
  $DRY_RUN || python experiments/extract_training_data.py \
    --traces "$out/traces.jsonl" \
    --skillbook "$out/skillbook.json" \
    --output "$out/training_data.jsonl" \
    2>&1 | tee -a "$out/phase3_extract.log"

  # Reuse colleague's tau2_stage2 SFT framework
  # NOTE: tau2_stage2/code/training/orchestration/train_pipeline.sh assumes its
  # own data_processed/ layout; we patch via env var to point at our extract.
  TRAINING_DATA="$out/training_data.jsonl" \
  TRAIN_OUTPUT_DIR="$out/llm_adapter" \
  RUN_CONFIG="$MODEL_SWEEP" \
  BUNDLE_ROOT="$BUNDLE_ROOT" \
  $DRY_RUN || bash "$BUNDLE_ROOT/code/training/orchestration/train_pipeline.sh" \
    2>&1 | tee "$out/phase3_train.log"

  $DRY_RUN && echo "  DRY: would invoke tau2_stage2 train_pipeline.sh with RUN_CONFIG=$MODEL_SWEEP"
}

phase4_router_train() {
  local cycle=$1
  local out="$RESULTS_DIR/cycle_$cycle"
  echo "  [Phase 4] Router training — cycle=$cycle"

  # Re-label traces using the NEW LLM (small_model boundary shifted)
  $DRY_RUN || python experiments/extract_training_data.py \
    --traces "$out/traces.jsonl" \
    --small-model "$out/llm_adapter/checkpoint-best" \
    --output "$out/router_train.jsonl" \
    --mode router_labels \
    2>&1 | tee -a "$out/phase4_label.log"

  local cmd=(
    python experiments/train_learnable_router.py
    --train-data "$out/router_train.jsonl"
    --output "$out/router.pkl"
  )
  $DRY_RUN && { echo "  DRY: ${cmd[*]}"; return; }
  "${cmd[@]}" 2>&1 | tee "$out/phase4_train.log"

  # Tune threshold on held-out
  $DRY_RUN || python experiments/tune_learnable_router_threshold.py \
    --router "$out/router.pkl" \
    --eval-data "$out/router_train.jsonl" \
    --output "$out/router_threshold.json" \
    2>&1 | tee "$out/phase4_tune.log"
}

phase5_e2e_ablation() {
  local cycle=$1
  local out="$RESULTS_DIR/cycle_$cycle"
  echo "  [Phase 5] E2E ablation — cycle=$cycle"

  local cmd=(
    python experiments/run_e2e_ablation.py
    --bench "$BENCH"
    --skillbook "$out/skillbook.json"
    --router "$out/router.pkl"
    --router-threshold "$out/router_threshold.json"
    --llm-adapter "$out/llm_adapter/checkpoint-best"
    --eval-tasks "$N_TASKS"
    --output "$out/e2e_ablation_summary.json"
  )
  $DRY_RUN && { echo "  DRY: ${cmd[*]}"; return; }
  "${cmd[@]}" 2>&1 | tee "$out/phase5.log"

  echo "  Cycle $cycle ablation:"
  python -c "
import json
d = json.load(open('$out/e2e_ablation_summary.json'))
for variant, m in d['variants'].items():
    print(f'    {variant:8s}  routing_acc={m[\"routing_acc\"]:.2%!|(MISSING)  task_pass={m[\"task_pass\"]:.2%!}(MISSING)')
" || true
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Cycle orchestration with schedule support
# ─────────────────────────────────────────────────────────────────────────────

run_cycle() {
  local cycle=$1
  echo "═══ Cycle $cycle / $((N_CYCLES-1))  schedule=$SCHEDULE ═══"
  local out="$RESULTS_DIR/cycle_$cycle"
  mkdir -p "$out"

  # Phase 1 (trace collection) always first
  phase1_collect_traces "$cycle"

  # Phases 2/3/4 ordered by $SCHEDULE
  # S = Skills, L = LLM, R = Router
  for stage in $(echo "$SCHEDULE" | grep -o .); do
    case "$stage" in
      S) phase2_skills_evolve "$cycle" ;;
      L) phase3_llm_train "$cycle" ;;
      R) phase4_router_train "$cycle" ;;
      *) echo "[FATAL] Unknown schedule stage: $stage"; exit 4 ;;
    esac
  done

  # Phase 5 always last
  phase5_e2e_ablation "$cycle"
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. Final aggregation
# ─────────────────────────────────────────────────────────────────────────────

aggregate() {
  echo "═══ Aggregation ═══"
  $DRY_RUN && { echo "  DRY: would aggregate cycles 0..$((N_CYCLES-1))"; return; }

  python experiments/scaling/aggregate_cycles.py \
    --experiment-dir "$RESULTS_DIR" \
    --n-cycles "$N_CYCLES" \
    --output-md "$RESULTS_DIR/final_ablation_table.md" \
    --output-png "$RESULTS_DIR/curve.png" \
    2>&1 | tee "$RESULTS_DIR/aggregate.log"

  echo
  echo "═══════════════════════════════════════════════════"
  echo "  DONE.  Final table:    $RESULTS_DIR/final_ablation_table.md"
  echo "        Iteration curve: $RESULTS_DIR/curve.png"
  echo "═══════════════════════════════════════════════════"
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

preflight

for ((cycle=RESUME_FROM; cycle<N_CYCLES; cycle++)); do
  run_cycle "$cycle"
done

aggregate
