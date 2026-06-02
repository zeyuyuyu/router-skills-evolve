#!/usr/bin/env bash
# run_full_pipeline.sh
#
# End-to-end: Skills evolve + Router training + LLM training + E2E ablation,
# multi-cycle iteration. Default bench: tau2-bench. SWE-Bench adapter is a stub.
#
# Quick smoke (no GPU / no API key required — uses mock adapter):
#   bash scaling/run_full_pipeline.sh --smoke --mock
#
# Real run (8×H200 or 8×A800; OPENAI_API_KEY + HF_TOKEN required):
#   bash scaling/run_full_pipeline.sh
#
# Required env vars (for non-mock runs):
#   OPENAI_API_KEY  Phase 1 large-model invocation + tau2 NL judge
#   HF_TOKEN        Qwen3 download (required for 35B-A3B)
#
# Optional env vars:
#   EXPERIMENT_NAME   default: scaling_$(date -u +%Y%m%d_%H%M%S)
#   BENCH             tau2_bench (default) | swe_bench (NotImplemented stub)
#   MODEL_SWEEP       one of experiments/tau2_stage2/code/training/configs/runs/*.yaml (basename)
#                     default: 05_qwen3_5_4b_273
#                     for smoke: smoke_2b
#   N_CYCLES          MERA default 4; main-branch 5/20 ran 8
#   SCHEDULE          Skills(S)/LLM(L)/Router(R) order. Default: SLR.
#                     Alternatives: LSR LRS SRL RSL RLS
#   SMALL_MODEL       default: deepseek/deepseek-v3.2
#   LARGE_MODEL       default: openai/gpt-5.4-2026-03-05
#   TAU2_DOMAIN       airline | retail | telecom (default: retail)
#   SKIP_LLM          set to 1 to skip Phase 3 (useful while colleague's
#                     train framework is being set up)

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 0. Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

SMOKE=false
MOCK=false
DRY_RUN=false
RESUME_FROM=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=true; shift ;;
    --mock) MOCK=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --resume) RESUME_FROM="$2"; shift 2 ;;
    --bench) BENCH="$2"; shift 2 ;;
    --model-config) MODEL_SWEEP="$2"; shift 2 ;;
    --n-cycles) N_CYCLES="$2"; shift 2 ;;
    --schedule) SCHEDULE="$2"; shift 2 ;;
    --skip-llm) SKIP_LLM=1; shift ;;
    -h|--help) sed -n '1,40p' "$0"; exit 0 ;;
    *) echo "[ERROR] Unknown flag: $1"; exit 2 ;;
  esac
done

# ─────────────────────────────────────────────────────────────────────────────
# 1. Config defaults
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${BUNDLE_ROOT:=$REPO_ROOT/experiments/tau2_stage2}"
: "${EXPERIMENT_NAME:=scaling_$(date -u +%Y%m%d_%H%M%S)}"
: "${BENCH:=tau2_bench}"
: "${MODEL_SWEEP:=05_qwen3_5_4b_273}"
: "${N_CYCLES:=4}"
: "${SCHEDULE:=SLR}"
: "${SMALL_MODEL:=deepseek/deepseek-v3.2}"
: "${LARGE_MODEL:=openai/gpt-5.4-2026-03-05}"
: "${TAU2_DOMAIN:=retail}"
: "${SKIP_LLM:=0}"

export TAU2_DOMAIN

if $SMOKE; then
  MODEL_SWEEP="smoke_2b"
  N_CYCLES=1
  N_TASKS=30
  SKIP_LLM=1   # smoke skips LLM training by default
else
  N_TASKS=848   # tau2-bench eval split size; SWE-Bench Lite ≈ 300
fi

if $MOCK; then
  export SCALING_MOCK=1
fi

RESULTS_DIR="$REPO_ROOT/results/$EXPERIMENT_NAME"
mkdir -p "$RESULTS_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Pre-flight
# ─────────────────────────────────────────────────────────────────────────────

preflight() {
  echo "═══ Pre-flight ═══"
  echo "  EXPERIMENT_NAME = $EXPERIMENT_NAME"
  echo "  BENCH           = $BENCH"
  echo "  MODEL_SWEEP     = $MODEL_SWEEP"
  echo "  N_CYCLES        = $N_CYCLES"
  echo "  SCHEDULE        = $SCHEDULE  (Skills→LLM→Router default)"
  echo "  TAU2_DOMAIN     = $TAU2_DOMAIN"
  echo "  RESULTS_DIR     = $RESULTS_DIR"
  echo "  SMOKE=$SMOKE MOCK=$MOCK DRY_RUN=$DRY_RUN RESUME_FROM=$RESUME_FROM SKIP_LLM=$SKIP_LLM"
  echo
  # Python detection (system may have python3 but no python)
  PYTHON="${PYTHON:-}"
  if [[ -z "$PYTHON" ]]; then
    if   command -v python3 >/dev/null 2>&1; then PYTHON=python3
    elif command -v python  >/dev/null 2>&1; then PYTHON=python
    else echo "[FATAL] No python interpreter found"; exit 3
    fi
  fi
  echo "  PYTHON          = $PYTHON ($($PYTHON --version 2>&1))"



  # Secret checks (skipped in mock mode)
  if ! $MOCK; then
    [[ -z "${OPENAI_API_KEY:-}" ]] && { echo "[FATAL] OPENAI_API_KEY not set (or run with --mock)"; exit 3; }
    [[ -z "${HF_TOKEN:-}" ]] && echo "[WARN] HF_TOKEN not set — large model download will be slow"
  fi

  # Disk
  free_gb=$(df -BG "$REPO_ROOT" 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G' || echo 999)
  if ! $MOCK && (( free_gb < 100 )); then
    echo "[FATAL] Need ≥100 GB free for non-mock runs; have ${free_gb}G"; exit 3
  fi

  # GPUs
  if command -v nvidia-smi >/dev/null 2>&1; then
    n_gpu=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
    echo "  GPUs detected: $n_gpu"
    if ! $SMOKE && ! $MOCK && (( n_gpu < 8 )); then
      echo "[WARN] Full run designed for 8 GPUs; have $n_gpu"
    fi
  fi

  # tau2_stage2 bundle presence (non-mock only)
  if [[ "$BENCH" == "tau2_bench" ]] && ! $MOCK && [[ ! -d "$BUNDLE_ROOT" ]]; then
    echo "[FATAL] BUNDLE_ROOT=$BUNDLE_ROOT does not exist."
    echo "        Merge codex/tau2-stage2-training-eval branch first:"
    echo "          git merge origin/codex/tau2-stage2-training-eval --no-edit"
    exit 3
  fi

  # Shepherd junk cleanup (see README common pitfall 5)
  shopt -s nullglob
  for f in "$REPO_ROOT"/shepherd.log "$REPO_ROOT"/shepherd_*.log "$REPO_ROOT"/.shepherd "$REPO_ROOT"/shepherd_logs "$REPO_ROOT"/shepherd_local_*; do
    [[ -e "$f" ]] && { echo "  Removing shepherd junk: $(basename "$f")"; $DRY_RUN || rm -rf "$f"; }
  done

  # snapshot config
  $DRY_RUN || cat > "$RESULTS_DIR/config_snapshot.json" <<EOF
{
  "experiment_name": "$EXPERIMENT_NAME",
  "bench": "$BENCH",
  "model_sweep": "$MODEL_SWEEP",
  "n_cycles": $N_CYCLES,
  "schedule": "$SCHEDULE",
  "tau2_domain": "$TAU2_DOMAIN",
  "small_model": "$SMALL_MODEL",
  "large_model": "$LARGE_MODEL",
  "smoke": $SMOKE,
  "mock": $MOCK,
  "skip_llm": $([ "$SKIP_LLM" -eq 1 ] && echo true || echo false),
  "resume_from": $RESUME_FROM,
  "git_sha": "$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)",
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "host": "$(hostname)"
}
EOF
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Phase implementations
# ─────────────────────────────────────────────────────────────────────────────

phase1_collect_traces() {
  local cycle=$1
  local out="$RESULTS_DIR/cycle_$cycle"
  mkdir -p "$out"
  echo "  [Phase 1] Trace collection — bench=$BENCH cycle=$cycle"

  # In cycles ≥1, the loop is closed with the PREVIOUS cycle's latest artifacts:
  #   - small model  = previous cycle's trained LLM adapter
  #   - router       = previous cycle's trained router.joblib
  #   - skillbook    = previous cycle's carried-over skillbook.json
  # (review 2026-05-21: phase 1 must route with the evolved router + skillbook,
  #  not just the evolved adapter.)
  local small_arg="$SMALL_MODEL"
  local prev_router="" prev_skillbook=""
  if (( cycle > 0 )); then
    local prev="$RESULTS_DIR/cycle_$((cycle-1))/llm_adapter/checkpoint-best"
    [[ -e "$prev" ]] && small_arg="$prev"
    prev_router="$RESULTS_DIR/cycle_$((cycle-1))/router/router.joblib"
    prev_skillbook="$RESULTS_DIR/cycle_$((cycle-1))/skillbook.json"
  fi

  local cmd=(
    "$PYTHON" "$REPO_ROOT/experiments/scaling/collect_traces.py"
    --bench "$BENCH"
    --n-tasks "$N_TASKS"
    --small-model "$small_arg"
    --large-model "$LARGE_MODEL"
    --cycle "$cycle"
    --out "$out/traces.jsonl"
  )
  [[ -n "$prev_router"    && -e "$prev_router"    ]] && cmd+=(--router "$prev_router")
  [[ -n "$prev_skillbook" && -e "$prev_skillbook" ]] && cmd+=(--skillbook "$prev_skillbook")
  $MOCK && cmd+=(--mock)
  $DRY_RUN && { echo "  DRY: ${cmd[*]}"; return; }
  "${cmd[@]}" 2>&1 | tee "$out/phase1.log"
}

phase2_skills_evolve() {
  local cycle=$1
  local out="$RESULTS_DIR/cycle_$cycle"
  echo "  [Phase 2] Skills evolve — cycle=$cycle"

  $DRY_RUN && { echo "  DRY: build skillbook from $out/traces.jsonl"; return; }

  # Build SkillBook directly from our traces using src/skills.py
  "$PYTHON" - <<PY 2>&1 | tee "$out/phase2.log"
import json, sys
from pathlib import Path
sys.path.insert(0, "$REPO_ROOT")
from src.skills import SkillBook

sb = SkillBook()
prev = Path("$RESULTS_DIR/cycle_$((cycle-1))/skillbook.json")
if $cycle > 0 and prev.exists():
    sb.load(prev)
    print(f"[skills] carried over from cycle $((cycle-1))")

n = 0
with open("$out/traces.jsonl") as fh:
    for line in fh:
        try:
            t = json.loads(line)
        except json.JSONDecodeError:
            continue
        prompt = t.get("prompt") or t.get("signature") or t.get("task_id", "")
        # Prefer the POLICY outcome (what the evolved router+skillbook actually
        # routed to) over the adapter's original small-first decision. Fall back
        # to final_* for cycle-0 / non-closed-loop traces, or when the policy
        # outcome is unknown (large_skipped). Review round 2, 2026-05-21.
        if t.get("policy_final_model") and t.get("policy_final_success") is not None:
            model = t["policy_final_model"]
            succ = bool(t["policy_final_success"])
        else:
            model = t.get("final_model", "")
            succ = bool(t.get("final_success", False))
        # Procedural skills (review item 2): store the successful completion as
        # an exemplar so the cluster's procedure can be distilled. Prefer the
        # large model's completion (the teacher) for hard tasks.
        completion = t.get("large_completion") or t.get("small_completion") or ""
        if prompt and model:
            sb.update(prompt, model, succ, t.get("task_id", ""), completion=completion)
            n += 1

# Distill a reusable procedure per cluster from the accumulated exemplars.
# Heuristic (no-API) by default; an LLM distiller can be wired here later for
# the full "agent sub-workflow" induction (tool-use steps / domain policy).
n_proc = sb.distill_all()

out_path = Path("$out/skillbook.json")
sb.save(out_path)
size = len(getattr(sb, "skills", {}))
print(f"[skills_evolve] ingested {n} traces  SkillBook size={size}  "
      f"procedures_distilled={n_proc}  wrote {out_path}")
PY
}

phase3_llm_train() {
  local cycle=$1
  local out="$RESULTS_DIR/cycle_$cycle"

  if [[ "$SKIP_LLM" -eq 1 ]]; then
    echo "  [Phase 3] LLM training — SKIPPED (SKIP_LLM=1 or smoke mode)"
    mkdir -p "$out/llm_adapter"
    echo "skipped" > "$out/llm_adapter/STATUS"
    return
  fi

  echo "  [Phase 3] LLM training — cycle=$cycle model=$MODEL_SWEEP"

  # Convert THIS cycle's traces into SFT prompt/completion pairs.
  # Use the bench-agnostic traces_to_sft.py (reads `prompt` + `large_completion`
  # straight from the trace). The old experiments/extract_training_data.py is
  # HumanEval-coupled (looks tasks up in HumanEval.jsonl and re-queries the
  # large model) and produces nothing for tau2/SWE traces — review 2026-05-21.
  $DRY_RUN || "$PYTHON" "$REPO_ROOT/experiments/scaling/traces_to_sft.py" \
    --traces "$out/traces.jsonl" \
    --output "$out/training_data.jsonl" \
    2>&1 | tee "$out/phase3_extract.log"

  # Default to scaling_traces so the per-cycle data actually trains the model.
  # (Previously defaulted to colleague_corpus, which ignored TRAINING_DATA — the
  #  evolve loop never reached LLM SFT. Set TAU2_TRAIN_MODE=colleague_corpus to
  #  restore the old fixed-corpus behaviour.)
  TRAINING_DATA="$out/training_data.jsonl" \
  TRAIN_OUTPUT_DIR="$out/llm_adapter" \
  RUN_CONFIG="$MODEL_SWEEP" \
  BUNDLE_ROOT="$BUNDLE_ROOT" \
  MODE="${TAU2_TRAIN_MODE:-scaling_traces}" \
    bash "$REPO_ROOT/experiments/scaling/tau2_train_wrapper.sh" \
    2>&1 | tee "$out/phase3_train.log"
}

phase4_router_train() {
  local cycle=$1
  local out="$RESULTS_DIR/cycle_$cycle"
  echo "  [Phase 4] Router training — cycle=$cycle"

  # Use scaling/train_router_simple.py (bench-agnostic, reads `prompt` from
  # trace rows directly). main-branch experiments/train_learnable_router.py
  # is HumanEval-coupled and won't work on tau2 / SWE traces.
  local cmd=(
    "$PYTHON" "$REPO_ROOT/experiments/scaling/train_router_simple.py"
    --traces "$out/traces.jsonl"
    --output-dir "$out/router"
  )
  $DRY_RUN && { echo "  DRY: ${cmd[*]}"; return; }
  "${cmd[@]}" 2>&1 | tee "$out/phase4_train.log"

  # Threshold tuning: simple-router defaults to 0.5. The dedicated tuner in
  # main branch is also HumanEval-coupled. For scaling we leave it at 0.5;
  # threshold sweeps can be done post-hoc against the saved router.
  echo '{"threshold": 0.5, "tuner": "skipped (uses scaling/train_router_simple default)"}' > "$out/router_threshold.json"
}

phase5_e2e_ablation() {
  local cycle=$1
  local out="$RESULTS_DIR/cycle_$cycle"
  echo "  [Phase 5] E2E ablation — cycle=$cycle"

  local thresh=0.5
  if [[ -f "$out/router_threshold.json" ]]; then
    thresh=$($PYTHON -c "import json; print(json.load(open('$out/router_threshold.json')).get('threshold', 0.5))" 2>/dev/null || echo 0.5)
  fi

  # Use scaling/run_e2e_ablation_simple.py (bench-agnostic).
  # main-branch run_e2e_ablation.py is HumanEval-coupled and won't work on tau2 traces.
  local cmd=(
    "$PYTHON" "$REPO_ROOT/experiments/scaling/run_e2e_ablation_simple.py"
    --traces "$out/traces.jsonl"
    --skillbook "$out/skillbook.json"
    --router-dir "$out/router"
    --router-threshold "$thresh"
    --output "$out/e2e_ablation_summary.json"
    --markdown-output "$out/e2e_ablation_summary.md"
  )
  $DRY_RUN && { echo "  DRY: ${cmd[*]}"; return; }
  "${cmd[@]}" 2>&1 | tee "$out/phase5.log"

  # Inject metadata so aggregate_cycles can plot it correctly
  $DRY_RUN || "$PYTHON" - <<PY
import json
from pathlib import Path
p = Path("$out/e2e_ablation_summary.json")
if p.exists():
    d = json.loads(p.read_text())
    d.update({"cycle": $cycle, "bench": "$BENCH",
              "model_config": "$MODEL_SWEEP", "schedule": "$SCHEDULE"})
    p.write_text(json.dumps(d, indent=2))
PY
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Cycle orchestration with schedule support
# ─────────────────────────────────────────────────────────────────────────────

run_cycle() {
  local cycle=$1
  echo "═══ Cycle $cycle / $((N_CYCLES-1))  schedule=$SCHEDULE ═══"
  mkdir -p "$RESULTS_DIR/cycle_$cycle"

  phase1_collect_traces "$cycle"

  for stage in $(echo "$SCHEDULE" | grep -o .); do
    case "$stage" in
      S) phase2_skills_evolve "$cycle" ;;
      L) phase3_llm_train     "$cycle" ;;
      R) phase4_router_train  "$cycle" ;;
      *) echo "[FATAL] Unknown schedule stage: $stage"; exit 4 ;;
    esac
  done

  phase5_e2e_ablation "$cycle"
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. Final aggregation
# ─────────────────────────────────────────────────────────────────────────────

aggregate() {
  echo "═══ Aggregation ═══"
  $DRY_RUN && { echo "  DRY: would aggregate cycles 0..$((N_CYCLES-1))"; return; }

  "$PYTHON" "$REPO_ROOT/experiments/scaling/aggregate_cycles.py" \
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
