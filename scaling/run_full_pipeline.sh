#!/usr/bin/env bash
# run_full_pipeline.sh
#
# End-to-end: Skills evolve + Router training + LLM training + E2E ablation,
# multi-cycle iteration. Default bench: tau2-bench. SWE-Bench adapter is a stub.
#
# Quick smoke (no GPU / no API key required — uses mock adapter):
#   bash scaling/run_full_pipeline.sh --smoke --mock
#
# Real run (8× high-memory GPUs; OPENAI_API_KEY + HF_TOKEN required):
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
#   DISTILLER_MODEL   LLM used to distill skill procedures (Phase 2).
#                     default: deepseek/deepseek-v3.2  (cheap, good enough)
#                     set to "" or "heuristic" to skip LLM distillation
#   TAU2_DOMAIN       airline | retail | telecom (default: retail)
#   TAU2_DOMAINS      comma-separated domains; overrides TAU2_DOMAIN for tau2
#                     task loading (e.g. retail,telecom,airline)
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
: "${DISTILLER_MODEL:=deepseek/deepseek-v3.2}"
: "${TAU2_DOMAIN:=retail}"
: "${TAU2_DOMAINS:=$TAU2_DOMAIN}"
: "${SKIP_LLM:=0}"
: "${RUN_HELDOUT_EVAL:=0}"
: "${SCALING_NUM_TRAIN_EPOCHS:=2}"
: "${GRPO_N_GENERATIONS:=8}"
: "${GRPO_EPOCHS:=1}"
: "${GRPO_LR:=5e-6}"
: "${GRPO_BETA:=0.04}"
: "${GRPO_ALGO:=grpo}"       # grpo | dapo  (DAPO = dynamic-sampling + clip-higher + token-loss)
: "${DAPO_CLIP_LOW:=0.2}"    # PPO ε lower bound (both algos)
: "${DAPO_CLIP_HIGH:=0.5}"   # DAPO ε upper bound for positive-advantage tokens

export TAU2_DOMAIN
export TAU2_DOMAINS
export SCALING_NUM_TRAIN_EPOCHS

# HumanEval bench: local code models + pytest, no remote agent API.
# Phase 3a uses train_small_model.py (bench-agnostic SFT) instead of the
# tau2-specific wrapper. Phase 3b runs GRPO with test-execution reward.
if [[ "$BENCH" == "humaneval" ]]; then
  SMALL_MODEL="${HE_SMALL_MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
  LARGE_MODEL="${HE_LARGE_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
  SKIP_LLM=0     # use bench-agnostic SFT path inside phase3_llm_train
  : "${SKIP_GRPO:=0}"   # GRPO enabled for HumanEval (has test executor)
else
  : "${SKIP_GRPO:=1}"   # GRPO disabled for tau2/SWE (no test executor yet)
fi

if $SMOKE; then
  MODEL_SWEEP="smoke_2b"
  N_CYCLES=1
  N_TASKS=30
  SKIP_LLM=1   # smoke skips LLM training by default
elif [[ "$BENCH" == "humaneval" ]]; then
  N_TASKS=82   # HumanEval train split (even indices of the 164 tasks)
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
  echo "  SMALL_MODEL     = $SMALL_MODEL"
  echo "  LARGE_MODEL     = $LARGE_MODEL"
  echo "  DISTILLER_MODEL = ${DISTILLER_MODEL:-heuristic (no LLM)}"
  echo "  SKIP_GRPO       = $SKIP_GRPO  (algo=$GRPO_ALGO K=$GRPO_N_GENERATIONS epochs=$GRPO_EPOCHS lr=$GRPO_LR  clip_low=$DAPO_CLIP_LOW clip_high=$DAPO_CLIP_HIGH)"
  echo "  TAU2_DOMAIN     = $TAU2_DOMAIN"
  echo "  TAU2_DOMAINS    = $TAU2_DOMAINS"
  echo "  RUN_HELDOUT_EVAL= $RUN_HELDOUT_EVAL"
  echo "  SFT_EPOCHS      = $SCALING_NUM_TRAIN_EPOCHS"
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
  if [[ "$PYTHON" == */* && "$PYTHON" != /* ]]; then
    PYTHON="$REPO_ROOT/$PYTHON"
  fi
  export PYTHON
  echo "  PYTHON          = $PYTHON ($($PYTHON --version 2>&1))"



  # Secret checks (skipped in mock mode). HumanEval uses LOCAL models + pytest,
  # so it needs no OPENAI_API_KEY (only HF for model download).
  if ! $MOCK && ! $DRY_RUN && [[ "$BENCH" != "humaneval" ]]; then
    [[ -z "${OPENAI_API_KEY:-}" ]] && { echo "[FATAL] OPENAI_API_KEY not set (or run with --mock)"; exit 3; }
    [[ -z "${HF_TOKEN:-}" ]] && echo "[WARN] HF_TOKEN not set — large model download will be slow"
  fi

  # Disk
  free_gb=$(df -BG "$REPO_ROOT" 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G' || echo 999)
  if ! $MOCK && ! $DRY_RUN && (( free_gb < 100 )); then
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

  if [[ "$BENCH" == "tau2_bench" ]]; then
    "$PYTHON" - "$REPO_ROOT" "$BENCH" "$N_TASKS" <<'PY'
import sys
from collections import Counter

repo, bench, n_tasks = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path.insert(0, repo)
from experiments.scaling.benches import load_adapter  # noqa: E402

adapter = load_adapter(bench)
train = adapter.load_tasks(n_tasks, split="train")
eval_ = adapter.load_tasks(n_tasks, split="eval")
train_ids = {str(t.get("task_id")) for t in train}
eval_ids = {str(t.get("task_id")) for t in eval_}
overlap = train_ids & eval_ids
train_domains = Counter(str(t.get("domain", "unknown")) for t in train)
eval_domains = Counter(str(t.get("domain", "unknown")) for t in eval_)
print(f"  tau2 train tasks: {len(train)}  domains={dict(train_domains)}")
print(f"  tau2 eval tasks:  {len(eval_)}  domains={dict(eval_domains)}")
print(f"  tau2 train/eval overlap: {len(overlap)}")
if overlap:
    raise SystemExit("[FATAL] tau2 train/eval task_id overlap detected")
PY
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
  "tau2_domains": "$TAU2_DOMAINS",
  "small_model": "$SMALL_MODEL",
  "large_model": "$LARGE_MODEL",
  "distiller_model": "${DISTILLER_MODEL:-heuristic}",
  "skip_grpo": $([ "$SKIP_GRPO" -eq 1 ] && echo true || echo false),
  "grpo_algo": "$GRPO_ALGO",
  "grpo_n_generations": $GRPO_N_GENERATIONS,
  "grpo_epochs": $GRPO_EPOCHS,
  "grpo_lr": "$GRPO_LR",
  "dapo_clip_low": "$DAPO_CLIP_LOW",
  "dapo_clip_high": "$DAPO_CLIP_HIGH",
  "smoke": $SMOKE,
  "mock": $MOCK,
  "skip_llm": $([ "$SKIP_LLM" -eq 1 ] && echo true || echo false),
  "run_heldout_eval": $([ "$RUN_HELDOUT_EVAL" = "1" ] && echo true || echo false),
  "scaling_num_train_epochs": "$SCALING_NUM_TRAIN_EPOCHS",
  "resume_from": $RESUME_FROM,
  "git_sha": "$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)",
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Phase implementations
# ─────────────────────────────────────────────────────────────────────────────

stop_local_vllm() {
  local ckpt="$1"
  local pid_file="$ckpt/vllm_serve.pid"
  [[ -f "$pid_file" ]] || return 0
  local pid
  pid=$(sed -n 's/.*PID=\([0-9][0-9]*\).*/\1/p' "$pid_file" | head -1)
  if [[ -n "$pid" ]]; then
    kill "$pid" 2>/dev/null || true
    sleep 2
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$pid_file"
}

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
  local local_student_ckpt="" local_student_port="" local_student_served=""
  if (( cycle > 0 )); then
    # Prefer GRPO adapter (stronger) over SFT checkpoint when both exist.
    local prev_grpo="$RESULTS_DIR/cycle_$((cycle-1))/grpo_adapter"
    local prev_sft="$RESULTS_DIR/cycle_$((cycle-1))/llm_adapter/checkpoint-best"
    local prev=""
    if   [[ -d "$prev_grpo" ]]; then prev="$prev_grpo"
    elif [[ -d "$prev_sft"  ]]; then prev="$prev_sft"
    fi
    if [[ -n "$prev" ]]; then
      local_student_ckpt="$prev"
      local_student_port="${TAU2_LOCAL_PORT:-8050}"
      local_student_served="${TAU2_LOCAL_SERVED_MODEL:-evol-llm-student}"
      small_arg="openai/$local_student_served"
    fi
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
    --split train
    --out "$out/traces.jsonl"
  )
  [[ -n "$prev_router"    && -e "$prev_router"    ]] && cmd+=(--router "$prev_router")
  [[ -n "$prev_skillbook" && -e "$prev_skillbook" ]] && cmd+=(--skillbook "$prev_skillbook")
  [[ "${SCALING_TRACE_RESUME:-0}" == "1" ]] && cmd+=(--resume)
  [[ "${SCALING_FORCE_BOTH:-0}" == "1" ]] && cmd+=(--force-both)
  $MOCK && cmd+=(--mock)
  $DRY_RUN && { echo "  DRY: ${cmd[*]}"; return; }

  if [[ "${SCALING_TRACE_RESUME:-0}" == "1" && -s "$out/traces.jsonl" ]]; then
    if "$PYTHON" - "$REPO_ROOT" "$BENCH" "$N_TASKS" "$out/traces.jsonl" <<'PY'
import json
import sys
from pathlib import Path

repo, bench, n_tasks, traces = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
sys.path.insert(0, repo)
from experiments.scaling.benches import load_adapter  # noqa: E402


def task_key(row):
    return str(row.get("task_id") or row.get("id") or "")


expected = {tid for tid in (task_key(t) for t in load_adapter(bench).load_tasks(n_tasks, split="train")) if tid}
seen = set()
with traces.open() as fh:
    for line in fh:
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = task_key(row)
        if task_id:
            seen.add(task_id)
missing = expected - seen
if expected and not missing:
    print(f"  [Phase 1] Trace collection already complete ({len(seen)}/{len(expected)}); skipping.")
    raise SystemExit(0)
print(f"  [Phase 1] Trace resume incomplete ({len(seen)}/{len(expected)}; missing={len(missing)}); continuing.")
raise SystemExit(1)
PY
    then
      return 0
    fi
  fi

  local rc=0
  if [[ -n "$local_student_ckpt" && "$MOCK" != "true" ]]; then
    echo "  [Phase 1] Starting local student vLLM from $local_student_ckpt on port $local_student_port"
    stop_local_vllm "$local_student_ckpt"
    rm -f "$out/phase1_vllm_start.log"
    TP_SIZE="${TAU2_LOCAL_TP_SIZE:-8}" \
      NUM_GPUS="${NUM_GPUS:-8}" \
      bash "$BUNDLE_ROOT/code/training/eval/vllm_serve.sh" "$local_student_ckpt" "$local_student_port" "${TAU2_LOCAL_GPU:-0}" \
      2>&1 | tee "$out/phase1_vllm_start.log" &
    local vllm_start_pid=$!
    for _ in $(seq 1 120); do
      if curl -sf "http://127.0.0.1:$local_student_port/v1/models" >/dev/null 2>&1; then
        break
      fi
      if ! kill -0 "$vllm_start_pid" 2>/dev/null; then
        wait "$vllm_start_pid" || true
        echo "[FATAL] local student vLLM starter exited before ready; see $out/phase1_vllm_start.log" >&2
        return 1
      fi
      sleep 5
    done
    if ! curl -sf "http://127.0.0.1:$local_student_port/v1/models" >/dev/null 2>&1; then
      echo "[FATAL] local student vLLM failed to become ready on port $local_student_port" >&2
      kill "$vllm_start_pid" 2>/dev/null || true
      stop_local_vllm "$local_student_ckpt"
      return 1
    fi
    TAU2_LOCAL_API_BASE="http://127.0.0.1:$local_student_port/v1" \
      TAU2_LOCAL_API_KEY="${TAU2_LOCAL_API_KEY:-EMPTY}" \
      TAU2_LOCAL_SERVED_MODEL="$local_student_served" \
      "${cmd[@]}" 2>&1 | tee "$out/phase1.log" || rc=${PIPESTATUS[0]}
    stop_local_vllm "$local_student_ckpt"
    kill "$vllm_start_pid" 2>/dev/null || true
    wait "$vllm_start_pid" 2>/dev/null || true
    return "$rc"
  fi

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
        if not prompt:
            continue
        tid = t.get("task_id", "")
        # tofix.md #2 + #4: build the REAL capability table from BOTH oracle
        # outcomes, keyed by CANONICAL ROLES ("small"/"large") rather than raw
        # model ids / adapter paths (which change each cycle and break
        # can_downgrade_to_small). The policy/deployment outcome stays in the
        # trace as metadata; stats record true small-vs-large capability per
        # signature. Exemplars attach to whichever role succeeded.
        sm = t.get("small_success")
        lg = t.get("large_success")
        if isinstance(sm, bool):
            sb.update(prompt, "small", sm, tid,
                      completion=(t.get("small_completion", "") if sm else ""))
        # large_skipped => large outcome is a placeholder, not real; skip it.
        if isinstance(lg, bool) and not t.get("large_skipped", False):
            sb.update(prompt, "large", lg, tid,
                      completion=(t.get("large_completion", "") if lg else ""))
        if isinstance(sm, bool) or isinstance(lg, bool):
            n += 1

# Distill a reusable procedure per cluster from the accumulated exemplars.
# If DISTILLER_MODEL is set (and not "heuristic"), use the LLM distiller;
# otherwise fall back to the no-API heuristic.
distiller_model = "$DISTILLER_MODEL"
if distiller_model and distiller_model != "heuristic":
    from src.skills import make_llm_distiller
    distiller = make_llm_distiller(distiller_model)
    distiller_tag = distiller_model
else:
    distiller = None
    distiller_tag = "heuristic"

n_proc = sb.distill_all(distiller=distiller)

out_path = Path("$out/skillbook.json")
sb.save(out_path)
size = len(getattr(sb, "skills", {}))
print(f"[skills_evolve] ingested {n} traces  SkillBook size={size}  "
      f"procedures_distilled={n_proc}  distiller={distiller_tag}  wrote {out_path}")
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

  echo "  [Phase 3] LLM SFT — cycle=$cycle"

  SKILLBOOK_ARG=""
  [[ -f "$out/skillbook.json" ]] && SKILLBOOK_ARG="--skillbook $out/skillbook.json"
  $DRY_RUN || "$PYTHON" "$REPO_ROOT/experiments/scaling/traces_to_sft.py" \
    --traces "$out/traces.jsonl" \
    --output "$out/training_data.jsonl" \
    $SKILLBOOK_ARG \
    2>&1 | tee "$out/phase3_extract.log"

  if [[ "$BENCH" == "humaneval" ]]; then
    # HumanEval: bench-agnostic SFT via train_small_model.py.
    # Warm-start from previous GRPO or SFT checkpoint if available.
    local warmstart_model="$SMALL_MODEL"
    local prev_grpo="$RESULTS_DIR/cycle_$((cycle-1))/grpo_adapter"
    local prev_sft="$RESULTS_DIR/cycle_$((cycle-1))/llm_adapter/checkpoint-best"
    if   (( cycle > 0 )) && [[ -d "$prev_grpo" ]]; then warmstart_model="$prev_grpo"
    elif (( cycle > 0 )) && [[ -d "$prev_sft"  ]]; then warmstart_model="$prev_sft"
    fi
    $DRY_RUN || "$PYTHON" "$REPO_ROOT/experiments/train_small_model.py" \
      --data "$out/training_data.jsonl" \
      --base-model "$warmstart_model" \
      --output "$out/llm_adapter" \
      --prompt-style "${HE_PROMPT_STYLE:-qwen-chat}" \
      --epochs "$SCALING_NUM_TRAIN_EPOCHS" \
      2>&1 | tee "$out/phase3_train.log"
  else
    # Tau2 / SWE: tau2-specific FSDP2 + FA2 training wrapper.
    TRAINING_DATA="$out/training_data.jsonl" \
    TRAIN_OUTPUT_DIR="$out/llm_adapter" \
    RUN_CONFIG="$MODEL_SWEEP" \
    BUNDLE_ROOT="$BUNDLE_ROOT" \
    MODE="${TAU2_TRAIN_MODE:-scaling_traces}" \
      bash "$REPO_ROOT/experiments/scaling/tau2_train_wrapper.sh" \
      2>&1 | tee "$out/phase3_train.log"
  fi
}

phase3b_grpo_train() {
  local cycle=$1
  local out="$RESULTS_DIR/cycle_$cycle"

  if [[ "$SKIP_GRPO" -eq 1 ]]; then
    echo "  [Phase 3b] GRPO — SKIPPED (SKIP_GRPO=1 or non-HumanEval bench)"
    return
  fi

  echo "  [Phase 3b] GRPO — cycle=$cycle algo=$GRPO_ALGO K=$GRPO_N_GENERATIONS epochs=$GRPO_EPOCHS lr=$GRPO_LR"

  # Warm-start from Phase 3a SFT checkpoint if available.
  local grpo_base="$SMALL_MODEL"
  local sft_ckpt="$out/llm_adapter/checkpoint-best"
  [[ -d "$sft_ckpt" ]] && grpo_base="$sft_ckpt"

  SKILLBOOK_ARG=""
  [[ -f "$out/skillbook.json" ]] && SKILLBOOK_ARG="--skillbook $out/skillbook.json"

  $DRY_RUN && { echo "  DRY: grpo_train --algo $GRPO_ALGO --model $grpo_base --output-dir $out/grpo_adapter K=$GRPO_N_GENERATIONS"; return; }
  GRPO_N_GENERATIONS="$GRPO_N_GENERATIONS" \
  GRPO_EPOCHS="$GRPO_EPOCHS" \
  GRPO_LR="$GRPO_LR" \
  GRPO_BETA="$GRPO_BETA" \
  GRPO_ALGO="$GRPO_ALGO" \
  DAPO_CLIP_LOW="$DAPO_CLIP_LOW" \
  DAPO_CLIP_HIGH="$DAPO_CLIP_HIGH" \
    "$PYTHON" "$REPO_ROOT/experiments/scaling/grpo_train_simple.py" \
    --model "$grpo_base" \
    --bench-data "${HE_DATA:-$REPO_ROOT/data/HumanEval.jsonl}" \
    --output-dir "$out/grpo_adapter" \
    --algo "$GRPO_ALGO" \
    --n-generations "$GRPO_N_GENERATIONS" \
    --epochs "$GRPO_EPOCHS" \
    --lr "$GRPO_LR" \
    --beta "$GRPO_BETA" \
    --clip-low "$DAPO_CLIP_LOW" \
    --clip-high "$DAPO_CLIP_HIGH" \
    --prompt-style "${HE_PROMPT_STYLE:-qwen-chat}" \
    $SKILLBOOK_ARG \
    2>&1 | tee "$out/phase3b_grpo.log"
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
      L) phase3_llm_train "$cycle"; phase3b_grpo_train "$cycle" ;;
      R) phase4_router_train  "$cycle" ;;
      *) echo "[FATAL] Unknown schedule stage: $stage"; exit 4 ;;
    esac
  done

  phase5_e2e_ablation "$cycle"
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. Held-out eval
# ─────────────────────────────────────────────────────────────────────────────

phase6_heldout_eval() {
  [[ "$RUN_HELDOUT_EVAL" == "1" ]] || { echo "═══ Held-out eval skipped (RUN_HELDOUT_EVAL=0) ═══"; return; }

  local cycle=$((N_CYCLES-1))
  local out="$RESULTS_DIR/heldout_eval"
  mkdir -p "$out"
  echo "═══ Held-out eval — split=eval cycle=$cycle ═══"

  # Prefer GRPO adapter over SFT checkpoint.
  local final_ckpt="$RESULTS_DIR/cycle_$cycle/grpo_adapter"
  [[ ! -d "$final_ckpt" ]] && final_ckpt="$RESULTS_DIR/cycle_$cycle/llm_adapter/checkpoint-best"
  local final_router="$RESULTS_DIR/cycle_$cycle/router/router.joblib"
  local final_skillbook="$RESULTS_DIR/cycle_$cycle/skillbook.json"
  local small_arg="$SMALL_MODEL"
  local local_student_ckpt="" local_student_port="" local_student_served=""
  if [[ -d "$final_ckpt" ]]; then
    local_student_ckpt="$final_ckpt"
    local_student_port="${TAU2_LOCAL_PORT:-8050}"
    local_student_served="${TAU2_LOCAL_SERVED_MODEL:-evol-llm-student}"
    small_arg="openai/$local_student_served"
  fi

  local cmd=(
    "$PYTHON" "$REPO_ROOT/experiments/scaling/collect_traces.py"
    --bench "$BENCH"
    --n-tasks "$N_TASKS"
    --small-model "$small_arg"
    --large-model "$LARGE_MODEL"
    --cycle "$cycle"
    --split eval
    --out "$out/traces.jsonl"
    --force-both
  )
  [[ -e "$final_router" ]] && cmd+=(--router "$final_router")
  [[ -e "$final_skillbook" ]] && cmd+=(--skillbook "$final_skillbook")
  $MOCK && cmd+=(--mock)
  $DRY_RUN && { echo "  DRY: ${cmd[*]}"; return; }

  local rc=0
  if [[ -n "$local_student_ckpt" && "$MOCK" != "true" ]]; then
    echo "  [Held-out eval] Starting local student vLLM from $local_student_ckpt on port $local_student_port"
    stop_local_vllm "$local_student_ckpt"
    rm -f "$out/vllm_start.log"
    TP_SIZE="${TAU2_LOCAL_TP_SIZE:-8}" \
      NUM_GPUS="${NUM_GPUS:-8}" \
      bash "$BUNDLE_ROOT/code/training/eval/vllm_serve.sh" "$local_student_ckpt" "$local_student_port" "${TAU2_LOCAL_GPU:-0}" \
      2>&1 | tee "$out/vllm_start.log" &
    local vllm_start_pid=$!
    for _ in $(seq 1 120); do
      if curl -sf "http://127.0.0.1:$local_student_port/v1/models" >/dev/null 2>&1; then
        break
      fi
      if ! kill -0 "$vllm_start_pid" 2>/dev/null; then
        wait "$vllm_start_pid" || true
        echo "[FATAL] held-out vLLM starter exited before ready; see $out/vllm_start.log" >&2
        return 1
      fi
      sleep 5
    done
    if ! curl -sf "http://127.0.0.1:$local_student_port/v1/models" >/dev/null 2>&1; then
      echo "[FATAL] held-out vLLM failed to become ready on port $local_student_port" >&2
      kill "$vllm_start_pid" 2>/dev/null || true
      stop_local_vllm "$local_student_ckpt"
      return 1
    fi
    TAU2_LOCAL_API_BASE="http://127.0.0.1:$local_student_port/v1" \
      TAU2_LOCAL_API_KEY="${TAU2_LOCAL_API_KEY:-EMPTY}" \
      TAU2_LOCAL_SERVED_MODEL="$local_student_served" \
      "${cmd[@]}" 2>&1 | tee "$out/collect.log" || rc=${PIPESTATUS[0]}
    stop_local_vllm "$local_student_ckpt"
    kill "$vllm_start_pid" 2>/dev/null || true
    wait "$vllm_start_pid" 2>/dev/null || true
    [[ "$rc" -eq 0 ]] || return "$rc"
  else
    "${cmd[@]}" 2>&1 | tee "$out/collect.log"
  fi

  local thresh=0.5
  if [[ -f "$RESULTS_DIR/cycle_$cycle/router_threshold.json" ]]; then
    thresh=$($PYTHON -c "import json; print(json.load(open('$RESULTS_DIR/cycle_$cycle/router_threshold.json')).get('threshold', 0.5))" 2>/dev/null || echo 0.5)
  fi
  "$PYTHON" "$REPO_ROOT/experiments/scaling/run_e2e_ablation_simple.py" \
    --traces "$out/traces.jsonl" \
    --skillbook "$final_skillbook" \
    --router-dir "$RESULTS_DIR/cycle_$cycle/router" \
    --router-threshold "$thresh" \
    --output "$out/e2e_ablation_summary.json" \
    --markdown-output "$out/e2e_ablation_summary.md" \
    2>&1 | tee "$out/ablation.log"

  "$PYTHON" - <<PY
import json
from pathlib import Path
p = Path("$out/e2e_ablation_summary.json")
if p.exists():
    d = json.loads(p.read_text())
    d.update({"cycle": $cycle, "bench": "$BENCH", "split": "eval",
              "model_config": "$MODEL_SWEEP", "schedule": "$SCHEDULE"})
    p.write_text(json.dumps(d, indent=2))
PY
}

# ─────────────────────────────────────────────────────────────────────────────
# 6. Final aggregation
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

phase6_heldout_eval
aggregate
