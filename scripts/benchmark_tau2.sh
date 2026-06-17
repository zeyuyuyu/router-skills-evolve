#!/usr/bin/env bash
# benchmark_tau2.sh
#
# Benchmark a base model on ONE tau-2 domain — NO training. Serves the model
# with vLLM, runs it as the tau-2 agent against the gpt-5.2 user simulator + NL
# judge, and reports the pass rate. This is the cycle-0 baseline that the
# scaling pipeline (run_full_pipeline.sh) would otherwise only reach after
# SFT/GRPO.
#
# tau-2 skills differ per category, so this runs ONE domain per invocation
# (TAU2_DOMAIN). Run it once per domain you care about.
#
# Quick wiring smoke (no GPU / no API key — synthetic rollouts):
#   bash scripts/benchmark_tau2.sh --mock
#
# Real benchmark (1 GPU+ ; OPENAI_API_KEY + CommonStack base for gpt-5.2):
#   OPENAI_API_KEY=... OPENAI_API_BASE=<commonstack> \
#     bash scripts/benchmark_tau2.sh
#
# Config (env vars; all optional):
#   MODEL           HF id or local dir of the agent under test
#                   default: Qwen/Qwen3.5-4B   ("qwen-3-4B")
#   TAU2_DOMAIN     retail | airline | telecom            default: retail
#   N_TASKS         tasks from the eval split             default: 9999 (all)
#   TAU2_USER_MODEL customer-side simulator + NL judge    default: openai/openai/gpt-5.2
#   LARGE_MODEL     oracle/fallback agent (kept = served policy so a base
#                   benchmark never bills a second agent)  default: the served model
#   TP_SIZE         tensor-parallel size for vLLM         default: 1
#   TAU2_LOCAL_GPU  lowest GPU index for the vLLM group   default: 0
#   PORT            vLLM port                             default: 8050
#   EXPERIMENT_NAME results subdir            default: bench_tau2_<domain>_<ts>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${BUNDLE_ROOT:=$REPO_ROOT/tau2_stage2}"

MOCK=false
for a in "$@"; do case "$a" in --mock) MOCK=true ;; -h|--help) sed -n '1,40p' "$0"; exit 0 ;; esac; done

: "${MODEL:=Qwen/Qwen3.5-4B}"
: "${TAU2_DOMAIN:=retail}"
: "${N_TASKS:=9999}"
: "${TAU2_USER_MODEL:=openai/openai/gpt-5.2}"
: "${PORT:=8050}"
: "${TP_SIZE:=1}"
: "${TAU2_LOCAL_GPU:=0}"
: "${TAU2_LOCAL_SERVED_MODEL:=evol-llm-student}"
: "${EXPERIMENT_NAME:=bench_tau2_${TAU2_DOMAIN}_$(date -u +%Y%m%d_%H%M%S)}"

# One domain per run (skills are per-category) — pin TAU2_DOMAINS to the single
# domain so the adapter never silently fans out across categories.
export TAU2_DOMAIN
export TAU2_DOMAINS="$TAU2_DOMAIN"
export TAU2_USER_MODEL
# Benchmark measures pass@1: a task where the agent produces nothing is a
# FAILURE to be counted, not a poisoned SFT trace to abort on. (The strict
# abort in collect_traces is for trace-collection runs feeding SFT.)
export SCALING_ALLOW_EMPTY_TRACE_FAILURES=1

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if   [[ -x "$REPO_ROOT/venv/bin/python" ]]; then PYTHON="$REPO_ROOT/venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then PYTHON=python3
  else PYTHON=python; fi
fi

SERVED="$TAU2_LOCAL_SERVED_MODEL"
SMALL_MODEL="openai/$SERVED"
: "${LARGE_MODEL:=$SMALL_MODEL}"   # default: don't bill a second (gpt-5.x) agent

OUT="$REPO_ROOT/results/$EXPERIMENT_NAME"
mkdir -p "$OUT"

echo "═══ tau-2 benchmark (no training) ═══"
echo "  MODEL        = $MODEL"
echo "  DOMAIN       = $TAU2_DOMAIN   (N_TASKS=$N_TASKS, eval split)"
echo "  AGENT        = $SMALL_MODEL   (served via vLLM)"
echo "  USER/JUDGE   = $TAU2_USER_MODEL"
echo "  LARGE/oracle = $LARGE_MODEL"
echo "  OUT          = $OUT"
echo "  MOCK         = $MOCK"
echo

run_collect() {
  local extra=("$@")
  "$PYTHON" "$REPO_ROOT/src/pipeline/collect_traces.py" \
    --bench tau2_bench \
    --n-tasks "$N_TASKS" \
    --small-model "$1" \
    --large-model "$LARGE_MODEL" \
    --split eval \
    --out "$OUT/traces.jsonl" \
    "${extra[@]:1}"
}

summarize() {
  "$PYTHON" - "$OUT/traces.jsonl" "$TAU2_DOMAIN" "$MODEL" <<'PY' | tee "$OUT/benchmark_summary.txt"
import json, sys
from pathlib import Path
traces, domain, model = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in traces.read_text().splitlines() if l.strip()]
rows = [r for r in rows if not r.get("error")]
n = len(rows)
small_ok = sum(1 for r in rows if r.get("small_success"))
cost = sum(float(r.get("small_cost") or 0.0) for r in rows)
print(f"tau-2 benchmark — domain={domain}  model={model}")
print(f"  tasks scored : {n}")
if n:
    print(f"  PASS@1 (agent): {small_ok}/{n} = {100*small_ok/n:.1f}%")
    print(f"  mean agent cost/task: ${cost/n:.4f}")
summ = {"domain": domain, "model": model, "n": n, "pass": small_ok,
        "pass_rate": (small_ok / n) if n else None}
Path(sys.argv[1]).with_name("benchmark_summary.json").write_text(json.dumps(summ, indent=2))
PY
}

if $MOCK; then
  export SCALING_MOCK=1
  echo "[mock] synthetic rollouts; skipping vLLM + gpt-5.2."
  run_collect "$SMALL_MODEL" --mock 2>&1 | tee "$OUT/collect.log"
  summarize
  echo "DONE (mock). Summary: $OUT/benchmark_summary.txt"
  exit 0
fi

[[ -z "${OPENAI_API_KEY:-}" ]] && { echo "[FATAL] OPENAI_API_KEY not set (needed for gpt-5.2 user sim + judge). Use --mock to validate wiring."; exit 3; }
[[ -d "$BUNDLE_ROOT/code/vendor/tau2-bench" ]] || {
  echo "[FATAL] tau2-bench vendor missing. Run first:"
  echo "          bash $BUNDLE_ROOT/code/training/orchestration/setup_env_server.sh"
  exit 3; }

# Resolve a HF model id to its local snapshot dir (vllm_serve.sh wants a dir).
if [[ -d "$MODEL" ]]; then
  MODEL_DIR="$MODEL"
else
  echo "  Resolving $MODEL to local snapshot…"
  MODEL_DIR="$("$PYTHON" - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1]))
PY
)"
  echo "  -> $MODEL_DIR"
fi

stop_vllm() {
  local pid_file="$MODEL_DIR/vllm_serve.pid"
  [[ -f "$pid_file" ]] || return 0
  local pid; pid=$(sed -n 's/.*PID=\([0-9][0-9]*\).*/\1/p' "$pid_file" | head -1)
  [[ -n "$pid" ]] && { kill "$pid" 2>/dev/null || true; sleep 2; kill -9 "$pid" 2>/dev/null || true; }
  rm -f "$pid_file"
}
trap stop_vllm EXIT

echo "  Serving $MODEL_DIR on port $PORT (TP=$TP_SIZE, GPU $TAU2_LOCAL_GPU)…"
stop_vllm
TP_SIZE="$TP_SIZE" NUM_GPUS="${NUM_GPUS:-8}" \
  bash "$BUNDLE_ROOT/code/training/eval/vllm_serve.sh" "$MODEL_DIR" "$PORT" "$TAU2_LOCAL_GPU" \
  2>&1 | tee "$OUT/vllm_start.log" &
VLLM_START_PID=$!
for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  if ! kill -0 "$VLLM_START_PID" 2>/dev/null; then
    wait "$VLLM_START_PID" || true
    echo "[FATAL] vLLM exited before ready; see $OUT/vllm_start.log"; exit 1
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || { echo "[FATAL] vLLM not ready on $PORT"; exit 1; }

TAU2_LOCAL_API_BASE="http://127.0.0.1:$PORT/v1" \
  TAU2_LOCAL_API_KEY="${TAU2_LOCAL_API_KEY:-EMPTY}" \
  TAU2_LOCAL_SERVED_MODEL="$SERVED" \
  run_collect "$SMALL_MODEL" 2>&1 | tee "$OUT/collect.log"

summarize
echo
echo "DONE. Summary: $OUT/benchmark_summary.txt   traces: $OUT/traces.jsonl"
