#!/usr/bin/env bash
# Watchdog for unattended real tau2 scaling runs.
#
# It resumes the same EXPERIMENT_NAME until one of the stop conditions is true:
#   - traces.jsonl has N_TASKS rows;
#   - e2e_ablation_summary.json exists;
#   - accumulated trace cost reaches SCALING_MAX_COST_USD;
#   - the same run command exits MAX_RESTARTS times.
#
# Required env:
#   EXPERIMENT_NAME
#
# Typical GPU usage:
#   EXPERIMENT_NAME=real_tau2_30_... \
#   BUNDLE_ENV=/path/to/.env \
#   PYTHON=/path/to/.venv/bin/python \
#   OPENAI_API_BASE=http://127.0.0.1:18082/v1 \
#   SCALING_MAX_COST_USD=2 \
#   SCALING_TASK_TIMEOUT_S=1800 \
#   SCALING_MAX_ZERO_COST_FAILURES=3 \
#   setsid -f bash scaling/watch_real_tau2.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${EXPERIMENT_NAME:?EXPERIMENT_NAME is required}"
: "${N_TASKS:=30}"
: "${N_CYCLES:=1}"
: "${WATCH_INTERVAL_S:=60}"
: "${MAX_RESTARTS:=6}"
: "${BUNDLE_ROOT:=$REPO_ROOT/experiments/tau2_stage2}"
: "${BUNDLE_ENV:=$BUNDLE_ROOT/.env}"
: "${OPENAI_API_BASE:=http://127.0.0.1:18082/v1}"
: "${OPENAI_BASE_URL:=$OPENAI_API_BASE}"
: "${SCALING_MAX_COST_USD:=2}"
: "${SCALING_TASK_TIMEOUT_S:=1800}"
: "${SCALING_MAX_ZERO_COST_FAILURES:=3}"
: "${TAU2_MAX_STEPS:=30}"
: "${TAU2_DOMAIN:=retail}"

RESULTS_DIR="$REPO_ROOT/results/$EXPERIMENT_NAME"
TRACE_FILE="$RESULTS_DIR/cycle_0/traces.jsonl"
SUMMARY_FILE="$RESULTS_DIR/cycle_0/e2e_ablation_summary.json"
WATCH_LOG="$RESULTS_DIR/watchdog.log"
mkdir -p "$RESULTS_DIR"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$WATCH_LOG"
}

trace_metric() {
  local field=$1
  "$PYTHON" - "$TRACE_FILE" "$field" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
field = sys.argv[2]
rows = []
if path.exists():
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
if field == "rows":
    print(len(rows))
elif field == "cost":
    print(sum(float(row.get("total_cost") or 0.0) for row in rows))
elif field == "success":
    print(sum(bool(row.get("final_success")) for row in rows))
else:
    raise SystemExit(f"unknown metric: {field}")
PY
}

is_running() {
  pgrep -af "$EXPERIMENT_NAME|collect_traces.py --bench tau2_bench --n-tasks $N_TASKS" >/dev/null 2>&1
}

stop_running() {
  pkill -f "$EXPERIMENT_NAME|collect_traces.py --bench tau2_bench --n-tasks $N_TASKS" >/dev/null 2>&1 || true
}

if [[ -f "$BUNDLE_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$BUNDLE_ENV"
  set +a
fi

export OPENAI_API_BASE OPENAI_BASE_URL
export COMMONSTACK_API_KEY="${COMMONSTACK_API_KEY:-${OPENAI_API_KEY:-}}"
export COMMONSTACK_BASE_URL="${COMMONSTACK_BASE_URL:-$OPENAI_API_BASE}"
export PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/experiments/tau2_stage2/code:${PYTHONPATH:-}"
export BUNDLE_ROOT TAU2_MAX_STEPS TAU2_DOMAIN
export SCALING_MAX_COST_USD SCALING_TASK_TIMEOUT_S SCALING_MAX_ZERO_COST_FAILURES

restarts=0
log "watchdog start experiment=$EXPERIMENT_NAME n_tasks=$N_TASKS cost_cap=$SCALING_MAX_COST_USD"

while true; do
  rows=$(trace_metric rows)
  cost=$(trace_metric cost)
  success=$(trace_metric success)
  log "status rows=$rows/$N_TASKS success=$success cost=$cost running=$(is_running && echo yes || echo no)"

  if [[ -f "$SUMMARY_FILE" ]]; then
    log "complete: found $SUMMARY_FILE"
    exit 0
  fi
  if (( rows >= N_TASKS )); then
    log "phase1 complete: rows=$rows; waiting for pipeline aggregation if process is still running"
    if ! is_running; then
      log "rows complete and no process; exiting"
      exit 0
    fi
  fi
  if "$PYTHON" - "$cost" "$SCALING_MAX_COST_USD" <<'PY'
import sys
cost = float(sys.argv[1])
cap = float(sys.argv[2])
raise SystemExit(0 if cost >= cap else 1)
PY
  then
    log "stop: cost cap reached cost=$cost cap=$SCALING_MAX_COST_USD"
    stop_running
    exit 0
  fi

  if is_running; then
    sleep "$WATCH_INTERVAL_S"
    continue
  fi

  if (( restarts >= MAX_RESTARTS )); then
    log "stop: max restarts reached ($MAX_RESTARTS)"
    exit 2
  fi
  restarts=$((restarts + 1))
  log "starting/resuming run restart=$restarts"
  bash "$REPO_ROOT/scaling/run_full_pipeline.sh" \
    --n-tasks "$N_TASKS" \
    --n-cycles "$N_CYCLES" \
    --skip-llm \
    >> "$RESULTS_DIR/run.log" 2>&1 &
  sleep "$WATCH_INTERVAL_S"
done
