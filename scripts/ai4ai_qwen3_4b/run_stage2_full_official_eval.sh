#!/usr/bin/env bash
# Official tau2 CLI eval for the AI4AI Qwen3-4B stage2-full SFT checkpoint.
#
# This script starts the SFT checkpoint on the remote 8-GPU node, exposes it as
# an OpenAI-compatible local endpoint, runs tau2 test100 with GPT-5.2 user/judge,
# and writes summary/raw JSON artifacts under results/ai4ai_4b_tau2/...
#
# Required:
#   COMMONSTACK_API_KEY or OPENAI_API_KEY
#
# Usage:
#   bash scripts/ai4ai_qwen3_4b/run_stage2_full_official_eval.sh
#   MODE=smoke bash scripts/ai4ai_qwen3_4b/run_stage2_full_official_eval.sh
#   MODE=summarize bash scripts/ai4ai_qwen3_4b/run_stage2_full_official_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_NAME="${RUN_NAME:-qwen3_4b_evolution_20260615}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/results/ai4ai_4b_tau2/$RUN_NAME}"
EVAL_DIR="${EVAL_DIR:-$RESULT_ROOT/eval_stage2_full_parserfix}"
LOG_DIR="$RESULT_ROOT/logs"

REMOTE_HOST="${REMOTE_HOST:-root@10.100.0.32}"
REMOTE_PORT="${REMOTE_PORT:-19810}"
REMOTE_REPO="${REMOTE_REPO:-/root/router-skills-evolve}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-$REMOTE_REPO/results/ai4ai_4b_tau2/$RUN_NAME/train/qwen3_4b_stage2_full_1epoch/checkpoint-final}"
SFT_SERVED_MODEL="${SFT_SERVED_MODEL:-qwen3-4b-sft}"
LOCAL_LB_PORT="${LOCAL_LB_PORT:-18160}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MODE="${MODE:-full}"

PY="${PY:-$REPO_ROOT/.venv-tau2/bin/python}"
TAU2="${TAU2:-$REPO_ROOT/.venv-tau2/bin/tau2}"
if [[ ! -x "$PY" ]]; then PY=python3; fi
if [[ ! -x "$TAU2" ]]; then TAU2=tau2; fi

if [[ -d "$REPO_ROOT/tau2_stage2/code/vendor/tau2-bench/data/simulations" ]]; then
  SIM_ROOT="$REPO_ROOT/tau2_stage2/code/vendor/tau2-bench/data/simulations"
else
  SIM_ROOT="$REPO_ROOT/experiments/tau2_stage2/code/vendor/tau2-bench/data/simulations"
fi

mkdir -p "$EVAL_DIR" "$LOG_DIR" "$RESULT_ROOT/manifests"

commonstack_key() {
  if [[ -n "${COMMONSTACK_API_KEY:-}" ]]; then
    printf '%s\n' "$COMMONSTACK_API_KEY"
  elif [[ -n "${OPENAI_API_KEY:-}" ]]; then
    printf '%s\n' "$OPENAI_API_KEY"
  else
    return 1
  fi
}

export_commonstack_env() {
  local key
  key="$(commonstack_key)" || {
    echo "Set COMMONSTACK_API_KEY or OPENAI_API_KEY before running eval." >&2
    exit 2
  }
  export OPENAI_API_KEY="$key"
  export OPENAI_API_BASE=https://api.commonstack.ai/v1
  export OPENAI_BASE_URL=https://api.commonstack.ai/v1
  export TAU2_NL_JUDGE_MODEL=openai/openai/gpt-5.2
  export TAU2_NL_JUDGE_ARGS_JSON
  TAU2_NL_JUDGE_ARGS_JSON="$("$PY" - <<'PY'
import json
import os
print(json.dumps({
    "temperature": 0.0,
    "api_base": "https://api.commonstack.ai/v1",
    "api_key": os.environ["OPENAI_API_KEY"],
    "custom_llm_provider": "openai",
}))
PY
)"
}

json_agent_args() {
  "$PY" - <<PY
import json
print(json.dumps({
    "temperature": 0.0,
    "api_base": "http://127.0.0.1:${LOCAL_LB_PORT}/v1",
    "api_key": "EMPTY",
    "custom_llm_provider": "openai",
}))
PY
}

json_user_args() {
  "$PY" - <<'PY'
import json
import os
print(json.dumps({
    "temperature": 0.0,
    "api_base": "https://api.commonstack.ai/v1",
    "api_key": os.environ["OPENAI_API_KEY"],
    "custom_llm_provider": "openai",
}))
PY
}

sync_server() {
  rsync -az -e "ssh -p $REMOTE_PORT -o BatchMode=yes -o StrictHostKeyChecking=no" \
    "$SCRIPT_DIR/hf_openai_server.py" \
    "$REMOTE_HOST:$REMOTE_REPO/scripts/ai4ai_qwen3_4b/hf_openai_server.py"
}

stop_remote_servers() {
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p "$REMOTE_PORT" "$REMOTE_HOST" "
set -euo pipefail
pgrep -f '[h]f_openai_server.py' | xargs -r /bin/kill -TERM 2>/dev/null || true
for _ in 1 2 3 4 5; do
  pgrep -f '[h]f_openai_server.py' >/dev/null || break
  sleep 1
done
pgrep -f '[h]f_openai_server.py' | xargs -r /bin/kill -KILL 2>/dev/null || true
" || true
  tmux kill-session -t ai4ai_qwen3_4b_lb 2>/dev/null || true
  pkill -f 'ssh -f -N .*1818[0-7]:127.0.0.1:1808[0-7]' 2>/dev/null || true
}

remote_python_expr='if [ -x tau2_stage2/code/.venv/bin/python ]; then echo tau2_stage2/code/.venv/bin/python; elif [ -x experiments/tau2_stage2/code/.venv/bin/python ]; then echo experiments/tau2_stage2/code/.venv/bin/python; else echo python3; fi'

start_sft_servers() {
  echo "SFT_CHECKPOINT=$SFT_CHECKPOINT" | tee "$RESULT_ROOT/manifests/stage2_full_parserfix_checkpoint.txt"
  sync_server
  stop_remote_servers
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p "$REMOTE_PORT" "$REMOTE_HOST" "
set -euo pipefail
cd '$REMOTE_REPO'
mkdir -p 'scripts/ai4ai_qwen3_4b' 'results/ai4ai_4b_tau2/$RUN_NAME/logs'
REMOTE_PY=\$($remote_python_expr)
for gpu in 0 1 2 3 4 5 6 7; do
  port=\$((18080+gpu))
  log='results/ai4ai_4b_tau2/$RUN_NAME/logs/stage2_full_parserfix_sft_server_gpu'\${gpu}'_port'\${port}'.log'
  nohup \"\$REMOTE_PY\" scripts/ai4ai_qwen3_4b/hf_openai_server.py \
    --model-path '$SFT_CHECKPOINT' \
    --served-model-name '$SFT_SERVED_MODEL' \
    --host 127.0.0.1 --port \"\$port\" --device \"cuda:\$gpu\" --dtype bfloat16 \
    > \"\$log\" 2>&1 &
  echo \$! > \"\${log%.log}.pid\"
done
"
}

wait_remote_servers() {
  for _ in $(seq 1 120); do
    local ready
    ready="$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p "$REMOTE_PORT" "$REMOTE_HOST" \
      'n=0; for p in 18080 18081 18082 18083 18084 18085 18086 18087; do curl -sf http://127.0.0.1:$p/v1/models >/dev/null && n=$((n+1)); done; echo $n' 2>/dev/null || echo 0)"
    echo "remote ready=$ready/8"
    [[ "$ready" == "8" ]] && return 0
    sleep 10
  done
  return 1
}

start_lb() {
  for gpu in 0 1 2 3 4 5 6 7; do
    local_port=$((18180+gpu))
    remote_port=$((18080+gpu))
    if ! curl -sf "http://127.0.0.1:${local_port}/v1/models" >/dev/null 2>&1; then
      ssh -f -N -o BatchMode=yes -o StrictHostKeyChecking=no \
        -L "${local_port}:127.0.0.1:${remote_port}" \
        -p "$REMOTE_PORT" "$REMOTE_HOST"
    fi
  done
  tmux kill-session -t ai4ai_qwen3_4b_lb 2>/dev/null || true
  tmux new-session -d -s ai4ai_qwen3_4b_lb \
    "cd '$REPO_ROOT' && '$PY' scripts/ai4ai_qwen3_4b/openai_lb_proxy.py --host 127.0.0.1 --port '$LOCAL_LB_PORT' \
      --upstream http://127.0.0.1:18180 --upstream http://127.0.0.1:18181 \
      --upstream http://127.0.0.1:18182 --upstream http://127.0.0.1:18183 \
      --upstream http://127.0.0.1:18184 --upstream http://127.0.0.1:18185 \
      --upstream http://127.0.0.1:18186 --upstream http://127.0.0.1:18187 \
      >> '$LOG_DIR/stage2_full_parserfix_lb.log' 2>&1"
  sleep 2
  curl -sf "http://127.0.0.1:${LOCAL_LB_PORT}/v1/models" >/dev/null
}

smoke_sft() {
  "$PY" - <<PY
import json
import urllib.request
req = urllib.request.Request(
    "http://127.0.0.1:${LOCAL_LB_PORT}/v1/chat/completions",
    data=json.dumps({
        "model": "${SFT_SERVED_MODEL}",
        "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
        "temperature": 0,
        "max_tokens": 8,
    }).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as resp:
    payload = json.loads(resp.read())
print(json.dumps(payload["choices"][0]["message"], ensure_ascii=False))
PY
}

run_tau2() {
  local domain="$1"
  local n="$2"
  local save="$3"
  local log="$LOG_DIR/${save}.log"
  echo "START tau2 domain=$domain n=$n save=$save at $(date -Is)" | tee -a "$LOG_DIR/stage2_full_parserfix.driver.log"
  "$TAU2" run \
    --domain "$domain" \
    --agent-llm "openai/$SFT_SERVED_MODEL" \
    --agent-llm-args "$(json_agent_args)" \
    --user-llm openai/openai/gpt-5.2 \
    --user-llm-args "$(json_user_args)" \
    --task-split-name test \
    --num-trials 1 \
    --num-tasks "$n" \
    --max-steps 200 \
    --max-errors 10 \
    --seed 300 \
    --max-concurrency 2 \
    --save-to "$save" \
    --log-level ERROR \
    > "$log" 2>&1
  echo "DONE tau2 domain=$domain n=$n save=$save at $(date -Is)" | tee -a "$LOG_DIR/stage2_full_parserfix.driver.log"
}

summarize_saves() {
  "$PY" - "$SIM_ROOT" "$EVAL_DIR" "$@" <<'PY'
import json
import shutil
import sys
from pathlib import Path

sim_root = Path(sys.argv[1])
eval_dir = Path(sys.argv[2])
saves = sys.argv[3:]
eval_dir.mkdir(parents=True, exist_ok=True)

summary = {
    "model": "qwen3-4b-sft",
    "checkpoint": "qwen3_4b_stage2_full_1epoch/checkpoint-final",
    "split": "test",
    "num_trials": 1,
    "seed": 300,
    "max_steps": 200,
    "user_model": "openai/openai/gpt-5.2",
    "judge": "openai/openai/gpt-5.2",
    "saves": {},
}
total_ok = total_n = total_tool = total_user_tags = total_unexpected = 0

for save in saves:
    src = sim_root / save / "results.json"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    data = json.loads(src.read_text())
    sims = data.get("simulations", data if isinstance(data, list) else [])
    ok = 0
    tool_calls = 0
    user_tags = 0
    unexpected = 0
    terms = {}
    rewards = []
    for sim in sims:
        reward = float((sim.get("reward_info") or {}).get("reward", 0.0))
        rewards.append(reward)
        ok += reward == 1.0
        terms[str(sim.get("termination_reason"))] = terms.get(str(sim.get("termination_reason")), 0) + 1
        for msg in sim.get("messages", []) or []:
            if msg.get("role") == "assistant":
                tool_calls += len(msg.get("tool_calls") or [])
                if "<user>" in (msg.get("content") or ""):
                    user_tags += 1
            if msg.get("role") == "tool" and "unexpected keyword argument" in (msg.get("content") or ""):
                unexpected += 1
    item = {
        "n": len(sims),
        "pass": int(ok),
        "pass_rate": ok / len(sims) if sims else 0.0,
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "assistant_tool_calls": tool_calls,
        "assistant_user_tags": user_tags,
        "unexpected_kwarg_errors": unexpected,
        "termination_reasons": terms,
    }
    summary["saves"][save] = item
    total_ok += int(ok)
    total_n += len(sims)
    total_tool += tool_calls
    total_user_tags += user_tags
    total_unexpected += unexpected
    shutil.copy2(src, eval_dir / f"{save}.results.json")

summary["overall"] = {
    "n": total_n,
    "pass": total_ok,
    "pass_rate": total_ok / total_n if total_n else 0.0,
    "assistant_tool_calls": total_tool,
    "assistant_user_tags": total_user_tags,
    "unexpected_kwarg_errors": total_unexpected,
}
(eval_dir / "stage2_full_parserfix_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
)
lines = [
    "# Qwen3-4B Stage2-Full Parserfix Official Tau2 Eval",
    "",
    f"- checkpoint: `{summary['checkpoint']}`",
    f"- user/judge: `{summary['user_model']}`",
    f"- split: `{summary['split']}`, seed: `{summary['seed']}`",
    "",
    "| save | pass | n | pass_rate | tool_calls | user_tags | unexpected_kwarg |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for save, item in summary["saves"].items():
    lines.append(
        f"| {save} | {item['pass']} | {item['n']} | {item['pass_rate']:.3f} | "
        f"{item['assistant_tool_calls']} | {item['assistant_user_tags']} | {item['unexpected_kwarg_errors']} |"
    )
o = summary["overall"]
lines.append(
    f"| total | {o['pass']} | {o['n']} | {o['pass_rate']:.3f} | "
    f"{o['assistant_tool_calls']} | {o['assistant_user_tags']} | {o['unexpected_kwarg_errors']} |"
)
(eval_dir / "stage2_full_parserfix_summary.md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
}

full_eval() {
  export_commonstack_env
  trap 'stop_remote_servers' EXIT
  start_sft_servers
  wait_remote_servers
  start_lb
  smoke_sft | tee "$LOG_DIR/stage2_full_parserfix_smoke.log"

  local a="stage2_full_parserfix_${RUN_ID}_test_airline20"
  local r="stage2_full_parserfix_${RUN_ID}_test_retail40"
  local t="stage2_full_parserfix_${RUN_ID}_test_telecom40"
  run_tau2 airline 20 "$a"
  run_tau2 retail 40 "$r"
  run_tau2 telecom 40 "$t"
  summarize_saves "$a" "$r" "$t"
}

case "$MODE" in
  full) full_eval ;;
  smoke)
    export_commonstack_env
    trap 'stop_remote_servers' EXIT
    start_sft_servers
    wait_remote_servers
    start_lb
    smoke_sft | tee "$LOG_DIR/stage2_full_parserfix_smoke.log"
    save="stage2_full_parserfix_${RUN_ID}_smoke_airline5"
    run_tau2 airline 5 "$save"
    summarize_saves "$save"
    ;;
  summarize)
    summarize_saves \
      stage2_full_parserfix_20260617_005840_test_airline20 \
      stage2_full_parserfix_20260617_005840_test_retail40 \
      stage2_full_parserfix_20260617_005840_test_telecom40
    ;;
  *)
    echo "unknown MODE=$MODE" >&2
    exit 2
    ;;
esac
