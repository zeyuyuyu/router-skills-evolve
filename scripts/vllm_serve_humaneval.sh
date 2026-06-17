#!/usr/bin/env bash
# vllm_serve_humaneval.sh
#
# Start a vLLM OpenAI-compatible server for ONE HumanEval model, using the
# isolated .vllm_venv (separate torch — avoids conflicting with the main
# training env's torch). Backgrounds the server and writes a PID file.
#
# Usage:
#   scripts/vllm_serve_humaneval.sh <model_id_or_path> <port> <gpu_id> [served_name]
#
# Returns once the /v1/models endpoint is ready (or fails after a timeout).
# The pipeline registers each served model in HE_VLLM_MAP so the HumanEval
# adapter routes generation to the local server instead of in-process HF.
set -euo pipefail

MODEL="$1"
PORT="$2"
GPU="${3:-0}"
SERVED="${4:-$MODEL}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VLLM_VENV="${HE_VLLM_VENV:-$REPO_ROOT/.vllm_venv}"
VLLM_BIN="$VLLM_VENV/bin/vllm"
LOG_DIR="${HE_VLLM_LOG_DIR:-$REPO_ROOT/results/_vllm_logs}"
mkdir -p "$LOG_DIR"

if [[ ! -x "$VLLM_BIN" ]]; then
  echo "[vllm_serve] FATAL: vllm not found at $VLLM_BIN" >&2
  echo "             create it with: python3 -m venv .vllm_venv && .vllm_venv/bin/pip install vllm" >&2
  exit 1
fi

SAFE_NAME="$(echo "$SERVED" | tr '/' '_')"
PID_FILE="$LOG_DIR/vllm_${SAFE_NAME}_${PORT}.pid"
SRV_LOG="$LOG_DIR/vllm_${SAFE_NAME}_${PORT}.log"

# A LoRA adapter dir (cycles >= 1) is served as base model + --enable-lora.
LORA_ARGS=()
if [[ -f "$MODEL/adapter_config.json" ]]; then
  BASE="$("$VLLM_VENV/bin/python" - "$MODEL" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1] + "/adapter_config.json"))
print(cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-Coder-1.5B-Instruct"))
PY
)"
  echo "[vllm_serve] LoRA adapter detected → base=$BASE + --enable-lora ($MODEL as '$SERVED')"
  LORA_ARGS=(--enable-lora --lora-modules "$SERVED=$MODEL")
  MODEL_TO_LOAD="$BASE"
else
  MODEL_TO_LOAD="$MODEL"
fi

echo "[vllm_serve] starting: model=$MODEL_TO_LOAD served=$SERVED port=$PORT gpu=$GPU"
# Put the venv's bin first on PATH so vLLM's runtime subprocesses (ninja for
# kernel compilation, etc.) resolve to the venv tools rather than failing.
# Driver/CUDA workaround (driver 575 / CUDA 12.9 on this box):
#   - vllm 0.22's bundled FLASH_ATTN (vllm-flash-attn Hopper) + flashinfer are
#     built for CUDA 13 → "CUDA driver insufficient" / "No module named flashinfer".
#   - TRITON_ATTN compiles at runtime (no cu13 prebuilt kernel) and
#     VLLM_USE_FLASHINFER_SAMPLER=0 falls back to the PyTorch-native sampler.
# On a CUDA-13-capable driver, override: HE_VLLM_ATTN_BACKEND=FLASH_ATTN
# VLLM_USE_FLASHINFER_SAMPLER=1 (faster).
CUDA_VISIBLE_DEVICES="$GPU" \
  PATH="$VLLM_VENV/bin:$PATH" \
  VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}" \
  nohup "$VLLM_BIN" serve "$MODEL_TO_LOAD" \
    --served-model-name "$SERVED" \
    --port "$PORT" \
    --gpu-memory-utilization "${HE_VLLM_GPU_UTIL:-0.85}" \
    --max-model-len "${HE_VLLM_MAX_LEN:-4096}" \
    --dtype bfloat16 \
    --trust-remote-code \
    --attention-backend "${HE_VLLM_ATTN_BACKEND:-TRITON_ATTN}" \
    "${LORA_ARGS[@]}" \
    > "$SRV_LOG" 2>&1 &
SERVER_PID=$!
echo "PID=$SERVER_PID" > "$PID_FILE"
echo "[vllm_serve] PID=$SERVER_PID  log=$SRV_LOG  pidfile=$PID_FILE"

# Wait for readiness (up to ~5 min — first load downloads/loads weights).
for _ in $(seq 1 150); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[vllm_serve] READY on port $PORT"
    exit 0
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[vllm_serve] FATAL: server exited before ready; see $SRV_LOG" >&2
    tail -20 "$SRV_LOG" >&2 || true
    exit 1
  fi
  sleep 2
done
echo "[vllm_serve] FATAL: not ready after timeout; see $SRV_LOG" >&2
kill "$SERVER_PID" 2>/dev/null || true
exit 1
