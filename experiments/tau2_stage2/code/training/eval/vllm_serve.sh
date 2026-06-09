#!/usr/bin/env bash
# code/training/eval/vllm_serve.sh
# Spin up vLLM in OpenAI-compatible mode for one trained checkpoint.
# Usage: vllm_serve.sh <checkpoint_dir> <port> [<gpu_id>]
set -euo pipefail

CKPT="$1"
PORT="$2"
GPU="${3:-0}"

if [[ ! -d "$CKPT" ]]; then
    echo "Checkpoint dir does not exist: $CKPT" >&2; exit 1
fi

# Sanity: GPU index must be a non-negative integer. The TP+GPU range check
# is done below once TP_SIZE is resolved.
if ! [[ "$GPU" =~ ^[0-9]+$ ]]; then
    echo "GPU must be a non-negative integer, got: $GPU" >&2; exit 1
fi

# Default Qwen3 tool parser. (Qwen3.5/3.6 share the same parser per vLLM 0.20+.)
TOOL_PARSER="qwen3_coder"

# Tensor parallelism. Three eval layouts are supported on the 8× H200 server:
#   TP=1: legacy one-GPU-per-checkpoint (original eval_all.sh).
#   TP=2: step-budget eval default for 2B/4B/9B. Up to 4 vLLM instances run
#         concurrently, each pinning a contiguous pair of GPUs (0..1, 2..3,
#         4..5, 6..7). Doubles per-GPU KV headroom over TP=1 → max_model_len
#         can grow from 32K to 128K without exhausting KV cache.
#   TP=8: 35B MoE — all 8 H200s back ONE eval; the ~70 GB bf16 weights
#         + KV cache cannot reliably fit on a single H200.
#
# `$GPU` is the LOWEST GPU index in the TP group. With TP>1 the visible
# devices become $GPU..$GPU+TP_SIZE-1. The caller (eval_all_step_budget.sh)
# is responsible for ensuring (GPU + TP_SIZE) <= 8 — i.e., the group fits
# within the 8 physical GPUs and does not overlap with another in-flight
# vLLM. The earlier implementation hardcoded 0..TP-1 here, which silently
# broke any multi-TP parallel layout (every instance would race for
# GPUs 0..TP-1).
TP_SIZE="${TP_SIZE:-1}"
# 8 H200s on the server; the (GPU, GPU+TP-1) range must fit inside [0, 7].
# Allow override via NUM_GPUS for non-standard machines.
NUM_GPUS="${NUM_GPUS:-8}"
if (( GPU + TP_SIZE > NUM_GPUS )); then
    echo "Invalid GPU range: GPU=$GPU + TP_SIZE=$TP_SIZE exceeds NUM_GPUS=$NUM_GPUS" >&2
    exit 1
fi
if [[ "$TP_SIZE" -gt 1 ]]; then
    CUDA_DEVICES=$(seq -s, "$GPU" $((GPU + TP_SIZE - 1)))
else
    CUDA_DEVICES="$GPU"
fi

# GPU memory utilization. With TP=1 a 9B (~18 GB bf16) fills a big share of
# one 141 GB H200, leaving the rest for KV at 0.85 util. With TP>=2 each GPU
# holds only 1/TP_SIZE of the weights so 0.92 leaves ample KV headroom:
#   * 9B/TP=2 → ~9 GB weights/GPU → ~121 GB KV/GPU → ~1.5M tokens of KV
#   * 4B/TP=2 → ~4 GB weights/GPU → ~126 GB KV/GPU → ~1.7M tokens of KV
#   * 35B/TP=8 → ~9 GB weights/GPU → ~121 GB KV/GPU → ~9.8M tokens of KV
# Any of these comfortably covers max_model_len=131072 with one in-flight
# request per harness process.
if [[ "$TP_SIZE" -gt 1 ]]; then
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
else
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
fi

# --dtype bfloat16: explicit; SFT bf16 checkpoints would otherwise rely on
# `auto` inferring from config.json — fine in practice but cheap to pin.
# --enable-prefix-caching: tau2 multi-turn reuses long system+tool prompts
# across every step; KV-cache reuse is a large speedup for this workload.
# --max-num-seqs 16: tau2 issues at most --max-concurrency=4 in-flight
# requests per invocation, but the harness only runs one rollout at a time
# per process — 16 is well over-provisioned and the KV pages are allocated
# on demand, so the higher cap costs nothing.
# --max-model-len 131072: 4× the SFT context (32K), well inside Qwen3.5/3.6's
# native 262144 (256K) max_position_embeddings (verified live against HF model
# config.json on 2026-05-23). Step-budget rollouts can run long:
#   * Method A on airline/8 (B=59): student does ALL 59 turns from scratch.
#     ~7K system+tools + 59 turns × ~1K tokens/turn ≈ 65-70K tokens at the
#     final turn — the prior 65K cap was right at the edge and would clip.
#   * Method B with late substitution + +10 drift-buffer baseline tail:
#     airline/8 max_steps=B+10=69 turns ≈ 75K tokens at the final turn.
# 128K gives ~2× headroom over the worst case while staying half of the
# model's native window (so RoPE positions stay well inside the trained
# range when YaRN/scaling is applied — Qwen3 ships rope_scaling in its
# own config.json, so vLLM picks it up without explicit flags).
# KV budget on H200: 9B/TP=2 @ 0.92 util → ~1.5M tokens of KV cache, with
# one in-flight request that is comfortably more than 128K of headroom.
# Override at the env-var level if needed: MAX_MODEL_LEN=262144 bash ...
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MOE_BACKEND="${MOE_BACKEND:-auto}"
VLLM_LOG="$CKPT/vllm_serve.log"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    vllm serve "$CKPT" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype bfloat16 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --language-model-only \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_PARSER" \
    --reasoning-parser qwen3 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --gdn-prefill-backend "${GDN_PREFILL_BACKEND:-triton}" \
    --moe-backend "$MOE_BACKEND" \
    --served-model-name "evol-llm-student" \
    >> "$VLLM_LOG" 2>&1 < /dev/null &

VLLM_PID=$!
echo "vLLM PID=$VLLM_PID port=$PORT ckpt=$CKPT" > "$CKPT/vllm_serve.pid"

# Health check — wait up to 10 minutes (120 × 5s). Aligned with the
# Python-side wait_for_vllm timeout in harness.py so the bash helper
# doesn't kill vllm before the harness gives up polling. Big checkpoints
# (35B Qwen3.6 MoE) routinely take 5+ min to load.
for _ in $(seq 1 120); do
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        echo "vLLM ready on port $PORT (PID $VLLM_PID)"
        wait "$VLLM_PID"
        exit $?
    fi
    # Bail early if the vllm process itself died (OOM, bad arg, etc.).
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM PID $VLLM_PID died before becoming ready; see $CKPT/vllm_serve.log" >&2
        exit 1
    fi
    sleep 5
done
echo "vLLM failed to start on port $PORT within timeout" >&2
kill "$VLLM_PID" 2>/dev/null || true
exit 1
