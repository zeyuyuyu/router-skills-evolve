#!/usr/bin/env bash
# setup_vllm_cu12_venv.sh
#
# Build .vllm_cu12_venv — a CUDA-12 vLLM that actually RUNS on this box's driver
# (575.57.08 / CUDA 12.9 ceiling). The other venv (.vllm_venv, vllm 0.22) is a
# cu13 build: its flash-attn / flashinfer / _C kernels are compiled for CUDA 13
# and fail with "CUDA driver version is insufficient for CUDA runtime version"
# (works only for a couple of Qwen2.5 models via TRITON_ATTN hacks, breaks on
# Qwen3-4B). vllm 0.11.0 is cu12 → native FLASH_ATTN, no workarounds.
#
# Pinned, verified-working combo on driver 575:
#   vllm 0.11.0 · torch 2.8.0+cu128 · transformers 4.57.x · fastapi 0.116.1 · starlette 0.41.3
# (vllm 0.11 pulls transformers 5.x + fastapi 0.137 which break it — we pin both back:
#  transformers <5 keeps `all_special_tokens_extended`, >=4.51 keeps Qwen3 support;
#  fastapi/starlette pre-`_IncludedRouter` avoid the API-server HTTP 500.)
#
# Usage: scripts/setup_vllm_cu12_venv.sh [venv_dir]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${1:-$REPO_ROOT/.vllm_cu12_venv}"
PY="${PYTHON:-python3}"

echo "[setup-cu12] building $VENV with $("$PY" --version 2>&1)"
rm -rf "$VENV"
"$PY" -m venv "$VENV"
PIP="$VENV/bin/pip"
"$PIP" install --upgrade pip wheel

# vLLM (pulls torch 2.8.0+cu128 — CUDA 12.8 ≤ driver 575's 12.9 ceiling).
"$PIP" install "vllm==0.11.0"
# Pin the deps vllm 0.11 over-resolves to incompatible latest versions.
"$PIP" install "transformers>=4.51,<5.0" "fastapi==0.116.1" "starlette==0.41.3"

"$VENV/bin/python" - <<'PY'
import torch, vllm, transformers, fastapi, starlette
print("[setup-cu12] OK:",
      "vllm", vllm.__version__, "| torch", torch.__version__, torch.version.cuda,
      "| transformers", transformers.__version__,
      "| fastapi", fastapi.__version__, "| starlette", starlette.__version__)
PY

echo "[setup-cu12] done. Smoke serve, e.g.:"
echo "  CUDA_VISIBLE_DEVICES=2 PATH=$VENV/bin:\$PATH $VENV/bin/vllm serve Qwen/Qwen3-4B \\"
echo "    --served-model-name qwen3-4b --port 8106 --dtype bfloat16 --trust-remote-code"
