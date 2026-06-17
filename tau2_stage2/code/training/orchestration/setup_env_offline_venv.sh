#!/usr/bin/env bash
# code/training/orchestration/setup_env_offline_venv.sh
# Install the training stack from offline_artifacts/ into code/.venv.
#
# Run this on the offline GPU machine after copying the bundle and
# offline_artifacts/.
#
# Usage:
#   BUNDLE_ROOT=/path/to/bundle bash code/training/orchestration/setup_env_offline_venv.sh
#
# Env vars:
#   BUNDLE_ROOT       default: inferred from this script path
#   ARTIFACTS_DIR     default: $BUNDLE_ROOT/offline_artifacts
#   PYTHON            default: python3.12
#   VENV_DIR          default: $BUNDLE_ROOT/code/.venv
#   MAX_JOBS_BUILD    default: 4
#   FLASH_ATTN_CUDA_ARCHS default: 90; H200 is sm90
#   LOCAL_MODELS_DIR  default: $ARTIFACTS_DIR/models
#   FORCE             default: 0; set 1 to bypass H200/CUDA preflight failures
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="${BUNDLE_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$BUNDLE_ROOT/offline_artifacts}"
PYTHON_BIN="${PYTHON:-python3.12}"
VENV_DIR="${VENV_DIR:-$BUNDLE_ROOT/code/.venv}"
MAX_JOBS_BUILD="${MAX_JOBS_BUILD:-4}"
FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-90}"
FORCE="${FORCE:-0}"

WHEELHOUSE="$ARTIFACTS_DIR/wheelhouse"
HF_HOME_DIR="$ARTIFACTS_DIR/hf_home"
LOCAL_MODELS_DIR="${EVOL_LOCAL_MODELS_DIR:-${LOCAL_MODELS_DIR:-$ARTIFACTS_DIR/models}}"
MANIFEST_DIR="$ARTIFACTS_DIR/manifests"
TRAIN_REQ="$MANIFEST_DIR/requirements_train_only.txt"
TORCH_CONSTRAINT="$MANIFEST_DIR/constraints_offline.txt"
VENV="$VENV_DIR/bin"

cd "$BUNDLE_ROOT"

echo "=== Preflight ==="
[[ -d "$WHEELHOUSE" ]] || { echo "Missing wheelhouse: $WHEELHOUSE" >&2; exit 1; }
[[ -d "$HF_HOME_DIR" ]] || { echo "Missing HF cache: $HF_HOME_DIR" >&2; exit 1; }
if [[ ! -f "$TRAIN_REQ" ]]; then
    mkdir -p "$MANIFEST_DIR"
    grep -vE '^(torch|flash-attn|vllm)==' "$BUNDLE_ROOT/code/training/requirements.txt" > "$TRAIN_REQ"
fi
if [[ ! -f "$TORCH_CONSTRAINT" ]]; then
cat > "$TORCH_CONSTRAINT" <<'EOF'
torch==2.11.0+cu130
cuda-bindings==13.0.3
cuda-pathfinder==1.2.2
fsspec==2026.2.0
setuptools==70.2.0
EOF
fi

if ! command -v nvcc >/dev/null 2>&1; then
    for CUDA_CANDIDATE in /usr/local/cuda-13.0 /usr/local/cuda-13 /usr/local/cuda; do
        if [[ -x "$CUDA_CANDIDATE/bin/nvcc" ]]; then
            export CUDA_HOME="$CUDA_CANDIDATE"
            export PATH="$CUDA_HOME/bin:$PATH"
            export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
            break
        fi
    done
elif [[ -z "${CUDA_HOME:-}" ]]; then
    NVCC_PATH="$(command -v nvcc)"
    export CUDA_HOME="$(cd "$(dirname "$NVCC_PATH")/.." && pwd)"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "H200"; then
        echo "Expected H200 GPUs (set FORCE=1 to override)" >&2
        [[ "$FORCE" == "1" ]] || exit 1
    fi
else
    echo "nvidia-smi not found (set FORCE=1 to override)" >&2
    [[ "$FORCE" == "1" ]] || exit 1
fi

if command -v nvcc >/dev/null 2>&1; then
    if ! nvcc --version | grep -q "release 13.0"; then
        echo "Expected nvcc CUDA 13.0 for this bundle (set FORCE=1 to override)" >&2
        nvcc --version >&2 || true
        [[ "$FORCE" == "1" ]] || exit 1
    fi
else
    echo "nvcc not found; flash-attn source build requires nvcc (set FORCE=1 to override)" >&2
    [[ "$FORCE" == "1" ]] || exit 1
fi
echo "  CUDA_HOME=${CUDA_HOME:-unset}"
echo "  nvcc=$(command -v nvcc 2>/dev/null || true)"

"$PYTHON_BIN" --version | grep -q "3.12" || {
    echo "Python 3.12 required; got $("${PYTHON_BIN}" --version 2>&1)" >&2
    exit 1
}

echo ""
echo "=== Stage 1: create venv ==="
"$PYTHON_BIN" -m venv "$VENV_DIR"

export HF_HOME="$HF_HOME_DIR"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
if [[ -d "$LOCAL_MODELS_DIR" ]]; then
    export EVOL_LOCAL_MODELS_DIR="$LOCAL_MODELS_DIR"
    export EVOL_REQUIRE_LOCAL_MODELS=1
else
    echo "  warn: local model directory not found: $LOCAL_MODELS_DIR" >&2
    echo "        will rely on HF_HOME cache instead" >&2
fi

echo ""
echo "=== Stage 2: pip tooling from wheelhouse ==="
"$VENV/python" -m pip install \
    --no-index --find-links "$WHEELHOUSE" \
    -U pip setuptools wheel packaging

echo ""
echo "=== Stage 3: torch 2.11.0 cu130 ==="
"$VENV/python" -m pip install \
    --no-index --find-links "$WHEELHOUSE" \
    --constraint "$TORCH_CONSTRAINT" \
    torch==2.11.0+cu130

"$VENV/python" - <<'PYEOF'
import torch
assert torch.cuda.is_available(), "CUDA not available"
assert torch.cuda.device_count() == 8, f"expected 8 GPUs, got {torch.cuda.device_count()}"
assert torch.version.cuda and torch.version.cuda.startswith("13"), torch.version.cuda
print(f"  torch {torch.__version__} CUDA {torch.version.cuda} on {torch.cuda.device_count()} GPUs OK")
PYEOF

echo ""
echo "=== Stage 4: flash-attn 2.8.3 source build ==="
"$VENV/python" -m pip install \
    --no-index --find-links "$WHEELHOUSE" \
    ninja packaging wheel psutil numpy einops
export MAX_JOBS="$MAX_JOBS_BUILD"
export FLASH_ATTN_CUDA_ARCHS
echo "  MAX_JOBS=$MAX_JOBS"
echo "  FLASH_ATTN_CUDA_ARCHS=$FLASH_ATTN_CUDA_ARCHS"
"$VENV/python" -m pip install \
    --no-index --find-links "$WHEELHOUSE" \
    --no-build-isolation \
    --no-deps \
    flash-attn==2.8.3

echo ""
echo "=== Stage 5: remaining training requirements ==="
"$VENV/python" -m pip install \
    --no-index --find-links "$WHEELHOUSE" \
    --constraint "$TORCH_CONSTRAINT" \
    -r "$TRAIN_REQ"

echo ""
echo "=== Stage 6: editable-install bundle code without dependency resolution ==="
"$VENV/python" -m pip install --no-deps -e "$BUNDLE_ROOT/code"

echo ""
echo "=== Stage 7: training import check ==="
"$VENV/python" <<'PYEOF'
import torch, transformers, accelerate, trl, peft, datasets
import flash_attn, matplotlib, pandas
print("=== Training imports OK ===")
print(f"  torch:        {torch.__version__} (CUDA {torch.version.cuda})")
print(f"  transformers: {transformers.__version__}")
print(f"  accelerate:   {accelerate.__version__}")
print(f"  trl:          {trl.__version__}")
print(f"  peft:         {peft.__version__}")
print(f"  datasets:     {datasets.__version__}")
print(f"  flash_attn:   {flash_attn.__version__}")
print(f"  matplotlib:   {matplotlib.__version__}")
print(f"  pandas:       {pandas.__version__}")
print(f"  CUDA devices: {torch.cuda.device_count()}")
PYEOF

echo ""
echo "=== Offline venv ready ==="
echo "VENV=$VENV"
echo "Use:"
echo "  export BUNDLE_ROOT=$BUNDLE_ROOT"
echo "  export VENV=$VENV"
echo "  export HF_HOME=$HF_HOME_DIR"
echo "  export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1"
if [[ -d "$LOCAL_MODELS_DIR" ]]; then
    echo "  export EVOL_LOCAL_MODELS_DIR=$LOCAL_MODELS_DIR EVOL_REQUIRE_LOCAL_MODELS=1"
fi
echo "  bash code/training/orchestration/train_pipeline.sh"
