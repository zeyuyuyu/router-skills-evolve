#!/usr/bin/env bash
# code/training/orchestration/setup_env_local.sh
# Local dev-box env setup mirroring the server.
# Target: WSL2/Linux laptop, CUDA 13.0 driver, 4 GB VRAM, conda available, Python 3.12.
#
# Mirrors setup_env_server.sh exactly — same Python version, same packages,
# same install order. The only difference is MAX_JOBS_BUILD defaulting to 4
# (laptop CPU) and not asserting H200 GPU count. flash-attn IS built locally
# so the same build path is exercised before cloud handoff.
#
# Usage:
#   BUNDLE_ROOT=/path/to/bundle bash code/training/orchestration/setup_env_local.sh
#
# Env vars:
#   BUNDLE_ROOT      (required) — path to bundle root
#   ENV_NAME         (default: tau2-stage2-local)
#   MAX_JOBS_BUILD   (default: 4 — laptop)
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:?set BUNDLE_ROOT}"
ENV_NAME="${ENV_NAME:-tau2-stage2-local}"
MAX_JOBS_BUILD="${MAX_JOBS_BUILD:-4}"

cd "$BUNDLE_ROOT"

echo "=== Stage 1: conda env + CUDA 13.0 toolkit ==="
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "  conda env '$ENV_NAME' already exists — reusing"
else
    conda create -n "$ENV_NAME" python=3.12 cuda-toolkit=13.0 -c nvidia -y
fi

# conda's cuda-nvcc activate hook references unbound NVCC_PREPEND_FLAGS;
# disable -u briefly so the hook doesn't trigger pipefail.
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
set -u

# Force CUDA_HOME / CUDA_PATH to conda's CUDA — system CUDA_HOME (often
# /usr/local/cuda-12.x) breaks torch.utils.cpp_extension's version check.
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"

python --version | grep -q "3.12" || { echo "Python 3.12 not active"; exit 1; }
nvcc --version | grep -q "release 13.0" || { echo "nvcc 13.0 not on PATH"; exit 1; }
nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi failed — need a CUDA GPU"; exit 1; }

echo ""
echo "=== Stage 2: torch 2.11.0 + cu130 ==="
# Force the PyTorch cu130 index for determinism (PyPI default ships cu12
# wheels for some torch versions). torchvision is unused, so omit it.
pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available — wheel did not resolve to cu130'
assert '13' in torch.version.cuda, f'expected CUDA 13.x, got {torch.version.cuda}'
print(f'  torch {torch.__version__} CUDA {torch.version.cuda} on {torch.cuda.device_count()} GPU(s) OK')
"

echo ""
echo "=== Stage 3: flash-attn — SKIPPED on local (cloud-only source build) ==="
echo "  flash-attn 2.8.3 source build needs ~16+ GB RAM with MAX_JOBS=4 and"
echo "  >1 hr at MAX_JOBS=1 — exceeds laptop budget. Local smoke configs use"
echo "  attn_implementation=eager so flash-attn isn't required for validation."
echo "  Server-side setup_env_server.sh DOES build it."
echo ""
echo "  OPTIONAL for local GPU model-loading tests:"
echo "    pip install --pre \"flash-attn-4[cu13]\""
echo "  Required ONLY when loading a transformers model with AutoModelForCausalLM"
echo "  on this local env (transformers 5.8 imports flash_attention.py at module"
echo "  load time and KeyErrors if neither flash-attn nor flash-attn-4 is installed)."
echo "  pytest does NOT need this — tests use helpers + mocks, never actually load"
echo "  a transformers model. flash-attn-4 is currently 4.0.0b12 (beta) and is NOT"
echo "  a drop-in replacement for the server's flash-attn 2.x (transformers' three"
echo "  attn_implementation paths flash_attention_2 / _3 / _4 are mutually exclusive)."

echo ""
echo "=== Stage 4: rest of requirements.txt (skip flash-attn + vllm) ==="
# vllm hard-pins flash-attn (both cloud-only). Build a temp requirements
# minus those two. bitsandbytes was removed from requirements.txt in iter-6.
TMP_REQ=$(mktemp --suffix=-local-requirements.txt)
grep -Ev '^(flash-attn|vllm)' "$BUNDLE_ROOT/code/training/requirements.txt" > "$TMP_REQ"
pip install -r "$TMP_REQ"
rm "$TMP_REQ"

echo ""
echo "=== Stage 5: smoke-test all imports (local skips flash-attn / vllm / bitsandbytes) ==="
python <<'PYEOF'
import torch, transformers, accelerate, trl, peft, datasets
import matplotlib, pandas
print("=== Local imports OK ===")
print(f"  torch:        {torch.__version__} (CUDA {torch.version.cuda})")
print(f"  transformers: {transformers.__version__}")
print(f"  accelerate:   {accelerate.__version__}")
print(f"  trl:          {trl.__version__}")
print(f"  peft:         {peft.__version__}")
print(f"  datasets:     {datasets.__version__}")
print(f"  matplotlib:   {matplotlib.__version__}")
print(f"  pandas:       {pandas.__version__}")
print(f"  CUDA devices: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"  VRAM (GPU 0): {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print()
print("  Skipped (cloud-only): flash-attn, vllm")
PYEOF

echo ""
echo "=== Local env ready: conda env '$ENV_NAME' ==="
echo "Activate with: conda activate $ENV_NAME"
