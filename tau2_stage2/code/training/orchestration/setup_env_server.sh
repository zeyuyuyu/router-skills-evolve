#!/usr/bin/env bash
# code/training/orchestration/setup_env_server.sh
# Server-side env setup for τ²-bench Stage-2 training.
# Target: 8× H200, CUDA 13.0, conda available, Python 3.12.
#
# Usage:
#   BUNDLE_ROOT=/path/to/bundle bash code/training/orchestration/setup_env_server.sh
#
# Env vars:
#   BUNDLE_ROOT  (required) — path to bundle root
#   ENV_NAME     (default: tau2-stage2)
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:?set BUNDLE_ROOT}"
ENV_NAME="${ENV_NAME:-tau2-stage2}"
MAX_JOBS_BUILD="${MAX_JOBS_BUILD:-4}"   # FA build peaks ~12-15 GB RAM per nvcc job
                                        # (Dao-AILab/flash-attention#1038); 4 is safe on
                                        # any H200 node, 8 needs >120 GB free RAM.
                                        # Operator can bump if `free -g` headroom allows.

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
nvidia-smi | grep -q "H200" || { echo "Expected H200 GPUs"; exit 1; }

echo ""
echo "=== Stage 2: torch 2.11.0 + cu130 (deterministic via PyTorch index) ==="
# torchvision is not used anywhere in the framework — drop it to avoid an
# unpinned co-install. The PyPI default index doesn't ship +cu130 wheels for
# every micro torch version; force the PyTorch cu130 index for determinism.
pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available — wheel did not resolve to cu130'
assert '13' in torch.version.cuda, f'expected CUDA 13.x, got {torch.version.cuda}'
assert torch.cuda.device_count() == 8, f'expected 8 GPUs, got {torch.cuda.device_count()}'
print(f'  torch {torch.__version__} CUDA {torch.version.cuda} on {torch.cuda.device_count()} GPUs OK')
"

echo ""
echo "=== Stage 3: flash-attn 2.8.3 (source build — no cu13 wheel exists) ==="
echo "  Building with MAX_JOBS=$MAX_JOBS_BUILD; expect 30-60 min on H200..."
# flash-attn 2.8.3 is built with --no-build-isolation, so its setup.py has
# to find ninja, packaging, wheel, psutil in the env at build time. ninja
# in particular is not a transitive dep of pip/conda — install it
# explicitly before invoking flash-attn's source build.
#
# Iter-16 re-verification (2026-05-11): Attempted to switch to
# flash-attn-4 4.0.0b12 with [cu13] extra (the official upstream
# Hopper-native path per Dao-AILab/flash-attention README). VERDICT: NO-GO
# for our pipeline. Three confirmed blockers:
#
#   1. TRL 1.4.0's FLASH_ATTENTION_VARIANTS set (defined in
#      trl/trainer/sft_trainer.py) does NOT include 'flash_attention_4'
#      — verified by direct inspection of installed source. With
#      packing=True, packing_strategy=bfd, padding_free=True (all 10 of
#      our production runs), TRL logs a warning but proceeds — risking
#      silent cross-sequence contamination during BFD packing. No TRL
#      release (latest 1.4.0) or main-branch commit has added FA4 yet
#      (last variant-set change was 2025-09-30, before FA4 existed).
#
#   2. Qwen3.6-35B-A3B uses GDN (Gated DeltaNet) on 75% of layers — the
#      `Qwen3_5MoeGatedDeltaNet` class bypasses _attn_implementation
#      entirely and calls FLA kernels directly. Run 08 sees ZERO FA4
#      benefit (only 25% of layers route through the dispatch), AND
#      transformers' GDN doesn't support packing in any case.
#
#   3. FA4 varlen backward is documented as unsupported on sm_90 per
#      transformers PR #42435 (the FA4 integration PR). Packing on H200
#      (SM 9.0) needs varlen backward for the training pass; would fail.
#
# FA3 (`attn_implementation="flash_attention_3"`) IS in TRL's variant set
# and emits real Hopper sm_90 kernels (TMA/WGMMA), so it would close the
# perf gap — but install is awkward (`cd flash-attention/hopper && python
# setup.py install` from the repo, separate from FA2). For now FA2 stays
# primary on the principle that proven > faster-but-untested.
pip install ninja packaging wheel psutil
export MAX_JOBS="$MAX_JOBS_BUILD"
pip install --no-build-isolation flash-attn==2.8.3
python -c "import flash_attn; print(f'  flash_attn {flash_attn.__version__} OK')"

echo ""
echo "=== Stage 4: rest of requirements.txt (incl. vllm 0.20.2 from PyPI) ==="
# Iter-16 (2026-05-11): The GitHub +cu129 wheel detour (formerly Stage 3b)
# is replaced by the standard PyPI install via this stage's
# `pip install -r requirements.txt`. The requirements.txt pin
# `vllm==0.20.2` resolves to the PyPI manylinux wheel, which is the
# CUDA-12.9-default build per docs.vllm.ai ("vLLM's binaries are compiled
# with CUDA 12.9 and public PyTorch release versions by default").
# CUDA forward-compat: cu12.9 binaries run on cu13 driver; vllm's kernels
# JIT to Hopper sm_90 at first launch. No +cu130 wheel exists on GitHub
# releases for any v0.20.x (verified via HTTP HEAD 2026-05-11).
pip install -r "$BUNDLE_ROOT/code/training/requirements.txt"
python -c "import vllm; print(f'  vllm {vllm.__version__} OK (PyPI cu12.9-default wheel on cu13 driver, fwd-compat)')"

echo ""
echo "=== Stage 5: τ²-bench (cloned at pinned SHA, editable install with [voice]) ==="
# τ²-bench is a runtime dependency for code/training/eval/harness.py, which
# spawns `python -m tau2.cli run …` to drive force-routed evaluation against
# the local vLLM. The bundle deliberately does not vendor the upstream repo
# (would add ~2.6 GB); clone fresh at the pinned SHA used during corpus
# collection so eval-side code matches the assumptions baked into the data.
#
# WHY [voice,knowledge]: tau2/__init__.py imports tau2.runner → tau2.evaluator
# → tau2.agent → tau2.voice → tau2.knowledge unconditionally (upstream design
# choice). Even though we never call any voice or knowledge path, the imports
# must resolve, so we install both extras + the system portaudio shared
# library that pyaudio binds to. portaudio is installed via conda-forge so
# we don't depend on `apt install portaudio19-dev` being available on the
# cloud node. Verified empirically: bare `pip install -e tau2-bench` fails
# with `scipy/pyaudio/elevenlabs` then `rank_bm25` missing on `import tau2`.
TAU2_DIR="$BUNDLE_ROOT/code/vendor/tau2-bench"
TAU2_PINNED_SHA="17e07b1da2bbc0cadfddeea36412686e0604127b"
TAU2_REPO="https://github.com/sierra-research/tau2-bench"
mkdir -p "$BUNDLE_ROOT/code/vendor"
if [ -d "$TAU2_DIR/.git" ]; then
    echo "  tau2-bench checkout already present at $TAU2_DIR — reusing"
else
    git clone "$TAU2_REPO" "$TAU2_DIR"
fi
git -C "$TAU2_DIR" fetch --depth=1 origin "$TAU2_PINNED_SHA" 2>/dev/null || \
    git -C "$TAU2_DIR" fetch origin
git -C "$TAU2_DIR" checkout "$TAU2_PINNED_SHA"
HEAD_SHA=$(git -C "$TAU2_DIR" rev-parse HEAD)
test "$HEAD_SHA" = "$TAU2_PINNED_SHA" || { echo "tau2 checkout SHA mismatch (got $HEAD_SHA, expected $TAU2_PINNED_SHA)"; exit 1; }

# portaudio system lib (pyaudio dlopen target). conda-forge so no sudo needed.
conda install -c conda-forge -y portaudio
pip install -e "$TAU2_DIR[voice,knowledge]"
python -c "import tau2; print(f'  tau2 imports OK')"

echo ""
echo "=== Stage 5b: editable-install the bundle's code/ package ==="
# pyproject.toml at code/ registers training/, pipeline/, adapters/, core/,
# scripts/ as importable packages. Without this install, `python -m
# training.train` only works when the cwd is code/. train_all.sh / eval_all.sh
# do `cd code/` before invoking, so this isn't strictly required — but making
# it pip-discoverable means the operator can `python -m training.eval.harness`
# from anywhere, and PYTHONPATH-less debug sessions Just Work.
pip install -e "$BUNDLE_ROOT/code"

echo ""
echo "=== Stage 6: smoke-test all imports ==="
python <<'PYEOF'
import torch, transformers, accelerate, trl, peft, datasets
import vllm, flash_attn, matplotlib, pandas
import tau2
# The harness drives tau2 via subprocess (`python -m tau2.cli run`), but
# cross-checking that the in-process import path is healthy too catches
# install-tree breakage before the first eval run.
from tau2.runner import build_text_orchestrator       # noqa: F401
from tau2.evaluator.evaluator import EvaluationType   # noqa: F401
print("=== All package imports OK ===")
print(f"  torch:        {torch.__version__} (CUDA {torch.version.cuda})")
print(f"  transformers: {transformers.__version__}")
print(f"  accelerate:   {accelerate.__version__}")
print(f"  trl:          {trl.__version__}")
print(f"  peft:         {peft.__version__}")
print(f"  datasets:     {datasets.__version__}")
print(f"  vllm:         {vllm.__version__}")
print(f"  flash_attn:   {flash_attn.__version__}")
print(f"  matplotlib:   {matplotlib.__version__}")
print(f"  pandas:       {pandas.__version__}")
print(f"  tau2 runtime: build_text_orchestrator + EvaluationType importable")
print(f"  CUDA devices: {torch.cuda.device_count()}")
PYEOF

echo ""
echo "=== Server env ready: conda env '$ENV_NAME' ==="
echo "Activate with: conda activate $ENV_NAME"
