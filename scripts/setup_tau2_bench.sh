#!/usr/bin/env bash
# setup_tau2_bench.sh
#
# Stage 5 ONLY of the colleague's setup_env_server.sh: clone the official
# τ²-bench at the pinned SHA into the bundle's vendor dir and editable-install
# it (+ the bundle's code/ package) into a DEDICATED venv — WITHOUT the cu13
# torch/flash-attn stages (we serve via .vllm_cu12_venv) and WITHOUT conda.
#
# venv: `.venv_tau2`, created from the main venv's python with
# --system-site-packages so it SEES the pipeline deps (src/, sklearn, openai,
# matplotlib, …) without reinstalling them; tau2-bench's deps land in this
# overlay only. Run the pipeline under it:
#   PYTHON=$REPO_ROOT/.venv_tau2/bin/python ... bash scripts/run_full_pipeline.sh --bench tau2_bench
#
# AUDIO / voice: `import tau2` unconditionally pulls the voice chain
# (tau2.agent.base.streaming → tau2.voice → scipy/soundfile/pyaudio). We never
# run audio, but the imports must resolve. The wheel-available pieces (scipy,
# soundfile, elevenlabs, librosa) are pip-installed; **pyaudio has no wheel here
# and needs system portaudio to build (no conda/sudo)** → we drop in a tiny stub
# so `import pyaudio` resolves. Real mic I/O would raise — we never call it.
# (This is why we install tau2-bench with [knowledge] only, NOT [voice]: the
#  [voice] extra would force pip to build pyaudio from source and abort.)
#
# Usage: scripts/setup_tau2_bench.sh
# Env: MAIN_VENV (default <repo>/venv), TAU2_VENV (default <repo>/.venv_tau2)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUNDLE_ROOT="${BUNDLE_ROOT:-$REPO_ROOT/tau2_stage2}"
MAIN_VENV="${MAIN_VENV:-$REPO_ROOT/venv}"
TAU2_VENV="${TAU2_VENV:-$REPO_ROOT/.venv_tau2}"

TAU2_DIR="$BUNDLE_ROOT/code/vendor/tau2-bench"
TAU2_PINNED_SHA="17e07b1da2bbc0cadfddeea36412686e0604127b"
TAU2_REPO="${TAU2_REPO:-https://github.com/sierra-research/tau2-bench}"

[[ -x "$MAIN_VENV/bin/python" ]] || { echo "[setup-tau2] FATAL: main venv python not at $MAIN_VENV"; exit 1; }

# 1) Dedicated venv that can see the main venv's packages (overlay only adds tau2).
if [[ ! -x "$TAU2_VENV/bin/python" ]]; then
  echo "[setup-tau2] creating $TAU2_VENV (--system-site-packages from main venv)"
  "$MAIN_VENV/bin/python" -m venv --system-site-packages "$TAU2_VENV"
  "$TAU2_VENV/bin/pip" install --upgrade pip wheel >/dev/null
fi
PIP="$TAU2_VENV/bin/pip"
PY="$TAU2_VENV/bin/python"

# 2) Clone + pin (idempotent).
mkdir -p "$BUNDLE_ROOT/code/vendor"
if [[ -d "$TAU2_DIR/.git" ]]; then
  echo "[setup-tau2] reusing existing checkout at $TAU2_DIR"
else
  echo "[setup-tau2] cloning $TAU2_REPO (~2.6 GB) ..."
  git clone "$TAU2_REPO" "$TAU2_DIR"
fi
git -C "$TAU2_DIR" fetch --depth=1 origin "$TAU2_PINNED_SHA" 2>/dev/null || git -C "$TAU2_DIR" fetch origin
git -C "$TAU2_DIR" checkout "$TAU2_PINNED_SHA"
HEAD_SHA="$(git -C "$TAU2_DIR" rev-parse HEAD)"
[[ "$HEAD_SHA" == "$TAU2_PINNED_SHA" ]] || { echo "[setup-tau2] FATAL: SHA mismatch ($HEAD_SHA != $TAU2_PINNED_SHA)"; exit 1; }

# 3) Voice-chain deps that DO have wheels (so `import tau2` resolves), no conda.
echo "[setup-tau2] installing wheel-available voice deps (scipy/soundfile/elevenlabs/librosa) ..."
"$PIP" install scipy soundfile elevenlabs librosa

# 4) pyaudio stub (no wheel here + needs system portaudio to build → stub it).
SP="$("$PY" -c "import site; print(site.getsitepackages()[0])")"
cat > "$SP/pyaudio.py" <<'PYSTUB'
"""Stub PyAudio — tau2.voice imports it at module load, but this pipeline never
runs live audio I/O. Real pyaudio needs system portaudio (no wheel / no conda
here). Any actual use raises clearly; imports + module-level constants resolve.
"""
paInt16 = 8; paInt32 = 2; paFloat32 = 1; paInt8 = 16; paUInt8 = 32
paContinue = 0; paComplete = 1; paAbort = 2

class PyAudio:  # noqa: N801
    def __init__(self, *a, **k):
        raise RuntimeError("pyaudio stub: live audio I/O unavailable in this env")
    def __getattr__(self, name):
        raise RuntimeError("pyaudio stub: live audio I/O unavailable in this env")

def get_format_from_width(width, unsigned=True):
    return paInt16

def __getattr__(name):  # PEP 562 — tolerate any other module-level access at import
    return None
PYSTUB
echo "[setup-tau2] wrote pyaudio stub → $SP/pyaudio.py"

# 5) Editable-install τ²-bench [knowledge] (NOT [voice]) + the bundle's code/ pkg.
echo "[setup-tau2] pip install -e tau2-bench[knowledge] + bundle code ..."
"$PIP" install -e "$TAU2_DIR[knowledge]"
"$PIP" install -e "$BUNDLE_ROOT/code"

# 6) Verify import + retail tasks present.
"$PY" - "$TAU2_DIR" <<'PY'
import sys
from pathlib import Path
tau2_dir = Path(sys.argv[1])
import tau2  # noqa: F401 — runs the voice import chain (scipy/soundfile/pyaudio-stub)
import tau2.utils.llm_utils  # noqa: F401 — patched by the adapter
from tau2.runner import build_text_orchestrator  # noqa: F401
tasks = tau2_dir / "data" / "tau2" / "domains" / "retail" / "tasks.json"
assert tasks.exists(), f"retail tasks.json missing at {tasks}"
print("[setup-tau2] OK: import tau2 + runner + retail tasks.json present (voice stubbed, no audio)")
PY

echo "[setup-tau2] done. Run tau2 for real, e.g.:"
echo "  PYTHON=$TAU2_VENV/bin/python EXPERIMENT_CONFIG=tau2_retail N_TASKS=20 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=6 \\"
echo "    bash scripts/run_full_pipeline.sh --bench tau2_bench --n-cycles 1 --model-config 06_qwen3_4b_273"
