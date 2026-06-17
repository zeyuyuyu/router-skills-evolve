#!/usr/bin/env bash
# setup_tau2_bench.sh
#
# Stage 5 ONLY of the colleague's setup_env_server.sh: clone the official
# τ²-bench at the pinned SHA into the bundle's vendor dir and editable-install
# it (+ the bundle's code/ package) into a venv — WITHOUT the cu13 torch /
# flash-attn stages (we serve via the separate .vllm_cu12_venv) and WITHOUT
# conda (portaudio comes from the PyAudio manylinux wheel, which bundles
# libportaudio.so — no system lib / sudo / conda needed).
#
# τ²-bench is the real benchmark data+runner the tau2 adapter wraps; the bundle
# deliberately does not vendor it (~2.6 GB). `import tau2` pulls voice+knowledge
# unconditionally, so both extras are installed even though we never call them.
#
# Usage:
#   scripts/setup_tau2_bench.sh
# Env:
#   VENV   target venv to install into   (default: <repo>/venv — the pipeline's
#          Phase-1 collect_traces runs under this and does `import tau2`)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUNDLE_ROOT="${BUNDLE_ROOT:-$REPO_ROOT/tau2_stage2}"
VENV="${VENV:-$REPO_ROOT/venv}"
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"

TAU2_DIR="$BUNDLE_ROOT/code/vendor/tau2-bench"
TAU2_PINNED_SHA="17e07b1da2bbc0cadfddeea36412686e0604127b"
TAU2_REPO="${TAU2_REPO:-https://github.com/sierra-research/tau2-bench}"

[[ -x "$PIP" ]] || { echo "[setup-tau2] FATAL: no pip at $PIP (set VENV=...)"; exit 1; }
echo "[setup-tau2] venv=$VENV  bundle=$BUNDLE_ROOT"

# 1) Clone + pin (idempotent).
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

# 2) portaudio without conda: the PyAudio manylinux wheel bundles libportaudio.so.
#    Install it first so the editable [voice] extra resolves to the wheel rather
#    than building pyaudio from sdist (which would need system portaudio headers).
echo "[setup-tau2] installing PyAudio wheel (bundles libportaudio, no conda/apt) ..."
"$PIP" install "pyaudio>=0.2.14"

# 3) Editable-install τ²-bench with the unconditional import extras.
echo "[setup-tau2] pip install -e tau2-bench[voice,knowledge] ..."
"$PIP" install -e "$TAU2_DIR[voice,knowledge]"

# 4) Editable-install the bundle's code/ package (registers adapters/, core/, ...).
echo "[setup-tau2] pip install -e $BUNDLE_ROOT/code ..."
"$PIP" install -e "$BUNDLE_ROOT/code"

# 5) Verify: import + a domain's tasks file is present.
echo "[setup-tau2] verifying ..."
"$PY" - "$TAU2_DIR" <<'PY'
import sys, importlib
from pathlib import Path
tau2_dir = Path(sys.argv[1])
import tau2  # noqa: F401 — triggers the voice/knowledge import chain
import tau2.utils.llm_utils  # noqa: F401 — patched by the adapter
from tau2.runner import build_text_orchestrator  # noqa: F401
tasks = tau2_dir / "data" / "tau2" / "domains" / "retail" / "tasks.json"
assert tasks.exists(), f"retail tasks.json missing at {tasks}"
print("[setup-tau2] OK: import tau2 + runner + retail tasks.json present")
PY

echo "[setup-tau2] done. Now you can run tau2 for real, e.g.:"
echo "  PYTHON=$VENV/bin/python EXPERIMENT_CONFIG=tau2_retail N_TASKS=20 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=6 \\"
echo "    bash scripts/run_full_pipeline.sh --bench tau2_bench --n-cycles 1 --model-config 06_qwen3_4b_273"
