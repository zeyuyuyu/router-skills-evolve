#!/usr/bin/env bash
# setup_tau2_bench.sh
#
# Stage 5 ONLY of the colleague's setup_env_server.sh: clone the official
# τ²-bench at the pinned SHA into the bundle's vendor dir and editable-install
# it (+ the bundle's code/ package) into a DEDICATED venv — WITHOUT the cu13
# torch / flash-attn stages (we serve via .vllm_cu12_venv) and WITHOUT conda.
#
# venv: `.venv_tau2`, created from the main venv's python with
# --system-site-packages so it SEES the pipeline deps (src/, sklearn, openai,
# matplotlib, …) without reinstalling them, while tau2-bench's own deps land in
# this overlay and never pollute the main venv. Run the pipeline under it:
#   PYTHON=$REPO_ROOT/.venv_tau2/bin/python ... bash scripts/run_full_pipeline.sh --bench tau2_bench
#
# Extras: default `knowledge` (rank_bm25) only — NO voice/audio (no pyaudio /
# elevenlabs / portaudio). If `import tau2` turns out to pull voice
# unconditionally at this SHA, the verify step fails loudly; re-run with
# TAU2_EXTRAS=voice,knowledge (and it'll grab the PyAudio manylinux wheel, which
# bundles libportaudio — still no conda).
#
# Usage:
#   scripts/setup_tau2_bench.sh
# Env: MAIN_VENV (default <repo>/venv), TAU2_VENV (default <repo>/.venv_tau2),
#      TAU2_EXTRAS (default "knowledge")
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUNDLE_ROOT="${BUNDLE_ROOT:-$REPO_ROOT/tau2_stage2}"
MAIN_VENV="${MAIN_VENV:-$REPO_ROOT/venv}"
TAU2_VENV="${TAU2_VENV:-$REPO_ROOT/.venv_tau2}"
TAU2_EXTRAS="${TAU2_EXTRAS:-knowledge}"

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

# 3) Editable-install τ²-bench (no voice by default) + the bundle's code/ package.
extras_suffix=""
[[ -n "$TAU2_EXTRAS" ]] && extras_suffix="[$TAU2_EXTRAS]"
echo "[setup-tau2] pip install -e tau2-bench$extras_suffix  (no voice/audio unless TAU2_EXTRAS includes it)"
"$PIP" install -e "$TAU2_DIR$extras_suffix"
echo "[setup-tau2] pip install -e $BUNDLE_ROOT/code ..."
"$PIP" install -e "$BUNDLE_ROOT/code"

# 4) Verify: import + a domain's tasks file present. Reports clearly if `import
#    tau2` needs voice (then re-run with TAU2_EXTRAS=voice,knowledge).
echo "[setup-tau2] verifying ..."
"$PY" - "$TAU2_DIR" <<'PY'
import sys
from pathlib import Path
tau2_dir = Path(sys.argv[1])
try:
    import tau2  # noqa: F401 — runs tau2/__init__ (may pull voice/knowledge)
    import tau2.utils.llm_utils  # noqa: F401 — patched by the adapter
    from tau2.runner import build_text_orchestrator  # noqa: F401
except ModuleNotFoundError as e:
    name = getattr(e, "name", "") or str(e)
    if any(k in name for k in ("pyaudio", "elevenlabs", "voice", "soundfile", "librosa")):
        print(f"[setup-tau2] import tau2 NEEDS voice deps ({name}). "
              f"Re-run: TAU2_EXTRAS=voice,knowledge scripts/setup_tau2_bench.sh", file=sys.stderr)
    raise
tasks = tau2_dir / "data" / "tau2" / "domains" / "retail" / "tasks.json"
assert tasks.exists(), f"retail tasks.json missing at {tasks}"
print("[setup-tau2] OK: import tau2 + runner + retail tasks.json present (no voice)")
PY

echo "[setup-tau2] done. Run tau2 for real, e.g.:"
echo "  PYTHON=$TAU2_VENV/bin/python EXPERIMENT_CONFIG=tau2_retail N_TASKS=20 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=6 \\"
echo "    bash scripts/run_full_pipeline.sh --bench tau2_bench --n-cycles 1 --model-config 06_qwen3_4b_273"
