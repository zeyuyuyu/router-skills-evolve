#!/usr/bin/env bash
# code/training/orchestration/prepare_offline_artifacts.sh
# Prepare files needed to install the training stack on an offline GPU box.
#
# Run this on a networked Linux machine, then copy offline_artifacts/ to the
# offline training machine together with this bundle.
#
# Usage:
#   BUNDLE_ROOT=/path/to/bundle bash code/training/orchestration/prepare_offline_artifacts.sh
#
# Env vars:
#   BUNDLE_ROOT        default: inferred from this script path
#   ARTIFACTS_DIR      default: $BUNDLE_ROOT/offline_artifacts
#   PYTHON             default: python3.12
#   PREP_VENV          default: /tmp/evol-offline-prep
#   DOWNLOAD_MODELS    default: 1
#   INCLUDE_EVAL_DEPS  default: 0; set 1 to also fetch vllm + tau2-bench deps
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="${BUNDLE_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$BUNDLE_ROOT/offline_artifacts}"
PYTHON_BIN="${PYTHON:-python3.12}"
PREP_VENV="${PREP_VENV:-/tmp/evol-offline-prep}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"
INCLUDE_EVAL_DEPS="${INCLUDE_EVAL_DEPS:-0}"

WHEELHOUSE="$ARTIFACTS_DIR/wheelhouse"
HF_HOME_DIR="$ARTIFACTS_DIR/hf_home"
SRC_DIR="$ARTIFACTS_DIR/src"
MANIFEST_DIR="$ARTIFACTS_DIR/manifests"
TRAIN_REQ="$MANIFEST_DIR/requirements_train_only.txt"
TORCH_CONSTRAINT="$MANIFEST_DIR/constraints_offline.txt"

mkdir -p "$WHEELHOUSE" "$HF_HOME_DIR" "$SRC_DIR" "$MANIFEST_DIR"

cd "$BUNDLE_ROOT"

echo "=== Stage 1: prep venv on networked machine ==="
"$PYTHON_BIN" -m venv "$PREP_VENV"
# shellcheck source=/dev/null
source "$PREP_VENV/bin/activate"
python -m pip install -U pip wheel setuptools packaging huggingface-hub hf_transfer

echo ""
echo "=== Stage 2: build training-only requirements ==="
grep -vE '^(torch|flash-attn|vllm)==' "$BUNDLE_ROOT/code/training/requirements.txt" > "$TRAIN_REQ"
echo "  wrote $TRAIN_REQ"
cat > "$TORCH_CONSTRAINT" <<'EOF'
torch==2.11.0+cu130
cuda-bindings==13.0.3
cuda-pathfinder==1.2.2
fsspec==2026.2.0
setuptools==70.2.0
EOF
echo "  wrote $TORCH_CONSTRAINT"

echo ""
echo "=== Stage 3: download torch 2.11.0 cu130 wheel set ==="
python -m pip download \
    --dest "$WHEELHOUSE" \
    --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.11.0

echo ""
echo "=== Stage 4: download training dependencies ==="
python -m pip download \
    --dest "$WHEELHOUSE" \
    --find-links "$WHEELHOUSE" \
    --constraint "$TORCH_CONSTRAINT" \
    -r "$TRAIN_REQ"

echo ""
echo "=== Stage 5: download flash-attn source package + build helpers ==="
python -m pip download \
    --dest "$WHEELHOUSE" \
    ninja packaging wheel psutil numpy einops setuptools pip

# flash-attn's setup imports torch while generating build metadata, so install
# the already-downloaded CUDA torch and metadata helpers into the prep venv
# before downloading the sdist. This is only for metadata; the real build
# still happens on H200.
python -m pip install \
    --no-index --find-links "$WHEELHOUSE" \
    --constraint "$TORCH_CONSTRAINT" \
    ninja packaging wheel psutil numpy einops
python -m pip install \
    --no-index --find-links "$WHEELHOUSE" \
    --constraint "$TORCH_CONSTRAINT" \
    torch==2.11.0+cu130
python -m pip download \
    --dest "$WHEELHOUSE" \
    --no-deps \
    --no-build-isolation \
    flash-attn==2.8.3
if [[ "$INCLUDE_EVAL_DEPS" == "1" ]]; then
    echo ""
    echo "=== Stage 6: optional eval deps (vllm + tau2-bench) ==="
    TAU2_DIR="$SRC_DIR/tau2-bench"
    TAU2_PINNED_SHA="17e07b1da2bbc0cadfddeea36412686e0604127b"
    TAU2_REPO="https://github.com/sierra-research/tau2-bench"
    if [[ -d "$TAU2_DIR/.git" ]]; then
        git -C "$TAU2_DIR" fetch origin
    else
        git clone "$TAU2_REPO" "$TAU2_DIR"
    fi
    git -C "$TAU2_DIR" checkout "$TAU2_PINNED_SHA"
    git -C "$TAU2_DIR" rev-parse HEAD > "$MANIFEST_DIR/tau2-bench.sha"

    python -m pip download \
        --dest "$WHEELHOUSE" \
        "$TAU2_DIR[voice,knowledge]"
    python -m pip download \
        --dest "$WHEELHOUSE" \
        vllm==0.20.2
else
    echo ""
    echo "=== Stage 6: optional eval deps skipped (INCLUDE_EVAL_DEPS=0) ==="
fi

if [[ "$DOWNLOAD_MODELS" == "1" ]]; then
    echo ""
    echo "=== Stage 7: download Hugging Face snapshots ==="
    export HF_HOME="$HF_HOME_DIR"
    export HF_HUB_ENABLE_HF_TRANSFER=1
    python <<'PYEOF'
from huggingface_hub import snapshot_download

models = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
    ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    ("Qwen/Qwen3.6-35B-A3B", "995ad96eacd98c81ed38be0c5b274b04031597b0"),
]

for repo_id, revision in models:
    print(f"Downloading {repo_id}@{revision}")
    snapshot_download(repo_id=repo_id, revision=revision)
PYEOF
else
    echo ""
    echo "=== Stage 7: model download skipped (DOWNLOAD_MODELS=0) ==="
fi

echo ""
echo "=== Stage 8: write manifests ==="
find "$WHEELHOUSE" -maxdepth 1 -type f -printf '%f\n' | sort \
    > "$MANIFEST_DIR/wheelhouse_files.txt"
find "$HF_HOME_DIR" -type f -printf '%P\n' | sort \
    > "$MANIFEST_DIR/hf_home_files.txt"
find "$WHEELHOUSE" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$MANIFEST_DIR/wheelhouse_sha256.txt"
du -sh "$ARTIFACTS_DIR" > "$MANIFEST_DIR/size.txt"

echo ""
echo "=== Offline artifacts ready ==="
echo "Artifacts:  $ARTIFACTS_DIR"
echo "Wheelhouse: $WHEELHOUSE"
echo "HF_HOME:    $HF_HOME_DIR"
echo "Manifest:   $MANIFEST_DIR"
