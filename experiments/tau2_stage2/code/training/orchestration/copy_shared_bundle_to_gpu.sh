#!/usr/bin/env bash
# Copy the prepared 3FS bundle to local disk on the offline GPU machine.
#
# Run this on wanyichen-gpu02 after the 3FS path is visible there.
#
# Usage:
#   TARGET_ROOT=/local_nvme/evol-llm-tau2-stage2-ship \
#     bash /mnt/3fs2/data/root/evol/tau2_stage2_offline_20260512/\
#evol-llm-tau2-stage2-ship/code/training/orchestration/copy_shared_bundle_to_gpu.sh
#
# Env vars:
#   SHARED_ROOT  default: /mnt/3fs2/data/root/evol/tau2_stage2_offline_20260512
#   TARGET_ROOT  required; local destination bundle path on the GPU machine
set -euo pipefail

SHARED_ROOT="${SHARED_ROOT:-/mnt/3fs2/data/root/evol/tau2_stage2_offline_20260512}"
TARGET_ROOT="${TARGET_ROOT:?set TARGET_ROOT to a local path on the GPU machine}"
SOURCE_BUNDLE="$SHARED_ROOT/evol-llm-tau2-stage2-ship"

[[ -d "$SOURCE_BUNDLE" ]] || {
    echo "Missing shared bundle: $SOURCE_BUNDLE" >&2
    exit 1
}

mkdir -p "$TARGET_ROOT"

echo "=== Copy bundle code/data/wheelhouse ==="
rsync -a --delete \
    --exclude 'code/.venv/' \
    --exclude 'train_outputs/' \
    --exclude 'offline_artifacts/models/' \
    "$SOURCE_BUNDLE/" "$TARGET_ROOT/"

echo ""
echo "=== Copy model directories ==="
mkdir -p "$TARGET_ROOT/offline_artifacts/models"
rsync -aL --delete \
    "$SOURCE_BUNDLE/offline_artifacts/models/" \
    "$TARGET_ROOT/offline_artifacts/models/"

cat <<EOF

=== Copy complete ===
Bundle: $TARGET_ROOT

Next commands on the GPU machine:

  cd "$TARGET_ROOT"
  BUNDLE_ROOT="\$PWD" bash code/training/orchestration/setup_env_offline_venv.sh

  export BUNDLE_ROOT="\$PWD"
  export VENV="\$PWD/code/.venv/bin"
  export HF_HOME="\$PWD/offline_artifacts/hf_home"
  export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
  export EVOL_LOCAL_MODELS_DIR="\$PWD/offline_artifacts/models"
  export EVOL_REQUIRE_LOCAL_MODELS=1

  MAX_STEPS_OVERRIDE=1 bash code/training/orchestration/train_all.sh
EOF
