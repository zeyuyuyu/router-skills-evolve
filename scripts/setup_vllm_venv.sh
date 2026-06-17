#!/usr/bin/env bash
# setup_vllm_venv.sh
#
# Build a vLLM venv by CLONING the main training env (torch 2.11.0+cu129 +
# trl/transformers/peft/datasets) and adding vLLM on top, with torch PINNED so
# the install cannot drift to cu130. Result: vLLM coexists with the training
# stack on the SAME torch/CUDA — enabling GRPO colocate/server weight-sync and
# avoiding the cu130 mismatch from a plain `pip install vllm`.
#
# Usage:
#   scripts/setup_vllm_venv.sh [venv_dir]
# Env:
#   MAIN_PYTHON   python of the env to clone   (default: <repo>/venv/bin/python)
#   VLLM_VERSION  vLLM to install              (default: 0.22.0 — cu129-compatible)
#   CLONE_MODE    full | linked                (default: linked)
#                 linked = --system-site-packages (fast: see main's packages,
#                          only install vLLM + its unique deps into an overlay)
#                 full   = pip freeze + reinstall everything (true copy, slower)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAIN_PYTHON="${MAIN_PYTHON:-$REPO_ROOT/venv/bin/python}"
VENV_DIR="${1:-$REPO_ROOT/.vllm_venv}"
VLLM_VERSION="${VLLM_VERSION:-0.22.0}"
CLONE_MODE="${CLONE_MODE:-linked}"
TORCH_PIN="torch==2.11.0+cu129"
CU_INDEX="https://download.pytorch.org/whl/cu129"

echo "[setup] main python : $MAIN_PYTHON"
echo "[setup] target venv : $VENV_DIR"
echo "[setup] vllm version: $VLLM_VERSION   clone_mode: $CLONE_MODE"
"$MAIN_PYTHON" -c "import torch; print('[setup] main torch:', torch.__version__, torch.version.cuda)"

# 1. Fresh venv
rm -rf "$VENV_DIR"
if [[ "$CLONE_MODE" == "linked" ]]; then
  "$MAIN_PYTHON" -m venv --system-site-packages "$VENV_DIR"
else
  "$MAIN_PYTHON" -m venv "$VENV_DIR"
fi
PIP="$VENV_DIR/bin/pip"
"$PIP" install --upgrade pip wheel >/dev/null

# 2. Replicate the main env (full clone only; linked sees it via system packages)
if [[ "$CLONE_MODE" == "full" ]]; then
  FREEZE=/tmp/main_freeze.txt
  "$MAIN_PYTHON" -m pip freeze --exclude-editable \
    | grep -vE '@ file://|^-e |^torch==' > "$FREEZE"
  echo "[setup] cloning $(wc -l < "$FREEZE") packages from main env"
  "$PIP" install --extra-index-url "$CU_INDEX" "$TORCH_PIN"
  "$PIP" install -r "$FREEZE" --extra-index-url "$CU_INDEX"
fi

# 3. Install vLLM with torch PINNED (constraint) so it can't pull cu130 torch.
CONSTRAINTS=/tmp/vllm_constraints.txt
printf '%s\n' "$TORCH_PIN" > "$CONSTRAINTS"
echo "[setup] installing vllm==$VLLM_VERSION (torch pinned to cu129)"
"$PIP" install "vllm==$VLLM_VERSION" -c "$CONSTRAINTS" --extra-index-url "$CU_INDEX"

# 4. Verify the combined stack imports and torch stayed on cu129.
"$VENV_DIR/bin/python" - <<'PY'
import torch, vllm
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm ", vllm.__version__)
for m in ("transformers", "trl", "peft", "datasets"):
    try:
        print(f"{m:12}", __import__(m).__version__)
    except Exception as e:  # noqa: BLE001
        print(f"{m:12} MISSING ({e})")
assert torch.version.cuda == "12.9", f"torch CUDA drifted to {torch.version.cuda}, expected 12.9"
print("[setup] verify OK — vllm + training stack on torch cu129")
PY
echo "[setup] DONE → $VENV_DIR"
echo "[setup] use with: HE_VLLM_VENV=$VENV_DIR  (serve script picks it up)"
