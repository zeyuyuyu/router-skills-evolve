# Offline No-Conda TODO

Last updated: 2026-05-19 21:50 CST

This file is the working checklist for running this bundle on `wanyichen-gpu02`
without conda and without internet access on the GPU machine. Update the
checkboxes and notes after each command.

## Current State

- [x] Bundle extracted at `/root/cwy/projects/evol/evol-llm-tau2-stage2-ship`.
- [x] Tarball integrity checked: `gzip -t` passed.
- [x] Python syntax check passed for `code/**/*.py`.
- [x] Shell syntax check passed for orchestration scripts.
- [x] YAML configs parse.
- [x] Data conversion smoke passed locally:
  - train: 6413 raw rows, 6156 kept, 257 dropped for no user prefix
  - val: 394 raw rows, 378 kept, 16 dropped for no user prefix
- [x] Pure Python tests passed locally: 66 passed.
- [x] Chat-template unit tests passed locally: 4 passed.
- [x] `OPENAI_API_KEY` is not present in the bundle or current shell.
- [x] `HF_TOKEN` is not present in the bundle or current shell.
- [x] Added helper script: `code/training/orchestration/prepare_offline_artifacts.sh`.
- [x] Added helper script: `code/training/orchestration/setup_env_offline_venv.sh`.
- [x] Added helper script: `code/training/orchestration/copy_shared_bundle_to_gpu.sh`.
- [x] Added offline local-model resolver so YAML model ids can map to copied
      model directories without editing all run configs.
- [x] `wanyichen-gpu02` hardware inspected through `10.100.0.62:35822`.
      The DNS alias does not resolve from this shell, but the IP/port path works.
- [x] Offline training wheelhouse prepared at `offline_artifacts/wheelhouse`
      (`DOWNLOAD_MODELS=0`, eval deps skipped).
- [x] 3FS model inventory checked on current machine. Some required models are
      present as direct HF-format directories, but not the full run set.
- [x] Complete local model directory prepared on 3FS for all training/smoke
      models under the shared bundle.
- [x] Shared 3FS handoff bundle prepared:
      `/mnt/3fs2/data/root/evol/tau2_stage2_offline_20260512/evol-llm-tau2-stage2-ship`.
- [x] Local model resolver verified against shared model dirs; 2B val
      chat-template check passed: 378/378 rows.
- [x] `10.100.0.62:35822` SSH checked: hostname
      `lg-cmc-b7r201-c06u06-h200-000061`, user `root`.
- [x] `wanyichen-gpu02` equivalent hardware inspected via `10.100.0.62:35822`:
      8x H200, Python 3.12.3, 3FS visible, about 211T free on `/`.
- [x] CUDA 13.0 `nvcc` found at `/usr/local/cuda-13.0/bin/nvcc`; setup script
      now auto-adds it to PATH if needed.
- [x] Shared bundle copied to GPU local disk:
      `/root/cwy/projects/evol/evol-llm-tau2-stage2-ship` on `10.100.0.62:35822`.
- [x] Copied GPU bundle is about 102G; model entries are real directories,
      not symlinks.
- [x] Fixed setup preflight to avoid `nvidia-smi | grep -q` SIGPIPE under
      `set -o pipefail`; it now uses `nvidia-smi --query-gpu=name`.
- [x] Fixed the same H200 preflight SIGPIPE issue in `train_all.sh`.
- [x] Fixed flash-attn offline install path: added `einops` to wheelhouse
      inputs and install flash-attn with `--no-deps` after helper deps.
- [x] Fixed flash-attn build target for H200: `FLASH_ATTN_CUDA_ARCHS=90`
      instead of default `80;90;100;120`.
- [x] No-conda venv installed on `wanyichen-gpu02` via `10.100.0.62:35822`.
- [x] Built flash-attn wheel saved back into wheelhouse:
      `flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl`.
- [x] GPU local gates passed: shell syntax OK; non-smoke tests
      `81 passed, 3 skipped`.
- [x] First production 2B one-step smoke found a real VRAM issue; DDP run
      configs were moved to a lower-memory batch/accum plan and
      `train_all.sh` now defaults `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- [x] 2B DDP one-step smoke passed on `wanyichen-gpu02` after the low-memory
      config patch; `STATUS=smoke_done`.
- [x] 4B DDP one-step smoke passed on `wanyichen-gpu02` after the same patch;
      `STATUS=smoke_done`.
- [x] Added `checkpoint-best` publication in `train.py` so eval uses the
      Trainer-selected best checkpoint path instead of falling back to final.
- [x] Split FSDP2 auto-wrap configs by model family: dense Qwen3.5 uses
      `accelerate_fsdp2_dense.yaml`; Qwen3.6-35B-A3B MoE uses
      `accelerate_fsdp2_moe.yaml`.
- [x] 9B and 35B staged one-step smoke train passed on `wanyichen-gpu02`.
      Both publish `checkpoint-best -> checkpoint-1`.
- [x] 35B-A3B MoE checkpoint key sanity passed: safetensors contain 693 HF
      parameter keys and 0 `_checkpoint_wrapped_module` keys.
- [x] Full train started on `10.100.0.62:35822` in tmux session
      `tau2-train`; watch `train.log`.
- [x] Full train runs `01_qwen3_5_2b_273` through
      `05_qwen3_5_4b_273` completed successfully.
- [x] `06_qwen3_5_9b_50` exposed a FSDP2 end-of-train issue in
      Transformers: `load_best_model_at_end` tries to copy regular tensors into
      DTensor-wrapped params. The failed output was archived under
      `train_outputs/06_qwen3_5_9b_50_failed_loadbest_20260515_162424`.
- [x] Patched FSDP runs `06`, `07`, and `08` to skip the in-process best-model
      reload. `train.py` still publishes `checkpoint-best` from
      `trainer.state`, so eval gets the intended best checkpoint.
- [x] Remote YAML regression test passed after the patch:
      `17 passed in 0.11s` in the GPU venv.
- [x] Full train restarted on `10.100.0.62:35822` at
      `2026-05-15 16:24 CST`. Runs `01`-`05` are skipped as done; run `06` is
      running again.
- [x] Full train runs `06_qwen3_5_9b_50`, `07_qwen3_5_9b_273`, and
      `08_qwen3_6_35b_a3b_273` completed successfully. The 35B-A3B run
      completed at `2026-05-18 02:44 CST`.
- [x] Current run on `2026-05-18 11:31 CST`:
      `09_qwen3_5_4b_273_lr1e5` is running at `60/210` steps with tqdm ETA
      about `21:26:14`.
- [x] User requested the experiment stop at `2026-05-18 11:37 CST`.
      The `tau2-train` tmux session and training processes were stopped, GPUs
      returned to idle, and `09_qwen3_5_4b_273_lr1e5/STATUS` was set to
      `interrupted`. Last recorded tqdm progress was `60/210`; partial
      checkpoint `checkpoint-42` is present.
- [x] User requested training resume at `2026-05-18 16:50 CST`. Added
      auto-resume-from-latest-checkpoint logic to `train.py`, synced it to GPU
      and 3FS, and restarted `tau2-train`. The log confirms
      `09_qwen3_5_4b_273_lr1e5` resumed from `checkpoint-42`.
- [x] On `2026-05-18 23:23-23:30 CST`, runs `09` and `10` received external
      `SIGTERM` and the tmux pipeline exited. This was not an OOM or Python
      training exception. `09` saved `checkpoint-84`; `10` had not saved a
      checkpoint yet.
- [x] User asked for status on `2026-05-19 21:40 CST`; GPUs were idle and
      `09`/`10` were marked failed from the external signal. Restarted
      `tau2-train` at `2026-05-19 21:41 CST`. The log confirms `09` resumed
      from `checkpoint-84`.
- [x] Per-run training loss plots for completed runs `01`-`05` were generated
      and copied to `/root/cwy/projects/evol/temp`.
- [ ] Full train passed on `wanyichen-gpu02`.
- [ ] Remaining train runs: finish `09_qwen3_5_4b_273_lr1e5`, then run
      `10_qwen3_5_4b_273_lr3e5`.
- [ ] Eval machine selected. GPU machine is offline, so eval needs either network
      access elsewhere or a temporary network exception.
- [ ] Final `SUMMARY.csv` and plots produced.

## What This Repo Does

This is a cloud handoff bundle for tau2-bench Stage-2 supervised fine-tuning.
It trains 10 Qwen student-model variants on the prepared Stage-2 corpus, then
evaluates the fully trained checkpoints with tau2-bench.

Training does not need OpenAI or external APIs once dependencies and Hugging
Face model weights are already local.

Eval does need OpenAI by default, because tau2's user simulator and NL judge
use `OPENAI_API_KEY`. Since `wanyichen-gpu02` cannot access the internet, the
recommended split is:

1. Train offline on `wanyichen-gpu02`.
2. Copy checkpoints to a networked GPU box for vLLM + tau2 eval.
3. Produce `SUMMARY.csv` and plots on the eval box.

## Machines

| Machine | Network | Role | Notes |
|---|---:|---|---|
| local/current box | yes | inspect bundle, prepare docs | Not suitable for full H200 training. |
| `wanyichen-gpu02` | no | offline training | Needs Python 3.12, CUDA/nvcc, venv, wheels, copied local model dirs. |
| networked GPU box | yes | eval | Needed for OpenAI-backed tau2 eval unless network is enabled on `wanyichen-gpu02`. |

## Shared 3FS Handoff Directory

Prepared on 2026-05-12:

```text
/mnt/3fs2/data/root/evol/tau2_stage2_offline_20260512/
  evol-llm-tau2-stage2-ship/
    code/
    data_processed/
    docs/
    offline_artifacts/
      wheelhouse/
      hf_home/
      manifests/
      models/
```

Important paths:

- Shared bundle:
  `/mnt/3fs2/data/root/evol/tau2_stage2_offline_20260512/evol-llm-tau2-stage2-ship`
- Shared local models:
  `/mnt/3fs2/data/root/evol/tau2_stage2_offline_20260512/evol-llm-tau2-stage2-ship/offline_artifacts/models`
- Model manifest:
  `offline_artifacts/manifests/local_models.tsv`

Model layout:

| Model id in YAML | Shared model dir | Size |
|---|---|---:|
| `Qwen/Qwen2.5-0.5B-Instruct` | `offline_artifacts/models/Qwen2.5-0.5B-Instruct` | 954M |
| `Qwen/Qwen3.5-2B` | `offline_artifacts/models/Qwen3.5-2B` | 4.3G |
| `Qwen/Qwen3.5-4B` | `offline_artifacts/models/Qwen3.5-4B` | 8.8G |
| `Qwen/Qwen3.5-9B` | `offline_artifacts/models/Qwen3.5-9B` | 19G |
| `Qwen/Qwen3.6-35B-A3B` | `offline_artifacts/models/Qwen3.6-35B-A3B` | 67G |

The 4B/9B/35B entries are symlinks to existing 3FS model directories in the
shared bundle. The GPU copy script uses `rsync -aL`, so those links are copied
as real model files onto the GPU local disk.

Size notes:

- Shared bundle itself, without dereferencing symlinks: about 8.0G.
- Actual GPU copy size for `offline_artifacts/models` with symlinks
  dereferenced: about 99G.
- Wheelhouse: about 2.7G.

Copy from 3FS to local disk on `wanyichen-gpu02`:

```bash
TARGET_ROOT=/local/path/evol-llm-tau2-stage2-ship \
  bash /mnt/3fs2/data/root/evol/tau2_stage2_offline_20260512/evol-llm-tau2-stage2-ship/code/training/orchestration/copy_shared_bundle_to_gpu.sh
```

After copying, use local model resolution for all offline train commands:

```bash
export EVOL_LOCAL_MODELS_DIR="$PWD/offline_artifacts/models"
export EVOL_REQUIRE_LOCAL_MODELS=1
```

## Expected Runtime

| Stage | Estimated Time | Notes |
|---|---:|---|
| Offline artifact preparation | 1-3 hours | Depends on internet speed and model downloads. |
| Copy artifacts to GPU machine | depends | Model cache is large. |
| venv install from local files | 30-90 min | `flash-attn` build can take 30-60 min. |
| smoke train | 10 min-1.5 hours per staged run | Observed 1-step train runtimes: 2B 291.8s, 4B 624.8s, 9B 579.3s, 35B 684.7s. Wall clock includes tokenization, eval, save, and FSDP gather overhead. |
| full train | likely multi-day train-only | Original README estimate was 12-15 hours, but it is no longer trusted after the low-memory DDP/FSDP changes. 35B alone is roughly 443 optimizer steps at the current global batch, so the full 10-run sweep should be treated as a long tmux job and watched from `train.log`. |
| eval | 3-6 hours | Requires OpenAI key and network. |
| full train + eval | full train multi-day + 3-6 hours eval | Eval cannot run on the offline GPU without OpenAI network access. |

## Phase 0 - Inspect `wanyichen-gpu02`

Goal: confirm the offline machine has the hardware and base system we need.

Run from a machine that can SSH to `wanyichen-gpu02`:

```bash
ssh wanyichen-gpu02 '
set -e
hostname
nvidia-smi
which python3 || true
python3 --version || true
which python3.12 || true
python3.12 --version || true
which nvcc || true
nvcc --version || true
df -h .
free -h
'
```

Expected:

- 8 H200 GPUs visible.
- Python 3.12 available, or root/admin can install Python 3.12 without conda.
- CUDA toolkit with `nvcc`, ideally CUDA 13.0.
- Enough disk for model cache, environments, and checkpoints.

Status:

- [ ] SSH access works.
- [ ] GPU count confirmed.
- [ ] CUDA/nvcc confirmed.
- [ ] Python 3.12 confirmed.
- [ ] Free disk recorded here:

Notes:

```text
2026-05-12: From /root/cwy/projects/evol, `ssh wanyichen-gpu02` failed with:
Could not resolve hostname wanyichen-gpu02: Name or service not known.
Need to run this phase from a machine on the same network or with the right
SSH alias / DNS config.

2026-05-12: User provided `10.100.0.62 35822`. SSH works:
- hostname: `lg-cmc-b7r201-c06u06-h200-000061`
- user: `root`
- GPUs: 8x NVIDIA H200, 143771 MiB each, driver 570.124.06
- Python: `/usr/bin/python3.12`, version 3.12.3
- 3FS: `/mnt/3fs` and `/mnt/3fs2` mounted
- disk: `/` has about 211T available
- CUDA 13.0 nvcc exists at `/usr/local/cuda-13.0/bin/nvcc`, but was not in PATH
  by default; `setup_env_offline_venv.sh` now auto-discovers it.
```

## Phase 1 - Prepare Offline Artifacts On A Networked Machine

Goal: create one directory that can be copied to the offline GPU box.

This phase prepares the training environment first. Do not force eval-only
packages onto `wanyichen-gpu02`: training does not need `vllm`, tau2-bench, or
OpenAI. Eval can run later on a networked GPU machine.

Output directory:

```text
offline_artifacts/
  wheelhouse/
  hf_home/
  models/
  src/tau2-bench/
  manifests/
```

Run from the bundle root on a networked Linux machine.

Recommended scripted path:

```bash
cd /path/to/evol-llm-tau2-stage2-ship
BUNDLE_ROOT="$PWD" \
DOWNLOAD_MODELS=1 \
INCLUDE_EVAL_DEPS=0 \
  bash code/training/orchestration/prepare_offline_artifacts.sh
```

Set `INCLUDE_EVAL_DEPS=1` only if the eval machine also needs offline
installation. For `wanyichen-gpu02`, keep it `0` because training does not need
`vllm` or tau2-bench.

Manual equivalent:

```bash
cd /path/to/evol-llm-tau2-stage2-ship
mkdir -p offline_artifacts/wheelhouse offline_artifacts/hf_home offline_artifacts/src offline_artifacts/manifests
python3.12 -m venv /tmp/evol-offline-prep
source /tmp/evol-offline-prep/bin/activate
python -m pip install -U pip wheel setuptools packaging huggingface-hub hf_transfer
```

Download Python wheels:

```bash
grep -vE '^(torch|flash-attn|vllm)==' code/training/requirements.txt \
  > offline_artifacts/manifests/requirements_train_only.txt

# Torch cu130 wheels live on the PyTorch index.
python -m pip download \
  --dest offline_artifacts/wheelhouse \
  --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.11.0

# Training deps only. vLLM is eval-only and is intentionally excluded here.
python -m pip download \
  --dest offline_artifacts/wheelhouse \
  -r offline_artifacts/manifests/requirements_train_only.txt

# flash-attn source package. It is built on the H200 machine with local nvcc.
python -m pip download \
  --dest offline_artifacts/wheelhouse \
  --no-deps \
  flash-attn==2.8.3

# Build/install helpers for source builds.
python -m pip download \
  --dest offline_artifacts/wheelhouse \
  ninja packaging wheel psutil setuptools pip
```

Optional eval-only artifacts:

Only do this if the eval machine also cannot use the internet. If eval runs on
a normal networked GPU box, it is simpler to install eval deps there directly.

Vendor tau2-bench at the pinned SHA:

```bash
git clone https://github.com/sierra-research/tau2-bench offline_artifacts/src/tau2-bench
git -C offline_artifacts/src/tau2-bench checkout 17e07b1da2bbc0cadfddeea36412686e0604127b

# Collect tau2-bench dependencies into the wheelhouse.
python -m pip download \
  --dest offline_artifacts/wheelhouse \
  "./offline_artifacts/src/tau2-bench[voice,knowledge]"

# vLLM is eval-only.
python -m pip download \
  --dest offline_artifacts/wheelhouse \
  vllm==0.20.2
```

Download Hugging Face model snapshots into the portable cache:

```bash
export HF_HOME="$PWD/offline_artifacts/hf_home"
export HF_HUB_ENABLE_HF_TRANSFER=1

python - <<'PY'
from huggingface_hub import snapshot_download

models = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
    ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    ("Qwen/Qwen3.6-35B-A3B", "995ad96eacd98c81ed38be0c5b274b04031597b0"),
]

for repo_id, revision in models:
    print("Downloading", repo_id, revision)
    snapshot_download(repo_id=repo_id, revision=revision)
PY
```

Write manifests:

```bash
find offline_artifacts/wheelhouse -maxdepth 1 -type f -printf '%f\n' | sort \
  > offline_artifacts/manifests/wheelhouse_files.txt
find offline_artifacts/hf_home -type f -printf '%P\n' | sort \
  > offline_artifacts/manifests/hf_home_files.txt
if [ -d offline_artifacts/src/tau2-bench/.git ]; then
  git -C offline_artifacts/src/tau2-bench rev-parse HEAD \
    > offline_artifacts/manifests/tau2-bench.sha
fi
du -sh offline_artifacts > offline_artifacts/manifests/size.txt
```

Status:

- [x] `offline_artifacts/wheelhouse` prepared.
- [x] Shared `offline_artifacts/models` prepared with all 5 required model ids.
- [ ] `offline_artifacts/hf_home` prepared. Not required for the current
      offline training route because local model directories are used instead.
- [ ] `offline_artifacts/src/tau2-bench` prepared if offline eval is needed.
- [x] Manifests written.
- [x] Artifacts copied to `wanyichen-gpu02` via `10.100.0.62:35822`.

Notes:

```text
2026-05-12: Ran
`BUNDLE_ROOT=$PWD DOWNLOAD_MODELS=0 INCLUDE_EVAL_DEPS=0 bash code/training/orchestration/prepare_offline_artifacts.sh`.
Result: wheelhouse ready, 94 files, offline_artifacts size about 2.7G.
Important fixes made after trial runs:
- constrained torch to `torch==2.11.0+cu130` so dependency resolution does not
  pull a non-cu130 torch candidate;
- installed torch/numpy/psutil into the prep venv before downloading
  `flash_attn-2.8.3.tar.gz`, because flash-attn metadata imports them.

2026-05-12: Prepared shared 3FS handoff bundle at
`/mnt/3fs2/data/root/evol/tau2_stage2_offline_20260512/evol-llm-tau2-stage2-ship`.
Downloaded missing models into `offline_artifacts/models`:
- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen3.5-2B`

Linked existing 3FS model dirs into `offline_artifacts/models`:
- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.5-9B`
- `Qwen/Qwen3.6-35B-A3B`

2026-05-12: Copied the shared bundle to GPU local disk with:
`TARGET_ROOT=/root/cwy/projects/evol/evol-llm-tau2-stage2-ship bash .../copy_shared_bundle_to_gpu.sh`.
Result on `10.100.0.62:35822`:
- bundle path: `/root/cwy/projects/evol/evol-llm-tau2-stage2-ship`
- total copied size: about 102G
- model dirs are real directories, not symlinks
```

## 3FS Model Inventory

Checked from the current machine on 2026-05-12. Both 3FS mounts are visible:

- `/mnt/3fs`
- `/mnt/3fs2`

The exact Hugging Face cache for the run configs is not complete. Instead,
the handoff now uses local HF-format model directories plus
`training.model_resolution`, so the YAMLs can keep their original `Qwen/...`
model ids.

Present:

| Needed for | Current YAML model | 3FS path | Status |
|---|---|---|---|
| runs 02-05, 09-10 | `Qwen/Qwen3.5-4B` | `/mnt/3fs2/data/axis_eval_share_model/Qwen35-4B_ffc4be93-6837-4bb0-9ac9-522e011c176a/hf` | present, HF format, 8.8G |
| runs 06-07 | `Qwen/Qwen3.5-9B` | `/mnt/3fs/data/shared_models/Qwen/Qwen3.5-9B` | present, HF format, 19G |
| run 08 | `Qwen/Qwen3.6-35B-A3B` | `/mnt/3fs/data/shared_models/Qwen/Qwen3.6-35B-A3B` | present, HF format, 67G |
| run 08 alt path | `Qwen/Qwen3.6-35B-A3B` | `/mnt/3fs2/data/axis_eval_share_model/Qwen36-35B-A3B_b9d63cdc-f5fc-4a32-b970-d5510049df90/hf` | present, HF format, 67G |

Originally missing / not found in the checked 3FS paths:

- `Qwen/Qwen3.5-2B` for run `01_qwen3_5_2b_273`
- `Qwen/Qwen2.5-0.5B-Instruct` for `runs_smoke/smoke_2b.yaml`

Resolved on 2026-05-12:

- Both missing models were downloaded into the shared bundle's
  `offline_artifacts/models`.
- All 5 model tokenizers were loaded locally with
  `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 EVOL_REQUIRE_LOCAL_MODELS=1`.
- `Qwen/Qwen3.5-2B` chat-template validation passed on local val cache:
  378/378 rows, 0 failures.
- No run YAML patch is needed.

## Phase 2 - Create No-Conda Environment On `wanyichen-gpu02`

Goal: create `code/.venv` and install the training stack without internet.

Run on `wanyichen-gpu02` from the bundle root.

Recommended scripted path:

```bash
cd /path/to/evol-llm-tau2-stage2-ship
BUNDLE_ROOT="$PWD" \
ARTIFACTS_DIR="$PWD/offline_artifacts" \
PYTHON=python3.12 \
  bash code/training/orchestration/setup_env_offline_venv.sh
```

Manual equivalent:

```bash
cd /path/to/evol-llm-tau2-stage2-ship
export BUNDLE_ROOT="$PWD"
export VENV="$BUNDLE_ROOT/code/.venv/bin"
export HF_HOME="$BUNDLE_ROOT/offline_artifacts/hf_home"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export EVOL_LOCAL_MODELS_DIR="$BUNDLE_ROOT/offline_artifacts/models"
export EVOL_REQUIRE_LOCAL_MODELS=1

python3.12 -m venv "$BUNDLE_ROOT/code/.venv"
"$VENV/python" -m pip install --no-index --find-links "$BUNDLE_ROOT/offline_artifacts/wheelhouse" -U pip setuptools wheel packaging
"$VENV/python" -m pip install --no-index --find-links "$BUNDLE_ROOT/offline_artifacts/wheelhouse" torch==2.11.0
"$VENV/python" -m pip install --no-index --find-links "$BUNDLE_ROOT/offline_artifacts/wheelhouse" ninja packaging wheel psutil

# flash-attn is a source build; keep MAX_JOBS conservative.
export MAX_JOBS="${MAX_JOBS:-4}"
"$VENV/python" -m pip install --no-index --find-links "$BUNDLE_ROOT/offline_artifacts/wheelhouse" --no-build-isolation flash-attn==2.8.3

"$VENV/python" -m pip install --no-index --find-links "$BUNDLE_ROOT/offline_artifacts/wheelhouse" -r "$BUNDLE_ROOT/offline_artifacts/manifests/requirements_train_only.txt"
"$VENV/python" -m pip install --no-deps -e "$BUNDLE_ROOT/code"
```

Important:

- The original `setup_env_server.sh` is conda-based. Do not use it for this
  route.
- All orchestration scripts support no-conda through `VENV=$BUNDLE_ROOT/code/.venv/bin`.
- Offline model loading is controlled by `EVOL_LOCAL_MODELS_DIR`; keep it set
  to `$BUNDLE_ROOT/offline_artifacts/models` on the GPU machine.
- If `flash-attn` fails, capture the full build log and check CUDA/nvcc first.
- This training venv intentionally does not require `vllm` or tau2-bench.
- If later using the same machine for eval, install `vllm` and tau2-bench as an
  extra step. tau2-bench voice imports may need a system `portaudio` library,
  which is one reason eval is cleaner on a networked machine.

Verification:

```bash
"$VENV/python" <<'PY'
import torch, transformers, accelerate, trl, peft, datasets
import flash_attn, matplotlib, pandas
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("trl", trl.__version__)
print("flash_attn", flash_attn.__version__)
print("training imports OK")
PY
```

Expected:

- `torch.cuda.is_available()` true.
- `torch.cuda.device_count()` is 8.
- `flash_attn` imports.
- Training imports pass.

Status:

- [x] venv created.
- [x] torch installed.
- [x] flash-attn built.
- [x] requirements installed.
- [x] bundle code installed editable.
- [x] import verification passed.

Notes:

```text
2026-05-12: Installed on `10.100.0.62:35822` at
`/root/cwy/projects/evol/evol-llm-tau2-stage2-ship/code/.venv`.
Import check passed:
- torch 2.11.0+cu130, CUDA 13.0, 8 GPUs
- transformers 5.8.0
- accelerate 1.13.0
- trl 1.4.0
- peft 0.19.1
- datasets 4.8.5
- flash_attn 2.8.3

The first flash-attn build attempted default archs `80;90;100;120`; stopped it
and rebuilt with `FLASH_ATTN_CUDA_ARCHS=90` for H200. Built wheel was copied to
`offline_artifacts/wheelhouse/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl`
and manifests were refreshed, so future installs should use the wheel.
```

## Phase 3 - Local Data And Config Gates On GPU Machine

Goal: catch cheap problems before launching GPU training.

Run from the bundle root:

```bash
export BUNDLE_ROOT="$PWD"
export VENV="$BUNDLE_ROOT/code/.venv/bin"
export HF_HOME="$BUNDLE_ROOT/offline_artifacts/hf_home"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export EVOL_LOCAL_MODELS_DIR="$BUNDLE_ROOT/offline_artifacts/models"
export EVOL_REQUIRE_LOCAL_MODELS=1

for s in code/training/orchestration/*.sh code/training/eval/vllm_serve.sh; do
  bash -n "$s"
done

cd "$BUNDLE_ROOT/code"
"$VENV/python" -m pytest training/tests -q --ignore=training/tests/test_smoke_train.py
cd "$BUNDLE_ROOT"
```

If full pytest is blocked by environment-specific warnings, run the already
validated pure-Python subset first:

```bash
cd "$BUNDLE_ROOT/code"
"$VENV/python" -m pytest -c /dev/null -q \
  training/tests/test_convert_to_prompt_completion.py \
  training/tests/test_domain_rebalance.py \
  training/tests/test_heldout_split.py \
  training/tests/test_tool_shuffle.py \
  training/tests/test_configs.py \
  training/tests/test_run_yamls.py \
  training/tests/test_summarize.py \
  training/tests/test_plotting.py \
  training/tests/test_local_provider.py \
  training/tests/test_harness.py \
  training/tests/test_validate_chat_template.py
cd "$BUNDLE_ROOT"
```

Status:

- [x] Shell syntax passed on GPU machine.
- [x] Unit tests passed on GPU machine.
- [x] Chat-template validation can read local copied model dirs.
- [ ] Offline HF cache validation is not required for the current local-model route.

Notes:

```text
2026-05-12 on `10.100.0.62:35822`:
- shell syntax passed for orchestration scripts and `vllm_serve.sh`
- non-smoke pytest passed: 81 passed, 3 skipped
- `train_all.sh` preflight converted data and validated all 2B/4B/9B/35B
  tokenizers from local copied model dirs:
  train 6156/6156 and val 378/378 passed for each model family.
```

## Phase 4 - Smoke Training

Goal: prove the training path works before committing to 12-15 hours.

First run one production config for one step using the real full data path:

```bash
cd /path/to/evol-llm-tau2-stage2-ship
export BUNDLE_ROOT="$PWD"
export VENV="$BUNDLE_ROOT/code/.venv/bin"
export HF_HOME="$BUNDLE_ROOT/offline_artifacts/hf_home"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export EVOL_LOCAL_MODELS_DIR="$BUNDLE_ROOT/offline_artifacts/models"
export EVOL_REQUIRE_LOCAL_MODELS=1

ONLY_RUN=01_qwen3_5_2b_273 MAX_STEPS_OVERRIDE=1 \
  bash code/training/orchestration/train_all.sh 2>&1 | tee smoke_train_01.log
```

Expected output:

- `train_outputs/_data_cache/train_prompt_completion.jsonl`
- `train_outputs/_data_cache/val_prompt_completion.jsonl`
- `train_outputs/_data_cache/heldout_task_ids.json`
- `train_outputs/_data_cache/heldout_tasks.jsonl`
- `train_outputs/01_qwen3_5_2b_273/checkpoint-best/`
- `train_outputs/01_qwen3_5_2b_273/STATUS` containing `smoke_done`
- `train_outputs/01_qwen3_5_2b_273/train_stdout.log`

Then run staged smoke checks for larger paths:

```bash
ONLY_RUN=06_qwen3_5_9b_50 MAX_STEPS_OVERRIDE=1 \
  bash code/training/orchestration/train_all.sh 2>&1 | tee smoke_train_06.log

ONLY_RUN=08_qwen3_6_35b_a3b_273 MAX_STEPS_OVERRIDE=1 \
  bash code/training/orchestration/train_all.sh 2>&1 | tee smoke_train_08.log
```

Expected:

- 9B FSDP2 smoke finishes with `STATUS=smoke_done`.
- 35B-A3B MoE smoke finishes with `STATUS=smoke_done`.
- No NaN losses in `training_log.json`.
- Peak VRAM is below H200 capacity.

Status:

- [x] 2B one-step smoke passed.
- [x] 4B one-step DDP smoke passed (`02_qwen3_5_4b_50`).
- [x] 9B FSDP2 smoke passed.
- [x] 35B-A3B MoE smoke passed.

Notes:

```text
2026-05-12: Initial command
`ONLY_RUN=01_qwen3_5_2b_273 MAX_STEPS_OVERRIDE=1 bash code/training/orchestration/train_all.sh`
reached the first training step, then OOMed on all 8 H200 ranks. Each process
was using about 139 GiB with the original DDP settings:
- per_device_train_batch_size=4
- gradient_accumulation_steps=2
- gradient_checkpointing=false
- max_seq_length=32768
- packing=true, padding_free=true

This means the environment, offline model loading, data conversion, heldout
split, and chat-template validation all work; the blocker was the production
DDP memory plan.

Applied fix:
- DDP runs 01-05 and 09-10 now use per_device_train_batch_size=2,
  gradient_accumulation_steps=4, per_device_eval_batch_size=2, and
  gradient_checkpointing=true. Effective global batch remains 64.
- `train_all.sh` now sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by default.

Next action: sync the patched configs to the shared bundle and GPU copy, move
the failed `train_outputs/01_qwen3_5_2b_273` aside, then rerun the 2B one-step
smoke.

2026-05-12: Re-ran the 2B one-step smoke after the DDP low-memory patch:
- command log: `smoke_train_01_lowmem.log`
- status: `train_outputs/01_qwen3_5_2b_273/STATUS=smoke_done`
- `global_step=1`
- `train_loss=0.16373974084854126`
- `eval_loss=0.24673070013523102`
- `train_runtime=291.8094s`
- observed training-step VRAM: about 37-45 GiB/GPU, leaving about 98-106 GiB
  free per H200
- output size: about 15G

Follow-up fix from the smoke: Trainer's internal `load_best_model_at_end`
uses direct `load_state_dict` and emits missing/unexpected key warnings for
Qwen3.5 checkpoints whose raw safetensor keys are `model.language_model.*`.
The checkpoints themselves load cleanly through `AutoModelForCausalLM`, but
`eval_all.sh` expects a stable `checkpoint-best` path. `train.py` now publishes
`checkpoint-best` as a symlink to `trainer.state.best_model_checkpoint`; the
existing smoke output was patched with `checkpoint-best -> checkpoint-1`.

2026-05-12: Re-ran one 4B DDP smoke after the same low-memory patch:
- command log: `smoke_train_02_lowmem.log`
- status: `train_outputs/02_qwen3_5_4b_50/STATUS=smoke_done`
- `global_step=1`
- `train_loss=0.19749587774276733`
- `eval_loss=0.19075793027877808`
- `train_runtime=624.8409s`
- observed training-step VRAM: about 66-85 GiB/GPU, leaving about 58-77 GiB
  free per H200
- output size: about 32G
- `checkpoint-best -> checkpoint-1` was published correctly.

2026-05-12: First 9B FSDP2 smoke attempt failed before the first training step:
- command log: `smoke_train_06_fsdp1.log`
- status: `train_outputs/06_qwen3_5_9b_50/STATUS=failed`
- error: Accelerate FSDP2 could not find `Qwen3_5MoeDecoderLayer` in the 9B
  dense model.

Root cause: `accelerate_fsdp2.yaml` mixed dense and MoE decoder layer classes
in `fsdp_transformer_layer_cls_to_wrap`. Accelerate FSDP2 validates that every
listed class exists in the current model.

Applied fix:
- Added `configs/accelerate_fsdp2_dense.yaml` with only `Qwen3_5DecoderLayer`.
- Added `configs/accelerate_fsdp2_moe.yaml` with only
  `Qwen3_5MoeDecoderLayer`.
- Runs 06/07 now point to the dense config; run 08 points to the MoE config.
- The old `configs/accelerate_fsdp2.yaml` is now a dense compatibility default.

2026-05-12: Re-ran 9B FSDP2 with `accelerate_fsdp2_dense.yaml`:
- command log: `smoke_train_06_fullstate_skipfinal.log`
- status: `train_outputs/06_qwen3_5_9b_50/STATUS=smoke_done`
- `global_step=1`
- `train_loss=0.217498779296875`
- `eval_loss=0.17539429664611816`
- `train_runtime=579.3011s`
- output size: about 67G
- `checkpoint-best -> checkpoint-1` was published correctly.
- FSDP final duplicate save is skipped; the `checkpoint-1` FULL_STATE_DICT
  checkpoint is the HF checkpoint used by eval.

2026-05-12: First 35B-A3B MoE attempts exposed three real issues:
- pre-train aux-loss sample forward failed under FSDP2 because CPU-efficient
  loading left the model on CPU while flash-attn has no CPU backend;
- 32k context OOMed in the Qwen3.6 MoE/linear-attention path;
- 16k context with whole-decoder MoE FSDP wrapping still OOMed during FSDP
  all-gather.

Applied fix:
- skip only the pre-train sample aux check for MoE+FSDP2; the real wrapped
  training forward still exercises router outputs;
- keep run 08 at `max_seq_length: 16384`;
- use `accelerate_fsdp2_moe.yaml` with fine-grained wrapping around the heavy
  MoE child modules and `fsdp_reshard_after_forward: true`.

2026-05-12: Re-ran 35B-A3B one-step smoke after those fixes:
- command log: `smoke_train_08_16k_finewrap.log`
- status: `train_outputs/08_qwen3_6_35b_a3b_273/STATUS=smoke_done`
- `global_step=1`
- `train_loss=0.130950927734375`
- `eval_loss=0.12861832976341248`
- `train_runtime=684.7032s`
- eval runtime: 80.8802s
- output size: about 259G
- observed train/eval VRAM stayed below H200 capacity, roughly 95-108 GiB/GPU
  during the successful run.
- `checkpoint-best -> checkpoint-1` was published correctly.
- safetensors sanity check: 693 keys, 0 keys containing
  `_checkpoint_wrapped_module`.
```

## Phase 5 - Full Offline Training

Goal: train all 10 configured runs on `wanyichen-gpu02`.

Run inside tmux:

```bash
cd /path/to/evol-llm-tau2-stage2-ship
export BUNDLE_ROOT="$PWD"
export VENV="$BUNDLE_ROOT/code/.venv/bin"
export HF_HOME="$BUNDLE_ROOT/offline_artifacts/hf_home"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export EVOL_LOCAL_MODELS_DIR="$BUNDLE_ROOT/offline_artifacts/models"
export EVOL_REQUIRE_LOCAL_MODELS=1

tmux new -s tau2-train
bash code/training/orchestration/train_pipeline.sh 2>&1 | tee train.log
```

Resume or watch:

```bash
tmux attach -t tau2-train
tail -f train.log
```

Expected final output per run:

- `train_outputs/<run_id>/STATUS` containing `done`
- `train_outputs/<run_id>/checkpoint-final/` or `checkpoint-best/`
- `train_outputs/<run_id>/train_stdout.log`
- `train_outputs/<run_id>/training_log.json`
- `train_outputs/<run_id>/_prep.json`

Runs:

| Run | Model | Strategy | Eval |
|---|---|---|---|
| `01_qwen3_5_2b_273` | Qwen/Qwen3.5-2B | ddp | yes |
| `02_qwen3_5_4b_50` | Qwen/Qwen3.5-4B | ddp | yes |
| `03_qwen3_5_4b_100` | Qwen/Qwen3.5-4B | ddp | yes |
| `04_qwen3_5_4b_200` | Qwen/Qwen3.5-4B | ddp | yes |
| `05_qwen3_5_4b_273` | Qwen/Qwen3.5-4B | ddp | yes |
| `06_qwen3_5_9b_50` | Qwen/Qwen3.5-9B | fsdp2 | yes |
| `07_qwen3_5_9b_273` | Qwen/Qwen3.5-9B | fsdp2 | yes |
| `08_qwen3_6_35b_a3b_273` | Qwen/Qwen3.6-35B-A3B | fsdp2 | yes |
| `09_qwen3_5_4b_273_lr1e5` | Qwen/Qwen3.5-4B | ddp | no, LR ablation |
| `10_qwen3_5_4b_273_lr3e5` | Qwen/Qwen3.5-4B | ddp | no, LR ablation |

Status:

- [x] Full training started.
- [ ] Full training completed.
- [ ] All 10 train runs have `STATUS=done`.
- [ ] Checkpoints copied or staged for eval.

Notes:

```text
2026-05-12: Phase 4 smoke gate passed for:
- 2B DDP one-step smoke
- 4B DDP one-step smoke (`02_qwen3_5_4b_50`)
- 9B FSDP2 one-step smoke
- 35B-A3B MoE FSDP2 one-step smoke

Before launching the full sweep, moved the `STATUS=smoke_done` directories to:
`train_outputs_smoke_20260512_192514/`

`train_all.sh` intentionally skips only `STATUS=done`, so plain full training
will rerun smoke directories; starting from a clean run directory avoids mixing
old 1-step artifacts with full-run checkpoints.

2026-05-12 19:25 CST: Full train started on `10.100.0.62:35822`:
- tmux session: `tau2-train`
- main log: `train.log`
- first run in progress: `01_qwen3_5_2b_273`

After the DDP low-memory patch and the 35B 16k/FSDP2 changes, the original
12-15 hour estimate is no longer trusted. Treat the full sweep as a long tmux
job, probably multi-day train-only, and watch `train.log` plus per-run
`train_outputs/<run_id>/train_stdout.log`.
```

## Phase 6 - Eval And Aggregation

Goal: evaluate trained checkpoints and produce final result tables and plots.

This phase needs network access to OpenAI unless tau2 eval is reconfigured to
use a local replacement for the user simulator and NL judge.

If `wanyichen-gpu02` remains offline, copy the bundle plus `train_outputs/` to
a networked GPU machine that has the same Python dependencies and vLLM.

Run on the eval machine:

```bash
cd /path/to/evol-llm-tau2-stage2-ship
export BUNDLE_ROOT="$PWD"
export VENV="$BUNDLE_ROOT/code/.venv/bin"
export OPENAI_API_KEY="sk-..."

bash code/training/orchestration/eval_pipeline.sh 2>&1 | tee eval.log
```

Expected output:

- `train_outputs/<run_id>/eval_results.json`
- `train_outputs/<run_id>/eval_rollouts.jsonl`
- `train_outputs/<run_id>/eval_stdout.log`
- `train_outputs/<winner>/eval_results_seed300.json`
- `train_outputs/<winner>/eval_results_seed301.json`
- `train_outputs/SUMMARY.csv`
- `train_outputs/plots/capacity_curve.png`
- `train_outputs/plots/cost_pareto.png`

Status:

- [ ] Eval machine ready.
- [ ] `OPENAI_API_KEY` available on eval machine.
- [ ] Eval completed.
- [ ] Summary generated.
- [ ] Plots generated.

Notes:

```text
TODO
```

## Phase 7 - Final Deliverables

Expected deliverable directory:

```text
train_outputs/
  SUMMARY.csv
  plots/
  01_qwen3_5_2b_273/
  02_qwen3_5_4b_50/
  ...
  10_qwen3_5_4b_273_lr3e5/
```

Minimum final report:

- Best run id:
- Best pass rate:
- Total task cost USD:
- Eval seed(s):
- Any failed or skipped run:
- Exact commit or bundle hash:
- Tarball SHA256:
  `e22bb5a549aee5fddebfa9d084fb2e1d051cbb91f2f6c96091d41055fcb1a5cc`

Status:

- [ ] Final artifacts archived.
- [ ] Final report written.
- [ ] TODO updated with actual timings and paths.

Notes:

```text
TODO
```

## Quick Commands

Copy prepared 3FS bundle to local disk on `wanyichen-gpu02`:

```bash
TARGET_ROOT=/local/path/evol-llm-tau2-stage2-ship \
  bash /mnt/3fs2/data/root/evol/tau2_stage2_offline_20260512/evol-llm-tau2-stage2-ship/code/training/orchestration/copy_shared_bundle_to_gpu.sh
```

Prepare offline artifacts on a networked machine:

```bash
BUNDLE_ROOT="$PWD" DOWNLOAD_MODELS=1 INCLUDE_EVAL_DEPS=0 \
  bash code/training/orchestration/prepare_offline_artifacts.sh
```

Install no-conda training venv on the offline GPU machine:

```bash
BUNDLE_ROOT="$PWD" ARTIFACTS_DIR="$PWD/offline_artifacts" \
  bash code/training/orchestration/setup_env_offline_venv.sh
```

Train-only, no OpenAI needed:

```bash
export BUNDLE_ROOT="$PWD"
export VENV="$BUNDLE_ROOT/code/.venv/bin"
export HF_HOME="$BUNDLE_ROOT/offline_artifacts/hf_home"
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export EVOL_LOCAL_MODELS_DIR="$BUNDLE_ROOT/offline_artifacts/models"
export EVOL_REQUIRE_LOCAL_MODELS=1
bash code/training/orchestration/train_pipeline.sh
```

Single run smoke:

```bash
ONLY_RUN=01_qwen3_5_2b_273 MAX_STEPS_OVERRIDE=1 \
  bash code/training/orchestration/train_all.sh
```

Eval-only, OpenAI needed:

```bash
export OPENAI_API_KEY="sk-..."
bash code/training/orchestration/eval_pipeline.sh
```

Check statuses:

```bash
find train_outputs -maxdepth 2 -name STATUS -print -exec cat {} \;
```

Find failures:

```bash
rg -n "FAILED|Traceback|CUDA out of memory|NaN|nan|error" train.log train_outputs -g '*.log' -g '*.json'
```
