# τ²-bench Stage-2 SFT — server-side training bundle

**Bundle date**: 2026-05-09
**Source bundle**: `evol-llm-tau2-stage2-2026-05-08` (private)
**Target hardware**: 8× H200, CUDA 13.0, conda-managed Python 3.12
**Goal**: from a single bash command, run 10 SFT runs on the τ²-bench Stage-2 corpus, evaluate each on held-out tasks, and emit a scaling-curve summary + plots.

This bundle is the cloud-handoff artifact. Everything inside is what the operator needs and nothing more — no raw collection data, no build manifests, no in-progress notebooks. It was extracted from a larger local development bundle after all 23 framework-build tasks were complete and locally validated against a CPU smoke run.

## Layout

```
.
├── README.md                         this file
├── .gitignore                        ignores caches + run artifacts (commits only inputs)
├── code/
│   ├── pyproject.toml                package manifest (registers training/, pipeline/, etc.)
│   ├── adapters/tau2_bench/          glue between pipeline.runner and τ²-bench (data-collection only)
│   ├── core/                         pricing tables, schemas, credit-rotation, slice classifier
│   ├── pipeline/                     pipeline.runner — used by data-collection paths only; eval bypasses it
│   ├── scripts/training_prep/        data_audit utility (used by tokenizer_reaudit.sh)
│   ├── vendor/                       (NOT TRACKED) tau2-bench cloned here by setup_env_server.sh
│   └── training/                     ← the framework
│       ├── train.py                  TRL SFTTrainer entrypoint (FSDP2 + bf16 + FA2 + BFD packing)
│       ├── requirements.txt          CUDA-13.0-pinned package list
│       ├── data/                     tool_shuffle, domain_rebalance, heldout_split (+ descriptor expand), format converter, chat-template validator
│       ├── eval/                     vLLM serve wrapper + tau2-CLI-driving harness (force-routed eval)
│       ├── orchestration/            train_all.sh, eval_all.sh, run_pipeline.sh, setup_env_{server,local}.sh, summarize.py, plotting.py, tokenizer_reaudit.sh
│       ├── configs/                  plan_c_prime.yaml, accelerate_{ddp,fsdp2}.yaml, runs/ (10 per-run YAMLs), runs_smoke/smoke_2b.yaml
│       └── tests/                    13 pytest modules (75 unit tests; all pass locally)
├── data_processed/stage2_v1/         6,413 train + 394 val SFT rows (the corpus)
├── train_outputs/_data_cache/        prepped prompt-completion cache (computed once, fingerprinted)
└── docs/
    ├── 2026-05-08-tau2-stage2-training-framework-design.md   design spec (what's built and why)
    ├── 2026-05-08-tau2-stage2-training-framework-plan.md     implementation plan (23-task breakdown)
    └── 2026-05-08-LOCAL-VS-CLOUD-handoff.md                  6-stage cloud rollout playbook
```

## Prerequisites (server-side)

The bundle assumes the operator is on a fresh Ubuntu 24.04 + CUDA 13.0 + 8×H200 image with the following already in place:

- **miniconda installed and initialized in bash**. `setup_env_server.sh` calls `eval "$(conda shell.bash hook)"` which requires the conda binary on `$PATH`. On a fresh image, run these THEN start a new shell before continuing to Quick start:
  ```bash
  # Install miniconda (one-time)
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p $HOME/miniconda3
  source $HOME/miniconda3/etc/profile.d/conda.sh
  conda init bash
  ```
  After running the block above, run `exec bash` (or open a new SSH session) so that conda's PATH/init lands in the parent shell. Then verify `command -v conda` resolves before continuing to Quick start below.
- **CUDA 13.0 toolkit** (`nvcc --version` reports 13.0). The flash-attn source build needs nvcc; cu130 wheels do the rest.
- **`OPENAI_API_KEY`** exported in env (eval phase uses OpenAI for the user simulator + NL-judge).
- **(Optional) `HF_TOKEN`** exported — Qwen3.5/3.6 are public so unauthenticated downloads work, but `HF_TOKEN` raises the rate limit and speeds the 35B-A3B download.

## Quick start (server-side)

The pipeline is split into two halves so the operator can **start training without an OpenAI API key**, then run eval later once the key is provisioned. (Training never calls OpenAI; eval needs the key for tau2's user-simulator + NL-judge.)

```bash
# 1. Set up the conda env (one-shot; ~30-60 min for the flash-attn source build).
BUNDLE_ROOT=$(pwd) bash code/training/orchestration/setup_env_server.sh

# 2. Activate.
conda activate tau2-stage2

# 3. Smoke-test the whole pipeline with a tiny model on 1 step.
BUNDLE_ROOT=$(pwd) MAX_STEPS_OVERRIDE=1 bash code/training/orchestration/train_all.sh

# 4. If smoke is green, run TRAINING (no OpenAI key needed; ~12-15 hr).
BUNDLE_ROOT=$(pwd) bash code/training/orchestration/train_pipeline.sh

# 5. Once you have your OpenAI key, run EVAL + aggregation (~3-6 hr).
export OPENAI_API_KEY=sk-...
BUNDLE_ROOT=$(pwd) bash code/training/orchestration/eval_pipeline.sh

# 6. Read summary + plots.
cat train_outputs/SUMMARY.csv
ls train_outputs/plots/
```

Or run end-to-end in one shot (only if the OpenAI key is already available):

```bash
export OPENAI_API_KEY=sk-...
BUNDLE_ROOT=$(pwd) bash code/training/orchestration/run_pipeline.sh
```

For step-by-step rollout (recommended on the first cloud run), see `docs/2026-05-08-LOCAL-VS-CLOUD-handoff.md` — it walks through 6 staged validations from "is the env right?" through to "all 10 runs passed eval".

## Local validation status

Before this bundle was cut, the framework was built and re-validated end-to-end on a 4 GB-VRAM Linux laptop (CUDA 13.0 / cu130 / Python 3.12 conda env mirroring the server stack):
- `pytest tests/` → **75 passed, 1 skipped** (gated overfit smoke runs in 391s on demand).
- Gated overfit smoke (200 steps on 1 example, masked correctness) → PASSED — final loss 0.0466 (threshold <0.05).
- E2E smoke via `train.py` + `runs_smoke/smoke_2b.yaml` → STATUS=done, checkpoint-final/, training_log.json, _prep.json all written. SHA-pinned model revision downloaded from HF Hub.
- Setup script flows: `setup_env_local.sh` recreates the laptop env idempotently; `setup_env_server.sh` was dry-run on the laptop through Stage 5 (tau2-bench clone+checkout+pip-install) — `import tau2`, `from tau2.runner import build_text_orchestrator`, `from tau2.evaluator.evaluator import EvaluationType`, `python -m pipeline.runner --help` all green.
- `bash -n` clean on all 7 shell scripts.
- Plotting on synthetic CSV → 4 PNGs rendered.

**Bugs caught and fixed during this validation pass** (in addition to the two already fixed before the tidy):
- `pytest-asyncio` was missing from `requirements.txt` while `pyproject.toml` set `asyncio_mode = "auto"` + `filterwarnings = ["error"]` — pytest INTERNALERRORs at startup. Added pin.
- `tau2-bench` requires `[voice,knowledge]` extras + `portaudio` system lib (`pyaudio` dlopen target) for `import tau2` to succeed. Setup script now installs both.
- Orchestration scripts hardcoded `VENV=$BUNDLE_ROOT/code/.venv/bin` — broken on conda-only servers. Now defaults to `$CONDA_PREFIX/bin`.
- The eval pipeline was structurally broken: `harness.py` invoked `pipeline.runner` with args that don't exist (`--mode eval`, `--provider local`, `--local-port`, `--tasks-jsonl`, …). Replaced with a thin tau2-CLI driver that bypasses `pipeline.runner` (which never had a `local` provider).
- `eval_all.sh` consumed `heldout_tasks.jsonl` but `train_all.sh` only emitted `heldout_task_ids.json`. Added an `expand_heldout_ids_to_descriptors` step.
- `requirements.txt` pinned `trl @ git+main` (non-deterministic) — switched to `trl==1.4.0` (released 2026-05-09; ships chunked_nll). Model `revision: main` → 40-char HF Hub commit SHAs for all 5 models.
- `flash-attn` build `MAX_JOBS=8` was unsafe (12-15 GB RAM/job; OOMs unless free RAM > 128 GB). Default lowered to 4.
- `warmup_ratio` is on the deprecation path in transformers 5.x. Migrated all 11 YAMLs to `warmup_steps: 0.05` (float<1 = ratio); train.py forwards either form.

**Phase-J post-compact deep audit caught five more (commit `746513f`):**
- `accelerate_fsdp2.yaml` had `fsdp_forward_prefetch: true` — accelerate 1.13.0 explicitly raises `ValueError("forward_prefetch is not yet implemented in FSDP2")` at config load (utils/dataclasses.py:1931). Removed forward_prefetch and the four FSDP1-only keys (`sharding_strategy`, `use_orig_params`, `sync_module_states`, `backward_prefetch`) that warn-and-override under FSDP2; top-level `dispatch_batches` lives on DataLoaderConfiguration not LaunchConfig.
- `train_all.sh`'s `accelerate launch ... | tee` pipeline under `set -e -o pipefail` aborted the entire 10-run sweep on the first failed run, before the failure-branch could mark it failed and continue. Added `|| true` so per-run STATUS-file checks own success/failure decisions.
- `eval_all.sh`'s seed-301 winner re-eval was a silent no-op: harness short-circuits on existing `eval_results.json` and the subsequent rename mislabeled the seed-300 file as seed-301. Now renames seed-300 outputs aside before invoking with seed 301, then renames the new outputs to `*_seed301.*`.
- `heldout_split.HELDOUT_LIMITS_BY_DOMAIN` had telecom 400/30 but every telecom row in `eval_tasks.jsonl` declared 100/10. Held-out telecom evals would have used a 4× over-provisioned step budget. Aligned to 100/10.
- `train.py:283` used `torch_dtype=` for `from_pretrained`; transformers 5.x renamed it to `dtype=` (kept-for-BC but warns at modeling_utils.py:1518-1521). Fixed.

**Phase-K (round 2) caught five more (commit pending):**
- `convert_to_prompt_completion.py`'s argparse did not declare `--drop-log`, but `train_all.sh` invokes it with that flag — cold-run would have crashed before the first conversion. Argparse now accepts it (lossless converter writes an empty file).
- `setup_env_server.sh`/`setup_env_local.sh` installed torch via plain PyPI (`pip install torch==2.11.0 torchvision`); cu130 wheels live only on `download.pytorch.org/whl/cu130`. Switched to `--index-url`, dropped unused torchvision.
- `setup_env_server.sh`'s flash-attn source build runs `--no-build-isolation` — needs `ninja` in the env at build time, which is not a transitive dep of pip/conda and not in `requirements.txt`. Added `pip install ninja packaging wheel psutil` before Stage 3, plus an explicit `export MAX_JOBS=$MAX_JOBS_BUILD`.
- `setup_env_server.sh` never `pip install -e $BUNDLE_ROOT/code` — only `cd code/`-prefixed `python -m` invocations worked. Added in new Stage 5b so operator can run from any cwd.
- Stage 6 sanity-import did not exercise tau2's runtime entry points (`build_text_orchestrator`, `EvaluationType`); now imported alongside the package list. Stage-5 stale comment about `pipeline.runner --provider local` rewritten to reflect the tau2-CLI direct flow.

## What's NOT exercised locally (cloud-only first runs)

Local validation runs on CPU with `attn_implementation=eager` and packing disabled, so the FA2 + FSDP2 + bf16 + packing + chunked_nll code paths are **not** exercised before this bundle. The full eval (vLLM + tau2 + OpenAI judge) is also untested locally — it was redesigned and unit-tested with mocks, but the live tau2-CLI subprocess path will only run on the server. The 6-stage rollout in `docs/2026-05-08-LOCAL-VS-CLOUD-handoff.md` is designed to surface failures cheaply (one tiny model, then one full-size model, then full sweep) before committing to all 10 runs.

## Data lineage

`data_processed/stage2_v1/` was built from a multi-epoch τ²-bench collection (multiple Qwen3.5-397B-A17B and Claude Opus 4.7 student-teacher mixtures across airline/retail/telecom). The corpus has 6,413 train + 394 val rows from 273 unique runs. `_build_meta.json` records the source-bundle SHA. Per-row provenance is in `audit/per_row_provenance.jsonl`.

## Known non-blockers

- `train_all.sh`'s `MAX_STEPS_OVERRIDE` only triggers `--smoke-test` (1 step). For arbitrary step counts, edit the run YAML's `max_steps`.
- The eval harness invokes the OpenAI API by default (user simulator + NL judge via `openai/gpt-5.2`). `eval_all.sh` requires `OPENAI_API_KEY`; override the user model with `--user-llm` if you want a different judge.
- For `08_qwen3_6_35b_a3b_273.yaml` (the Qwen3.6-MoE/GDN run), training works on the core stack but optimal GDN throughput benefits from `pip install flash-linear-attention causal-conv1d`. Not strictly required — see `docs/2026-05-08-LOCAL-VS-CLOUD-handoff.md` for details.

## License & attribution

τ²-bench is vendored at SHA `17e07b1d` from https://github.com/sierra-research/tau2-bench (Apache 2.0). The setup script clones at the pinned SHA — see `code/training/orchestration/setup_env_server.sh`.
