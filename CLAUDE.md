# CLAUDE.md

Guidance for Claude Code instances working in this repo. Read [README.md](README.md)
for the full system narrative; this file is the operational quick-reference and the
list of non-obvious design decisions you must respect.

## What this is

On-policy iterative distillation: a **Router** sends each task to a small or large
model, **Skills** distill a reusable solving procedure from past traces (fed to the
small model's prompt), and the **LLM** is fine-tuned (SFT + GRPO) on the collected
data. The three co-evolve over N cycles. Main entrypoint: `scripts/run_full_pipeline.sh`.

## Repo layout (post-2026-06 refactor)

- `src/` — importable library. `src/skills.py`, `src/models.py`, `src/config.py`,
  `src/train_plots.py`, and `src/pipeline/` (the pipeline stages: `collect_traces.py`,
  `traces_to_sft.py`, `train_small_model.py`, `grpo_train_simple.py`,
  `train_router_simple.py`, `run_e2e_ablation_simple.py`, `aggregate_cycles.py`,
  `benches/`, `tau2_train_wrapper.sh`). Imported as `from src.pipeline.X import …`.
- `scripts/` — shell orchestration (`run_full_pipeline.sh`, `vllm_serve_humaneval.sh`,
  `setup_vllm_venv.sh`, `benchmark_tau2.sh`).
- `config/` — experiment input recipes (`*.env`), loaded via `--config <name>` or
  `EXPERIMENT_CONFIG=<name>`. See `config/README.md`.
- `tau2_stage2/` — vendored tau2 SFT framework (`BUNDLE_ROOT`). `data/`, `results/`.
- The old `experiments/` tree is gone; its contents moved to `src/pipeline/` and `tau2_stage2/`.

## Commands

```bash
# Smoke test (no GPU / no API key — mock adapter). Fastest sanity check.
bash scripts/run_full_pipeline.sh --bench humaneval --smoke --mock
bash scripts/run_full_pipeline.sh --smoke --mock            # tau2 (default bench)

# Unit tests — pytest is NOT in the venv by default; install it first.
venv/bin/pip install pytest && venv/bin/python -m pytest tests/ -q

# Real runs
bash scripts/run_full_pipeline.sh --bench humaneval --n-cycles 4   # SFT + GRPO (needs GPU)
SKIP_LLM=1 SKIP_GRPO=1 bash scripts/run_full_pipeline.sh --bench humaneval  # Skills+Router only, no GPU

# Compile-check a python edit before claiming it works
venv/bin/python -m py_compile <file.py>
```

- Use `venv/bin/python` (Python 3.12), not a bare `python`.
- `*.ipynb` is **gitignored** — notebook edits won't show in `git status`. To verify a
  notebook runs: `jupyter nbconvert --to notebook --execute --output /tmp/_chk.ipynb <nb>`.

## Pipeline phases (per cycle, default schedule SLR)

| Phase | Script | Output |
|---|---|---|
| 1 collect traces | `src/pipeline/collect_traces.py` | `traces.jsonl` (run-both oracle) |
| 2 skills evolve | inline Python in the shell script → `src/skills.py` | `skillbook.json` |
| 3a SFT | `traces_to_sft.py` + `train_small_model.py` (HE) / `tau2_train_wrapper.sh` (tau2) | `llm_adapter/checkpoint-best` |
| 3b GRPO | `src/pipeline/grpo_train_simple.py` (HumanEval) | `grpo_adapter/` |
| 4 router | `src/pipeline/train_router_simple.py` | `router/router.joblib` |
| 5 ablation | `src/pipeline/run_e2e_ablation_simple.py` | `e2e_ablation_summary.json` |
| 6 aggregate | `src/pipeline/aggregate_cycles.py` | `final_ablation_table.md`, `curve.png` |

Bench branches: `humaneval` (local models + pytest reward, GRPO on) vs `tau2_bench`
(remote agent API, FSDP2 SFT, GRPO off by default). `swe_bench` adapter is a stub.

## Design decisions you must NOT silently break

1. **Single global skill.** `extract_signature()` in `src/skills.py` always returns
   `"coding"`. Do not reintroduce per-cluster signatures without explicit ask — the
   whole routing/ablation design now assumes one bucket.
2. **Router owns routing; Skills only distill procedure.** `collect_traces._policy_decision`
   routes via the learned router ONLY. The SkillBook `can_downgrade_to_small` verdict is
   computed but recorded as a diagnostic field (`policy_skill_verdict`) — it must NOT
   override `route`. (With one global skill the verdict is identical for every task and
   would steamroll the per-prompt router.)
3. **Router trains on the RAW prompt, no procedure prefix.** Procedure is a constant
   under one skill → zero discriminative value, and the label already reflects the
   procedure-augmented run. Train and inference both use the raw prompt — keep them aligned.
4. **Procedure format is shared across SFT / GRPO / inference:** `f"{procedure}\n\n---\n\n{problem}"`.
   Changing it in one place silently creates a train/inference mismatch.
5. **Ablation is four-arm:** `large / skills / router / full`. There is no standalone
   `base` arm — `skills` (always-small + procedure) is the small baseline. `VARIANT_ORDER`
   in `aggregate_cycles.py` and the markdown loop in `run_e2e_ablation_simple.py` must
   stay in sync.
6. **Checkpoint priority for the next cycle's small model:** `grpo_adapter/` >
   `llm_adapter/checkpoint-best` > base `SMALL_MODEL`.
7. **SkillBook stats keys are canonical roles `"small"`/`"large"`,** never raw model IDs
   (IDs change every cycle and break `can_downgrade_to_small`).
8. **GRPO rollout temperature must be > 0.** The tau2 litellm patch drops `seed` from every
   completion, so sampling temperature is the ONLY source of intra-group divergence — greedy
   (temp 0) collapses all K rollouts → zero advantage → no gradient. Use **0.7–1.0** for
   training rollouts (default `GRPO_TEMPERATURE=1.0`; lower toward 0.7 if multi-turn
   trajectories destabilize). Use **greedy (temp 0)** for held-out eval / deployed routing —
   never reuse the training temperature there.

## Gotchas

- **Concurrent modification.** A "shepherd" automation and collaborators commit to this
  repo (see `git log` for `shepherd:` / `refined pipeline` commits). Files you edited may
  get committed by another process, and new files (e.g. tau2 GRPO work) may appear
  mid-session. Re-read a file before assuming its state; check `git log`/`git diff` before
  large doc edits.
- **Cycle ≥ 1 needs the previous LLM served:** `vllm_serve.sh` brings up the checkpoint as
  an OpenAI-compatible server (port 8050) so `--small-model openai/evol-llm-student` works.
  `run_full_pipeline.sh` launches/kills it automatically; manual runs must manage it.
- **Qwen3 thinking mode off** (`enable_thinking=False`) — tau2 corpus has no CoT, enabling
  it goes OOD.
- **flash-attn build:** `MAX_JOBS=4` max, else OOM.
- **vLLM on driver 575.57.08 (CUDA 12.9): use the cu12 venv.** The old `.vllm_venv`
  (vllm 0.22) is a **cu13** build — flash-attn / flashinfer / `_C` kernels are compiled
  for CUDA 13 and fail with `CUDA driver version is insufficient` (only a couple of
  Qwen2.5 models limp along via TRITON_ATTN hacks; Qwen3-4B breaks). The fix is
  **`.vllm_cu12_venv`** (vllm 0.11.0 / torch 2.8+cu128 / transformers 4.57 / fastapi
  0.116.1 / starlette 0.41.3) — native FLASH_ATTN, no workarounds. Build it with
  `scripts/setup_vllm_cu12_venv.sh`. Both serve scripts (`scripts/vllm_serve_humaneval.sh`
  and `tau2_stage2/code/training/eval/vllm_serve.sh`) now default to `.vllm_cu12_venv`.
  Switches `HE_USE_VLLM` (Phase 1 serving) / `GRPO_USE_VLLM` (Phase 3b weight-sync) can
  now be set to 1. In-process HF generate (both 0) still works and needs no vLLM.
- **SFT on hard-tasks-only collapses the model.** With ~19 teacher pairs, grad_accum
  produces nan-grad steps and pass@1 drops to ~0. Always run with `SFT_INCLUDE_SUCCESS=1`
  (also behaviour-clones solved tasks → ~77 pairs, stable). See `config/humaneval_dapo_gpt.env`.
- **Trimmed tree:** the only entrypoint is `scripts/run_full_pipeline.sh`. The old standalone
  HumanEval scripts (`run_evolve.py`, `extract_training_data.py`, `train_small_model_grpo.py`,
  `train_learnable_router.py`, `run_e2e_ablation.py`, the DPO/GRPO variants) and the BERT
  `src/learned_router/` + `src/router.py` were removed — don't reintroduce them.

## Conventions

- Match surrounding code style; comments in this repo are often bilingual (中文 + English) — follow the file you're in.
- Commit/push only when asked. If on `main`, branch first.
- Verify edits compile (`py_compile`) and, for pipeline logic, run the `--smoke --mock`
  path before claiming success.
