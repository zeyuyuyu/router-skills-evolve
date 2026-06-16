# CLAUDE.md

Guidance for Claude Code instances working in this repo. Read [README.md](README.md)
for the full system narrative; this file is the operational quick-reference and the
list of non-obvious design decisions you must respect.

## What this is

On-policy iterative distillation: a **Router** sends each task to a small or large
model, **Skills** distill a reusable solving procedure from past traces (fed to the
small model's prompt), and the **LLM** is fine-tuned (SFT + GRPO) on the collected
data. The three co-evolve over N cycles. Main entrypoint: `scaling/run_full_pipeline.sh`.

## Commands

```bash
# Smoke test (no GPU / no API key — mock adapter). Fastest sanity check.
bash scaling/run_full_pipeline.sh --bench humaneval --smoke --mock
bash scaling/run_full_pipeline.sh --smoke --mock            # tau2 (default bench)

# Unit tests — pytest is NOT in the venv by default; install it first.
venv/bin/pip install pytest && venv/bin/python -m pytest tests/ -q

# Real runs
bash scaling/run_full_pipeline.sh --bench humaneval --n-cycles 4   # SFT + GRPO (needs GPU)
SKIP_LLM=1 SKIP_GRPO=1 bash scaling/run_full_pipeline.sh --bench humaneval  # Skills+Router only, no GPU

# Compile-check a python edit before claiming it works
venv/bin/python -m py_compile <file.py>
```

- Use `venv/bin/python` (Python 3.12), not a bare `python`.
- `*.ipynb` is **gitignored** — notebook edits won't show in `git status`. To verify a
  notebook runs: `jupyter nbconvert --to notebook --execute --output /tmp/_chk.ipynb <nb>`.

## Pipeline phases (per cycle, default schedule SLR)

| Phase | Script | Output |
|---|---|---|
| 1 collect traces | `experiments/scaling/collect_traces.py` | `traces.jsonl` (run-both oracle) |
| 2 skills evolve | inline Python in the shell script → `src/skills.py` | `skillbook.json` |
| 3a SFT | `traces_to_sft.py` + `train_small_model.py` (HE) / `tau2_train_wrapper.sh` (tau2) | `llm_adapter/checkpoint-best` |
| 3b GRPO | `experiments/scaling/grpo_train_simple.py` (HumanEval) | `grpo_adapter/` |
| 4 router | `experiments/scaling/train_router_simple.py` | `router/router.joblib` |
| 5 ablation | `experiments/scaling/run_e2e_ablation_simple.py` | `e2e_ablation_summary.json` |
| 6 aggregate | `experiments/scaling/aggregate_cycles.py` | `final_ablation_table.md`, `curve.png` |

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
- **Trimmed tree:** the only entrypoint is `scaling/run_full_pipeline.sh`. The old standalone
  HumanEval scripts (`run_evolve.py`, `extract_training_data.py`, `train_small_model_grpo.py`,
  `train_learnable_router.py`, `run_e2e_ablation.py`, the DPO/GRPO variants) and the BERT
  `src/learned_router/` + `src/router.py` were removed — don't reintroduce them.

## Conventions

- Match surrounding code style; comments in this repo are often bilingual (中文 + English) — follow the file you're in.
- Commit/push only when asked. If on `main`, branch first.
- Verify edits compile (`py_compile`) and, for pipeline logic, run the `--smoke --mock`
  path before claiming success.
