# Local-vs-Cloud Responsibility Split

**Constraint**: Local dev box has **~4 GB VRAM**. Cloud target has **8× H200 (~1.13 TB)**. Most of the framework MUST be developed locally without ever loading a real Qwen3.5-2B+ model, but the actual training runs ONLY on the cloud.

This document maps every task in `2026-05-08-tau2-stage2-training-framework-plan.md` to:

- **Local-OK**: can fully implement + test on the 4 GB box (CPU or tiny-model GPU)
- **Local-partial**: can implement + smoke-test locally, but full validation requires cloud
- **Cloud-only**: cannot meaningfully test locally; first run on cloud must be a staged smoke

This is the source of truth for what local sign-off DOES and DOES NOT prove.

---

## Per-task split

| Task | What it builds | Local-OK / partial / cloud-only | Local verification recipe |
|---|---|---|---|
| 1. Scaffolding | Package layout, requirements.txt, fixtures | **Local-OK** | `pytest tests/ --collect-only` |
| 2. `filters.py` | Quality filters | **Local-OK** | `pytest tests/test_filters.py` (pure Python) |
| 3. `tool_shuffle.py` | Tool-schema augmentation | **Local-OK** | `pytest tests/test_tool_shuffle.py` |
| 4. `domain_rebalance.py` | T=0.5 sampling | **Local-OK** | `pytest tests/test_domain_rebalance.py` |
| 5. `heldout_split.py` | Held-out task split | **Local-OK** | `pytest tests/test_heldout_split.py` |
| 6. `convert_to_prompt_completion.py` | Row → TRL format | **Local-OK** | Run on real `train.jsonl` (no model needed); verify drop count <1% |
| 7. `validate_chat_template.py` | Chat-template check | **Local-partial** | Tokenizer-only download (Qwen3.5-2B tokenizer ~14 MB). Real Qwen3.6 tokenizer download (~50 MB) — fits in disk, no GPU. Run for all 4 tokenizers locally. |
| 8. Plan + accelerate YAMLs | Configs | **Local-OK** | `python -c "import yaml; yaml.safe_load(...)"` round-trip |
| 9. Per-run YAMLs (×10) | Configs | **Local-OK** | YAML parse round-trip |
| 10. Tokenizer re-audit | Per-tokenizer length stats | **Local-partial** | Tokenizer-only — same as Task 7. CPU-bound, ~5 min per tokenizer × 4 tokenizers = ~20 min on dev box. |
| 11. `train.py` helpers | argparse, subsample, filter | **Local-OK** | `pytest tests/test_train_helpers.py` |
| 12. `train.py` dataset prep | Tokenizer + dataset loading | **Local-partial** | Smoke with `Qwen2.5-0.5B-Instruct` (1.2 GB weights, fits in 4 GB at fp16) on CPU; verify dataset prep + chat template apply. **Does NOT verify FA2, FSDP2, bf16.** |
| 13. `train.py` SFTTrainer body | Full trainer + MoE assertion | **Local-partial** | Smoke with 0.5B model + `attn_implementation="eager"` + max_seq_length=1024 + 1 step. **Does NOT verify FA2, FSDP2, bf16, MoE aux loss, packing.** |
| 14. `vllm_serve.sh` | vLLM serve script | **Cloud-only** | vLLM doesn't run usefully on CPU. Local: shell-syntax check only (`bash -n vllm_serve.sh`). |
| 15. `local_provider.py` | LiteLLM provider for vLLM | **Local-OK** | `pytest tests/test_local_provider.py` (no actual vLLM needed for unit tests) |
| 16. `harness.py` | Eval orchestration | **Cloud-only** | Local: import-check only (`PYTHONPATH=code python -c "from training.eval import harness"`). Real test requires vLLM endpoint. |
| 17. `train_all.sh` | Phase 1 entry | **Cloud-only** | Local: shell-syntax check (`bash -n train_all.sh`); preflight will fail without H200 (expected). |
| 18. `eval_all.sh` | Phase 2 entry | **Cloud-only** | Same as 17. |
| 19. `summarize.py` | SUMMARY.csv | **Local-OK** | `pytest tests/test_summarize.py` |
| 20. `plotting.py` | Aggregate plots | **Local-OK** | Synthetic SUMMARY.csv → render plots → verify PNGs exist |
| 21. Single-batch overfit smoke | Mask correctness gate | **Local-OK** | `SKIP_SMOKE_TRAIN=0 pytest tests/test_smoke_train.py` (~2 min CPU) |
| 22. `run_pipeline.sh` | Top-level entry | **Cloud-only** | Local: shell-syntax check |
| 23. End-to-end CPU smoke | Full chain check | **Local-OK** | The smoke override config with 0.5B model |

---

## Local sign-off acceptance criteria

Before committing to **any** cloud-side run, the local dev box MUST pass these gates:

```bash
# Run all gates from BUNDLE ROOT (pyproject registers `training` as the
# import root; PYTHONPATH=code keeps `python -m training.X` resolvable
# without needing `pip install -e code/` first).
cd /path/to/bundle

# 1. All unit tests
PYTHONPATH=code python -m pytest code/training/tests/ -v --ignore=code/training/tests/test_smoke_train.py

# 2. Chat-template validation against real data + real tokenizers (Qwen3.5-2B as proxy; 3.6 if downloaded)
# (Run from bundle root, not from code/training; pyproject registers `training` as the import root.)
PYTHONPATH=code python -m training.data.validate_chat_template \
    --src train_outputs/_data_cache/train_prompt_completion.jsonl \
    --tokenizer Qwen/Qwen3.5-2B \
    --out /tmp/local_validation.json
# Expect: n_pass == n_total, exit 0

# 3. Single-batch overfit (verifies completion_only_loss masking)
SKIP_SMOKE_TRAIN=0 PYTHONPATH=code python -m pytest code/training/tests/test_smoke_train.py::test_overfit_one_example -v -s
# Expect: final loss < 0.05

# 4. End-to-end smoke (Task 23)
# Train Qwen2.5-0.5B for 1 step on CPU; verify STATUS=smoke_done (Phase-N).
# (Run from bundle root; --smoke-test makes train.py write smoke_done, which
# train_all.sh now accepts as success.)
CUDA_VISIBLE_DEVICES="" PYTHONPATH=code python -m training.train \
    --run-config code/training/configs/runs_smoke/smoke_2b.yaml \
    --plan-config code/training/configs/plan_c_prime.yaml \
    --bundle-root . \
    --smoke-test

# 5. All shell scripts syntax-clean
for s in code/training/orchestration/*.sh code/training/eval/vllm_serve.sh; do bash -n "$s" || echo "FAIL: $s"; done

# 6. All run YAMLs parse + schema-correct
PYTHONPATH=code python -c "
import yaml, glob
for p in sorted(glob.glob('code/training/configs/runs/*.yaml')):
    d = yaml.safe_load(open(p))
    assert d['run_id']
    assert d['model']['name'].startswith('Qwen/')
    assert d['training']['per_device_train_batch_size'] >= 1
    print(p, 'OK')
"

# 7. Plotting works on synthetic SUMMARY.csv
PYTHONPATH=code python -c "
import csv, tempfile, pathlib
from training.orchestration import plotting
with tempfile.TemporaryDirectory() as d:
    root = pathlib.Path(d)
    rid_dir = root / '01_test'; rid_dir.mkdir()
    (root / 'SUMMARY.csv').write_text(
        'run_id,model_name,n_runs_kept,pass_rate,total_task_cost_usd,eval_loss\\n'
        '01_test,Qwen/Qwen3.5-2B,273,0.42,0.005,1.5\\n'
    )
    rows = plotting.load_summary(root / 'SUMMARY.csv')
    plotting.plot_capacity_curve(rows, root / 'plots' / 'capacity_curve.png')
    plotting.plot_cost_pareto(rows, root / 'plots' / 'cost_pareto.png')
    assert (root / 'plots' / 'capacity_curve.png').exists()
    print('plots OK')
"
```

If all 7 pass on the dev box, the framework is **code-correct**. It is NOT yet **infrastructure-correct** — that's what the staged cloud rollout is for.

---

## Cloud-side staged rollout (FIRST cloud run)

Once the framework is in the cloud server's filesystem, **do NOT immediately invoke `train_pipeline.sh` or `run_pipeline.sh`**. Run a staged rollout that catches infrastructure issues cheaply:

### Stage 1 — Environment verification (~5 min, $0)

```bash
# Verify the cluster matches the spec.
nvidia-smi  # 8× H200, 141 GB each
nvcc --version  # CUDA 13.0
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import flash_attn; print(flash_attn.__version__)"  # expect 2.8.3 (pinned)
python -c "import vllm; print(vllm.__version__)"              # expect 0.20.2 (pinned, iter-16)
# Verify all 4 fine-tune-target model SHAs are still served by HF Hub
# (metadata-only check; no weight download). `huggingface-cli repo show`
# is NOT a real subcommand — use HfApi.model_info instead.
python - <<'PYEOF'
from huggingface_hub import HfApi
api = HfApi()
for repo, sha in [
    ("Qwen/Qwen3.5-2B",       "15852e8c16360a2fea060d615a32b45270f8a8fc"),
    ("Qwen/Qwen3.5-4B",       "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    ("Qwen/Qwen3.5-9B",       "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    ("Qwen/Qwen3.6-35B-A3B",  "995ad96eacd98c81ed38be0c5b274b04031597b0"),
]:
    info = api.model_info(repo, revision=sha)
    print(f"OK {repo} @ {info.sha[:10]}")
PYEOF
echo "OPENAI_API_KEY: ${OPENAI_API_KEY:0:8}..."  # confirm set, mask value
```

### Stage 2 — Single-GPU FA2 smoke on Qwen3.5-2B (~10 min, $0)

```bash
# Run #1 (2B-273) for ONLY 5 steps. Verifies: FA2 builds, FSDP2-DDP path runs, bf16 stable, packing works.
cd /path/to/bundle
ONLY_RUN=01_qwen3_5_2b_273 MAX_STEPS_OVERRIDE=5 \
    bash code/training/orchestration/train_all.sh
# Expect: STATUS=done, training_log.json with non-NaN losses.
```

**Note (iter-15, 2026-05-11):** `MAX_STEPS_OVERRIDE=N` now passes through to `train.py --max-steps N`, capping steps on FULL data + FULL batch/accum so Stages 2/3/4 actually run 5/20/20 steps under realistic distributed conditions. STATUS is written as `smoke_done` so `eval_all.sh` skips the partial checkpoint. Earlier iterations only flipped `--smoke-test` (which truncated to 1 step + 10/5 rows + bs=1) regardless of the override value.

### Stage 3 — FSDP2 smoke on Qwen3.5-9B (~30 min, $0)

```bash
ONLY_RUN=06_qwen3_5_9b_50 MAX_STEPS_OVERRIDE=20 \
    bash code/training/orchestration/train_all.sh
# Expect: 8 GPUs all engaged, FSDP2 sharding works, peak VRAM <100 GB/GPU.
```

### Stage 4 — MoE smoke on Qwen3.6-35B-A3B (~45 min, $0)

```bash
ONLY_RUN=08_qwen3_6_35b_a3b_273 MAX_STEPS_OVERRIDE=20 \
    bash code/training/orchestration/train_all.sh
# Expect: aux_loss > 0 assertion passes, peak VRAM <130 GB/GPU,
# throughput ≥50% of expected (else fall back to MS-Swift per Spec §6.1.2).
```

If Stage 4 fails the throughput check or OOMs, the operator should:
1. Drop `max_seq_length` to 16384 in run #8's YAML and re-run Stage 4.
2. If still failing, switch run #8 to the `Qwen/Qwen3.5-35B-A3B` fallback per Spec §11 risks.

### Stage 5 — Single full eval (~3 hr, ~$10 OpenAI)

```bash
# After at least one trained checkpoint exists:
ONLY_RUN=01_qwen3_5_2b_273 \
    bash code/training/orchestration/eval_all.sh
# Expect: vLLM starts, 35 rollouts complete, eval_results.json has non-zero pass_rate.
```

### Stage 6 — Full pipeline (~17-21 hr, ~$70 OpenAI)

Two equivalent options — pick the one that matches your API-key timing.

**Option A — split (recommended when the OpenAI key arrives late).** Training (~12-15 hr) runs without a key; eval (~3-6 hr) runs once the key is exported.

```bash
# Phase 1: training only — no OPENAI_API_KEY needed
bash code/training/orchestration/train_pipeline.sh

# … later, once the OpenAI key is provisioned …
export OPENAI_API_KEY=sk-...
bash code/training/orchestration/eval_pipeline.sh
# Expect: SUMMARY.csv with 10 rows, all plots in train_outputs/plots/.
```

**Option B — end-to-end (only if the OpenAI key is already in env).**

```bash
export OPENAI_API_KEY=sk-...
bash code/training/orchestration/run_pipeline.sh
# Expect: SUMMARY.csv with 10 rows, all plots in train_outputs/plots/.
```

---

## What the local dev box CANNOT verify (and what to watch for on cloud)

| Risk | Local-test gap | First cloud sign of trouble |
|---|---|---|
| FlashAttention 2 numerical bug | Local uses `eager` attention | NaN losses or sudden spikes in step 1-50 |
| FSDP2 sharding policy mismatch | Local uses DDP single-GPU | OOM at step 0; or "tensor not contiguous" errors |
| bf16 numerical instability | Local uses fp32 | Loss diverges after epoch 1; gradient norm explodes |
| MoE aux_loss not flowing | Local uses dense 0.5B | Aux loss assertion in Task 13 fires; OR loss plateau (load balancing collapse) |
| Packing + padding-free + tool calls | Local uses `packing=False` | Cross-document attention bleed (eval pass-rate craters) |
| `dispatch_batches=False` not honored | Local uses 1 GPU | Each rank loads same data; train loss correct but val loss inflated |
| chunked_nll + reshard_after_forward=False | Local doesn't shard | OOM at step 0 with "all-gather peak memory" error |
| vLLM checkpoint compatibility | Local doesn't serve | `vllm serve` fails to load FSDP2-saved checkpoint format |
| OpenAI rate limits | Local doesn't call OpenAI | Eval rollouts time out; tau2-CLI litellm backoff retries |
| Tokenizer re-audit ≠ Qwen3.6 | Local downloads tokenizer files (~50 MB), works | Should be identical to local; flag if not |

The **staged rollout** above is specifically designed to catch each of these one stage at a time, so a failure costs minutes-to-hours, not the full ~20 hr pipeline.

---

## Pinning checklist before cloud handoff

1. `requirements.txt` pinned to specific versions (no `git+main`); ✓ done — `trl==1.4.0` released 2026-05-09.
2. Tokenizer download cached: HF cache copied to a known location on cloud (or trust HF Hub at runtime).
3. Model revision SHAs pinned in each run YAML's `model.revision` field; ✓ done — 5 SHAs verified live against HF Hub API on 2026-05-09 and committed.
4. `BUNDLE_ROOT` consistent across all scripts (no hardcoded `/home/tonyy/...`); ✓ verified — only one informational reference in the design spec doc.
5. `OPENAI_API_KEY` rotated before sharing the bundle (separate from local dev key) — operator action.
6. Bundle git committed with the design spec + this plan + this handoff doc; ✓ done.

## Optional perf extras for Qwen3.6-35B-A3B (run 08)

Qwen3.6-35B-A3B uses Gated Delta Network (GDN) attention. transformers 5.8.0
ships an in-tree Triton GDN path that's correct on the core stack, so run 08
trains and evals as-is. For best throughput on H200, the upstream `flash-linear-attention`
authors recommend installing their hand-tuned chunked GDN kernels:

```bash
conda activate tau2-stage2
pip install "flash-linear-attention>=0.3.2" "causal-conv1d>=1.4.0"
```

This is purely a perf upgrade; correctness is unchanged. `mamba-ssm` is **not**
needed (Qwen3.5/3.6 use GDN, not classic Mamba SSM). vLLM 0.20.1 already ships
its own in-tree GDN kernel for inference; no extra packages needed for eval.
7. SHA256 checksum of the bundle tarball recorded for integrity check on the cloud side
