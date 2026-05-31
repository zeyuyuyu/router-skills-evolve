# τ²-bench Stage-2 Training Framework — Design

**Date**: 2026-05-08
**Owner**: Tony
**Bundle root**: `evol-llm-tau2-stage2-2026-05-08/`
**Source data SHA**: `git_sha: 3af58dcd757b5b5d8ab95796c4f75942b0455fea`
**Stage-2 corpus partition SHA**: `5d1983ba1461b0c5e819872a8eb7aaf6438e74a77b282b2fbc878ee5e09fe07a`

> **Status note (2026-05-09 post-validation):** This design doc captures the
> original architecture. Several aspects diverged during the build + Phase
> A-L deep validation passes. The README's "Bugs caught and fixed" section
> documents every divergence, but the headline ones are:
>
> 1. **Eval pipeline does NOT go through `pipeline.runner --provider local`**
>    (any "local provider" / "pipeline.runner" mention below is stale).
>    The new harness drives `tau2.cli` directly via subprocess; see
>    `code/training/eval/harness.py` for the canonical flow.
> 2. **`accelerate_fsdp2.yaml` dropped the FSDP1-only knobs** (forward_prefetch,
>    backward_prefetch, sharding_strategy, use_orig_params, sync_module_states,
>    top-level dispatch_batches). FSDP2 either rejects them outright
>    (forward_prefetch raises ValueError) or silently overrides them.
> 3. **`heldout_split.HELDOUT_LIMITS_BY_DOMAIN` uses 100/10 across all three**
>    domains to match what `eval_tasks.jsonl` actually declared.
>
> The code is authoritative — read the actual files when in doubt.

## 1. Goal

Train a small student LLM on the stage-2 SFT corpus such that the student, when used as the **lowest-tier replacement model** in τ²-bench's force-routed pipeline, satisfies two objectives:

1. **Hard constraint**: per-task inference cost < `glm-4.5-air` ($0.13/M input, ~$0.0212 per task at median).
2. **Soft maximization**: total-task-cost minimization via high step-replacement rate. Total cost formula:

   ```
   total_task_cost(student, k) = $0.0995 × (1 − k) + per_task_cost(student) × k
   ```

   where `k` = fraction of steps the student successfully replaces and `$0.0995` is the baseline `qwen3.5-397b-a17b` cost per full task.

The experiment also produces a **data-scaling curve** (sweep B) at the cost-sweet-spot anchor and a **capacity curve** at full data, in one shared infrastructure.

## 2. Background

### 2.1 Corpus shape (from `audit/audit_for_training.json`)

| Metric | Value |
|---|---|
| Train rows | 6,413 (273 unique runs) |
| Val rows | 394 (16 runs) |
| Eval task descriptors | 35 |
| Diag task descriptors | 5 |
| Domain split | telecom 60%, retail 21%, airline 20% |
| Phase split | locked 52%, baseline_alt 48% |
| Per-row token p99 / max | 13,025 / 19,925 (Qwen2.5 tokenizer) |
| Per-row loss-target tokens p50 | 76 (telecom), 44 (airline), 45 (retail) |
| Per-run sum total tokens p50 / p95 / max | 150K / 421K / 561K |
| **Loss-target tokens / epoch** | **697,618** |
| Naive total tokens / epoch | 53.2M |
| Sequence-packed (by run) tokens / epoch | 2.69M |
| **Redundancy factor** | **19.75×** |

### 2.2 Critical constraints

- **No CoT in data** — collection used standard chat-completion API without extended thinking. Qwen3.5/3.6 thinking mode must be disabled (`enable_thinking=False`).
- **Multi-epoch training mandatory** — 700K loss-target tokens is 6 orders below chinchilla-optimal for any candidate student, so training must run multiple epochs with early-stopping on val loss.
- **Sequence packing is load-bearing** — without restructuring rows into per-run sequences, training processes ~20× redundant tokens per epoch.

### 2.3 Compute envelope

- **Cloud target**: 8× NVIDIA H200 (~141 GB each, ~1.13 TB total), CUDA 13.0.
- **Local dev box**: ~4 GB VRAM. Used only for code-correctness smoke tests; no actual training.

## 3. Experiment design — Plan C'

A cross-shape grid: data-sweep at the cheap-anchor capacity, capacity-sweep at full data, with one cross-validation point.

| Run | Model | Data (n_runs) | Compute (5 epochs) | Purpose |
|---:|---|---:|---:|---|
| 1 | `Qwen/Qwen3.5-2B` | 273 (full) | ~5 GPU-hr | Capacity floor — cheapest possible deployment |
| 2 | `Qwen/Qwen3.5-4B` | 50 | ~2 GPU-hr | Data sweep low |
| 3 | `Qwen/Qwen3.5-4B` | 100 | ~4 GPU-hr | Data sweep mid-low |
| 4 | `Qwen/Qwen3.5-4B` | 200 | ~6 GPU-hr | Data sweep mid-high |
| 5 | `Qwen/Qwen3.5-4B` | 273 (full) | ~8 GPU-hr | Data sweep full + 4B capacity reference |
| 6 | `Qwen/Qwen3.5-9B` | 50 | ~6 GPU-hr | Data-shape sanity check at higher capacity |
| 7 | `Qwen/Qwen3.5-9B` | 273 (full) | ~15 GPU-hr | Capacity high + data sweep validation |
| 8 | `Qwen/Qwen3.6-35B-A3B` | 273 (full) | ~35 GPU-hr (+30 buffer if FSDP2 slow) | Capacity ceiling — tests "more steps replaced" hypothesis. **Hybrid architecture (Gated DeltaNet + Gated Attention) — see §11 architecture-confound risk** |
| 9–10 | `Qwen/Qwen3.5-4B` | 273 + LR ∈ {1e-5, 3e-5} | 2 × 8 GPU-hr | LR sanity check vs default 2e-5 |

**Total: 10 runs, ~96 GPU-hr training.** With per-run eval (~2 GPU-hr each = ~20 GPU-hr) and OOM/debug buffer (~15 GPU-hr), grand total **~130–140 GPU-hr**. Wall clock on 8× H200 with parallelization: **~16–20 hours.**

### 3.1 Subsample strategy

All data subsamples use **run-stratified sampling by domain** (preserves the natural 60/20/20 telecom/retail/airline mix). Random seed = 42 for reproducibility. Rounding: target counts use `max(1, round(target_n × n_domain / n_total))` per domain; if the sum exceeds `target_n`, drop the longest-tail domain's last sampled run; if under, add the next sampled run from the largest domain. Once the run set is chosen for band B, every row from those runs is included; partial-run subsampling is not allowed.

### 3.2 Mode

**Mode L+H** (locked + baseline_alt phases) is used for all runs. This uses the entire 6,413-row corpus. Mode L (locked-only) is not in scope for this experiment.

### 3.3 Decisions deferred to follow-up

- 2B data scaling sweep: only run if 2B-273 is competitive on pass-rate (≥80% of 4B-273).
- Mode L vs L+H ablation: not in this experiment.
- 27B dense or 122B-A10B: violates cost ceiling and/or memory; skipped.

## 4. Architecture

```
bundle-root/
├── data_processed/stage2_v1/        # source data (immutable, never written)
├── code/
│   ├── scripts/training_prep/       # already exists: data_audit.py
│   └── training/                    # NEW
│       ├── __init__.py
│       ├── configs/
│       │   ├── plan_c_prime.yaml    # the 10-run plan
│       │   └── runs/                # one YAML per run
│       │       ├── 01_2b_273.yaml
│       │       ├── 02_4b_50.yaml
│       │       └── ...
│       ├── data/
│       │   ├── convert_to_prompt_completion.py   # row → TRL prompt/completion format
│       │   ├── validate_chat_template.py         # pre-launch chat-template check
│       │   ├── filters.py                        # quality filters
│       │   ├── domain_rebalance.py               # T=0.5 temperature sampling
│       │   └── tool_shuffle.py                   # tool-schema-order augmentation
│       ├── train.py                 # entry point — trains ONE run
│       ├── eval/
│       │   ├── harness.py           # force-routed τ² eval
│       │   ├── vllm_serve.sh        # spin up vLLM per checkpoint
│       │   └── heldout_eval.py      # 15-task held-out generalization eval
│       ├── orchestration/
│       │   ├── train_all.sh         # Phase 1 — train all 10 runs
│       │   ├── eval_all.sh          # Phase 2 — eval all 8 main + winner re-eval
│       │   ├── run_pipeline.sh      # convenience: train_all && eval_all && summarize
│       │   ├── summarize.py         # Phase 3 — aggregate → SUMMARY.csv + plots
│       │   └── plotting.py          # per-run + aggregate plots
│       └── tests/                   # smoke + unit tests
└── train_outputs/                   # NEW — all artifacts
    ├── _data_cache/
    │   ├── train_prompt_completion.jsonl    # converted train rows
    │   ├── val_prompt_completion.jsonl      # converted val rows
    │   ├── heldout_tasks.jsonl              # 15 task-id-disjoint descriptors
    │   └── validation_report.json           # chat-template validation results
    ├── <run_id>/
    │   ├── checkpoint-final/
    │   ├── checkpoint-best/         # best val loss
    │   ├── training_log.json
    │   ├── eval_results.json
    │   ├── eval_results_heldout.json        # held-out task eval
    │   ├── plots/                   # per-run plots
    │   └── STATUS                   # running | done | failed
    ├── SUMMARY.csv
    └── plots/                       # aggregate plots
```

### 4.1 Two-phase orchestration

```bash
# Phase 1 — Train all 10 runs (~16-20 hr wall clock)
bash code/training/orchestration/train_all.sh \
    --bundle-root /path/to/bundle \
    --output-root /path/to/train_outputs \
    --plan plan_c_prime.yaml
# Output: train_outputs/<run_id>/{checkpoint-best/, training_log.json, plots/} for all 10

# Phase 2 — Eval all main runs in parallel on 8 GPUs (~3 hr wall clock)
bash code/training/orchestration/eval_all.sh \
    --output-root /path/to/train_outputs \
    --plan plan_c_prime.yaml
# Output: train_outputs/<run_id>/eval_results.json for runs 1–8 + winner re-eval + held-out eval

# Phase 3 — Aggregate (instant)
python code/training/orchestration/summarize.py --root /path/to/train_outputs
# Output: SUMMARY.csv, plots/{capacity_curve, data_scaling, cost_pareto, lr_check}.png
```

**Why two-phase**: Layer 1 val loss handles in-training monitoring. Layer 2 (per-epoch diag rollouts) is dropped per Option C cost cuts. Layer 3 full eval was always at-end-of-training only. Separating gives cleaner phase boundaries, easier failure isolation, and lets eval parallelize across all 8 GPUs simultaneously (huge speedup vs sequential).

### 4.1 Stack

| Layer | Choice | Rationale |
|---|---|---|
| Trainer | TRL `SFTTrainer` | Mature, supports FSDP/bf16 |
| Distributed | `accelerate launch` + FSDP | Standard for ≥9B on H200 |
| Mixed precision | bf16 throughout | H200 native; safer than fp16 for MoE |
| Inference (eval) | vLLM OpenAI-compatible mode | `vllm serve <ckpt>` integrates with existing `pipeline.runner` via a new `local` provider. **vLLM ≥ 0.7 for Qwen3.5; vLLM ≥ 0.19.0 for Qwen3.6.** Tool parsing: `--tool-call-parser qwen3_coder` (renamed from `hermes`/`qwen3`). |
| Tokenizer | Qwen3.5 native, `enable_thinking=False` | Data has no CoT |
| Eval framework | Existing `pipeline.runner` + new `local` provider | Reuses τ²-bench machinery — same metric definitions as collection |

Not chosen: Axolotl, Llama-Factory. They assume "train on every assistant turn"; our schema requires training only on the `_target_index` assistant message.

## 5. Data preparation pipeline

Transform stage-2 row-format → TRL prompt/completion format with quality filters, validation, domain rebalancing, and tool-schema augmentation. Replaces the earlier "custom collator" approach with TRL primitives + a dataset-level converter.

### 5.1 `convert_to_prompt_completion.py`

Reads each row from `data_processed/stage2_v1/train.jsonl` (6,413 rows) and emits TRL-compatible records:

```json
{
  "prompt": [
    {"role": "system", "content": "<system from domain_assets>"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "..."},
    ... messages[0..._target_index-1] ...
  ],
  "completion": [
    messages[_target_index]    // the single target — assistant text or tool_call
  ],
  "tools": <tools schema from domain_assets/<domain>_tools.json>,
  "_meta": <provenance from row._p>
}
```

Then in `SFTConfig`: `completion_only_loss=True`, `assistant_only_loss=False`. TRL handles masking out of the box — only the completion (target) message contributes to loss; everything in the prompt (including intermediate assistant turns) is masked.

### 5.2 Quality filters (run during conversion)

Drop rows that fail any of these:

| Rule | Why |
|---|---|
| `target_tokens < 5` | Garbage targets contribute noise |
| Unparseable `tool_calls[].function.arguments` JSON | Chat template will fail |
| `tool` role without preceding `assistant.tool_calls` | Orphan tool roles break template |
| Empty `content` AND empty `tool_calls` on target | Already 0 in current data, but defensive |

Expected drop rate: <1% of rows based on data audit (0 empty targets, 0 orphan tools observed).

### 5.3 Chat-template validation

`validate_chat_template.py` runs before training launches **for each model family** (since Qwen3.5 and Qwen3.6 have different chat templates):
- Apply the model's chat template with `enable_thinking=False` to every prompt+completion pair.
- Verify the resulting text is parseable round-trip via `tokenizer.encode → decode`.
- Verify no double-insertion of `<think></think>` (data must not already contain that string).
- Verify `tool_call_id` consistency (every `tool` role's `tool_call_id` matches a preceding `tool_calls[].id`).
- Output: `train_outputs/_data_cache/validation_report_<model_family>.json` with pass/fail per row. Hard-stop training if any row fails.

### 5.3.1 Tokenizer re-audit on actual training tokenizers

The existing `audit_for_training.json` was computed against `Qwen/Qwen2.5-1.5B-Instruct` tokenizer. Before launch, **re-run `data_audit.py`** against each actual training tokenizer:

- `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B` (likely identical token counts; same tokenizer family)
- `Qwen/Qwen3.6-35B-A3B` (chosen — has padded vocab 248,320, may produce different token counts vs Qwen3.5; re-audit required)

Verify per-row max stays ≤32K; if Qwen3.6 produces longer sequences, escalate seq_len or rebuild data_audit.

### 5.4 Domain rebalancing (T=0.5 temperature sampling)

Source distribution: telecom 60%, retail 21%, airline 20%. Without rebalancing, the student over-fits telecom-style behavior.

Apply temperature sampling at the dataset-iterator level:
```
weight(domain) ∝ count(domain)^T  with T=0.5
→ effective sampling: airline ~28%, retail ~30%, telecom ~42%
```

Implementation: `WeightedRandomSampler` in the DataLoader. **Reference**: arxiv 2410.04579 — upsampling outperforms per-token loss weighting for SFT stability.

### 5.5 Tool-schema-order augmentation

The order of tools in the schema is irrelevant to correctness but the model can pick it up as a spurious signal. At training time, randomly permute the tool list per-batch. Effectively 2–3× data augmentation, free.

Default ON for all runs. Disable via `tool_shuffle: false` in run YAML if isolating effects.

### 5.6 Run-grouping for prefix dedup (handled by TRL packing)

We do NOT pre-restructure rows into one-sequence-per-run. Instead, rely on TRL's BFD packing + `padding_free=True` + FlashAttention 2 to:
- Pack multiple short prompt/completion pairs into 32K windows
- Maintain block-diagonal attention so packed examples don't attend to each other
- Reset `position_ids` per example

Expected packing efficiency: total packed-window tokens within 10% of 2.69M per epoch (down from 53.2M naive). Logged in `training_log.json`.

### 5.7 Held-out task-id eval split

**New**: hold back 5 task_ids from each domain (15 tasks total) entirely from training and standard val. These tasks never appear in train.jsonl or val.jsonl after conversion. Used for the **generalization eval** (separate from the existing 35-task `eval_tasks.jsonl` which shares task_ids with training).

This addresses the gap that the existing val set shares task_ids with train (only seeds differ). Generalization-to-unseen-tasks is the more rigorous test.

Output: `train_outputs/_data_cache/heldout_tasks.jsonl`.

### 5.8 Single-batch overfit sanity check (pre-launch acceptance gate)

Before launching the full 10-run plan, run a sanity check on each model architecture (2B, 4B, 9B, 35B-A3B):

```
pdb=1, ga=1, lr=1e-4, 200 steps on ONE row
→ training loss should decrease to <0.05
→ verify completion_only_loss masks correctly: dump 5 random batches' label masks,
  confirm ONLY the completion's tokens have label != -100
```

If this fails for any architecture, do not launch the run. Fix masking/data first.

## 6. Training procedure

### 6.1 Hyperparameters (verified against Qwen3 official recipes + TRL+H200 best practices, 2026-05-08)

| Hyperparameter | Value | Notes |
|---|---|---|
| Optimizer | AdamW (β₁=0.9, β₂=0.95, weight_decay=0.01) | β₂=0.95 confirmed standard in Qwen3 Megatron-SWIFT recipes |
| LR (default) | **3e-5 (2B), 2e-5 (4B), 1e-5 (9B), 1e-5 (35B-A3B)** | 2B previously 5e-5 — too hot per Qwen2.5 SFT examples; 9B previously 2e-5 — lowered for stability |
| LR variants for 4B-273 sanity | **{1e-5, 2e-5, 3e-5}** | Was {1e-5, 2e-5, 4e-5}; dropped 4e-5 high end |
| LR schedule | Cosine decay to 10% min | Universal default |
| Warmup | **5% of total steps** (was 3%) | Safer at small total step count |
| Epochs | 5 (extendable to 10 if val loss not plateaued) | Aggressive early-stopping limits overfitting |
| Sequence length | 32,768 | Fits all rows (max 19,925) with headroom |
| Effective batch size | 64 sequences | Via gradient accumulation |
| **Per-GPU batch × grad-accum (8 GPUs → eff batch 64)** | **see table below** | Verified against published H200 VRAM benchmarks |
| Activation checkpointing | On for 9B/35B-A3B, off for 2B/4B | **`gradient_checkpointing_kwargs={"use_reentrant": False}`** mandatory under FSDP2 |
| FSDP | **FSDP2** (full-shard for 9B/35B, DDP-replicated for 2B/4B) | FSDP2 over FSDP1: 7% lower peak memory, ~1.5% faster, native partial-freeze, better checkpointing |
| `torch.compile` | **False for 9B/35B**, optional for 2B/4B | Known FSDP2 dtype-mismatch bugs at 32K seq |
| Mixed precision | bf16 (mixed, NOT pure bf16-true) | fp32 master weights via FSDP MixedPrecision |
| Gradient clipping | 1.0 | Universal default |
| Liger Kernel | **disabled** (`use_liger_kernel=False`) | TRL #3781: silently drops `assistant_masks` |
| Loss type | **`loss_type="chunked_nll"`** | ~30–50% less peak VRAM at long context |
| FSDP reshard after forward | **False** (with chunked_nll) | Required combination |
| Tokenizer EOS | **`eos_token='<|im_end|>'`** explicit | Avoid Qwen3 BOS confusion |
| Tokenizer PAD | **`pad_token='<|endoftext|>'`** explicit (NOT same as EOS) | Reusing `<|im_end|>` as pad breaks generation; Qwen3 has dedicated `<|endoftext|>` for padding |
| DataLoader workers | **`dataloader_num_workers=4`** | Default 0 starves H200s |
| Accelerate dispatch | **`dispatch_batches=False`** | Required for FSDP2 — each rank loads its own shard |
| Pin memory | `dataloader_pin_memory=True` (default) | Verify not overridden |
| Checkpoint save | Save **`tokenizer_config.json` + `chat_template.jinja`** alongside model weights | vLLM serving needs both |
| Mode | **L+H** (locked + baseline_alt, all 6,413 rows) | Doubles loss-target signal |
| Thinking | **`enable_thinking=False`** in `tokenizer.apply_chat_template` | Data has no CoT |
| Sequence packing | **`packing=True, packing_strategy="bfd"`** + **`padding_free=True`** | NOT "wrapped"; FA2 mandatory for cross-doc attention isolation |
| Attention impl | **`attn_implementation="flash_attention_2"`** | Required for `padding_free` and packing correctness |
| Loss masking | **TRL prompt/completion format** + `completion_only_loss=True`, `assistant_only_loss=False` | Replaces earlier "custom collator" plan; safer and standard |
| Seeds | data=42, model_init=1234, torch=1234 | Reproducibility |

### 6.1.1 Per-config VRAM safety table (verified)

Per-GPU peak VRAM at the recommended settings on 141 GB H200, with FSDP2 + FA2 + bf16 mixed + non-reentrant AC + chunked_nll:

| Run | Distrib. | per_device_train_batch_size | grad_accum_steps | AC | Peak VRAM/GPU | Headroom |
|---|---|---:|---:|---|---:|---:|
| 2B (run 1) | DDP | 4 | 2 | off | ~22–28 GB | 110+ GB ✅ |
| 4B (runs 2–5, 9, 10) | DDP/HSDP | 4 | 2 | off | ~38–50 GB | 90+ GB ✅ |
| 9B (runs 6, 7) | FSDP2 full-shard | **2** | **4** | on | ~55–75 GB | 65+ GB ✅ |
| **Qwen3.6**-35B-A3B (run 8) | FSDP2 full-shard | **1** | **8** | on (mandatory) | **~95–120 GB** (hybrid arch may differ — verify in smoke test) | **20–40 GB ⚠️** |

### 6.1.2 35B-A3B VRAM contingency

The 35B-A3B run is the only configuration with tight VRAM headroom. **Pre-launch smoke test**: run 100 training steps on the 50-run band before the full job. Monitor peak VRAM via `torch.cuda.max_memory_allocated()`. Escalate via this fallback ladder if needed:

1. **Drop seq_len 32K → 16K** (saves ~30% activation memory; loses airline tail-context headroom).
2. **Switch to MS-Swift** (EP+FSDP for Qwen3-MoE; 2–3× faster than vanilla FSDP2; battle-tested for this exact model class).
3. **LoRA fallback** (rank=64, alpha=128) — confounds the scaling-curve story but at least produces a result.

### 6.1.3 MoE-specific (Qwen3.6-35B-A3B only)

- `model.config.output_router_logits = True` — required for load-balancing loss to flow
- `model.config.router_aux_loss_coef = 0.001` — **Qwen3MoE HF default** (NOT 0.01; 0.01 would 10× over-weight aux loss and dominate SFT loss)
- **First-step assertion**: `assert outputs.aux_loss is not None and outputs.aux_loss > 0` — TRL has had silent-drop bugs (#2197, #4070); pre-flight check guarantees aux loss flows
- No expert dropout in published recipe
- QLoRA 4-bit not recommended on MoE (Unsloth) — full bf16 only

### 6.1.4 Qwen3.6-specific (Qwen3.6-35B-A3B only)

Qwen3.6 introduces a **hybrid architecture** distinct from Qwen3.5:

- **Layer pattern**: `10 × (3 × Gated-DeltaNet → MoE + 1 × Gated-Attention → MoE)` — 75% linear-attention layers, 25% standard self-attention
- **Attention dims**: Gated DeltaNet (32V/16QK heads, head_dim=128) + Gated Attention (16Q/2KV, head_dim=256)
- **Vocab**: 248,320 padded (re-audit token counts on Qwen3.6 tokenizer per §5.3.1)
- **Chat template**: similar to 3.5 but adds a `preserve_thinking` kwarg (default False — no change needed for our zero-CoT data; explicitly set `preserve_thinking=False` in `apply_chat_template` to be safe)
- **vLLM**: requires version **≥ 0.19.0** for inference (vs ≥ 0.7 for Qwen3.5)
- **FSDP2 compatibility**: Qwen3.6 is newer; the hybrid Gated DeltaNet kernels are less battle-tested under FSDP2 sharding. **Pre-launch 100-step smoke test is mandatory** before committing to the full ~35 GPU-hr run.
- **Tool-calling**: same OpenAI-compatible schema as Qwen3.5; vLLM uses `--tool-call-parser qwen3_coder`
- **Known bug**: empty-args loop on multi-turn tool calls ([pi-mono#3325](https://github.com/badlogic/pi-mono/issues/3325)). Validate post-train; if it manifests in eval, fall back to `Qwen/Qwen3.5-35B-A3B`.

### 6.2 Memory budget (35B-A3B reference)

```
Params (bf16):           35B × 2 =  70 GB
Gradients (bf16):        35B × 2 =  70 GB
Optimizer (Adam fp32):   35B × 8 = 280 GB
Total stored:                      420 GB → 52.5 GB/GPU sharded ×8
Activations (32K, mb=2):           ~30 GB/GPU
TOTAL per GPU:                     ~83 GB / 141 GB available
```

35B-A3B fits comfortably. 9B and below trivially fit.

### 6.3 Per-run output

For each run `<run_id>`:
- `checkpoint-final/` — last-epoch weights (the trained student).
- `checkpoint-best/` — best val-loss checkpoint (used for eval).
- `training_log.json` — step-level losses, LR, throughput, gradient norms, memory.
- `eval_results.json` — full force-routed eval after final checkpoint (see §7.3).
- `plots/{loss_curves,lr_schedule,gradient_norm,throughput,eval_diag}.png`.
- `STATUS` — `running` → `done` | `failed`.

## 7. Evaluation strategy

Three layers, scaled to cost.

### 7.1 Layer 1 — Validation loss (every step block)

Standard HF Trainer val-loss computation on `val.jsonl` (394 rows, 16 runs). Teacher-forced next-token prediction loss. No vLLM, no rollout. Logged every 50 steps; full val pass every 1 epoch.

**Early-stopping rule**: stop training if epoch-end val loss is higher than the best seen so far for **2 consecutive epochs**. **Best-checkpoint selection**: the checkpoint with lowest val loss across all epochs becomes `checkpoint-best`; `checkpoint-final` is always the last-epoch weights regardless of val loss.

### 7.2 Layer 2 — Diag-task pass-rate (every epoch)

After each epoch:
1. Save checkpoint.
2. Spin up vLLM on it (port assigned from a pool to avoid collisions).
3. Run **5 diag tasks × 2 seeds = 10 force-routed rollouts** via `pipeline.runner --provider local`.
4. Record pass-rate, per-task `replacement_rate_k`, NL-judge reward.
5. Append to `training_log.json` and update `eval_diag.png`.

Cost: ~5–10 min per epoch. Early-warning system for catastrophic regressions (e.g., model collapses to outputting nothing).

### 7.3 Layer 3 — Full force-routed eval (end of all training)

**Two-phase orchestration**: All 10 training runs complete first; then all evals run in parallel on the freed-up GPUs. See §6.4 for the orchestration script split.

**Eval procedure** for each main checkpoint (1 seed = task seed 300; LR-variant runs 9, 10 skip Layer 3):
1. Spin up vLLM on `checkpoint-best` (port assigned from a pool to avoid collisions).
2. Run **35 eval tasks × 1 seed = 35 rollouts** via `pipeline.runner --provider local` with the trained student configured as the lowest tier in `tau2_bench.yaml`.
3. Compute headline metrics:
   - `pass_rate` (fraction of tasks succeeded end-to-end).
   - `replacement_rate_k` (mean fraction of steps the student handled).
   - `total_task_cost_usd` (using the project's standard formula).
   - NL-judge per-task scores.
4. Write `eval_results.json` and the per-run row in `SUMMARY.csv`.

After all 8 main runs are evaluated, identify the **winning config** (highest pass_rate among configs with `total_task_cost < $0.0212`) and run a **2nd-seed re-eval** (35 tasks × seed 301 = 35 rollouts) for headline robustness.

### 7.4 Held-out task generalization eval (additional, no extra OpenAI cost beyond what's already budgeted)

Each main checkpoint is also evaluated on the 15 held-out task descriptors from §5.7. These are tasks whose `task_id` never appeared in train. Same eval pipeline; recorded as a separate column in `SUMMARY.csv`. Tests true generalization vs. the existing 35-task eval set which shares task_ids with train.

Cost: 15 tasks × 1 seed × 8 ckpts = 120 rollouts ≈ $16 additional OpenAI. **Updated total OpenAI bill: ~$70.**

Eval compute: ~2 hr per checkpoint; runs in parallel across 8 GPUs at end of training. **Total Phase 2 wall-clock: ~3 hr.**

### 7.4 Aggregate analysis (`summarize.py`)

After all runs complete:

| Plot | x-axis | y-axis | Series |
|---|---|---|---|
| `aggregate/capacity_curve.png` | log(active params) | pass-rate AND total task cost | {Qwen3.5-2B, Qwen3.5-4B, Qwen3.5-9B, Qwen3.6-35B-A3B} at 273 runs (annotated with architecture-confound caveat for 35B point) |
| `aggregate/data_scaling.png` | log(n_runs) | pass-rate AND total task cost | 4B at {50, 100, 200, 273} + 9B at {50, 273} for shape validation |
| `aggregate/cost_pareto.png` | total task cost ($) | pass-rate (%) | All 10 configs + glm-4.5-air baseline + qwen3.5-397b baseline |
| `aggregate/lr_check.png` | LR (log scale) | val loss + pass-rate | 4B-273 at LR ∈ {1e-5, 2e-5, 4e-5} |

`SUMMARY.csv` columns: `run_id, model, n_train_runs, n_train_rows, lr, n_epochs, val_loss_best, pass_rate, replacement_rate_k, total_task_cost_usd, training_gpu_hours, eval_gpu_hours`.

## 8. Cost & compute budget

### 8.1 GPU compute

| Phase | GPU-hr |
|---|---:|
| Run 1 — 2B-273 | 5 |
| Runs 2–5 — 4B sweep | 20 |
| Runs 9–10 — 4B-273 LR variants | 16 |
| Run 6 — 9B-50 | 6 |
| Run 7 — 9B-273 | 16 |
| Run 8 — 35B-A3B-273 | 35 |
| Eval (Layer 3 only; Layer 2 amortized into training) | 20 |
| Buffer (debug, OOM retries, vLLM warm-up) | 15 |
| **Total GPU-hours** | **~133** |

At commercial rates (~$3.50/GPU-hr H200 on Lambda/Runpod): **~$465**. On owned/leased cluster: opportunity cost only.

### 8.2 External API (real out-of-pocket) — post Option C cost cuts

τ²-bench eval routes user simulator and NL judge through `openai/gpt-5.2`. Per-rollout estimate: ~$0.135 (sim + judge).

| Item | Volume | Cost |
|---|---|---:|
| Layer 3 main eval (35 tasks × 1 seed × 8 main runs) | 280 rollouts | ~$38 |
| Held-out task eval (15 tasks × 1 seed × 8 main runs) | 120 rollouts | ~$16 |
| Winner re-eval (35 tasks × 1 extra seed) | 35 rollouts | ~$5 |
| Buffer (25%) | — | ~$15 |
| **Total OpenAI** | | **~$70–80** |

**Note**: Layer 2 mid-training diag rollouts are dropped under Option C — val loss handles catastrophic-drop detection.

**Estimated real out-of-pocket cost: ~$70–80 (OpenAI bill) since compute is owned (8× H200 server, no rental cost).**

## 9. Reproducibility & artifacts

- **Frozen environment**: `code/training/requirements.txt` pinned via `pip freeze`.
- **Pinned model revisions**: each per-run YAML records the exact HF commit SHA for the model + tokenizer (queried via `huggingface_hub.list_repo_commits`).
- **Seeds**: all RNG paths seeded.
- **Data immutability check**: `restructure_runs.py` verifies the SHA256 of `train.jsonl` matches the value in `_build_meta.json` before running.
- **Determinism**: `torch.use_deterministic_algorithms(True)` where compatible; set `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- **Logging**: every run records bundle git commit, source data SHA, environment freeze, full hyperparameters, and host info in `training_log.json`.
- **Artifact transferability**: `train_outputs/` is self-contained and can be copied wholesale to another machine for re-eval or analysis.

## 10. Out of scope (explicit non-goals)

- LoRA / QLoRA — confounds scaling-curve interpretation.
- Hyperparameter search beyond the LR sanity check at 4B-273.
- DPO / RLHF / preference learning — pure SFT only.
- Sweep C (forcing schedule ablation) — separate experiment.
- Multi-GPU eval inference — single vLLM instance per checkpoint is sufficient.
- 27B / 122B / 397B students — outside cost or memory ceilings.
- Custom NL-judge — reuse the gpt-5.2 judge from collection for metric continuity.
- Domain transfer eval (out-of-domain tasks) — not in stage-2 corpus.

## 11. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| 4B's data-scaling shape doesn't transfer to 9B | Medium | Run 6 (9B-50) as cross-validation; if shapes diverge, plan a 9B data sweep follow-up. |
| 35B-A3B overfits on 700K loss-target tokens | High | Aggressive early-stopping on val loss; checkpoint-best selection; epoch ≤ 5. |
| **35B-A3B VRAM tightness on H200** (peak ~95–120 GB / 141 GB) | **Medium–High** | Pre-launch smoke test on the 50-run band (100 steps), monitor `torch.cuda.max_memory_allocated()`. If >130 GB peak, escalate fallback ladder: (1) drop seq 32K→16K, (2) switch to MS-Swift EP+FSDP, (3) LoRA fallback. |
| **`Qwen/Qwen3.6-35B-A3B` HF repo availability** | Low–Medium | Verify the repo exists before launch (`huggingface-cli repo show Qwen/Qwen3.6-35B-A3B`). Fallback: `Qwen/Qwen3.5-35B-A3B` (clean architecture comparison) or drop the 35B point. |
| **🔴 Architecture confound (3.5 students vs 3.6 35B)** | **Certain (accepted)** | Qwen3.6 uses hybrid Gated-DeltaNet + Gated-Attention; smaller students are Qwen3.5 standard Transformer. The capacity scaling curve {2B, 4B, 9B, 35B-A3B} mixes scale-effect with architecture/version-effect. **In any published result, this MUST be explicitly caveated** — the headline "35B beats 9B by Δ%" cannot be cleanly attributed to scale alone. Mitigation: report 4B / 9B scaling curve separately as the "clean" result; report 35B as a "best-attainable-with-current-OSS-MoE" point with an architecture note. |
| **Qwen3.6 hybrid-attention FSDP2 stability** | **Medium** | Hybrid Gated DeltaNet kernels are newer and less tested under FSDP2 sharding than vanilla self-attention. Pre-launch 100-step smoke test mandatory. If unstable, fall back to `Qwen/Qwen3.5-35B-A3B` (well-tested with vanilla FSDP2). |
| **Qwen3.6 tool-call empty-args bug** ([pi-mono#3325](https://github.com/badlogic/pi-mono/issues/3325)) | Low–Medium | Known issue in Qwen3.6 multi-turn tool calling. If it manifests post-train (eval rollouts loop on empty args), fall back to `Qwen/Qwen3.5-35B-A3B`. |
| **35B-A3B vanilla FSDP2 slower than budgeted** | **Medium** | Documented OOM/slowness with vanilla FSDP2 for Qwen3-MoE class. Compute budget includes +30 GPU-hr buffer; switch to MS-Swift if 100-step smoke test shows <50% of expected throughput. |
| **`completion_only_loss` masks incorrectly** | Low | Pre-launch single-batch overfit test (§5.8) verifies. If wrong, training won't converge. |
| **Data validation fails for some rows** | Low | Hard-stop the run if any row fails chat-template validation; don't silently drop. |
| `enable_thinking=False` chat-template double-insertion | Low | `validate_chat_template.py` checks for pre-existing `<think></think>` strings. |
| vLLM serving incompatibility with FSDP2-trained weights | Low | vLLM 0.7+ supports Qwen3.5 + FSDP2 checkpoints. Smoke-tested with run 1's checkpoint before launching the rest. |
| OpenAI gpt-5.2 rate limits during full eval | Medium | Eval is staggered serially per checkpoint, not parallel. Built-in pipeline.runner backoff handles 429s. |
| Out-of-pocket cost overrun on OpenAI bill | Low | Hard cap via OpenAI usage limit; abort if exceeded. |
| Local dev environment blocks code completion (no GPU for testing) | Low | Smoke tests run on CPU + 0.5B model in 4-bit; unit tests pure-Python (no GPU required). |

## 12. Acceptance criteria

The framework is complete when:

1. `bash code/training/orchestration/run_all.sh` runs end-to-end from a freshly-checked-out bundle on the cloud server.
2. Each of the 10 runs produces a `STATUS=done` flag, all required artifacts (`checkpoint-final/`, `eval_results.json`, plots), and SUMMARY.csv has 10 rows.
3. Aggregate plots `capacity_curve.png`, `data_scaling.png`, `cost_pareto.png`, `lr_check.png` are generated.
4. At least one trained student satisfies the hard cost constraint (`per_task_cost < $0.0212`) AND has `pass_rate > 0` (sanity).
5. `tests/training/` smoke + unit tests all pass on the dev box (CPU only).

The experiment **succeeds** if at least one trained config has total task cost lower than glm-4.5-air's baseline at non-trivial pass-rate (>50%). Even if no config wins, the curves provide publishable scaling-curve data — the experiment is informative either way.
