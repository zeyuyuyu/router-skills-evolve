"""Training entry point — trains ONE run.

Usage:
    accelerate launch --config_file <accel.yaml> -m training.train \\
        --run-config code/training/configs/runs/<run_id>.yaml \\
        --plan-config code/training/configs/plan_c_prime.yaml \\
        --bundle-root /path/to/bundle

Sub-task 11 builds the skeleton + helpers. Sub-tasks 12 + 13 implement the
SFTTrainer body (data prep + model loading + training + checkpointing +
MoE aux-loss assertion). Heavy ML imports (torch, transformers, trl) are
lazy — they live INSIDE main(), so the helpers stay unit-testable on a
plain CPU dev box without those packages installed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path


def _sha256_of(path: Path) -> str | None:
    """Compute sha256 of a file; return None if absent (provenance helper)."""
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

import yaml


def load_run_config(path: Path) -> dict:
    """Load a per-run YAML config."""
    return yaml.safe_load(path.read_text())


def load_plan_config(path: Path) -> dict:
    """Load the plan-level YAML config."""
    return yaml.safe_load(path.read_text())


def build_runs_by_domain(rows: list[dict]) -> dict[str, list[str]]:
    """Group run_dirs by domain from converted prompt/completion rows.

    Each row's `_meta` is the per-row provenance; we pick out
    `(domain, run_dir)` pairs and de-duplicate.
    """
    seen: set[tuple[str, str]] = set()
    out: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        meta = r["_meta"]
        key = (meta["domain"], meta["run_dir"])
        if key in seen:
            continue
        seen.add(key)
        out[meta["domain"]].append(meta["run_dir"])
    return dict(out)


def subsample_runs_stratified(
    runs_by_domain: dict[str, list[str]], target_n: int, seed: int
) -> list[str]:
    """Sample target_n unique run_dirs total, stratified by domain.

    Per Spec §3.1: preserves the natural 60/20/20 telecom/retail/airline mix
    via per-domain quota = round(target_n * count_d / total). If quotas don't
    sum to target_n, adjust by trimming from the head or topping up from
    the largest unused domain pool. Seeded for reproducibility (sort then
    shuffle, since `set` iteration order is non-deterministic).
    """
    rng = random.Random(seed)
    total_pool = sum(len(rs) for rs in runs_by_domain.values())
    if target_n >= total_pool:
        all_runs: list[str] = []
        for d in sorted(runs_by_domain):
            all_runs.extend(sorted(runs_by_domain[d]))
        return all_runs

    quotas = {
        d: max(1, round(target_n * len(rs) / total_pool))
        for d, rs in runs_by_domain.items()
    }

    sampled: list[str] = []
    for d in sorted(runs_by_domain):
        pool = sorted(runs_by_domain[d])
        rng.shuffle(pool)
        sampled.extend(pool[: min(quotas[d], len(pool))])

    if len(sampled) > target_n:
        sampled = sampled[:target_n]
    elif len(sampled) < target_n:
        sampled_set = set(sampled)
        unused: list[str] = []
        for d in sorted(runs_by_domain, key=lambda x: -len(runs_by_domain[x])):
            for r in sorted(runs_by_domain[d]):
                if r not in sampled_set:
                    unused.append(r)
        rng.shuffle(unused)
        sampled.extend(unused[: target_n - len(sampled)])
    return sampled


def filter_rows_by_runs(rows: list[dict], run_dirs: set[str]) -> list[dict]:
    """Keep only rows whose `_meta.run_dir` is in the allowed set."""
    return [r for r in rows if r["_meta"]["run_dir"] in run_dirs]


def apply_heldout_filter(
    train_rows: list[dict], heldout_ids_path: Path
) -> tuple[list[dict], int, bool]:
    """Drop train rows whose (domain, task_id) matches any heldout pair.

    train_all.sh writes heldout_task_ids.json into _data_cache/. Without
    this filter, the model would train on the exact tasks held out for
    heldout-eval, inflating heldout pass_rate (iter-7 audit verified 8%
    of train rows match heldout pairs at seed=42).

    Returns (filtered_rows, n_dropped, heldout_file_present). When the
    file is absent, returns the input rows unchanged and present=False
    so the caller can emit a warning to stderr.
    """
    if not heldout_ids_path.exists():
        return train_rows, 0, False
    heldout_by_domain = json.loads(heldout_ids_path.read_text())
    heldout_pairs = {
        (domain, task_id)
        for domain, ids in heldout_by_domain.items()
        for task_id in ids
    }
    n_before = len(train_rows)
    filtered = [
        r for r in train_rows
        if (r["_meta"]["domain"], r["_meta"].get("task_id")) not in heldout_pairs
    ]
    return filtered, n_before - len(filtered), True


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file as a list of dicts."""
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _latest_numbered_checkpoint(output_dir: Path) -> Path | None:
    checkpoints: list[tuple[int, Path]] = []
    for p in output_dir.glob("checkpoint-*"):
        if not p.is_dir() or p.name in {"checkpoint-best", "checkpoint-final"}:
            continue
        suffix = p.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            checkpoints.append((int(suffix), p))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def publish_best_checkpoint(
    output_dir: Path,
    best_model_checkpoint: str | None,
    best_global_step: int | None = None,
) -> str | None:
    """Expose Trainer's best checkpoint at the stable path eval_all.sh expects."""
    src: Path | None = None
    if best_model_checkpoint:
        src = Path(best_model_checkpoint)
    elif best_global_step:
        candidate = output_dir / f"checkpoint-{best_global_step}"
        if candidate.exists():
            src = candidate
    if src is None:
        src = _latest_numbered_checkpoint(output_dir)
    if src is None:
        return None

    src = src.resolve()
    if not src.exists():
        return None

    dst = output_dir / "checkpoint-best"
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists():
        shutil.rmtree(dst)

    try:
        os.symlink(os.path.relpath(src, start=output_dir), dst, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)
    return str(dst)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-config", required=True, type=Path)
    ap.add_argument("--plan-config", required=True, type=Path)
    ap.add_argument("--bundle-root", required=True, type=Path)
    ap.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a minimal config (10 rows, 1 step) for code-correctness.",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Cap training to N steps with full data and full batch sizes. "
            "Writes STATUS=smoke_done so eval_all.sh skips the partial "
            "checkpoint. Use --smoke-test for the 10/5-row 1-step tiny "
            "smoke; --max-steps is for the staged rollout (Stage 2/3/4 in "
            "LOCAL-VS-CLOUD-handoff.md) where the operator wants real "
            "distributed config under real bs/accum, just N steps deep."
        ),
    )
    args = ap.parse_args(argv)

    # ---------- Phase A: config + paths ----------
    run_cfg = load_run_config(args.run_config)
    plan_cfg = load_plan_config(args.plan_config)
    bundle_root: Path = args.bundle_root.resolve()
    rid = run_cfg["run_id"]

    output_dir = (bundle_root / run_cfg["training"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_status = None
    status_file = output_dir / "STATUS"
    if status_file.exists():
        previous_status = status_file.read_text(errors="ignore").strip()
    (output_dir / "STATUS").write_text("running\n")

    # HF Hub robustness — set BEFORE the lazy imports below, since
    # transformers caches HF_HUB_* settings at first hub call. Defaults:
    #   HF_HUB_DOWNLOAD_TIMEOUT — per-request timeout in seconds. Default 10s
    #     is tight when downloading 35B-A3B's many shards on variable WAN
    #     latency; a transient slow chunk would raise and abort the run.
    #   HF_HUB_ENABLE_HF_TRANSFER — use the rust hf_transfer client (faster
    #     resumable downloads). transformers 5.8 picks it up if installed.
    # hf_hub_download has built-in resume + exponential backoff on 5xx/429
    # per huggingface_hub.file_download._http_get; raising the timeout
    # extends the window before it counts as failure.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # ---------- Phase B: lazy heavy imports ----------
    import math

    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        EarlyStoppingCallback,
        TrainerCallback,
    )
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    class NaNGuardCallback(TrainerCallback):
        """Abort training if loss becomes NaN or Inf.

        SFTTrainer's default behavior is to KEEP training through NaN loss
        — gradients propagate, parameters poison, the final checkpoint
        contains NaN weights, and downstream vLLM serves gibberish at eval.
        This callback raises at the first NaN/Inf in `loss`, before the
        next save_model() can write the corrupt weights.

        Particularly important for the 35B-A3B MoE run: bf16 + chunked_nll
        + activation checkpointing has more numerical knobs than smaller
        dense runs and is more likely to hit a transient NaN.
        """

        def on_log(self, args, state, control, **kwargs):  # noqa: D401
            if not state.log_history:
                return
            entry = state.log_history[-1]
            # Check both 'loss' (train logs) and 'eval_loss' (eval-strategy=steps
            # logs). Eval logs have eval_loss but NOT loss; checking only 'loss'
            # would silently skip eval-side NaN (FSDP runs 06/07/08 all use
            # eval_strategy=steps, so eval logs interleave with train logs).
            for key in ("loss", "eval_loss"):
                value = entry.get(key)
                if value is None:
                    continue
                if math.isnan(value) or math.isinf(value):
                    raise RuntimeError(
                        f"NaN/Inf {key} at step {state.global_step} ({key}={value}). "
                        f"Aborting before save_model() can persist corrupt weights."
                    )

    # ---------- Phase C: tokenizer ----------
    from training.model_resolution import (
        from_pretrained_kwargs,
        resolve_model_source,
    )

    model_cfg = run_cfg["model"]
    model_source = resolve_model_source(
        model_cfg["name"], model_cfg.get("revision", "main")
    )
    if model_source.local_path:
        print(
            f"MODEL-RESOLUTION: {model_source.original_name} -> "
            f"{model_source.local_path}"
        )
    tok = AutoTokenizer.from_pretrained(
        model_source.name,
        **from_pretrained_kwargs(model_source, trust_remote_code=True),
    )
    if tok.pad_token_id is None:
        # Spec §6.1 — use <|endoftext|> as pad. Fail loud if it resolves to
        # unk_token_id (means tokenizer has no real <|endoftext|> token).
        eot_id = tok.convert_tokens_to_ids("<|endoftext|>")
        if eot_id is None or eot_id == tok.unk_token_id:
            raise RuntimeError(
                f"Tokenizer for {model_cfg['name']} has no pad_token AND "
                f"<|endoftext|> resolves to unk_token_id={tok.unk_token_id} "
                f"(or None). Cannot set a sensible pad token. Override in "
                f"the run config or pick a different pad strategy."
            )
        tok.pad_token = "<|endoftext|>"
        tok.pad_token_id = eot_id

    # ---------- Phase D: load converted dataset ----------
    cache_dir = bundle_root / "train_outputs" / "_data_cache"
    train_jsonl = cache_dir / "train_prompt_completion.jsonl"
    val_jsonl = cache_dir / "val_prompt_completion.jsonl"
    if not train_jsonl.exists():
        raise FileNotFoundError(
            f"Converted train cache not found at {train_jsonl}. Run "
            f"`python -m training.data.convert_to_prompt_completion "
            f"--bundle-root {bundle_root}` first."
        )
    train_rows = _read_jsonl(train_jsonl)
    val_rows = _read_jsonl(val_jsonl) if val_jsonl.exists() else []

    # ---------- Phase D.5: heldout-leakage filter ----------
    # train_all.sh writes heldout_task_ids.json into _data_cache/ AFTER the
    # cache is built. The cache itself includes every (domain, task_id),
    # so without this filter the model would train on the exact tasks
    # held-out for eval_results_heldout.json — inflating its pass rate.
    # Iter-7 audit verified 8% of train rows match heldout pairs at seed 42.
    # Apply at train-time so each per-run subsample sees a clean train set.
    heldout_ids_path = cache_dir / "heldout_task_ids.json"
    n_before = len(train_rows)
    train_rows, n_dropped, heldout_present = apply_heldout_filter(
        train_rows, heldout_ids_path
    )
    if heldout_present:
        if n_dropped > 0:
            print(
                f"HELDOUT-FILTER: dropped {n_dropped}/{n_before} train rows "
                f"matching heldout (domain, task_id) pairs"
            )
    else:
        # train_all.sh writes the JSON at Phase 1 (lines 108-140). If it's
        # missing here, the operator invoked train.py directly (e.g., the
        # local smoke recipe in LOCAL-VS-CLOUD-handoff.md Stage 4). Without
        # the filter, ~8% of train rows match heldout (domain, task_id)
        # pairs at seed=42 — silently inflating heldout pass_rate. Warn
        # loudly so the operator notices before a full sweep.
        msg = (
            "WARNING: heldout_task_ids.json not found in cache_dir; "
            "SKIPPING heldout-leakage filter. ~8% of train rows likely "
            "overlap heldout tasks at seed=42. Run train_all.sh end-to-end "
            "to write the file, OR ignore if you intend a no-eval run."
        )
        print(msg, file=sys.stderr)

    # ---------- Phase E: run-stratified subsample ----------
    runs_by_domain = build_runs_by_domain(train_rows)
    n_train_runs = run_cfg["data"]["n_train_runs"]
    subsample_seed = plan_cfg["partition"]["subsample_seed"]
    keep_runs = set(
        subsample_runs_stratified(
            runs_by_domain, target_n=n_train_runs, seed=subsample_seed
        )
    )
    train_rows = filter_rows_by_runs(train_rows, keep_runs)
    n_runs_kept = len(keep_runs)

    # ---------- Phase F: smoke override ----------
    if args.smoke_test:
        train_rows = train_rows[:10]
        val_rows = val_rows[:5]
        # Force minimal hyperparams; the runs_smoke YAML should already
        # encode these, but we belt-and-suspender override here too.
        run_cfg["training"]["num_train_epochs"] = 1
        run_cfg["training"]["max_steps"] = 1
        run_cfg["training"]["per_device_train_batch_size"] = 1
        run_cfg["training"]["gradient_accumulation_steps"] = 1
    if args.max_steps is not None:
        # --max-steps N (set via MAX_STEPS_OVERRIDE in train_all.sh) caps
        # max_steps on the FULL data + FULL batch/accum, so the operator's
        # staged rollout (LOCAL-VS-CLOUD-handoff.md Stages 2/3/4) actually
        # runs N steps under realistic distributed conditions. Diverges
        # from --smoke-test (which truncates data + bsz to 1 for code-
        # correctness). Combinable: --smoke-test + --max-steps overrides
        # both. STATUS is written as smoke_done either way to keep
        # eval_all.sh from running on the partial checkpoint.
        run_cfg["training"]["max_steps"] = int(args.max_steps)

    # ---------- Phase G: tool-schema-order shuffle ----------
    from training.data.tool_shuffle import shuffle_tools_in_schema

    rng = random.Random(subsample_seed)
    if plan_cfg.get("data", {}).get("tool_shuffle", True):
        for r in train_rows:
            r["tools"] = shuffle_tools_in_schema(r["tools"], rng)

    # ---------- Phase H: domain rebalance weights (diagnostic only) ----------
    from training.data.domain_rebalance import compute_per_row_weights

    domain_temp = plan_cfg.get("data", {}).get("domain_temperature", 0.5)
    train_rows_for_weights = [
        {"_p": {"domain": r["_meta"]["domain"]}} for r in train_rows
    ]
    sample_weights = compute_per_row_weights(train_rows_for_weights, domain_temp)

    # ---------- Phase I: HF Datasets ----------
    # TRL 1.4.0 reads per-row chat_template_kwargs in sft_trainer.py:1454
    # to parameterize apply_chat_template. Injecting
    # {"enable_thinking": False} on every row pins the train-time render
    # to match validate_chat_template.py:72 + the <think>-stripped data
    # from convert_to_prompt_completion._sanitize_message. Without this,
    # the Qwen3.5/3.6 template's empty-think branch injects empty
    # <think>\n\n</think>\n\n around each assistant turn, diverging from
    # both validation and the data. Iter-9 verified TRL 1.4.0 has no
    # SFTConfig.chat_template_kwargs field; the only working hook is
    # per-row on the dataset.
    for r in train_rows:
        r["chat_template_kwargs"] = {"enable_thinking": False}
    for r in val_rows:
        r["chat_template_kwargs"] = {"enable_thinking": False}
    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows) if val_rows else None

    # ---------- Phase J: prep dump ----------
    domain_counts: dict[str, int] = defaultdict(int)
    for r in train_rows:
        domain_counts[r["_meta"]["domain"]] += 1
    if sample_weights:
        sw_min = min(sample_weights)
        sw_max = max(sample_weights)
        sw_mean = sum(sample_weights) / len(sample_weights)
    else:
        sw_min = sw_max = sw_mean = 0.0
    prep_info = {
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
        "n_runs_kept": n_runs_kept,
        "keep_runs": sorted(keep_runs),
        "domain_counts": dict(domain_counts),
        "sample_weight_dist": {
            "min": sw_min,
            "max": sw_max,
            "mean": sw_mean,
            "n": len(sample_weights),
        },
        "domain_temperature": domain_temp,
        "tool_shuffle": plan_cfg.get("data", {}).get("tool_shuffle", True),
        "smoke_test": bool(args.smoke_test),
        "model_name": model_cfg["name"],
        "resolved_model_name": model_source.name,
        "resolved_model_local_path": model_source.local_path,
        # Corpus fingerprint — closes the provenance chain
        # source_jsonl → converted_cache → checkpoint. Without this an
        # operator cannot verify post-hoc which converted corpus this
        # checkpoint trained on, especially if the cache was regenerated
        # between runs.
        "train_jsonl_sha256": _sha256_of(train_jsonl),
        "val_jsonl_sha256": _sha256_of(val_jsonl) if val_rows else None,
    }
    (output_dir / "_prep.json").write_text(json.dumps(prep_info, indent=2) + "\n")
    print(
        f"PREP: rid={rid} n_train={len(train_rows)} n_val={len(val_rows)} "
        f"n_runs_kept={n_runs_kept} domains={dict(domain_counts)}"
    )

    # ---------- Phase K: model ----------
    config = AutoConfig.from_pretrained(
        model_source.name,
        **from_pretrained_kwargs(model_source, trust_remote_code=True),
    )
    if model_cfg.get("is_moe", False):
        moe_cfg = run_cfg.get("moe") or {}
        config.output_router_logits = moe_cfg.get("output_router_logits", True)
        config.router_aux_loss_coef = moe_cfg.get("router_aux_loss_coef", 0.001)

    train_cfg = run_cfg["training"]
    # Map dtype: bf16 in YAML → torch.bfloat16; otherwise float32 (CPU smoke).
    if train_cfg.get("bf16", False):
        torch_dtype = torch.bfloat16
    elif train_cfg.get("fp16", False):
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # transformers 5.x renamed `torch_dtype` → `dtype` (legacy still works
    # but emits a warning; verified against modeling_utils.py:1518-1521).
    model = AutoModelForCausalLM.from_pretrained(
        model_source.name,
        config=config,
        dtype=torch_dtype,
        attn_implementation=train_cfg.get("attn_implementation", "eager"),
        **from_pretrained_kwargs(model_source, trust_remote_code=True),
    )
    # NOTE: gradient_checkpointing is enabled via SFTConfig below (Phase L)
    # AND via accelerate_fsdp2.yaml's fsdp_activation_checkpointing on the
    # 8-GPU path. We previously also called model.gradient_checkpointing_enable()
    # manually here — that was a third, redundant enable. Dropped: HF Trainer
    # honors SFTConfig.gradient_checkpointing + gradient_checkpointing_kwargs
    # (forwarded below) and FSDP plugin layers on top.

    # ---------- Phase L: SFTConfig ----------
    # NOTE: TRL@main renamed `max_seq_length` → `max_length` (deprecated in
    # 0.16, removed in 0.20). Map from our run-config key here.
    sft_kwargs: dict = {
        "output_dir": str(output_dir),
        "num_train_epochs": train_cfg.get("num_train_epochs", 1),
        "learning_rate": float(train_cfg.get("learning_rate", 3e-5)),
        "lr_scheduler_type": train_cfg.get("lr_scheduler_type", "cosine"),
        "weight_decay": train_cfg.get("weight_decay", 0.01),
        "adam_beta1": train_cfg.get("adam_beta1", 0.9),
        "adam_beta2": train_cfg.get("adam_beta2", 0.95),
        "max_grad_norm": train_cfg.get("max_grad_norm", 1.0),
        "per_device_train_batch_size": train_cfg.get("per_device_train_batch_size", 1),
        "gradient_accumulation_steps": train_cfg.get("gradient_accumulation_steps", 1),
        "per_device_eval_batch_size": train_cfg.get("per_device_eval_batch_size", 1),
        "bf16": train_cfg.get("bf16", False),
        "fp16": train_cfg.get("fp16", False),
        "gradient_checkpointing": train_cfg.get("gradient_checkpointing", False),
        # Run YAMLs declare gradient_checkpointing_kwargs.use_reentrant=false
        # (runs 06/07/08); previously this block was IGNORED — the manual
        # model.gradient_checkpointing_enable() call hardcoded use_reentrant=False
        # and HF Trainer's internal enable defaulted to use_reentrant=True,
        # fighting each other. Forward the YAML value (default: non-reentrant).
        "gradient_checkpointing_kwargs": train_cfg.get(
            "gradient_checkpointing_kwargs", {"use_reentrant": False}
        ),
        # SFT-specific: TRL@main uses `max_length` (not `max_seq_length`).
        "max_length": train_cfg.get("max_seq_length", 1024),
        "packing": train_cfg.get("packing", False),
        "packing_strategy": train_cfg.get("packing_strategy", "bfd"),
        "padding_free": train_cfg.get("padding_free", False),
        "completion_only_loss": train_cfg.get("completion_only_loss", True),
        "assistant_only_loss": train_cfg.get("assistant_only_loss", False),
        "use_liger_kernel": train_cfg.get("use_liger_kernel", False),
        "loss_type": train_cfg.get("loss_type", "default"),
        "dataloader_num_workers": train_cfg.get("dataloader_num_workers", 0),
        # NOTE: enable_thinking=False is enforced via PER-ROW
        # chat_template_kwargs (see Phase I below), NOT here. TRL 1.4.0's
        # SFTConfig has no chat_template_kwargs field — passing it as a
        # SFTConfig kwarg raises TypeError at construct. TRL reads
        # chat_template_kwargs from each dataset example inside tokenize_fn
        # (sft_trainer.py:1454,1461,1492).
        "eval_strategy": train_cfg.get("eval_strategy", "no"),
        "eval_steps": train_cfg.get("eval_steps", 50),
        "save_strategy": train_cfg.get("save_strategy", "no"),
        "save_total_limit": train_cfg.get("save_total_limit", 1),
        "save_only_model": train_cfg.get("save_only_model", False),
        "load_best_model_at_end": train_cfg.get("load_best_model_at_end", False),
        "metric_for_best_model": train_cfg.get("metric_for_best_model", "eval_loss"),
        "greater_is_better": train_cfg.get("greater_is_better", False),
        "logging_steps": train_cfg.get("logging_steps", 10),
        "seed": train_cfg.get("seed", 1234),
        "data_seed": train_cfg.get("data_seed", 42),
        "report_to": "none",
    }
    # Warmup scheduling. TRL/transformers 5.x deprecates `warmup_ratio`
    # (slated for removal in v5.2; the current 5.8.0 still emits a
    # DeprecationWarning). The forward path is `warmup_steps`, which now
    # accepts a float<1 interpreted as a ratio of total steps. We forward
    # whichever the run config sets, preferring `warmup_steps`.
    if "warmup_steps" in train_cfg and train_cfg["warmup_steps"] is not None:
        sft_kwargs["warmup_steps"] = train_cfg["warmup_steps"]
    elif "warmup_ratio" in train_cfg and train_cfg["warmup_ratio"] is not None:
        sft_kwargs["warmup_ratio"] = train_cfg["warmup_ratio"]
    else:
        sft_kwargs["warmup_steps"] = 0.05

    # cosine_min_lr_ratio: every run YAML declares 0.10 (min LR = 10% of
    # initial). Plain "cosine" decays to 0 — the YAML key was previously
    # silently dropped. transformers 5.x exposes "cosine_with_min_lr" which
    # honors lr_scheduler_kwargs.min_lr_rate (a 0..1 ratio). Upgrade the
    # scheduler iff the run requests a non-zero floor.
    if (
        train_cfg.get("lr_scheduler_type") == "cosine"
        and train_cfg.get("cosine_min_lr_ratio") is not None
        and float(train_cfg["cosine_min_lr_ratio"]) > 0
    ):
        sft_kwargs["lr_scheduler_type"] = "cosine_with_min_lr"
        sft_kwargs["lr_scheduler_kwargs"] = {
            "min_lr_rate": float(train_cfg["cosine_min_lr_ratio"]),
        }
    # Optional max_steps (smoke uses this).
    if "max_steps" in train_cfg and train_cfg["max_steps"] is not None:
        sft_kwargs["max_steps"] = int(train_cfg["max_steps"])
    sft_config = SFTConfig(**sft_kwargs)

    # ---------- Phase M: Trainer + early stopping ----------
    callbacks = [NaNGuardCallback()]
    # Early stopping only when load_best_model_at_end + an eval strategy are set,
    # else HF Trainer will refuse it.
    if train_cfg.get("load_best_model_at_end", False) and train_cfg.get(
        "eval_strategy", "no"
    ) != "no":
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=train_cfg.get(
                    "early_stopping_patience", 2
                )
            )
        )
    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=callbacks,
    )

    # ---------- Phase N: MoE aux-loss assertion (Spec §6.1.3) ----------
    if model_cfg.get("is_moe", False):
        if run_cfg.get("distributed", {}).get("strategy") == "fsdp2":
            print(
                "MOE-AUX-CHECK: skipping pre-train sample forward under FSDP2; "
                "the wrapped training forward will exercise router outputs."
            )
        else:
            trainer.model.eval()
            with torch.no_grad():
                dl = trainer.get_train_dataloader()
                sample_batch = next(iter(dl))
                # Move tensors to model device.
                device = next(trainer.model.parameters()).device
                sample_batch = {
                    k: (v.to(device) if hasattr(v, "to") else v)
                    for k, v in sample_batch.items()
                }
                out = trainer.model(**sample_batch, output_router_logits=True)
            has_aux = (
                getattr(out, "aux_loss", None) is not None
                or getattr(out, "router_logits", None) is not None
            )
            if not has_aux:
                raise RuntimeError(
                    f"MoE run {rid} ({model_cfg['name']}, e.g. Qwen3 MoE) was "
                    f"requested but the model output does not carry aux_loss or "
                    f"router_logits even with output_router_logits=True. The aux "
                    f"load-balancing loss will silently be missing. Refusing to "
                    f"train. Check that `is_moe: true` matches the actual model "
                    f"and that the transformers version exposes aux outputs."
                )
            trainer.model.train()

    # ---------- Phase O: train + checkpoint ----------
    resume_from_checkpoint = None
    if not args.max_steps and os.environ.get("EVOL_DISABLE_AUTO_RESUME", "0") != "1":
        latest_checkpoint = _latest_numbered_checkpoint(output_dir)
        if latest_checkpoint is not None and previous_status not in {"done", "smoke_done"}:
            resume_from_checkpoint = latest_checkpoint
            print(
                f"RESUME: {rid} from {resume_from_checkpoint} "
                f"(previous STATUS={previous_status!r})",
                flush=True,
            )

    train_result = trainer.train(
        resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None
    )
    final_dir = output_dir / "checkpoint-final"
    is_main_process = trainer.is_world_process_zero()
    published_best_checkpoint = None
    if is_main_process:
        published_best_checkpoint = publish_best_checkpoint(
            output_dir,
            trainer.state.best_model_checkpoint,
            getattr(trainer.state, "best_global_step", None),
        )
    # FSDP FULL_STATE_DICT already writes a normal HF checkpoint during the
    # Trainer save step (`checkpoint-<global_step>`). A second final
    # trainer.save_model() can spend a long time re-gathering the full state on
    # rank0 and is redundant because eval_all.sh prefers checkpoint-best.
    skip_final_model_save = run_cfg.get("distributed", {}).get("strategy") == "fsdp2"
    if not skip_final_model_save:
        trainer.save_model(str(final_dir))
        if is_main_process:
            tok.save_pretrained(str(final_dir))

    # ---------- Phase P: training_log.json + STATUS=done ----------
    metrics = {}
    try:
        # train_result.metrics is the canonical dict on HF Trainer.
        metrics = dict(train_result.metrics)
    except Exception:
        pass
    # Persist trainer.state.log_history — per-step train loss + per-eval
    # eval_loss + lr trace. Without this the operator has no loss curve
    # from saved artifacts (only the final scalars + stdout), and
    # plot_per_run_loss_curve / plot_lr_check both emit empty PNGs.
    # HF Trainer's train_result.metrics does NOT carry eval_loss either —
    # it only lives in log_history. Surface the final eval_loss into
    # metrics so SUMMARY.csv consumers (summarize.py:50) find it.
    try:
        log_history = list(trainer.state.log_history)
    except Exception:
        log_history = []
    if "eval_loss" not in metrics:
        for entry in reversed(log_history):
            if "eval_loss" in entry:
                metrics["eval_loss"] = entry["eval_loss"]
                break
    log = {
        "run_id": rid,
        "metrics": metrics,
        "history": log_history,
        "model_name": model_cfg["name"],
        "resolved_model_name": model_source.name,
        "resolved_model_local_path": model_source.local_path,
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
        "n_runs_kept": n_runs_kept,
        "git_sha": os.environ.get("GIT_SHA", "?"),
        "published_best_checkpoint": published_best_checkpoint,
        "skipped_final_model_save": skip_final_model_save,
    }
    # Per-run trainer state fields useful for post-hoc debugging — none of
    # these are in train_result.metrics; they live on trainer.state. Iter-6
    # audit flagged that without these the operator can't tell from saved
    # artifacts whether a run hit max_steps before max_epochs, which
    # checkpoint was selected as best, or what its best_metric value was.
    try:
        log["global_step"] = trainer.state.global_step
        log["epoch"] = trainer.state.epoch
        log["best_metric"] = trainer.state.best_metric
        log["best_model_checkpoint"] = trainer.state.best_model_checkpoint
    except Exception:
        pass
    if is_main_process:
        (output_dir / "training_log.json").write_text(json.dumps(log, indent=2) + "\n")
    # Distinguish smoke runs from real runs in STATUS so train_all.sh's
    # `grep -q "^done$"` skip-check doesn't treat a 1-step smoke as a
    # completed full run. Without this, an operator who follows the
    # README quick-start (`MAX_STEPS_OVERRIDE=1 train_all.sh` for smoke,
    # then plain `run_pipeline.sh` for the real sweep) ends up with
    # 1-step-trained checkpoints feeding the eval phase silently.
    status = "smoke_done" if (args.smoke_test or args.max_steps is not None) else "done"
    if is_main_process:
        (output_dir / "STATUS").write_text(status + "\n")
        print(f"DONE: {rid} -> {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
