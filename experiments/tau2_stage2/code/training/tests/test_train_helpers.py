import json
from pathlib import Path

import pytest

from training.train import (
    load_run_config,
    load_plan_config,
    subsample_runs_stratified,
    filter_rows_by_runs,
    build_runs_by_domain,
    apply_heldout_filter,
    publish_best_checkpoint,
)


def test_sft_config_accepts_train_py_kwargs():
    """Regression: train.py's full sft_kwargs dict must be valid for TRL SFTConfig.

    Iter-9 caught: iter-8 fix passed `chat_template_kwargs` to SFTConfig —
    TRL 1.4.0 has no such field, so every run would have crashed at
    trainer construction. Iter-12 caught (during user's final verification):
    eval_strategy=steps + save_strategy=epoch + load_best_model_at_end=True
    is REJECTED by transformers 5.x — strategies must match (or
    save_strategy=best). The fix was save_strategy: epoch → best in all 10
    run YAMLs.

    This test mirrors the FULL sft_kwargs dict (including the eval/save
    strategy combination from the production run YAMLs) so future drift —
    either a non-SFTConfig kwarg OR an incompatible strategy combination —
    surfaces in pytest before it reaches the H200.

    Keep in sync with `code/training/train.py:347` `sft_kwargs` builder
    AND with the eval/save knobs in `code/training/configs/runs/*.yaml`.
    """
    trl = pytest.importorskip("trl")
    from trl import SFTConfig

    # Full kwarg set mirroring train.py + a representative run YAML (05).
    sft_kwargs = {
        "output_dir": "/tmp/test_sft_config",
        "num_train_epochs": 1,
        "learning_rate": 3e-5,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "max_grad_norm": 1.0,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "per_device_eval_batch_size": 1,
        "bf16": False,
        "fp16": False,
        "gradient_checkpointing": False,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "max_length": 1024,
        "packing": False,
        "packing_strategy": "bfd",
        "padding_free": False,
        "completion_only_loss": True,
        "assistant_only_loss": False,
        "use_liger_kernel": False,
        "loss_type": "default",
        "dataloader_num_workers": 0,
        # Eval/save/load_best — the iter-12 collision surface.
        # eval_strategy=epoch matches production run YAMLs after iter-13
        # (eval_steps=50 was too coarse for small-cohort runs — 0 evals
        # fired). Per-epoch eval keeps cohort comparisons apples-to-apples.
        "eval_strategy": "epoch",
        "save_strategy": "best",
        "save_total_limit": 2,
        "save_only_model": False,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        # Misc
        "logging_steps": 10,
        "report_to": "none",
        "warmup_steps": 0.05,
        "lr_scheduler_kwargs": {"min_lr_rate": 0.1},
        "seed": 42,
        "data_seed": 42,
    }
    cfg = SFTConfig(**sft_kwargs)
    assert cfg.output_dir == "/tmp/test_sft_config"
    # chat_template_kwargs is intentionally NOT a field — TRL 1.4.0 reads
    # it per-row from the dataset. Confirm absence so we don't regress.
    assert "chat_template_kwargs" not in SFTConfig.__dataclass_fields__


def test_load_run_config(tmp_path: Path):
    yaml_text = """\
run_id: testrun
model:
  name: Qwen/Qwen3.5-2B
  revision: main
  is_moe: false
data:
  n_train_runs: 100
training:
  output_dir: out
  num_train_epochs: 5
  learning_rate: 0.00003
distributed:
  strategy: ddp
  accelerate_config: foo.yaml
moe: null
"""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml_text)
    cfg = load_run_config(cfg_path)
    assert cfg["run_id"] == "testrun"
    assert cfg["data"]["n_train_runs"] == 100


def test_subsample_runs_stratified_proportional():
    """50 of 273 runs sampled — should preserve 60/20/20 mix approximately."""
    runs_by_domain = {
        "telecom": [f"t{i}" for i in range(165)],
        "retail":  [f"r{i}" for i in range(54)],
        "airline": [f"a{i}" for i in range(54)],
    }
    sampled = subsample_runs_stratified(runs_by_domain, target_n=50, seed=42)
    counts = {"telecom": 0, "retail": 0, "airline": 0}
    for r in sampled:
        if r.startswith("t"): counts["telecom"] += 1
        elif r.startswith("r"): counts["retail"] += 1
        elif r.startswith("a"): counts["airline"] += 1
    assert sum(counts.values()) == 50
    assert 28 <= counts["telecom"] <= 32
    assert 8 <= counts["retail"] <= 12
    assert 8 <= counts["airline"] <= 12


def test_subsample_full_returns_all():
    runs_by_domain = {"telecom": ["t0", "t1"], "airline": ["a0"]}
    sampled = subsample_runs_stratified(runs_by_domain, target_n=3, seed=42)
    assert set(sampled) == {"t0", "t1", "a0"}


def test_subsample_deterministic():
    """Same seed → same selection."""
    runs_by_domain = {
        "telecom": [f"t{i}" for i in range(20)],
        "airline": [f"a{i}" for i in range(10)],
    }
    a = subsample_runs_stratified(runs_by_domain, target_n=10, seed=42)
    b = subsample_runs_stratified(runs_by_domain, target_n=10, seed=42)
    assert a == b


def test_filter_rows_by_runs():
    rows = [
        {"_meta": {"run_dir": "airline/0_300_cs"}, "x": 1},
        {"_meta": {"run_dir": "airline/1_300_cs"}, "x": 2},
        {"_meta": {"run_dir": "telecom/5_300_cs"}, "x": 3},
    ]
    keep = filter_rows_by_runs(rows, run_dirs={"airline/0_300_cs", "telecom/5_300_cs"})
    assert len(keep) == 2
    assert {r["x"] for r in keep} == {1, 3}


def test_apply_heldout_filter_drops_matching_pairs(tmp_path: Path):
    """Heldout filter drops rows whose (domain, task_id) is in the heldout map."""
    import json as _json
    ids_path = tmp_path / "heldout_task_ids.json"
    ids_path.write_text(_json.dumps({
        "airline": ["t1", "t3"],
        "telecom": ["t5"],
    }))
    rows = [
        {"_meta": {"domain": "airline", "task_id": "t1"}, "x": 1},  # drop
        {"_meta": {"domain": "airline", "task_id": "t2"}, "x": 2},  # keep
        {"_meta": {"domain": "airline", "task_id": "t3"}, "x": 3},  # drop
        {"_meta": {"domain": "telecom", "task_id": "t5"}, "x": 4},  # drop
        {"_meta": {"domain": "retail",  "task_id": "t5"}, "x": 5},  # keep — different domain
    ]
    filtered, n_dropped, present = apply_heldout_filter(rows, ids_path)
    assert present is True
    assert n_dropped == 3
    assert {r["x"] for r in filtered} == {2, 5}


def test_publish_best_checkpoint_uses_stable_symlink(tmp_path: Path):
    output_dir = tmp_path / "run"
    best = output_dir / "checkpoint-3"
    best.mkdir(parents=True)
    (best / "model.safetensors").write_text("placeholder")

    published = publish_best_checkpoint(output_dir, str(best))

    dst = output_dir / "checkpoint-best"
    assert published == str(dst)
    assert dst.is_dir()
    assert (dst / "model.safetensors").read_text() == "placeholder"
    if dst.is_symlink():
        assert dst.readlink() == Path("checkpoint-3")


def test_publish_best_checkpoint_falls_back_to_best_global_step(tmp_path: Path):
    output_dir = tmp_path / "run"
    best = output_dir / "checkpoint-7"
    best.mkdir(parents=True)
    (best / "model.safetensors").write_text("placeholder")

    published = publish_best_checkpoint(
        output_dir,
        best_model_checkpoint=None,
        best_global_step=7,
    )

    dst = output_dir / "checkpoint-best"
    assert published == str(dst)
    assert (dst / "model.safetensors").read_text() == "placeholder"


def test_publish_best_checkpoint_falls_back_to_latest_checkpoint(tmp_path: Path):
    output_dir = tmp_path / "run"
    for step in (1, 5, 3):
        ckpt = output_dir / f"checkpoint-{step}"
        ckpt.mkdir(parents=True)
        (ckpt / "step.txt").write_text(str(step))
    (output_dir / "checkpoint-final").mkdir()

    published = publish_best_checkpoint(output_dir, best_model_checkpoint=None)

    dst = output_dir / "checkpoint-best"
    assert published == str(dst)
    assert (dst / "step.txt").read_text() == "5"


def test_apply_heldout_filter_no_file_passthrough(tmp_path: Path):
    """Iter-15: missing heldout_task_ids.json → no-op pass-through, present=False.

    The caller (train.main) prints a stderr warning when present=False so
    the operator notices before a full sweep silently ships 8% leakage.
    This test pins the contract.
    """
    ids_path = tmp_path / "heldout_task_ids.json"  # deliberately missing
    rows = [
        {"_meta": {"domain": "airline", "task_id": "t1"}, "x": 1},
        {"_meta": {"domain": "telecom", "task_id": "t5"}, "x": 2},
    ]
    filtered, n_dropped, present = apply_heldout_filter(rows, ids_path)
    assert present is False
    assert n_dropped == 0
    assert filtered == rows


def test_apply_heldout_filter_warning_emitted_via_main_else_branch():
    """Iter-15 regression: verify train.main's else-branch prints WARNING.

    Pure source-grep test — confirms the warning text is present in
    train.py source so a refactor doesn't silently drop it. The actual
    warning behavior is also covered by the apply_heldout_filter unit
    tests; this test pins the user-visible string.
    """
    from training import train as train_mod
    import inspect
    src = inspect.getsource(train_mod.main)
    assert "WARNING: heldout_task_ids.json not found" in src
    assert "SKIPPING heldout-leakage filter" in src


def test_build_runs_by_domain():
    rows = [
        {"_meta": {"domain": "airline", "run_dir": "airline/0_300_cs"}},
        {"_meta": {"domain": "airline", "run_dir": "airline/0_300_cs"}},  # dup run
        {"_meta": {"domain": "airline", "run_dir": "airline/1_300_cs"}},
        {"_meta": {"domain": "telecom", "run_dir": "telecom/5_300_cs"}},
    ]
    out = build_runs_by_domain(rows)
    assert set(out["airline"]) == {"airline/0_300_cs", "airline/1_300_cs"}
    assert out["telecom"] == ["telecom/5_300_cs"]
