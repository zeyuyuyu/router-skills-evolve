"""Verify all 10 per-run YAMLs parse and have required schema."""
import yaml
from pathlib import Path

import pytest

_RUNS_DIR = Path(__file__).resolve().parents[1] / "configs" / "runs"

EXPECTED_RUNS = [
    "01_qwen3_5_2b_273", "02_qwen3_5_4b_50", "03_qwen3_5_4b_100",
    "04_qwen3_5_4b_200", "05_qwen3_5_4b_273", "06_qwen3_5_9b_50",
    "07_qwen3_5_9b_273", "08_qwen3_6_35b_a3b_273",
    "09_qwen3_5_4b_273_lr1e5", "10_qwen3_5_4b_273_lr3e5",
]


@pytest.mark.parametrize("run_id", EXPECTED_RUNS)
def test_run_yaml_loads(run_id: str):
    p = _RUNS_DIR / f"{run_id}.yaml"
    assert p.exists(), f"missing: {p}"
    d = yaml.safe_load(p.read_text())
    assert d["run_id"] == run_id
    assert d["model"]["name"].startswith("Qwen/")
    assert d["training"]["per_device_train_batch_size"] >= 1
    expected_max_seq_length = 16384 if run_id == "08_qwen3_6_35b_a3b_273" else 32768
    assert d["training"]["max_seq_length"] == expected_max_seq_length
    assert d["training"]["completion_only_loss"] is True
    assert d["training"]["assistant_only_loss"] is False
    assert d["training"]["loss_type"] == "chunked_nll"
    assert d["training"]["packing_strategy"] == "bfd"
    assert d["training"]["padding_free"] is True
    assert d["training"]["use_liger_kernel"] is False
    assert d["distributed"]["strategy"] in ("ddp", "fsdp2")


def test_unique_run_ids():
    ids = []
    for r in EXPECTED_RUNS:
        d = yaml.safe_load((_RUNS_DIR / f"{r}.yaml").read_text())
        ids.append(d["run_id"])
    assert len(ids) == len(set(ids)) == 10


def test_moe_run_has_correct_aux_loss_coef():
    """CRITICAL: router_aux_loss_coef must be 0.001 (Spec §6.1.3)."""
    p = _RUNS_DIR / "08_qwen3_6_35b_a3b_273.yaml"
    d = yaml.safe_load(p.read_text())
    assert d["model"]["is_moe"] is True
    assert d["moe"]["router_aux_loss_coef"] == 0.001
    assert d["moe"]["output_router_logits"] is True
    assert d["moe"]["preserve_thinking"] is False


def test_dense_runs_have_null_moe():
    for r in EXPECTED_RUNS:
        if r == "08_qwen3_6_35b_a3b_273":
            continue
        d = yaml.safe_load((_RUNS_DIR / f"{r}.yaml").read_text())
        assert d["model"]["is_moe"] is False
        assert d["moe"] is None


def test_ddp_runs_use_ddp_accelerate_config():
    for r in ["01_qwen3_5_2b_273", "02_qwen3_5_4b_50", "05_qwen3_5_4b_273", "09_qwen3_5_4b_273_lr1e5"]:
        d = yaml.safe_load((_RUNS_DIR / f"{r}.yaml").read_text())
        assert d["distributed"]["strategy"] == "ddp"
        assert "ddp" in d["distributed"]["accelerate_config"]


def test_fsdp2_runs_use_fsdp2_accelerate_config():
    for r in ["06_qwen3_5_9b_50", "07_qwen3_5_9b_273", "08_qwen3_6_35b_a3b_273"]:
        d = yaml.safe_load((_RUNS_DIR / f"{r}.yaml").read_text())
        assert d["distributed"]["strategy"] == "fsdp2"
        assert "fsdp2" in d["distributed"]["accelerate_config"]


def test_fsdp2_runs_use_model_family_specific_accelerate_config():
    dense_runs = ["06_qwen3_5_9b_50", "07_qwen3_5_9b_273"]
    for r in dense_runs:
        d = yaml.safe_load((_RUNS_DIR / f"{r}.yaml").read_text())
        assert d["model"]["is_moe"] is False
        assert d["distributed"]["accelerate_config"].endswith("accelerate_fsdp2_dense.yaml")

    moe = yaml.safe_load((_RUNS_DIR / "08_qwen3_6_35b_a3b_273.yaml").read_text())
    assert moe["model"]["is_moe"] is True
    assert moe["distributed"]["accelerate_config"].endswith("accelerate_fsdp2_moe.yaml")


def test_fsdp2_runs_save_only_model():
    for r in ["06_qwen3_5_9b_50", "07_qwen3_5_9b_273", "08_qwen3_6_35b_a3b_273"]:
        d = yaml.safe_load((_RUNS_DIR / f"{r}.yaml").read_text())
        assert d["training"]["save_only_model"] is True


def test_lr_values_per_tier():
    # 2B → 3e-5; 4B (regular) → 2e-5; 9B/35B → 1e-5; 4B LR variants → 1e-5, 3e-5.
    cases = [
        ("01_qwen3_5_2b_273", 3.0e-5),
        ("02_qwen3_5_4b_50", 2.0e-5),
        ("03_qwen3_5_4b_100", 2.0e-5),
        ("04_qwen3_5_4b_200", 2.0e-5),
        ("05_qwen3_5_4b_273", 2.0e-5),
        ("06_qwen3_5_9b_50", 1.0e-5),
        ("07_qwen3_5_9b_273", 1.0e-5),
        ("08_qwen3_6_35b_a3b_273", 1.0e-5),
        ("09_qwen3_5_4b_273_lr1e5", 1.0e-5),
        ("10_qwen3_5_4b_273_lr3e5", 3.0e-5),
    ]
    for run_id, expected_lr in cases:
        d = yaml.safe_load((_RUNS_DIR / f"{run_id}.yaml").read_text())
        assert d["training"]["learning_rate"] == expected_lr, f"{run_id}: got {d['training']['learning_rate']}"
