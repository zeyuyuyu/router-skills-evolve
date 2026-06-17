"""Smoke tests that the three top-level training configs parse cleanly."""

import yaml
from pathlib import Path

_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_plan_c_prime_parses():
    d = yaml.safe_load((_CONFIGS / "plan_c_prime.yaml").read_text())
    assert d["name"] == "plan_c_prime"
    assert len(d["runs"]) == 10
    # All run ids are unique and follow the naming convention.
    ids = [r["id"] for r in d["runs"]]
    assert len(set(ids)) == 10
    for rid in ids:
        assert rid[:2].isdigit() and rid[2] == "_"


def test_accelerate_fsdp2_parses():
    d = yaml.safe_load((_CONFIGS / "accelerate_fsdp2.yaml").read_text())
    assert d["distributed_type"] == "FSDP"
    assert d["mixed_precision"] == "bf16"
    assert d["fsdp_config"]["fsdp_version"] == 2
    assert d["num_processes"] == 8


def test_accelerate_ddp_parses():
    d = yaml.safe_load((_CONFIGS / "accelerate_ddp.yaml").read_text())
    assert d["distributed_type"] == "MULTI_GPU"
    assert d["mixed_precision"] == "bf16"
    assert d["num_processes"] == 8
