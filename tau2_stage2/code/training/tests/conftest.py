"""Shared pytest fixtures."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def bundle_root() -> Path:
    """Path to the bundle root (parent of code/)."""
    return Path(__file__).resolve().parents[3]

@pytest.fixture
def stage2_root(bundle_root: Path) -> Path:
    """Path to data_processed/stage2_v1."""
    return bundle_root / "data_processed" / "stage2_v1"

@pytest.fixture
def sample_train_row(stage2_root: Path) -> dict:
    """First row of train.jsonl as a sample."""
    with (stage2_root / "train.jsonl").open() as f:
        return json.loads(f.readline())

@pytest.fixture
def sample_train_rows_10(stage2_root: Path) -> list:
    """First 10 rows of train.jsonl."""
    rows = []
    with (stage2_root / "train.jsonl").open() as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            rows.append(json.loads(line))
    return rows
