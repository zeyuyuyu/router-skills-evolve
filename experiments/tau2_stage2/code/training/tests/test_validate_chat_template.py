"""Tests for chat-template pre-launch validator (Spec §5.3).

The validator is a HARD-STOP gate before training: every row must render
with the model's chat template and survive an encode/decode round-trip.

Tests that need a real Qwen tokenizer use a session-scoped fixture that
skips cleanly when the network or HF Hub is unavailable.
"""
import json
from pathlib import Path

import pytest

# Skip the whole module if transformers is not available.
pytest.importorskip("transformers")

from training.data.validate_chat_template import (
    contains_pre_existing_thinking_block,
    validate_jsonl,
    validate_row,
)


GOOD_ROW = {
    "prompt": [
        {"role": "system", "content": "You are a helpful agent."},
        {"role": "user", "content": "Hello"},
    ],
    "completion": [{"role": "assistant", "content": "Hi! How can I help?"}],
    "tools": [],
    "_meta": {"row_id": "good"},
}

BAD_THINKING = {
    "prompt": [
        {"role": "system", "content": "Sys"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "<think></think>actual reply"},
    ],
    "completion": [{"role": "assistant", "content": "ok"}],
    "tools": [],
    "_meta": {"row_id": "bad_thinking"},
}


@pytest.fixture(scope="session")
def qwen_tokenizer():
    """Session-scoped tokenizer; skip cleanly if download/load fails."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True
        )
    except Exception as e:  # pragma: no cover - network-dependent
        pytest.skip(f"Qwen tokenizer unavailable: {e}")


def test_contains_pre_existing_thinking_block():
    """Pure-Python check; no tokenizer needed."""
    assert contains_pre_existing_thinking_block(BAD_THINKING) is True
    assert contains_pre_existing_thinking_block(GOOD_ROW) is False


def test_validate_good_row(qwen_tokenizer):
    """A clean row passes validation."""
    ok, reason = validate_row(GOOD_ROW, qwen_tokenizer)
    assert ok is True, f"Expected pass, got reason: {reason!r}"
    assert reason == ""


def test_validate_thinking_block_fails(qwen_tokenizer):
    """A row with pre-existing <think>...</think> fails the data-quality check."""
    ok, reason = validate_row(BAD_THINKING, qwen_tokenizer)
    assert ok is False
    assert "thinking" in reason.lower()


def test_validate_jsonl_writes_report(qwen_tokenizer, tmp_path: Path):
    """End-to-end: write input jsonl, validate, read report."""
    src = tmp_path / "src.jsonl"
    src.write_text(
        json.dumps(GOOD_ROW) + "\n" + json.dumps(BAD_THINKING) + "\n"
    )
    out = tmp_path / "report.json"
    summary = validate_jsonl(src, qwen_tokenizer, out)
    assert summary["n_total"] == 2
    assert summary["n_pass"] == 1
    assert summary["n_fail"] == 1
    assert out.exists()
    on_disk = json.loads(out.read_text())
    assert on_disk["n_total"] == 2
    # The bad row should be reported with a 'thinking' reason.
    assert any(
        "thinking" in f["reason"].lower() for f in on_disk["failures"]
    )
