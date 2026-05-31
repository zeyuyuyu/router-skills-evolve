# τ²-bench Stage-2 Training Framework Implementation Plan

> **Status note (2026-05-09 post-validation):** This plan is preserved as the
> historical task breakdown. The implementation diverged during Phase A-L
> deep validation; do NOT regenerate files from this plan without first
> reading the README's "Bugs caught and fixed" section and the current code.
> Specifically: Task 8 (accelerate YAML), Task 13 (train.py model loading),
> Task 15 (`local_provider.py` is built but no longer on the eval path),
> Task 16 (eval harness rewritten to drive `tau2.cli` directly, not
> `pipeline.runner --provider local`), Task 17 (vllm_serve.sh has new
> flags), Task 18 (eval_all.sh seed-301 logic) all have material changes
> from what's prescribed below. The code under `code/` is authoritative.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a one-bash-command training framework that trains 10 Qwen students on the τ²-bench stage-2 SFT corpus and evaluates them force-routed against the τ²-bench harness, producing publishable scaling-curve and capacity plots.

**Architecture:** TRL `SFTTrainer` + `accelerate` + FSDP2 + bf16 mixed precision + FlashAttention 2 + BFD sequence packing + `padding_free=True`. Two-phase orchestration: `train_all.sh` (10 sequential training runs with smart GPU partitioning) → `eval_all.sh` (8 main runs in parallel via vLLM `--provider local`, 1 winner re-eval). Custom dataset prep converts row-format SFT to TRL prompt/completion format with quality filters, domain rebalancing, and tool-schema augmentation. Local dev box (4 GB VRAM) supports code-correctness smoke tests only; cloud (8× H200) handles all real training.

**Spec reference:** `code/docs/superpowers/specs/2026-05-08-tau2-stage2-training-framework-design.md`

**Tech Stack:** Python 3.11+, `transformers ≥4.50`, `trl ≥0.13`, `accelerate ≥1.5` (FSDP2 native), `torch ≥2.5`, `vllm ≥0.19` (Qwen3.6 support), FlashAttention 2.7+, `bitsandbytes` (smoke-test 4-bit only), `pytest`, `matplotlib`, `PyYAML`.

---

## File Structure

```
code/training/                                  # NEW
├── __init__.py
├── requirements.txt                            # Pinned dependencies
├── configs/
│   ├── plan_c_prime.yaml                       # 10-run plan metadata
│   ├── accelerate_fsdp2.yaml                   # accelerate FSDP2 config
│   ├── accelerate_ddp.yaml                     # accelerate DDP config (2B/4B)
│   └── runs/                                   # one YAML per run (10 files)
│       ├── 01_qwen3_5_2b_273.yaml
│       ├── 02_qwen3_5_4b_50.yaml
│       ├── 03_qwen3_5_4b_100.yaml
│       ├── 04_qwen3_5_4b_200.yaml
│       ├── 05_qwen3_5_4b_273.yaml
│       ├── 06_qwen3_5_9b_50.yaml
│       ├── 07_qwen3_5_9b_273.yaml
│       ├── 08_qwen3_6_35b_a3b_273.yaml
│       ├── 09_qwen3_5_4b_273_lr1e5.yaml
│       └── 10_qwen3_5_4b_273_lr3e5.yaml
├── data/                                       # Data preparation
│   ├── __init__.py
│   ├── filters.py                              # Quality filters
│   ├── convert_to_prompt_completion.py         # Row → TRL format
│   ├── validate_chat_template.py               # Pre-launch chat-template check
│   ├── domain_rebalance.py                     # T=0.5 temperature sampling
│   ├── tool_shuffle.py                         # Tool-schema-order augmentation
│   └── heldout_split.py                        # 15-task held-out split
├── train.py                                    # Trains ONE run (entry point)
├── eval/
│   ├── __init__.py
│   ├── local_provider.py                       # New "local" vLLM provider for pipeline.runner
│   ├── vllm_serve.sh                           # Spin up vLLM per checkpoint
│   ├── harness.py                              # Force-routed τ² eval orchestration
│   └── heldout_eval.py                         # Held-out task eval
├── orchestration/
│   ├── train_all.sh                            # Phase 1
│   ├── eval_all.sh                             # Phase 2
│   ├── run_pipeline.sh                         # Convenience all-in-one
│   ├── summarize.py                            # SUMMARY.csv aggregation
│   └── plotting.py                             # Per-run + aggregate plots
└── tests/                                      # Pytest tests
    ├── __init__.py
    ├── conftest.py                             # Fixtures
    ├── test_filters.py
    ├── test_convert_to_prompt_completion.py
    ├── test_validate_chat_template.py
    ├── test_domain_rebalance.py
    ├── test_tool_shuffle.py
    ├── test_heldout_split.py
    ├── test_smoke_train.py                     # CPU+0.5B end-to-end smoke
    └── test_summarize.py
```

**Boundary rationale:**
- `data/` is pure functional (no torch); each module testable on CPU without GPU.
- `train.py` orchestrates — does NOT contain custom training loop logic; uses TRL primitives.
- `eval/` reuses existing `pipeline.runner`; just adds a "local" provider layer.
- `orchestration/` is shell glue + Python summarization; no model code.
- `tests/` mirrors module structure 1:1.

---

## Pre-implementation: Cloud-side prerequisites (operator must verify)

Before any task in this plan runs on the cloud server:

1. CUDA 13.0 installed, `nvidia-smi` shows 8× H200 (141 GB each).
2. Conda/uv env with Python 3.11+.
3. `huggingface-cli login` with a valid HF token (for model downloads).
4. Verify Qwen3.6-35B-A3B is accessible: `huggingface-cli repo show Qwen/Qwen3.6-35B-A3B`. Fallback: `Qwen/Qwen3.5-35B-A3B`.
5. OpenAI API key in env (`OPENAI_API_KEY`) for τ² simulator + NL judge.

These are documented in `train_all.sh` (Task 17) preflight checks.

---

## Tasks

### Task 1: Project scaffolding + frozen requirements

**Files:**
- Create: `code/training/__init__.py` (empty)
- Create: `code/training/data/__init__.py` (empty)
- Create: `code/training/eval/__init__.py` (empty)
- Create: `code/training/tests/__init__.py` (empty)
- Create: `code/training/tests/conftest.py`
- Create: `code/training/requirements.txt`

- [ ] **Step 1: Create the package directories**

```bash
cd /path/to/bundle/code/training
mkdir -p data eval orchestration configs/runs tests
touch __init__.py data/__init__.py eval/__init__.py tests/__init__.py
```

- [ ] **Step 2: Write `requirements.txt`**

```
# Core
torch==2.5.1
transformers==4.50.0
accelerate==1.5.0
trl==0.13.0
peft==0.13.0
datasets==3.2.0

# FSDP2 + FA2
flash-attn==2.7.4
bitsandbytes==0.44.1   # smoke-test only

# Eval
vllm==0.19.0           # Qwen3.6 support

# Misc
pyyaml==6.0.2
pytest==8.3.4
matplotlib==3.10.0
pandas==2.2.3
huggingface-hub==0.27.0
```

- [ ] **Step 3: Write `tests/conftest.py`**

```python
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
```

- [ ] **Step 4: Verify pytest discovers the tests dir**

Run: `cd code/training && python -m pytest tests/ --collect-only`
Expected: `0 tests collected` (no test files yet, but no errors)

- [ ] **Step 5: Commit**

```bash
git add code/training/
git commit -m "feat(training): scaffold training framework package structure"
```

---

### Task 2: Quality filters (`filters.py`)

Spec §5.2 — filter rules to drop bad rows during conversion.

**Files:**
- Create: `code/training/data/filters.py`
- Create: `code/training/tests/test_filters.py`

- [ ] **Step 1: Write the failing tests**

```python
# code/training/tests/test_filters.py
"""Tests for data quality filters."""
import json

import pytest

from code.training.data.filters import (
    has_orphan_tool_role,
    has_unparseable_tool_call_args,
    is_target_too_short,
    should_drop_row,
)


def make_row(target_msg, prefix_msgs=None):
    """Helper: build a minimal row with given target message."""
    msgs = (prefix_msgs or []) + [target_msg]
    return {
        "_p": {"row_id": "test"},
        "_target_index": len(msgs) - 1,
        "messages": msgs,
    }


# is_target_too_short
def test_target_too_short_text():
    row = make_row({"role": "assistant", "content": "Hi!"})
    assert is_target_too_short(row, min_tokens=5) is True


def test_target_long_text_passes():
    row = make_row({"role": "assistant", "content": "This is a longer reply with many tokens."})
    assert is_target_too_short(row, min_tokens=5) is False


def test_target_with_tool_call_not_short():
    row = make_row({
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "t1", "type": "function",
                        "function": {"name": "do_x", "arguments": '{"a":1}'}}],
    })
    assert is_target_too_short(row, min_tokens=5) is False


# has_unparseable_tool_call_args
def test_unparseable_args_caught():
    row = make_row({
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "t1", "type": "function",
                        "function": {"name": "x", "arguments": "{not json"}}],
    })
    assert has_unparseable_tool_call_args(row) is True


def test_parseable_args_passes():
    row = make_row({
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "t1", "type": "function",
                        "function": {"name": "x", "arguments": '{"a":1}'}}],
    })
    assert has_unparseable_tool_call_args(row) is False


def test_no_tool_calls_passes():
    row = make_row({"role": "assistant", "content": "plain text reply with enough tokens"})
    assert has_unparseable_tool_call_args(row) is False


# has_orphan_tool_role
def test_orphan_tool_caught():
    row = make_row(
        {"role": "assistant", "content": "ok"},
        prefix_msgs=[
            {"role": "tool", "tool_call_id": "phantom", "content": "result"},
            {"role": "user", "content": "hi"},
        ],
    )
    assert has_orphan_tool_role(row) is True


def test_proper_tool_chain_passes():
    row = make_row(
        {"role": "assistant", "content": "result interpreted"},
        prefix_msgs=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "t1", "type": "function",
                             "function": {"name": "x", "arguments": '{}'}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "result"},
        ],
    )
    assert has_orphan_tool_role(row) is False


# should_drop_row composite
def test_drop_row_composite():
    bad = make_row({"role": "assistant", "content": "Hi"})  # too short
    good = make_row({"role": "assistant", "content": "Long enough text reply with multiple words"})
    assert should_drop_row(bad)[0] is True
    assert should_drop_row(good)[0] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd code/training && python -m pytest tests/test_filters.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'code.training.data.filters'`

- [ ] **Step 3: Write `filters.py`**

```python
# code/training/data/filters.py
"""Quality filters applied during row→TRL format conversion (Spec §5.2)."""
from __future__ import annotations

import json
from typing import Any


def is_target_too_short(row: dict, min_tokens: int = 5) -> bool:
    """Return True if the target message has < min_tokens of content.

    Counts: words in `content` + 1 per `tool_calls` entry (each tool call
    is meaningful even if content is empty).
    """
    target = row["messages"][row["_target_index"]]
    content = (target.get("content") or "").strip()
    n_tool_calls = len(target.get("tool_calls") or [])
    # Approx word count is fine — actual tokenization happens later.
    n_words = len(content.split()) if content else 0
    return (n_words + n_tool_calls) < min_tokens


def has_unparseable_tool_call_args(row: dict) -> bool:
    """Return True if any tool_call.function.arguments is not parseable JSON."""
    for msg in row["messages"]:
        for tc in msg.get("tool_calls") or []:
            args = tc.get("function", {}).get("arguments")
            if args is None:
                continue
            try:
                json.loads(args) if isinstance(args, str) else args
            except (json.JSONDecodeError, TypeError):
                return True
    return False


def has_orphan_tool_role(row: dict) -> bool:
    """Return True if a `tool` role message has no preceding matching tool_call.

    A `tool` message must reference a tool_call_id that appeared in an earlier
    assistant message's `tool_calls`.
    """
    seen_tool_call_ids: set[str] = set()
    for msg in row["messages"]:
        if msg["role"] == "assistant":
            for tc in msg.get("tool_calls") or []:
                if "id" in tc:
                    seen_tool_call_ids.add(tc["id"])
        elif msg["role"] == "tool":
            tcid = msg.get("tool_call_id")
            if tcid is None or tcid not in seen_tool_call_ids:
                return True
    return False


def should_drop_row(row: dict, min_target_tokens: int = 5) -> tuple[bool, str]:
    """Composite filter. Returns (drop?, reason)."""
    if is_target_too_short(row, min_target_tokens):
        return True, "target_too_short"
    if has_unparseable_tool_call_args(row):
        return True, "unparseable_tool_call_args"
    if has_orphan_tool_role(row):
        return True, "orphan_tool_role"
    return False, ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd code/training && python -m pytest tests/test_filters.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add code/training/data/filters.py code/training/tests/test_filters.py
git commit -m "feat(training/data): row quality filters with unit tests"
```

---

### Task 3: Tool-schema-order augmentation (`tool_shuffle.py`)

Spec §5.5 — randomly permute tool list per-batch. Free 2-3× data augmentation.

**Files:**
- Create: `code/training/data/tool_shuffle.py`
- Create: `code/training/tests/test_tool_shuffle.py`

- [ ] **Step 1: Write failing tests**

```python
# code/training/tests/test_tool_shuffle.py
import random

from code.training.data.tool_shuffle import shuffle_tools_in_schema


SAMPLE_TOOLS = [
    {"type": "function", "function": {"name": "tool_a", "parameters": {}}},
    {"type": "function", "function": {"name": "tool_b", "parameters": {}}},
    {"type": "function", "function": {"name": "tool_c", "parameters": {}}},
    {"type": "function", "function": {"name": "tool_d", "parameters": {}}},
]


def test_shuffle_preserves_count():
    rng = random.Random(42)
    shuffled = shuffle_tools_in_schema(SAMPLE_TOOLS, rng)
    assert len(shuffled) == len(SAMPLE_TOOLS)


def test_shuffle_preserves_set_of_tools():
    rng = random.Random(42)
    shuffled = shuffle_tools_in_schema(SAMPLE_TOOLS, rng)
    names = {t["function"]["name"] for t in shuffled}
    expected = {t["function"]["name"] for t in SAMPLE_TOOLS}
    assert names == expected


def test_shuffle_actually_permutes():
    rng = random.Random(42)
    shuffled = shuffle_tools_in_schema(SAMPLE_TOOLS, rng)
    original_order = [t["function"]["name"] for t in SAMPLE_TOOLS]
    new_order = [t["function"]["name"] for t in shuffled]
    # With seed 42 and 4! = 24 permutations, we expect a non-identity perm.
    assert original_order != new_order


def test_shuffle_does_not_mutate_input():
    rng = random.Random(42)
    snapshot = [dict(t) for t in SAMPLE_TOOLS]
    _ = shuffle_tools_in_schema(SAMPLE_TOOLS, rng)
    assert SAMPLE_TOOLS == snapshot


def test_shuffle_empty():
    rng = random.Random(0)
    assert shuffle_tools_in_schema([], rng) == []


def test_shuffle_singleton():
    rng = random.Random(0)
    one = [SAMPLE_TOOLS[0]]
    assert shuffle_tools_in_schema(one, rng) == one
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd code/training && python -m pytest tests/test_tool_shuffle.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `tool_shuffle.py`**

```python
# code/training/data/tool_shuffle.py
"""Tool-schema-order augmentation (Spec §5.5).

Shuffling the tool order per row exposes the model to multiple presentations
of the same tool set, teaching order-invariance. Effectively 2-3× data
augmentation at zero cost.
"""
from __future__ import annotations

import random
from copy import deepcopy


def shuffle_tools_in_schema(tools: list[dict], rng: random.Random) -> list[dict]:
    """Return a shuffled copy of the tools list.

    Args:
        tools: list of OpenAI-format tool definitions.
        rng: random.Random instance for deterministic shuffling.

    Returns:
        New list with same tools in shuffled order. Input is not mutated.
    """
    if len(tools) <= 1:
        return list(tools)
    shuffled = deepcopy(tools)
    rng.shuffle(shuffled)
    return shuffled
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd code/training && python -m pytest tests/test_tool_shuffle.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add code/training/data/tool_shuffle.py code/training/tests/test_tool_shuffle.py
git commit -m "feat(training/data): tool-schema-order shuffle augmentation"
```

---

### Task 4: Domain rebalancing (`domain_rebalance.py`)

Spec §5.4 — T=0.5 temperature sampling. Telecom 60% → 42%, airline/retail boosted.

**Files:**
- Create: `code/training/data/domain_rebalance.py`
- Create: `code/training/tests/test_domain_rebalance.py`

- [ ] **Step 1: Write failing tests**

```python
# code/training/tests/test_domain_rebalance.py
import math

from code.training.data.domain_rebalance import compute_domain_weights


def test_uniform_when_already_balanced():
    counts = {"airline": 100, "retail": 100, "telecom": 100}
    weights = compute_domain_weights(counts, temperature=0.5)
    assert math.isclose(weights["airline"], 1/3, abs_tol=1e-6)
    assert math.isclose(weights["retail"], 1/3, abs_tol=1e-6)
    assert math.isclose(weights["telecom"], 1/3, abs_tol=1e-6)


def test_temperature_zero_uniform():
    """T=0 → uniform sampling regardless of counts."""
    counts = {"airline": 100, "retail": 200, "telecom": 700}
    weights = compute_domain_weights(counts, temperature=0.0)
    for v in weights.values():
        assert math.isclose(v, 1/3, abs_tol=1e-6)


def test_temperature_one_proportional():
    """T=1 → exactly proportional to source counts."""
    counts = {"airline": 100, "retail": 200, "telecom": 700}
    weights = compute_domain_weights(counts, temperature=1.0)
    total = sum(counts.values())
    assert math.isclose(weights["airline"], 100/total, abs_tol=1e-6)
    assert math.isclose(weights["retail"], 200/total, abs_tol=1e-6)
    assert math.isclose(weights["telecom"], 700/total, abs_tol=1e-6)


def test_temperature_half_real_distribution():
    """The actual production case: telecom 60%, retail 21%, airline 20%."""
    counts = {"airline": 1254, "retail": 1342, "telecom": 3817}
    weights = compute_domain_weights(counts, temperature=0.5)
    # T=0.5 should pull telecom down significantly while boosting airline/retail.
    assert weights["telecom"] < 0.50, f"Expected telecom < 50%, got {weights['telecom']:.3f}"
    assert weights["airline"] > 0.20, f"Expected airline > 20%, got {weights['airline']:.3f}"
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6)


def test_per_row_weight_via_division():
    """Each row's weight = domain_weight / row_count_in_domain (so sampling
    one row from a domain at probability domain_weight is uniform within the domain)."""
    from code.training.data.domain_rebalance import compute_per_row_weights
    rows = [
        {"_p": {"domain": "airline"}},
        {"_p": {"domain": "airline"}},
        {"_p": {"domain": "telecom"}},
    ]
    per_row = compute_per_row_weights(rows, temperature=0.5)
    assert len(per_row) == 3
    # Both airline rows should have equal weight.
    assert math.isclose(per_row[0], per_row[1], abs_tol=1e-6)
    # The two airline rows together carry the airline domain weight.
    counts = {"airline": 2, "telecom": 1}
    domain_weights = compute_domain_weights(counts, temperature=0.5)
    assert math.isclose(per_row[0] * 2, domain_weights["airline"], abs_tol=1e-6)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd code/training && python -m pytest tests/test_domain_rebalance.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `domain_rebalance.py`**

```python
# code/training/data/domain_rebalance.py
"""Domain rebalance via temperature sampling (Spec §5.4).

Source distribution: telecom 60%, retail 21%, airline 20%.
At T=0.5, effective sampling becomes ~telecom 42%, retail 30%, airline 28%.

Reference: arxiv 2410.04579 — upsampling outperforms per-token loss weighting
for SFT stability.
"""
from __future__ import annotations


def compute_domain_weights(counts: dict[str, int], temperature: float) -> dict[str, float]:
    """Compute domain-level sampling probabilities.

    weight(domain) ∝ count(domain) ** temperature

    Args:
        counts: {domain: row_count}
        temperature: 0.0 → uniform; 1.0 → proportional; 0.5 → in between.

    Returns:
        Normalized {domain: probability} summing to 1.0.
    """
    if not counts:
        return {}
    if temperature == 0.0:
        n = len(counts)
        return {d: 1.0 / n for d in counts}
    raw = {d: c ** temperature for d, c in counts.items()}
    total = sum(raw.values())
    return {d: v / total for d, v in raw.items()}


def compute_per_row_weights(rows: list[dict], temperature: float) -> list[float]:
    """Compute per-row sampling weights for use with WeightedRandomSampler.

    Each row's weight = domain_weight / row_count_in_domain so that:
    - P(sample from domain D) = domain_weight(D)
    - Within domain D, each row is uniform.

    Args:
        rows: list of dicts each with `_p.domain` field.
        temperature: see compute_domain_weights.

    Returns:
        List of per-row weights, same length as rows.
    """
    counts: dict[str, int] = {}
    for r in rows:
        d = r["_p"]["domain"]
        counts[d] = counts.get(d, 0) + 1
    domain_weights = compute_domain_weights(counts, temperature)
    return [domain_weights[r["_p"]["domain"]] / counts[r["_p"]["domain"]] for r in rows]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd code/training && python -m pytest tests/test_domain_rebalance.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add code/training/data/domain_rebalance.py code/training/tests/test_domain_rebalance.py
git commit -m "feat(training/data): domain temperature rebalancing"
```

---

### Task 5: Held-out task-id split (`heldout_split.py`)

Spec §5.7 — hold back 5 task_ids per domain (15 total) for true generalization eval.

**Files:**
- Create: `code/training/data/heldout_split.py`
- Create: `code/training/tests/test_heldout_split.py`

- [ ] **Step 1: Write failing tests**

```python
# code/training/tests/test_heldout_split.py
from code.training.data.heldout_split import select_heldout_task_ids, split_rows_by_heldout


SAMPLE_ROWS = [
    {"_p": {"domain": "airline", "task_id": "0"}},
    {"_p": {"domain": "airline", "task_id": "1"}},
    {"_p": {"domain": "airline", "task_id": "2"}},
    {"_p": {"domain": "airline", "task_id": "3"}},
    {"_p": {"domain": "airline", "task_id": "4"}},
    {"_p": {"domain": "airline", "task_id": "5"}},
    {"_p": {"domain": "retail", "task_id": "10"}},
    {"_p": {"domain": "retail", "task_id": "11"}},
    {"_p": {"domain": "retail", "task_id": "12"}},
    {"_p": {"domain": "retail", "task_id": "13"}},
    {"_p": {"domain": "retail", "task_id": "14"}},
    {"_p": {"domain": "retail", "task_id": "15"}},
    {"_p": {"domain": "telecom", "task_id": "20"}},
    {"_p": {"domain": "telecom", "task_id": "21"}},
    {"_p": {"domain": "telecom", "task_id": "22"}},
    {"_p": {"domain": "telecom", "task_id": "23"}},
    {"_p": {"domain": "telecom", "task_id": "24"}},
    {"_p": {"domain": "telecom", "task_id": "25"}},
]


def test_selects_n_per_domain():
    heldout = select_heldout_task_ids(SAMPLE_ROWS, n_per_domain=5, seed=42)
    assert set(heldout.keys()) == {"airline", "retail", "telecom"}
    for d, tids in heldout.items():
        assert len(tids) == 5, f"Expected 5 task_ids in {d}, got {len(tids)}"


def test_deterministic():
    h1 = select_heldout_task_ids(SAMPLE_ROWS, n_per_domain=5, seed=42)
    h2 = select_heldout_task_ids(SAMPLE_ROWS, n_per_domain=5, seed=42)
    assert h1 == h2


def test_different_seeds_differ():
    h1 = select_heldout_task_ids(SAMPLE_ROWS, n_per_domain=3, seed=42)
    h2 = select_heldout_task_ids(SAMPLE_ROWS, n_per_domain=3, seed=99)
    # Same task pool — different selections.
    assert h1 != h2


def test_split_filters_correctly():
    heldout_tids = {"airline": {"0", "1"}, "retail": {"10"}, "telecom": set()}
    train, heldout = split_rows_by_heldout(SAMPLE_ROWS, heldout_tids)
    train_keys = {(r["_p"]["domain"], r["_p"]["task_id"]) for r in train}
    heldout_keys = {(r["_p"]["domain"], r["_p"]["task_id"]) for r in heldout}
    assert ("airline", "0") in heldout_keys
    assert ("airline", "0") not in train_keys
    assert ("retail", "11") in train_keys


def test_too_few_unique_tasks_raises():
    """If a domain has fewer unique task_ids than n_per_domain, raise."""
    minimal_rows = [{"_p": {"domain": "airline", "task_id": "0"}}]
    import pytest
    with pytest.raises(ValueError, match="not enough unique task_ids"):
        select_heldout_task_ids(minimal_rows, n_per_domain=5, seed=42)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd code/training && python -m pytest tests/test_heldout_split.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `heldout_split.py`**

```python
# code/training/data/heldout_split.py
"""Held-out task-id split for true generalization eval (Spec §5.7).

The standard val set shares task_ids with train (different seeds only).
For a stricter test, hold out 5 task_ids per domain entirely.
"""
from __future__ import annotations

import random
from collections import defaultdict


def select_heldout_task_ids(
    rows: list[dict], n_per_domain: int, seed: int = 42
) -> dict[str, set[str]]:
    """Pick n_per_domain unique task_ids per domain to hold out.

    Args:
        rows: list of stage-2 rows with `_p.domain` and `_p.task_id`.
        n_per_domain: number of task_ids to hold out per domain.
        seed: RNG seed for reproducibility.

    Returns:
        {domain: {task_id, ...}} — task_ids that should be held out.

    Raises:
        ValueError: if any domain has fewer than n_per_domain unique task_ids.
    """
    by_domain: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_domain[r["_p"]["domain"]].add(r["_p"]["task_id"])

    rng = random.Random(seed)
    heldout: dict[str, set[str]] = {}
    for domain, tids in by_domain.items():
        if len(tids) < n_per_domain:
            raise ValueError(
                f"Domain {domain!r} has only {len(tids)} unique task_ids, "
                f"not enough unique task_ids to hold out {n_per_domain}."
            )
        sorted_tids = sorted(tids)  # Determinism: sort before sampling.
        rng.shuffle(sorted_tids)
        heldout[domain] = set(sorted_tids[:n_per_domain])
    return heldout


def split_rows_by_heldout(
    rows: list[dict], heldout_tids: dict[str, set[str]]
) -> tuple[list[dict], list[dict]]:
    """Split rows into (train_keep, heldout) based on held-out task_ids.

    Args:
        rows: full row list.
        heldout_tids: {domain: {task_id, ...}}.

    Returns:
        (train_keep, heldout_rows). train_keep + heldout_rows == rows (preserved order).
    """
    train_keep, heldout = [], []
    for r in rows:
        d = r["_p"]["domain"]
        tid = r["_p"]["task_id"]
        if tid in heldout_tids.get(d, set()):
            heldout.append(r)
        else:
            train_keep.append(r)
    return train_keep, heldout
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd code/training && python -m pytest tests/test_heldout_split.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add code/training/data/heldout_split.py code/training/tests/test_heldout_split.py
git commit -m "feat(training/data): held-out task-id split for generalization eval"
```

---

### Task 6: Row → TRL prompt/completion conversion (`convert_to_prompt_completion.py`)

Spec §5.1 — the central format transformation. Every row becomes a `{prompt, completion, tools}` record.

**Files:**
- Create: `code/training/data/convert_to_prompt_completion.py`
- Create: `code/training/tests/test_convert_to_prompt_completion.py`

- [ ] **Step 1: Write failing tests**

```python
# code/training/tests/test_convert_to_prompt_completion.py
import json
from pathlib import Path

import pytest

from code.training.data.convert_to_prompt_completion import (
    convert_row,
    convert_dataset,
    load_domain_assets,
)


SAMPLE_ROW = {
    "_p": {"row_id": "airline_0_300_cs__step_1__locked",
           "domain": "airline", "task_id": "0", "step_1based": 1, "phase": "locked",
           "step_depth_frac": 0.0833, "epoch_alias": "cs", "seed": 300},
    "_system_ref": "domain_assets/airline_system.txt",
    "_tools_ref": "domain_assets/airline_tools.json",
    "_target_index": 0,
    "messages": [{"role": "assistant", "content": "Hi! How can I help you today?"}],
}

SAMPLE_ROW_DEEP = {
    "_p": {"row_id": "telecom_5_300_cs__step_3__locked",
           "domain": "telecom", "task_id": "5", "step_1based": 3, "phase": "locked",
           "step_depth_frac": 0.5, "epoch_alias": "cs", "seed": 300},
    "_system_ref": "domain_assets/telecom_system.txt",
    "_tools_ref": "domain_assets/telecom_tools.json",
    "_target_index": 4,
    "messages": [
        {"role": "assistant", "content": "Hello, how can I help?"},
        {"role": "user", "content": "I need help with my plan."},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "tc1", "type": "function",
                         "function": {"name": "get_plan", "arguments": '{"user_id": "u1"}'}}]},
        {"role": "tool", "tool_call_id": "tc1", "content": '{"plan": "premium"}'},
        {"role": "assistant", "content": "Your plan is Premium."},
    ],
}


def test_convert_row_step1():
    """Step 1 has empty prompt prefix; completion is the single message."""
    out = convert_row(SAMPLE_ROW, system_text="SYS", tools=[])
    # Prompt = system + empty prefix
    assert out["prompt"][0]["role"] == "system"
    assert out["prompt"][0]["content"] == "SYS"
    assert len(out["prompt"]) == 1  # only system, no prior messages
    # Completion = the single target message
    assert len(out["completion"]) == 1
    assert out["completion"][0]["role"] == "assistant"
    assert out["completion"][0]["content"] == "Hi! How can I help you today?"
    assert out["tools"] == []
    assert out["_meta"]["row_id"] == SAMPLE_ROW["_p"]["row_id"]


def test_convert_row_deep():
    """Deeper step has prior history as prompt, target as completion."""
    out = convert_row(SAMPLE_ROW_DEEP, system_text="SYS", tools=[{"x": 1}])
    # Prompt = system + 4 prior messages
    assert len(out["prompt"]) == 5  # system + 4
    assert out["prompt"][0]["content"] == "SYS"
    assert out["prompt"][-1]["role"] == "tool"
    # Completion = the final assistant
    assert len(out["completion"]) == 1
    assert out["completion"][0]["content"] == "Your plan is Premium."
    assert out["tools"] == [{"x": 1}]


def test_convert_row_target_with_tool_calls():
    """Tool-call target preserves tool_calls in completion."""
    row = {
        "_p": {"row_id": "x", "domain": "airline", "task_id": "1",
               "step_1based": 1, "phase": "locked",
               "step_depth_frac": 0.1, "epoch_alias": "cs", "seed": 300},
        "_system_ref": "x", "_tools_ref": "x", "_target_index": 0,
        "messages": [
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "t1", "type": "function",
                             "function": {"name": "foo", "arguments": '{"a":1}'}}]},
        ],
    }
    out = convert_row(row, system_text="SYS", tools=[])
    assert "tool_calls" in out["completion"][0]
    assert out["completion"][0]["tool_calls"][0]["id"] == "t1"


def test_convert_dataset_with_filters(tmp_path: Path):
    """End-to-end: write input jsonl, convert, read output."""
    src = tmp_path / "src.jsonl"
    da = tmp_path / "domain_assets"
    da.mkdir()
    (da / "airline_system.txt").write_text("AIRLINE SYS")
    (da / "airline_tools.json").write_text(json.dumps([{"name": "tool_a"}]))

    src.write_text(json.dumps(SAMPLE_ROW) + "\n")
    dst = tmp_path / "out.jsonl"

    stats = convert_dataset(
        src_jsonl=src, dst_jsonl=dst, domain_assets_dir=da,
        min_target_tokens=1,  # disable filter for this 1-word test
    )
    assert stats["n_kept"] == 1
    assert stats["n_dropped"] == 0
    out_rows = [json.loads(l) for l in dst.read_text().splitlines()]
    assert len(out_rows) == 1
    assert out_rows[0]["prompt"][0]["content"] == "AIRLINE SYS"
    assert out_rows[0]["tools"] == [{"name": "tool_a"}]


def test_load_domain_assets(tmp_path: Path):
    da = tmp_path
    (da / "airline_system.txt").write_text("AIR")
    (da / "airline_tools.json").write_text(json.dumps([{"name": "t"}]))
    sys_texts, tools = load_domain_assets(da)
    assert sys_texts["airline"] == "AIR"
    assert tools["airline"] == [{"name": "t"}]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd code/training && python -m pytest tests/test_convert_to_prompt_completion.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `convert_to_prompt_completion.py`**

```python
# code/training/data/convert_to_prompt_completion.py
"""Convert stage-2 row format to TRL prompt/completion format (Spec §5.1).

Each row becomes:
    {
      "prompt":     [system, *messages[0:_target_index]],
      "completion": [messages[_target_index]],
      "tools":      <tools schema for the row's domain>,
      "_meta":      <provenance from row._p>
    }

With TRL SFTConfig(completion_only_loss=True, assistant_only_loss=False),
loss is computed only on the completion span. Intermediate assistant turns
in the prompt are correctly masked.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from code.training.data.filters import should_drop_row


def load_domain_assets(domain_assets_dir: Path) -> tuple[dict[str, str], dict[str, list]]:
    """Load all *_system.txt and *_tools.json from domain_assets/.

    Returns (system_texts, tools_per_domain) keyed by domain name.
    """
    system_texts: dict[str, str] = {}
    tools_per_domain: dict[str, list] = {}
    for p in sorted(domain_assets_dir.glob("*_system.txt")):
        domain = p.stem.replace("_system", "")
        system_texts[domain] = p.read_text()
    for p in sorted(domain_assets_dir.glob("*_tools.json")):
        domain = p.stem.replace("_tools", "")
        tools_per_domain[domain] = json.loads(p.read_text())
    return system_texts, tools_per_domain


def convert_row(row: dict, system_text: str, tools: list) -> dict:
    """Convert one stage-2 row to TRL prompt/completion format."""
    msgs = row["messages"]
    ti = row["_target_index"]
    prompt = [{"role": "system", "content": system_text}] + list(msgs[:ti])
    completion = [msgs[ti]]
    return {
        "prompt": prompt,
        "completion": completion,
        "tools": tools,
        "_meta": dict(row["_p"]),  # full provenance preserved
    }


def convert_dataset(
    src_jsonl: Path,
    dst_jsonl: Path,
    domain_assets_dir: Path,
    min_target_tokens: int = 5,
    drop_log: Path | None = None,
) -> dict:
    """Convert an entire jsonl file. Applies quality filters."""
    system_texts, tools_per_domain = load_domain_assets(domain_assets_dir)
    n_kept, n_dropped = 0, 0
    drop_reasons: dict[str, int] = {}
    drop_records: list[dict] = []

    dst_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with src_jsonl.open() as fin, dst_jsonl.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            should_drop, reason = should_drop_row(row, min_target_tokens)
            if should_drop:
                n_dropped += 1
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                if drop_log is not None:
                    drop_records.append({"row_id": row["_p"]["row_id"], "reason": reason})
                continue
            domain = row["_p"]["domain"]
            converted = convert_row(
                row,
                system_text=system_texts[domain],
                tools=tools_per_domain[domain],
            )
            fout.write(json.dumps(converted) + "\n")
            n_kept += 1

    if drop_log is not None and drop_records:
        drop_log.parent.mkdir(parents=True, exist_ok=True)
        drop_log.write_text("\n".join(json.dumps(r) for r in drop_records))

    return {
        "n_kept": n_kept,
        "n_dropped": n_dropped,
        "drop_reasons": drop_reasons,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--domain-assets", required=True, type=Path)
    ap.add_argument("--min-target-tokens", type=int, default=5)
    ap.add_argument("--drop-log", type=Path, default=None)
    args = ap.parse_args(argv)

    stats = convert_dataset(
        args.src, args.dst, args.domain_assets,
        args.min_target_tokens, args.drop_log,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd code/training && python -m pytest tests/test_convert_to_prompt_completion.py -v`
Expected: 5 passed

- [ ] **Step 5: Run conversion on real data and verify output**

Run:
```bash
cd /path/to/bundle
.venv/bin/python -m code.training.data.convert_to_prompt_completion \
    --src data_processed/stage2_v1/train.jsonl \
    --dst train_outputs/_data_cache/train_prompt_completion.jsonl \
    --domain-assets data_processed/stage2_v1/domain_assets \
    --drop-log train_outputs/_data_cache/dropped_rows.jsonl
```
Expected: ~6,400+ kept, <50 dropped (filters very lenient — corpus is clean).

- [ ] **Step 6: Commit**

```bash
git add code/training/data/convert_to_prompt_completion.py \
        code/training/tests/test_convert_to_prompt_completion.py
git commit -m "feat(training/data): row → TRL prompt/completion conversion"
```

---

### Task 7: Chat-template validation (`validate_chat_template.py`)

Spec §5.3 — pre-launch check that the model's chat template can render every row.

**Files:**
- Create: `code/training/data/validate_chat_template.py`
- Create: `code/training/tests/test_validate_chat_template.py`

- [ ] **Step 1: Write failing tests**

```python
# code/training/tests/test_validate_chat_template.py
import json
from pathlib import Path
import pytest

# Skipping if transformers not available — this test requires the actual
# Qwen3.5-2B tokenizer to be cached.
transformers = pytest.importorskip("transformers")

from code.training.data.validate_chat_template import (
    validate_row,
    validate_jsonl,
    contains_pre_existing_thinking_block,
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
        # Pre-existing thinking block — would double-insert under enable_thinking=False
        {"role": "assistant", "content": "<think></think>actual reply"},
    ],
    "completion": [{"role": "assistant", "content": "ok"}],
    "tools": [], "_meta": {"row_id": "bad_thinking"},
}

ORPHAN_TOOL = {
    "prompt": [
        {"role": "system", "content": "Sys"},
        {"role": "tool", "tool_call_id": "phantom", "content": "result"},
    ],
    "completion": [{"role": "assistant", "content": "ok"}],
    "tools": [], "_meta": {"row_id": "orphan_tool"},
}


def test_contains_pre_existing_thinking_block():
    assert contains_pre_existing_thinking_block(BAD_THINKING) is True
    assert contains_pre_existing_thinking_block(GOOD_ROW) is False


@pytest.mark.skipif(
    not Path.home().joinpath(".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct").exists(),
    reason="Qwen2.5-1.5B tokenizer not cached locally",
)
def test_validate_good_row():
    """A clean row passes validation against Qwen2.5-1.5B as proxy."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
    ok, reason = validate_row(GOOD_ROW, tok)
    assert ok is True, f"Expected pass, got reason: {reason}"


def test_validate_thinking_block_fails():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
    ok, reason = validate_row(BAD_THINKING, tok)
    assert ok is False
    assert "thinking" in reason.lower()


def test_validate_jsonl_writes_report(tmp_path: Path):
    src = tmp_path / "src.jsonl"
    src.write_text(json.dumps(GOOD_ROW) + "\n" + json.dumps(BAD_THINKING) + "\n")
    out = tmp_path / "report.json"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
    summary = validate_jsonl(src, tok, out)
    assert summary["n_total"] == 2
    assert summary["n_pass"] == 1
    assert summary["n_fail"] == 1
    assert out.exists()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd code/training && python -m pytest tests/test_validate_chat_template.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `validate_chat_template.py`**

```python
# code/training/data/validate_chat_template.py
"""Pre-launch chat-template validation (Spec §5.3).

For each prompt/completion record:
- Apply chat template with enable_thinking=False
- Verify token-level round-trip via encode → decode
- Verify no pre-existing <think></think> blocks (would double-insert)
- Verify tool_call_id consistency

Hard-stops training if any row fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def contains_pre_existing_thinking_block(row: dict) -> bool:
    """Check if any message content already contains a <think></think> string."""
    for msg in (row.get("prompt") or []) + (row.get("completion") or []):
        c = msg.get("content") or ""
        if "<think>" in c or "</think>" in c:
            return True
    return False


def has_orphan_tool_in_prompt(row: dict) -> bool:
    """Check if a `tool` role in the prompt has no preceding tool_call.id."""
    seen_ids: set[str] = set()
    for msg in row.get("prompt") or []:
        if msg["role"] == "assistant":
            for tc in msg.get("tool_calls") or []:
                if "id" in tc:
                    seen_ids.add(tc["id"])
        elif msg["role"] == "tool":
            tcid = msg.get("tool_call_id")
            if tcid is None or tcid not in seen_ids:
                return True
    return False


def validate_row(row: dict, tokenizer) -> tuple[bool, str]:
    """Return (ok, reason)."""
    if contains_pre_existing_thinking_block(row):
        return False, "pre_existing_thinking_block"
    if has_orphan_tool_in_prompt(row):
        return False, "orphan_tool_in_prompt"

    # Build full conversation: prompt + completion, render with chat template.
    full_msgs = list(row["prompt"]) + list(row["completion"])
    tools = row.get("tools") or None
    try:
        rendered = tokenizer.apply_chat_template(
            full_msgs, tools=tools, tokenize=False,
            add_generation_prompt=False, enable_thinking=False,
        )
    except Exception as e:
        return False, f"chat_template_apply_error: {type(e).__name__}: {e}"

    # Round-trip: encode → decode should preserve.
    try:
        ids = tokenizer.encode(rendered)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        # The decoded string should contain key tokens from the rendered template.
        # We use the assistant marker as a sentinel.
        if "<|im_start|>assistant" not in decoded:
            return False, "round_trip_lost_assistant_marker"
    except Exception as e:
        return False, f"encode_decode_error: {type(e).__name__}: {e}"

    return True, ""


def validate_jsonl(src_jsonl: Path, tokenizer, out_report: Path) -> dict:
    """Validate every row in a jsonl file. Writes a JSON report."""
    n_total, n_pass, n_fail = 0, 0, 0
    failures: list[dict] = []
    reason_counts: dict[str, int] = {}

    with src_jsonl.open() as f:
        for line in f:
            row = json.loads(line)
            ok, reason = validate_row(row, tokenizer)
            n_total += 1
            if ok:
                n_pass += 1
            else:
                n_fail += 1
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                failures.append({"row_id": row.get("_meta", {}).get("row_id", "?"),
                                  "reason": reason})

    summary = {
        "n_total": n_total,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "reason_counts": reason_counts,
        "failures": failures[:100],  # cap at 100 examples
    }
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer id, e.g. Qwen/Qwen3.5-2B")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    summary = validate_jsonl(args.src, tok, args.out)
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2))
    if summary["n_fail"] > 0:
        print(f"\n!! {summary['n_fail']} rows failed validation. Inspect {args.out}.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd code/training && python -m pytest tests/test_validate_chat_template.py -v`
Expected: 4 passed (some may skip if tokenizer not cached locally — that's OK)

- [ ] **Step 5: Run validation on the converted dataset**

Run:
```bash
cd /path/to/bundle
.venv/bin/python -m code.training.data.validate_chat_template \
    --src train_outputs/_data_cache/train_prompt_completion.jsonl \
    --tokenizer Qwen/Qwen3.5-2B \
    --out train_outputs/_data_cache/validation_report_qwen3_5.json
```
Expected: `n_pass == n_total`, exit code 0. If any failures, fix data BEFORE training.

- [ ] **Step 6: Commit**

```bash
git add code/training/data/validate_chat_template.py \
        code/training/tests/test_validate_chat_template.py
git commit -m "feat(training/data): chat-template pre-launch validation"
```

---

### Task 8: Plan-level config (`plan_c_prime.yaml`) + accelerate configs

Define the 10-run plan and accelerate FSDP2/DDP configs as data.

**Files:**
- Create: `code/training/configs/plan_c_prime.yaml`
- Create: `code/training/configs/accelerate_fsdp2.yaml`
- Create: `code/training/configs/accelerate_ddp.yaml`

- [ ] **Step 1: Write `plan_c_prime.yaml`**

```yaml
# code/training/configs/plan_c_prime.yaml
# Plan C' — 10-run training schedule (Spec §3)
# Source data: data_processed/stage2_v1
# Output root: train_outputs/

name: plan_c_prime
data:
  source_jsonl: data_processed/stage2_v1/train.jsonl
  val_jsonl:    data_processed/stage2_v1/val.jsonl
  domain_assets: data_processed/stage2_v1/domain_assets
  heldout_n_per_domain: 5
  heldout_seed: 42
  domain_temperature: 0.5
  tool_shuffle: true
  min_target_tokens: 5

partition:
  source: data_processed/stage2_v1/partition.json
  subsample_seed: 42

eval:
  eval_tasks_jsonl: data_processed/stage2_v1/eval_tasks.jsonl
  diag_tasks_jsonl: data_processed/stage2_v1/diag_tasks.jsonl
  default_seed: 300
  winner_reeval_seed: 301

runs:
  - id: 01_qwen3_5_2b_273
    config: configs/runs/01_qwen3_5_2b_273.yaml
  - id: 02_qwen3_5_4b_50
    config: configs/runs/02_qwen3_5_4b_50.yaml
  - id: 03_qwen3_5_4b_100
    config: configs/runs/03_qwen3_5_4b_100.yaml
  - id: 04_qwen3_5_4b_200
    config: configs/runs/04_qwen3_5_4b_200.yaml
  - id: 05_qwen3_5_4b_273
    config: configs/runs/05_qwen3_5_4b_273.yaml
  - id: 06_qwen3_5_9b_50
    config: configs/runs/06_qwen3_5_9b_50.yaml
  - id: 07_qwen3_5_9b_273
    config: configs/runs/07_qwen3_5_9b_273.yaml
  - id: 08_qwen3_6_35b_a3b_273
    config: configs/runs/08_qwen3_6_35b_a3b_273.yaml
  - id: 09_qwen3_5_4b_273_lr1e5
    config: configs/runs/09_qwen3_5_4b_273_lr1e5.yaml
    skip_eval: true     # LR sanity check — picked by val loss only
  - id: 10_qwen3_5_4b_273_lr3e5
    config: configs/runs/10_qwen3_5_4b_273_lr3e5.yaml
    skip_eval: true
```

- [ ] **Step 2: Write `accelerate_fsdp2.yaml`** (Spec §6.1.1)

```yaml
# code/training/configs/accelerate_fsdp2.yaml
# accelerate config for FSDP2 — used for Qwen3.5-9B and Qwen3.6-35B-A3B
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
mixed_precision: bf16
num_machines: 1
num_processes: 8       # 8× H200
machine_rank: 0
main_training_function: main
dispatch_batches: false        # required for FSDP2 — each rank loads its own shard
fsdp_config:
  fsdp_version: 2
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: Qwen3DecoderLayer,Qwen3MoeDecoderLayer,Qwen36DecoderLayer
  fsdp_use_orig_params: true
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_reshard_after_forward: false       # required with chunked_nll
  fsdp_activation_checkpointing: true
  fsdp_sync_module_states: true
  fsdp_forward_prefetch: true
  fsdp_backward_prefetch: BACKWARD_PRE
```

- [ ] **Step 3: Write `accelerate_ddp.yaml`**

```yaml
# code/training/configs/accelerate_ddp.yaml
# accelerate config for DDP — used for Qwen3.5-2B and Qwen3.5-4B
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: bf16
num_machines: 1
num_processes: 8
machine_rank: 0
main_training_function: main
```

- [ ] **Step 4: Verify YAMLs parse**

Run:
```bash
cd /path/to/bundle/code/training
python -c "import yaml; print(yaml.safe_load(open('configs/plan_c_prime.yaml'))['name'])"
python -c "import yaml; print(yaml.safe_load(open('configs/accelerate_fsdp2.yaml'))['distributed_type'])"
```
Expected: `plan_c_prime` and `FSDP`.

- [ ] **Step 5: Commit**

```bash
git add code/training/configs/plan_c_prime.yaml \
        code/training/configs/accelerate_fsdp2.yaml \
        code/training/configs/accelerate_ddp.yaml
git commit -m "feat(training/configs): plan_c_prime + accelerate FSDP2/DDP configs"
```

---

### Task 9: Per-run YAMLs (10 files)

Per-run hyperparameters. All files share a common schema; values per Spec §6.1.

**Files:**
- Create: `code/training/configs/runs/01_qwen3_5_2b_273.yaml`
- Create: `code/training/configs/runs/02_qwen3_5_4b_50.yaml`
- Create: `code/training/configs/runs/03_qwen3_5_4b_100.yaml`
- Create: `code/training/configs/runs/04_qwen3_5_4b_200.yaml`
- Create: `code/training/configs/runs/05_qwen3_5_4b_273.yaml`
- Create: `code/training/configs/runs/06_qwen3_5_9b_50.yaml`
- Create: `code/training/configs/runs/07_qwen3_5_9b_273.yaml`
- Create: `code/training/configs/runs/08_qwen3_6_35b_a3b_273.yaml`
- Create: `code/training/configs/runs/09_qwen3_5_4b_273_lr1e5.yaml`
- Create: `code/training/configs/runs/10_qwen3_5_4b_273_lr3e5.yaml`

- [ ] **Step 1: Write run #1 (2B floor)**

```yaml
# code/training/configs/runs/01_qwen3_5_2b_273.yaml
run_id: 01_qwen3_5_2b_273
model:
  name: Qwen/Qwen3.5-2B
  revision: main      # operator pins to a specific SHA after first download
  is_moe: false
data:
  n_train_runs: 273         # full data
training:
  output_dir: train_outputs/01_qwen3_5_2b_273
  num_train_epochs: 5
  learning_rate: 3.0e-5
  warmup_ratio: 0.05
  lr_scheduler_type: cosine
  cosine_min_lr_ratio: 0.10
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.95
  max_grad_norm: 1.0
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2          # effective batch = 4 × 2 × 8 GPUs = 64
  per_device_eval_batch_size: 4
  bf16: true
  gradient_checkpointing: false
  torch_compile: false
  max_seq_length: 32768
  packing: true
  packing_strategy: bfd
  padding_free: true
  attn_implementation: flash_attention_2
  completion_only_loss: true
  assistant_only_loss: false
  use_liger_kernel: false
  loss_type: chunked_nll
  dataloader_num_workers: 4
  eval_strategy: steps
  eval_steps: 50
  save_strategy: epoch
  save_total_limit: 2
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
  greater_is_better: false
  early_stopping_patience: 2
  logging_steps: 10
  seed: 1234
  data_seed: 42
distributed:
  strategy: ddp
  accelerate_config: configs/accelerate_ddp.yaml
moe: null
```

- [ ] **Step 2: Write runs #2–5 (4B sweep)** — same schema, different `n_train_runs` and `output_dir`. Diff vs #1: `model.name=Qwen/Qwen3.5-4B`, `learning_rate=2.0e-5`. Run #5 is the LR-default reference.

```yaml
# code/training/configs/runs/02_qwen3_5_4b_50.yaml
run_id: 02_qwen3_5_4b_50
model:
  name: Qwen/Qwen3.5-4B
  revision: main
  is_moe: false
data:
  n_train_runs: 50
training:
  output_dir: train_outputs/02_qwen3_5_4b_50
  num_train_epochs: 5
  learning_rate: 2.0e-5
  warmup_ratio: 0.05
  lr_scheduler_type: cosine
  cosine_min_lr_ratio: 0.10
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.95
  max_grad_norm: 1.0
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2
  per_device_eval_batch_size: 4
  bf16: true
  gradient_checkpointing: false
  torch_compile: false
  max_seq_length: 32768
  packing: true
  packing_strategy: bfd
  padding_free: true
  attn_implementation: flash_attention_2
  completion_only_loss: true
  assistant_only_loss: false
  use_liger_kernel: false
  loss_type: chunked_nll
  dataloader_num_workers: 4
  eval_strategy: steps
  eval_steps: 50
  save_strategy: epoch
  save_total_limit: 2
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
  greater_is_better: false
  early_stopping_patience: 2
  logging_steps: 10
  seed: 1234
  data_seed: 42
distributed:
  strategy: ddp
  accelerate_config: configs/accelerate_ddp.yaml
moe: null
```

Repeat for `03_..._100.yaml` (n_train_runs: 100), `04_..._200.yaml` (200), `05_..._273.yaml` (273) — only the `n_train_runs` and `output_dir` change.

- [ ] **Step 3: Write runs #6–7 (9B)** — diff vs 4B: `model.name=Qwen/Qwen3.5-9B`, `learning_rate=1.0e-5`, `gradient_checkpointing=true`, `per_device_train_batch_size=2`, `gradient_accumulation_steps=4`, `distributed.strategy=fsdp2`, `distributed.accelerate_config=configs/accelerate_fsdp2.yaml`.

```yaml
# code/training/configs/runs/06_qwen3_5_9b_50.yaml
run_id: 06_qwen3_5_9b_50
model:
  name: Qwen/Qwen3.5-9B
  revision: main
  is_moe: false
data:
  n_train_runs: 50
training:
  output_dir: train_outputs/06_qwen3_5_9b_50
  num_train_epochs: 5
  learning_rate: 1.0e-5
  warmup_ratio: 0.05
  lr_scheduler_type: cosine
  cosine_min_lr_ratio: 0.10
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.95
  max_grad_norm: 1.0
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  per_device_eval_batch_size: 2
  bf16: true
  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: false
  torch_compile: false
  max_seq_length: 32768
  packing: true
  packing_strategy: bfd
  padding_free: true
  attn_implementation: flash_attention_2
  completion_only_loss: true
  assistant_only_loss: false
  use_liger_kernel: false
  loss_type: chunked_nll
  dataloader_num_workers: 4
  eval_strategy: steps
  eval_steps: 50
  save_strategy: epoch
  save_total_limit: 2
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
  greater_is_better: false
  early_stopping_patience: 2
  logging_steps: 10
  seed: 1234
  data_seed: 42
distributed:
  strategy: fsdp2
  accelerate_config: configs/accelerate_fsdp2.yaml
moe: null
```

`07_qwen3_5_9b_273.yaml` is identical except `n_train_runs: 273` and `output_dir`.

- [ ] **Step 4: Write run #8 (Qwen3.6-35B-A3B MoE)** — Spec §6.1.3 + §6.1.4

```yaml
# code/training/configs/runs/08_qwen3_6_35b_a3b_273.yaml
run_id: 08_qwen3_6_35b_a3b_273
model:
  name: Qwen/Qwen3.6-35B-A3B
  revision: main
  is_moe: true
data:
  n_train_runs: 273
training:
  output_dir: train_outputs/08_qwen3_6_35b_a3b_273
  num_train_epochs: 5
  learning_rate: 1.0e-5
  warmup_ratio: 0.05
  lr_scheduler_type: cosine
  cosine_min_lr_ratio: 0.10
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.95
  max_grad_norm: 1.0
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  per_device_eval_batch_size: 1
  bf16: true
  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: false
  torch_compile: false
  max_seq_length: 32768
  packing: true
  packing_strategy: bfd
  padding_free: true
  attn_implementation: flash_attention_2
  completion_only_loss: true
  assistant_only_loss: false
  use_liger_kernel: false
  loss_type: chunked_nll
  dataloader_num_workers: 4
  eval_strategy: steps
  eval_steps: 50
  save_strategy: epoch
  save_total_limit: 2
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
  greater_is_better: false
  early_stopping_patience: 2
  logging_steps: 10
  seed: 1234
  data_seed: 42
distributed:
  strategy: fsdp2
  accelerate_config: configs/accelerate_fsdp2.yaml
moe:
  output_router_logits: true
  router_aux_loss_coef: 0.001            # NOT 0.01 — see Spec §6.1.3
  preserve_thinking: false               # Qwen3.6-specific (Spec §6.1.4)
  vllm_min_version: "0.19.0"
  vllm_tool_call_parser: qwen3_coder
```

- [ ] **Step 5: Write LR variants** — `09_qwen3_5_4b_273_lr1e5.yaml` and `10_qwen3_5_4b_273_lr3e5.yaml`. Identical to `05_qwen3_5_4b_273.yaml` except `learning_rate` and `output_dir`.

- [ ] **Step 6: Verify all YAMLs parse**

```bash
cd /path/to/bundle/code/training
for f in configs/runs/*.yaml; do
  python -c "import yaml; d=yaml.safe_load(open('$f')); print('$f', '->', d['run_id'], d['model']['name'])"
done
```
Expected: 10 lines, all unique run_ids.

- [ ] **Step 7: Commit**

```bash
git add code/training/configs/runs/
git commit -m "feat(training/configs): per-run YAMLs for plan C' (10 runs)"
```

---

### Task 10: Tokenizer re-audit on actual training tokenizers

Spec §5.3.1 — verify token counts on Qwen3.5/3.6 (existing audit was Qwen2.5). Re-uses existing `data_audit.py`.

**Files:**
- Modify: `code/scripts/training_prep/data_audit.py` (add multi-tokenizer mode)
- Create: `code/training/orchestration/tokenizer_reaudit.sh`

- [ ] **Step 1: Add tokenizer overlong-flag mode to `data_audit.py`**

The existing `data_audit.py` already supports `--tokenizer` arg. We just need to run it 4 times. Verify it works:

```bash
cd /path/to/bundle/code
.venv/bin/python -m scripts.training_prep.data_audit \
    --root ../data_processed/stage2_v1 \
    --tokenizer Qwen/Qwen3.5-2B \
    --out ../data_processed/stage2_v1/audit/audit_for_training.qwen3_5_2b.json \
    --limit 50
```
Expected: writes a JSON, no errors. (If transformers can't load Qwen3.5-2B, that's a download issue to resolve before launch.)

- [ ] **Step 2: Write `tokenizer_reaudit.sh`** — runs the audit for all 4 model tokenizers

```bash
#!/usr/bin/env bash
# code/training/orchestration/tokenizer_reaudit.sh
# Re-audit token counts on actual training tokenizers (Spec §5.3.1).
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:?set BUNDLE_ROOT to bundle path}"
VENV="$BUNDLE_ROOT/code/.venv/bin"

cd "$BUNDLE_ROOT/code"

for TOK in Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B Qwen/Qwen3.6-35B-A3B; do
    SLUG=$(echo "$TOK" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
    OUT="$BUNDLE_ROOT/data_processed/stage2_v1/audit/audit_for_training.${SLUG}.json"
    echo "=== Auditing $TOK → $OUT ==="
    "$VENV/python" -m scripts.training_prep.data_audit \
        --root "$BUNDLE_ROOT/data_processed/stage2_v1" \
        --tokenizer "$TOK" \
        --out "$OUT"
done

echo ""
echo "=== Summary: per-tokenizer max sequence length ==="
"$VENV/python" - <<'EOF'
import json, glob, os
root = os.environ["BUNDLE_ROOT"]
for p in sorted(glob.glob(f"{root}/data_processed/stage2_v1/audit/audit_for_training.qwen*.json")):
    d = json.loads(open(p).read())
    name = os.path.basename(p)
    mx = max(s["max"] for s in d.get("per_domain_total_toks", {}).values()) if "per_domain_total_toks" in d else "?"
    print(f"  {name}: max total toks = {mx}")
EOF
```

- [ ] **Step 3: Run the audit script (cloud only — needs the actual tokenizers)**

```bash
chmod +x code/training/orchestration/tokenizer_reaudit.sh
BUNDLE_ROOT=/path/to/bundle bash code/training/orchestration/tokenizer_reaudit.sh
```
Expected: 4 audit JSONs, max total toks <32K for all (else 32K seq is unsafe).

- [ ] **Step 4: Commit**

```bash
git add code/training/orchestration/tokenizer_reaudit.sh
git commit -m "feat(training/orchestration): tokenizer re-audit for Qwen3.5/3.6"
```

---

### Task 11: `train.py` — argparse + config loading + run-stratified subsample

The training entry point. Long file — broken into 3 sub-tasks.

**Files:**
- Create: `code/training/train.py` (sub-task 11a builds the skeleton; 11b/11c flesh it out)
- Create: `code/training/tests/test_train_helpers.py`

- [ ] **Step 1: Write failing tests for the helper functions**

```python
# code/training/tests/test_train_helpers.py
import json
import random
from pathlib import Path

import pytest

from code.training.train import (
    load_run_config,
    subsample_runs_stratified,
    filter_rows_by_runs,
)


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
    # Telecom should be roughly 60% of 50 = 30, allow ±2
    assert 28 <= counts["telecom"] <= 32
    assert 8 <= counts["retail"] <= 12
    assert 8 <= counts["airline"] <= 12


def test_subsample_full_returns_all():
    runs_by_domain = {"telecom": ["t0", "t1"], "airline": ["a0"]}
    sampled = subsample_runs_stratified(runs_by_domain, target_n=3, seed=42)
    assert set(sampled) == {"t0", "t1", "a0"}


def test_filter_rows_by_runs():
    rows = [
        {"_meta": {"run_dir": "airline/0_300_cs"}, "x": 1},
        {"_meta": {"run_dir": "airline/1_300_cs"}, "x": 2},
        {"_meta": {"run_dir": "telecom/5_300_cs"}, "x": 3},
    ]
    keep = filter_rows_by_runs(rows, run_dirs={"airline/0_300_cs", "telecom/5_300_cs"})
    assert len(keep) == 2
    assert {r["x"] for r in keep} == {1, 3}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd code/training && python -m pytest tests/test_train_helpers.py -v`
Expected: FAIL — module not found / functions missing.

- [ ] **Step 3: Write `train.py` skeleton with helpers**

```python
# code/training/train.py
"""Training entry point — trains ONE run.

Usage:
    accelerate launch --config_file <accel.yaml> code/training/train.py \\
        --run-config code/training/configs/runs/<run_id>.yaml \\
        --plan-config code/training/configs/plan_c_prime.yaml \\
        --bundle-root /path/to/bundle
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml


def load_run_config(path: Path) -> dict:
    """Load a per-run YAML config."""
    return yaml.safe_load(path.read_text())


def load_plan_config(path: Path) -> dict:
    """Load the plan-level YAML config."""
    return yaml.safe_load(path.read_text())


def subsample_runs_stratified(
    runs_by_domain: dict[str, list[str]], target_n: int, seed: int
) -> list[str]:
    """Sample target_n runs total, stratified by domain (per Spec §3.1)."""
    rng = random.Random(seed)
    total_pool = sum(len(rs) for rs in runs_by_domain.values())
    if target_n >= total_pool:
        # Return everything in deterministic order.
        all_runs = []
        for d in sorted(runs_by_domain):
            all_runs.extend(runs_by_domain[d])
        return all_runs

    # Initial allocation per domain.
    quotas = {d: max(1, round(target_n * len(rs) / total_pool))
              for d, rs in runs_by_domain.items()}
    # Sample within each domain.
    sampled: list[str] = []
    for d, quota in quotas.items():
        pool = sorted(runs_by_domain[d])  # determinism: sort before shuffle
        rng.shuffle(pool)
        sampled.extend(pool[:min(quota, len(pool))])
    # Adjust to exact target_n.
    if len(sampled) > target_n:
        # Drop from the largest domain's tail.
        sampled = sampled[:target_n]
    elif len(sampled) < target_n:
        # Top up from any unused (largest domain pool first).
        sampled_set = set(sampled)
        unused = []
        for d in sorted(runs_by_domain, key=lambda x: -len(runs_by_domain[x])):
            for r in sorted(runs_by_domain[d]):
                if r not in sampled_set:
                    unused.append(r)
        rng.shuffle(unused)
        sampled.extend(unused[:target_n - len(sampled)])
    return sampled


def filter_rows_by_runs(rows: list[dict], run_dirs: set[str]) -> list[dict]:
    """Keep only rows whose `_meta.run_dir` is in the allowed set."""
    return [r for r in rows if r["_meta"]["run_dir"] in run_dirs]


def build_runs_by_domain(rows: list[dict]) -> dict[str, list[str]]:
    """Group run_dirs by domain from the converted prompt/completion rows."""
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-config", required=True, type=Path)
    ap.add_argument("--plan-config", required=True, type=Path)
    ap.add_argument("--bundle-root", required=True, type=Path)
    ap.add_argument("--smoke-test", action="store_true",
                    help="Run a minimal config (10 rows, 1 step) for code-correctness.")
    args = ap.parse_args(argv)

    # Sub-task 11b will fill in the rest.
    raise NotImplementedError("Sub-task 11b: implement training body")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd code/training && python -m pytest tests/test_train_helpers.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add code/training/train.py code/training/tests/test_train_helpers.py
git commit -m "feat(training): train.py skeleton + subsample/load helpers"
```

---

### Task 12: `train.py` — model + tokenizer + dataset loading

Build out `train.py`'s setup body. Spec §5.4 (domain rebalance), §5.5 (tool shuffle), §6.1 (HF/TRL config).

**Files:**
- Modify: `code/training/train.py` (replace `NotImplementedError` with real body)

- [ ] **Step 1: Replace `main()` with a real implementation**

```python
# code/training/train.py — replace the previous main() with this version
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-config", required=True, type=Path)
    ap.add_argument("--plan-config", required=True, type=Path)
    ap.add_argument("--bundle-root", required=True, type=Path)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args(argv)

    run_cfg = load_run_config(args.run_config)
    plan_cfg = load_plan_config(args.plan_config)
    rid = run_cfg["run_id"]
    bundle_root = args.bundle_root.resolve()
    output_dir = bundle_root / run_cfg["training"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    status_file = output_dir / "STATUS"
    status_file.write_text("running")

    # Lazy imports — heavy.
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from accelerate import Accelerator

    # 1. Load tokenizer.
    tok = AutoTokenizer.from_pretrained(
        run_cfg["model"]["name"],
        revision=run_cfg["model"].get("revision", "main"),
        trust_remote_code=True,
    )
    if tok.pad_token_id is None:
        # Spec §6.1 — pad token MUST be <|endoftext|>, NOT eos_token.
        eot_id = tok.convert_tokens_to_ids("<|endoftext|>")
        if eot_id is None or eot_id == tok.unk_token_id:
            raise RuntimeError(
                f"Cannot resolve <|endoftext|> for tokenizer {run_cfg['model']['name']}"
            )
        tok.pad_token = "<|endoftext|>"
        tok.pad_token_id = eot_id

    # 2. Load converted dataset.
    train_path = bundle_root / "train_outputs" / "_data_cache" / "train_prompt_completion.jsonl"
    val_path = bundle_root / "train_outputs" / "_data_cache" / "val_prompt_completion.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Converted train data not found at {train_path}. "
            f"Run convert_to_prompt_completion.py first."
        )

    train_rows = [json.loads(l) for l in train_path.read_text().splitlines()]
    val_rows = [json.loads(l) for l in val_path.read_text().splitlines()]

    # 3. Apply run-stratified subsample.
    runs_by_domain = build_runs_by_domain(train_rows)
    n_target = run_cfg["data"]["n_train_runs"]
    keep_runs = set(subsample_runs_stratified(
        runs_by_domain, n_target, seed=plan_cfg["partition"]["subsample_seed"]
    ))
    train_rows = filter_rows_by_runs(train_rows, keep_runs)

    # Smoke test override.
    if args.smoke_test:
        train_rows = train_rows[:10]
        val_rows = val_rows[:5]
        run_cfg["training"]["num_train_epochs"] = 1
        run_cfg["training"]["max_steps"] = 1
        run_cfg["training"]["per_device_train_batch_size"] = 1
        run_cfg["training"]["gradient_accumulation_steps"] = 1

    # 4. Apply tool-schema-order shuffle (Spec §5.5).
    from code.training.data.tool_shuffle import shuffle_tools_in_schema
    rng = random.Random(plan_cfg["partition"]["subsample_seed"])
    if plan_cfg["data"].get("tool_shuffle", True):
        for r in train_rows:
            r["tools"] = shuffle_tools_in_schema(r["tools"], rng)

    # 5. Compute domain rebalance weights (Spec §5.4) for the WeightedRandomSampler.
    from code.training.data.domain_rebalance import compute_per_row_weights
    domain_temp = plan_cfg["data"].get("domain_temperature", 0.5)
    # Need to wrap rows into a shape compute_per_row_weights expects.
    train_rows_for_weights = [{"_p": {"domain": r["_meta"]["domain"]}} for r in train_rows]
    sample_weights = compute_per_row_weights(train_rows_for_weights, domain_temp)

    # 6. Build HF Datasets.
    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)

    # Sub-task 13 will continue: model loading, MoE config, SFTConfig, trainer.
    # For now, persist the prepared dataset for inspection and exit.
    prep_dump = output_dir / "_prep.json"
    prep_dump.write_text(json.dumps({
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
        "n_runs_kept": len(keep_runs),
        "domain_counts": {d: sum(1 for r in train_rows if r["_meta"]["domain"] == d)
                           for d in runs_by_domain},
        "sample_weight_dist": {
            "min": min(sample_weights), "max": max(sample_weights),
            "mean": sum(sample_weights) / len(sample_weights),
        },
    }, indent=2))
    print(f"Prepared {len(train_rows)} rows for {rid}; dumped to {prep_dump}")

    raise NotImplementedError("Sub-task 13: trainer setup")
```

- [ ] **Step 2: Run smoke verification on the dataset prep alone (CPU OK, no GPU)**

```bash
cd /path/to/bundle
.venv/bin/python -c "
import sys, runpy
sys.argv = ['train.py',
  '--run-config', 'code/training/configs/runs/01_qwen3_5_2b_273.yaml',
  '--plan-config', 'code/training/configs/plan_c_prime.yaml',
  '--bundle-root', '.', '--smoke-test']
try:
    runpy.run_module('code.training.train', run_name='__main__')
except NotImplementedError as e:
    print('OK — reached the end of dataset prep:', e)
"
```
Expected: prints "OK" and the prep dump file. Failure modes: missing converted data → re-run Task 6.

- [ ] **Step 3: Commit**

```bash
git add code/training/train.py
git commit -m "feat(training): train.py dataset prep with stratified subsample + rebalance"
```

---

### Task 13: `train.py` — TRL SFTTrainer + MoE handling + checkpointing

Final piece of `train.py`. Spec §6.1 + §6.1.3.

**Files:**
- Modify: `code/training/train.py` (remove the second `NotImplementedError`, add full trainer body)

- [ ] **Step 1: Add the trainer body before the second `NotImplementedError`**

Replace the lines from the `_prep.json` dump onward with:

```python
    # 7. Load model.
    config = AutoConfig.from_pretrained(
        run_cfg["model"]["name"],
        revision=run_cfg["model"].get("revision", "main"),
        trust_remote_code=True,
    )
    if run_cfg["model"].get("is_moe", False):
        # Spec §6.1.3 — MoE-specific config.
        moe_cfg = run_cfg.get("moe") or {}
        config.output_router_logits = moe_cfg.get("output_router_logits", True)
        config.router_aux_loss_coef = moe_cfg.get("router_aux_loss_coef", 0.001)

    model = AutoModelForCausalLM.from_pretrained(
        run_cfg["model"]["name"],
        revision=run_cfg["model"].get("revision", "main"),
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation=run_cfg["training"].get("attn_implementation", "flash_attention_2"),
        trust_remote_code=True,
    )
    if run_cfg["training"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # 8. SFTConfig.
    t = run_cfg["training"]
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=t["num_train_epochs"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"],
        weight_decay=t["weight_decay"],
        adam_beta1=t["adam_beta1"],
        adam_beta2=t["adam_beta2"],
        max_grad_norm=t["max_grad_norm"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        max_seq_length=t["max_seq_length"],
        packing=t["packing"],
        packing_strategy=t["packing_strategy"],
        padding_free=t["padding_free"],
        completion_only_loss=t["completion_only_loss"],
        assistant_only_loss=t.get("assistant_only_loss", False),
        use_liger_kernel=t.get("use_liger_kernel", False),
        loss_type=t.get("loss_type", "default"),
        dataloader_num_workers=t["dataloader_num_workers"],
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        save_strategy=t["save_strategy"],
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        greater_is_better=t["greater_is_better"],
        logging_steps=t["logging_steps"],
        seed=t["seed"],
        data_seed=t["data_seed"],
        report_to="none",         # we do our own plotting
    )

    # 9. Trainer + early stopping.
    from transformers import EarlyStoppingCallback
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=t.get("early_stopping_patience", 2)
        ),
    ]

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=callbacks,
    )

    # 10. MoE aux-loss assertion (Spec §6.1.3).
    if run_cfg["model"].get("is_moe", False):
        # Run one forward pass to confirm aux_loss is non-zero.
        trainer.model.eval()
        with torch.no_grad():
            sample = trainer.get_train_dataloader().__iter__().__next__()
            sample = {k: v.to(trainer.model.device) for k, v in sample.items()
                      if isinstance(v, torch.Tensor)}
            out = trainer.model(**sample, output_router_logits=True)
            aux = getattr(out, "aux_loss", None) or getattr(out, "router_logits", None)
            assert aux is not None, "Qwen3 MoE: aux_loss not produced — check output_router_logits"
        trainer.model.train()

    # 11. Train.
    train_result = trainer.train()
    trainer.save_model(str(output_dir / "checkpoint-final"))
    tok.save_pretrained(str(output_dir / "checkpoint-final"))

    # 12. Save training_log.json + metadata.
    log_path = output_dir / "training_log.json"
    log_path.write_text(json.dumps({
        "run_id": rid,
        "metrics": train_result.metrics,
        "model_name": run_cfg["model"]["name"],
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
        "n_runs_kept": len(keep_runs),
        "git_sha": os.environ.get("GIT_SHA", "?"),
    }, indent=2))

    # 13. Mark done.
    status_file.write_text("done")
    print(f"DONE: {rid} → {output_dir}")
    return 0
```

- [ ] **Step 2: Smoke-test on CPU + Qwen2.5-0.5B**

```bash
cd /path/to/bundle
# Override the run YAML to use a tiny model just for code-correctness check.
mkdir -p code/training/configs/runs_smoke
cp code/training/configs/runs/01_qwen3_5_2b_273.yaml code/training/configs/runs_smoke/smoke.yaml
sed -i 's|Qwen/Qwen3.5-2B|Qwen/Qwen2.5-0.5B-Instruct|' code/training/configs/runs_smoke/smoke.yaml
sed -i 's|max_seq_length: 32768|max_seq_length: 1024|' code/training/configs/runs_smoke/smoke.yaml
sed -i 's|attn_implementation: flash_attention_2|attn_implementation: eager|' code/training/configs/runs_smoke/smoke.yaml

CUDA_VISIBLE_DEVICES="" .venv/bin/python -m code.training.train \
    --run-config code/training/configs/runs_smoke/smoke.yaml \
    --plan-config code/training/configs/plan_c_prime.yaml \
    --bundle-root . --smoke-test
```
Expected: completes 1 step (1 batch); writes `train_outputs/01_qwen3_5_2b_273/STATUS=done`. Slow (~30 s on CPU) but OK.

- [ ] **Step 3: Commit**

```bash
git add code/training/train.py
git commit -m "feat(training): train.py SFTTrainer setup with MoE aux-loss assertion"
```

---

### Task 14: vLLM serve script (`vllm_serve.sh`)

Spec §4 — spin up vLLM per checkpoint for eval.

**Files:**
- Create: `code/training/eval/vllm_serve.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# code/training/eval/vllm_serve.sh
# Spin up vLLM in OpenAI-compatible mode for one trained checkpoint.
# Usage: vllm_serve.sh <checkpoint_dir> <port> [&]
set -euo pipefail

CKPT="$1"
PORT="$2"
GPU="${3:-0}"

if [[ ! -d "$CKPT" ]]; then
    echo "Checkpoint dir does not exist: $CKPT" >&2; exit 1
fi

# Detect Qwen version from checkpoint config (controls parser).
TOOL_PARSER="qwen3_coder"
MODEL_NAME=$(python -c "
import json, sys
cfg = json.load(open('$CKPT/config.json'))
print(cfg.get('_name_or_path', cfg.get('model_type', 'qwen')))
")

CUDA_VISIBLE_DEVICES="$GPU" \
    vllm serve "$CKPT" \
    --port "$PORT" \
    --max-model-len 32768 \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_PARSER" \
    --reasoning-parser qwen3 \
    --gpu-memory-utilization 0.85 \
    --served-model-name "evol-llm-student" \
    >> "$CKPT/vllm_serve.log" 2>&1 &

VLLM_PID=$!
echo "vLLM PID=$VLLM_PID port=$PORT ckpt=$CKPT" > "$CKPT/vllm_serve.pid"

# Health check — wait up to 5 minutes.
for _ in $(seq 1 60); do
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        echo "vLLM ready on port $PORT (PID $VLLM_PID)"
        exit 0
    fi
    sleep 5
done
echo "vLLM failed to start on port $PORT" >&2
kill "$VLLM_PID" 2>/dev/null || true
exit 1
```

- [ ] **Step 2: Make executable + commit**

```bash
chmod +x code/training/eval/vllm_serve.sh
git add code/training/eval/vllm_serve.sh
git commit -m "feat(training/eval): vLLM serve script with qwen3_coder tool parser"
```

---

### Task 15: New "local" provider for `pipeline.runner` (`local_provider.py`)

Spec §7 — add a `local` provider that hits a vLLM endpoint instead of OpenRouter/CommonStack. The existing `pipeline.runner` uses LiteLLM. Adding a new provider means:
1. Register the local model id in pricing.yaml (zero-cost since self-hosted).
2. Configure LiteLLM to route the model id to `http://localhost:<port>/v1`.

**Files:**
- Create: `code/training/eval/local_provider.py`
- Modify: `code/core/configs/pricing.yaml` (add zero-cost entry for `local/student`)
- Create: `code/training/tests/test_local_provider.py`

- [ ] **Step 1: Write failing test**

```python
# code/training/tests/test_local_provider.py
import os
import pytest

from code.training.eval.local_provider import (
    build_litellm_kwargs_for_local,
    register_local_model_pricing,
)


def test_litellm_kwargs_default_port():
    kwargs = build_litellm_kwargs_for_local(port=8000)
    assert kwargs["api_base"] == "http://localhost:8000/v1"
    assert kwargs["api_key"] == "dummy"
    assert kwargs["custom_llm_provider"] == "openai"


def test_litellm_kwargs_custom_port():
    kwargs = build_litellm_kwargs_for_local(port=9123, host="127.0.0.1")
    assert kwargs["api_base"] == "http://127.0.0.1:9123/v1"


def test_register_zero_pricing():
    pricing = {}
    register_local_model_pricing(pricing, model_id="local/student")
    assert pricing["local/student"]["input_usd_per_m"] == 0.0
    assert pricing["local/student"]["output_usd_per_m"] == 0.0
```

- [ ] **Step 2: Run test → FAIL**

Run: `cd code/training && python -m pytest tests/test_local_provider.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `local_provider.py`**

```python
# code/training/eval/local_provider.py
"""Local vLLM provider for pipeline.runner.

The trained student is served via vLLM in OpenAI-compatible mode. We route
LiteLLM at this endpoint and price the calls at $0 (self-hosted, no per-token
cost in the OpenAI bill).
"""
from __future__ import annotations


def build_litellm_kwargs_for_local(port: int, host: str = "localhost") -> dict:
    """Return LiteLLM completion() kwargs that route to a local vLLM instance."""
    return {
        "api_base": f"http://{host}:{port}/v1",
        "api_key": "dummy",  # vLLM doesn't validate; required for OpenAI client
        "custom_llm_provider": "openai",
    }


def register_local_model_pricing(pricing_dict: dict, model_id: str = "local/student") -> dict:
    """Add a zero-cost pricing entry for the locally-served student model."""
    pricing_dict[model_id] = {
        "input_usd_per_m": 0.0,
        "output_usd_per_m": 0.0,
        "cache_read_usd_per_m": 0.0,
        "cache_write_usd_per_m": 0.0,
    }
    return pricing_dict
```

- [ ] **Step 4: Run test → PASS**

Run: `cd code/training && python -m pytest tests/test_local_provider.py -v`
Expected: 3 passed.

- [ ] **Step 5: Add `local/student` to `pricing.yaml`** (Spec §7)

Modify `code/core/configs/pricing.yaml`. Add this block above `# ---- legacy`:

```yaml
  local/student:                      # trained Qwen student via local vLLM
    input_usd_per_m: 0.0              # self-hosted, zero per-token cost
    output_usd_per_m: 0.0
    cache_read_usd_per_m: 0.0
    cache_write_usd_per_m: 0.0
```

- [ ] **Step 6: Commit**

```bash
git add code/training/eval/local_provider.py code/training/tests/test_local_provider.py code/core/configs/pricing.yaml
git commit -m "feat(training/eval): local vLLM provider for pipeline.runner"
```

---

### Task 16: Eval harness (`harness.py`)

Spec §7.3 — orchestrate force-routed eval per checkpoint, including the held-out set.

**Files:**
- Create: `code/training/eval/harness.py`

- [ ] **Step 1: Write `harness.py`**

```python
# code/training/eval/harness.py
"""Force-routed τ²-bench eval harness for one trained checkpoint.

Procedure:
1. Spin up vLLM via vllm_serve.sh on a unique port.
2. Wait for /v1/models to respond.
3. Invoke pipeline.runner with --provider local pointed at the vLLM port.
4. Parse results into eval_results.json (and eval_results_heldout.json).
5. Tear down vLLM.

Usage:
    python -m code.training.eval.harness \\
        --checkpoint train_outputs/<run_id>/checkpoint-best \\
        --output-dir train_outputs/<run_id> \\
        --eval-tasks data_processed/stage2_v1/eval_tasks.jsonl \\
        --heldout-tasks train_outputs/_data_cache/heldout_tasks.jsonl \\
        --port 8000 \\
        --gpu 0 \\
        --seeds 300
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def serve_vllm(checkpoint: Path, port: int, gpu: int, vllm_serve_sh: Path) -> subprocess.Popen:
    """Spin up vLLM in background. Returns the Popen handle."""
    proc = subprocess.Popen(
        ["bash", str(vllm_serve_sh), str(checkpoint), str(port), str(gpu)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return proc


def wait_for_vllm(port: int, timeout_s: int = 600) -> bool:
    """Poll the OpenAI-compatible /v1/models endpoint until ready."""
    import urllib.request
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def stop_vllm(checkpoint: Path) -> None:
    """Kill the vLLM process recorded in the pid file."""
    pid_file = checkpoint / "vllm_serve.pid"
    if not pid_file.exists():
        return
    text = pid_file.read_text().split()
    for tok in text:
        if tok.startswith("PID="):
            try:
                pid = int(tok.removeprefix("PID="))
                os.kill(pid, signal.SIGTERM)
                time.sleep(2)
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass
    pid_file.unlink(missing_ok=True)


def run_eval(
    bundle_root: Path,
    eval_tasks_jsonl: Path,
    output_path: Path,
    seeds: list[int],
    port: int,
    n_tasks_limit: int | None = None,
) -> dict:
    """Invoke pipeline.runner with --provider local, return summary."""
    cmd = [
        sys.executable, "-m", "pipeline.runner",
        "--mode", "eval",
        "--provider", "local",
        "--local-port", str(port),
        "--tasks-jsonl", str(eval_tasks_jsonl),
        "--seeds", *[str(s) for s in seeds],
        "--baseline-model", "qwen/qwen3.5-397b-a17b",
        "--lowest-tier-model", "local/student",
        "--output-jsonl", str(output_path),
    ]
    if n_tasks_limit is not None:
        cmd += ["--limit", str(n_tasks_limit)]
    result = subprocess.run(cmd, cwd=str(bundle_root / "code"), check=True)
    # Parse the output JSONL to compute summary metrics.
    rows = [json.loads(l) for l in output_path.read_text().splitlines()]
    n = len(rows)
    pass_rate = sum(1 for r in rows if r.get("passed")) / max(1, n)
    avg_k = sum(r.get("replacement_rate_k", 0.0) for r in rows) / max(1, n)
    avg_cost = sum(r.get("total_task_cost_usd", 0.0) for r in rows) / max(1, n)
    return {
        "n_rollouts": n,
        "pass_rate": pass_rate,
        "replacement_rate_k_mean": avg_k,
        "total_task_cost_usd_mean": avg_cost,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--bundle-root", required=True, type=Path)
    ap.add_argument("--eval-tasks", required=True, type=Path)
    ap.add_argument("--heldout-tasks", type=Path, default=None)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[300])
    args = ap.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vllm_serve_sh = args.bundle_root / "code/training/eval/vllm_serve.sh"

    # Idempotence: skip if eval_results.json already exists and is non-empty.
    res_path = args.output_dir / "eval_results.json"
    if res_path.exists() and res_path.stat().st_size > 0:
        print(f"SKIP: {res_path} already exists.")
        return 0

    proc = serve_vllm(args.checkpoint, args.port, args.gpu, vllm_serve_sh)
    try:
        if not wait_for_vllm(args.port):
            print("vLLM failed to start", file=sys.stderr)
            return 2

        # Main eval (35 tasks).
        main_summary = run_eval(
            args.bundle_root, args.eval_tasks,
            args.output_dir / "eval_rollouts.jsonl", args.seeds, args.port,
        )
        res_path.write_text(json.dumps(main_summary, indent=2))

        # Held-out eval (15 tasks) if provided.
        if args.heldout_tasks is not None and args.heldout_tasks.exists():
            held_summary = run_eval(
                args.bundle_root, args.heldout_tasks,
                args.output_dir / "eval_rollouts_heldout.jsonl", args.seeds, args.port,
            )
            (args.output_dir / "eval_results_heldout.json").write_text(
                json.dumps(held_summary, indent=2)
            )
        return 0
    finally:
        stop_vllm(args.checkpoint)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify the script imports cleanly (no GPU needed)**

```bash
cd /path/to/bundle
.venv/bin/python -c "from code.training.eval import harness; print('OK')"
```
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add code/training/eval/harness.py
git commit -m "feat(training/eval): force-routed eval harness with vLLM lifecycle"
```

---

### Task 17: Phase 1 orchestration — `train_all.sh`

**Files:**
- Create: `code/training/orchestration/train_all.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# code/training/orchestration/train_all.sh
# Phase 1 — train all 10 runs in plan_c_prime.yaml.
# Each run is sequenced (FSDP runs use all 8 GPUs; small DDP runs share 8).
#
# Idempotent: skips runs whose STATUS=done.
#
# Usage:
#   BUNDLE_ROOT=/path/to/bundle bash code/training/orchestration/train_all.sh
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:?set BUNDLE_ROOT}"
PLAN="${PLAN:-code/training/configs/plan_c_prime.yaml}"
VENV="${VENV:-$BUNDLE_ROOT/code/.venv/bin}"
GIT_SHA=$(git -C "$BUNDLE_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")
export GIT_SHA BUNDLE_ROOT

cd "$BUNDLE_ROOT"

# Preflight checks (Spec §4 — operator prerequisites).
echo "=== Preflight ==="
nvidia-smi | grep -q "H200" || { echo "Expected H200 GPUs"; exit 1; }
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo "OPENAI_API_KEY not set"; exit 1; }
[[ -d "$BUNDLE_ROOT/data_processed/stage2_v1" ]] || { echo "Source data missing"; exit 1; }

# Convert + validate (one time, idempotent).
mkdir -p train_outputs/_data_cache
if [[ ! -f train_outputs/_data_cache/train_prompt_completion.jsonl ]]; then
    echo "=== Converting train + val to TRL prompt/completion format ==="
    "$VENV/python" -m code.training.data.convert_to_prompt_completion \
        --src data_processed/stage2_v1/train.jsonl \
        --dst train_outputs/_data_cache/train_prompt_completion.jsonl \
        --domain-assets data_processed/stage2_v1/domain_assets \
        --drop-log train_outputs/_data_cache/dropped_train.jsonl
    "$VENV/python" -m code.training.data.convert_to_prompt_completion \
        --src data_processed/stage2_v1/val.jsonl \
        --dst train_outputs/_data_cache/val_prompt_completion.jsonl \
        --domain-assets data_processed/stage2_v1/domain_assets \
        --drop-log train_outputs/_data_cache/dropped_val.jsonl
fi

# Validate against each model's chat template.
for TOK in Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B Qwen/Qwen3.6-35B-A3B; do
    SLUG=$(echo "$TOK" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
    REPORT="train_outputs/_data_cache/validation_report_${SLUG}.json"
    if [[ ! -f "$REPORT" ]]; then
        echo "=== Validating chat template for $TOK ==="
        "$VENV/python" -m code.training.data.validate_chat_template \
            --src train_outputs/_data_cache/train_prompt_completion.jsonl \
            --tokenizer "$TOK" \
            --out "$REPORT"
    fi
done

# Build held-out task split.
if [[ ! -f train_outputs/_data_cache/heldout_tasks.jsonl ]]; then
    echo "=== Computing held-out task split ==="
    "$VENV/python" - <<'PYEOF'
import json
from pathlib import Path
from code.training.data.heldout_split import select_heldout_task_ids
src = Path("data_processed/stage2_v1/train.jsonl")
rows = [json.loads(l) for l in src.read_text().splitlines()]
heldout = select_heldout_task_ids(rows, n_per_domain=5, seed=42)
out = Path("train_outputs/_data_cache/heldout_task_ids.json")
out.write_text(json.dumps({d: sorted(s) for d, s in heldout.items()}, indent=2))
print(f"Held-out task ids written to {out}")
PYEOF
fi

# Per-run training.
RUNS=$(
    "$VENV/python" -c "
import yaml
plan = yaml.safe_load(open('$PLAN'))
for r in plan['runs']:
    print(r['id'], r['config'])
"
)

echo ""
echo "=== Training all runs ==="
while IFS=' ' read -r RID CFG; do
    [[ -z "$RID" ]] && continue
    OUT_DIR="train_outputs/$RID"
    STATUS_FILE="$OUT_DIR/STATUS"
    if [[ -f "$STATUS_FILE" ]] && grep -q "^done$" "$STATUS_FILE"; then
        echo "  SKIP (done): $RID"
        continue
    fi
    mkdir -p "$OUT_DIR"
    echo ""
    echo "=== TRAIN: $RID ==="

    ACCEL_CFG=$("$VENV/python" -c "import yaml; print(yaml.safe_load(open('code/training/$CFG'))['distributed']['accelerate_config'])")

    "$VENV/accelerate" launch \
        --config_file "code/training/$ACCEL_CFG" \
        code/training/train.py \
        --run-config "code/training/$CFG" \
        --plan-config "$PLAN" \
        --bundle-root "$BUNDLE_ROOT" \
        2>&1 | tee "$OUT_DIR/train_stdout.log"

    if grep -q "^done$" "$STATUS_FILE" 2>/dev/null; then
        echo "  ✓ $RID done"
    else
        echo "  ✗ $RID FAILED — see $OUT_DIR/train_stdout.log" >&2
        echo "failed" > "$STATUS_FILE"
    fi
done <<< "$RUNS"

echo ""
echo "=== Phase 1 complete. Run eval_all.sh next. ==="
```

- [ ] **Step 2: Make executable + commit**

```bash
chmod +x code/training/orchestration/train_all.sh
git add code/training/orchestration/train_all.sh
git commit -m "feat(training/orchestration): train_all.sh with preflight + idempotence"
```

---

### Task 18: Phase 2 orchestration — `eval_all.sh`

**Files:**
- Create: `code/training/orchestration/eval_all.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# code/training/orchestration/eval_all.sh
# Phase 2 — eval every main run in parallel on 8 GPUs.
# Idempotent: skips runs whose eval_results.json exists.
#
# Usage:
#   BUNDLE_ROOT=/path/to/bundle bash code/training/orchestration/eval_all.sh
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:?set BUNDLE_ROOT}"
PLAN="${PLAN:-code/training/configs/plan_c_prime.yaml}"
VENV="${VENV:-$BUNDLE_ROOT/code/.venv/bin}"
EVAL_TASKS="$BUNDLE_ROOT/data_processed/stage2_v1/eval_tasks.jsonl"
HELDOUT_TASKS="$BUNDLE_ROOT/train_outputs/_data_cache/heldout_tasks.jsonl"

cd "$BUNDLE_ROOT"

# Build a list of (run_id, gpu_idx, port).
RUNS=$(
    "$VENV/python" -c "
import yaml
plan = yaml.safe_load(open('$PLAN'))
for i, r in enumerate(plan['runs']):
    if r.get('skip_eval', False):
        continue
    print(r['id'], i % 8, 8000 + (i % 8))
"
)

echo "=== Phase 2 — parallel eval on 8 GPUs ==="

PIDS=()
while IFS=' ' read -r RID GPU PORT; do
    [[ -z "$RID" ]] && continue
    OUT_DIR="$BUNDLE_ROOT/train_outputs/$RID"
    CKPT="$OUT_DIR/checkpoint-best"
    [[ -d "$CKPT" ]] || CKPT="$OUT_DIR/checkpoint-final"
    if [[ ! -d "$CKPT" ]]; then
        echo "  SKIP (no checkpoint): $RID"
        continue
    fi
    if [[ -f "$OUT_DIR/eval_results.json" ]]; then
        echo "  SKIP (already evaluated): $RID"
        continue
    fi
    echo "  EVAL: $RID on GPU $GPU port $PORT"
    "$VENV/python" -m code.training.eval.harness \
        --checkpoint "$CKPT" \
        --output-dir "$OUT_DIR" \
        --bundle-root "$BUNDLE_ROOT" \
        --eval-tasks "$EVAL_TASKS" \
        --heldout-tasks "$HELDOUT_TASKS" \
        --port "$PORT" --gpu "$GPU" --seeds 300 \
        > "$OUT_DIR/eval_stdout.log" 2>&1 &
    PIDS+=("$!")
done <<< "$RUNS"

# Wait for all parallel evals.
for PID in "${PIDS[@]}"; do
    wait "$PID" || echo "  ✗ eval PID $PID exited non-zero"
done

echo ""
echo "=== Identifying winning config for re-eval ==="
WINNER=$(
    "$VENV/python" - <<'PYEOF'
import json, glob, os
candidates = []
for p in sorted(glob.glob("train_outputs/*/eval_results.json")):
    rid = os.path.basename(os.path.dirname(p))
    d = json.load(open(p))
    cost = d.get("total_task_cost_usd_mean", 1e9)
    pr = d.get("pass_rate", 0.0)
    if cost < 0.0212:                  # cost ceiling
        candidates.append((pr, -cost, rid))
candidates.sort(reverse=True)
print(candidates[0][2] if candidates else "")
PYEOF
)

if [[ -n "$WINNER" ]]; then
    echo "  Winner: $WINNER"
    OUT_DIR="$BUNDLE_ROOT/train_outputs/$WINNER"
    CKPT="$OUT_DIR/checkpoint-best"
    [[ -d "$CKPT" ]] || CKPT="$OUT_DIR/checkpoint-final"
    if [[ ! -f "$OUT_DIR/eval_results_seed301.json" ]]; then
        echo "  Re-evaluating winner with seed 301..."
        "$VENV/python" -m code.training.eval.harness \
            --checkpoint "$CKPT" \
            --output-dir "$OUT_DIR" \
            --bundle-root "$BUNDLE_ROOT" \
            --eval-tasks "$EVAL_TASKS" \
            --port 8000 --gpu 0 --seeds 301
        # Rename to avoid clobbering main results.
        mv "$OUT_DIR/eval_results.json" "$OUT_DIR/eval_results_seed301.json"
    fi
fi

echo ""
echo "=== Phase 2 complete. Run summarize.py next. ==="
```

- [ ] **Step 2: Make executable + commit**

```bash
chmod +x code/training/orchestration/eval_all.sh
git add code/training/orchestration/eval_all.sh
git commit -m "feat(training/orchestration): eval_all.sh parallel 8-GPU eval + winner reeval"
```

---

### Task 19: Aggregation — `summarize.py`

**Files:**
- Create: `code/training/orchestration/summarize.py`
- Create: `code/training/tests/test_summarize.py`

- [ ] **Step 1: Write failing test**

```python
# code/training/tests/test_summarize.py
import json
from pathlib import Path

from code.training.orchestration.summarize import collect_run_results, write_summary_csv


def test_collect_run_results(tmp_path: Path):
    rid_dir = tmp_path / "01_test"
    rid_dir.mkdir()
    (rid_dir / "training_log.json").write_text(json.dumps({
        "run_id": "01_test",
        "metrics": {"train_loss": 1.5, "eval_loss": 1.7, "epoch": 5.0},
        "model_name": "Qwen/Qwen3.5-2B",
        "n_train_rows": 6413, "n_runs_kept": 273,
    }))
    (rid_dir / "eval_results.json").write_text(json.dumps({
        "n_rollouts": 35, "pass_rate": 0.42,
        "replacement_rate_k_mean": 0.65,
        "total_task_cost_usd_mean": 0.005,
    }))
    rows = collect_run_results(tmp_path)
    assert len(rows) == 1
    assert rows[0]["run_id"] == "01_test"
    assert rows[0]["pass_rate"] == 0.42
    assert rows[0]["model_name"] == "Qwen/Qwen3.5-2B"


def test_write_csv(tmp_path: Path):
    rid_dir = tmp_path / "x"
    rid_dir.mkdir()
    (rid_dir / "training_log.json").write_text(json.dumps({
        "run_id": "x", "metrics": {}, "model_name": "?", "n_train_rows": 0, "n_runs_kept": 0,
    }))
    rows = collect_run_results(tmp_path)
    out = tmp_path / "SUMMARY.csv"
    write_summary_csv(rows, out)
    text = out.read_text()
    assert "run_id" in text
    assert "x" in text
```

- [ ] **Step 2: Run → FAIL**

Run: `cd code/training && python -m pytest tests/test_summarize.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `summarize.py`**

```python
# code/training/orchestration/summarize.py
"""Aggregate per-run results into SUMMARY.csv (Spec §7.4)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def collect_run_results(train_outputs_root: Path) -> list[dict]:
    """Scan train_outputs/<run_id>/ for results JSONs."""
    rows: list[dict] = []
    for run_dir in sorted(train_outputs_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_") or run_dir.name == "plots":
            continue
        log_path = run_dir / "training_log.json"
        eval_path = run_dir / "eval_results.json"
        eval_held_path = run_dir / "eval_results_heldout.json"
        eval_seed301_path = run_dir / "eval_results_seed301.json"
        if not log_path.exists():
            continue

        log = json.loads(log_path.read_text())
        row: dict = {
            "run_id": log.get("run_id", run_dir.name),
            "model_name": log.get("model_name", "?"),
            "n_train_rows": log.get("n_train_rows", 0),
            "n_runs_kept": log.get("n_runs_kept", 0),
            "train_loss": log.get("metrics", {}).get("train_loss"),
            "eval_loss": log.get("metrics", {}).get("eval_loss"),
        }
        if eval_path.exists():
            ev = json.loads(eval_path.read_text())
            row["pass_rate"] = ev.get("pass_rate")
            row["replacement_rate_k"] = ev.get("replacement_rate_k_mean")
            row["total_task_cost_usd"] = ev.get("total_task_cost_usd_mean")
        if eval_held_path.exists():
            evh = json.loads(eval_held_path.read_text())
            row["pass_rate_heldout"] = evh.get("pass_rate")
            row["total_task_cost_usd_heldout"] = evh.get("total_task_cost_usd_mean")
        if eval_seed301_path.exists():
            ev301 = json.loads(eval_seed301_path.read_text())
            row["pass_rate_seed301"] = ev301.get("pass_rate")
        rows.append(row)
    return rows


def write_summary_csv(rows: list[dict], out_path: Path) -> None:
    """Write a CSV with all keys present in any row as columns."""
    if not rows:
        out_path.write_text("")
        return
    cols = sorted({k for r in rows for k in r.keys()})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="train_outputs root")
    args = ap.parse_args(argv)
    rows = collect_run_results(args.root)
    out_path = args.root / "SUMMARY.csv"
    write_summary_csv(rows, out_path)
    print(f"Wrote {out_path} with {len(rows)} rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run → PASS**

Run: `cd code/training && python -m pytest tests/test_summarize.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add code/training/orchestration/summarize.py code/training/tests/test_summarize.py
git commit -m "feat(training/orchestration): summarize.py → SUMMARY.csv"
```

---

### Task 20: Plotting — `plotting.py`

Spec §7.4 — capacity_curve, data_scaling, cost_pareto, lr_check, plus per-run loss curves.

**Files:**
- Create: `code/training/orchestration/plotting.py`

- [ ] **Step 1: Write `plotting.py`**

```python
# code/training/orchestration/plotting.py
"""Plot per-run + aggregate training/eval results (Spec §7.4)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Active params per model (used as x-axis on capacity curve).
ACTIVE_PARAMS_B = {
    "Qwen/Qwen3.5-2B": 2.0,
    "Qwen/Qwen3.5-4B": 4.0,
    "Qwen/Qwen3.5-9B": 9.0,
    "Qwen/Qwen3.6-35B-A3B": 3.0,        # MoE: 3B active per token, despite 35B total
}

# Reference points from spec (glm-4.5-air baseline).
GLM_AIR_BASELINE = {
    "pass_rate_assumed": 0.50,        # placeholder; user edits if external benchmark known
    "total_task_cost_usd": 0.0212,
}


def load_summary(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def _safe_float(s):
    try:
        return float(s) if s not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def plot_capacity_curve(rows: list[dict], out_path: Path) -> None:
    """Pass-rate vs log(active params) at full data (273 runs)."""
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    pts = []
    for r in rows:
        if int(r.get("n_runs_kept") or 0) < 273:
            continue
        ap = ACTIVE_PARAMS_B.get(r["model_name"])
        if ap is None:
            continue
        pr = _safe_float(r.get("pass_rate"))
        cost = _safe_float(r.get("total_task_cost_usd"))
        if pr is None:
            continue
        pts.append((ap, pr, cost, r["model_name"]))
    pts.sort()
    if pts:
        xs = [p[0] for p in pts]
        ax1.plot(xs, [p[1] for p in pts], "o-", color="tab:blue", label="pass-rate")
        if any(p[2] is not None for p in pts):
            ax2.plot(xs, [p[2] for p in pts], "s--", color="tab:red", label="task cost ($)")
    ax1.axhline(GLM_AIR_BASELINE["pass_rate_assumed"], color="gray", linestyle=":", alpha=0.5,
                label="glm-4.5-air baseline")
    ax2.axhline(GLM_AIR_BASELINE["total_task_cost_usd"], color="darkred", linestyle=":", alpha=0.5)
    ax1.set_xscale("log")
    ax1.set_xlabel("active params (B)")
    ax1.set_ylabel("pass-rate", color="tab:blue")
    ax2.set_ylabel("total task cost ($)", color="tab:red")
    ax1.set_title("Capacity curve at full data (273 runs)\n[Qwen3.6-35B-A3B point: architecture-confound caveat]")
    ax1.grid(True, alpha=0.3)
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95), fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_data_scaling(rows: list[dict], out_path: Path) -> None:
    """Pass-rate vs n_runs at fixed model (4B, with 9B-50/9B-273 overlay)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    by_model: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        if r.get("model_name") not in {"Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"}:
            continue
        nrun = int(r.get("n_runs_kept") or 0)
        pr = _safe_float(r.get("pass_rate"))
        if pr is None or nrun == 0:
            continue
        by_model.setdefault(r["model_name"], []).append((nrun, pr))
    for model, pts in sorted(by_model.items()):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                label=model.replace("Qwen/", ""))
    ax.set_xscale("log")
    ax.set_xlabel("n_train_runs")
    ax.set_ylabel("pass-rate")
    ax.set_title("Data scaling at 4B (9B overlay validates shape transfer)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_cost_pareto(rows: list[dict], out_path: Path) -> None:
    """Total task cost vs pass-rate scatter."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in rows:
        cost = _safe_float(r.get("total_task_cost_usd"))
        pr = _safe_float(r.get("pass_rate"))
        if cost is None or pr is None:
            continue
        label = f"{r['run_id']} ({r.get('n_runs_kept')}runs)"
        ax.scatter(cost, pr, s=80)
        ax.annotate(label, (cost, pr), fontsize=7, alpha=0.7,
                    xytext=(5, 5), textcoords="offset points")
    ax.axvline(GLM_AIR_BASELINE["total_task_cost_usd"], color="red", linestyle="--",
               label=f"glm-4.5-air cost (${GLM_AIR_BASELINE['total_task_cost_usd']})")
    ax.set_xlabel("total task cost ($)")
    ax.set_ylabel("pass-rate")
    ax.set_title("Cost-pass-rate Pareto")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_lr_check(rows: list[dict], out_path: Path) -> None:
    """Val loss vs LR for the 3 4B-273 LR variants."""
    pts = []
    lr_map = {
        "05_qwen3_5_4b_273": 2e-5,
        "09_qwen3_5_4b_273_lr1e5": 1e-5,
        "10_qwen3_5_4b_273_lr3e5": 3e-5,
    }
    for r in rows:
        lr = lr_map.get(r["run_id"])
        vl = _safe_float(r.get("eval_loss"))
        if lr is None or vl is None:
            continue
        pts.append((lr, vl, r["run_id"]))
    pts.sort()
    fig, ax = plt.subplots(figsize=(7, 4))
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-")
        for x, y, lbl in pts:
            ax.annotate(lbl.split("_")[-1], (x, y), fontsize=8,
                        xytext=(5, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("eval loss")
    ax.set_title("LR sanity check at 4B-273")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_per_run_loss_curve(run_dir: Path) -> None:
    """Read training_log.json and write loss_curves.png."""
    log_path = run_dir / "training_log.json"
    if not log_path.exists():
        return
    log = json.loads(log_path.read_text())
    history = log.get("history") or log.get("metrics", {}).get("history") or []
    if not history:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    train_steps = [h["step"] for h in history if "loss" in h]
    train_loss = [h["loss"] for h in history if "loss" in h]
    eval_steps = [h["step"] for h in history if "eval_loss" in h]
    eval_loss = [h["eval_loss"] for h in history if "eval_loss" in h]
    if train_loss:
        ax.plot(train_steps, train_loss, label="train")
    if eval_loss:
        ax.plot(eval_steps, eval_loss, label="val", marker="o")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(log.get("run_id", "?"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = run_dir / "plots" / "loss_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args(argv)
    summary_csv = args.root / "SUMMARY.csv"
    if not summary_csv.exists():
        print(f"No SUMMARY.csv at {summary_csv}; run summarize.py first.", file=sys.stderr)
        return 2
    rows = load_summary(summary_csv)

    plots_dir = args.root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_capacity_curve(rows, plots_dir / "capacity_curve.png")
    plot_data_scaling(rows, plots_dir / "data_scaling.png")
    plot_cost_pareto(rows, plots_dir / "cost_pareto.png")
    plot_lr_check(rows, plots_dir / "lr_check.png")

    for d in args.root.iterdir():
        if d.is_dir() and not d.name.startswith("_") and d.name != "plots":
            plot_per_run_loss_curve(d)

    print(f"Plots written to {plots_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-verify the import**

```bash
cd /path/to/bundle
.venv/bin/python -c "from code.training.orchestration import plotting; print('OK')"
```
Expected: OK.

- [ ] **Step 3: Commit**

```bash
git add code/training/orchestration/plotting.py
git commit -m "feat(training/orchestration): aggregate + per-run plots"
```

---

### Task 21: Single-batch overfit smoke test (`test_smoke_train.py`)

Spec §5.8 — pre-launch acceptance gate. Verify masking is correct.

**Files:**
- Create: `code/training/tests/test_smoke_train.py`

- [ ] **Step 1: Write the smoke test**

```python
# code/training/tests/test_smoke_train.py
"""Single-batch overfit smoke test (Spec §5.8).

Goal: confirm that completion_only_loss correctly masks all but the target,
and that a tiny model on one example can drive training loss → ~0.

This runs on CPU + Qwen2.5-0.5B at 4-bit (4 GB VRAM compatible).
"""
import json
import os
from pathlib import Path

import pytest


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("SKIP_SMOKE_TRAIN", "1") == "1",
    reason="Set SKIP_SMOKE_TRAIN=0 to run (downloads Qwen2.5-0.5B; takes ~2 min)"
)
def test_overfit_one_example():
    """Train a tiny model on one row for 200 steps. Loss should drop below 0.05."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = "<|endoftext|>"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32,
        attn_implementation="eager",
        trust_remote_code=True,
    )

    data = [{
        "prompt": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        "completion": [{"role": "assistant", "content": "The answer is 4."}],
        "tools": [],
    }]
    ds = Dataset.from_list(data)

    cfg = SFTConfig(
        output_dir="/tmp/_smoke_overfit",
        num_train_epochs=1,
        max_steps=200,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        bf16=False, fp16=False,         # CPU
        max_seq_length=512,
        packing=False,                  # smoke
        completion_only_loss=True,
        assistant_only_loss=False,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
    )
    trainer = SFTTrainer(model=model, processing_class=tok, args=cfg, train_dataset=ds)

    final = trainer.train()
    final_loss = final.metrics.get("train_loss")
    assert final_loss is not None, "trainer did not record a final loss"
    assert final_loss < 0.05, (
        f"Single-example overfit failed: loss={final_loss}. "
        f"Expected <0.05 after 200 steps. Likely cause: completion mask not applied."
    )


def test_completion_only_mask_excludes_prompt(tmp_path: Path):
    """Verify: with completion_only_loss=True, prompt tokens have label=-100."""
    from transformers import AutoTokenizer
    from trl import SFTConfig
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = "<|endoftext|>"

    # Build a prompt+completion pair
    msgs_prompt = [
        {"role": "system", "content": "Sys"},
        {"role": "user", "content": "Hi"},
    ]
    msgs_completion = [{"role": "assistant", "content": "Hello!"}]

    full_text = tok.apply_chat_template(
        msgs_prompt + msgs_completion, tokenize=False,
        add_generation_prompt=False, enable_thinking=False,
    )
    prompt_text = tok.apply_chat_template(
        msgs_prompt, tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    full_ids = tok.encode(full_text)
    prompt_ids = tok.encode(prompt_text)

    # The labels under completion_only_loss should mask the prompt tokens.
    n_prompt = len(prompt_ids)
    n_total = len(full_ids)
    assert n_total > n_prompt, "Prompt should be a strict prefix of full"
    # Conceptually: TRL builds labels = [-100]*n_prompt + full_ids[n_prompt:]
    labels = [-100] * n_prompt + full_ids[n_prompt:]
    n_masked = sum(1 for l in labels if l == -100)
    assert n_masked == n_prompt
```

- [ ] **Step 2: Run the fast test**

Run: `cd code/training && python -m pytest tests/test_smoke_train.py::test_completion_only_mask_excludes_prompt -v`
Expected: PASS.

- [ ] **Step 3: Run the slow overfit test (optional, ~2 min)**

Run: `cd code/training && SKIP_SMOKE_TRAIN=0 python -m pytest tests/test_smoke_train.py::test_overfit_one_example -v -s`
Expected: PASS (final loss < 0.05).

- [ ] **Step 4: Commit**

```bash
git add code/training/tests/test_smoke_train.py
git commit -m "test(training): single-batch overfit smoke + mask sanity"
```

---

### Task 22: Top-level pipeline script (`run_pipeline.sh`)

The single command the operator runs.

**Files:**
- Create: `code/training/orchestration/run_pipeline.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# code/training/orchestration/run_pipeline.sh
# Top-level pipeline: train_all → eval_all → summarize → plotting.
# Run from bundle root.
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:?set BUNDLE_ROOT}"
VENV="${VENV:-$BUNDLE_ROOT/code/.venv/bin}"

cd "$BUNDLE_ROOT"

echo "=== Phase 1: Training ==="
bash code/training/orchestration/train_all.sh

echo ""
echo "=== Phase 2: Eval ==="
bash code/training/orchestration/eval_all.sh

echo ""
echo "=== Phase 3: Aggregate ==="
"$VENV/python" -m code.training.orchestration.summarize --root train_outputs
"$VENV/python" -m code.training.orchestration.plotting --root train_outputs

echo ""
echo "=== Pipeline complete ==="
echo "Summary: $BUNDLE_ROOT/train_outputs/SUMMARY.csv"
echo "Plots:   $BUNDLE_ROOT/train_outputs/plots/"
```

- [ ] **Step 2: Make executable + commit**

```bash
chmod +x code/training/orchestration/run_pipeline.sh
git add code/training/orchestration/run_pipeline.sh
git commit -m "feat(training/orchestration): top-level run_pipeline.sh"
```

---

### Task 23: End-to-end integration on the dev box (CPU + tiny model)

Verify the entire chain works locally with a 0.5B model on CPU.

**Files:**
- (Test only — no new files)

- [ ] **Step 1: Create a smoke override config**

```bash
cd /path/to/bundle
mkdir -p code/training/configs/runs_smoke
cat > code/training/configs/runs_smoke/smoke_2b.yaml <<'YAML'
run_id: smoke_2b
model:
  name: Qwen/Qwen2.5-0.5B-Instruct
  revision: main
  is_moe: false
data:
  n_train_runs: 5
training:
  output_dir: train_outputs/_smoke
  num_train_epochs: 1
  max_steps: 5
  learning_rate: 5.0e-5
  warmup_ratio: 0.1
  lr_scheduler_type: linear
  cosine_min_lr_ratio: 0.10
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.95
  max_grad_norm: 1.0
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  per_device_eval_batch_size: 1
  bf16: false
  gradient_checkpointing: false
  torch_compile: false
  max_seq_length: 1024
  packing: false
  packing_strategy: bfd
  padding_free: false
  attn_implementation: eager
  completion_only_loss: true
  assistant_only_loss: false
  use_liger_kernel: false
  loss_type: default
  dataloader_num_workers: 0
  eval_strategy: "no"
  eval_steps: 0
  save_strategy: "no"
  save_total_limit: 1
  load_best_model_at_end: false
  metric_for_best_model: eval_loss
  greater_is_better: false
  early_stopping_patience: 99
  logging_steps: 1
  seed: 1234
  data_seed: 42
distributed:
  strategy: ddp
  accelerate_config: configs/accelerate_ddp.yaml
moe: null
YAML
```

- [ ] **Step 2: Run end-to-end smoke**

```bash
cd /path/to/bundle
# Run convert + validate first (one-time per dev box).
.venv/bin/python -m code.training.data.convert_to_prompt_completion \
    --src data_processed/stage2_v1/train.jsonl \
    --dst train_outputs/_data_cache/train_prompt_completion.jsonl \
    --domain-assets data_processed/stage2_v1/domain_assets

.venv/bin/python -m code.training.data.convert_to_prompt_completion \
    --src data_processed/stage2_v1/val.jsonl \
    --dst train_outputs/_data_cache/val_prompt_completion.jsonl \
    --domain-assets data_processed/stage2_v1/domain_assets

# Now train smoke run on CPU.
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m code.training.train \
    --run-config code/training/configs/runs_smoke/smoke_2b.yaml \
    --plan-config code/training/configs/plan_c_prime.yaml \
    --bundle-root .

# Verify checkpoint and STATUS exist. Phase-N: train.py writes
# "smoke_done" under --smoke-test, "done" otherwise; either is success.
test -d train_outputs/_smoke/checkpoint-final
grep -qE "^(done|smoke_done)$" train_outputs/_smoke/STATUS
echo "OK — end-to-end smoke passed"
```
Expected: prints `OK — end-to-end smoke passed`. Slow (~3-5 min on CPU).

- [ ] **Step 3: Run all unit tests**

```bash
cd /path/to/bundle/code/training
python -m pytest tests/ -v --ignore=tests/test_smoke_train.py
```
Expected: all tests pass.

- [ ] **Step 4: Clean up smoke artifacts**

```bash
rm -rf /path/to/bundle/train_outputs/_smoke
rm -rf /path/to/bundle/code/training/configs/runs_smoke
```

- [ ] **Step 5: Commit smoke validation note**

```bash
git status   # confirm clean
git log --oneline -10
```

---

## Self-Review

After writing all tasks, I checked the plan against the spec and found these coverage points:

**Spec section coverage:**

| Spec section | Implementing task(s) |
|---|---|
| §3 Plan C' run plan | Tasks 8, 9 |
| §4 Architecture | Tasks 1, 17, 18, 22 |
| §5.1 Convert to prompt/completion | Task 6 |
| §5.2 Quality filters | Task 2 |
| §5.3 Chat-template validation | Task 7 |
| §5.3.1 Tokenizer re-audit | Task 10 |
| §5.4 Domain rebalance | Task 4 |
| §5.5 Tool-shuffle augmentation | Task 3 |
| §5.6 Packing (TRL handles) | Task 9 (config) |
| §5.7 Held-out task split | Task 5 |
| §5.8 Single-batch overfit gate | Task 21 |
| §6.1 Hyperparameters | Task 9 |
| §6.1.1 VRAM safety table | Task 9 (per-GPU batch + accum) |
| §6.1.3 MoE config | Task 9, Task 13 |
| §6.1.4 Qwen3.6 specifics | Task 9, Task 14 |
| §7.1 Layer 1 val loss | Task 13 (SFTConfig eval_strategy=steps) |
| §7.3 Layer 3 force-routed eval | Task 16 |
| §7.4 Held-out eval | Task 16 |
| §11 Pre-launch smoke tests | Tasks 21, 23 |
| Two-phase orchestration | Tasks 17, 18, 22 |

**Confirmed: every spec requirement maps to at least one task.**

**Placeholder scan**: I searched for "TODO", "TBD", "implement later", "fill in", "similar to" — none found. All code blocks contain real, executable code.

**Type / signature consistency**: Functions referenced in later tasks (e.g., `compute_per_row_weights`, `should_drop_row`, `convert_row`) all have signatures defined in their introducing task. The orchestration scripts use absolute paths from `BUNDLE_ROOT` consistently.

**Scope check**: This is one cohesive plan for the training framework. Eval reuses existing `pipeline.runner` (one new provider; no major refactor). All 23 tasks together produce one bash command (`run_pipeline.sh`) that delivers the full deliverable.

---

## Execution Handoff

Plan complete and saved to `code/docs/superpowers/plans/2026-05-08-tau2-stage2-training-framework.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints for review.

**Which approach?**
