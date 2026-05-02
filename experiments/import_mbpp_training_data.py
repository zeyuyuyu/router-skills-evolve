#!/usr/bin/env python3
"""Import MBPP as SFT/eval data with executable tests.

The existing ``data/training_data.jsonl`` only contains a few hard HumanEval
examples. MBPP provides many small Python tasks with reference code and tests,
which makes it a practical next dataset for the model-evolution track.

Output rows use the same shape as ``extract_training_data.py``:

    instruction / input / output / test / entry_point

so they can be consumed by ``train_small_model.py`` and
``evaluate_finetuned_model.py``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def infer_entry_point(code: str, tests: Iterable[str]) -> Optional[str]:
    match = re.search(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code, re.MULTILINE)
    if match:
        return match.group(1)
    for test in tests:
        match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", test)
        if match and match.group(1) not in {"assert", "all", "any", "len", "range"}:
            return match.group(1)
    return None


def build_check_function(entry_point: str, tests: Iterable[str], setup_code: str = "") -> str:
    body = []
    for raw in tests:
        test = str(raw).strip()
        if not test:
            continue
        test = re.sub(rf"\b{re.escape(entry_point)}\s*\(", "candidate(", test)
        body.append(f"    {test}")
    if not body:
        body.append("    pass")

    parts = []
    if setup_code.strip():
        parts.append(setup_code.strip())
        parts.append("")
    parts.append("def check(candidate):")
    parts.extend(body)
    return "\n".join(parts)


def row_to_sample(row: Dict, source_split: str) -> Optional[Dict]:
    code = str(row.get("code") or "").strip()
    prompt = str(row.get("text") or row.get("prompt") or "").strip()
    tests = row.get("test_list") or row.get("tests") or []
    if isinstance(tests, str):
        tests = [tests]
    setup_code = str(row.get("test_setup_code") or "").strip()
    entry_point = infer_entry_point(code, tests)
    if not code or not prompt or not entry_point or not tests:
        return None

    instruction = (
        f"Write a Python function named `{entry_point}` for the following task:\n"
        f"{prompt}"
    )
    task_id = row.get("task_id", "")
    return {
        "task_id": f"MBPP/{task_id}" if task_id != "" else "",
        "instruction": instruction,
        "input": "",
        "output": code,
        "test": build_check_function(entry_point, tests, setup_code),
        "entry_point": entry_point,
        "source": "mbpp",
        "source_split": source_split,
    }


def load_split(dataset_name: str, config: Optional[str], split: str):
    from datasets import load_dataset

    if config:
        return load_dataset(dataset_name, config, split=split)
    return load_dataset(dataset_name, split=split)


def convert_split(dataset_name: str, config: Optional[str], split: str, limit: Optional[int]) -> List[Dict]:
    dataset = load_split(dataset_name, config, split)
    samples: List[Dict] = []
    for row in dataset:
        sample = row_to_sample(dict(row), split)
        if sample is None:
            continue
        samples.append(sample)
        if limit is not None and len(samples) >= limit:
            break
    return samples


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import MBPP into local SFT/eval JSONL")
    parser.add_argument("--dataset", default="google-research-datasets/mbpp")
    parser.add_argument("--config", default="sanitized", help="Dataset config; pass '' for none")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--train-output", default="data/mbpp/mbpp_train.jsonl")
    parser.add_argument("--eval-output", default="data/mbpp/mbpp_eval.jsonl")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    config = args.config or None

    train_samples = convert_split(args.dataset, config, args.train_split, args.max_train)
    eval_samples = convert_split(args.dataset, config, args.eval_split, args.max_eval)

    print(f"train: {len(train_samples)} samples from {args.train_split}")
    print(f"eval : {len(eval_samples)} samples from {args.eval_split}")
    if train_samples:
        print("example:")
        print(json.dumps(train_samples[0], indent=2)[:1000])

    if args.dry_run:
        print("--dry-run: skipped writing files")
        return

    train_output = Path(args.train_output)
    eval_output = Path(args.eval_output)
    if not train_output.is_absolute():
        train_output = root / train_output
    if not eval_output.is_absolute():
        eval_output = root / eval_output
    write_jsonl(train_output, train_samples)
    write_jsonl(eval_output, eval_samples)
    print(f"saved train: {train_output}")
    print(f"saved eval : {eval_output}")


if __name__ == "__main__":
    main()
