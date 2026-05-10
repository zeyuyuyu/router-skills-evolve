#!/usr/bin/env python3
"""Build a unified dataset bundle for router/model evolution experiments.

The goal is to stop relying on a tiny 20-example smoke test. This builder
creates a reproducible dataset layout with:

- code tasks: prompt + reference code + executable tests
- router tasks: prompt + small/large route labels or weak tier labels
- summary stats and a dataset card

The generated data is intentionally ignored by git; commit the schema, scripts,
stats, and small samples rather than large generated artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.import_mbpp_training_data import row_to_sample as mbpp_row_to_sample


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UNCOMMONROUTE_REPO = "CommonstackAI/UncommonRoute"
DEFAULT_UNCOMMONROUTE_FILES = ["bench/data/all.jsonl"]
ROUTE_SMALL = 0
ROUTE_LARGE = 1
LABEL_TO_ROUTE = {ROUTE_SMALL: "small", ROUTE_LARGE: "large"}


def stable_id(*parts: str) -> str:
    raw = "\n".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_rows(rows: List[Dict], train_ratio: float, dev_ratio: float, seed: int) -> Dict[str, List[Dict]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    return {
        "train": shuffled[:n_train],
        "dev": shuffled[n_train:n_train + n_dev],
        "test": shuffled[n_train + n_dev:],
    }


def tier_to_label(tier: str) -> int:
    normalized = str(tier or "").strip().upper()
    if normalized in {"SIMPLE", "MEDIUM"}:
        return ROUTE_SMALL
    if normalized in {"COMPLEX", "REASONING"}:
        return ROUTE_LARGE
    raise ValueError(f"Unsupported expected_tier: {tier!r}")


def infer_signature(prompt: str) -> str:
    """Local copy of the lightweight signature heuristic.

    Avoid importing src.* here so dataset builds do not require API dependencies.
    """
    p = prompt.lower()
    length = len(prompt)
    if length < 200:
        len_b = "S"
    elif length < 500:
        len_b = "M"
    else:
        len_b = "L"

    tags = []
    if any(k in p for k in ["list", "array"]):
        tags.append("list")
    if any(k in p for k in ["string", "str"]):
        tags.append("str")
    if any(k in p for k in ["number", "integer", "float"]):
        tags.append("num")
    if any(k in p for k in ["sort", "order"]):
        tags.append("sort")
    if any(k in p for k in ["prime", "factor", "divisor"]):
        tags.append("theory")
    if any(k in p for k in ["encode", "decode", "cipher", "cyclic", "shift"]):
        tags.append("crypto")
    if any(k in p for k in ["poly", "recursion", "recursive"]):
        tags.append("advanced")
    if any(k in p for k in ["true", "false", "check", "boolean"]):
        tags.append("bool")
    if not tags:
        tags.append("general")
    return f"{len_b}|{'/'.join(sorted(set(tags)))}"


def humaneval_rows(path: Path) -> List[Dict]:
    rows = []
    for row in read_jsonl(path):
        prompt = row["prompt"]
        reference = prompt + row.get("canonical_solution", "")
        rows.append(
            {
                "id": row.get("task_id") or f"humaneval/{stable_id(prompt)}",
                "source": "humaneval",
                "source_split": "all",
                "task_type": "code_generation",
                "prompt": prompt,
                "instruction": prompt,
                "input": "",
                "reference_solution": reference,
                "output": reference,
                "entry_point": row.get("entry_point", ""),
                "test": row.get("test", ""),
                "has_tests": bool(row.get("test")),
                "has_reference": bool(row.get("canonical_solution")),
                "signature": infer_signature(prompt),
                "metadata": {"original_task_id": row.get("task_id", "")},
            }
        )
    return rows


def hard_trace_rows(path: Path) -> List[Dict]:
    rows = []
    if not path.exists():
        return rows
    for row in read_jsonl(path):
        prompt = row.get("instruction") or row.get("prompt") or ""
        output = row.get("output") or row.get("reference_solution") or ""
        if not prompt or not output:
            continue
        rows.append(
            {
                "id": row.get("task_id") or f"hard/{stable_id(prompt, output)}",
                "source": "hard_traces",
                "source_split": "all",
                "task_type": "code_generation",
                "prompt": prompt,
                "instruction": prompt,
                "input": row.get("input", ""),
                "reference_solution": output,
                "output": output,
                "entry_point": row.get("entry_point", ""),
                "test": row.get("test", ""),
                "has_tests": bool(row.get("test")),
                "has_reference": True,
                "signature": row.get("signature") or infer_signature(prompt),
                "metadata": {
                    "source_model": row.get("source_model", ""),
                    "generation_cost_usd": row.get("generation_cost_usd", 0.0),
                },
            }
        )
    return rows


def load_dataset_split(dataset_name: str, config: Optional[str], split: str):
    from datasets import load_dataset

    if config:
        return load_dataset(dataset_name, config, split=split)
    return load_dataset(dataset_name, split=split)


def mbpp_rows(
    dataset_name: str,
    config: Optional[str],
    splits: Iterable[str],
    max_per_split: Optional[int],
) -> List[Dict]:
    rows = []
    for split in splits:
        dataset = load_dataset_split(dataset_name, config, split)
        count = 0
        for raw in dataset:
            sample = mbpp_row_to_sample(dict(raw), split)
            if not sample:
                continue
            prompt = sample["instruction"]
            rows.append(
                {
                    "id": sample.get("task_id") or f"mbpp/{split}/{stable_id(prompt)}",
                    "source": "mbpp",
                    "source_split": split,
                    "task_type": "code_generation",
                    "prompt": prompt,
                    "instruction": prompt,
                    "input": sample.get("input", ""),
                    "reference_solution": sample.get("output", ""),
                    "output": sample.get("output", ""),
                    "entry_point": sample.get("entry_point", ""),
                    "test": sample.get("test", ""),
                    "has_tests": bool(sample.get("test")),
                    "has_reference": bool(sample.get("output")),
                    "signature": infer_signature(prompt),
                    "metadata": {},
                }
            )
            count += 1
            if max_per_split is not None and count >= max_per_split:
                break
    return rows


def raw_github_text(repo: str, path: str) -> str:
    try:
        url = subprocess.check_output(
            ["gh", "api", f"repos/{repo}/contents/{path}", "--jq", ".download_url"],
            text=True,
        ).strip()
    except Exception:
        url = f"https://raw.githubusercontent.com/{repo}/main/{path}"
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def load_uncommonroute_source(repo: str, path: str) -> List[Dict]:
    local = ROOT / path
    if local.exists():
        return read_jsonl(local)
    text = raw_github_text(repo, path)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def uncommonroute_rows(repo: str, files: Iterable[str], max_rows: Optional[int]) -> List[Dict]:
    rows = []
    for file_path in files:
        source_rows = load_uncommonroute_source(repo, file_path)
        for idx, row in enumerate(source_rows):
            prompt = str(row.get("prompt", "")).strip()
            tier = str(row.get("expected_tier", "")).strip().upper()
            if not prompt or not tier:
                continue
            try:
                label = tier_to_label(tier)
            except ValueError:
                continue
            rows.append(
                {
                    "id": f"uncommonroute/{Path(file_path).stem}/{idx}",
                    "source": "uncommonroute_bench",
                    "source_split": Path(file_path).stem,
                    "task_type": "router_supervision",
                    "prompt": prompt,
                    "router_label": int(label),
                    "router_label_name": LABEL_TO_ROUTE[int(label)],
                    "tier": tier,
                    "category": row.get("category", ""),
                    "lang": row.get("lang", ""),
                    "weak_label": True,
                    "signature": infer_signature(prompt),
                    "metadata": {"source_file": file_path},
                }
            )
            if max_rows is not None and len(rows) >= max_rows:
                return rows
    return rows


def dedupe_by_key(rows: Iterable[Dict], key_fields: List[str]) -> List[Dict]:
    seen = set()
    out = []
    for row in rows:
        key = tuple(row.get(field, "") for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def stats_for(rows: List[Dict]) -> Dict:
    stats = {
        "total": len(rows),
        "by_source": Counter(row.get("source", "") for row in rows),
        "by_source_split": Counter(f"{row.get('source')}:{row.get('source_split')}" for row in rows),
        "by_task_type": Counter(row.get("task_type", "") for row in rows),
        "has_tests": sum(1 for row in rows if row.get("has_tests")),
        "has_reference": sum(1 for row in rows if row.get("has_reference")),
    }
    router_labels = Counter(row.get("router_label_name", "") for row in rows if "router_label_name" in row)
    if router_labels:
        stats["router_labels"] = router_labels
    return json.loads(json.dumps(stats, default=dict))


def write_dataset_card(path: Path, stats: Dict, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Evolver Dataset

This dataset bundle is built by `experiments/build_evolver_dataset.py`.

## Purpose

The bundle is meant for EMNLP-style evolver experiments where router learning
and model learning are evaluated together. It separates executable code tasks
from router-only weak supervision so experiments can report clearly what signal
is used.

## Schema

### Code tasks

- `id`
- `source`, `source_split`
- `task_type = code_generation`
- `prompt` / `instruction`
- `reference_solution` / `output`
- `entry_point`
- `test`
- `has_tests`, `has_reference`
- `signature`
- `metadata`

### Router tasks

- `id`
- `source`, `source_split`
- `task_type = router_supervision`
- `prompt`
- `router_label`, `router_label_name`
- `tier`, `category`, `lang`
- `weak_label`
- `signature`

## Statistics

```json
{json.dumps(stats, indent=2, ensure_ascii=False)}
```

## Build arguments

```json
{json.dumps(vars(args), indent=2, ensure_ascii=False)}
```

## Notes

- UncommonRoute bench labels are weak router labels, not executable code tasks.
- MBPP and HumanEval contain executable tests and references.
- Existing `data/training_data.jsonl` hard traces are high-value but small.
"""
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified evolver dataset bundle")
    parser.add_argument("--output-dir", default="data/evolver_dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)

    parser.add_argument("--include-humaneval", action="store_true", default=True)
    parser.add_argument("--humaneval-path", default="data/HumanEval.jsonl")
    parser.add_argument("--include-hard-traces", action="store_true", default=True)
    parser.add_argument("--hard-traces-path", default="data/training_data.jsonl")

    parser.add_argument("--include-mbpp", action="store_true", default=True)
    parser.add_argument("--mbpp-dataset", default="google-research-datasets/mbpp")
    parser.add_argument("--mbpp-config", default="sanitized")
    parser.add_argument("--mbpp-splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--max-mbpp-per-split", type=int, default=None)

    parser.add_argument("--include-uncommonroute", action="store_true", default=True)
    parser.add_argument("--uncommonroute-repo", default=DEFAULT_UNCOMMONROUTE_REPO)
    parser.add_argument("--uncommonroute-files", nargs="+", default=DEFAULT_UNCOMMONROUTE_FILES)
    parser.add_argument("--max-uncommonroute", type=int, default=None)

    parser.add_argument("--skip-external", action="store_true", help="Use only local files")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    code_rows: List[Dict] = []
    router_rows: List[Dict] = []

    if args.include_humaneval:
        path = ROOT / args.humaneval_path
        if path.exists():
            code_rows.extend(humaneval_rows(path))
        else:
            print(f"skip HumanEval: missing {path}")

    if args.include_hard_traces:
        path = ROOT / args.hard_traces_path
        code_rows.extend(hard_trace_rows(path))

    if args.include_mbpp and not args.skip_external:
        code_rows.extend(
            mbpp_rows(
                args.mbpp_dataset,
                args.mbpp_config or None,
                args.mbpp_splits,
                args.max_mbpp_per_split,
            )
        )
    elif args.include_mbpp:
        print("skip MBPP: --skip-external")

    if args.include_uncommonroute and not args.skip_external:
        router_rows.extend(
            uncommonroute_rows(
                args.uncommonroute_repo,
                args.uncommonroute_files,
                args.max_uncommonroute,
            )
        )
    elif args.include_uncommonroute:
        print("skip UncommonRoute: --skip-external")

    code_rows = dedupe_by_key(code_rows, ["source", "id"])
    router_rows = dedupe_by_key(router_rows, ["prompt"])

    code_splits = split_rows(code_rows, args.train_ratio, args.dev_ratio, args.seed)
    router_splits = split_rows(router_rows, args.train_ratio, args.dev_ratio, args.seed)

    if args.dry_run:
        print("code", stats_for(code_rows))
        print("router", stats_for(router_rows))
        print("--dry-run: skipped writing files")
        return

    for split, rows in code_splits.items():
        write_jsonl(output_dir / "code_tasks" / f"{split}.jsonl", rows)
    for split, rows in router_splits.items():
        write_jsonl(output_dir / "router_tasks" / f"{split}.jsonl", rows)

    stats = {
        "code_tasks": stats_for(code_rows),
        "router_tasks": stats_for(router_rows),
        "splits": {
            "code_tasks": {split: len(rows) for split, rows in code_splits.items()},
            "router_tasks": {split: len(rows) for split, rows in router_splits.items()},
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    write_dataset_card(output_dir / "DATASET_CARD.md", stats, args)

    sample = {
        "code_task": code_rows[0] if code_rows else None,
        "router_task": router_rows[0] if router_rows else None,
    }
    with open(output_dir / "sample.json", "w") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)

    print(f"saved dataset bundle: {output_dir}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
