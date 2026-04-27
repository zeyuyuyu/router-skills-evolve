#!/usr/bin/env python3
"""Import UncommonRoute benchmark tier labels as weak router data.

The UncommonRoute benchmark labels prompts by tier:
  SIMPLE / MEDIUM / COMPLEX / REASONING

For this repo's binary learned router we map:
  SIMPLE, MEDIUM      -> small
  COMPLEX, REASONING  -> large

The output JSONL can be mixed with real traces through
``train_learnable_router.py --router-data``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.learned_router.data import LABEL_TO_ROUTE, ROUTE_LARGE, ROUTE_SMALL


DEFAULT_FILES = ["bench/data/all.jsonl"]
DEFAULT_REPO = "CommonstackAI/UncommonRoute"


def tier_to_label(tier: str) -> int:
    normalized = str(tier or "").strip().upper()
    if normalized in {"SIMPLE", "MEDIUM"}:
        return ROUTE_SMALL
    if normalized in {"COMPLEX", "REASONING"}:
        return ROUTE_LARGE
    raise ValueError(f"Unsupported expected_tier: {tier!r}")


def download_via_gh(repo: str, path: str) -> str:
    """Fetch a file from GitHub using gh when available."""
    try:
        url = subprocess.check_output(
            ["gh", "api", f"repos/{repo}/contents/{path}", "--jq", ".download_url"],
            text=True,
        ).strip()
    except Exception as exc:
        raise RuntimeError(
            "Failed to resolve GitHub download URL. Ensure gh is installed/authenticated "
            f"or pass a local file path. repo={repo} path={path}"
        ) from exc
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def iter_jsonl_text(text: str) -> Iterable[Dict]:
    for line in text.splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def load_source(repo: str, path_or_file: str, root: Path) -> List[Dict]:
    path = Path(path_or_file)
    if not path.is_absolute():
        local_path = root / path
    else:
        local_path = path
    if local_path.exists():
        with open(local_path) as f:
            return [json.loads(line) for line in f if line.strip()]
    return list(iter_jsonl_text(download_via_gh(repo, path_or_file)))


def convert_rows(rows: Iterable[Dict], source_name: str, *, prefix: str) -> List[Dict]:
    converted: List[Dict] = []
    for idx, row in enumerate(rows):
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue
        tier = row.get("expected_tier", "")
        try:
            label = tier_to_label(tier)
        except ValueError:
            continue
        converted.append(
            {
                "task_id": f"{prefix}/{idx}",
                "prompt": prompt,
                "label": label,
                "label_name": LABEL_TO_ROUTE[label],
                "signature": "",
                "source_trace": source_name,
                "small_success": label == ROUTE_SMALL,
                "large_success": True if label == ROUTE_LARGE else None,
                "tier": str(tier).upper(),
                "category": row.get("category", ""),
                "lang": row.get("lang", ""),
                "weak_label": True,
            }
        )
    return converted


def dedupe_examples(examples: Iterable[Dict]) -> List[Dict]:
    seen = set()
    deduped: List[Dict] = []
    for example in examples:
        key = example["prompt"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(example)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Import UncommonRoute bench data as router labels")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo owner/name")
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
        help="UncommonRoute bench JSONL file(s), GitHub paths or local paths",
    )
    parser.add_argument(
        "--output",
        default="data/router_training/uncommonroute_bench.jsonl",
        help="Output router-data JSONL",
    )
    parser.add_argument("--no-dedupe", action="store_true", help="Keep duplicate prompts")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    all_examples: List[Dict] = []
    for source in args.files:
        rows = load_source(args.repo, source, root)
        prefix = f"uncommonroute/{Path(source).stem}"
        converted = convert_rows(rows, f"{args.repo}/{source}", prefix=prefix)
        print(f"{source}: loaded {len(rows)} rows -> {len(converted)} router examples")
        all_examples.extend(converted)

    if not args.no_dedupe:
        before = len(all_examples)
        all_examples = dedupe_examples(all_examples)
        print(f"dedupe: {before} -> {len(all_examples)}")

    counts = {"small": 0, "large": 0}
    for example in all_examples:
        counts[example["label_name"]] += 1

    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"saved: {output}")
    print(f"counts: {counts}")


if __name__ == "__main__":
    main()
