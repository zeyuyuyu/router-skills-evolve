"""Standalone Phase-4 reassembly from saved per-(task, seed) artifacts.

CLI:
    python -m pipeline.assemble --adapter tau2_bench --subset retail [--clean]

No API calls. Pure disk operation. Walks `tasks/<id>/seed_<n>/` directories,
loads `phase1_repriced.json` + `phase3_attempts.json`, and reruns Phase 4
once per (task, seed) pair. Use --clean to wipe `samples.jsonl` and per-run
`emitted.ok` markers before rebuilding (recommended after upstream
artifact changes).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from core.configs.loader import load_pricing, load_tier_pool
from pipeline.io import (
    artifact_paths, domain_root,
    load_baseline_repriced, load_exploration,
)
from pipeline.phases.phase4_assembly import run_phase4

_SEED_DIR_RE = re.compile(r"^seed_(\d+)$")


def _task_ids(root: Path) -> list[str]:
    tasks_dir = root / "tasks"
    if not tasks_dir.exists():
        return []
    return sorted(d.name for d in tasks_dir.iterdir() if d.is_dir())


def _seeds_for_task(root: Path, task_id: str) -> list[int]:
    """Sorted list of seeds that have a seed_<n>/ dir under this task."""
    task_dir = root / "tasks" / task_id
    if not task_dir.exists():
        return []
    seeds: list[int] = []
    for d in task_dir.iterdir():
        if not d.is_dir():
            continue
        m = _SEED_DIR_RE.match(d.name)
        if m:
            seeds.append(int(m.group(1)))
    return sorted(seeds)


def reassemble_subset(
    data_dir: Path, adapter: str, subset: str, *, clean: bool = False,
) -> int:
    """Run Phase 4 over every saved (task, seed). Returns # records emitted."""
    root = domain_root(data_dir, adapter, subset)
    sup = root / "samples.jsonl"
    if clean:
        if sup.exists():
            sup.unlink()
        for task_id in _task_ids(root):
            for seed in _seeds_for_task(root, task_id):
                paths = artifact_paths(data_dir, adapter, subset,
                                       task_id=task_id, seed=seed)
                if paths.emitted_marker.exists():
                    paths.emitted_marker.unlink()

    repo = Path(__file__).resolve().parents[1]
    tier_pool = load_tier_pool(
        repo / "core" / "configs" / "tier_pools" / f"{adapter}.yaml"
    )
    pricing = load_pricing(repo / "core" / "configs" / "pricing.yaml")

    n = 0
    for task_id in _task_ids(root):
        for seed in _seeds_for_task(root, task_id):
            paths = artifact_paths(data_dir, adapter, subset,
                                   task_id=task_id, seed=seed)
            if not (paths.repriced.exists() and paths.phase3.exists()):
                continue
            repriced = load_baseline_repriced(paths.repriced)
            exploration = load_exploration(paths.phase3)
            records = run_phase4(
                baseline=repriced, analysis=None, exploration=exploration,
                adapter_name=adapter, subset=subset,
                supervision_path=paths.supervision,
                emitted_marker=paths.emitted_marker,
                baseline_model=tier_pool.baseline_model,
                baseline_tier=tier_pool.baseline_tier,
                pricing=pricing,
                seed=seed,
            )
            n += len(records)
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", default="tau2_bench")
    p.add_argument("--subset", required=True)
    p.add_argument("--data-dir", default="data", type=Path)
    p.add_argument("--clean", action="store_true",
                   help="Delete samples.jsonl and per-run emitted.ok markers first.")
    args = p.parse_args()
    n = reassemble_subset(args.data_dir, args.adapter, args.subset,
                          clean=args.clean)
    print(f"Emitted {n} supervision records to "
          f"{args.data_dir}/{args.adapter}/{args.subset}/samples.jsonl")


if __name__ == "__main__":
    main()
