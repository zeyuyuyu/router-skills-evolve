"""Per-task artifact paths + read/write helpers.

Centralizes the task-first per-seed on-disk layout:

    data/<adapter>/<subset>/
      tasks/
        <task_id>/
          seed_<n>/
            phase0_baseline.json
            phase0_baseline_confirm.json    (--baseline-confirm-runs >= 2)
            phase0_baseline_retry.json      (--baseline-max-attempts >= 2,
                                             written only if attempt 1 failed)
            phase1_repriced.json
            phase3_attempts.json
            phase3_partial.json              (mid-flight checkpoint)
            emitted.ok                       Phase 4 idempotency marker
      samples.jsonl                          append-only SFT dataset
      progress.json                          per-(task, seed) summary index

Each (task, seed) is an independent run. Different seeds for the same
task live in sibling `seed_<n>/` subdirs and never overwrite each other.
`samples.jsonl` and `progress.json` aggregate across all tasks AND seeds.

Every save/load uses Pydantic's strict round-trip; schema drift fails loudly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.schemas.artifacts import (
    BaselineArtifact,
    BaselineResult,
    ExplorationLog,
    Phase3Checkpoint,
)


@dataclass(frozen=True)
class TaskArtifactPaths:
    task_dir: Path             # tasks/<id>/  — shared across seeds
    seed_dir: Path             # tasks/<id>/seed_<n>/  — this run's home
    baseline: Path
    baseline_confirm: Path
    baseline_retry: Path        # only used when --baseline-max-attempts >= 2
    repriced: Path
    phase3: Path
    phase3_partial: Path       # mid-flight Phase-3 checkpoint; deleted on completion
    supervision: Path          # shared at subset root, across all (task, seed)
    emitted_marker: Path       # per-(task, seed), lives inside seed_dir


def domain_root(data_dir: Path, adapter: str, subset: str) -> Path:
    return data_dir / adapter / subset


def artifact_paths(
    data_dir: Path, adapter: str, subset: str, *, task_id: str, seed: int
) -> TaskArtifactPaths:
    root = domain_root(data_dir, adapter, subset)
    task_dir = root / "tasks" / task_id
    seed_dir = task_dir / f"seed_{seed}"
    return TaskArtifactPaths(
        task_dir=task_dir,
        seed_dir=seed_dir,
        baseline=seed_dir / "phase0_baseline.json",
        baseline_confirm=seed_dir / "phase0_baseline_confirm.json",
        baseline_retry=seed_dir / "phase0_baseline_retry.json",
        repriced=seed_dir / "phase1_repriced.json",
        phase3=seed_dir / "phase3_attempts.json",
        phase3_partial=seed_dir / "phase3_partial.json",
        supervision=root / "samples.jsonl",
        emitted_marker=seed_dir / "emitted.ok",
    )


def _write_json(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def save_baseline(art: BaselineArtifact, path: Path) -> None:
    _write_json(path, art.model_dump_json(indent=2))


def load_baseline(path: Path) -> BaselineArtifact:
    return BaselineArtifact.model_validate_json(path.read_text())


def save_baseline_repriced(r: BaselineResult, path: Path) -> None:
    _write_json(path, r.model_dump_json(indent=2))


def load_baseline_repriced(path: Path) -> BaselineResult:
    return BaselineResult.model_validate_json(path.read_text())


def save_exploration(e: ExplorationLog, path: Path) -> None:
    _write_json(path, e.model_dump_json(indent=2))


def load_exploration(path: Path) -> ExplorationLog:
    return ExplorationLog.model_validate_json(path.read_text())


def save_phase3_checkpoint(c: Phase3Checkpoint, path: Path) -> None:
    _write_json(path, c.model_dump_json(indent=2))


def load_phase3_checkpoint(path: Path) -> Phase3Checkpoint:
    return Phase3Checkpoint.model_validate_json(path.read_text())
