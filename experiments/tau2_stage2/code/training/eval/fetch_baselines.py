"""Pull phase0_baseline.json (and sibling files) from the two HF dataset backups
into a canonical layout under $BUNDLE_ROOT/eval_baselines/.

WHY:
    The portable bundle ships only `data_processed/stage2_v1/` (SFT corpus),
    NOT the raw per-(task, seed) baseline trajectories. The step-budget eval
    needs the actual `phase0_baseline.json` for each of the 35 eval tasks at
    each available seed:
      - 50 (task, seed) pairs live in `TonyYun/evol-llm-portable-data-2026-05-13`
        (single epoch: canonical_apr29_qwen_cs).
      - 13 (task, seed) pairs live in `TonyYun/evol-llm-data-backup-2026-05-13`
        under raw/may01_qwen_or/ (_qo), raw/may04_anthropic_or/ (_o4), or
        raw/may06_anthropic_or/ (_o6) — picked per the run_dirs suffix in
        eval_tasks.jsonl.

OUTPUT LAYOUT (canonical, consumed by step_budget_harness.py):
    $BUNDLE_ROOT/eval_baselines/<domain>/tasks/<task_id>/seed_<seed>/
        phase0_baseline.json
        phase1_repriced.json     (optional sibling — kept for posterity)
        phase3_attempts.json     (optional sibling)
        emitted.ok               (sentinel)

USAGE:
    HF_TOKEN=... BUNDLE_ROOT=/path/to/bundle \
        python -m training.eval.fetch_baselines
        [--bundle-root <path>]
        [--data-extra-root <path>]   # where the big-backup epoch dirs land
                                     # default: $BUNDLE_ROOT/data_extra/

Idempotent: re-running with all files present does nothing. Skips downloads
when the canonical file already exists.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Both HF datasets are PRIVATE — require HF_TOKEN.
_HF_REPO_SIDECAR = "TonyYun/evol-llm-portable-data-2026-05-13"
_HF_REPO_BACKUP = "TonyYun/evol-llm-data-backup-2026-05-13"

# Run-dirs suffix in eval_tasks.jsonl → epoch dir name in the LARGE backup.
# These are the epochs where each non-canonical baseline lives. The 13 missing
# (task, seed) pairs (see step-budget design doc) split:
#   _qo (qwen via OpenRouter)         → raw/may01_qwen_or/
#   _o4 (Claude Opus collected May 4) → raw/may04_anthropic_or/
#   _o6 (Claude Opus collected May 6) → raw/may06_anthropic_or/
_EPOCH_PATHS = {
    "_qo": "raw/may01_qwen_or",
    "_o4": "raw/may04_anthropic_or",
    "_o6": "raw/may06_anthropic_or",
}

# Per-task explicit HF path mapping for the 13 (task, seed) pairs missing from
# the sidecar dataset. Verified 2026-05-22 against the large backup repo's
# /api/datasets/.../tree endpoint. Key: (domain, task_id, seed). Value: the
# DIRECTORY path inside the HF repo containing phase0_baseline.json + siblings.
#
# These were chosen by matching the `run_dirs` suffix in eval_tasks.jsonl
# against the per-epoch backup structure:
#   - airline 17/301: only _qo recorded; pick the canonical shard_00.
#   - airline 25, 30, 48: only _qo, single shard per task.
#   - airline 27 (both seeds): _o4 → may04_anthropic_or.
#   - retail 24, 28 (seed 300): _o6 → may06_anthropic_or.
#   - retail 61, 83: _qo → may01_qwen_or.
_BACKUP_PATH_MAP: dict[tuple[str, str, int], str] = {
    ("airline", "17", 301): "raw/may01_qwen_or/data_or_airline_retry/shard_00/tau2_bench/airline/tasks/17/seed_301",
    ("airline", "25", 300): "raw/may01_qwen_or/data_or_airline_retry/shard_01/tau2_bench/airline/tasks/25/seed_300",
    ("airline", "25", 301): "raw/may01_qwen_or/data_or_airline_retry/shard_01/tau2_bench/airline/tasks/25/seed_301",
    ("airline", "27", 300): "raw/may04_anthropic_or/data_anthropic_2026-05-04/airline_fail/tau2_bench/airline/tasks/27/seed_300",
    ("airline", "27", 301): "raw/may04_anthropic_or/data_anthropic_2026-05-04/airline_fail/tau2_bench/airline/tasks/27/seed_301",
    ("airline", "30", 300): "raw/may01_qwen_or/data_or_airline_retry/shard_01/tau2_bench/airline/tasks/30/seed_300",
    ("airline", "30", 301): "raw/may01_qwen_or/data_or_airline_retry/shard_01/tau2_bench/airline/tasks/30/seed_301",
    ("airline", "48", 300): "raw/may01_qwen_or/data_or_airline_retry/shard_02/tau2_bench/airline/tasks/48/seed_300",
    ("retail",  "24", 300): "raw/may06_anthropic_or/retail/tau2_bench/retail/tasks/24/seed_300",
    ("retail",  "28", 300): "raw/may06_anthropic_or/retail/tau2_bench/retail/tasks/28/seed_300",
    ("retail",  "61", 300): "raw/may01_qwen_or/data_or_retail_retry/shard_02/tau2_bench/retail/tasks/61/seed_300",
    ("retail",  "61", 301): "raw/may01_qwen_or/data_or_retail_retry/shard_02/tau2_bench/retail/tasks/61/seed_301",
    ("retail",  "83", 300): "raw/may01_qwen_or/data_or_retail_retry/shard_03/tau2_bench/retail/tasks/83/seed_300",
}

_SIBLING_FILENAMES = (
    "phase0_baseline.json",
    "phase1_repriced.json",
    "phase3_attempts.json",
    "emitted.ok",
)


def _load_eval_tasks(eval_tasks_jsonl: Path) -> list[dict]:
    rows: list[dict] = []
    for line in eval_tasks_jsonl.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _canonical_dest(bundle_root: Path, domain: str, task_id: str, seed: int) -> Path:
    return bundle_root / "eval_baselines" / domain / "tasks" / task_id / f"seed_{seed}"


def _missing_files(dest: Path) -> list[str]:
    """Return the list of sibling filenames not yet present at dest."""
    if not dest.exists():
        return list(_SIBLING_FILENAMES)
    return [fn for fn in _SIBLING_FILENAMES if not (dest / fn).exists()]


def fetch_from_sidecar(
    *,
    hf_token: str,
    bundle_root: Path,
    eval_tasks: list[dict],
) -> tuple[int, int]:
    """Pull every (task, seed) pair from the sidecar HF dataset that's present
    there. Returns (n_fetched, n_skipped_existing)."""
    from huggingface_hub import snapshot_download

    # The sidecar dataset has both data/ and data_unified/ trees. We use
    # data_unified/ (canonical structure mirroring data_raw/.../data_unified/...
    # per per_row_provenance.jsonl source_paths).
    allow_patterns: list[str] = []
    needs: list[tuple[str, str, int]] = []
    for t in eval_tasks:
        dom, tid = t["domain"], t["task_id"]
        for seed in t.get("available_seeds", []):
            dest = _canonical_dest(bundle_root, dom, tid, seed)
            if not _missing_files(dest):
                continue  # already have all siblings
            needs.append((dom, tid, seed))
            for fn in _SIBLING_FILENAMES:
                allow_patterns.append(f"data_unified/tau2_bench/{dom}/tasks/{tid}/seed_{seed}/{fn}")

    if not allow_patterns:
        return (0, len(eval_tasks))

    sidecar_local = bundle_root / "data_extra" / "_hf_sidecar_cache"
    sidecar_local.mkdir(parents=True, exist_ok=True)

    print(f"[fetch_baselines] sidecar: snapshot_download for "
          f"{len(needs)} (task, seed) pairs ({len(allow_patterns)} files)")
    snapshot_download(
        repo_id=_HF_REPO_SIDECAR,
        repo_type="dataset",
        token=hf_token,
        local_dir=str(sidecar_local),
        allow_patterns=allow_patterns,
    )

    fetched = 0
    for dom, tid, seed in needs:
        src_dir = sidecar_local / "data_unified" / "tau2_bench" / dom / "tasks" / tid / f"seed_{seed}"
        if not src_dir.is_dir():
            print(f"[fetch_baselines] sidecar MISS: {dom}/{tid}/seed_{seed} not in sidecar tree "
                  f"(falling through to backup)", file=sys.stderr)
            continue
        dest = _canonical_dest(bundle_root, dom, tid, seed)
        dest.mkdir(parents=True, exist_ok=True)
        for fn in _SIBLING_FILENAMES:
            sp = src_dir / fn
            if sp.exists():
                shutil.copy2(sp, dest / fn)
                fetched += 1
    return (fetched, len(eval_tasks) * 2 - fetched)


def fetch_from_backup(
    *,
    hf_token: str,
    bundle_root: Path,
    eval_tasks: list[dict],
) -> int:
    """Pull the 13 (task, seed) pairs that live only in the big backup repo.

    Uses _BACKUP_PATH_MAP for explicit per-(task, seed) HF paths. Returns the
    number of files copied into the canonical layout.
    """
    from huggingface_hub import snapshot_download

    # Determine which keys are STILL missing locally (idempotency).
    still_missing: dict[tuple[str, str, int], str] = {}
    for (dom, tid, seed), hf_path in _BACKUP_PATH_MAP.items():
        dest = _canonical_dest(bundle_root, dom, tid, seed)
        if _missing_files(dest):
            still_missing[(dom, tid, seed)] = hf_path

    if not still_missing:
        print("[fetch_baselines] backup: all 13 backup-only baselines already present locally — skipping")
        return 0

    allow_patterns = [f"{path}/*" for path in still_missing.values()]
    backup_local = bundle_root / "data_extra" / "_hf_backup_cache"
    backup_local.mkdir(parents=True, exist_ok=True)

    print(f"[fetch_baselines] backup: snapshot_download for {len(still_missing)} task dirs")
    snapshot_download(
        repo_id=_HF_REPO_BACKUP,
        repo_type="dataset",
        token=hf_token,
        local_dir=str(backup_local),
        allow_patterns=allow_patterns,
    )

    copied = 0
    for (dom, tid, seed), hf_path in still_missing.items():
        src_dir = backup_local / hf_path
        if not src_dir.is_dir():
            print(f"[fetch_baselines] backup MISS after download: {hf_path}", file=sys.stderr)
            continue
        dest = _canonical_dest(bundle_root, dom, tid, seed)
        dest.mkdir(parents=True, exist_ok=True)
        for fn in _SIBLING_FILENAMES:
            sp = src_dir / fn
            if sp.exists():
                shutil.copy2(sp, dest / fn)
                copied += 1
    return copied


def verify_coverage(*, bundle_root: Path, eval_tasks: list[dict]) -> tuple[int, list[tuple]]:
    """Return (n_pairs_covered, list_of_missing) for the eval set."""
    covered = 0
    missing: list[tuple] = []
    for t in eval_tasks:
        dom, tid = t["domain"], t["task_id"]
        for seed in t.get("available_seeds", []):
            dest = _canonical_dest(bundle_root, dom, tid, seed)
            if (dest / "phase0_baseline.json").exists():
                covered += 1
            else:
                missing.append((dom, tid, seed))
    return covered, missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle-root", type=Path, default=Path(os.environ.get("BUNDLE_ROOT", ".")))
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                    help="HuggingFace token. Defaults to $HF_TOKEN.")
    ap.add_argument("--keep-extra-cache", action="store_true",
                    help="Keep the data_extra/ download cache (useful for re-runs).")
    args = ap.parse_args(argv)

    if not args.hf_token:
        print("ERROR: HF_TOKEN is required (export it or pass --hf-token).", file=sys.stderr)
        return 2

    bundle_root = args.bundle_root.resolve()
    eval_tasks_jsonl = bundle_root / "data_processed" / "stage2_v1" / "eval_tasks.jsonl"
    if not eval_tasks_jsonl.exists():
        print(f"ERROR: eval_tasks.jsonl not found at {eval_tasks_jsonl}", file=sys.stderr)
        return 2

    eval_tasks = _load_eval_tasks(eval_tasks_jsonl)
    print(f"[fetch_baselines] {len(eval_tasks)} eval tasks; "
          f"{sum(len(t.get('available_seeds', [])) for t in eval_tasks)} (task, seed) pairs to acquire")

    # Step 1: sidecar (cheap, covers ~50 of 63).
    n_fetched, _ = fetch_from_sidecar(
        hf_token=args.hf_token, bundle_root=bundle_root, eval_tasks=eval_tasks,
    )
    print(f"[fetch_baselines] sidecar: copied {n_fetched} files")

    # Step 2: backup (the 13 epoch-specific ones).
    n_copied = fetch_from_backup(
        hf_token=args.hf_token, bundle_root=bundle_root, eval_tasks=eval_tasks,
    )
    print(f"[fetch_baselines] backup: copied {n_copied} files")

    # Step 3: verify
    covered, missing = verify_coverage(bundle_root=bundle_root, eval_tasks=eval_tasks)
    total_pairs = sum(len(t.get("available_seeds", [])) for t in eval_tasks)
    print(f"[fetch_baselines] coverage: {covered}/{total_pairs} (task, seed) pairs")
    if missing:
        print("[fetch_baselines] MISSING after fetch:", file=sys.stderr)
        for m in missing:
            print(f"   - {m}", file=sys.stderr)
        return 3

    # Optional: clean download cache to save disk.
    if not args.keep_extra_cache:
        cache_dir = bundle_root / "data_extra"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"[fetch_baselines] cleaned download cache at {cache_dir}")

    print(f"[fetch_baselines] DONE — {covered} baselines available under {bundle_root}/eval_baselines/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
