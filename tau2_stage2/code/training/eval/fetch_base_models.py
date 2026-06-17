"""Snapshot-download the four untrained Qwen base models for the step-budget eval.

Pins to the SAME HF revisions the SFT training started from. Source of truth
for the pinned SHAs is `stage2_ship/docs/2026-05-08-LOCAL-VS-CLOUD-handoff.md:137-140`
(verified live against HF Hub API on 2026-05-09 and again on 2026-05-23).

WHY pin to the same revision:
    HuggingFace model repos are mutable. If the publisher pushed a tokenizer
    or weight update between training-time and eval-time, the "base" model we
    evaluate as the scaling-curve floor would NOT be the same starting weights
    the SFT trained from. The "untrained vs trained" comparison would be
    confounded by HF model drift. Pinning eliminates that risk: the base eval
    is exactly "the trained run minus the SFT step."

OUTPUT LAYOUT:
    $BUNDLE_ROOT/base_models/<dirname>/...   (full snapshot, vLLM-loadable)

DISK FOOTPRINT (approximate):
    Qwen3.5-2B   ~ 4 GB
    Qwen3.5-4B   ~ 8 GB
    Qwen3.5-9B   ~18 GB
    Qwen3.6-35B-A3B ~70 GB  (A3B = 3B active / 35B total)
    TOTAL: ~100 GB

USAGE (server-only — do NOT run locally; this would download 100 GB):
    HF_TOKEN=... BUNDLE_ROOT=/path/to/bundle \
        HF_HUB_ENABLE_HF_TRANSFER=1 \
        python -m training.eval.fetch_base_models

Idempotent: existing complete snapshots are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Pinned per stage2_ship/docs/2026-05-08-LOCAL-VS-CLOUD-handoff.md:137-140.
# (Sentence in the doc: "Model revision SHAs pinned in each run YAML's
# model.revision field; ✓ done — 5 SHAs verified live against HF Hub API on
# 2026-05-09 and committed.")
PINNED_BASES: list[tuple[str, str, str]] = [
    # (target_name,         hf_repo,                  pinned_sha)
    ("base_Qwen3.5-2B",     "Qwen/Qwen3.5-2B",        "15852e8c16360a2fea060d615a32b45270f8a8fc"),
    ("base_Qwen3.5-4B",     "Qwen/Qwen3.5-4B",        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    ("base_Qwen3.5-9B",     "Qwen/Qwen3.5-9B",        "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    ("base_Qwen3.6-35B-A3B","Qwen/Qwen3.6-35B-A3B",   "995ad96eacd98c81ed38be0c5b274b04031597b0"),
]

# Skip these files at download time — they aren't needed for vLLM inference
# and bloat the cache. None for now: vLLM consumes the full snapshot. Keep
# this as a hook for future ignoring (e.g., onnx variants if added).
_IGNORE_PATTERNS: list[str] = []


def _is_snapshot_complete(local_dir: Path, expected_sha: str) -> bool:
    """A snapshot is "complete" if config.json + tokenizer config files +
    safetensors index (multi-file checkpoints) or a single safetensors
    (small checkpoints) are present, AND a STAMP file matches expected_sha.

    The STAMP is written by this script after download; resuming a partial
    snapshot_download mid-download won't write it. Conservative — favors
    re-download over silently using a partial.
    """
    stamp = local_dir / ".pinned_sha"
    if not stamp.exists():
        return False
    try:
        if stamp.read_text().strip() != expected_sha:
            return False
    except OSError:
        return False
    if not (local_dir / "config.json").exists():
        return False
    # Tokenizer presence (any of these): tokenizer.json, tokenizer_config.json,
    # or vocab.json + merges.txt (legacy).
    tok_ok = (
        (local_dir / "tokenizer.json").exists()
        or (local_dir / "tokenizer_config.json").exists()
        or ((local_dir / "vocab.json").exists() and (local_dir / "merges.txt").exists())
    )
    if not tok_ok:
        return False
    # Weights: either a sharded index (.safetensors.index.json) or a single
    # safetensors / .bin file.
    has_shards = (local_dir / "model.safetensors.index.json").exists()
    has_single = any(local_dir.glob("*.safetensors")) or any(local_dir.glob("pytorch_model*.bin"))
    return has_shards or has_single


def fetch_one(
    *,
    target_name: str,
    hf_repo: str,
    pinned_sha: str,
    base_models_root: Path,
    hf_token: str,
    enable_hf_transfer: bool,
) -> dict:
    """Download a single base model. Returns a small metadata dict."""
    from huggingface_hub import HfApi, snapshot_download

    local_dir = base_models_root / target_name

    # Verify the SHA still exists on HF before any disk activity.
    api = HfApi()
    try:
        info = api.model_info(hf_repo, revision=pinned_sha, token=hf_token)
    except Exception as e:
        return {
            "target_name": target_name,
            "ok": False,
            "error": f"model_info failed: {type(e).__name__}: {e}",
        }

    if _is_snapshot_complete(local_dir, pinned_sha):
        return {
            "target_name": target_name,
            "ok": True,
            "skipped": True,
            "local_dir": str(local_dir),
            "pinned_sha": pinned_sha,
            "n_siblings_in_repo": len(info.siblings),
        }

    local_dir.mkdir(parents=True, exist_ok=True)

    # HF_HUB_ENABLE_HF_TRANSFER=1 enables the rust-based fast downloader
    # (hf_transfer package). The training env has it installed
    # (stage2_ship/code/training/requirements.txt line 13). It's roughly
    # 2-3x faster than the default HTTP downloader on multi-GB checkpoints.
    if enable_hf_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    print(f"[fetch_base_models] downloading {hf_repo}@{pinned_sha[:8]} → {local_dir}")
    snapshot_download(
        repo_id=hf_repo,
        revision=pinned_sha,
        repo_type="model",
        token=hf_token,
        local_dir=str(local_dir),
        ignore_patterns=_IGNORE_PATTERNS,
        # local_dir_use_symlinks defaults to "auto" — copies for small files,
        # symlinks for huge ones. Either way, vllm_serve.sh can load from
        # local_dir.
    )

    # Stamp the pinned SHA so resume detection works.
    (local_dir / ".pinned_sha").write_text(pinned_sha)

    return {
        "target_name": target_name,
        "ok": True,
        "skipped": False,
        "local_dir": str(local_dir),
        "pinned_sha": pinned_sha,
        "n_siblings_in_repo": len(info.siblings),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle-root", type=Path, default=Path(os.environ.get("BUNDLE_ROOT", ".")))
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--target", default=None,
                    help="If given, fetch only this one base model (e.g. base_Qwen3.5-4B).")
    ap.add_argument("--verify-only", action="store_true",
                    help="Skip downloads; just verify pinned SHAs exist on HF + which targets are complete locally.")
    ap.add_argument("--no-hf-transfer", action="store_true",
                    help="Disable the hf_transfer fast downloader (debugging only — slower).")
    args = ap.parse_args(argv)

    if not args.hf_token:
        print("ERROR: HF_TOKEN is required.", file=sys.stderr)
        return 2
    bundle_root = args.bundle_root.resolve()
    base_models_root = bundle_root / "base_models"

    targets = PINNED_BASES
    if args.target:
        targets = [t for t in PINNED_BASES if t[0] == args.target]
        if not targets:
            print(f"ERROR: unknown target {args.target!r}; choose one of "
                  f"{[t[0] for t in PINNED_BASES]}", file=sys.stderr)
            return 2

    results: list[dict] = []
    if args.verify_only:
        from huggingface_hub import HfApi
        api = HfApi()
        for target_name, hf_repo, pinned_sha in targets:
            local_complete = _is_snapshot_complete(base_models_root / target_name, pinned_sha)
            try:
                _ = api.model_info(hf_repo, revision=pinned_sha, token=args.hf_token)
                hf_ok = True
                hf_err = None
            except Exception as e:
                hf_ok = False
                hf_err = f"{type(e).__name__}: {e}"
            results.append({
                "target_name": target_name, "hf_repo": hf_repo,
                "pinned_sha": pinned_sha, "hf_revision_exists": hf_ok,
                "hf_error": hf_err, "local_snapshot_complete": local_complete,
            })
        print(json.dumps(results, indent=2))
        return 0 if all(r["hf_revision_exists"] for r in results) else 1

    enable_xfer = not args.no_hf_transfer
    for target_name, hf_repo, pinned_sha in targets:
        res = fetch_one(
            target_name=target_name,
            hf_repo=hf_repo,
            pinned_sha=pinned_sha,
            base_models_root=base_models_root,
            hf_token=args.hf_token,
            enable_hf_transfer=enable_xfer,
        )
        results.append(res)
        if res.get("ok"):
            status = "SKIP (already complete)" if res.get("skipped") else "DONE"
            print(f"[fetch_base_models] {target_name}: {status}")
        else:
            print(f"[fetch_base_models] {target_name}: ERROR — {res.get('error')}", file=sys.stderr)
            return 3

    # Write a manifest so summarize.py / plotting can join targets ↔ HF metadata.
    manifest = base_models_root / "manifest.json"
    manifest.write_text(json.dumps(results, indent=2))
    print(f"[fetch_base_models] manifest at {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
