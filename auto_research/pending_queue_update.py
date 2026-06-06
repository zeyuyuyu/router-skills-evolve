#!/usr/bin/env python3
"""
Pending queue update bootstrap — last updated 2026-06-06 by daily pipeline.

RECOVERY NOTE (2026-06-06): The cloud idea-gen routine could not push the full
184K experiment list due to MCP socket size limits. The full 53-experiment version
is preserved in git history at SHA 649b8a4fe5f11b319ce3cdd5c205b9124028eda3.

To restore and apply all experiments on A800:

    # Step 1: restore the full 53-experiment script from git history
    git show 649b8a4fe5f11b319ce3cdd5c205b9124028eda3:auto_research/pending_queue_update.py \
        > /tmp/pending_queue_update_full.py

    # Step 2: run it to apply experiments 1-53
    python3 /tmp/pending_queue_update_full.py

    # Step 3: apply today's new experiments (EXP-054, EXP-055)
    python3 auto_research/pending_queue_update_2026_06_06.py

Alternatively, this script auto-detects git and does all three steps:
    python3 auto_research/pending_queue_update.py
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")
REPO_ROOT = Path(__file__).resolve().parent.parent
FULL_SCRIPT_SHA = "649b8a4fe5f11b319ce3cdd5c205b9124028eda3"
FULL_SCRIPT_PATH = "auto_research/pending_queue_update.py"


def restore_and_run_full():
    """Restore the 53-experiment script from git history and run it."""
    try:
        result = subprocess.run(
            ["git", "show", f"{FULL_SCRIPT_SHA}:{FULL_SCRIPT_PATH}"],
            capture_output=True, text=True, cwd=REPO_ROOT, timeout=30
        )
        if result.returncode != 0:
            print(f"WARNING: git show failed: {result.stderr[:200]}")
            return False
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="pqu_full_")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                f.write(result.stdout)
            print(f"Restored 53-experiment script to {tmp_path}")
            run = subprocess.run([sys.executable, tmp_path], timeout=60)
            return run.returncode == 0
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as e:
        print(f"WARNING: restore_and_run_full failed: {e}")
        return False


def run_daily_patch(patch_name):
    patch_path = REPO_ROOT / "auto_research" / patch_name
    if not patch_path.exists():
        print(f"WARNING: patch not found: {patch_path}")
        return False
    run = subprocess.run([sys.executable, str(patch_path)], timeout=60)
    return run.returncode == 0


def main():
    if not STATE_PATH.exists():
        print(f"ERROR: {STATE_PATH} not found. Are you on the A800?")
        return 1

    with open(STATE_PATH) as f:
        state = json.load(f)
    queue = state.get("queue", [])
    history = state.get("history", [])
    all_ids = {e.get("id", "") for e in queue} | {e.get("id", "") for e in history}

    # Check if the older 53 experiments need to be applied
    needs_full = not any("exp_2026_05_15" in eid for eid in all_ids)
    if needs_full:
        print("Older experiments not in state — restoring full script from git history...")
        if not restore_and_run_full():
            print("ERROR: Could not restore full experiment set. Run manually:")
            print(f"  git show {FULL_SCRIPT_SHA}:{FULL_SCRIPT_PATH} > /tmp/pqu.py")
            print("  python3 /tmp/pqu.py")
            return 1
    else:
        print(f"Older experiments already in state ({len(queue)} queued, {len(history)} history). Skipping full restore.")

    # Apply today's patch (EXP-054, EXP-055)
    print("\nApplying 2026-06-06 patch (EXP-054, EXP-055)...")
    run_daily_patch("pending_queue_update_2026_06_06.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
