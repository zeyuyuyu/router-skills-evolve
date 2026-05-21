#!/bin/bash
# Cron entry for the auto-research orchestrator (v2: now also syncs from GitHub).
# Drop into crontab:
#   */30 * * * * /data0/home/zeyuwang/auto_research/cron_entry.sh >> /data0/home/zeyuwang/auto_research/logs/cron.log 2>&1

cd /data0/home/zeyuwang/auto_research
export PYTHONUNBUFFERED=1
export http_proxy=http://127.0.0.1:1080
export https_proxy=http://127.0.0.1:1080

# Guard against overlapping ticks
LOCK=/tmp/auto_research_tick.lock
exec 9>$LOCK
if ! flock -n 9; then
  echo "[$(date -u)] another tick still holds the lock, skipping" >&2
  exit 0
fi

# ---------- Sync from GitHub (cloud agent's commits) ----------
REPO=/data0/home/zeyuwang/router-skills-evolve
PENDING="$REPO/auto_research/pending_queue_update.py"
APPLIED_MARKER="/data0/home/zeyuwang/auto_research/.last_applied_pending_sha"

if [ -d "$REPO/.git" ]; then
  # Stash uncommitted .py mods, fetch, hard reset, unstash. Idempotent.
  cd "$REPO"
  git stash push -q -m "auto-sync-$(date -u +%s)" experiments/*.py 2>/dev/null || true
  if git fetch origin main 2>/dev/null; then
    git reset --hard origin/main 2>/dev/null
  fi
  git stash pop -q 2>/dev/null || true

  # Apply pending queue update if it exists and content changed
  if [ -f "$PENDING" ]; then
    CUR_SHA=$(sha256sum "$PENDING" | awk '{print $1}')
    LAST_SHA=""
    [ -f "$APPLIED_MARKER" ] && LAST_SHA=$(cat "$APPLIED_MARKER")
    if [ "$CUR_SHA" != "$LAST_SHA" ]; then
      echo "[$(date -u)] applying pending_queue_update.py (sha $CUR_SHA != $LAST_SHA)"
      python3 "$PENDING" || true
      echo "$CUR_SHA" > "$APPLIED_MARKER"
    fi
  fi
  cd /data0/home/zeyuwang/auto_research
fi

# ---------- Run orchestrator ----------
python3 /data0/home/zeyuwang/auto_research/orchestrator.py
