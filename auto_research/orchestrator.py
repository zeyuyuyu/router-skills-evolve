#!/usr/bin/env python3
"""Auto-research orchestrator.

Single-shot driver invoked by cron. Each invocation:
  1. Check `running` list. For each running, poll if done. If done, collect
     metrics, move to history, free slot.
  2. While free slot available and queue non-empty, pop highest-priority
     experiment, launch it, record pid.
  3. Persist state.json atomically.

Designed to be idempotent so a missed cron tick is safe.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

STATE_PATH = Path(os.environ.get("AUTO_RES_STATE", "/data0/home/zeyuwang/auto_research/state.json"))
LOG_ROOT = Path("/data0/home/zeyuwang/auto_research/logs")
LOG_ROOT.mkdir(parents=True, exist_ok=True)

ORCHESTRATOR_LOG = LOG_ROOT / "orchestrator.log"


def log(msg: str) -> None:
    ts = datetime.datetime.utcnow().isoformat()
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(ORCHESTRATOR_LOG, "a") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if not STATE_PATH.exists():
        raise FileNotFoundError(f"state file missing: {STATE_PATH}")
    return json.loads(STATE_PATH.read_text())


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    tmp.replace(STATE_PATH)


def free_disk_gb(path: str) -> float:
    s = shutil.disk_usage(path)
    return s.free / (1024 ** 3)


def is_alive(pid: int) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def pick_free_gpus(state: dict, need: int = 1) -> list[int]:
    """Find GPUs not held by our running experiments. Returns list of indices."""
    used = set()
    for r in state.get("running", []):
        for g in r.get("gpus", []):
            used.add(int(g))
    # we have GPUs 0-7 on A800, but 2,3 often held by other users
    candidates = [0, 1, 4, 5, 6, 7]
    free = [g for g in candidates if g not in used]
    return free[:need]


def launch_experiment(exp: dict, state: dict) -> dict | None:
    """Launch one experiment. Returns the running-record dict, or None on failure."""
    runner = Path(__file__).parent / "runner.py"
    exp_id = exp["id"]
    log_path = LOG_ROOT / f"{exp_id}.log"

    gpus = pick_free_gpus(state, need=1)
    if not gpus:
        log(f"no free GPU for {exp_id}, skipping")
        return None

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    env["http_proxy"] = "http://127.0.0.1:1080"
    env["https_proxy"] = "http://127.0.0.1:1080"

    spec_path = LOG_ROOT / f"{exp_id}.spec.json"
    spec_path.write_text(json.dumps(exp, indent=2))

    cmd = ["python3", str(runner), str(spec_path)]
    log(f"launching {exp_id} on GPU {gpus} -> {log_path}")

    with open(log_path, "a") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            cwd="/data0/home/zeyuwang/auto_research",
        )

    return {
        "id": exp_id,
        "pid": proc.pid,
        "gpus": gpus,
        "started_at": datetime.datetime.utcnow().isoformat(),
        "log": str(log_path),
        "spec_path": str(spec_path),
        "exp": exp,
    }


def collect_finished(record: dict) -> dict:
    """When a running record completes, collect its result."""
    result_path = LOG_ROOT / f"{record['id']}.result.json"
    result = None
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text())
        except Exception as e:
            log(f"failed to parse result for {record['id']}: {e}")
    return {
        **record,
        "finished_at": datetime.datetime.utcnow().isoformat(),
        "result": result,
    }


def count_today_finished(history: list) -> int:
    today = datetime.datetime.utcnow().date().isoformat()
    return sum(1 for h in history if (h.get("finished_at", "")[:10] == today))


def tick(state: dict) -> dict:
    cfg = state["config"]

    # 1) check disk
    free_gb = free_disk_gb(cfg.get("project_root", "/data0/home/zeyuwang"))
    log(f"free disk: {free_gb:.1f} GB")

    if free_gb < cfg.get("max_disk_gb_floor", 50):
        log(f"disk floor breached ({free_gb:.1f} GB < {cfg.get('max_disk_gb_floor')}), NOT launching new experiments")
        new_allowed = False
    else:
        new_allowed = True

    # 2) check running list
    still_running = []
    for r in state.get("running", []):
        pid = r.get("pid")
        if is_alive(pid):
            still_running.append(r)
            log(f"  running {r['id']} (pid {pid}) still alive")
        else:
            finished = collect_finished(r)
            state.setdefault("history", []).append(finished)
            log(f"  collected {r['id']} -> history (result={'ok' if finished.get('result') else 'no-result-file'})")
    state["running"] = still_running

    # 3) daily cap check
    today_count = count_today_finished(state.get("history", []))
    daily_cap = cfg.get("max_experiments_per_day", 30)
    if today_count >= daily_cap:
        log(f"daily cap reached ({today_count} >= {daily_cap}), not launching new")
        new_allowed = False

    # 3.5) queue starvation guard: if both queue and running are empty,
    # invoke the local rule-based idea generator to perpetuate motion.
    if new_allowed and not state.get('running') and not state.get('queue'):
        idea_path = '/data0/home/zeyuwang/auto_research/idea_gen_local.py'
        try:
            log('queue + running both empty: persisting state then invoking idea_gen_local.py')
            save_state(state)
            rc = subprocess.call(['python3', idea_path])
            log(f'idea_gen_local rc={rc}')
            # reload state to pick up newly injected experiments
            state.clear(); state.update(load_state())
        except Exception as e:
            log(f'idea_gen_local invocation failed: {e!r}')

    # 4) launch up to max_parallel
    max_parallel = cfg.get("max_parallel", 4)
    while new_allowed and len(state.get("running", [])) < max_parallel and state.get("queue"):
        next_idx = max(range(len(state["queue"])), key=lambda i: state["queue"][i].get("priority", 0))
        exp = state["queue"].pop(next_idx)
        rec = launch_experiment(exp, state)
        if rec is None:
            # GPU shortage; put back and stop
            state["queue"].insert(next_idx, exp)
            log("ran out of GPUs, will retry next tick")
            break
        state.setdefault("running", []).append(rec)

    # 5) status summary
    log(
        f"end state: running={len(state.get('running', []))} "
        f"queued={len(state.get('queue', []))} history={len(state.get('history', []))}"
    )
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log("=== orchestrator tick ===")
    state = load_state()

    if args.dry_run:
        log("DRY RUN: will report state but not launch anything")
        log(f"running={len(state.get('running', []))} queued={len(state.get('queue', []))}")
        return

    new_state = tick(state)
    save_state(new_state)
    log("=== tick done ===")


if __name__ == "__main__":
    main()
