"""Force-routed τ²-bench eval harness for one trained checkpoint.

Procedure:
1. Spin up vLLM via vllm_serve.sh on a unique port (one trained checkpoint per port).
2. Wait for /v1/models to respond (vLLM is OpenAI-compatible).
3. Group eval tasks by (domain, max_steps, max_errors). Tau2's CLI takes a
   single max_steps / max_errors per invocation, so we issue one
   `tau2 run` per group, routing the agent at the local vLLM endpoint via
   `--agent-llm-args '{"api_base": ..., "api_key": "EMPTY"}'`. Tau2 writes a
   Results JSON for each group.
4. Flatten all SimulationRuns into per-rollout records (passed/cost/etc.) and
   aggregate via summarize_rollouts → eval_results.json + eval_rollouts.jsonl.
5. Tear down vLLM.

This module deliberately does NOT depend on `pipeline.runner --provider local`.
The earlier design called for that, but pipeline.runner does not implement a
`local` provider (its choices are commonstack / openrouter), and the runner's
probe-all infrastructure (multi-tier baseline + step-replacement search) is
not what we want for a force-routed eval anyway. Going through tau2's CLI
keeps the eval path narrow and trivially auditable per upstream τ²-bench.

Usage:
    python -m training.eval.harness \\
        --checkpoint train_outputs/<run_id>/checkpoint-best \\
        --output-dir train_outputs/<run_id> \\
        --bundle-root . \\
        --eval-tasks data_processed/stage2_v1/eval_tasks.jsonl \\
        --port 8000 --gpu 0 --seed 300
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# Default user-simulator + judge models. Matches the simulator_user_model
# value baked into eval_tasks.jsonl. Override via env or CLI if needed.
DEFAULT_USER_LLM = "openai/gpt-5.2"

# vLLM serves the checkpoint under this `--served-model-name`; matches
# vllm_serve.sh. The agent_llm name passed to tau2 must agree.
SERVED_MODEL_NAME = "evol-llm-student"


def serve_vllm(checkpoint: Path, port: int, gpu: int, vllm_serve_sh: Path) -> subprocess.Popen:
    """Spin up vLLM in background. Returns the Popen handle."""
    proc = subprocess.Popen(
        ["bash", str(vllm_serve_sh), str(checkpoint), str(port), str(gpu)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return proc


def wait_for_vllm(port: int, timeout_s: int = 600, launcher: subprocess.Popen | None = None) -> bool:
    """Poll the OpenAI-compatible /v1/models endpoint until ready.

    If `launcher` is provided, also bail early when the bash launcher exits
    non-zero (e.g., vllm rejected an arg, ran out of GPU memory). The
    launcher itself backgrounds vllm and returns within seconds, so we
    treat any non-zero exit as fatal but a zero exit (with vllm still
    coming up) as expected — keep polling /v1/models until the deadline.
    """
    import urllib.request
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        if launcher is not None:
            rc = launcher.poll()
            if rc is not None and rc != 0:
                return False
        time.sleep(5)
    return False


def stop_vllm(checkpoint: Path) -> None:
    """Kill the vLLM process recorded in the pid file."""
    pid_file = checkpoint / "vllm_serve.pid"
    if not pid_file.exists():
        return
    text = pid_file.read_text().split()
    for tok in text:
        if tok.startswith("PID="):
            try:
                pid = int(tok.removeprefix("PID="))
                os.kill(pid, signal.SIGTERM)
                time.sleep(2)
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass
    pid_file.unlink(missing_ok=True)


def load_eval_tasks_grouped(eval_tasks_jsonl: Path) -> dict[tuple[str, int, int], list[str]]:
    """Load eval tasks; group by (domain, max_steps, max_errors) → task_ids.

    Tau2's CLI takes a single max_steps/max_errors per invocation, so we
    invoke once per distinct (domain, limits) tuple to honor each task's
    declared budget without overrunning easier ones.
    """
    groups: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    for line in eval_tasks_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["domain"], int(r["max_steps"]), int(r["max_errors"]))
        groups[key].append(r["task_id"])
    return dict(groups)


def run_tau2_eval_for_group(
    *,
    domain: str,
    task_ids: list[str],
    max_steps: int,
    max_errors: int,
    port: int,
    output_path: Path,
    seed: int,
    user_llm: str = DEFAULT_USER_LLM,
    max_concurrency: int = 4,
) -> dict:
    """Invoke `tau2 run` for one (domain, limits) group; return parsed Results."""
    agent_llm_args = json.dumps(
        {"api_base": f"http://localhost:{port}/v1", "api_key": "EMPTY"}
    )
    cmd = [
        sys.executable, "-m", "tau2.cli", "run",
        "--domain", domain,
        "--agent-llm", f"openai/{SERVED_MODEL_NAME}",
        "--agent-llm-args", agent_llm_args,
        "--user-llm", user_llm,
        "--task-ids", *task_ids,
        "--seed", str(seed),
        "--max-steps", str(max_steps),
        "--max-errors", str(max_errors),
        "--max-concurrency", str(max_concurrency),
        "--save-to", str(output_path),
        "--auto-resume",
        "--log-level", "WARNING",
    ]
    subprocess.run(cmd, check=True)
    return json.loads(output_path.read_text())


def flatten_simulations_to_rollouts(
    tau2_results: dict, *, default_replacement_rate_k: float = 1.0
) -> list[dict]:
    """Convert tau2 Results JSON → list of per-rollout dicts for summarize_rollouts.

    Force-routed eval means the student is the agent for every step, so
    `replacement_rate_k = 1.0` for all rollouts. `total_task_cost_usd` is
    the sum of agent_cost (always 0 for self-hosted vLLM unless tau2 prices
    it via litellm) and user_cost (real OpenAI cost on the user simulator).
    """
    rollouts: list[dict] = []
    for sim in tau2_results.get("simulations", []):
        reward_info = sim.get("reward_info") or {}
        reward = float(reward_info.get("reward", 0.0))
        agent_cost = float(sim.get("agent_cost") or 0.0)
        user_cost = float(sim.get("user_cost") or 0.0)
        rollouts.append({
            "task_id": sim.get("task_id"),
            "seed": sim.get("seed"),
            "passed": reward >= 0.5,
            "reward": reward,
            "replacement_rate_k": default_replacement_rate_k,
            "total_task_cost_usd": agent_cost + user_cost,
            "agent_cost_usd": agent_cost,
            "user_cost_usd": user_cost,
            "termination_reason": sim.get("termination_reason"),
        })
    return rollouts


def summarize_rollouts(rows: list[dict]) -> dict:
    """Compute aggregate metrics from a list of rollout result dicts.

    Preserves the legacy contract used by test_harness.py: missing keys
    default to (0.0, False).
    """
    n = len(rows)
    pass_rate = sum(1 for r in rows if r.get("passed")) / max(1, n) if n else 0.0
    avg_k = (sum(r.get("replacement_rate_k", 0.0) for r in rows) / n) if n else 0.0
    avg_cost = (sum(r.get("total_task_cost_usd", 0.0) for r in rows) / n) if n else 0.0
    return {
        "n_rollouts": n,
        "pass_rate": pass_rate,
        "replacement_rate_k_mean": avg_k,
        "total_task_cost_usd_mean": avg_cost,
    }


def summarize_with_per_domain(rollouts: list[dict], task_id_to_domain: dict[str, str]) -> dict:
    """Aggregate stats with per-domain breakdown."""
    overall = summarize_rollouts(rollouts)
    per_domain: dict[str, list[dict]] = defaultdict(list)
    for r in rollouts:
        domain = task_id_to_domain.get(r.get("task_id"), "unknown")
        per_domain[domain].append(r)
    overall["per_domain"] = {d: summarize_rollouts(rs) for d, rs in per_domain.items()}
    return overall


def run_eval(
    *,
    bundle_root: Path,
    eval_tasks_jsonl: Path,
    output_dir: Path,
    seed: int,
    port: int,
    user_llm: str = DEFAULT_USER_LLM,
    max_concurrency: int = 4,
    n_tasks_limit: int | None = None,
    rollouts_basename: str = "eval_rollouts.jsonl",
) -> dict:
    """Run full eval: per-group tau2 invocations → per-rollout records → summary."""
    groups = load_eval_tasks_grouped(eval_tasks_jsonl)
    # Build a task_id → domain map for per-domain aggregation.
    task_id_to_domain = {}
    for line in eval_tasks_jsonl.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            task_id_to_domain[row["task_id"]] = row["domain"]

    if n_tasks_limit is not None:
        # For smoke runs: only keep the first N task_ids across all groups.
        capped: dict[tuple[str, int, int], list[str]] = {}
        kept = 0
        for key, ids in groups.items():
            if kept >= n_tasks_limit:
                break
            take = ids[: max(0, n_tasks_limit - kept)]
            if take:
                capped[key] = take
                kept += len(take)
        groups = capped

    output_dir.mkdir(parents=True, exist_ok=True)
    rollouts_jsonl = output_dir / rollouts_basename
    all_rollouts: list[dict] = []
    for (domain, max_steps, max_errors), task_ids in sorted(groups.items()):
        group_save = output_dir / f"tau2_results_{domain}_s{max_steps}_e{max_errors}_seed{seed}.json"
        results: dict | None = None
        # Reuse cached results ONLY when the cache covers EXACTLY the current
        # task_ids set. A prior crashed tau2 invocation can leave a partial
        # JSON (fewer sims than tasks); a prior wider invocation can leave
        # MORE sims (extra tasks). Both must be re-run / filtered.
        task_id_set = set(map(str, task_ids))
        if group_save.exists() and group_save.stat().st_size > 0:
            cached = json.loads(group_save.read_text())
            cached_sims_all = cached.get("simulations", [])
            cached_task_ids = {str(s.get("task_id")) for s in cached_sims_all}
            if task_id_set.issubset(cached_task_ids):
                # Cache covers all current task_ids — keep only the ones
                # in scope and reuse.
                cached["simulations"] = [
                    s for s in cached_sims_all
                    if str(s.get("task_id")) in task_id_set
                ]
                results = cached
            else:
                missing = sorted(task_id_set - cached_task_ids)[:5]
                print(
                    f"WARN: cached {group_save.name} covers "
                    f"{len(cached_task_ids & task_id_set)}/{len(task_id_set)} "
                    f"current task_ids (missing e.g. {missing}); re-running group.",
                    file=sys.stderr,
                )
        if results is None:
            results = run_tau2_eval_for_group(
                domain=domain,
                task_ids=task_ids,
                max_steps=max_steps,
                max_errors=max_errors,
                port=port,
                output_path=group_save,
                seed=seed,
                user_llm=user_llm,
                max_concurrency=max_concurrency,
            )
        all_rollouts.extend(flatten_simulations_to_rollouts(results))

    rollouts_jsonl.write_text("".join(json.dumps(r) + "\n" for r in all_rollouts))
    return summarize_with_per_domain(all_rollouts, task_id_to_domain)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--bundle-root", required=True, type=Path)
    ap.add_argument("--eval-tasks", required=True, type=Path)
    ap.add_argument("--heldout-tasks", type=Path, default=None,
                    help="Optional held-out eval task descriptors JSONL.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=300,
                    help="Single seed for this eval run; for pass^k, run with multiple --seed values.")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="DEPRECATED — use --seed once per eval. Kept for back-compat with eval_all.sh.")
    ap.add_argument("--user-llm", default=DEFAULT_USER_LLM)
    ap.add_argument("--max-concurrency", type=int, default=4)
    args = ap.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vllm_serve_sh = args.bundle_root / "code/training/eval/vllm_serve.sh"

    # Honor legacy --seeds (single-element list); take the first.
    seed = args.seed if args.seeds is None else int(args.seeds[0])

    res_path = args.output_dir / "eval_results.json"
    if res_path.exists() and res_path.stat().st_size > 0:
        print(f"SKIP: {res_path} already exists.")
        return 0

    proc = serve_vllm(args.checkpoint, args.port, args.gpu, vllm_serve_sh)
    try:
        if not wait_for_vllm(args.port, launcher=proc):
            print("vLLM failed to start", file=sys.stderr)
            return 2

        main_summary = run_eval(
            bundle_root=args.bundle_root,
            eval_tasks_jsonl=args.eval_tasks,
            output_dir=args.output_dir,
            seed=seed,
            port=args.port,
            user_llm=args.user_llm,
            max_concurrency=args.max_concurrency,
            rollouts_basename="eval_rollouts.jsonl",
        )
        res_path.write_text(json.dumps(main_summary, indent=2))

        if args.heldout_tasks is not None and args.heldout_tasks.exists():
            held_summary = run_eval(
                bundle_root=args.bundle_root,
                eval_tasks_jsonl=args.heldout_tasks,
                output_dir=args.output_dir,
                seed=seed,
                port=args.port,
                user_llm=args.user_llm,
                max_concurrency=args.max_concurrency,
                rollouts_basename="eval_rollouts_heldout.jsonl",
            )
            (args.output_dir / "eval_results_heldout.json").write_text(
                json.dumps(held_summary, indent=2)
            )
        return 0
    finally:
        stop_vllm(args.checkpoint)


if __name__ == "__main__":
    sys.exit(main())
