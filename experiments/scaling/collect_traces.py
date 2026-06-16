#!/usr/bin/env python3
"""Phase 1: bench-agnostic trace collection.

For each task in the chosen bench, run BOTH a small and a large model and
write one trace row per task. Output schema is the same one consumed by
`src/skills.py`, `experiments/train_learnable_router.py`, and
`experiments/run_e2e_ablation.py`.

Closed-loop routing (added per review 2026-05-21)
-------------------------------------------------
In a multi-cycle run, cycle k makes its routing decisions using the trained
router from cycle k-1 (the LLM adapter is wired via --small-model; the
carried-over SkillBook supplies the procedure prefix to the small model, but
NOT routing — see _policy_decision). Pass
``--router cycle_{k-1}/router/router.joblib`` and
``--skillbook cycle_{k-1}/skillbook.json`` and each trace row gains the policy
decision the current system *would* make:

    policy_route          "small" | "large"  (from the learned router alone)
    policy_router_prob    P(needs large) from the trained router, or null
    policy_skill_verdict  True/False/None from the SkillBook — DIAGNOSTIC ONLY,
                          does not affect policy_route (single global skill)
    policy_final_model    model the policy would have billed
    policy_total_cost     cost under the policy decision (small, or small+large
                          if the policy probes small then falls back)

The oracle small/large outcomes are still recorded (both models are run) so the
NEXT cycle's router trainer keeps clean labels; the policy_* fields close the
loop without introducing label bias. If --router/--skillbook are omitted the
behaviour is byte-for-byte the original (cycle-0 / ablation use).

Usage:
    python experiments/scaling/collect_traces.py \
        --bench tau2_bench \
        --n-tasks 30 \
        --small-model deepseek/deepseek-v3.2 \
        --large-model openai/gpt-5.4-2026-03-05 \
        --router results/scaling_xxx/cycle_0/router/router.joblib \
        --skillbook results/scaling_xxx/cycle_0/skillbook.json \
        --out results/scaling_xxx/cycle_1/traces.jsonl

For smoke testing without GPUs/API keys:
    SCALING_MOCK=1 python experiments/scaling/collect_traces.py --bench tau2_bench --n-tasks 30 --out /tmp/traces.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Make `experiments.scaling.benches` importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scaling.benches import load_adapter  # noqa: E402


class FatalTraceCollectionError(RuntimeError):
    """Abort collection without writing a poisoned trace row."""


_TRACE_WORKER_ADAPTER = None
_TRACE_WORKER_SKILLBOOK = None


def _init_trace_worker(bench: str, skillbook_path: str | None = None) -> None:
    global _TRACE_WORKER_ADAPTER, _TRACE_WORKER_SKILLBOOK
    _TRACE_WORKER_ADAPTER = load_adapter(bench)
    _TRACE_WORKER_SKILLBOOK = _load_skillbook(skillbook_path)


def _run_trace_worker(payload: tuple[str, dict, str, str, int, bool]) -> dict:
    bench, task, small_model, large_model, cycle, force_both = payload
    adapter = _TRACE_WORKER_ADAPTER or load_adapter(bench)
    return adapter.run_task_pair(
        task,
        small_model=small_model,
        large_model=large_model,
        cycle=cycle,
        force_both=force_both,
        skillbook=_TRACE_WORKER_SKILLBOOK,
    )


def _is_provider_cap_error(exc: Exception) -> bool:
    text = str(exc).lower()
    needles = (
        "max cost limit exceeded",
        "limit exceed",
        "ratelimiterror",
        "error code: 429",
    )
    return any(needle in text for needle in needles)


def _validate_trace_or_abort(trace: dict, task_id: str) -> None:
    small_completion = trace.get("small_completion") or ""
    large_completion = trace.get("large_completion") or ""
    if not small_completion and not large_completion:
        if os.environ.get("SCALING_ALLOW_EMPTY_TRACE_FAILURES", "0") == "1":
            print(
                f"[collect_traces] WARN empty completions for task_id={task_id}; "
                "recording failure row because SCALING_ALLOW_EMPTY_TRACE_FAILURES=1",
                file=sys.stderr,
            )
            return
        raise FatalTraceCollectionError(
            f"empty completions for task_id={task_id}; aborting to avoid poisoned traces"
        )
    large_was_run = not bool(trace.get("large_skipped", False)) and int(trace.get("attempts_count") or 0) >= 2
    if large_was_run and not large_completion:
        raise FatalTraceCollectionError(
            f"empty large completion for task_id={task_id}; aborting to avoid poisoned traces"
        )


def _load_router(path: str | None):
    """Load the scaling sklearn router (joblib). Returns the pipeline or None."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[collect_traces] WARN router not found at {p}; routing without it", file=sys.stderr)
        return None
    try:
        import joblib
        pipe = joblib.load(p)
        print(f"[collect_traces] loaded router from {p}", file=sys.stderr)
        return pipe
    except Exception as e:  # noqa: BLE001
        print(f"[collect_traces] WARN could not load router ({e}); routing without it", file=sys.stderr)
        return None


def _load_skillbook(path: str | None):
    """Load the carried-over SkillBook. Returns SkillBook or None."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[collect_traces] WARN skillbook not found at {p}; routing without it", file=sys.stderr)
        return None
    try:
        from src.skills import SkillBook
        sb = SkillBook()
        sb.load(p)
        print(f"[collect_traces] loaded SkillBook ({len(sb.skills)} signatures) from {p}", file=sys.stderr)
        return sb
    except Exception as e:  # noqa: BLE001
        print(f"[collect_traces] WARN could not load skillbook ({e}); routing without it", file=sys.stderr)
        return None


def _policy_decision(prompt, router_pipe, skillbook, small_model,
                     router_threshold=0.5, min_rate=0.8, min_samples=1):
    """Decide route ('small'/'large') from the learned router alone.

    The learned router owns routing (per-task P(needs large)). The SkillBook is
    NOT consulted for routing: with a single global skill its verdict is the same
    for every task and would steamroll the router's per-task signal. We keep the
    SkillBook's role to procedure distillation only (the adapter prepends
    get_procedure() to the small model). `skill_verdict` is still computed and
    returned as a diagnostic field, but it no longer overrides `route`.
    Returns (route, router_prob, skill_verdict).
    """
    router_prob = None
    route = "small"
    if router_pipe is not None and prompt:
        try:
            prob_large = float(router_pipe.predict_proba([prompt])[0][1])
            router_prob = prob_large
            route = "large" if prob_large >= router_threshold else "small"
        except Exception:  # noqa: BLE001
            pass

    # Diagnostic only — recorded in policy_skill_verdict, does NOT affect routing.
    skill_verdict = None
    if skillbook is not None and prompt:
        try:
            from src.skills import extract_signature
            sig = extract_signature(prompt)
            skill = skillbook.skills.get(sig)
            if skill is not None:
                skill_verdict = skill.can_downgrade_to_small("small", min_rate, min_samples)
        except Exception:  # noqa: BLE001
            pass

    return route, router_prob, skill_verdict


def _apply_policy(trace, route, router_prob, skill_verdict, small_model, large_model):
    """Augment a (run-both) trace with what the policy would have billed.

    Keeps oracle small/large outcomes intact for label-clean router training;
    adds policy_* fields reflecting the cycle-(k-1) routing decision.
    """
    small_ok = trace.get("small_success")
    large_ok = trace.get("large_success")
    small_cost = trace.get("small_cost", 0.0) or 0.0
    large_cost = trace.get("large_cost", 0.0) or 0.0
    large_skipped = bool(trace.get("large_skipped", False))

    if route == "large":
        if large_skipped:
            # large was NOT actually run (cost-control skip): we cannot honestly
            # bill a large outcome. Mark unknown rather than fabricate.
            policy_final_model = large_model
            policy_success = None
            policy_cost = None
            policy_decision = "policy:route_large(unknown:large_skipped)"
        else:
            policy_final_model = large_model
            policy_success = bool(large_ok)
            policy_cost = large_cost
            policy_decision = "policy:route_large"
    else:
        # policy routes to small; on small failure it falls back to large
        if small_ok:
            policy_final_model = small_model
            policy_success = True
            policy_cost = small_cost
            policy_decision = "policy:small_ok"
        else:
            policy_final_model = large_model
            policy_success = bool(large_ok)
            policy_cost = small_cost + large_cost
            policy_decision = "policy:small_fail->large"

    trace.update({
        "policy_route": route,
        "policy_router_prob": router_prob,
        "policy_skill_verdict": skill_verdict,
        "policy_final_model": policy_final_model,
        "policy_final_success": policy_success,
        "policy_total_cost": policy_cost,
        "policy_decision": policy_decision,
    })
    return trace


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", required=True, choices=["tau2_bench", "swe_bench", "humaneval"])
    ap.add_argument("--n-tasks", type=int, required=True)
    ap.add_argument("--small-model", required=True)
    ap.add_argument("--large-model", required=True)
    ap.add_argument("--out", required=True, help="output traces.jsonl path")
    ap.add_argument("--cycle", type=int, default=0)
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--router", default=None,
                    help="path to previous cycle's router.joblib (closed-loop routing)")
    ap.add_argument("--skillbook", default=None,
                    help="path to previous cycle's skillbook.json (closed-loop routing)")
    ap.add_argument("--router-threshold", type=float, default=0.5)
    ap.add_argument("--resume", action="store_true",
                    help="append to an existing traces.jsonl and skip task_ids already present")
    ap.add_argument("--force-both", action="store_true",
                    help="run both small and large for every task, even without router/skillbook")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SCALING_TRACE_WORKERS", "1")),
                    help="parallel task workers for trace collection; default from SCALING_TRACE_WORKERS or 1")
    ap.add_argument("--mock", action="store_true",
                    help="generate synthetic deterministic traces (no API/GPU). Equivalent to SCALING_MOCK=1.")
    args = ap.parse_args()

    if args.mock:
        os.environ["SCALING_MOCK"] = "1"

    adapter = load_adapter(args.bench)
    router_pipe = _load_router(args.router)
    skillbook = _load_skillbook(args.skillbook)
    closed_loop = router_pipe is not None or skillbook is not None
    print(f"[collect_traces] bench={args.bench} cycle={args.cycle} split={args.split} "
          f"closed_loop={closed_loop} "
          f"workers={max(1, args.workers)} "
          f"mock={os.environ.get('SCALING_MOCK', '0') == '1'}", file=sys.stderr)

    tasks = adapter.load_tasks(args.n_tasks, split=args.split)
    print(f"[collect_traces] loaded {len(tasks)} tasks", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_success = 0
    n_policy_small = 0
    n_policy_large = 0
    existing_task_ids: set[str] = set()
    if args.resume and out_path.exists():
        with out_path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                task_id = str(row.get("task_id", ""))
                if task_id:
                    existing_task_ids.add(task_id)
                if row.get("final_success"):
                    n_success += 1
                if row.get("policy_route") == "small":
                    n_policy_small += 1
                elif row.get("policy_route") == "large":
                    n_policy_large += 1
        print(f"[collect_traces] resume enabled; loaded {len(existing_task_ids)} existing task_ids",
              file=sys.stderr)

    mode = "a" if args.resume else "w"
    t0 = time.time()
    tasks_to_run = [
        (i, task)
        for i, task in enumerate(tasks, 1)
        if str(task.get("task_id", f"unknown-{i}")) not in existing_task_ids
    ]

    def handle_trace(fh, i: int, task: dict, trace: dict, completed: int) -> None:
        nonlocal n_success, n_policy_small, n_policy_large
        task_id = str(task.get("task_id", f"unknown-{i}"))
        _validate_trace_or_abort(trace, task_id)
        if closed_loop:
            route, rprob, sverd = _policy_decision(
                trace.get("prompt", "") or task.get("prompt", ""),
                router_pipe, skillbook, args.small_model,
                router_threshold=args.router_threshold,
            )
            trace = _apply_policy(trace, route, rprob, sverd,
                                  args.small_model, args.large_model)
            if route == "small":
                n_policy_small += 1
            else:
                n_policy_large += 1
        fh.write(json.dumps(trace, ensure_ascii=False) + "\n")
        fh.flush()
        if trace.get("final_success"):
            n_success += 1
        total_done = len(existing_task_ids) + completed
        if total_done % 10 == 0 or total_done == len(tasks):
            elapsed = time.time() - t0
            print(f"[collect_traces] {total_done}/{len(tasks)}  "
                  f"success={n_success}/{total_done}  elapsed={elapsed:.1f}s",
                  file=sys.stderr)

    with out_path.open(mode) as fh:
        if max(1, args.workers) == 1:
            for completed, (i, task) in enumerate(tasks_to_run, 1):
                task_id = str(task.get("task_id", f"unknown-{i}"))
                try:
                    # closed_loop => force both models so the policy annotation has
                    # a REAL large outcome even when small also succeeded (review
                    # round 2, 2026-05-21). Without this, routing to large on a
                    # small-OK task would bill a fake skip placeholder.
                    trace = adapter.run_task_pair(
                        task,
                        small_model=args.small_model,
                        large_model=args.large_model,
                        cycle=args.cycle,
                        force_both=(closed_loop or args.force_both),
                        skillbook=skillbook,
                    )
                    handle_trace(fh, i, task, trace, completed)
                except FatalTraceCollectionError:
                    raise
                except Exception as e:  # noqa: BLE001
                    if _is_provider_cap_error(e):
                        raise FatalTraceCollectionError(
                            f"provider cap/rate limit while collecting task_id={task_id}: {e}"
                        ) from e
                    print(f"[collect_traces] task {task.get('task_id')} FAILED: {e}", file=sys.stderr)
                    fh.write(json.dumps({
                        "task_id": task.get("task_id", f"unknown-{i}"),
                        "signature": "",
                        "decision": "error",
                        "attempts": 0,
                        "attempts_count": 0,
                        "final_success": False,
                        "final_model": "",
                        "total_cost": 0.0,
                        "round": args.cycle,
                        "error": str(e),
                    }) + "\n")
                    fh.flush()
        else:
            force_both = closed_loop or args.force_both
            with ProcessPoolExecutor(
                max_workers=max(1, args.workers),
                initializer=_init_trace_worker,
                initargs=(args.bench, args.skillbook),
            ) as pool:
                futures = {
                    pool.submit(
                        _run_trace_worker,
                        (args.bench, task, args.small_model, args.large_model, args.cycle, force_both),
                    ): (i, task)
                    for i, task in tasks_to_run
                }
                for completed, fut in enumerate(as_completed(futures), 1):
                    i, task = futures[fut]
                    task_id = str(task.get("task_id", f"unknown-{i}"))
                    try:
                        trace = fut.result()
                        handle_trace(fh, i, task, trace, completed)
                    except FatalTraceCollectionError:
                        raise
                    except Exception as e:  # noqa: BLE001
                        if _is_provider_cap_error(e):
                            raise FatalTraceCollectionError(
                                f"provider cap/rate limit while collecting task_id={task_id}: {e}"
                            ) from e
                        print(f"[collect_traces] task {task.get('task_id')} FAILED: {e}", file=sys.stderr)
                        fh.write(json.dumps({
                            "task_id": task.get("task_id", f"unknown-{i}"),
                            "signature": "",
                            "decision": "error",
                            "attempts": 0,
                            "attempts_count": 0,
                            "final_success": False,
                            "final_model": "",
                            "total_cost": 0.0,
                            "round": args.cycle,
                            "error": str(e),
                        }) + "\n")
                        fh.flush()

    elapsed = time.time() - t0
    msg = (f"[collect_traces] DONE  out={out_path}  "
           f"final_success={n_success}/{len(tasks)} ({100*n_success/max(1,len(tasks)):.1f}%)  "
           f"elapsed={elapsed:.1f}s")
    if closed_loop:
        msg += f"  policy_route: small={n_policy_small} large={n_policy_large}"
    print(msg, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
