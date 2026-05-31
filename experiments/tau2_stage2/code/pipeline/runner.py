"""Offline probe-all pipeline runner (multi-provider, multi-seed).

Usage:
    # Single seed (default — same as historical behavior)
    python -m pipeline.runner --adapter tau2_bench --subset retail --limit 5

    # k=4 multi-seed (per charter §3.1 pass^k @ k=4)
    python -m pipeline.runner --adapter tau2_bench --subset retail \\
        --limit 5 --seeds 300,301,302,303

Wires per (task, seed):
    adapter.run_task (tau2 runner via LiteLLM → CommonStack)
    -> reprice via pricing.yaml                              (Phase 1)
    -> adapter.run_task_with_substitution per step/tier/candidate,
       sequential-lock with evolving trajectory + ACTIONS grader
                                                             (Phase 3 probe-all)
    -> emit JSONL supervision records                        (Phase 4)

Each (task, seed) pair is an independent unit. Failure on one seed does
not gate any other seed; pass^k is computed post-hoc from `progress.json`.

Phase 2 (analyzer) and Phase 3.5 (below-lock verify) are not in the
default flow — sequential probe-all covers them. The `analyzer_client`
parameter is retained for a future opt-in audit flow.

The scheduler is per-phase resumable: each phase checks if its persisted
artifact already exists and reuses it instead of re-running. A second
invocation over the same `data_dir` performs zero LLM calls.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from adapters.tau2_bench.adapter import Tau2BenchAdapter
from core.configs.loader import load_pricing, load_tier_pool
from core.schemas.artifacts import (
    LLMSpec,
    RunTaskConfig,
)
from pipeline._litellm_compat import install_fuzzy_model_cost, register_pricing_with_litellm
from pipeline._resolve import _resolve_model
from pipeline.config.settings import load_settings
from pipeline.phases.phase3_search import Phase3Context

# Re-export so existing callers (e.g. tests, sibling modules) can keep
# `from pipeline.runner import _resolve_model`. The implementation lives
# in `pipeline._resolve` to break the runner ↔ phase3_search import cycle.
__all__ = ["_resolve_model", "main", "run_pipeline_for_tasks"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _select_provider_endpoint(
    settings, provider: str
) -> tuple[str, str, list[str]]:
    """Return (api_base, api_key, fallback_keys) for the given provider.

    Exits via sys.exit() with a clear message if --provider openrouter is
    selected but OPENROUTER_API_KEY is not set in .env. CommonStack always
    succeeds because COMMONSTACK_API_KEY is required at Settings load time.
    """
    if provider == "openrouter":
        if not settings.openrouter_api_key:
            sys.exit(
                "--provider openrouter but OPENROUTER_API_KEY is not set in .env"
            )
        return (
            settings.openrouter_base_url,
            settings.openrouter_api_key,
            list(settings.openrouter_api_keys[1:]),
        )
    return (
        settings.commonstack_base_url,
        settings.commonstack_api_key,
        list(settings.commonstack_api_keys[1:]),
    )


async def run_pipeline_for_tasks(
    *,
    adapter,
    analyzer_client=None,   # unused under probe-all; kept for the future audit flow
    run_cfg: RunTaskConfig,
    phase3_ctx: Phase3Context,
    pricing,
    tier_pool,
    data_dir: Path,
    subset: str,
    tasks: list[dict],
    seeds: list[int],
    baseline_confirm_runs: int = 1,
    baseline_max_attempts: int = 1,
    only_task: str | None = None,
    fallback_api_keys: list[str] | None = None,
) -> list[dict]:
    """Per-(task, seed) resumable pipeline scheduler (probe-all flow).

    Outer loop: tasks. Inner loop: seeds. For each (task, seed):
    Phase 0 → 0b (confirm) → 1 → 3 (probe-all) → 4. Each pair has its
    own seed-scoped artifact dir under `tasks/<id>/seed_<n>/`, so seeds
    never overwrite each other.

    `seeds` is the explicit list of seeds (e.g. `[300, 301, 302, 303]`
    for k=4). `run_cfg.seed` is overridden by the per-seed value inside
    the loop, so the seed in `run_cfg` only matters as a structural
    placeholder.

    Phase 2 (analyzer) and Phase 3.5 (below-lock verify) are dropped
    from the default flow — probe-all with sequential trajectory
    evolution covers both. `analyzer_client` is kept in the signature
    for a future opt-in audit mode but is not called here.

    Every phase is skip-if-artifact-exists; a second invocation over the
    same `data_dir` performs zero LLM calls and writes zero new files.
    """
    from adapters.tau2_bench.adapter import (
        deserialize_baseline_messages, serialize_baseline,
    )
    from pipeline.io import (
        artifact_paths, load_baseline,
        load_baseline_repriced, load_exploration,
        load_phase3_checkpoint,
        save_baseline, save_baseline_repriced,
        save_exploration,
    )
    from pipeline.phases.phase1_baseline import reprice_baseline_artifact
    from pipeline.phases.phase3_search import run_phase3_probe_all
    from pipeline.phases.phase4_assembly import run_phase4

    if not seeds:
        raise ValueError("`seeds` must contain at least one seed value.")

    from core.credit_rotation import is_credit_exhaustion

    fallback_keys = list(fallback_api_keys or [])

    results: list[dict] = []
    for task in tasks:
        task_id = str(task["id"])
        if only_task is not None and task_id != only_task:
            continue
        for seed in seeds:
            # Per-(task, seed) try/swap-key/retry loop. Credit-exhaustion
            # errors swap to the next fallback key and re-attempt the same
            # (task, seed); any saved phase artifacts (e.g. a Phase-0
            # baseline that finished before Phase 3 hit the wall) are
            # reused on retry, so the retry is cheap.
            while True:
                try:
                    result = await _run_one(
                        adapter=adapter, task=task, task_id=task_id, seed=seed,
                        run_cfg=run_cfg, phase3_ctx=phase3_ctx,
                        pricing=pricing, tier_pool=tier_pool,
                        data_dir=data_dir, subset=subset,
                        baseline_confirm_runs=baseline_confirm_runs,
                        baseline_max_attempts=baseline_max_attempts,
                        serialize_baseline=serialize_baseline,
                        deserialize_baseline_messages=deserialize_baseline_messages,
                        artifact_paths=artifact_paths,
                        load_baseline=load_baseline,
                        load_baseline_repriced=load_baseline_repriced,
                        load_exploration=load_exploration,
                        load_phase3_checkpoint=load_phase3_checkpoint,
                        save_baseline=save_baseline,
                        save_baseline_repriced=save_baseline_repriced,
                        save_exploration=save_exploration,
                        reprice_baseline_artifact=reprice_baseline_artifact,
                        run_phase3_probe_all=run_phase3_probe_all,
                        run_phase4=run_phase4,
                    )
                    break
                except BaseException as exc:
                    if is_credit_exhaustion(exc) and fallback_keys:
                        next_key = fallback_keys.pop(0)
                        print(f"[runner] credit-exhaustion on (task={task_id}, "
                              f"seed={seed}); swapping to fallback key "
                              f"({len(fallback_keys)} remaining). "
                              f"cause: {exc!s}")
                        run_cfg = _with_api_key(run_cfg, next_key)
                        phase3_ctx = _ctx_with_api_key(phase3_ctx, next_key)
                        continue  # retry same (task, seed) with new key
                    # Either non-credit error (transient API timeout like
                    # Cloudflare 524, model-side ValueError on empty
                    # AssistantMessage, etc.) OR credit-exhaustion with no
                    # fallback keys remaining. In the latter case, the 429
                    # may be a transient rate-limit dressed up as a quota
                    # message — observed empirically on 2026-04-29 when key
                    # 2's probe succeeded seconds after the runner saw 429
                    # on the same key. Don't kill the shard for either:
                    # log, record the pair as failed in progress.json, move
                    # on. A future re-launch will retry from saved phase
                    # artifacts, so non-deterministic failures get a free
                    # second chance.
                    import traceback
                    label = ("credit-exhaustion (no fallback left)"
                             if is_credit_exhaustion(exc)
                             else "non-credit error")
                    print(f"[runner] {label} on (task={task_id}, seed={seed}); "
                          f"recording as failed and continuing. cause: "
                          f"{type(exc).__name__}: {str(exc)[:200]}",
                          flush=True)
                    traceback.print_exc()
                    result = {
                        "task_id": task_id, "seed": seed,
                        "passed": False,
                        "error": (f"crash: {type(exc).__name__}: "
                                  f"{str(exc)[:300]}"),
                        "baseline_usd": 0.0,
                    }
                    break  # exit retry loop, move to next (task, seed)

            results.append(result)
            # Persist progress.json after each (task, seed) pair.
            out_dir = data_dir / adapter.benchmark_name / subset
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "progress.json").write_text(json.dumps(results, indent=2))

    return results


def _with_api_key(run_cfg: RunTaskConfig, key: str) -> RunTaskConfig:
    """Return a copy of `run_cfg` with `api_key` swapped on both LLMSpecs."""
    new_agent = run_cfg.agent.model_copy(
        update={"args": {**run_cfg.agent.args, "api_key": key}})
    new_user = run_cfg.user.model_copy(
        update={"args": {**run_cfg.user.args, "api_key": key}})
    return run_cfg.model_copy(update={"agent": new_agent, "user": new_user})


def _ctx_with_api_key(ctx: Phase3Context, key: str) -> Phase3Context:
    """Return a copy of `ctx` with `api_key` swapped on commonstack_args
    AND on the embedded baseline_run_cfg, so probe-substitution calls
    use the new key too."""
    import dataclasses
    return dataclasses.replace(
        ctx,
        commonstack_args={**ctx.commonstack_args, "api_key": key},
        baseline_run_cfg=_with_api_key(ctx.baseline_run_cfg, key),
    )


async def _run_one(
    *,
    adapter, task, task_id, seed,
    run_cfg, phase3_ctx, pricing, tier_pool,
    data_dir, subset,
    baseline_confirm_runs,
    baseline_max_attempts,
    serialize_baseline, deserialize_baseline_messages,
    artifact_paths,
    load_baseline, load_baseline_repriced, load_exploration, load_phase3_checkpoint,
    save_baseline, save_baseline_repriced, save_exploration,
    reprice_baseline_artifact, run_phase3_probe_all, run_phase4,
) -> dict:
    """Run the full pipeline for one (task, seed) pair."""
    import dataclasses
    paths = artifact_paths(data_dir, adapter.benchmark_name, subset,
                           task_id=task_id, seed=seed)
    # Per-seed RunTaskConfig — only `seed` differs from the caller's run_cfg.
    seed_run_cfg = run_cfg.model_copy(update={"seed": seed})
    # Per-seed Phase3Context — `baseline_run_cfg.seed` must match so probe
    # substitutions reuse this seed's trajectory. Phase3Context is a frozen
    # dataclass; mutate via dataclasses.replace.
    seed_ctx = dataclasses.replace(phase3_ctx, baseline_run_cfg=seed_run_cfg)

    # ---------------- Phase 0 — baseline (run #1)
    if paths.baseline.exists():
        art1 = load_baseline(paths.baseline)
    else:
        tr = adapter.run_task(task, seed_run_cfg, domain=subset)
        art1 = serialize_baseline(
            tr, task_id=task_id, domain=subset, seed=seed,
            baseline_model=tier_pool.baseline_model,
            simulator_model=tier_pool.simulator.model,
            max_steps=seed_run_cfg.max_steps, max_errors=seed_run_cfg.max_errors,
        )
        save_baseline(art1, paths.baseline)

    # ---------------- Phase 0b — baseline confirm (when N >= 2)
    confirm_passed = art1.passed
    if baseline_confirm_runs >= 2:
        if paths.baseline_confirm.exists():
            art2 = load_baseline(paths.baseline_confirm)
        else:
            tr2 = adapter.run_task(task, seed_run_cfg, domain=subset)
            art2 = serialize_baseline(
                tr2, task_id=task_id, domain=subset, seed=seed,
                baseline_model=tier_pool.baseline_model,
                simulator_model=tier_pool.simulator.model,
                max_steps=seed_run_cfg.max_steps, max_errors=seed_run_cfg.max_errors,
            )
            save_baseline(art2, paths.baseline_confirm)
        confirm_passed = confirm_passed and art2.passed

    # ---------------- Phase 0a-retry — retry-on-fail (when --baseline-max-attempts >= 2)
    # Distinct from --baseline-confirm-runs above, which always runs the
    # confirm pass and only sets confirm_passed. This block runs the second
    # attempt ONLY if attempt 1 failed, and uses ITS trajectory for all
    # downstream phases when it passes. Resumability: a second attempt's
    # artifact is stored separately at paths.baseline_retry; both attempts
    # are skipped on re-launch via .exists().
    chosen_baseline = art1
    if baseline_max_attempts >= 2 and not art1.passed:
        if paths.baseline_retry.exists():
            art_retry = load_baseline(paths.baseline_retry)
        else:
            tr_retry = adapter.run_task(task, seed_run_cfg, domain=subset)
            art_retry = serialize_baseline(
                tr_retry, task_id=task_id, domain=subset, seed=seed,
                baseline_model=tier_pool.baseline_model,
                simulator_model=tier_pool.simulator.model,
                max_steps=seed_run_cfg.max_steps, max_errors=seed_run_cfg.max_errors,
            )
            save_baseline(art_retry, paths.baseline_retry)
        if art_retry.passed:
            chosen_baseline = art_retry

    # ---------------- Phase 1 — reprice (no API)
    if paths.repriced.exists():
        repriced = load_baseline_repriced(paths.repriced)
    else:
        repriced = reprice_baseline_artifact(chosen_baseline, pricing=pricing)
        save_baseline_repriced(repriced, paths.repriced)

    # If baseline failed, write Phase-4 marker (so we don't retry) and continue
    if not repriced.passed:
        run_phase4(
            baseline=repriced, analysis=None,
            exploration=_empty_exploration(),
            adapter_name=adapter.benchmark_name, subset=subset,
            supervision_path=paths.supervision,
            emitted_marker=paths.emitted_marker,
            baseline_model=tier_pool.baseline_model,
            baseline_tier=tier_pool.baseline_tier,
            pricing=pricing,
            confirm_passed=False,
            seed=seed,
        )
        return {"task_id": task_id, "seed": seed, "passed": False,
                "error": repriced.error or "baseline_failed",
                "baseline_usd": repriced.actual_usd}

    # ---------------- Phase 3 — probe-all search (analyzer-free)
    baseline_messages = deserialize_baseline_messages(chosen_baseline)
    if paths.phase3.exists():
        phase3_log = load_exploration(paths.phase3)
    else:
        resume_from = (
            load_phase3_checkpoint(paths.phase3_partial)
            if paths.phase3_partial.exists()
            else None
        )
        phase3_log = run_phase3_probe_all(
            adapter=adapter, task=task,
            baseline=repriced,
            baseline_messages=baseline_messages,
            ctx=seed_ctx,
            checkpoint_path=paths.phase3_partial,
            resume_from=resume_from,
        )
        save_exploration(phase3_log, paths.phase3)
        # Final artifact written; stale checkpoint no longer needed.
        if paths.phase3_partial.exists():
            paths.phase3_partial.unlink()

    # ---------------- Phase 4 — assemble
    records = run_phase4(
        baseline=repriced, analysis=None, exploration=phase3_log,
        adapter_name=adapter.benchmark_name, subset=subset,
        supervision_path=paths.supervision,
        emitted_marker=paths.emitted_marker,
        baseline_model=tier_pool.baseline_model,
        baseline_tier=tier_pool.baseline_tier,
        pricing=pricing,
        confirm_passed=confirm_passed,
        seed=seed,
    )

    probed_locks = [
        l for l in phase3_log.locked_results
        if l.source in ("phase3", "phase3_5")
    ]
    return {
        "task_id": task_id, "seed": seed, "passed": True,
        "baseline_usd": repriced.actual_usd,
        "simulator_usd": repriced.simulator_usd,
        "locked_steps": [l.step for l in probed_locks],
        "locked_models": [l.model_id for l in probed_locks],
        "supervision_records": len(records),
        "confirm_passed": confirm_passed,
    }


def _empty_exploration():
    from core.schemas.artifacts import ExplorationLog
    return ExplorationLog(attempts=[], locked_results=[])


def _parse_seeds(s: str) -> list[int]:
    """Parse '300,301,302,303' or '300' → list[int]. Strips whitespace."""
    return [int(p.strip()) for p in s.split(",") if p.strip()]


def _resolve_task_slice(
    adapter: Any,
    *,
    subset: str,
    task_ids_file: str | None,
    offset: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Pick the (subset, offset, limit, task_ids_file) slice of tasks.

    With no manifest: tasks.json[offset:offset+limit].
    With manifest: read IDs in file order, slice [offset:offset+limit] of THAT
    list, then join against tasks.json. Missing IDs raise ValueError so a
    typo doesn't silently shrink the run.
    """
    all_tasks = adapter.load_tasks(subset)
    if task_ids_file is None:
        raw = all_tasks[offset : offset + limit]
        return [t if isinstance(t, dict) else dict(t) for t in raw]

    with open(task_ids_file) as f:
        manifest_ids = [line.strip() for line in f if line.strip()]
    sliced_ids = manifest_ids[offset : offset + limit]
    by_id = {str(t["id"]): t for t in all_tasks}
    missing = [tid for tid in sliced_ids if tid not in by_id]
    if missing:
        raise ValueError(
            f"task IDs from manifest not found in {subset} tasks.json "
            f"({len(missing)} missing, first 5: {missing[:5]})"
        )
    # Preserve manifest order — that's the user-controlled execution order.
    out = [by_id[tid] for tid in sliced_ids]
    return [t if isinstance(t, dict) else dict(t) for t in out]


async def _main(args: argparse.Namespace) -> None:
    install_fuzzy_model_cost()  # litellm cost-lookup tolerance for OR snapshots
    settings = load_settings()
    repo = _repo_root()
    pricing = load_pricing(repo / "core" / "configs" / "pricing.yaml",
                           provider=args.provider)
    register_pricing_with_litellm(pricing)
    tier_pool_name = args.tier_pool or args.adapter
    tier_pool = load_tier_pool(
        repo / "core" / "configs" / "tier_pools" / f"{tier_pool_name}.yaml"
    )

    api_base, api_key, fallback_api_keys = _select_provider_endpoint(
        settings, args.provider
    )
    cs_args = {
        "api_base": api_base,
        "api_key": api_key,            # primary; swapped on credit error
        "custom_llm_provider": "openai",
    }
    # Everything after the primary api_key is a fallback the runner rotates
    # to if the active key returns a credit-exhaustion error mid-(task, seed).
    # Single-key setups have an empty fallback list.

    # Analyzer client unused under probe-all; retained for a future audit flow.
    analyzer_client = None

    adapter = Tau2BenchAdapter(
        vendor_root=repo / "vendor" / "tau2-bench", domain=args.subset
    )

    seeds = _parse_seeds(args.seeds)

    # Structural run_cfg — seed is overridden per-seed inside the runner.
    # We use the first seed here so it's a valid value.
    # Map canonical tier-pool ids to provider-specific (api_id, extra_body).
    provider = args.provider
    agent_id, agent_extra = _resolve_model(
        tier_pool.baseline_model, provider, tier_pool.provider_overrides
    )
    user_id, user_extra = _resolve_model(
        tier_pool.simulator.model, provider, tier_pool.provider_overrides
    )
    agent_args = {**cs_args, **({"extra_body": agent_extra} if agent_extra else {})}
    user_args = {**cs_args, **({"extra_body": user_extra} if user_extra else {})}

    run_cfg = RunTaskConfig(
        agent=LLMSpec(model=f"openai/{agent_id}", args=agent_args),
        user=LLMSpec(model=f"openai/{user_id}", args=user_args),
        seed=seeds[0],
        max_steps=args.max_steps,
        max_errors=args.max_errors,
    )

    # Probes grade with ALL_IGNORE_BASIS (env + communicate + action,
    # multiplied unconditionally — no NL-assertion judge). Plain ALL refuses
    # when a task's reward_basis contains NL_ASSERTION (retail: 112/114
    # tasks), so we bypass the basis filter and always require env + action
    # + communicate to pass. This keeps the search loop deterministic and
    # the pass/fail criterion stricter-or-equal to what baseline Opus was
    # graded on. The final Exp-2 pass-rate measurement (out of scope for
    # collection) can re-grade with ALL_WITH_NL_ASSERTIONS.
    from tau2.evaluator.evaluator import EvaluationType
    phase3_ctx = Phase3Context(
        tier_pool=dict(tier_pool.tiers),
        search_tiers=list(tier_pool.search_tiers),
        pricing=pricing,
        commonstack_args=cs_args,
        baseline_run_cfg=run_cfg,
        domain=args.subset,
        baseline_model=tier_pool.baseline_model,
        baseline_tier=tier_pool.baseline_tier,
        evaluation_type=EvaluationType.ALL_IGNORE_BASIS,
        # Provider-aware sub_spec resolution at the 3 phase3 sub_spec sites.
        # `provider` is bound from --provider via args.provider; provider_overrides
        # come from the tier_pool YAML (see core/configs/tier_pools/<adapter>.yaml).
        provider=provider,
        provider_overrides=tier_pool.provider_overrides,
    )

    if args.task_offset < 0:
        raise ValueError(f"--task-offset must be >= 0, got {args.task_offset}")
    tasks = _resolve_task_slice(
        adapter, subset=args.subset,
        task_ids_file=args.task_ids_file,
        offset=args.task_offset, limit=args.limit,
    )

    results = await run_pipeline_for_tasks(
        adapter=adapter,
        analyzer_client=analyzer_client,
        run_cfg=run_cfg,
        phase3_ctx=phase3_ctx,
        pricing=pricing,
        tier_pool=tier_pool,
        data_dir=settings.data_dir,
        subset=args.subset,
        tasks=tasks,
        seeds=seeds,
        baseline_confirm_runs=args.baseline_confirm_runs,
        baseline_max_attempts=args.baseline_max_attempts,
        only_task=args.only_task,
        fallback_api_keys=fallback_api_keys,
    )

    # pass^k @ k=len(seeds): fraction of tasks where ALL seeds passed.
    by_task: dict[str, list[bool]] = {}
    for r in results:
        by_task.setdefault(str(r["task_id"]), []).append(bool(r.get("passed")))
    pass_at_k = (
        sum(1 for passes in by_task.values() if all(passes) and len(passes) == len(seeds))
        / len(by_task)
    ) if by_task else 0.0

    summary = {
        "n_runs": len(results),
        "n_tasks": len(by_task),
        "k": len(seeds),
        "seeds": seeds,
        "passed_runs": sum(1 for r in results if r.get("passed")),
        f"pass^{len(seeds)}": pass_at_k,
        "total_baseline_usd": sum(r.get("baseline_usd") or 0.0 for r in results),
        "total_supervision_records": sum(
            r.get("supervision_records") or 0 for r in results
        ),
    }
    print(json.dumps({"summary": summary}, indent=2))


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="tau2_bench")
    parser.add_argument("--subset", required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--task-offset", type=int, default=0,
        help="Number of tasks to skip from the start of tasks.json (or from "
             "the manifest, if --task-ids-file is set) before applying --limit. "
             "Use for sharding a single subset across multiple runner instances "
             "(each writing to its own EVOL_DATA_DIR). Default: 0 (no skip).",
    )
    parser.add_argument(
        "--task-ids-file", default=None,
        help="Path to a newline-separated file of task IDs. When set, the "
             "runner targets exactly those IDs (in file order) instead of "
             "tasks.json order. --task-offset and --limit slice the file. "
             "Use to sample non-contiguous task IDs (e.g. type-balanced "
             "telecom sampling).",
    )
    parser.add_argument(
        "--seeds", default="300",
        help="Comma-separated seed list. Default: '300' (single-seed). "
             "For charter §3.1 pass^k @ k=4 use '--seeds 300,301,302,303'.",
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--baseline-confirm-runs", type=int, default=1)
    parser.add_argument(
        "--baseline-max-attempts",
        type=int,
        default=1,
        help=("Maximum number of baseline attempts per (task, seed). Default 1 (no retry). "
              "If set to 2, the runner re-runs baseline once when the first attempt fails; "
              "the trajectory of the FIRST passing attempt is used for downstream phases. "
              "Distinct from --baseline-confirm-runs (which runs a confirm pass regardless)."),
    )
    parser.add_argument("--only-task", default=None)
    parser.add_argument(
        "--provider",
        choices=["commonstack", "openrouter"],
        default="commonstack",
        help="Inference provider (default: commonstack).",
    )
    parser.add_argument(
        "--tier-pool",
        default=None,
        help="Tier-pool config name (without .yaml). Resolved against "
             "core/configs/tier_pools/<name>.yaml. Defaults to --adapter, "
             "matching the historical behavior. Use --tier-pool tau2_bench_anthropic "
             "to run with the Claude Opus 4.7 baseline pool.",
    )
    return parser


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
