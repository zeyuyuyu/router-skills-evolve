"""Held-out task-id split for true generalization eval (Spec §5.7).

The standard val set shares task_ids with train (different seeds only).
For a stricter test, hold out 5 task_ids per domain entirely.
"""
from __future__ import annotations

import random
from collections import defaultdict


def select_heldout_task_ids(
    rows: list[dict], n_per_domain: int, seed: int = 42
) -> dict[str, set[str]]:
    """Pick n_per_domain unique task_ids per domain to hold out.

    Args:
        rows: list of stage-2 rows with `_p.domain` and `_p.task_id`.
        n_per_domain: number of task_ids to hold out per domain.
        seed: RNG seed for reproducibility.

    Returns:
        {domain: {task_id, ...}} — task_ids that should be held out.

    Raises:
        ValueError: if any domain has fewer than n_per_domain unique task_ids.
    """
    by_domain: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_domain[r["_p"]["domain"]].add(r["_p"]["task_id"])

    rng = random.Random(seed)
    heldout: dict[str, set[str]] = {}
    for domain, tids in by_domain.items():
        if len(tids) < n_per_domain:
            raise ValueError(
                f"Domain {domain!r} has only {len(tids)} unique task_ids, "
                f"not enough unique task_ids to hold out {n_per_domain}."
            )
        sorted_tids = sorted(tids)  # Determinism: sort before sampling.
        rng.shuffle(sorted_tids)
        heldout[domain] = set(sorted_tids[:n_per_domain])
    return heldout


def split_rows_by_heldout(
    rows: list[dict], heldout_tids: dict[str, set[str]]
) -> tuple[list[dict], list[dict]]:
    """Split rows into (train_keep, heldout) based on held-out task_ids.

    Args:
        rows: full row list.
        heldout_tids: {domain: {task_id, ...}}.

    Returns:
        (train_keep, heldout_rows). train_keep + heldout_rows == rows (preserved order).
    """
    train_keep, heldout = [], []
    for r in rows:
        d = r["_p"]["domain"]
        tid = r["_p"]["task_id"]
        if tid in heldout_tids.get(d, set()):
            heldout.append(r)
        else:
            train_keep.append(r)
    return train_keep, heldout


# Per-domain default eval limits for held-out tasks. Tau2 doesn't expose a
# domain → (max_steps, max_errors) constant, so we anchor on what
# eval_tasks.jsonl declared during the corpus build. Verified empirically
# against data_processed/stage2_v1/eval_tasks.jsonl:
#   - airline: 100/10 dominant (6/9 rows; 3 diag rows at 25/1 not in the
#              held-out path)
#   - retail:  100/10 dominant (10/18 rows; 5 diag at 25/1 + 3 long at
#              400/30 also not in the held-out path)
#   - telecom: 100/10 — ALL 8 telecom eval rows declared this; previous
#              400/30 here was a corpus-divergent guess.
HELDOUT_LIMITS_BY_DOMAIN: dict[str, dict[str, int]] = {
    "airline": {"max_steps": 100, "max_errors": 10},
    "retail":  {"max_steps": 100, "max_errors": 10},
    "telecom": {"max_steps": 100, "max_errors": 10},
}


def expand_heldout_ids_to_descriptors(
    heldout_ids: dict[str, list[str] | set[str]],
) -> list[dict]:
    """Expand `{domain: [task_id, ...]}` into eval-harness-ready descriptors.

    Output JSONL row shape mirrors data_processed/stage2_v1/eval_tasks.jsonl
    in the four fields the new harness consumes (task_id / domain /
    max_steps / max_errors). Other eval_tasks.jsonl fields aren't required
    by the eval path; they're carried forward as `null` for diff-friendliness
    if a downstream tool ever wants them.
    """
    descriptors: list[dict] = []
    for domain in sorted(heldout_ids):
        tids = heldout_ids[domain]
        if isinstance(tids, set):
            tids = sorted(tids)
        else:
            tids = sorted(tids)  # determinism
        limits = HELDOUT_LIMITS_BY_DOMAIN.get(domain)
        if limits is None:
            raise ValueError(
                f"No held-out limits configured for domain {domain!r}. "
                f"Add a HELDOUT_LIMITS_BY_DOMAIN entry."
            )
        for tid in tids:
            descriptors.append({
                "task_id": str(tid),
                "domain": domain,
                "max_steps": limits["max_steps"],
                "max_errors": limits["max_errors"],
                "rep_rate_pinned": None,
                "available_seeds": None,
                "expected_locked_steps": None,
                "system_ref": None,
                "tools_ref": None,
            })
    return descriptors
