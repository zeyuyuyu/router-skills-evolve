from training.data.heldout_split import (
    HELDOUT_LIMITS_BY_DOMAIN,
    expand_heldout_ids_to_descriptors,
    select_heldout_task_ids,
    split_rows_by_heldout,
)


SAMPLE_ROWS = [
    {"_p": {"domain": "airline", "task_id": "0"}},
    {"_p": {"domain": "airline", "task_id": "1"}},
    {"_p": {"domain": "airline", "task_id": "2"}},
    {"_p": {"domain": "airline", "task_id": "3"}},
    {"_p": {"domain": "airline", "task_id": "4"}},
    {"_p": {"domain": "airline", "task_id": "5"}},
    {"_p": {"domain": "retail", "task_id": "10"}},
    {"_p": {"domain": "retail", "task_id": "11"}},
    {"_p": {"domain": "retail", "task_id": "12"}},
    {"_p": {"domain": "retail", "task_id": "13"}},
    {"_p": {"domain": "retail", "task_id": "14"}},
    {"_p": {"domain": "retail", "task_id": "15"}},
    {"_p": {"domain": "telecom", "task_id": "20"}},
    {"_p": {"domain": "telecom", "task_id": "21"}},
    {"_p": {"domain": "telecom", "task_id": "22"}},
    {"_p": {"domain": "telecom", "task_id": "23"}},
    {"_p": {"domain": "telecom", "task_id": "24"}},
    {"_p": {"domain": "telecom", "task_id": "25"}},
]


def test_selects_n_per_domain():
    heldout = select_heldout_task_ids(SAMPLE_ROWS, n_per_domain=5, seed=42)
    assert set(heldout.keys()) == {"airline", "retail", "telecom"}
    for d, tids in heldout.items():
        assert len(tids) == 5, f"Expected 5 task_ids in {d}, got {len(tids)}"


def test_deterministic():
    h1 = select_heldout_task_ids(SAMPLE_ROWS, n_per_domain=5, seed=42)
    h2 = select_heldout_task_ids(SAMPLE_ROWS, n_per_domain=5, seed=42)
    assert h1 == h2


def test_different_seeds_differ():
    h1 = select_heldout_task_ids(SAMPLE_ROWS, n_per_domain=3, seed=42)
    h2 = select_heldout_task_ids(SAMPLE_ROWS, n_per_domain=3, seed=99)
    # Same task pool — different selections.
    assert h1 != h2


def test_split_filters_correctly():
    heldout_tids = {"airline": {"0", "1"}, "retail": {"10"}, "telecom": set()}
    train, heldout = split_rows_by_heldout(SAMPLE_ROWS, heldout_tids)
    train_keys = {(r["_p"]["domain"], r["_p"]["task_id"]) for r in train}
    heldout_keys = {(r["_p"]["domain"], r["_p"]["task_id"]) for r in heldout}
    assert ("airline", "0") in heldout_keys
    assert ("airline", "0") not in train_keys
    assert ("retail", "11") in train_keys


def test_too_few_unique_tasks_raises():
    """If a domain has fewer unique task_ids than n_per_domain, raise."""
    minimal_rows = [{"_p": {"domain": "airline", "task_id": "0"}}]
    import pytest
    with pytest.raises(ValueError, match="not enough unique task_ids"):
        select_heldout_task_ids(minimal_rows, n_per_domain=5, seed=42)


def test_expand_heldout_ids_to_descriptors_three_domains():
    """Each (domain, task_id) → one descriptor with the right limits."""
    ids = {"airline": ["7", "3"], "retail": ["12"], "telecom": ["20", "21"]}
    descriptors = expand_heldout_ids_to_descriptors(ids)
    # 5 total: 2 + 1 + 2.
    assert len(descriptors) == 5
    # task_ids sorted within each domain (determinism).
    airline_ids = [d["task_id"] for d in descriptors if d["domain"] == "airline"]
    assert airline_ids == ["3", "7"]
    # Each descriptor has the four required fields the harness consumes.
    for d in descriptors:
        for k in ("task_id", "domain", "max_steps", "max_errors"):
            assert k in d, f"missing required key {k!r}"
        # Limits match the per-domain table.
        assert d["max_steps"] == HELDOUT_LIMITS_BY_DOMAIN[d["domain"]]["max_steps"]
        assert d["max_errors"] == HELDOUT_LIMITS_BY_DOMAIN[d["domain"]]["max_errors"]


def test_expand_heldout_unknown_domain_raises():
    """Unknown domain → ValueError so we don't silently emit broken descriptors."""
    import pytest
    with pytest.raises(ValueError, match="No held-out limits configured"):
        expand_heldout_ids_to_descriptors({"banking": ["0"]})


def test_expand_heldout_accepts_set_input():
    """Accepts both list and set values per the input type."""
    descriptors = expand_heldout_ids_to_descriptors({"airline": {"5", "1"}})
    assert [d["task_id"] for d in descriptors] == ["1", "5"]


def test_expand_heldout_format_matches_eval_tasks_subset():
    """Descriptor schema is a subset of data_processed/stage2_v1/eval_tasks.jsonl
    in the four fields harness.load_eval_tasks_grouped consumes."""
    descriptors = expand_heldout_ids_to_descriptors({"airline": ["0"]})
    d = descriptors[0]
    # Same field names as eval_tasks.jsonl rows.
    assert d["task_id"] == "0"
    assert d["domain"] == "airline"
    assert isinstance(d["max_steps"], int)
    assert isinstance(d["max_errors"], int)
