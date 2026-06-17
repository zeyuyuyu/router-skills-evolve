import random

from training.data.tool_shuffle import shuffle_tools_in_schema


SAMPLE_TOOLS = [
    {"type": "function", "function": {"name": "tool_a", "parameters": {}}},
    {"type": "function", "function": {"name": "tool_b", "parameters": {}}},
    {"type": "function", "function": {"name": "tool_c", "parameters": {}}},
    {"type": "function", "function": {"name": "tool_d", "parameters": {}}},
]


def test_shuffle_preserves_count():
    rng = random.Random(42)
    shuffled = shuffle_tools_in_schema(SAMPLE_TOOLS, rng)
    assert len(shuffled) == len(SAMPLE_TOOLS)


def test_shuffle_preserves_set_of_tools():
    rng = random.Random(42)
    shuffled = shuffle_tools_in_schema(SAMPLE_TOOLS, rng)
    names = {t["function"]["name"] for t in shuffled}
    expected = {t["function"]["name"] for t in SAMPLE_TOOLS}
    assert names == expected


def test_shuffle_actually_permutes():
    rng = random.Random(42)
    shuffled = shuffle_tools_in_schema(SAMPLE_TOOLS, rng)
    original_order = [t["function"]["name"] for t in SAMPLE_TOOLS]
    new_order = [t["function"]["name"] for t in shuffled]
    # With seed 42 and 4! = 24 permutations, we expect a non-identity perm.
    assert original_order != new_order


def test_shuffle_does_not_mutate_input():
    rng = random.Random(42)
    snapshot = [dict(t) for t in SAMPLE_TOOLS]
    _ = shuffle_tools_in_schema(SAMPLE_TOOLS, rng)
    assert SAMPLE_TOOLS == snapshot


def test_shuffle_empty():
    rng = random.Random(0)
    empty = []
    out = shuffle_tools_in_schema(empty, rng)
    assert out == []
    assert out is not empty  # always a fresh list


def test_shuffle_singleton():
    rng = random.Random(0)
    one = [SAMPLE_TOOLS[0]]
    out = shuffle_tools_in_schema(one, rng)
    assert out == one
    assert out is not one  # always a fresh list
