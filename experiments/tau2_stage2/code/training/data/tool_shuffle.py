"""Tool-schema-order augmentation (Spec §5.5).

Shuffling the tool order per row exposes the model to multiple presentations
of the same tool set, teaching order-invariance. Effectively 2-3x data
augmentation at zero cost.
"""
from __future__ import annotations

import random


def shuffle_tools_in_schema(tools: list[dict], rng: random.Random) -> list[dict]:
    """Return a shuffled copy of the tools list.

    Args:
        tools: list of OpenAI-format tool definitions.
        rng: random.Random instance for deterministic shuffling.

    Returns:
        New list with same tools in shuffled order. The outer list is never
        mutated; tool dicts are shared by reference (callers must treat them
        as read-only). A fresh list is returned even for empty/singleton
        inputs.
    """
    shuffled = list(tools)
    if len(shuffled) > 1:
        rng.shuffle(shuffled)
    return shuffled
