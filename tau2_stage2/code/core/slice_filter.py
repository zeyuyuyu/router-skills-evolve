"""Deterministic slice taxonomy per PROJECT_CHARTER.md §3.5.

Sole source of truth for training eligibility AND Exp-3 student-substitution
eligibility. Phase-2 LLM output does NOT inform it.
"""
from __future__ import annotations

import json
from typing import Any

from core.schemas.artifacts import SliceType

_ANALYTICAL_INPUT_CHAR_THRESHOLD = 8000   # ~2k tokens @ ~4 chars/token
_ANALYTICAL_TOOL_RESULTS_THRESHOLD = 3


def classify_slice(step: dict[str, Any]) -> SliceType:
    """step = {input_messages, response, input_meta?}."""
    messages: list[dict] = step["input_messages"]
    response: dict = step["response"]
    meta: dict = step.get("input_meta") or {}

    tool_calls = response.get("tool_calls") or []
    content = (response.get("content") or "").strip()

    if len(tool_calls) >= 2:
        return "multi_tool"

    tool_results_in_input = sum(1 for m in messages if m.get("role") == "tool")
    total_input_chars = sum(len(m.get("content") or "") for m in messages)

    if len(tool_calls) == 1:
        if tool_results_in_input == 0:
            return "single_tool_call"
        return "other"

    if meta.get("response_format") and _is_json(content):
        return "strict_json"

    if tool_results_in_input == 1 and _looks_short_or_json(content):
        return "tool_result_to_json"

    if _is_clarify(content):
        return "clarify"

    if (total_input_chars > _ANALYTICAL_INPUT_CHAR_THRESHOLD
            or tool_results_in_input >= _ANALYTICAL_TOOL_RESULTS_THRESHOLD):
        return "analytical"

    return "other"


def _is_json(content: str) -> bool:
    content = content.strip()
    if not content.startswith(("{", "[")):
        return False
    try:
        json.loads(content)
        return True
    except json.JSONDecodeError:
        return False


def _looks_short_or_json(content: str) -> bool:
    return _is_json(content) or (0 < len(content) <= 500)


def _is_clarify(content: str) -> bool:
    if not content:
        return False
    if len(content) > 200:
        return False
    return content.rstrip().endswith("?")
