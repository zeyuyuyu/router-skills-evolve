"""Trace loading utilities for training a learnable router.

The training label is conservative:
- ``small``: historical trace shows the cheap model solved the task.
- ``large``: cheap model failed and the expensive model was needed.

This converts existing Router+Skills traces into supervised data for a
classifier. The classifier learns from prompt text instead of hand-written
signature buckets.
"""

from __future__ import annotations

import glob
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.config import LARGE_MODEL, SMALL_MODEL, calc_cost
from src.skills import extract_signature


ROUTE_SMALL = 0
ROUTE_LARGE = 1
LABEL_TO_ROUTE = {ROUTE_SMALL: "small", ROUTE_LARGE: "large"}
ROUTE_TO_LABEL = {v: k for k, v in LABEL_TO_ROUTE.items()}


@dataclass(frozen=True)
class RouterTraceExample:
    """A single supervised router training example."""

    task_id: str
    prompt: str
    label: int
    label_name: str
    signature: str
    source_trace: str
    small_success: bool
    large_success: Optional[bool]

    def to_dict(self) -> Dict:
        return asdict(self)


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_humaneval_prompts(path: Path) -> Dict[str, str]:
    """Load HumanEval prompts keyed by task_id."""
    prompts: Dict[str, str] = {}
    for task in load_jsonl(path):
        task_id = task.get("task_id")
        prompt = task.get("prompt")
        if task_id and prompt:
            prompts[task_id] = prompt
    return prompts


def expand_trace_paths(patterns: Sequence[str], root: Optional[Path] = None) -> List[Path]:
    """Expand one or more trace paths/globs."""
    paths: List[Path] = []
    base = root or Path.cwd()
    for pattern in patterns:
        raw = Path(pattern)
        glob_pattern = str(raw if raw.is_absolute() else base / raw)
        matches = sorted(glob.glob(glob_pattern))
        if matches:
            paths.extend(Path(m) for m in matches)
        elif raw.exists():
            paths.append(raw)
        elif not raw.is_absolute() and (base / raw).exists():
            paths.append(base / raw)

    seen = set()
    deduped: List[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(path)
    return deduped


def _model_matches(model_id: str, expected: str, loose_name: str) -> bool:
    model_id = model_id or ""
    return model_id == expected or loose_name in model_id.lower()


def infer_label_from_trace(
    trace: Dict,
    small_model: str = SMALL_MODEL,
    large_model: str = LARGE_MODEL,
) -> Tuple[Optional[int], bool, Optional[bool]]:
    """Infer route label from one trace record.

    Supports both the current trace format where ``attempts`` is a list and
    older summary traces where ``attempts`` is an int plus decision/final_model.
    """
    attempts = trace.get("attempts")
    small_success = False
    small_seen = False
    small_failed = False
    large_success: Optional[bool] = None

    if isinstance(attempts, list):
        for attempt in attempts:
            model = attempt.get("model", "")
            success = bool(attempt.get("success", False))
            if _model_matches(model, small_model, "deepseek"):
                small_seen = True
                small_success = small_success or success
                small_failed = small_failed or not success
            elif _model_matches(model, large_model, "gpt-5.4"):
                large_success = success

        if small_success:
            return ROUTE_SMALL, True, large_success
        if small_failed and large_success:
            return ROUTE_LARGE, False, True
        if not small_seen:
            final_model = trace.get("final_model", "")
            final_success = bool(trace.get("final_success", trace.get("success", False)))
            if _model_matches(final_model, large_model, "gpt-5.4") and final_success:
                return ROUTE_LARGE, False, True
        return None, small_success, large_success

    if isinstance(attempts, int):
        final_model = trace.get("final_model", "")
        decision = trace.get("decision", "")
        final_success = bool(trace.get("final_success", trace.get("success", False)))

        if _model_matches(final_model, small_model, "deepseek") and (
            "small_ok" in decision.lower() or "use_small" in decision.lower() or final_success
        ):
            return ROUTE_SMALL, True, None
        if attempts >= 2 or "fallback" in decision.lower() or _model_matches(final_model, large_model, "gpt-5.4"):
            large_ok = final_success if _model_matches(final_model, large_model, "gpt-5.4") else None
            return ROUTE_LARGE, False, large_ok

    return None, small_success, large_success


def load_router_examples(
    trace_patterns: Sequence[str],
    tasks_path: Path,
    *,
    root: Optional[Path] = None,
    small_model: str = SMALL_MODEL,
    large_model: str = LARGE_MODEL,
    dedupe_by_task: bool = True,
) -> List[RouterTraceExample]:
    """Load supervised examples from traces and HumanEval prompts."""
    task_prompts = load_humaneval_prompts(tasks_path)
    trace_paths = expand_trace_paths(trace_patterns, root=root)
    if not trace_paths:
        raise FileNotFoundError(f"No trace files matched: {trace_patterns}")

    examples: List[RouterTraceExample] = []
    seen_tasks = set()
    for trace_path in trace_paths:
        for trace in load_jsonl(trace_path):
            task_id = trace.get("task_id", "")
            if not task_id or task_id not in task_prompts:
                continue
            if dedupe_by_task and task_id in seen_tasks:
                continue

            label, small_success, large_success = infer_label_from_trace(
                trace, small_model=small_model, large_model=large_model
            )
            if label is None:
                continue

            prompt = task_prompts[task_id]
            examples.append(
                RouterTraceExample(
                    task_id=task_id,
                    prompt=prompt,
                    label=label,
                    label_name=LABEL_TO_ROUTE[label],
                    signature=trace.get("signature") or extract_signature(prompt),
                    source_trace=str(trace_path),
                    small_success=small_success,
                    large_success=large_success,
                )
            )
            seen_tasks.add(task_id)
    return examples


def split_examples(
    examples: Sequence[RouterTraceExample],
    eval_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[RouterTraceExample], List[RouterTraceExample]]:
    """Deterministic stratified-ish train/eval split."""
    import random

    by_label: Dict[int, List[RouterTraceExample]] = {ROUTE_SMALL: [], ROUTE_LARGE: []}
    for example in examples:
        by_label.setdefault(example.label, []).append(example)

    train: List[RouterTraceExample] = []
    eval_: List[RouterTraceExample] = []
    rng = random.Random(seed)
    for group in by_label.values():
        rng.shuffle(group)
        if len(group) <= 1:
            train.extend(group)
            continue
        eval_size = max(1, int(round(len(group) * eval_ratio)))
        eval_size = min(eval_size, len(group) - 1)
        eval_.extend(group[:eval_size])
        train.extend(group[eval_size:])
    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


def class_counts(examples: Iterable[RouterTraceExample]) -> Dict[str, int]:
    counts = {"small": 0, "large": 0}
    for example in examples:
        counts[example.label_name] += 1
    return counts


def estimate_attempt_costs(
    trace_patterns: Sequence[str],
    root: Optional[Path] = None,
    small_model: str = SMALL_MODEL,
    large_model: str = LARGE_MODEL,
) -> Tuple[float, float]:
    """Estimate average small/large call cost from traces."""
    small_costs: List[float] = []
    large_costs: List[float] = []

    for path in expand_trace_paths(trace_patterns, root=root):
        for trace in load_jsonl(path):
            attempts = trace.get("attempts")
            if not isinstance(attempts, list):
                continue
            for attempt in attempts:
                model = attempt.get("model", "")
                cost = float(attempt.get("cost", 0.0) or 0.0)
                if cost <= 0:
                    continue
                if _model_matches(model, small_model, "deepseek"):
                    small_costs.append(cost)
                elif _model_matches(model, large_model, "gpt-5.4"):
                    large_costs.append(cost)

    # Practical defaults based on short HumanEval prompts; real traces override.
    return (
        sum(small_costs) / len(small_costs) if small_costs else calc_cost(small_model, 300, 300),
        sum(large_costs) / len(large_costs) if large_costs else calc_cost(large_model, 300, 300),
    )
