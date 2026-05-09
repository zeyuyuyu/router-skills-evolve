#!/usr/bin/env python3
"""Offline end-to-end ablation for skills/router/model-evolution tracks.

The paper discussion should not report isolated "skills", "router training",
and "LLM training" numbers only. This script puts them into one comparable
table:

1. Base always-small with fallback.
2. SkillBook-style signature routing.
3. Optional learned router checkpoint.
4. Optional LLM base/adapter eval summaries.

The routing part is offline and uses labelled router examples, so it does not
call LLM APIs. Labels mean:

- small: small model should pass.
- large: small model would fail and large/fallback is needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

SMALL_MODEL = "deepseek/deepseek-v3.2"
LARGE_MODEL = "openai/gpt-5.4-2026-03-05"
ROUTE_SMALL = 0
ROUTE_LARGE = 1
LABEL_TO_ROUTE = {ROUTE_SMALL: "small", ROUTE_LARGE: "large"}


def evaluate_predictions(examples, predictions: List[int], small_cost: float, large_cost: float) -> Dict:
    correct = 0
    predicted_small = 0
    predicted_large = 0
    fallback_count = 0
    learned_cost = 0.0
    true_large = sum(1 for ex in examples if ex.label == ROUTE_LARGE)
    true_small = len(examples) - true_large
    tp_large = fp_large = fn_large = 0

    always_large_cost = len(examples) * large_cost
    always_small_cost = 0.0
    for ex in examples:
        always_small_cost += small_cost
        if ex.label == ROUTE_LARGE:
            always_small_cost += large_cost

    for ex, pred in zip(examples, predictions):
        correct += int(pred == ex.label)
        if pred == ROUTE_LARGE:
            predicted_large += 1
            learned_cost += large_cost
            if ex.label == ROUTE_LARGE:
                tp_large += 1
            else:
                fp_large += 1
        else:
            predicted_small += 1
            learned_cost += small_cost
            if ex.label == ROUTE_LARGE:
                fallback_count += 1
                fn_large += 1
                learned_cost += large_cost

    precision_large = tp_large / (tp_large + fp_large) if (tp_large + fp_large) else 0.0
    recall_large = tp_large / true_large if true_large else 0.0
    f1_large = (
        2 * precision_large * recall_large / (precision_large + recall_large)
        if (precision_large + recall_large)
        else 0.0
    )
    return {
        "total_examples": len(examples),
        "true_small": true_small,
        "true_large": true_large,
        "accuracy": correct / len(examples) if examples else 0.0,
        "precision_large": precision_large,
        "recall_large": recall_large,
        "f1_large": f1_large,
        "predicted_small": predicted_small,
        "predicted_large": predicted_large,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / len(examples) if examples else 0.0,
        "fallback_rate_on_large": fallback_count / true_large if true_large else 0.0,
        "cost": learned_cost,
        "always_small_with_fallback_cost": always_small_cost,
        "always_large_cost": always_large_cost,
        "cost_vs_always_large": learned_cost / always_large_cost if always_large_cost else 0.0,
        "cost_vs_always_small_with_fallback": learned_cost / always_small_cost if always_small_cost else 0.0,
    }


class Example:
    def __init__(self, task_id: str, prompt: str, label: int, signature: str):
        self.task_id = task_id
        self.prompt = prompt
        self.label = label
        self.signature = signature


def read_jsonl(path: Path) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def expand_paths(patterns: Iterable[str], root: Path) -> List[Path]:
    import glob

    paths = []
    for pattern in patterns:
        raw = Path(pattern)
        glob_pattern = str(raw if raw.is_absolute() else root / raw)
        matches = sorted(glob.glob(glob_pattern))
        if matches:
            paths.extend(Path(m) for m in matches)
        elif raw.exists():
            paths.append(raw)
        elif (root / raw).exists():
            paths.append(root / raw)
    return paths


def infer_signature(prompt: str) -> str:
    p = prompt.lower()
    length = len(prompt)
    if length < 200:
        len_b = "S"
    elif length < 500:
        len_b = "M"
    else:
        len_b = "L"
    tags = []
    if any(k in p for k in ["list", "array"]):
        tags.append("list")
    if any(k in p for k in ["string", "str"]):
        tags.append("str")
    if any(k in p for k in ["number", "integer", "float"]):
        tags.append("num")
    if any(k in p for k in ["sort", "order"]):
        tags.append("sort")
    if any(k in p for k in ["prime", "factor", "divisor"]):
        tags.append("theory")
    if any(k in p for k in ["encode", "decode", "cipher", "cyclic", "shift"]):
        tags.append("crypto")
    if any(k in p for k in ["poly", "recursion", "recursive"]):
        tags.append("advanced")
    if any(k in p for k in ["true", "false", "check", "boolean"]):
        tags.append("bool")
    if not tags:
        tags.append("general")
    return f"{len_b}|{'/'.join(sorted(set(tags)))}"


def load_humaneval_prompts(tasks_path: Path) -> Dict[str, str]:
    return {row["task_id"]: row["prompt"] for row in read_jsonl(tasks_path)}


def trace_label(trace: Dict) -> Optional[int]:
    attempts = trace.get("attempts")
    small_success = False
    small_failed = False
    large_success = False
    if isinstance(attempts, list):
        for attempt in attempts:
            model = attempt.get("model", "")
            success = bool(attempt.get("success", False))
            if "deepseek" in model:
                small_success = small_success or success
                small_failed = small_failed or not success
            if "gpt-5.4" in model:
                large_success = large_success or success
        if small_success:
            return ROUTE_SMALL
        if small_failed and large_success:
            return ROUTE_LARGE
    elif isinstance(attempts, int):
        decision = str(trace.get("decision", "")).lower()
        final_model = trace.get("final_model", "")
        if "fallback" in decision or attempts >= 2 or "gpt-5.4" in final_model:
            return ROUTE_LARGE
        if "small_ok" in decision or "use_small" in decision or "deepseek" in final_model:
            return ROUTE_SMALL
    return None


def load_trace_examples(trace_patterns: Iterable[str], tasks_path: Path, root: Path) -> List[Example]:
    prompts = load_humaneval_prompts(tasks_path)
    examples = []
    seen = set()
    for path in expand_paths(trace_patterns, root):
        for trace in read_jsonl(path):
            task_id = trace.get("task_id", "")
            if task_id in seen or task_id not in prompts:
                continue
            label = trace_label(trace)
            if label is None:
                continue
            prompt = prompts[task_id]
            examples.append(Example(task_id, prompt, label, trace.get("signature") or infer_signature(prompt)))
            seen.add(task_id)
    return examples


def load_router_data_examples(patterns: Iterable[str], root: Path) -> List[Example]:
    examples = []
    seen = set()
    for path in expand_paths(patterns, root):
        for row in read_jsonl(path):
            prompt = row.get("prompt", "")
            label = row.get("label")
            label_name = str(row.get("label_name", "")).lower()
            if label is None and label_name == "small":
                label = ROUTE_SMALL
            if label is None and label_name == "large":
                label = ROUTE_LARGE
            if label not in {ROUTE_SMALL, ROUTE_LARGE} or not prompt:
                continue
            key = row.get("task_id") or prompt
            if key in seen:
                continue
            seen.add(key)
            examples.append(
                Example(
                    key,
                    prompt,
                    int(label),
                    row.get("signature") or infer_signature(prompt),
                )
            )
    return examples


def load_combined_examples(trace_patterns: Iterable[str], tasks_path: Path, router_data_patterns: Iterable[str], root: Path) -> List[Example]:
    return load_trace_examples(trace_patterns, tasks_path, root) + load_router_data_examples(router_data_patterns, root)


def split_examples(examples: List[Example], eval_ratio: float, seed: int):
    import random

    by_label = {ROUTE_SMALL: [], ROUTE_LARGE: []}
    for ex in examples:
        by_label.setdefault(ex.label, []).append(ex)
    rng = random.Random(seed)
    train, eval_ = [], []
    for group in by_label.values():
        rng.shuffle(group)
        if len(group) <= 1:
            train.extend(group)
            continue
        n_eval = max(1, int(round(len(group) * eval_ratio)))
        n_eval = min(n_eval, len(group) - 1)
        eval_.extend(group[:n_eval])
        train.extend(group[n_eval:])
    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


def class_counts(examples: Iterable[Example]) -> Dict[str, int]:
    counts = {"small": 0, "large": 0}
    for ex in examples:
        counts[LABEL_TO_ROUTE[ex.label]] += 1
    return counts


def estimate_costs(trace_patterns: Iterable[str], root: Path):
    small_costs, large_costs = [], []
    for path in expand_paths(trace_patterns, root):
        for trace in read_jsonl(path):
            attempts = trace.get("attempts")
            if not isinstance(attempts, list):
                continue
            for attempt in attempts:
                model = attempt.get("model", "")
                cost = float(attempt.get("cost", 0.0) or 0.0)
                if cost <= 0:
                    continue
                if "deepseek" in model:
                    small_costs.append(cost)
                elif "gpt-5.4" in model:
                    large_costs.append(cost)
    return (
        sum(small_costs) / len(small_costs) if small_costs else 0.00009531933333333333,
        sum(large_costs) / len(large_costs) if large_costs else 0.00525,
    )


def train_skillbook_policy(examples, min_samples: int) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"small": 0, "large": 0})
    for ex in examples:
        key = ex.signature
        if ex.label == ROUTE_LARGE:
            stats[key]["large"] += 1
        else:
            stats[key]["small"] += 1
    return dict(stats)


def predict_skillbook(stats: Dict[str, Dict[str, int]], examples, large_rate_threshold: float, min_samples: int) -> List[int]:
    preds = []
    global_counts = Counter()
    for values in stats.values():
        global_counts["small"] += values["small"]
        global_counts["large"] += values["large"]
    default_pred = ROUTE_LARGE if global_counts["large"] > global_counts["small"] else ROUTE_SMALL

    for ex in examples:
        values = stats.get(ex.signature)
        if not values:
            preds.append(default_pred)
            continue
        total = values["small"] + values["large"]
        if total < min_samples:
            preds.append(default_pred)
            continue
        large_rate = values["large"] / total if total else 0.0
        preds.append(ROUTE_LARGE if large_rate >= large_rate_threshold else ROUTE_SMALL)
    return preds


def load_eval_metrics(path: str | Path | None) -> Optional[Dict]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return data.get("metrics", data)


def write_markdown(path: Path, payload: Dict) -> None:
    rows = []
    for name, metrics in payload["routing"].items():
        rows.append(
            "| {name} | {accuracy:.2%} | {f1:.2%} | {fallback:.2%} | {cost:.2%} | {large} |".format(
                name=name,
                accuracy=metrics["accuracy"],
                f1=metrics["f1_large"],
                fallback=metrics["fallback_rate"],
                cost=metrics["cost_vs_always_large"],
                large=metrics["predicted_large"],
            )
        )

    text = [
        "# End-to-End Ablation Report",
        "",
        "## Routing ablation",
        "",
        "| Variant | Accuracy | Large F1 | Fallback Rate | Cost vs Always-Large | Predicted Large |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "## LLM evolution",
        "",
    ]
    if payload.get("llm"):
        for name, metrics in payload["llm"].items():
            text.append(f"- **{name}**: `{metrics}`")
    else:
        text.append("- No LLM eval metrics provided.")
    text.append("")
    text.append("## Dataset")
    text.append("")
    text.append(f"- Train examples for SkillBook: {payload['dataset']['train_examples']}")
    text.append(f"- Eval examples: {payload['dataset']['eval_examples']}")
    text.append(f"- Eval class counts: `{payload['dataset']['eval_class_counts']}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text))


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline e2e ablation for evolver")
    parser.add_argument("--traces", nargs="+", default=["data/traces/*.jsonl"])
    parser.add_argument("--router-data", nargs="*", default=[])
    parser.add_argument("--tasks", default="data/HumanEval.jsonl")
    parser.add_argument("--learned-router-model", default="")
    parser.add_argument("--learned-threshold", type=float, default=0.5)
    parser.add_argument("--eval-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skill-large-rate-threshold", type=float, default=0.5)
    parser.add_argument("--skill-min-samples", type=int, default=1)
    parser.add_argument("--llm-base-eval", default="")
    parser.add_argument("--llm-adapter-eval", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default="")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    tasks_path = Path(args.tasks)
    if not tasks_path.is_absolute():
        tasks_path = root / tasks_path

    examples = load_combined_examples(args.traces, tasks_path, args.router_data, root)
    train_examples, eval_examples = split_examples(examples, args.eval_ratio, args.seed)
    small_cost, large_cost = estimate_costs(args.traces, root)

    routing = {}
    routing["base_always_small_fallback"] = evaluate_predictions(
        eval_examples,
        [ROUTE_SMALL] * len(eval_examples),
        small_cost,
        large_cost,
    )
    routing["always_large"] = evaluate_predictions(
        eval_examples,
        [ROUTE_LARGE] * len(eval_examples),
        small_cost,
        large_cost,
    )

    skill_stats = train_skillbook_policy(train_examples, args.skill_min_samples)
    routing["skillbook_signature"] = evaluate_predictions(
        eval_examples,
        predict_skillbook(
            skill_stats,
            eval_examples,
            args.skill_large_rate_threshold,
            args.skill_min_samples,
        ),
        small_cost,
        large_cost,
    )

    if args.learned_router_model:
        from src.learned_router.policy import LearnedRouterPolicy

        policy = LearnedRouterPolicy.from_pretrained(
            args.learned_router_model,
            threshold=args.learned_threshold,
        )
        probs = policy.router.predict_proba([ex.prompt for ex in eval_examples])
        routing["learned_router"] = evaluate_predictions(
            eval_examples,
            [ROUTE_LARGE if p >= args.learned_threshold else ROUTE_SMALL for p in probs],
            small_cost,
            large_cost,
        )

    llm = {}
    base_metrics = load_eval_metrics(args.llm_base_eval)
    adapter_metrics = load_eval_metrics(args.llm_adapter_eval)
    if base_metrics:
        llm["base"] = base_metrics
    if adapter_metrics:
        llm["adapter"] = adapter_metrics

    payload = {
        "args": vars(args),
        "dataset": {
            "all_examples": len(examples),
            "train_examples": len(train_examples),
            "eval_examples": len(eval_examples),
            "eval_class_counts": class_counts(eval_examples),
            "small_cost": small_cost,
            "large_cost": large_cost,
        },
        "routing": routing,
        "llm": llm,
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    print(f"Saved: {output}")

    if args.markdown_output:
        md = Path(args.markdown_output)
        if not md.is_absolute():
            md = root / md
        write_markdown(md, payload)
        print(f"Saved markdown: {md}")


if __name__ == "__main__":
    main()
