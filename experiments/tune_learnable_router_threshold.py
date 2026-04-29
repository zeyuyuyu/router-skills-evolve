#!/usr/bin/env python3
"""Tune the decision threshold for a trained learnable router.

The router outputs P(route_to_large). Lower thresholds route more prompts
directly to the large model, reducing fallback risk but increasing cost. This
script sweeps thresholds offline and selects the cheapest threshold that meets
simple safety constraints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import LARGE_MODEL, SMALL_MODEL
from src.learned_router.data import (
    ROUTE_LARGE,
    ROUTE_SMALL,
    class_counts,
    estimate_attempt_costs,
    load_combined_router_examples,
)
from src.learned_router.policy import LearnedRouterPolicy


def evaluate_threshold(examples, p_large, threshold: float, small_cost: float, large_cost: float) -> dict:
    correct = 0
    predicted_small = 0
    predicted_large = 0
    fallback_count = 0
    learned_cost = 0.0
    always_small_cost = 0.0
    always_large_cost = len(examples) * large_cost

    true_large = sum(1 for ex in examples if ex.label == ROUTE_LARGE)
    true_small = len(examples) - true_large
    tp_large = 0
    fp_large = 0
    fn_large = 0

    for ex, prob in zip(examples, p_large):
        pred = ROUTE_LARGE if prob >= threshold else ROUTE_SMALL
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

        always_small_cost += small_cost
        if ex.label == ROUTE_LARGE:
            always_small_cost += large_cost

    precision_large = tp_large / (tp_large + fp_large) if (tp_large + fp_large) else 0.0
    recall_large = tp_large / true_large if true_large else 0.0
    f1_large = (
        2 * precision_large * recall_large / (precision_large + recall_large)
        if (precision_large + recall_large)
        else 0.0
    )

    return {
        "threshold": threshold,
        "accuracy": correct / len(examples),
        "precision_large": precision_large,
        "recall_large": recall_large,
        "f1_large": f1_large,
        "predicted_small": predicted_small,
        "predicted_large": predicted_large,
        "true_small": true_small,
        "true_large": true_large,
        "tp_large": tp_large,
        "fp_large": fp_large,
        "fn_large": fn_large,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / len(examples),
        "fallback_rate_on_large": fallback_count / true_large if true_large else 0.0,
        "learned_cost": learned_cost,
        "always_small_with_fallback_cost": always_small_cost,
        "always_large_cost": always_large_cost,
        "learned_vs_always_large": learned_cost / always_large_cost if always_large_cost else 0.0,
        "learned_vs_always_small_with_fallback": learned_cost / always_small_cost if always_small_cost else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune learned-router threshold offline")
    parser.add_argument("--model", required=True, help="Trained router directory")
    parser.add_argument("--traces", nargs="+", default=["data/traces/*.jsonl"])
    parser.add_argument("--router-data", nargs="*", default=[])
    parser.add_argument("--tasks", default="data/HumanEval.jsonl")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--start", type=float, default=0.05)
    parser.add_argument("--end", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument(
        "--max-fallback-rate",
        type=float,
        default=0.02,
        help="Select cheapest threshold with fallback_rate <= this value.",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.90,
        help="Select cheapest threshold with route accuracy >= this value.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    tasks_path = Path(args.tasks)
    if not tasks_path.is_absolute():
        tasks_path = root / tasks_path

    examples = load_combined_router_examples(
        trace_patterns=args.traces,
        tasks_path=tasks_path,
        router_data_patterns=args.router_data,
        root=root,
    )
    if not examples:
        raise RuntimeError("No labelled router examples found")

    policy = LearnedRouterPolicy.from_pretrained(args.model, device=args.device)
    p_large = policy.router.predict_proba(
        [ex.prompt for ex in examples],
        batch_size=args.batch_size,
        device=policy.device,
    )

    small_cost, large_cost = estimate_attempt_costs(
        args.traces,
        root=root,
        small_model=SMALL_MODEL,
        large_model=LARGE_MODEL,
    )

    thresholds = []
    current = args.start
    while current <= args.end + 1e-9:
        thresholds.append(round(current, 6))
        current += args.step

    rows = [
        evaluate_threshold(examples, p_large, threshold, small_cost, large_cost)
        for threshold in thresholds
    ]

    feasible = [
        row
        for row in rows
        if row["fallback_rate"] <= args.max_fallback_rate and row["accuracy"] >= args.min_accuracy
    ]
    recommended = min(feasible, key=lambda row: row["learned_cost"]) if feasible else None

    best_accuracy = max(rows, key=lambda row: row["accuracy"])
    cheapest = min(rows, key=lambda row: row["learned_cost"])
    lowest_fallback = min(rows, key=lambda row: (row["fallback_rate"], row["learned_cost"]))

    payload = {
        "total_examples": len(examples),
        "class_counts": class_counts(examples),
        "avg_small_cost": small_cost,
        "avg_large_cost": large_cost,
        "constraints": {
            "max_fallback_rate": args.max_fallback_rate,
            "min_accuracy": args.min_accuracy,
        },
        "recommended": recommended,
        "best_accuracy": best_accuracy,
        "cheapest": cheapest,
        "lowest_fallback": lowest_fallback,
        "sweep": rows,
    }

    print("=" * 80)
    print("Learnable router threshold tuning")
    print("=" * 80)
    print(f"total_examples: {payload['total_examples']}")
    print(f"class_counts: {payload['class_counts']}")
    print(f"avg_small_cost: {small_cost}")
    print(f"avg_large_cost: {large_cost}")
    print(f"constraints: {payload['constraints']}")
    print(f"recommended: {recommended}")
    print(f"best_accuracy: {best_accuracy}")
    print(f"cheapest: {cheapest}")
    print(f"lowest_fallback: {lowest_fallback}")

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved threshold tuning: {output_path}")


if __name__ == "__main__":
    main()
