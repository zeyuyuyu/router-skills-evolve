#!/usr/bin/env python3
"""Offline evaluation for a trained learnable router.

This replays labelled trace examples instead of calling LLMs. It estimates the
trade-off a router would make:
- SMALL prediction: pay small cost; if the example label is LARGE, fallback pays
  large cost too but final accuracy is protected.
- LARGE prediction: pay large cost directly.
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained learnable router offline")
    parser.add_argument("--model", required=True, help="Trained router directory")
    parser.add_argument("--traces", nargs="+", default=["data/traces/*.jsonl"])
    parser.add_argument(
        "--router-data",
        nargs="*",
        default=[],
        help="Additional supervised router JSONL files.",
    )
    parser.add_argument("--tasks", default="data/HumanEval.jsonl")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
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

    policy = LearnedRouterPolicy.from_pretrained(
        args.model,
        threshold=args.threshold,
        device=args.device,
    )
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

    correct = 0
    predicted_small = 0
    predicted_large = 0
    fallback_count = 0
    learned_cost = 0.0
    always_small_cost = 0.0
    always_large_cost = len(examples) * large_cost
    records = []

    for ex, prob_large in zip(examples, p_large):
        pred = ROUTE_LARGE if prob_large >= policy.threshold else ROUTE_SMALL
        correct += int(pred == ex.label)
        if pred == ROUTE_SMALL:
            predicted_small += 1
            learned_cost += small_cost
            if ex.label == ROUTE_LARGE:
                fallback_count += 1
                learned_cost += large_cost
        else:
            predicted_large += 1
            learned_cost += large_cost

        always_small_cost += small_cost
        if ex.label == ROUTE_LARGE:
            always_small_cost += large_cost

        records.append(
            {
                "task_id": ex.task_id,
                "label": ex.label_name,
                "prediction": "large" if pred == ROUTE_LARGE else "small",
                "p_large": prob_large,
                "signature": ex.signature,
            }
        )

    metrics = {
        "total_examples": len(examples),
        "class_counts": class_counts(examples),
        "threshold": policy.threshold,
        "accuracy": correct / len(examples),
        "predicted_small": predicted_small,
        "predicted_large": predicted_large,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / len(examples),
        "avg_small_cost": small_cost,
        "avg_large_cost": large_cost,
        "learned_cost": learned_cost,
        "always_small_with_fallback_cost": always_small_cost,
        "always_large_cost": always_large_cost,
        "learned_vs_always_large": learned_cost / always_large_cost if always_large_cost else 0,
    }

    print("=" * 80)
    print("Learnable router offline evaluation")
    print("=" * 80)
    for key, value in metrics.items():
        print(f"{key}: {value}")

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"metrics": metrics, "records": records}, f, indent=2)
        print(f"\nSaved evaluation: {output_path}")


if __name__ == "__main__":
    main()
