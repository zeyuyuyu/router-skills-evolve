#!/usr/bin/env python3
"""Train a BERT-style learnable router from traces.

The router learns a binary policy:
  0 -> try the cheap/small model first
  1 -> route directly to the expensive/large model

This is intentionally different from ``train_small_model*.py``. Those scripts
fine-tune the generator; this script trains the router itself.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import LARGE_MODEL, SMALL_MODEL
from src.learned_router.data import (
    class_counts,
    load_router_examples,
    split_examples,
)
from src.learned_router.model import BertRouter, BertRouterConfig


def check_deps() -> None:
    missing = []
    for pkg in ["torch", "transformers", "datasets", "sklearn"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append("scikit-learn" if pkg == "sklearn" else pkg)
    if missing:
        print(f"Missing deps: {', '.join(missing)}")
        print("Install with: pip install torch transformers datasets scikit-learn")
        sys.exit(1)


def examples_to_dataset(examples, tokenizer, max_length: int):
    from datasets import Dataset

    dataset = Dataset.from_list(
        [{"prompt": ex.prompt, "label": ex.label, "task_id": ex.task_id} for ex in examples]
    )

    def tokenize(batch):
        encoded = tokenizer(
            batch["prompt"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        encoded["labels"] = batch["label"]
        return encoded

    return dataset.map(tokenize, batched=True, remove_columns=["prompt", "label", "task_id"])


def compute_metrics_fn():
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="binary",
            pos_label=1,
            zero_division=0,
        )
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision_large": precision,
            "recall_large": recall,
            "f1_large": f1,
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Train a learnable BERT router from traces")
    parser.add_argument(
        "--traces",
        nargs="+",
        default=["data/traces/*.jsonl"],
        help="Trace JSONL path(s) or glob(s)",
    )
    parser.add_argument("--tasks", default="data/HumanEval.jsonl", help="HumanEval JSONL path")
    parser.add_argument(
        "--base-model",
        default="google/bert_uncased_L-2_H-128_A-2",
        help="HF encoder model",
    )
    parser.add_argument("--output", default="outputs/learned-router", help="Output directory")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--eval-ratio", type=float, default=0.25)
    parser.add_argument("--epochs", type=float, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    tasks_path = Path(args.tasks)
    if not tasks_path.is_absolute():
        tasks_path = root / tasks_path
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    examples = load_router_examples(args.traces, tasks_path, root=root)
    if not examples:
        raise RuntimeError("No supervised router examples could be built from traces")

    train_examples, eval_examples = split_examples(examples, args.eval_ratio, args.seed)
    print("=" * 80)
    print("Learnable router data")
    print("=" * 80)
    print(f"Total: {len(examples)} {class_counts(examples)}")
    print(f"Train: {len(train_examples)} {class_counts(train_examples)}")
    print(f"Eval : {len(eval_examples)} {class_counts(eval_examples)}")
    print(f"Example: {examples[0].task_id} -> {examples[0].label_name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "router_dataset.jsonl", "w") as f:
        for example in examples:
            f.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")

    if args.dry_run:
        print("--dry-run: wrote dataset preview, skipped model training")
        return

    check_deps()

    import torch
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

    cfg = BertRouterConfig(
        base_model=args.base_model,
        max_length=args.max_length,
        small_model=SMALL_MODEL,
        large_model=LARGE_MODEL,
        threshold=args.threshold,
    )
    router = BertRouter.from_pretrained_base(cfg)

    train_dataset = examples_to_dataset(train_examples, router.tokenizer, args.max_length)
    eval_dataset = examples_to_dataset(eval_examples, router.tokenizer, args.max_length) if eval_examples else None

    train_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=1,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="macro_f1" if eval_dataset is not None else None,
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=router.model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=router.tokenizer,
        data_collator=DataCollatorWithPadding(router.tokenizer),
        compute_metrics=compute_metrics_fn() if eval_dataset is not None else None,
    )
    trainer.train()
    metrics = trainer.evaluate() if eval_dataset is not None else {}

    router.save(output_dir)
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(
            {
                "total_examples": len(examples),
                "train_examples": len(train_examples),
                "eval_examples": len(eval_examples),
                "class_counts": class_counts(examples),
                "metrics": metrics,
                "args": vars(args),
            },
            f,
            indent=2,
        )
    print(f"Saved learned router to {output_dir}")


if __name__ == "__main__":
    main()
