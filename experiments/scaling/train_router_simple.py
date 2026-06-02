#!/usr/bin/env python3
"""Bench-agnostic router trainer for the scaling pipeline.

Why this exists: `experiments/train_learnable_router.py` is hard-coupled to
`data/HumanEval.jsonl` (looks up prompts by task_id from HumanEval). For tau2
or SWE-Bench traces, that lookup fails. This trainer reads `prompt` directly
from the trace row, so it works on any bench whose adapter writes the standard
trace schema (see `experiments/scaling/benches/__init__.py`).

Algorithm: TF-IDF over prompt → LogisticRegression. Label = 1 if the small
model failed (i.e. we needed to route to large), 0 if small succeeded.

Serialization: uses `joblib` (sklearn's recommended format). The router
artifact is INTERNAL — only load `.joblib` files you yourself produced.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_examples(traces_path: Path):
    prompts: list[str] = []
    labels: list[int] = []
    skipped = 0
    with traces_path.open() as fh:
        for line in fh:
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            prompt = t.get("prompt") or t.get("signature") or ""
            if not prompt:
                skipped += 1
                continue

            small_ok = t.get("small_success")
            if isinstance(small_ok, bool):
                label = 0 if small_ok else 1
            else:
                decision = (t.get("decision") or "").lower()
                if "small_ok" in decision:
                    label = 0
                elif "fail" in decision or "large" in decision:
                    label = 1
                else:
                    skipped += 1
                    continue

            prompts.append(prompt)
            labels.append(label)

    return prompts, labels, skipped


def train(prompts, labels, seed: int = 42):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError:
        print("[router] ERROR: scikit-learn not installed. `pip install scikit-learn`", file=sys.stderr)
        sys.exit(2)

    # Edge case: all-same labels. This is a real signal (e.g. the small model
    # failed every tau2 task), so fit a constant router instead of trying a
    # stratified split with a synthetic one-off minority class.
    if len(set(labels)) < 2:
        from sklearn.dummy import DummyClassifier
        from sklearn.metrics import accuracy_score, f1_score

        constant = labels[0]
        print(f"[router] WARN only one class in labels ({set(labels)}); "
              f"using constant router={constant}", file=sys.stderr)
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=4096, ngram_range=(1, 2))),
            ("clf", DummyClassifier(strategy="constant", constant=constant)),
        ])
        # Include one synthetic complementary row so predict_proba exposes both
        # columns for downstream thresholding, while metrics stay on real rows.
        fit_prompts = list(prompts) + ["__synthetic_complementary_class__"]
        fit_labels = list(labels) + [1 - constant]
        pipe.fit(fit_prompts, fit_labels)
        y_pred = pipe.predict(prompts)
        return pipe, {
            "n_train": len(prompts),
            "n_eval": len(prompts),
            "accuracy": float(accuracy_score(labels, y_pred)),
            "f1_large": float(f1_score(labels, y_pred, pos_label=1, zero_division=0)),
            "constant_router": int(constant),
        }

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=4096, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=400, random_state=seed, class_weight="balanced")),
    ])

    stratify = labels if len(set(labels)) > 1 else None
    X_train, X_eval, y_train, y_eval = train_test_split(
        prompts, labels, test_size=0.25, random_state=seed, stratify=stratify
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_eval)
    metrics = {
        "n_train": len(X_train),
        "n_eval": len(X_eval),
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "f1_large": float(f1_score(y_eval, y_pred, pos_label=1, zero_division=0)),
    }
    return pipe, metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    traces_path = Path(args.traces)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts, labels, skipped = load_examples(traces_path)
    print(f"[router] loaded {len(prompts)} examples ({sum(labels)} label=1, "
          f"{len(prompts)-sum(labels)} label=0); skipped {skipped}", file=sys.stderr)
    if len(prompts) < 4:
        print(f"[router] ERROR too few examples ({len(prompts)}); need ≥4. "
              f"Check upstream collect_traces output.", file=sys.stderr)
        sys.exit(3)

    pipe, metrics = train(prompts, labels, seed=args.seed)

    try:
        import joblib
    except ImportError:
        print("[router] ERROR: joblib not installed. `pip install joblib`", file=sys.stderr)
        sys.exit(2)
    joblib.dump(pipe, out_dir / "router.joblib")
    meta = {
        "metrics": metrics,
        "threshold_default": 0.5,
        "n_examples_total": len(prompts),
        "label_distribution": {"0_small_ok": labels.count(0), "1_need_large": labels.count(1)},
        "seed": args.seed,
        "vectorizer": "tfidf(4096, 1-2gram)",
        "classifier": "LogisticRegression(class_weight=balanced)",
    }
    (out_dir / "router_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[router] DONE  acc={metrics['accuracy']:.3f}  f1_large={metrics['f1_large']:.3f}  "
          f"out={out_dir}/router.joblib", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
