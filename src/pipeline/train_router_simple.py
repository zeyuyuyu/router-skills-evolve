#!/usr/bin/env python3
"""Bench-agnostic router trainer for the scaling pipeline.

Reads `prompt` + `small_success` from each trace row and trains a per-prompt
router: label = 1 if the small model failed (needs large), 0 if it succeeded.

Featurizers (compared, best kept) — set `ROUTER_FEATURIZER`:
  - tfidf : TfidfVectorizer(4096,1-2gram) + LogReg          (no GPU, default-safe)
  - emb   : Qwen2.5-Coder-1.5B last-token embedding + LogReg (needs torch+GPU)
  - both  : train both, report CV AUC for each, KEEP THE BEST   (default)

The embedding router (`EmbeddingRouter`) re-embeds prompts at inference, so the
saved `router.joblib` is self-contained: `joblib.load(...).predict_proba(prompts)`
works in collect_traces / run_e2e_ablation_simple exactly like the tfidf pipeline.

Serialization: `joblib`. Router artifacts are INTERNAL — only load ones you produced.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


try:
    from .router_emb import EmbeddingRouter  # deployable embedding router
except ImportError:  # script run without package context
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.pipeline.router_emb import EmbeddingRouter


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


def _cv_auc(make_estimator, X, y, seed=42):
    """Repeated stratified 5-fold mean AUC/acc/F1 (robust on small sets)."""
    import numpy as np
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    y = np.asarray(y)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
    aucs, accs, f1s = [], [], []
    for tr, te in cv.split(np.zeros(len(y)), y):
        est = make_estimator()
        Xtr = X[tr] if hasattr(X, "__getitem__") and not isinstance(X, list) else [X[i] for i in tr]
        Xte = X[te] if hasattr(X, "__getitem__") and not isinstance(X, list) else [X[i] for i in te]
        est.fit(Xtr, y[tr])
        p = est.predict_proba(Xte)[:, 1]
        try:
            aucs.append(roc_auc_score(y[te], p))
        except ValueError:
            pass
        pred = (p >= 0.5).astype(int)
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, pos_label=1, zero_division=0))
    return {"auc": float(np.mean(aucs)) if aucs else float("nan"),
            "acc": float(np.mean(accs)), "f1_large": float(np.mean(f1s))}


def train(prompts, labels, seed: int = 42):
    """Train tfidf and/or embedding routers per ROUTER_FEATURIZER; keep best by CV AUC.
    Returns (best_fitted_estimator, meta_dict)."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    which = os.environ.get("ROUTER_FEATURIZER", "both").lower()
    emb_model = os.environ.get("ROUTER_EMB_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")

    if len(set(labels)) < 2:
        print(f"[router] WARN only one class ({set(labels)}); adding synthetic row", file=sys.stderr)
        prompts = list(prompts) + ["__synthetic_complementary_class__"]
        labels = list(labels) + [1 - labels[0]]
    y = list(labels)

    def mk_tfidf():
        return Pipeline([("tfidf", TfidfVectorizer(max_features=4096, ngram_range=(1, 2))),
                         ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    def mk_logreg():
        return LogisticRegression(max_iter=1000, class_weight="balanced")

    results = {}   # name -> cv metrics
    candidates = {}  # name -> (fit_fn, predict-ready estimator builder)

    # --- TF-IDF candidate ---
    if which in ("tfidf", "both"):
        try:
            results["tfidf"] = _cv_auc(mk_tfidf, prompts, y, seed)
            print(f"[router] tfidf      CV: auc={results['tfidf']['auc']:.3f} "
                  f"acc={results['tfidf']['acc']:.3f} f1={results['tfidf']['f1_large']:.3f}", file=sys.stderr)
            candidates["tfidf"] = "tfidf"
        except Exception as e:  # noqa: BLE001
            print(f"[router] tfidf failed: {e}", file=sys.stderr)

    # --- Embedding candidate ---
    Xemb = None
    if which in ("emb", "both"):
        try:
            er = EmbeddingRouter(emb_model, mk_logreg())
            Xemb = er.embed(prompts)
            results["emb"] = _cv_auc(mk_logreg, Xemb, y, seed)
            print(f"[router] emb({emb_model.split('/')[-1]}) CV: auc={results['emb']['auc']:.3f} "
                  f"acc={results['emb']['acc']:.3f} f1={results['emb']['f1_large']:.3f}", file=sys.stderr)
            candidates["emb"] = "emb"
        except Exception as e:  # noqa: BLE001
            print(f"[router] embedding featurizer unavailable ({type(e).__name__}: {e}); "
                  f"using tfidf only", file=sys.stderr)

    if not results:
        raise RuntimeError("no router featurizer succeeded")

    # pick best by CV AUC (nan-safe)
    best = max(results, key=lambda k: (results[k]["auc"] if results[k]["auc"] == results[k]["auc"] else -1))
    print(f"[router] BEST featurizer = {best} (auc={results[best]['auc']:.3f})", file=sys.stderr)

    # fit best on ALL data
    if best == "tfidf":
        est = mk_tfidf(); est.fit(prompts, y)
    else:
        clf = mk_logreg(); clf.fit(Xemb, y)
        est = EmbeddingRouter(emb_model, clf)

    meta = {"chosen_featurizer": best, "cv_metrics": results,
            "metrics": results[best],  # back-compat: top-level metrics = chosen
            "n_examples_total": len(prompts),
            "label_distribution": {"0_small_ok": y.count(0), "1_need_large": y.count(1)},
            "seed": seed, "emb_model": emb_model, "featurizer_mode": which}
    return est, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts, labels, skipped = load_examples(Path(args.traces))
    print(f"[router] loaded {len(prompts)} examples ({sum(labels)} label=1, "
          f"{len(prompts)-sum(labels)} label=0); skipped {skipped}", file=sys.stderr)
    if len(prompts) < 4:
        print(f"[router] ERROR too few examples ({len(prompts)}); need ≥4.", file=sys.stderr)
        sys.exit(3)

    est, meta = train(prompts, labels, seed=args.seed)
    try:
        import joblib
    except ImportError:
        print("[router] ERROR: joblib not installed.", file=sys.stderr)
        sys.exit(2)
    joblib.dump(est, out_dir / "router.joblib")
    meta["threshold_default"] = 0.5
    (out_dir / "router_meta.json").write_text(json.dumps(meta, indent=2))
    m = meta["metrics"]
    print(f"[router] DONE  chosen={meta['chosen_featurizer']}  auc={m['auc']:.3f} "
          f"acc={m['acc']:.3f}  out={out_dir}/router.joblib", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
