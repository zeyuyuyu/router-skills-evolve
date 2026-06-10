#!/usr/bin/env python3
"""Bench-agnostic E2E ablation for the scaling pipeline.

Why this exists: `experiments/run_e2e_ablation.py` is hard-coupled to
`data/HumanEval.jsonl`. This script consumes the bench-agnostic trace schema
(see `experiments/scaling/benches/__init__.py`) plus a router artifact from
`train_router_simple.py`, and produces four-variant numbers in the same
schema the main-branch ablation uses (so aggregate_cycles.py just works).

Variants:
    base    always-small (no skills, no router)
    large   always-large baseline
    skills  SkillBook-style signature routing (no learned router)
    router  + learned router
    full    + LLM training (LLM column populated only if --llm-eval given;
            otherwise full == router for routing metrics, and task_pass uses
            small_success/large_success from traces)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load_traces(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def true_label(trace: dict) -> int | None:
    """1 = small failed (needed large); 0 = small was enough."""
    s = trace.get("small_success")
    if isinstance(s, bool):
        return 0 if s else 1
    decision = (trace.get("decision") or "").lower()
    if "small_ok" in decision:
        return 0
    if "fail" in decision or "large" in decision:
        return 1
    return None


def score_routing(labels: list[int], preds: list[int]) -> dict:
    if not labels:
        return {"routing_acc": 0, "large_f1": 0, "fallback": 0, "cost_vs_large": 0,
                "n_eval": 0}
    n = len(labels)
    acc = sum(int(a == b) for a, b in zip(labels, preds)) / n
    # F1 on the "large" class (label=1)
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
    # Fallback rate = predicted-small but actually needed-large
    fallback = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0) / n
    # Cost relative to always-large. Approximate: small=1, large=10 (placeholder).
    SMALL_COST, LARGE_COST = 0.001, 0.01
    predicted_large = sum(preds)
    cost = predicted_large * LARGE_COST + (n - predicted_large) * SMALL_COST
    cost_vs_large = cost / (n * LARGE_COST)
    return {
        "routing_acc": acc,
        "large_f1": f1,
        "fallback": fallback,
        "cost_vs_large": cost_vs_large,
        "n_eval": n,
    }


def predict_base(traces: list[dict]) -> list[int]:
    """Always small (0)."""
    return [0 for _ in traces]


def predict_large(traces: list[dict]) -> list[int]:
    """Always large (1). Requires traces where large_success is real."""
    return [1 for _ in traces]


def predict_skills(traces: list[dict], skillbook_path: Path | None) -> list[int]:
    """SkillBook signature-based routing.

    Predict large (1) if the signature has historical large-success rate
    above threshold; else small (0). Falls back to always-small if no skillbook.
    """
    if skillbook_path is None or not skillbook_path.exists():
        return predict_base(traces)
    try:
        sb_data = json.loads(skillbook_path.read_text())
    except Exception:  # noqa: BLE001
        return predict_base(traces)

    # tofix.md #5: SkillBook.save() writes {"skills": [{"signature", "stats":
    # {model_or_role: [successes, total]}, ...}]}. The old code did
    # sb_data.get(sig) on the top-level dict (which only has key "skills"), so
    # every lookup missed and `skills` silently degraded to always-small.
    # Build sig -> stats from the real schema. Stats are keyed by canonical role
    # "small"/"large" (tofix.md #2); we also tolerate legacy model-name keys.
    by_sig: dict[str, dict] = {}
    skills_list = sb_data.get("skills") if isinstance(sb_data, dict) else None
    if isinstance(skills_list, list):
        for sk in skills_list:
            if isinstance(sk, dict) and sk.get("signature"):
                by_sig[sk["signature"]] = sk.get("stats", {}) or {}
    elif isinstance(sb_data, dict):
        # legacy flat {sig: {...}} fallback
        for sig, entry in sb_data.items():
            if isinstance(entry, dict):
                by_sig[sig] = entry.get("stats", entry)

    def _rate(stat) -> float:
        # stat is [successes, total] or a dict with success_rate
        if isinstance(stat, (list, tuple)) and len(stat) == 2 and stat[1]:
            return stat[0] / stat[1]
        if isinstance(stat, dict):
            return float(stat.get("success_rate", 0) or 0)
        return 0.0

    def needs_large(sig: str) -> bool:
        stats = by_sig.get(sig)
        if not isinstance(stats, dict) or not stats:
            return False
        small_score = large_score = 0.0
        for key, stat in stats.items():
            r = _rate(stat)
            k = key.lower()
            if k == "small" or "deepseek" in k or "qwen" in k:
                small_score = max(small_score, r)
            elif k == "large" or "gpt" in k or "claude" in k:
                large_score = max(large_score, r)
        return large_score > small_score + 0.2

    return [1 if needs_large(t.get("signature", "")) else 0 for t in traces]


def predict_router(traces: list[dict], router_dir: Path, threshold: float) -> list[int]:
    """Learned router from train_router_simple.py."""
    if not (router_dir / "router.joblib").exists():
        return predict_base(traces)
    try:
        import joblib
    except ImportError:
        return predict_base(traces)
    pipe = joblib.load(router_dir / "router.joblib")
    prompts = [t.get("prompt") or t.get("signature", "") for t in traces]
    # Use predict_proba if available, fall back to predict
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(prompts)
        return [1 if p[1] >= threshold else 0 for p in proba]
    return [int(x) for x in pipe.predict(prompts)]


def task_pass_rate(traces: list[dict], predictions: list[int]) -> float:
    """Simulated task pass rate under a given routing decision.

    If routed to small: success = small_success
    If routed to large: success = large_success (or final_success as fallback)
    """
    n_ok = 0
    n_total = 0
    for t, pred in zip(traces, predictions):
        if pred == 0:
            s = t.get("small_success")
            ok = s if isinstance(s, bool) else t.get("final_success", False)
        else:
            l = t.get("large_success")
            ok = l if isinstance(l, bool) else t.get("final_success", False)
        n_total += 1
        if ok:
            n_ok += 1
    return n_ok / n_total if n_total else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--skillbook", default="")
    ap.add_argument("--router-dir", default="")
    ap.add_argument("--router-threshold", type=float, default=0.5)
    ap.add_argument("--llm-task-pass", type=float, default=None,
                    help="Optional: explicit task-pass rate for the LLM variant "
                         "if you have a separate LLM eval. Otherwise full=router.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--markdown-output", default="")
    args = ap.parse_args()

    traces = load_traces(Path(args.traces))
    if not traces:
        print("[ablation] ERROR no traces loaded", file=sys.stderr)
        return 2

    labels = [true_label(t) for t in traces]
    keep = [(t, l) for t, l in zip(traces, labels) if l is not None]
    traces_kept = [x[0] for x in keep]
    labels_kept = [x[1] for x in keep]
    print(f"[ablation] {len(traces)} rows, {len(traces_kept)} with labels", file=sys.stderr)

    variants = {}
    # base
    p_base = predict_base(traces_kept)
    variants["base"] = score_routing(labels_kept, p_base)
    variants["base"]["task_pass"] = task_pass_rate(traces_kept, p_base)
    # always-large baseline
    p_large = predict_large(traces_kept)
    variants["large"] = score_routing(labels_kept, p_large)
    variants["large"]["task_pass"] = task_pass_rate(traces_kept, p_large)
    # skills
    p_sk = predict_skills(traces_kept, Path(args.skillbook) if args.skillbook else None)
    variants["skills"] = score_routing(labels_kept, p_sk)
    variants["skills"]["task_pass"] = task_pass_rate(traces_kept, p_sk)
    # router
    p_r = predict_router(traces_kept, Path(args.router_dir) if args.router_dir else Path(""),
                        args.router_threshold)
    variants["router"] = score_routing(labels_kept, p_r)
    variants["router"]["task_pass"] = task_pass_rate(traces_kept, p_r)
    # full = router routing + LLM-improved small (if --llm-task-pass given)
    variants["full"] = dict(variants["router"])
    if args.llm_task_pass is not None:
        variants["full"]["task_pass"] = float(args.llm_task_pass)

    out = {
        "variants": variants,
        "label_distribution": dict(Counter(labels_kept)),
        "n_traces_total": len(traces),
        "n_traces_with_label": len(traces_kept),
        "router_threshold": args.router_threshold,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"[ablation] DONE  out={args.output}", file=sys.stderr)
    for v, m in variants.items():
        print(f"  {v:8s}  routing_acc={m['routing_acc']:.2%}  task_pass={m['task_pass']:.2%}",
              file=sys.stderr)

    if args.markdown_output:
        lines = ["| Variant | Routing Acc | Large F1 | Fallback | Cost vs Large | Task Pass |",
                 "|---|---:|---:|---:|---:|---:|"]
        for v in ["base", "skills", "router", "full"]:
            m = variants[v]
            lines.append(f"| {v} | {m['routing_acc']:.2%} | {m['large_f1']:.2%} "
                         f"| {m['fallback']:.2%} | {m['cost_vs_large']:.2%} | {m['task_pass']:.2%} |")
        Path(args.markdown_output).write_text("\n".join(lines) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
