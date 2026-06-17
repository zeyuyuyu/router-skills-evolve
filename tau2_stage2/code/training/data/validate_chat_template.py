"""Pre-launch chat-template validation (Spec §5.3).

For each prompt/completion record:
- Verify no pre-existing ``<think></think>`` blocks (would double-insert
  under ``enable_thinking=False``).
- Apply the chat template with ``enable_thinking=False`` and verify it
  succeeds.
- Verify token-level round-trip via encode → decode preserves the
  ``<|im_start|>assistant`` marker.

Hard-stops training if any row fails.

Note: orphan-tool-role checks are intentionally NOT called from
``validate_row``. τ²-bench is dual-actor — the user agent can issue tool
calls, producing legitimate ``tool`` messages with no preceding *assistant*
``tool_call``. The chat template renders such "orphans" correctly. We rely
on ``apply_chat_template``'s own pass/fail as the verdict for tool-call
structure. ``has_orphan_tool_in_prompt`` is preserved as a diagnostic
helper for ad-hoc analysis.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def contains_pre_existing_thinking_block(row: dict) -> bool:
    """Check if any message content contains ``<think>`` / ``</think>`` tokens."""
    for msg in (row.get("prompt") or []) + (row.get("completion") or []):
        c = msg.get("content") or ""
        if "<think>" in c or "</think>" in c:
            return True
    return False


def has_orphan_tool_in_prompt(row: dict) -> bool:
    """Diagnostic: any ``tool``-role message in the prompt with no matching tool_call.

    NOT called from ``validate_row`` — kept for analysis/debug only. τ²-bench
    has legitimate user-side tool calls that look like orphans but render
    fine through the chat template.
    """
    seen_ids: set[str] = set()
    for msg in row.get("prompt") or []:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                tcid = tc.get("id")
                if tcid is not None:
                    seen_ids.add(tcid)
        elif msg.get("role") == "tool":
            tcid = msg.get("tool_call_id")
            if tcid is None or tcid not in seen_ids:
                return True
    return False


def validate_row(row: dict, tokenizer) -> tuple[bool, str]:
    """Return ``(ok, reason)``. ``reason`` is empty string when ok."""
    if contains_pre_existing_thinking_block(row):
        return False, "pre_existing_thinking_block"

    full_msgs = list(row["prompt"]) + list(row["completion"])
    tools = row.get("tools") or None
    try:
        rendered = tokenizer.apply_chat_template(
            full_msgs,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except Exception as e:
        return False, f"chat_template_apply_error: {type(e).__name__}: {e}"

    try:
        ids = tokenizer.encode(rendered)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if "<|im_start|>assistant" not in decoded:
            return False, "round_trip_lost_assistant_marker"
    except Exception as e:
        return False, f"encode_decode_error: {type(e).__name__}: {e}"

    return True, ""


def validate_jsonl(src_jsonl: Path, tokenizer, out_report: Path) -> dict:
    """Validate every row in a jsonl file.

    Writes a JSON report to ``out_report`` and returns the summary dict.
    """
    n_total, n_pass, n_fail = 0, 0, 0
    failures: list[dict] = []
    reason_counts: dict[str, int] = {}

    with src_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ok, reason = validate_row(row, tokenizer)
            n_total += 1
            if ok:
                n_pass += 1
            else:
                n_fail += 1
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                failures.append(
                    {
                        "row_id": row.get("_meta", {}).get("row_id", "?"),
                        "reason": reason,
                    }
                )

    summary = {
        "n_total": n_total,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "reason_counts": reason_counts,
        "failures": failures[:100],
    }
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Validate chat-template renderability of stage-2 training data."
    )
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument(
        "--tokenizer",
        required=True,
        help="HF tokenizer id, e.g. Qwen/Qwen2.5-1.5B-Instruct",
    )
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    from training.model_resolution import (
        from_pretrained_kwargs,
        resolve_model_source,
    )

    source = resolve_model_source(args.tokenizer)
    if source.local_path:
        print(
            f"MODEL-RESOLUTION: {source.original_name} -> {source.local_path}",
            file=sys.stderr,
        )
    tok = AutoTokenizer.from_pretrained(
        source.name,
        **from_pretrained_kwargs(source, trust_remote_code=True),
    )
    summary = validate_jsonl(args.src, tok, args.out)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "failures"}, indent=2
        )
    )
    if summary["n_fail"] > 0:
        print(
            f"\n!! {summary['n_fail']} rows failed validation. Inspect {args.out}.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
