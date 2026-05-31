"""Convert stage-2 row format to TRL prompt/completion format (Spec §5.1).

Each row becomes:
    {
      "prompt":     [system, *messages[0:_target_index]],
      "completion": [messages[_target_index]],
      "tools":      <tools schema for the row's domain>,
      "_meta":      <provenance from row._p>
    }

With TRL SFTConfig(completion_only_loss=True, assistant_only_loss=False),
loss is computed only on the completion span. Intermediate assistant turns
in the prompt are correctly masked.

`convert_row` itself is lossless. `convert_dataset` drops rows whose prompt
prefix (msgs[:_target_index]) contains NO user message — Qwen3.5/3.6 chat
templates raise `Exception("No user query found in messages.")` on such rows,
which causes `validate_chat_template.py` (run by `train_all.sh:64-78`) to
exit 1 before any GPU work. There are 257 such rows in stage2_v1
(single-message [assistant] greeting rows with no preceding user turn).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_THINK_PAIRED = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_ORPHAN_CLOSE = re.compile(r"\A.*?</think>\s*", re.DOTALL)
_THINK_ORPHAN_OPEN = re.compile(r"<think>.*", re.DOTALL)


def _sanitize_message(msg: dict) -> dict:
    """Return a new message dict with chat-template-safe content.

    - `content: None` becomes `""` (Qwen Jinja templates can't concatenate None).
    - Pre-existing thinking artifacts (`<think>...</think>`, orphan `</think>`
      preamble, or orphan `<think>` opener with no close) are stripped. Data
      is supposed to be CoT-free; leaving any of these in place would
      double-inject under Qwen3+ with `enable_thinking=False`.
    """
    out = dict(msg)
    c = out.get("content")
    if c is None:
        out["content"] = ""
        return out
    if isinstance(c, str):
        if "<think>" in c and "</think>" in c:
            c = _THINK_PAIRED.sub("", c)
        if "</think>" in c:
            c = _THINK_ORPHAN_CLOSE.sub("", c)
        if "<think>" in c:
            c = _THINK_ORPHAN_OPEN.sub("", c)
        out["content"] = c
    return out


def load_domain_assets(domain_assets_dir: Path) -> tuple[dict[str, str], dict[str, list]]:
    """Load all *_system.txt and *_tools.json from domain_assets/.

    Returns (system_texts, tools_per_domain) keyed by domain name.
    """
    system_texts: dict[str, str] = {}
    tools_per_domain: dict[str, list] = {}
    for p in sorted(domain_assets_dir.glob("*_system.txt")):
        domain = p.stem.replace("_system", "")
        system_texts[domain] = p.read_text()
    for p in sorted(domain_assets_dir.glob("*_tools.json")):
        domain = p.stem.replace("_tools", "")
        tools_per_domain[domain] = json.loads(p.read_text())
    return system_texts, tools_per_domain


def convert_row(row: dict, system_text: str, tools: list) -> dict:
    """Convert one stage-2 row to TRL prompt/completion format.

    Sanitizes every message via `_sanitize_message` (None content → "",
    strips pre-existing `<think>...</think>` blocks).
    """
    msgs = row["messages"]
    ti = row["_target_index"]
    prompt = [{"role": "system", "content": system_text}] + [
        _sanitize_message(m) for m in msgs[:ti]
    ]
    completion = [_sanitize_message(msgs[ti])]
    return {
        "prompt": prompt,
        "completion": completion,
        "tools": tools,
        "_meta": dict(row["_p"]),
    }


def convert_dataset(
    src_jsonl: Path,
    dst_jsonl: Path,
    domain_assets_dir: Path,
    drop_log: Path | None = None,
) -> dict:
    """Convert an entire jsonl file. Sanitizes every message; drops user-less rows.

    Filter: drops any row whose prompt prefix (`msgs[:_target_index]`) lacks
    a user message — Qwen3.5/3.6 chat templates raise "No user query found
    in messages." for such rows, blocking train_all.sh's per-tokenizer
    chat-template validation before any GPU work. `convert_row` stays
    lossless; the filter lives here so the data-loss decision is explicit.

    If `drop_log` is provided, dropped rows are recorded one-per-line with
    their reason + provenance so the operator can audit what was discarded.
    """
    system_texts, tools_per_domain = load_domain_assets(domain_assets_dir)
    n_kept = 0
    n_dropped_no_user = 0

    dst_jsonl.parent.mkdir(parents=True, exist_ok=True)
    drop_fout = None
    if drop_log is not None:
        drop_log.parent.mkdir(parents=True, exist_ok=True)
        drop_fout = drop_log.open("w")
    try:
        with src_jsonl.open() as fin, dst_jsonl.open("w") as fout:
            for line in fin:
                row = json.loads(line)
                msgs = row["messages"]
                ti = row["_target_index"]
                prefix = msgs[:ti]
                if not any(m["role"] == "user" for m in prefix):
                    n_dropped_no_user += 1
                    if drop_fout is not None:
                        drop_fout.write(json.dumps({
                            "reason": "no_user_in_prefix",
                            "row_id": row.get("_p", {}).get("row_id"),
                            "domain": row.get("_p", {}).get("domain"),
                            "task_id": row.get("_p", {}).get("task_id"),
                            "_target_index": ti,
                            "n_msgs": len(msgs),
                            "prefix_roles": [m["role"] for m in prefix],
                        }) + "\n")
                    continue
                domain = row["_p"]["domain"]
                converted = convert_row(
                    row,
                    system_text=system_texts[domain],
                    tools=tools_per_domain[domain],
                )
                fout.write(json.dumps(converted) + "\n")
                n_kept += 1
    finally:
        if drop_fout is not None:
            drop_fout.close()

    return {"n_kept": n_kept, "n_dropped_no_user": n_dropped_no_user}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--domain-assets", required=True, type=Path)
    ap.add_argument(
        "--drop-log",
        type=Path,
        default=None,
        help="Optional jsonl path; rows dropped by the converter are written here.",
    )
    args = ap.parse_args(argv)

    stats = convert_dataset(
        args.src, args.dst, args.domain_assets, drop_log=args.drop_log
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
