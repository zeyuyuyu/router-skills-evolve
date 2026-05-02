#!/usr/bin/env python3
"""Build DPO pairs for MBPP-style code tasks.

For each task we use the reference solution as ``chosen`` and ask a base model
to generate a candidate. Candidates that fail the task tests become
``rejected``. This gives the LLM-evolution track a preference-learning dataset
instead of another pure imitation/SFT dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.evaluate_finetuned_model import load_jsonl, sample_to_task
from experiments.train_small_model import format_prompt


def iter_limited(rows: Iterable[Dict], limit: int | None) -> List[Dict]:
    rows = list(rows)
    return rows if limit is None else rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MBPP DPO data from base-model failures")
    parser.add_argument("--data", required=True, help="MBPP-style JSONL with output/test/entry_point")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prompt-style", choices=["alpaca", "code", "qwen-chat"], default="qwen-chat")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent / data_path
    rows = iter_limited(load_jsonl(data_path), args.limit)
    tasks = [sample_to_task(row) for row in rows]
    print(f"Loaded {len(tasks)} tasks from {data_path}")

    if args.dry_run:
        print("--dry-run: skipped model generation")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.models import extract_code, run_humaneval_test

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    pairs = []
    for idx, (row, task) in enumerate(zip(rows, tasks), start=1):
        prompt = format_prompt(task["prompt"], style=args.prompt_style)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        stop_token = "<|im_end|>" if args.prompt_style == "qwen-chat" else None
        stop_token_id = tokenizer.convert_tokens_to_ids(stop_token) if stop_token else tokenizer.eos_token_id
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_token_id,
            )
        new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
        rejected_completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        rejected_code = extract_code(rejected_completion, task["entry_point"], task["prompt"])
        success, error = run_humaneval_test(task, rejected_code)
        status = "skip-pass" if success else "pair"
        print(f"[{idx:03d}/{len(tasks):03d}] {task['task_id']}: {status} {error}")
        if success:
            continue
        pairs.append(
            {
                "task_id": task["task_id"],
                "prompt": task["prompt"],
                "chosen": row["output"],
                "rejected": rejected_code,
                "rejected_error": error,
                "source": row.get("source", "mbpp"),
                "prompt_style": args.prompt_style,
            }
        )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent.parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} DPO pairs: {output_path}")


if __name__ == "__main__":
    main()
