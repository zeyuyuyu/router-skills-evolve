#!/usr/bin/env python3
"""Evaluate a base or LoRA-finetuned code model on HumanEval-style samples.

This script is intentionally local/offline: it loads a HuggingFace causal LM,
generates code, then reuses the repository's HumanEval-style execution checks.
It is useful for measuring whether ``train_small_model.py`` improved the small
model on the hard tasks extracted from traces.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.train_small_model import format_prompt


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sample_to_task(sample: Dict) -> Dict:
    """Convert training_data.jsonl or HumanEval rows to a common task shape."""
    if {"prompt", "test", "entry_point"}.issubset(sample):
        return {
            "task_id": sample.get("task_id", ""),
            "prompt": sample["prompt"],
            "test": sample["test"],
            "entry_point": sample["entry_point"],
        }
    if {"instruction", "test", "entry_point"}.issubset(sample):
        return {
            "task_id": sample.get("task_id", ""),
            "prompt": sample["instruction"],
            "test": sample["test"],
            "entry_point": sample["entry_point"],
        }
    raise ValueError(
        "Each sample must contain either prompt/test/entry_point or "
        "instruction/test/entry_point"
    )


def iter_limited(rows: Iterable[Dict], limit: int | None) -> List[Dict]:
    rows = list(rows)
    return rows if limit is None else rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local base/LoRA code model")
    parser.add_argument("--data", required=True, help="JSONL with HumanEval-style tests")
    parser.add_argument("--base-model", required=True, help="HF base model or local path")
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter path")
    parser.add_argument("--output", default=None, help="Output JSON report")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--prompt-style",
        choices=["alpaca", "code", "qwen-chat"],
        default="alpaca",
        help="Prompt template; must match SFT training prompt style.",
    )
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent / data_path
    rows = iter_limited(load_jsonl(data_path), args.limit)
    tasks = [sample_to_task(row) for row in rows]
    print(f"Loaded {len(tasks)} eval tasks from {data_path}")

    if args.dry_run:
        print("--dry-run: data format OK, skipped model loading")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from src.models import extract_code, run_humaneval_test

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=args.device_map,
        quantization_config=quantization_config,
        trust_remote_code=args.trust_remote_code,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    results = []
    for idx, task in enumerate(tasks, start=1):
        prompt = format_prompt(task["prompt"], style=args.prompt_style)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        stop_token = "<|im_end|>" if args.prompt_style == "qwen-chat" else None
        stop_token_id = tokenizer.convert_tokens_to_ids(stop_token) if stop_token else None
        if isinstance(stop_token_id, int) and stop_token_id >= 0:
            generation_kwargs["eos_token_id"] = stop_token_id
        if args.temperature > 0:
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["top_p"] = args.top_p
        with torch.no_grad():
            generated = model.generate(**inputs, **generation_kwargs)
        new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        code = extract_code(completion, task["entry_point"], task["prompt"])
        success, error = run_humaneval_test(task, code)
        results.append(
            {
                "task_id": task.get("task_id", ""),
                "success": success,
                "error": error,
                "completion": completion,
                "generated_code": code,
            }
        )
        icon = "OK" if success else "FAIL"
        print(f"[{idx:03d}/{len(tasks):03d}] {task.get('task_id', '')}: {icon} {error}")

    success_count = sum(1 for row in results if row["success"])
    metrics = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "total": len(results),
        "success": success_count,
        "success_rate": success_count / len(results) if results else 0.0,
    }
    print("=" * 80)
    print(metrics)

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = Path(__file__).parent.parent / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"metrics": metrics, "results": results}, f, indent=2)
        print(f"Saved evaluation: {output_path}")


if __name__ == "__main__":
    main()
