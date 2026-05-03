#!/usr/bin/env python3
"""Tiny local GRPO/REINFORCE trainer for code tasks with executable tests.

This script avoids TRL so it can run on hosts where TRL and torch versions are
out of sync. It samples multiple completions for each prompt, executes the
task tests, then updates a LoRA adapter with group-relative rewards:

    advantage = reward - mean(rewards for the same prompt)

Only generated completion tokens are trained. This is intentionally small and
experimental; it is meant to validate the model-evolution loop before moving to
a larger RL stack.
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


def resolve_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).parent.parent / p
    return p


def build_lora_config(rank: int, alpha: int):
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def completion_logprob(model, input_ids, attention_mask, prompt_len: int):
    """Average log-probability of generated tokens only."""
    import torch

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    # labels position i predicts original token i+1, so generated tokens start at
    # original prompt_len. In shifted labels this corresponds to prompt_len - 1.
    start = max(prompt_len - 1, 0)
    mask = torch.zeros_like(token_log_probs, dtype=torch.bool)
    mask[:, start:] = True
    mask &= labels.ne(-100)
    selected = token_log_probs[mask]
    if selected.numel() == 0:
        return token_log_probs.mean() * 0.0
    return selected.mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local GRPO/REINFORCE trainer for code tasks")
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--n-generations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--prompt-style", choices=["alpaca", "code", "qwen-chat"], default="qwen-chat")
    parser.add_argument("--reward-baseline", choices=["group", "zero"], default="group")
    parser.add_argument("--kl-coef", type=float, default=0.0, help="Reserved for future KL penalty")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = iter_limited(load_jsonl(resolve_path(args.data)), args.limit)
    tasks = [sample_to_task(row) for row in rows]
    print(f"Loaded {len(tasks)} RL tasks")
    if args.dry_run:
        print("--dry-run: skipped model loading/training")
        return

    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.models import extract_code, run_humaneval_test

    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model = get_peft_model(model, build_lora_config(args.lora_r, args.lora_alpha))
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    stop_token_id = tokenizer.eos_token_id
    if args.prompt_style == "qwen-chat":
        qwen_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(qwen_end, int) and qwen_end >= 0:
            stop_token_id = qwen_end

    history = []
    for epoch in range(args.epochs):
        total_reward = 0.0
        total_samples = 0
        for task_idx, task in enumerate(tasks, start=1):
            prompt = format_prompt(task["prompt"], style=args.prompt_style)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            prompt_len = inputs["input_ids"].shape[-1]

            # Sample without gradient; we optimize log-prob of sampled completions below.
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=args.n_generations,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=stop_token_id,
                )

            rewards = []
            samples = []
            for seq in generated:
                completion_ids = seq[prompt_len:]
                completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
                code = extract_code(completion, task["entry_point"], task["prompt"])
                success, error = run_humaneval_test(task, code)
                reward = 1.0 if success else 0.0
                rewards.append(reward)
                samples.append((seq.unsqueeze(0), completion, code, error))

            mean_reward = sum(rewards) / len(rewards)
            losses = []
            for reward, (seq, _completion, _code, _error) in zip(rewards, samples):
                advantage = reward - mean_reward if args.reward_baseline == "group" else reward
                if abs(advantage) < 1e-8:
                    continue
                seq = seq.to(model.device)
                attention_mask = torch.ones_like(seq, device=model.device)
                avg_logprob = completion_logprob(model, seq, attention_mask, prompt_len)
                losses.append(-float(advantage) * avg_logprob)

            if losses:
                loss = torch.stack(losses).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_value = float(loss.detach().cpu())
            else:
                loss_value = 0.0

            total_reward += sum(rewards)
            total_samples += len(rewards)
            if task_idx % 10 == 0 or task_idx == len(tasks):
                print(
                    f"epoch={epoch + 1} task={task_idx}/{len(tasks)} "
                    f"batch_reward={mean_reward:.2f} avg_reward={total_reward / total_samples:.3f} "
                    f"loss={loss_value:.4f}"
                )

        history.append({"epoch": epoch + 1, "avg_reward": total_reward / max(total_samples, 1)})

    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(output_dir / "rl_training_info.json", "w") as f:
        json.dump({"args": vars(args), "history": history}, f, indent=2)
    print(f"Saved RL adapter: {output_dir}")


if __name__ == "__main__":
    main()
