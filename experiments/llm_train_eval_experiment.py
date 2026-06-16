#!/usr/bin/env python3
"""
实验：SFT vs GRPO vs base 性能对比

流程：
  1. 从已有 traces 提取 SFT 数据（skills 提炼后的 traces）
  2a. SFT 训练 Qwen2.5-Coder-1.5B LoRA
  2b. GRPO 训练 Qwen2.5-Coder-1.5B LoRA（直接用 HumanEval test 做 reward）
  3. 在 eval split（82 题）上对比 base vs SFT vs GRPO

用法：
  CUDA_VISIBLE_DEVICES=6 python3 experiments/llm_train_eval_experiment.py \
      --traces results/scaling_20260616_183651/cycle_0/traces.jsonl \
      --skillbook results/scaling_20260616_183651/cycle_0/skillbook.json \
      --out results/llm_train_experiment
"""
import argparse
import gc
import json
import os
import signal
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "scaling"))
sys.path.insert(0, str(REPO / "experiments" / "scaling" / "benches"))

SMALL_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def _free_gpu():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────
# Step 1: SFT 数据提取
# ─────────────────────────────────────────────────────────────

def step1_extract_sft(traces_path, skillbook_path, out_dir):
    print("\n" + "="*60)
    print("Step 1: 提取 SFT 数据")
    print("="*60)

    from traces_to_sft import convert
    sft_path = out_dir / "training_data.jsonl"
    stats = convert(traces_path, sft_path, skillbook_path=skillbook_path)
    print(f"traces_total   : {stats['traces_total']}")
    print(f"hard_tasks     : {stats['hard_tasks']}  (small_fail & large_ok)")
    print(f"sft_pairs      : {stats['written']}")
    print(f"with_procedure : {stats.get('with_procedure', 0)}")
    print(f"→ {sft_path}")
    return sft_path, stats


# ─────────────────────────────────────────────────────────────
# Step 2a: SFT LoRA 训练
# ─────────────────────────────────────────────────────────────

def step2a_sft(sft_path, out_dir):
    print("\n" + "="*60)
    print("Step 2a: SFT LoRA 训练 Qwen2.5-Coder-1.5B")
    print("="*60)

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig

    adapter_dir = out_dir / "sft_adapter"
    rows = [json.loads(l) for l in open(sft_path)]
    print(f"训练样本数: {len(rows)}")

    tok = AutoTokenizer.from_pretrained(SMALL_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def fmt(row):
        text = (
            "<|im_start|>system\n"
            "You are a Python coding assistant. Return only valid Python code. "
            "Do not include Markdown fences, explanations, examples, or tests."
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{row['prompt']}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{row['completion'].rstrip()}\n<|im_end|>\n"
        )
        return {"text": text}

    dataset = Dataset.from_list([fmt(r) for r in rows])

    model = AutoModelForCausalLM.from_pretrained(
        SMALL_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_args = SFTConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        dataloader_num_workers=0,
        dataset_text_field="text",
        max_length=2048,
    )
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(str(adapter_dir))
    print(f"→ SFT adapter: {adapter_dir}")

    del model, trainer
    _free_gpu()
    return adapter_dir


# ─────────────────────────────────────────────────────────────
# Step 2b: GRPO 训练
# ─────────────────────────────────────────────────────────────

def _run_test(task, completion):
    """执行 HumanEval 测试，返回 1.0/0.0"""
    import re

    code = completion
    m = re.search(r"```python\n(.*?)```", code, re.DOTALL)
    if m:
        code = m.group(1).strip()
    elif "```" in code:
        m2 = re.search(r"```\n?(.*?)```", code, re.DOTALL)
        if m2:
            code = m2.group(1).strip()

    if f"def {task['entry_point']}" not in code:
        code = task["prompt"] + code

    full = f"{code}\n\n{task['test']}\n\ncheck({task['entry_point']})\n"
    try:
        old = signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(5)
        exec(full, {})  # noqa: S102
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        return 1.0
    except Exception:
        signal.alarm(0)
        return 0.0


def step2b_grpo(out_dir):
    print("\n" + "="*60)
    print("Step 2b: GRPO 训练 Qwen2.5-Coder-1.5B")
    print("="*60)

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig
    from trl import GRPOTrainer, GRPOConfig

    adapter_dir = out_dir / "grpo_adapter"

    # 用 train split（偶数索引，82 题）做 GRPO 训练
    from humaneval.adapter import Adapter
    he_adapter = Adapter()
    train_tasks = he_adapter.load_tasks(82, split="train")
    print(f"GRPO 训练任务数: {len(train_tasks)}")

    def make_prompt(task):
        return (
            "<|im_start|>system\n"
            "You are a Python coding assistant. Return only valid Python code. "
            "Do not include Markdown fences, explanations, examples, or tests."
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{task['prompt']}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    # prompt -> task lookup for reward function
    task_by_prompt = {make_prompt(t): t for t in train_tasks}

    dataset = Dataset.from_list([
        {"prompt": make_prompt(t)} for t in train_tasks
    ])

    def reward_fn(prompts, completions, **kwargs):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            task = task_by_prompt.get(prompt)
            rewards.append(_run_test(task, completion) if task else 0.0)
        mean_r = sum(rewards) / len(rewards)
        print(f"  [reward] batch mean={mean_r:.2f} ({int(sum(rewards))}/{len(rewards)} passed)")
        return rewards

    tok = AutoTokenizer.from_pretrained(SMALL_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        SMALL_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )

    grpo_cfg = GRPOConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-6,
        num_generations=4,          # 每题 sample 4 次
        max_completion_length=512,
        beta=0.04,                  # KL penalty
        warmup_steps=2,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        processing_class=tok,
        peft_config=lora_cfg,
    )
    trainer.train()
    trainer.save_model(str(adapter_dir))
    print(f"→ GRPO adapter: {adapter_dir}")

    del model, trainer
    _free_gpu()
    return adapter_dir


# ─────────────────────────────────────────────────────────────
# Step 3: eval split 评估
# ─────────────────────────────────────────────────────────────

def step3_evaluate(models_to_eval, out_dir):
    print("\n" + "="*60)
    print("Step 3: eval split（82 题）评估")
    print("="*60)

    from humaneval.adapter import Adapter
    he_adapter = Adapter()
    eval_tasks = he_adapter.load_tasks(82, split="eval")
    print(f"eval tasks: {len(eval_tasks)}")

    results = {}
    for label, model_id in models_to_eval:
        print(f"\n  [{label}] 评估 {model_id} ...")
        passed = 0
        for i, task in enumerate(eval_tasks):
            ok, _ = he_adapter._gen_and_test(str(model_id), task)
            if ok:
                passed += 1
            if (i + 1) % 20 == 0:
                print(f"    [{i+1}/{len(eval_tasks)}] pass={passed}")
        rate = passed / len(eval_tasks)
        print(f"  {label}: {passed}/{len(eval_tasks)} = {rate:.1%}")
        results[label] = {"passed": passed, "total": len(eval_tasks), "pass_rate": rate}
        # 释放显存给下一个模型
        if hasattr(he_adapter, "_cache"):
            he_adapter._cache.clear()
        _free_gpu()

    return results


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--skillbook", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-sft", action="store_true")
    ap.add_argument("--skip-grpo", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: SFT 数据
    sft_path, sft_stats = step1_extract_sft(
        Path(args.traces),
        Path(args.skillbook) if args.skillbook else None,
        out_dir,
    )

    # Step 2a: SFT
    sft_dir = out_dir / "sft_adapter"
    if args.skip_sft and sft_dir.exists():
        print("\nStep 2a: 跳过 SFT（--skip-sft），使用已有 adapter")
    else:
        sft_dir = step2a_sft(sft_path, out_dir)

    # Step 2b: GRPO
    grpo_dir = out_dir / "grpo_adapter"
    if args.skip_grpo and grpo_dir.exists():
        print("\nStep 2b: 跳过 GRPO（--skip-grpo），使用已有 adapter")
    else:
        grpo_dir = step2b_grpo(out_dir)

    # Step 3: 三路对比
    eval_results = step3_evaluate(
        [
            ("base", SMALL_MODEL),
            ("sft",  sft_dir),
            ("grpo", grpo_dir),
        ],
        out_dir,
    )

    # 汇总
    print("\n" + "="*60)
    print("结果汇总（eval split，82 题）")
    print("="*60)
    base_pass = eval_results["base"]["passed"]
    for label, res in eval_results.items():
        delta = res["passed"] - base_pass
        delta_str = f"({delta:+d})" if label != "base" else ""
        print(f"  {label:<8}: {res['passed']}/{res['total']} = {res['pass_rate']:.1%}  {delta_str}")

    summary = {"sft_stats": sft_stats, "eval_results": eval_results}
    (out_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n→ {out_dir}/experiment_summary.json")


if __name__ == "__main__":
    main()
