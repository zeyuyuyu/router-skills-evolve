#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) 训练 - 真正的 RL

DeepSeek-R1 用的方法。

原理:
  对每个 prompt sample N 个回答
  跑 HumanEval test case 得 reward (0/1)
  advantage = (reward - mean) / std
  PPO 式策略更新 (带 KL penalty)

优点 (vs SFT/DPO):
  - 不需要 teacher model 的代码
  - 直接优化 "任务通过率"
  - 可以学到超越 teacher 的策略

依赖:
  pip install torch transformers peft datasets accelerate bitsandbytes trl>=0.18.0

用法:
  python3 experiments/train_small_model_grpo.py \\
      --tasks data/HumanEval.jsonl \\
      --base-model "MiniMaxAI/MiniMax-M2" \\
      --output outputs/minimax-m2-grpo-v1 \\
      --n-generations 4 \\
      --use-4bit

⚠️ 注意:
  - 比 SFT/DPO 慢 4x (每步要生成 N 次)
  - 显存需求更高 (policy + ref policy)
  - 建议先用 DPO 验证 pipeline, 再上 GRPO
"""

import sys
import json
import signal
import argparse
from pathlib import Path


def check_deps():
    missing = []
    for pkg in ["torch", "transformers", "peft", "datasets", "trl"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"❌ 缺依赖: {', '.join(missing)}")
        sys.exit(1)


def format_prompt(instruction: str) -> str:
    """Alpaca 风格"""
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def extract_code_from_completion(completion: str, entry_point: str, prompt: str) -> str:
    """从生成的 completion 中提取 Python 代码"""
    code = completion
    # 去掉 markdown fence
    if "```python" in code:
        start = code.index("```python") + len("```python")
        end = code.index("```", start) if "```" in code[start:] else len(code)
        code = code[start:end].strip()
    elif "```" in code:
        start = code.index("```") + 3
        end = code.index("```", start) if "```" in code[start+3:] else len(code)
        code = code[start:end].strip()
    
    # 如果代码里没有 def entry_point, 拼接 prompt
    if f"def {entry_point}" not in code and prompt:
        # prompt 本身包含函数签名, 所以直接拼
        body_lines = code.split("\n")
        indented = []
        for line in body_lines:
            if line.strip() and not line.startswith((" ", "\t")):
                indented.append("    " + line)
            else:
                indented.append(line)
        code = prompt + "\n".join(indented)
    
    return code


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("timeout")


def run_test_reward(task: dict, completion: str) -> float:
    """跑 HumanEval test, 返回 reward ∈ [0, 1]"""
    try:
        code = extract_code_from_completion(
            completion, task["entry_point"], task["prompt"]
        )
        
        full_code = f"""
{code}

{task['test']}

check({task['entry_point']})
"""
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(5)  # 5 秒超时
        
        exec(full_code, {})
        signal.alarm(0)
        return 1.0  # 通过
    except Exception:
        signal.alarm(0)
        return 0.0  # 失败


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="data/HumanEval.jsonl")
    parser.add_argument("--n-tasks", type=int, default=60,
                       help="用多少题训练 (前 N 题)")
    parser.add_argument("--base-model", default="MiniMaxAI/MiniMax-M2")
    parser.add_argument("--output", default="outputs/grpo_model")
    parser.add_argument("--n-generations", type=int, default=4,
                       help="每个 prompt sample 几次 (推荐 4-8)")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)  # GRPO 显存大
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-6)  # RL 超小 lr
    parser.add_argument("--kl-coef", type=float, default=0.04)  # KL penalty
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    check_deps()

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import GRPOTrainer, GRPOConfig

    # ========================================================================
    # 1. 加载 HumanEval 任务
    # ========================================================================
    print("=" * 80)
    print(f"📂 加载 HumanEval: {args.tasks}")
    print("=" * 80)
    
    tasks_path = Path(args.tasks)
    if not tasks_path.is_absolute():
        tasks_path = Path(__file__).parent.parent / args.tasks
    
    all_tasks = []
    with open(tasks_path) as f:
        for line in f:
            all_tasks.append(json.loads(line.strip()))
    
    tasks = all_tasks[:args.n_tasks]
    task_by_prompt = {format_prompt(t["prompt"]): t for t in tasks}
    
    print(f"✅ 用 {len(tasks)} 题训练")
    
    # 数据集格式: 每条是一个 prompt (模型会 sample 多次)
    dataset = Dataset.from_list([
        {"prompt": format_prompt(t["prompt"]), "task_id": t["task_id"]}
        for t in tasks
    ])

    if args.dry_run:
        print(f"\n🏁 --dry-run: 数据 ✅ ({len(tasks)} tasks). 跳过模型加载和训练.")
        return
    
    # ========================================================================
    # 2. 加载模型
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"🤖 Base model: {args.base_model}")
    print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # ========================================================================
    # 3. 定义 reward function
    # ========================================================================
    def reward_fn(prompts, completions, **kwargs):
        """
        对每个 (prompt, completion) 跑 HumanEval test, 返回 reward.
        
        TRL 的 GRPOTrainer 会传:
          prompts: list of prompts (原始)
          completions: list of generations
        """
        rewards = []
        for prompt, completion in zip(prompts, completions):
            task = task_by_prompt.get(prompt)
            if not task:
                rewards.append(0.0)
                continue
            
            r = run_test_reward(task, completion)
            rewards.append(r)
        
        print(f"[reward] batch mean: {sum(rewards)/len(rewards):.2f} "
              f"({int(sum(rewards))}/{len(rewards)} passed)")
        return rewards
    
    # ========================================================================
    # 4. GRPO 训练
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"🏃 GRPO 训练 (n_gen={args.n_generations}, kl={args.kl_coef})")
    print("=" * 80)
    
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent.parent / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_generations=args.n_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_new_tokens,
        beta=args.kl_coef,  # KL penalty
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
    )
    
    # TRL 1.2+: use processing_class for tokenizer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    
    trainer.train()
    trainer.save_model(str(output_dir))
    
    with open(output_dir / "training_info.json", "w") as f:
        json.dump({
            "method": "GRPO",
            "base_model": args.base_model,
            "n_tasks": len(tasks),
            "n_generations": args.n_generations,
            "kl_coef": args.kl_coef,
            "learning_rate": args.lr,
            "epochs": args.epochs,
        }, f, indent=2)
    
    print(f"\n✅ GRPO 训练完成!")
    print(f"   Output: {output_dir}")


if __name__ == "__main__":
    main()
