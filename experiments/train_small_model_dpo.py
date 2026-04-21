#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) 训练

原理:
  给模型看 (chosen, rejected) 对, 让它偏向 chosen, 远离 rejected
  Loss: -log(σ(β * (logP(chosen) - logP(rejected))))

优点 (vs SFT):
  - 学 "对比" 比学 "模仿" 更强
  - 明确告诉模型 "不要这样写"
  - 数据利用效率高 (一条对比 = 一次正 + 一次负信号)

依赖:
  pip install transformers peft datasets accelerate bitsandbytes trl

用法:
  python3 experiments/train_small_model_dpo.py \\
      --data data/dpo_data.jsonl \\
      --base-model "MiniMaxAI/MiniMax-M2" \\
      --output outputs/minimax-m2-dpo-v1 \\
      --use-4bit
"""

import sys
import json
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
        print("pip install torch transformers peft datasets accelerate bitsandbytes trl")
        sys.exit(1)


def format_prompt(instruction: str) -> str:
    """Alpaca 风格"""
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="DPO data JSONL")
    parser.add_argument("--base-model", default="MiniMaxAI/MiniMax-M2")
    parser.add_argument("--output", default="outputs/dpo_model")
    parser.add_argument("--beta", type=float, default=0.1,
                       help="DPO β: 越大越保守 (推荐 0.1-0.5)")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)  # DPO 显存 2x
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)  # DPO 用小 lr
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    check_deps()

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import DPOTrainer, DPOConfig

    # ========================================================================
    # 1. 加载数据
    # ========================================================================
    print("=" * 80)
    print(f"📂 加载 DPO 数据: {args.data}")
    print("=" * 80)
    
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent / args.data
    
    if not data_path.exists():
        print(f"❌ 数据不存在: {data_path}")
        print("先运行: python3 experiments/extract_dpo_data.py")
        sys.exit(1)
    
    raw = []
    with open(data_path) as f:
        for line in f:
            raw.append(json.loads(line.strip()))
    
    print(f"✅ {len(raw)} 条 DPO 样本")
    
    # 转 DPO 格式 (trl 要求的字段名: prompt, chosen, rejected)
    dpo_data = [
        {
            "prompt": format_prompt(s["prompt"]),
            "chosen": s["chosen"],
            "rejected": s["rejected"],
        }
        for s in raw
    ]
    dataset = Dataset.from_list(dpo_data)
    
    print(f"\n样本预览:")
    print(f"  Prompt: {dpo_data[0]['prompt'][:80]}...")
    print(f"  Chosen ({len(dpo_data[0]['chosen'])} chars): {dpo_data[0]['chosen'][:60]}...")
    print(f"  Rejected ({len(dpo_data[0]['rejected'])} chars): {dpo_data[0]['rejected'][:60]}...")

    if args.dry_run:
        print(f"\n🏁 --dry-run: 数据 ✅ ({len(dpo_data)} 条). 跳过模型加载和训练.")
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
    
    # ========================================================================
    # 3. LoRA 配置
    # ========================================================================
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # ========================================================================
    # 4. DPO 训练
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"🏃 DPO 训练 (β={args.beta})")
    print("=" * 80)
    
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent.parent / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_seq_len,
        max_prompt_length=args.max_seq_len // 2,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )
    
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )
    
    trainer.train()
    trainer.save_model(str(output_dir))
    
    # ========================================================================
    # 5. 保存配置
    # ========================================================================
    with open(output_dir / "training_info.json", "w") as f:
        json.dump({
            "method": "DPO",
            "base_model": args.base_model,
            "training_samples": len(raw),
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "beta": args.beta,
            "epochs": args.epochs,
            "learning_rate": args.lr,
        }, f, indent=2)
    
    print(f"\n✅ DPO 训练完成!")
    print(f"   Output: {output_dir}")


if __name__ == "__main__":
    main()
