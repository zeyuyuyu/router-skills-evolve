#!/usr/bin/env python3
"""
训练小模型 (LoRA fine-tuning)

目标: 让小模型学会处理"之前失败的题" (来自 traces 的 hard examples)

用法:
    # 1. 准备 requirements (如需):
    pip install transformers peft datasets accelerate bitsandbytes trl

    # 2. 训练
    python3 experiments/train_small_model.py \\
        --data data/training_data.jsonl \\
        --base-model "MiniMaxAI/MiniMax-M2" \\
        --output outputs/minimax-m2-finetuned \\
        --lora-r 16 --epochs 3

环境要求:
    - GPU: 2 × A100 80G (for MiniMax-M2 ~20B)
    -      8 × A100 80G (for DeepSeek-V3.2 37B activate MoE)
    - Disk: 100-200GB (模型权重 + 训练数据)
    - Python 3.10+
    - CUDA 12.1+

推荐流程 (第一次跑):
    1. 从小模型开始: --base-model "MiniMaxAI/MiniMax-M2" (20B, 好训)
    2. 看训练效果
    3. 满意后扩到更大模型
"""

import sys
import json
import argparse
from pathlib import Path


def check_deps():
    """检查依赖是否安装"""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    try:
        import peft
    except ImportError:
        missing.append("peft")
    try:
        import datasets
    except ImportError:
        missing.append("datasets")
    try:
        import trl
    except ImportError:
        missing.append("trl")
    
    if missing:
        print(f"❌ 缺少依赖: {', '.join(missing)}")
        print("请运行: pip install transformers peft datasets accelerate bitsandbytes trl")
        sys.exit(1)


def format_prompt(instruction: str, input_text: str = "") -> str:
    """
    统一的 prompt 格式 (Alpaca 风格).
    训练时和推理时必须一致!
    """
    if input_text:
        return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    return f"""### Instruction:
{instruction}

### Response:
"""


def format_training_sample(sample: dict) -> dict:
    """把训练样本转成 (prompt, completion) 对"""
    prompt = format_prompt(sample["instruction"], sample.get("input", ""))
    return {
        "prompt": prompt,
        "completion": sample["output"],
        "text": prompt + sample["output"],  # 拼接后的完整训练文本
    }


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune small model")
    parser.add_argument(
        "--data", required=True,
        help="Training data JSONL (from extract_training_data.py)",
    )
    parser.add_argument(
        "--base-model", default="MiniMaxAI/MiniMax-M2",
        help="HuggingFace model ID to fine-tune",
    )
    parser.add_argument(
        "--output", default="outputs/finetuned_model",
        help="Output directory",
    )
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument(
        "--use-4bit", action="store_true",
        help="Use 4-bit quantization (省显存, 适合单卡)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只加载数据和模型, 不真的训练 (测试环境)",
    )
    args = parser.parse_args()

    check_deps()

    # 导入 (延迟到这里, 因为 check_deps 要先通过)
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, 
        TrainingArguments, BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    # ========================================================================
    # 1. 加载训练数据
    # ========================================================================
    print("=" * 80)
    print(f"📂 加载训练数据: {args.data}")
    print("=" * 80)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent / args.data

    if not data_path.exists():
        print(f"❌ 数据文件不存在: {data_path}")
        print("请先运行: python3 experiments/extract_training_data.py")
        sys.exit(1)

    raw_samples = []
    with open(data_path) as f:
        for line in f:
            raw_samples.append(json.loads(line.strip()))

    print(f"✅ 加载 {len(raw_samples)} 条训练样本")

    if len(raw_samples) < 10:
        print(f"⚠️  样本太少 ({len(raw_samples)} 条), 建议至少 100 条")
        print("可以: ")
        print("  1. 跑更多 run_evolve.py 收集更多 traces")
        print("  2. 混入 MBPP / HumanEval+ 等更难的题")

    # 转成训练格式
    formatted = [format_training_sample(s) for s in raw_samples]
    dataset = Dataset.from_list(formatted)

    print(f"\n📝 训练文本示例 (前 200 字符):")
    print(formatted[0]["text"][:200] + "...")

    # Dry-run 到这里就够了, 不用加载模型
    if args.dry_run:
        print(f"\n🏁 --dry-run: 数据加载 ✅, 格式化 ✅. 跳过模型加载和训练.")
        print(f"   训练样本数: {len(raw_samples)}")
        print(f"   Dataset columns: {dataset.column_names}")
        print(f"   下一步: 去掉 --dry-run, 真实训练")
        return

    # ========================================================================
    # 2. 加载 base model + tokenizer
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"🤖 加载 base model: {args.base_model}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit 量化节省显存
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        print("   使用 4-bit 量化 (QLoRA)")

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
    # 3. 加 LoRA adapter
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"🎯 配置 LoRA (rank={args.lora_r}, alpha={args.lora_alpha})")
    print("=" * 80)

    # 找模型的 linear 层名字 (不同模型有不同的模块名)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    # ========================================================================
    # 4. 训练
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"🏃 开始训练")
    print("=" * 80)

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent.parent / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
    )

    trainer.train()
    trainer.save_model(str(output_dir))

    # ========================================================================
    # 5. 保存配置
    # ========================================================================
    with open(output_dir / "training_info.json", "w") as f:
        json.dump({
            "base_model": args.base_model,
            "training_samples": len(raw_samples),
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "epochs": args.epochs,
            "learning_rate": args.lr,
        }, f, indent=2)

    print(f"\n✅ 训练完成!")
    print(f"   Output: {output_dir}")
    print(f"\n下一步:")
    print(f"   1. 部署 LoRA adapter 到 inference 服务")
    print(f"   2. 更新 CommonStack / UncommonRoute 模型池, 加入 finetuned 模型")
    print(f"   3. 重跑 run_evolve.py, 对比 fine-tune 前后效果")


if __name__ == "__main__":
    main()
