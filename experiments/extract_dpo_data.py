#!/usr/bin/env python3
"""
提取 DPO 训练数据

格式: (prompt, chosen, rejected)
- prompt:   HumanEval 题目
- chosen:   大模型的正确代码 (来自已有 training_data.jsonl)  
- rejected: 小模型生成的失败代码 (本脚本重新跑小模型产生)

为什么要单独脚本？
  因为 SFT 只需要 chosen (大模型答案), 
  DPO 还需要 rejected (小模型的错误答案)

用法:
    python3 experiments/extract_dpo_data.py \\
        --sft-data data/training_data.jsonl \\
        --output data/dpo_data.jsonl
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import SMALL_MODEL, solve_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sft-data", default="data/training_data.jsonl",
        help="SFT 数据 (有 chosen = 大模型代码)"
    )
    parser.add_argument("--output", default="data/dpo_data.jsonl")
    parser.add_argument(
        "--samples-per-prompt", type=int, default=3,
        help="每个 prompt 生成几次 rejected (取最差)"
    )
    args = parser.parse_args()

    root = Path(__file__).parent.parent

    sft_path = Path(args.sft_data)
    if not sft_path.is_absolute():
        sft_path = root / sft_path

    if not sft_path.exists():
        print(f"❌ SFT 数据不存在: {sft_path}")
        print("先运行: python3 experiments/extract_training_data.py")
        return

    # 加载 HumanEval (需要 test cases)
    from experiments.extract_training_data import load_humaneval_tasks
    human_eval = load_humaneval_tasks()

    # 加载 SFT 数据
    sft_samples = []
    with open(sft_path) as f:
        for line in f:
            sft_samples.append(json.loads(line.strip()))
    
    print(f"📚 加载 SFT 数据: {len(sft_samples)} 条")
    print(f"🔄 每条 prompt 用小模型 ({SMALL_MODEL}) 生成 {args.samples_per_prompt} 次")
    print(f"   取失败的作为 rejected")
    print()

    dpo_samples = []
    
    for i, sft_sample in enumerate(sft_samples):
        tid = sft_sample["task_id"]
        prompt = sft_sample["instruction"]
        chosen = sft_sample["output"]
        
        if tid not in human_eval:
            print(f"  [{i+1}/{len(sft_samples)}] {tid}: 缺 test case, 跳过")
            continue
        
        task = human_eval[tid]
        
        # 多次 sample 小模型, 找失败的
        print(f"  [{i+1}/{len(sft_samples)}] {tid}:", end=" ", flush=True)
        
        rejected = None
        for j in range(args.samples_per_prompt):
            result = solve_task(SMALL_MODEL, task)
            status = "✅" if result["success"] else "❌"
            print(f"{status}", end=" ", flush=True)
            
            if not result["success"] and rejected is None:
                rejected = result["generated_code"]
        
        if rejected is None:
            print(f"小模型都成功了, 跳过 (这题没必要 DPO)")
            continue
        
        print(f"→ 取第一次失败的做 rejected")
        
        dpo_samples.append({
            "task_id": tid,
            "prompt": prompt,
            "chosen": chosen,       # 大模型的正确代码
            "rejected": rejected,   # 小模型的错误代码
            "signature": sft_sample.get("signature", ""),
        })
    
    # 保存
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for s in dpo_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    print(f"\n💾 保存 {len(dpo_samples)} 条 DPO 样本: {output_path}")
    print(f"\n下一步:")
    print(f"  python3 experiments/train_small_model_dpo.py \\")
    print(f"    --data {args.output} \\")
    print(f"    --base-model 'MiniMaxAI/MiniMax-M2' \\")
    print(f"    --use-4bit")


if __name__ == "__main__":
    main()
