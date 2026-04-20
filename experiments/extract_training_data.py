#!/usr/bin/env python3
"""
从 traces 中提取训练数据

支持 3 种输入格式:
  1. 新 traces (来自 run_evolve.py) - attempts 是 list
  2. 老 traces (来自 harder_benchmark 实验) - attempts 是 int + decision 字段
  3. 对比 JSON (有 floor/ceiling) - 直接对比 small vs large 结果

输出: Alpaca 格式的 JSONL, 可喂给 train_small_model.py

用法:
    # 从新 traces
    python3 experiments/extract_training_data.py \\
        --traces "data/traces/*.jsonl" --output data/training_data.jsonl

    # 从老对比 JSON
    python3 experiments/extract_training_data.py \\
        --comparison-json path/to/experiment_results.json \\
        --output data/training_data.jsonl
"""

import sys
import json
import argparse
import glob
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import SMALL_MODEL, LARGE_MODEL, extract_signature, solve_task


# ============================================================================
# 加载数据
# ============================================================================

def load_humaneval_tasks() -> dict:
    """加载 HumanEval 任务 (task_id → task)"""
    paths = [
        Path(__file__).parent.parent / "data" / "HumanEval.jsonl",
        Path(__file__).parent.parent.parent / "uncommonroute_skill_experiment" /
            "data" / "human-eval" / "data" / "HumanEval.jsonl",
    ]
    for p in paths:
        if p.exists():
            tasks = {}
            with open(p) as f:
                for line in f:
                    t = json.loads(line)
                    tasks[t["task_id"]] = t
            return tasks
    raise FileNotFoundError("HumanEval.jsonl not found")


# ============================================================================
# 从 JSONL traces 找 hard tasks
# ============================================================================

def find_hard_tasks_from_traces(traces: list) -> set:
    """
    从 traces 中找 hard tasks:
    - 小模型失败（success=false 且 model=deepseek）
    - 大模型成功（success=true 且 model=gpt-5.4）
    
    支持两种 trace 格式:
      A) attempts 是 list: [{"model": "...", "success": bool}, ...]
      B) attempts 是 int + decision 字段: "probe:small→complex:fallback"
    """
    hard = set()
    per_task = defaultdict(lambda: {"small_failed": False, "large_success": False})

    for t in traces:
        tid = t.get("task_id", "")
        if not tid:
            continue

        attempts = t.get("attempts")

        # Format A: list
        if isinstance(attempts, list):
            for att in attempts:
                m = att.get("model", "")
                if "deepseek" in m and not att["success"]:
                    per_task[tid]["small_failed"] = True
                if "gpt-5.4" in m and att["success"]:
                    per_task[tid]["large_success"] = True
            continue

        # Format B: int + decision
        if isinstance(attempts, int):
            decision = t.get("decision", "")
            final_model = t.get("final_model", "")
            final_success = t.get("success", t.get("final_success", False))

            if attempts >= 2:  # escalated
                per_task[tid]["small_failed"] = True  # 小模型尝试过且失败
                if "gpt-5.4" in final_model and final_success:
                    per_task[tid]["large_success"] = True

    for tid, status in per_task.items():
        if status["small_failed"] and status["large_success"]:
            hard.add(tid)

    return hard


# ============================================================================
# 从对比 JSON (floor vs ceiling) 找 hard tasks
# ============================================================================

def find_hard_from_comparison(comparison_json_path: str) -> set:
    """
    从 full_comparison/harder_benchmark 类的 JSON 找 hard tasks:
    - floor (Pure DeepSeek) 失败
    - ceiling (Pure GPT-5.4) 成功
    """
    with open(comparison_json_path) as f:
        data = json.load(f)

    floor_results = data.get("floor", {}).get("results", [])
    ceiling_results = data.get("ceiling", {}).get("results", [])

    floor_by_id = {r["task_id"]: r for r in floor_results}
    ceiling_by_id = {r["task_id"]: r for r in ceiling_results}

    hard = set()
    for tid, floor_r in floor_by_id.items():
        if tid not in ceiling_by_id:
            continue
        if not floor_r.get("success") and ceiling_by_id[tid].get("success"):
            hard.add(tid)

    return hard


# ============================================================================
# 生成训练样本 (用大模型跑一次拿到 ground truth 代码)
# ============================================================================

def generate_training_samples(
    hard_task_ids: set,
    human_eval_tasks: dict,
    include_test_cases: bool = False,
) -> list:
    """
    对每个 hard task:
    - 用大模型 (gpt-5.4) 跑一次, 拿正确代码当 ground truth
    """
    samples = []
    print(f"\n🔄 重跑 {len(hard_task_ids)} 个 hard tasks (用 {LARGE_MODEL})...")

    hard_list = sorted(hard_task_ids)
    for i, tid in enumerate(hard_list):
        if tid not in human_eval_tasks:
            print(f"  [{i+1}/{len(hard_list)}] {tid}: 原数据缺失, 跳过")
            continue

        task = human_eval_tasks[tid]
        result = solve_task(LARGE_MODEL, task)

        status = "✅" if result["success"] else "❌"
        print(f"  [{i+1}/{len(hard_list)}] {tid}: {status} ${result['cost_usd']:.5f}")

        if not result["success"]:
            print(f"      ⚠️  大模型竟然也没过, 跳过")
            continue

        sig = extract_signature(task["prompt"])
        sample = {
            "task_id": tid,
            "instruction": task["prompt"],
            "input": "",
            "output": result["generated_code"],
            "signature": sig,
            "source_model": LARGE_MODEL,
            "generation_cost_usd": result["cost_usd"],
        }
        if include_test_cases:
            sample["test"] = task["test"]
            sample["entry_point"] = task["entry_point"]
        samples.append(sample)

    return samples


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Extract training data from traces")
    parser.add_argument(
        "--traces",
        default="data/traces/*.jsonl",
        help="Trace JSONL files (glob pattern)",
    )
    parser.add_argument(
        "--comparison-json",
        default=None,
        help="Alternative: load from comparison JSON (has floor+ceiling)",
    )
    parser.add_argument(
        "--output",
        default="data/training_data.jsonl",
        help="Output training data JSONL",
    )
    parser.add_argument(
        "--include-test-cases",
        action="store_true",
        help="Include HumanEval test cases in each sample (for later eval)",
    )
    args = parser.parse_args()

    root = Path(__file__).parent.parent

    # 加载 HumanEval tasks
    human_eval = load_humaneval_tasks()
    print(f"📚 加载 HumanEval: {len(human_eval)} 题")

    # 获取 hard tasks
    hard = set()

    if args.comparison_json:
        cp_path = Path(args.comparison_json)
        if not cp_path.is_absolute():
            cp_path = root / cp_path
        print(f"📂 Comparison JSON: {cp_path}")
        hard = find_hard_from_comparison(str(cp_path))
        print(f"   找到 {len(hard)} 个 hard tasks (floor 失败 + ceiling 成功)")
    else:
        traces_pattern = args.traces
        if not Path(traces_pattern).is_absolute():
            traces_pattern = str(root / traces_pattern)
        print(f"📂 Traces pattern: {traces_pattern}")

        files = sorted(glob.glob(traces_pattern))
        if not files:
            print("❌ 没找到 trace 文件")
            return

        traces = []
        for fp in files:
            with open(fp) as f:
                for line in f:
                    try:
                        traces.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue

        print(f"   加载 {len(traces)} 条 trace (从 {len(files)} 个文件)")
        hard = find_hard_tasks_from_traces(traces)
        print(f"   找到 {len(hard)} 个 hard tasks")

    if not hard:
        print("\n⚠️  没有 hard tasks! 可能原因:")
        print("   - 小模型全部成功 (benchmark 太简单, 换更难的)")
        print("   - trace 格式不对")
        return

    print(f"\n🎯 Hard tasks: {sorted(hard)[:10]}{'...' if len(hard) > 10 else ''}")

    # 生成训练样本
    samples = generate_training_samples(hard, human_eval, args.include_test_cases)

    if not samples:
        return

    # 保存
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n💾 保存 {len(samples)} 条训练样本: {output_path}")

    # 分布
    sig_dist = Counter(s["signature"] for s in samples)
    print(f"\n📊 按 signature 分布:")
    for sig, n in sig_dist.most_common():
        print(f"   {sig:<30} × {n}")

    # 示例
    print(f"\n📝 样本预览 (第 1 条):")
    s = samples[0]
    print(f"   task_id:     {s['task_id']}")
    print(f"   signature:   {s['signature']}")
    print(f"   instruction: {s['instruction'][:100].strip()}...")
    print(f"   output:      {s['output'][:100].strip()}...")

    print(f"\n✅ 训练数据就绪: {output_path}")
    print(f"\n下一步:")
    print(f"  python3 experiments/train_small_model.py \\")
    print(f"    --data {args.output} \\")
    print(f"    --base-model 'MiniMaxAI/MiniMax-M2' \\")
    print(f"    --output outputs/minimax-m2-v1 \\")
    print(f"    --use-4bit")


if __name__ == "__main__":
    main()
