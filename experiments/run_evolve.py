#!/usr/bin/env python3
"""
跑 Router + Skills Evolve 实验

用法:
    python3 experiments/run_evolve.py --n 30 --rounds 3
    python3 experiments/run_evolve.py --n 164 --rounds 4   # 全量 HumanEval

输出:
    - data/traces/traces_<timestamp>.jsonl   (所有 trace)
    - data/skills/skills_<timestamp>.json    (学到的 skills)
    - results/evolve_<timestamp>.json        (每轮 metrics)
"""

import sys
import json
import time
import argparse
from pathlib import Path
from collections import Counter

# 让 src 可以被 import
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import SMALL_MODEL, LARGE_MODEL, SkillBook, RouterWithSkills


# ============================================================================
# HumanEval 数据加载
# ============================================================================

def load_humaneval(max_n: int = 164) -> list:
    """加载 HumanEval 数据"""
    # 尝试多个可能的路径
    paths = [
        Path(__file__).parent.parent / "data" / "HumanEval.jsonl",
        Path(__file__).parent.parent.parent / "uncommonroute_skill_experiment" / "data" / "human-eval" / "data" / "HumanEval.jsonl",
    ]
    
    data_path = None
    for p in paths:
        if p.exists():
            data_path = p
            break
    
    if not data_path:
        raise FileNotFoundError(
            "HumanEval.jsonl not found. Place it at data/HumanEval.jsonl "
            "or run: git clone https://github.com/openai/human-eval"
        )
    
    tasks = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= max_n:
                break
            tasks.append(json.loads(line))
    return tasks


# ============================================================================
# 运行
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Router+Skills Evolve experiment")
    parser.add_argument("--n", type=int, default=30, help="Number of tasks (max 164)")
    parser.add_argument("--rounds", type=int, default=3, help="Number of evolve rounds")
    parser.add_argument("--min-rate", type=float, default=0.8, help="Min success rate to downgrade")
    parser.add_argument("--min-samples", type=int, default=1, help="Min samples before decision")
    parser.add_argument("--save-traces", action="store_true", default=True, help="Save traces")
    parser.add_argument("--start", type=int, default=0, help="HumanEval start index")
    args = parser.parse_args()

    print("=" * 80)
    print(f"🧬 Router + Skills Evolve 实验")
    print(f"   任务: HumanEval[{args.start}:{args.start+args.n}] ({args.n} 题)")
    print(f"   轮数: {args.rounds}")
    print(f"   Small: {SMALL_MODEL}")
    print(f"   Large: {LARGE_MODEL}")
    print("=" * 80)

    # 加载任务
    all_tasks = load_humaneval(args.start + args.n)
    tasks = all_tasks[args.start:args.start + args.n]
    per_round = len(tasks) // args.rounds

    # 初始化 Router
    skill_book = SkillBook()
    router = RouterWithSkills(
        small_model=SMALL_MODEL,
        large_model=LARGE_MODEL,
        skill_book=skill_book,
        min_rate=args.min_rate,
        min_samples=args.min_samples,
    )

    # 存储
    all_traces = []
    round_stats = []

    # 开跑
    for round_num in range(args.rounds):
        round_tasks = tasks[round_num * per_round:(round_num + 1) * per_round]
        skills_start = len(skill_book.skills)

        print(f"\n{'─'*80}")
        print(f"🔄 Round {round_num}: {len(round_tasks)} 题, skills={skills_start}")
        print(f"{'─'*80}")

        round_traces = []
        for i, task in enumerate(round_tasks):
            result = router.solve(task)
            result["round"] = round_num
            round_traces.append(result)
            all_traces.append(result)

            icon = "✅" if result["final_success"] else "❌"
            first_m = result["attempts"][0]["model"].split("/")[-1][:20]
            final_m = (
                result["final_model"].split("/")[-1][:20]
                if result["final_model"] else "FAIL"
            )
            print(f"  [{i+1:2d}] {task['task_id']:<15} {icon} "
                  f"{result['attempts_count']}次 ({first_m}→{final_m}) "
                  f"${result['total_cost']:.5f} | {result['decision'][:35]}")

        # 每轮统计
        total = len(round_traces)
        success = sum(1 for r in round_traces if r["final_success"])
        cost = sum(r["total_cost"] for r in round_traces)
        attempts = sum(r["attempts_count"] for r in round_traces)

        stats = {
            "round": round_num,
            "tasks": total,
            "skills_start": skills_start,
            "skills_end": len(skill_book.skills),
            "success": success,
            "success_rate": success / total,
            "total_cost": cost,
            "avg_cost": cost / total,
            "avg_attempts": attempts / total,
        }
        round_stats.append(stats)

        print(f"\n📊 Round {round_num}: {success}/{total} ({success/total:.0%}) "
              f"成本 ${cost:.5f}, skills {skills_start}→{len(skill_book.skills)}, "
              f"平均尝试 {attempts/total:.2f}")

    # 总结
    print("\n" + "=" * 80)
    print("📈 Evolve 趋势")
    print("=" * 80)
    print(f"\n{'Round':<8} {'Skills':<12} {'成功率':<10} {'单题成本':<14} {'平均尝试':<10}")
    print("-" * 60)
    for s in round_stats:
        skills_str = f"{s['skills_start']}→{s['skills_end']}"
        print(f"{s['round']:<8} {skills_str:<12} "
              f"{s['success_rate']:<10.0%} ${s['avg_cost']:<13.5f} "
              f"{s['avg_attempts']:<10.2f}")

    total = sum(s["tasks"] for s in round_stats)
    total_success = sum(s["success"] for s in round_stats)
    total_cost = sum(s["total_cost"] for s in round_stats)
    print(f"\n🎯 总计: {total_success}/{total} ({total_success/total:.0%}) 成本 ${total_cost:.5f}")

    # 学到的 Skills 分析
    print(f"\n📋 Skills 总数: {len(skill_book.skills)}")
    verdicts = Counter()
    for sig, skill in skill_book.skills.items():
        v = skill.can_downgrade_to_small(SMALL_MODEL, args.min_rate, args.min_samples)
        if v is True:
            verdicts["✅ small ok"] += 1
        elif v is False:
            verdicts["❌ skip small"] += 1
        else:
            verdicts["? 数据不足"] += 1
    for v, n in verdicts.items():
        print(f"   {v}: {n}")

    # 保存
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if args.save_traces:
        traces_path = Path(__file__).parent.parent / "data" / "traces" / f"traces_{timestamp}.jsonl"
        traces_path.parent.mkdir(parents=True, exist_ok=True)
        with open(traces_path, "w") as f:
            for t in all_traces:
                f.write(json.dumps(t) + "\n")
        print(f"\n💾 Traces: {traces_path}")

    skills_path = Path(__file__).parent.parent / "data" / "skills" / f"skills_{timestamp}.json"
    skill_book.save(skills_path)
    print(f"💾 Skills: {skills_path}")

    results_path = Path(__file__).parent.parent / "results" / f"evolve_{timestamp}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "args": vars(args),
            "round_stats": round_stats,
            "total_success": total_success,
            "total_tasks": total,
            "total_cost": total_cost,
            "skills_summary": dict(verdicts),
        }, f, indent=2)
    print(f"💾 Results: {results_path}")


if __name__ == "__main__":
    main()
