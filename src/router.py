"""Router + Skills 路由决策 + 执行 (含 fallback)"""

from typing import Dict, List, Optional, Tuple

from .models import solve_task
from .skills import SkillBook, extract_signature


class RouterWithSkills:
    """
    Router + Skills 的完整实现。
    
    策略 (Escalation + Skills):
    1. 查 Skills 推荐: 这类题用啥模型
    2a. 若推荐便宜模型 → 直接用；失败 fallback 到大模型
    2b. 若无推荐 (未见过) → probe 便宜模型；失败 fallback
    2c. 若推荐大模型 (历史失败多) → 直接用大模型
    3. 更新 Skills (用新结果 evolve)

    保证:
    - 准确率: 通过 fallback 兜底
    - 省钱: 只要有 cluster 确认 small 能做, 就用 small
    """

    def __init__(
        self,
        small_model: str,
        large_model: str,
        skill_book: Optional[SkillBook] = None,
        min_rate: float = 0.8,  # small 成功率 >= 此值才降级
        min_samples: int = 1,   # 至少几个样本才做决策
    ):
        self.small_model = small_model
        self.large_model = large_model
        self.skill_book = skill_book or SkillBook()
        self.min_rate = min_rate
        self.min_samples = min_samples

    def decide(self, prompt: str) -> Tuple[str, str]:
        """
        决策: 这道题先用哪个模型?
        
        Returns:
            (model_id, decision_reason)
        """
        sig = extract_signature(prompt)
        skill = self.skill_book.skills.get(sig)

        if skill is None:
            return self.small_model, f"probe:no_skill({sig})"

        verdict = skill.can_downgrade_to_small(
            self.small_model, self.min_rate, self.min_samples
        )

        if verdict is True:
            return self.small_model, f"skill:small_ok({sig})"
        elif verdict is False:
            return self.large_model, f"skill:skip_small({sig})"
        else:
            return self.small_model, f"probe:insufficient_data({sig})"

    def solve(self, task: Dict) -> Dict:
        """
        完整解题流程: 决策 → 调模型 → 测试 → fallback (若需要) → 更新 skills。
        
        Returns:
            {
              "task_id": str,
              "signature": str,
              "decision": str,              # 第一次决策理由
              "attempts": [
                {"model": str, "success": bool, "cost": float}, ...
              ],
              "final_success": bool,
              "final_model": str,
              "total_cost": float,
              "attempts_count": int,
            }
        """
        prompt = task["prompt"]
        sig = extract_signature(prompt)
        first_model, decision = self.decide(prompt)

        attempts = []
        total_cost = 0
        final_success = False
        final_model = None

        # Attempt 1: decision 模型
        r1 = solve_task(first_model, task)
        attempts.append({
            "model": r1["actual_model"],
            "success": r1["success"],
            "cost": r1["cost_usd"],
        })
        total_cost += r1["cost_usd"]
        self.skill_book.update(prompt, r1["actual_model"], r1["success"], task.get("task_id", ""))

        if r1["success"]:
            final_success = True
            final_model = r1["actual_model"]
        else:
            # Fallback: 升级到 large model (若还没用过)
            if first_model != self.large_model:
                r2 = solve_task(self.large_model, task)
                attempts.append({
                    "model": r2["actual_model"],
                    "success": r2["success"],
                    "cost": r2["cost_usd"],
                })
                total_cost += r2["cost_usd"]
                self.skill_book.update(prompt, r2["actual_model"], r2["success"], task.get("task_id", ""))

                if r2["success"]:
                    final_success = True
                    final_model = r2["actual_model"]

        return {
            "task_id": task.get("task_id", ""),
            "signature": sig,
            "decision": decision,
            "attempts": attempts,
            "attempts_count": len(attempts),
            "final_success": final_success,
            "final_model": final_model,
            "total_cost": total_cost,
        }
