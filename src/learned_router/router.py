"""Router integration for learned prompt policies."""

from __future__ import annotations

from typing import Dict, Optional

from src.models import solve_task
from src.skills import SkillBook, extract_signature

from .policy import LearnedRouterPolicy


class RouterWithLearnedPolicy:
    """
    Drop-in router that replaces hand-written SkillBook decisions with a learned
    prompt classifier.

    The learned policy predicts the first model. If it chooses the small model
    and that attempt fails, the router keeps the same large-model fallback
    guarantee as RouterWithSkills.
    """

    def __init__(
        self,
        small_model: str,
        large_model: str,
        policy: LearnedRouterPolicy,
        skill_book: Optional[SkillBook] = None,
    ):
        self.small_model = small_model
        self.large_model = large_model
        self.policy = policy
        self.skill_book = skill_book or SkillBook()

    def decide(self, prompt: str) -> tuple[str, str]:
        return self.policy.decide(prompt, self.small_model, self.large_model)

    def solve(self, task: Dict) -> Dict:
        prompt = task["prompt"]
        sig = extract_signature(prompt)
        first_model, decision = self.decide(prompt)

        attempts = []
        total_cost = 0.0
        final_success = False
        final_model = None

        r1 = solve_task(first_model, task)
        attempts.append(
            {
                "model": r1["actual_model"],
                "success": r1["success"],
                "cost": r1["cost_usd"],
            }
        )
        total_cost += r1["cost_usd"]
        self.skill_book.update(prompt, r1["actual_model"], r1["success"], task.get("task_id", ""))

        if r1["success"]:
            final_success = True
            final_model = r1["actual_model"]
        elif first_model != self.large_model:
            r2 = solve_task(self.large_model, task)
            attempts.append(
                {
                    "model": r2["actual_model"],
                    "success": r2["success"],
                    "cost": r2["cost_usd"],
                }
            )
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
