"""Inference wrapper for a trained BERT router."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from .model import BertRouter


class LearnedRouterPolicy:
    """Load a trained router checkpoint and return small/large decisions."""

    def __init__(
        self,
        router: BertRouter,
        *,
        threshold: float | None = None,
        device: str | None = None,
    ):
        self.router = router
        self.threshold = router.config.threshold if threshold is None else threshold
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        threshold: float | None = None,
        device: str | None = None,
    ) -> "LearnedRouterPolicy":
        router = BertRouter.load(path, device=device)
        return cls(router, threshold=threshold, device=device)

    def predict_large_probability(self, prompt: str) -> float:
        return self.router.predict_proba([prompt], device=self.device)[0]

    def decide(self, prompt: str, small_model: str, large_model: str) -> Tuple[str, str]:
        p_large = self.predict_large_probability(prompt)
        if p_large >= self.threshold:
            return large_model, f"learned:large(p_large={p_large:.3f})"
        return small_model, f"learned:small(p_large={p_large:.3f})"

