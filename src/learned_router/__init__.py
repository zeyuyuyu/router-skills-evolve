"""Trainable router components.

This package is intentionally separate from the historical SkillBook router:
SkillBook is a statistics table, while the classes here learn a prompt -> route
policy from traces.
"""

from .data import RouterTraceExample, load_router_examples
from .model import BertRouter, BertRouterConfig
from .policy import LearnedRouterPolicy
from .router import RouterWithLearnedPolicy

__all__ = [
    "RouterTraceExample",
    "load_router_examples",
    "BertRouter",
    "BertRouterConfig",
    "LearnedRouterPolicy",
    "RouterWithLearnedPolicy",
]
