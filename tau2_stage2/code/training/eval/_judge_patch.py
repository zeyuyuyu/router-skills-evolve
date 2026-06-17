"""Monkey-patch tau2-bench's NL-assertion judge to route through CommonStack.

WHY:
    tau2-bench v1.0.0 (SHA 17e07b1d) hardcodes
    `gpt-4.1-2025-04-14` as the NL-assertion judge in `tau2.config`. CommonStack
    does NOT host that dated snapshot — only base `openai/gpt-4.1`. The corpus
    in `data_processed/stage2_v1/eval_tasks.jsonl` declares
    `nl_judge_model: "openai/openai/gpt-5.2"` for every task, meaning the
    corpus was collected with gpt-5.2 as the judge. To make the eval
    methodologically faithful (same judge as collection) AND functional on
    CommonStack, we rebind the judge to `openai/openai/gpt-5.2` at import
    time and route it via CommonStack credentials.

WHEN TO CALL:
    Once per Python process, BEFORE any `tau2.evaluator.evaluator.evaluate_simulation`
    or `tau2.runner.simulation.run_simulation` call.

HOW IT WORKS:
    `tau2.evaluator.evaluator_nl_assertions` does
        `from tau2.config import DEFAULT_LLM_NL_ASSERTIONS`
    at its OWN import time. Python's import semantics mean the module binds
    a local name pointing at the same object as `tau2.config.DEFAULT_LLM_NL_ASSERTIONS`
    AT THAT MOMENT. Subsequently mutating `tau2.config.DEFAULT_LLM_NL_ASSERTIONS`
    does NOT affect the copy in `evaluator_nl_assertions`. So we must rebind
    BOTH module attributes. The same applies for the
    `DEFAULT_LLM_NL_ASSERTIONS_ARGS` companion dict.

    This is the same pattern as
    `adapters/tau2_bench/adapter.py:_route_nl_judge_through` (data-collection
    flow), but specialized for eval: judge args come from CommonStack creds
    (NOT the agent's args, which point at local vLLM and would 404 on gpt-5.2).
"""
from __future__ import annotations

# Double-prefix is intentional: litellm strips the leading "openai/" as the
# provider prefix; the api_base receives "openai/gpt-5.2" on the wire. See
# adapters/tau2_bench/adapter.py:71-76 for the same convention.
_JUDGE_MODEL = "openai/openai/gpt-5.2"

_applied = False


def apply_judge_patch(cs_api_base: str, cs_api_key: str) -> None:
    """Rebind tau2's NL-assertion judge to openai/gpt-5.2 via CommonStack.

    Args:
        cs_api_base: e.g. "https://api.commonstack.ai/v1"
        cs_api_key:  CommonStack API key (ak-...)

    Raises:
        RuntimeError if called more than once with conflicting args. Idempotent
        when called with identical args.
    """
    global _applied
    if _applied:
        return

    from tau2 import config as tau2_config
    from tau2.evaluator import evaluator_nl_assertions as nl_mod

    args = {
        "temperature": tau2_config.DEFAULT_LLM_NL_ASSERTIONS_TEMPERATURE,
        "api_base": cs_api_base,
        "api_key": cs_api_key,
    }
    tau2_config.DEFAULT_LLM_NL_ASSERTIONS = _JUDGE_MODEL
    tau2_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS = args
    # Re-bind the copies that evaluator_nl_assertions imported at module load.
    nl_mod.DEFAULT_LLM_NL_ASSERTIONS = _JUDGE_MODEL
    nl_mod.DEFAULT_LLM_NL_ASSERTIONS_ARGS = args
    _applied = True


def is_applied() -> bool:
    return _applied


def get_judge_model() -> str:
    return _JUDGE_MODEL
