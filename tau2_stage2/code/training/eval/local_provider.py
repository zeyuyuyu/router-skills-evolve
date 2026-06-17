"""Local vLLM provider — RETAINED BUT NOT ON THE LIVE EVAL PATH.

Originally written to plug the local vLLM endpoint into `pipeline.runner
--provider local`. The eval pipeline was rewritten in commit ec726e6 to
drive `tau2.cli run` directly via subprocess (see
`code/training/eval/harness.py:run_tau2_eval_for_group`), which builds
its agent-llm-args JSON inline. Neither helper here is imported by any
orchestration script or production code path; only the unit test
imports them.

Kept rather than deleted because the helpers (build_litellm_kwargs_for_local,
register_local_model_pricing) are small, tested, and a reasonable starting
point if the eval flow ever migrates back to a LiteLLM-based agent layer
instead of the tau2-CLI subprocess.
"""
from __future__ import annotations


def build_litellm_kwargs_for_local(port: int, host: str = "localhost") -> dict:
    """Return LiteLLM completion() kwargs that route to a local vLLM instance.

    vLLM exposes an OpenAI-compatible server at /v1/. LiteLLM treats this as
    a regular OpenAI endpoint via custom_llm_provider="openai" + api_base.
    The api_key is unvalidated by vLLM but required by the OpenAI client.
    """
    return {
        "api_base": f"http://{host}:{port}/v1",
        "api_key": "dummy",
        "custom_llm_provider": "openai",
    }


def register_local_model_pricing(
    pricing_dict: dict, model_id: str = "local/student"
) -> dict:
    """Add a zero-cost pricing entry for the locally-served student model."""
    pricing_dict[model_id] = {
        "input_usd_per_m": 0.0,
        "output_usd_per_m": 0.0,
        "cache_read_usd_per_m": 0.0,
        "cache_write_usd_per_m": 0.0,
    }
    return pricing_dict
