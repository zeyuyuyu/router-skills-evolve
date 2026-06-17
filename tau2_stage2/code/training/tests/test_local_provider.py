"""Tests for the local vLLM provider configuration."""
import pytest

from training.eval.local_provider import (
    build_litellm_kwargs_for_local,
    register_local_model_pricing,
)


def test_litellm_kwargs_default_port():
    kwargs = build_litellm_kwargs_for_local(port=8000)
    assert kwargs["api_base"] == "http://localhost:8000/v1"
    assert kwargs["api_key"] == "dummy"
    assert kwargs["custom_llm_provider"] == "openai"


def test_litellm_kwargs_custom_host_port():
    kwargs = build_litellm_kwargs_for_local(port=9123, host="127.0.0.1")
    assert kwargs["api_base"] == "http://127.0.0.1:9123/v1"


def test_register_zero_pricing():
    pricing = {}
    register_local_model_pricing(pricing, model_id="local/student")
    assert pricing["local/student"]["input_usd_per_m"] == 0.0
    assert pricing["local/student"]["output_usd_per_m"] == 0.0
    assert pricing["local/student"]["cache_read_usd_per_m"] == 0.0
    assert pricing["local/student"]["cache_write_usd_per_m"] == 0.0


def test_register_does_not_overwrite_existing():
    pricing = {"openai/gpt-5.2": {"input_usd_per_m": 1.0}}
    register_local_model_pricing(pricing, model_id="local/student")
    # Existing entry untouched.
    assert pricing["openai/gpt-5.2"]["input_usd_per_m"] == 1.0
    assert "local/student" in pricing
