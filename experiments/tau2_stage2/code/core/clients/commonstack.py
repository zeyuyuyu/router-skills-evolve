"""CommonStack client — the SOLE LLM gateway for v1.

CommonStack is OpenAI-compatible: POST {base_url}/chat/completions with the
usual {model, messages, tools, tool_choice, temperature, max_tokens} payload
and a Bearer auth header. See docs.commonstack.ai for specifics.

Retry policy: exponential backoff + jitter on {429, 500, 502, 503, 504}.
Hard-stop on 401 (auth) and 402 (credits) — those require human action.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass

import httpx

_RETRIABLE_STATUS = {429, 500, 502, 503, 504}
_HARD_STOP_STATUS = {401, 402}


@dataclass(frozen=True)
class CallResult:
    success: bool
    content: str
    tool_calls: list[dict]
    input_tokens: int
    output_tokens: int
    cached_read_tokens: int
    cached_write_tokens: int
    actual_usd: float       # priced upstream by cost_accounting.py, kept 0 here
    latency_ms: int
    error: str | None
    model_snapshot: str | None = None  # what the provider reports back (§11.3)


class CommonStackClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        max_retries: int = 3,
        retry_base_delay_s: float = 2.0,
        retry_max_delay_s: float = 60.0,
        timeout_s: float = 120.0,
    ) -> None:
        if not api_key:
            raise ValueError("CommonStack api_key is empty")
        self._key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._retry_base_delay_s = retry_base_delay_s
        self._retry_max_delay_s = retry_max_delay_s
        self._timeout_s = timeout_s

    async def call(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        extra_headers: dict | None = None,
        extra_body: dict | None = None,
    ) -> CallResult:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        if extra_body:
            payload.update(extra_body)
        start = time.perf_counter()
        resp: httpx.Response | None = None
        async with httpx.AsyncClient(timeout=self._timeout_s) as http:
            for attempt in range(self._max_retries + 1):
                resp = await http.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._key}",
                        "Content-Type": "application/json",
                        **(extra_headers or {}),
                    },
                    json=payload,
                )
                if resp.status_code in _HARD_STOP_STATUS:
                    break  # don't retry auth/credits failures
                if resp.status_code in _RETRIABLE_STATUS and attempt < self._max_retries:
                    delay = min(
                        self._retry_max_delay_s,
                        self._retry_base_delay_s * (2 ** attempt),
                    ) * (0.5 + random.random())
                    await asyncio.sleep(delay)
                    continue
                break
        assert resp is not None  # loop always assigns
        if resp.status_code < 400:
            return _parse_success(resp.json(), start)
        return CallResult(
            success=False, content="", tool_calls=[],
            input_tokens=0, output_tokens=0,
            cached_read_tokens=0, cached_write_tokens=0,
            actual_usd=0.0,
            latency_ms=int((time.perf_counter() - start) * 1000),
            error=f"HTTP {resp.status_code}: {resp.text[:200]}",
        )


def _parse_success(body: dict, start_perf: float) -> CallResult:
    choice = body["choices"][0]["message"]
    usage = body.get("usage", {}) or {}
    details = (usage.get("prompt_tokens_details") or {})
    return CallResult(
        success=True,
        content=choice.get("content") or "",
        tool_calls=choice.get("tool_calls") or [],
        input_tokens=int(usage.get("prompt_tokens", 0)),
        output_tokens=int(usage.get("completion_tokens", 0)),
        cached_read_tokens=int(details.get("cached_tokens", 0)),
        cached_write_tokens=int(usage.get("cache_creation_input_tokens", 0)),
        actual_usd=0.0,        # pricing applied by core/cost_accounting.py
        latency_ms=int((time.perf_counter() - start_perf) * 1000),
        error=None,
        model_snapshot=body.get("model"),
    )
