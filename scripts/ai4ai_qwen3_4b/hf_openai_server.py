#!/usr/bin/env python3
"""Tiny OpenAI-compatible chat server for local HF causal LMs.

This is intentionally small: it exists for offline GPU nodes where vLLM is not
available but tau2/LiteLLM still need a `/v1/chat/completions` endpoint.
Qwen3.5 emits tool calls as XML via its chat template; the server parses that
format back into OpenAI-style `tool_calls`.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


FUNCTION_RE = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    re.DOTALL,
)
JSON_TOOL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)
PARAM_RE = re.compile(
    r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)
PYTHON_CALL_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$",
    re.DOTALL,
)


def _jsonable_arg(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _schema_by_function(tools: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(tools, list):
        return out
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") or {}
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        params = fn.get("parameters") or {}
        if isinstance(name, str) and isinstance(params, dict):
            out[name] = params
    return out


def _schema_type(schema: dict[str, Any]) -> str | None:
    typ = schema.get("type")
    if isinstance(typ, str):
        return typ
    for key in ("anyOf", "oneOf"):
        variants = schema.get(key)
        if isinstance(variants, list):
            for item in variants:
                if isinstance(item, dict) and isinstance(item.get("type"), str):
                    if item["type"] != "null":
                        return item["type"]
    return None


def _coerce_tool_args(name: str, args: dict[str, Any], tools: Any) -> dict[str, Any]:
    schema = _schema_by_function(tools).get(name) or {}
    properties = schema.get("properties") or {}
    if not isinstance(properties, dict):
        return {key: value for key, value in args.items() if value is not None}
    allowed = set(properties)
    coerced = {
        key: value
        for key, value in args.items()
        if key in allowed and value is not None
    }
    for key, value in list(coerced.items()):
        prop_schema = properties.get(key) or {}
        if not isinstance(prop_schema, dict):
            continue
        typ = _schema_type(prop_schema)
        try:
            if typ == "string" and value is not None:
                coerced[key] = str(value)
            elif typ == "integer" and value not in (None, ""):
                coerced[key] = int(value)
            elif typ == "number" and value not in (None, ""):
                coerced[key] = float(value)
            elif typ == "boolean" and isinstance(value, str):
                coerced[key] = value.strip().lower() in {"1", "true", "yes", "on"}
        except (TypeError, ValueError):
            pass
    return coerced


def _tool_call_object(name: str, args: dict[str, Any], tools: Any) -> dict[str, Any]:
    args = _coerce_tool_args(name, args, tools)
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False),
        },
    }


def _parse_python_tool_call(body: str) -> tuple[str, dict[str, Any]] | None:
    """Parse SFT traces like `get_order({'order_id': 'x'})`.

    The first AI4AI SFT pass distilled teacher completions that wrapped tool
    calls as text inside `<tool_call>name({...})</tool_call>`. Qwen's native
    chat template expects structured tool calls, so convert this legacy
    Python-call shape into OpenAI-compatible arguments at serving time.
    """
    body = body.strip()
    match = PYTHON_CALL_RE.match(body)
    if not match:
        return None
    name = match.group(1)
    arg_src = match.group(2).strip()
    if not arg_src:
        return name, {}

    try:
        parsed = ast.parse(f"f({arg_src})", mode="eval")
    except SyntaxError:
        return None
    call = parsed.body
    if not isinstance(call, ast.Call):
        return None

    args: dict[str, Any] = {}
    if call.args:
        try:
            first = ast.literal_eval(call.args[0])
        except (ValueError, SyntaxError):
            first = None
        if isinstance(first, dict):
            args.update(first)
    for kw in call.keywords:
        if kw.arg is None:
            continue
        try:
            args[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            return None
    return name, args


def parse_qwen_tool_calls(text: str, tools: Any = None) -> tuple[str, list[dict[str, Any]]]:
    """Return assistant content plus OpenAI-compatible tool call objects."""
    tool_calls: list[dict[str, Any]] = []
    for match in FUNCTION_RE.finditer(text):
        name = match.group(1).strip()
        body = match.group(2)
        args = {
            p.group(1).strip(): _jsonable_arg(p.group(2))
            for p in PARAM_RE.finditer(body)
        }
        tool_calls.append(_tool_call_object(name, args, tools))
    xml_stripped = FUNCTION_RE.sub("", text)
    for match in JSON_TOOL_RE.finditer(xml_stripped):
        body = match.group(1).strip()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            parsed_python_call = _parse_python_tool_call(body)
            if parsed_python_call is None:
                continue
            name, args = parsed_python_call
            tool_calls.append(_tool_call_object(name, args, tools))
            continue
        if not isinstance(payload, dict):
            continue
        name = payload.get("name")
        args = payload.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if isinstance(name, str) and isinstance(args, dict):
            tool_calls.append(_tool_call_object(name, args, tools))
    content = JSON_TOOL_RE.sub("", xml_stripped).strip()
    return content, tool_calls


def strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    if text.startswith("<think>"):
        return ""
    return text.strip()


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make OpenAI messages palatable to the Qwen chat template."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        item: dict[str, Any] = {
            "role": role,
            "content": msg.get("content") or "",
        }
        if role == "assistant" and msg.get("tool_calls"):
            calls = []
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                calls.append({"function": {"name": fn.get("name", ""), "arguments": args}})
            item["tool_calls"] = calls
        out.append(item)
    return out


class ModelState:
    def __init__(
        self,
        model_path: Path,
        served_model_name: str,
        device: str,
        dtype: str,
        attn_implementation: str,
        trust_remote_code: bool,
    ) -> None:
        self.served_model_name = served_model_name
        self.lock = threading.Lock()
        torch_dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
            dtype=torch_dtype,
            device_map={"": device},
            attn_implementation=attn_implementation,
        )
        self.model.eval()

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        messages = normalize_messages(request.get("messages") or [])
        tools = request.get("tools")
        max_new_tokens = int(
            request.get("max_tokens")
            or request.get("max_completion_tokens")
            or 1024
        )
        temperature = float(request.get("temperature") or 0.0)
        top_p = float(request.get("top_p") or 1.0)
        do_sample = temperature > 0.0

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs.update({"temperature": temperature, "top_p": top_p})

        with self.lock, torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output[0][inputs["input_ids"].shape[-1] :]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        content, tool_calls = parse_qwen_tool_calls(strip_thinking(raw), tools=tools)

        completion_tokens = int(new_tokens.shape[-1])
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        message: dict[str, Any] = {
            "role": "assistant",
            "content": None if tool_calls and not content else content,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


class Handler(BaseHTTPRequestHandler):
    state: ModelState

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[hf_openai_server] {self.address_string()} {fmt % args}", flush=True)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.state.served_model_name,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "local",
                        }
                    ],
                },
            )
            return
        self._send_json(404, {"error": {"message": f"not found: {self.path}"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": f"not found: {self.path}"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length) or b"{}")
            if request.get("stream"):
                raise ValueError("stream=true is not supported")
            response = self.state.generate(request)
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": {"message": str(exc), "type": type(exc).__name__}})
            return
        self._send_json(200, response)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--served-model-name", default="evol-llm-student")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--no-trust-remote-code", action="store_true")
    args = ap.parse_args()

    Handler.state = ModelState(
        args.model_path,
        args.served_model_name,
        args.device,
        args.dtype,
        args.attn_implementation,
        trust_remote_code=not args.no_trust_remote_code,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"[hf_openai_server] serving {args.served_model_name} "
        f"from {args.model_path} on {args.host}:{args.port}",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
