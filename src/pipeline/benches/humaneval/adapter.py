"""HumanEval bench adapter for the scaling pipeline.

Lets `scripts/run_full_pipeline.sh` run the original HumanEval experiment with
the CURRENT pipeline (procedural SkillBook, closed-loop routing, learnable
router), not just tau2/SWE. Unlike the tau2 adapter (which calls a remote
agent API), HumanEval runs LOCAL code models + pytest — no API key needed.

small / large are HF model ids (or LoRA adapter paths in cycles >= 1):
    small = Qwen/Qwen2.5-Coder-1.5B-Instruct   (default)
    large = Qwen/Qwen2.5-Coder-3B-Instruct     (default)

Each `run_task_pair` generates code with the small model and runs the task's
HumanEval test; the large model is run when small fails (or always under
force_both, for clean closed-loop oracle outcomes). Trace rows match the schema
in `src/pipeline/benches/__init__.py` and carry `small_completion` /
`large_completion` (the generated code) so `traces_to_sft.py` can build SFT
pairs.

SCALING_MOCK=1 => deterministic synthetic traces (no GPU), for smoke tests.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
HUMANEVAL_PATH = REPO_ROOT / "data" / "HumanEval.jsonl"


def _extract_signature(prompt: str) -> str:
    """Reuse the main-branch signature so skills cluster the same way."""
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from src.skills import extract_signature
        return extract_signature(prompt)
    except Exception:  # noqa: BLE001
        head = (prompt or "").strip().split("\n")[0][:80]
        return f"{head}::len_bucket={len(prompt or '') // 200}"


def _format_multiturn_prompt(messages: list[dict], style: str) -> str:
    """Build a multi-turn prompt string for the given chat style.

    `messages` is a list of {role: "user"|"assistant", content: str}.
    The returned string ends with the assistant-turn open tag so the model
    continues directly.
    """
    if style == "qwen-chat":
        SYSTEM = (
            "You are a Python coding assistant. Return only valid Python code. "
            "Do not include Markdown fences, explanations, examples, or tests."
        )
        parts = [f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"]
        for msg in messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)
    if style == "code":
        # Flatten: keep system header, then show each turn labelled
        parts = [
            "You are a Python coding assistant.\n"
            "Complete the following task by returning only valid Python code.\n\n"
        ]
        for msg in messages:
            label = "Task" if msg["role"] == "user" else "Previous attempt"
            parts.append(f"{label}:\n{msg['content']}\n\n")
        parts.append("Python code:\n")
        return "".join(parts)
    # alpaca fallback
    parts = []
    for msg in messages:
        if msg["role"] == "user":
            parts.append(f"### Instruction:\n{msg['content']}\n\n### Response:\n")
        else:
            parts.append(f"{msg['content']}\n\n")
    return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Backend resolution: a model_id is served by one of three backends.
#   "api"  → CommonStack OpenAI-compatible API (concurrent, e.g. openai/*,
#            deepseek/*). Used for teacher models that are frontier/hosted.
#   "vllm" → local vLLM server (OpenAI-compatible HTTP), if a URL is registered
#            for this model in HE_VLLM_MAP. Continuous batching → concurrent.
#   "hf"   → in-process transformers generate (default fallback, single request).
#
# This keeps the pipeline backward compatible: with no HE_VLLM_MAP set and a
# local HF model id, behaviour is identical to the original adapter.
# ─────────────────────────────────────────────────────────────────────────────

def _is_api_model(model_id: str) -> bool:
    """True if model_id is a CommonStack hosted model (→ API backend).

    Distinguishes lowercase provider-prefixed CommonStack ids (openai/gpt-…,
    deepseek/deepseek-v3.2) from HuggingFace repos / local paths
    (Qwen/Qwen2.5-…, deepseek-ai/…, /abs/path/to/adapter).
    """
    if not model_id or model_id.startswith(("/", "./", "../")):
        return False
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from src.config import DYNAMIC_PRICES
        if model_id in DYNAMIC_PRICES:
            return True
    except Exception:  # noqa: BLE001
        pass
    # Known CommonStack provider prefixes (lowercase). HF repos use different
    # casing/orgs (Qwen/, deepseek-ai/) and won't match.
    api_prefixes = (
        "openai/", "deepseek/", "anthropic/", "google/", "minimax/",
        "zai-org/", "qwen/", "moonshotai/", "x-ai/", "xiaomi/",
    )
    return model_id.startswith(api_prefixes)


def _vllm_url_for(model_id: str) -> str:
    """Return the local vLLM base_url serving model_id, or '' if none.

    HE_VLLM_MAP is a JSON object {model_id: base_url}. Set by the pipeline when
    it starts local vLLM servers for the small/large models.
    """
    raw = os.environ.get("HE_VLLM_MAP", "")
    if not raw:
        return ""
    try:
        import json as _json
        return _json.loads(raw).get(model_id, "") or ""
    except Exception:  # noqa: BLE001
        return ""


def _resolve_backend(model_id: str) -> str:
    """Pick the backend for a model id: 'api' | 'vllm' | 'hf'."""
    if _is_api_model(model_id):
        return "api"
    if _vllm_url_for(model_id):
        return "vllm"
    return "hf"


class Adapter:
    """HumanEval adapter — local code models + pytest."""

    def __init__(self) -> None:
        self.mock = os.environ.get("SCALING_MOCK", "0") == "1"
        self.max_new_tokens = int(os.environ.get("HE_MAX_NEW_TOKENS", "768"))
        self.prompt_style = os.environ.get("HE_PROMPT_STYLE", "qwen-chat")
        self.temperature = float(os.environ.get("HE_TEMPERATURE", "0.0"))
        # Max repair attempts before giving up / escalating to large model.
        # Turn 1 = first attempt; turns 2..N = error-feedback retries.
        self.max_repair_turns = int(os.environ.get("HE_MAX_REPAIR_TURNS", "3"))
        self._cache: dict[str, Any] = {}  # model_id -> (model, tokenizer)

    # ------------------------------------------------------------------ load
    def load_tasks(self, n: int, split: str = "train") -> list[dict]:
        import json
        _he_data = os.environ.get("HE_DATA")
        path = Path(_he_data) if _he_data else HUMANEVAL_PATH
        # A relative HE_DATA (e.g. "data/he_mbpp.jsonl" from config) resolves
        # against the repo root, not the caller's cwd.
        if _he_data and not path.is_absolute():
            path = REPO_ROOT / path
        tasks = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line)
                row = {
                    "task_id": t["task_id"],
                    "prompt": t["prompt"],
                    "entry_point": t["entry_point"],
                    "test": t["test"],
                    "_raw": t,
                }
                if "split" in t:
                    row["split"] = t["split"]
                tasks.append(row)
        # If rows carry an explicit "split" field (merged HumanEval+MBPP in
        # data/he_mbpp.jsonl), honor it; otherwise fall back to the deterministic
        # even=train / odd=eval positional split (plain HumanEval.jsonl).
        if any("split" in t for t in tasks):
            tasks = [t for t in tasks if t.get("split") == split]
        elif split == "eval":
            tasks = tasks[1::2]
        elif split == "train":
            tasks = tasks[0::2]
        return tasks[:n]

    # ----------------------------------------------------------- run pair
    def run_task_pair(self, task: dict, small_model: str, large_model: str,
                      cycle: int, force_both: bool = False,
                      skillbook=None) -> dict:
        sig = _extract_signature(task.get("prompt", ""))
        # Look up distilled procedure for this cluster (small model only).
        # Mirrors the SFT training format so the fine-tuned model sees the same
        # prompt shape at inference time as it did during training.
        procedure = ""
        if skillbook is not None:
            try:
                procedure = skillbook.get_procedure(task.get("prompt", "")) or ""
            except Exception:  # noqa: BLE001
                pass

        if self.mock:
            return self._mock_run(task, small_model, large_model, sig, cycle, force_both)

        # Both models get the procedure prefix (DEFAULT). This matches the SFT
        # training format (procedure+problem) for the small model, and for the large
        # (teacher) model it makes the teacher demonstration's prompt match the SFT
        # input — removing a teacher-forcing prompt/target mismatch — and can help on
        # format-heavy tasks. Cycle 0 has no skillbook yet (procedure=""), so this is
        # a no-op there and naturally takes effect from cycle 1 onward. Set
        # LARGE_USE_SKILLS=0 to restore raw-prompt teacher.
        large_proc = "" if os.environ.get("LARGE_USE_SKILLS") == "0" else procedure
        s_ok, s_code, s_turns = self._gen_and_test(small_model, task, procedure=procedure)
        large_skipped = False
        if s_ok and not force_both:
            l_ok, l_code, l_turns, large_skipped = False, "", [], True
            decision = "probe:small->small_OK"
            final_model, final_success = small_model, True
        else:
            l_ok, l_code, l_turns = self._gen_and_test(large_model, task, procedure=large_proc)
            if s_ok:
                decision = "oracle:small_OK+large_run"
                final_model, final_success = small_model, True
            else:
                decision = f"probe:small_fail->large_{'OK' if l_ok else 'fail'}"
                final_model, final_success = large_model, l_ok

        return {
            "task_id": task["task_id"],
            "signature": sig,
            "decision": decision,
            "attempts": 1 if (s_ok and not force_both) else 2,
            "attempts_count": 1 if (s_ok and not force_both) else 2,
            "final_success": final_success,
            "final_model": final_model,
            "total_cost": 0.0,
            "round": cycle,
            "small_success": s_ok,
            "large_success": l_ok,
            "small_cost": 0.0,
            "large_cost": 0.0,
            "large_skipped": large_skipped,
            "prompt": task.get("prompt", ""),
            "small_completion": s_code,
            "large_completion": "" if large_skipped else l_code,
            # Full per-turn trajectories for multi-turn SFT data extraction.
            "small_turns": s_turns,
            "large_turns": l_turns,
        }

    # ----------------------------------------------------------- internals
    def _get_model(self, model_id: str):
        if model_id in self._cache:
            return self._cache[model_id]
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # model_id may be a base HF id or a LoRA adapter dir (cycles >= 1).
        adapter_dir = Path(model_id)
        is_adapter = (adapter_dir / "adapter_config.json").exists()
        if is_adapter:
            import json as _json
            base = _json.loads((adapter_dir / "adapter_config.json").read_text()).get(
                "base_model_name_or_path", os.environ.get("HE_SMALL_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct"))
            tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, model_id)
        else:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model.eval()
        self._cache[model_id] = (model, tok)
        return model, tok

    # System instruction shared by all chat backends (api / vllm). The hf
    # backend embeds the same text in _format_multiturn_prompt.
    _CHAT_SYSTEM = (
        "You are a Python coding assistant. Return only valid Python code. "
        "Do not include Markdown fences, explanations, examples, or tests."
    )

    def _openai_client_for(self, model_id: str, backend: str):
        """Build (and cache) an OpenAI-compatible client for api/vllm backends."""
        cache_key = f"_client::{backend}::{model_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        from openai import OpenAI
        if backend == "api":
            sys.path.insert(0, str(REPO_ROOT))
            from src.config import COMMONSTACK_API_KEY, COMMONSTACK_BASE_URL
            client = OpenAI(api_key=COMMONSTACK_API_KEY, base_url=COMMONSTACK_BASE_URL)
        else:  # vllm
            base_url = _vllm_url_for(model_id)
            client = OpenAI(api_key=os.environ.get("HE_VLLM_API_KEY", "EMPTY"),
                            base_url=base_url)
        self._cache[cache_key] = client
        return client

    def _generate_completion(self, model_id: str, messages: list[dict],
                             backend: str) -> str:
        """Generate one completion for a multi-turn message list via `backend`."""
        if backend in ("api", "vllm"):
            client = self._openai_client_for(model_id, backend)
            chat = [{"role": "system", "content": self._CHAT_SYSTEM}, *messages]
            resp = client.chat.completions.create(
                model=model_id,
                messages=chat,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
            return resp.choices[0].message.content or ""

        # hf backend (default): in-process transformers generate
        import torch
        model, tok = self._get_model(model_id)
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": tok.pad_token_id,
            "eos_token_id": tok.eos_token_id,
        }
        if self.prompt_style == "qwen-chat":
            qend = tok.convert_tokens_to_ids("<|im_end|>")
            if isinstance(qend, int) and qend >= 0:
                gen_kwargs["eos_token_id"] = qend
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
        prompt_str = _format_multiturn_prompt(messages, self.prompt_style)
        inputs = tok(prompt_str, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        return tok.decode(new_tokens, skip_special_tokens=True)

    def _gen_and_test(self, model_id: str, task: dict,
                      procedure: str = "") -> tuple[bool, str, list]:
        """Generate code and test it, retrying with error feedback (ReAct repair).

        Returns (ok, final_code, turns) where turns is a list of
        {turn, code, ok, error} dicts — one per attempt. The small model uses
        `procedure` as a prefix on the first turn; the large model gets the raw
        prompt (procedure="" by default). Repair turns are capped at
        self.max_repair_turns; the model escalates if still failing after that.

        Generation backend is resolved per model_id: api (CommonStack), vllm
        (local server), or hf (in-process). See _resolve_backend.
        """
        sys.path.insert(0, str(REPO_ROOT))
        from src.models import extract_code, run_humaneval_test

        backend = _resolve_backend(model_id)

        raw_problem = task["prompt"]
        first_user_content = f"{procedure}\n\n---\n\n{raw_problem}" if procedure else raw_problem
        messages = [{"role": "user", "content": first_user_content}]
        turns = []

        for turn_idx in range(self.max_repair_turns):
            completion = self._generate_completion(model_id, messages, backend)
            code = extract_code(completion, task["entry_point"], task["prompt"])
            ok, error = run_humaneval_test(task, code)

            turns.append({"turn": turn_idx + 1, "code": code, "ok": ok, "error": error})

            if ok:
                break

            if turn_idx < self.max_repair_turns - 1:
                # Feed the error back as the next user turn
                messages.append({"role": "assistant", "content": code})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your code failed with the following error:\n\n"
                        f"{error}\n\n"
                        "Analyze the error and return a corrected version of the complete function."
                    ),
                })

        final = turns[-1]
        return final["ok"], final["code"], turns

    def _mock_run(self, task, small_model, large_model, sig, cycle, force_both):
        h = int(hashlib.md5(f"{task['task_id']}|{cycle}".encode()).hexdigest(), 16)
        s_ok = (h % 10) < 6
        l_ok = (h % 10) < 9
        prompt = task.get("prompt", "")
        # always emit the generated code (pass or fail) — matches the real
        # adapter + satisfies the no-empty-trace guard.
        l_code = f"# mock large attempt for {task['task_id']} (ok={l_ok})"
        s_code = f"# mock small attempt for {task['task_id']} (ok={s_ok})"
        if s_ok and not force_both:
            return {"task_id": task["task_id"], "signature": sig,
                    "decision": "probe:small->small_OK", "attempts": 1, "attempts_count": 1,
                    "final_success": True, "final_model": small_model, "total_cost": 0.0,
                    "round": cycle, "small_success": True, "large_success": False,
                    "small_cost": 0.0, "large_cost": 0.0, "large_skipped": True,
                    "prompt": prompt, "small_completion": s_code, "large_completion": ""}
        return {"task_id": task["task_id"], "signature": sig,
                "decision": (f"oracle:small_OK+large_run" if s_ok else
                             f"probe:small_fail->large_{'OK' if l_ok else 'fail'}"),
                "attempts": 2, "attempts_count": 2,
                "final_success": (True if s_ok else l_ok),
                "final_model": (small_model if s_ok else large_model), "total_cost": 0.0,
                "round": cycle, "small_success": s_ok, "large_success": l_ok,
                "small_cost": 0.0, "large_cost": 0.0, "large_skipped": False,
                "prompt": prompt, "small_completion": s_code, "large_completion": l_code}
