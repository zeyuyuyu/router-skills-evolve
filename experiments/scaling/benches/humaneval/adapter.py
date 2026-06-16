"""HumanEval bench adapter for the scaling pipeline.

Lets `scaling/run_full_pipeline.sh` run the original HumanEval experiment with
the CURRENT pipeline (procedural SkillBook, closed-loop routing, learnable
router), not just tau2/SWE. Unlike the tau2 adapter (which calls a remote
agent API), HumanEval runs LOCAL code models + pytest — no API key needed.

small / large are HF model ids (or LoRA adapter paths in cycles >= 1):
    small = Qwen/Qwen2.5-Coder-1.5B-Instruct   (default)
    large = Qwen/Qwen2.5-Coder-3B-Instruct     (default)

Each `run_task_pair` generates code with the small model and runs the task's
HumanEval test; the large model is run when small fails (or always under
force_both, for clean closed-loop oracle outcomes). Trace rows match the schema
in `experiments/scaling/benches/__init__.py` and carry `small_completion` /
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


class Adapter:
    """HumanEval adapter — local code models + pytest."""

    def __init__(self) -> None:
        self.mock = os.environ.get("SCALING_MOCK", "0") == "1"
        self.max_new_tokens = int(os.environ.get("HE_MAX_NEW_TOKENS", "768"))
        self.prompt_style = os.environ.get("HE_PROMPT_STYLE", "qwen-chat")
        self.temperature = float(os.environ.get("HE_TEMPERATURE", "0.0"))
        self._cache: dict[str, Any] = {}  # model_id -> (model, tokenizer)

    # ------------------------------------------------------------------ load
    def load_tasks(self, n: int, split: str = "train") -> list[dict]:
        import json
        path = Path(os.environ.get("HE_DATA", str(HUMANEVAL_PATH)))
        tasks = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line)
                tasks.append({
                    "task_id": t["task_id"],
                    "prompt": t["prompt"],
                    "entry_point": t["entry_point"],
                    "test": t["test"],
                    "_raw": t,
                })
        # deterministic split: even-indexed = train, odd = eval (stable across cycles)
        if split == "eval":
            tasks = tasks[1::2]
        elif split == "train":
            tasks = tasks[0::2]
        return tasks[:n]

    # ----------------------------------------------------------- run pair
    def run_task_pair(self, task: dict, small_model: str, large_model: str,
                      cycle: int, force_both: bool = False) -> dict:
        sig = _extract_signature(task.get("prompt", ""))
        if self.mock:
            return self._mock_run(task, small_model, large_model, sig, cycle, force_both)

        s_ok, s_code = self._gen_and_test(small_model, task)
        large_skipped = False
        if s_ok and not force_both:
            l_ok, l_code, large_skipped = False, "", True
            decision = "probe:small->small_OK"
            final_model, final_success = small_model, True
        else:
            l_ok, l_code = self._gen_and_test(large_model, task)
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
            "total_cost": 0.0,  # local models; cost tracked elsewhere if needed
            "round": cycle,
            "small_success": s_ok,
            "large_success": l_ok,
            "small_cost": 0.0,
            "large_cost": 0.0,
            "large_skipped": large_skipped,
            "prompt": task.get("prompt", ""),
            # Record the ACTUAL generated code whenever the model ran (pass or
            # fail) — it is the real trace and satisfies the no-empty-trace
            # guard. traces_to_sft keeps only small-fail/large-OK rows and uses
            # large_completion, so failing code never becomes an SFT target.
            "small_completion": s_code,
            "large_completion": "" if large_skipped else l_code,
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

    def _gen_and_test(self, model_id: str, task: dict) -> tuple[bool, str]:
        import torch
        sys.path.insert(0, str(REPO_ROOT))
        from src.models import extract_code, run_humaneval_test
        from experiments.train_small_model import format_prompt
        model, tok = self._get_model(model_id)
        prompt = format_prompt(task["prompt"], style=self.prompt_style)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
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
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        new = out[0][inputs["input_ids"].shape[-1]:]
        completion = tok.decode(new, skip_special_tokens=True)
        code = extract_code(completion, task["entry_point"], task["prompt"])
        ok, _err = run_humaneval_test(task, code)
        return bool(ok), code

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
