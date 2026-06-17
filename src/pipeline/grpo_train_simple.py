#!/usr/bin/env python3
"""Phase 3b: GRPO on-policy RL for the small coding model.

After SFT (Phase 3a) warm-starts the model on teacher + self-repair data,
GRPO further refines it with a direct test-execution reward signal. The reward
is binary: passing all HumanEval tests = 1.0, failing = 0.0.

Key design choices
------------------
- Group size K (--n-generations): generate K completions per prompt, normalize
  rewards within the group → advantage = (r - mean) / (std + eps). This is the
  GRPO estimator from DeepSeekMath; no value network needed.
- Warm-start: loads the Phase 3a SFT checkpoint (LoRA adapter or full model).
  Merges LoRA weights before wrapping with a fresh LoRA for GRPO, so the
  reference policy is clean.
- Procedure prefix: same format as SFT training — skillbook procedure prepended
  to each prompt so the inference context matches training.
- Output: LoRA adapter saved to grpo_adapter/. The pipeline's Phase 1 checkpoint
  handoff prefers grpo_adapter/ over llm_adapter/checkpoint-best.

Usage:
    python src/pipeline/grpo_train_simple.py \\
        --model results/.../cycle_N/llm_adapter \\
        --bench-data data/HumanEval.jsonl \\
        --output-dir results/.../cycle_N/grpo_adapter \\
        [--skillbook results/.../cycle_N/skillbook.json] \\
        [--n-generations 8] [--epochs 1] [--lr 5e-6]

Algorithm comparison (--algo flag)
------------------------------------
GRPO  loss_type="grpo"  clip(r, 1-ε, 1+ε)            sequence-level norm
DAPO  loss_type="dapo"  clip(r, 1-ε_low, 1+ε_high)   token-level global norm
      + dynamic sampling: zero-grad for all-same-reward groups (reward wrapper)

Both implemented via TRL 1.6 native GRPOConfig. No subclassing needed.

Requires: trl >= 1.6, transformers, peft, datasets
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_humaneval_tasks(data_path: str, split: str = "train") -> list[dict]:
    tasks = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    if split == "train":
        return tasks[0::2]
    if split == "eval":
        return tasks[1::2]
    return tasks


def _load_procedures(skillbook_path: str | None):
    if not skillbook_path:
        return lambda _: ""
    p = Path(skillbook_path)
    if not p.exists():
        return lambda _: ""
    try:
        from src.skills import SkillBook
        sb = SkillBook()
        sb.load(p)
        print(f"[grpo] loaded skillbook ({len(sb.skills)} clusters) from {skillbook_path}",
              flush=True)
        return sb.get_procedure
    except Exception as e:  # noqa: BLE001
        print(f"[grpo] WARN could not load skillbook: {e}", flush=True)
        return lambda _: ""


def _build_dataset(tasks: list[dict], get_procedure, prompt_style: str):
    """Build a HF Dataset with formatted prompts and task metadata.

    The `prompt` column is the full formatted prompt string fed to GRPOTrainer.
    Extra columns (entry_point, test_code, raw_problem) are passed as kwargs to
    the reward function via TRL's automatic column forwarding.
    """
    from datasets import Dataset
    from src.pipeline.train_small_model import format_prompt

    rows = []
    for t in tasks:
        problem = t["prompt"]
        procedure = get_procedure(problem)
        raw_prompt = f"{procedure}\n\n---\n\n{problem}" if procedure else problem
        rows.append({
            "prompt": format_prompt(raw_prompt, style=prompt_style),
            "task_id": t["task_id"],
            "entry_point": t["entry_point"],
            "test_code": t["test"],
            "raw_problem": problem,
        })
    return Dataset.from_list(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Reward function
# ─────────────────────────────────────────────────────────────────────────────

def _make_reward_fn():
    """Return a GRPO reward function backed by HumanEval test execution.

    TRL 0.9+ passes extra dataset columns as kwargs. The reward is binary:
    1.0 if all tests pass, 0.0 otherwise.
    """
    from src.models import extract_code, run_humaneval_test

    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        entry_points = kwargs.get("entry_point", [""] * len(completions))
        test_codes   = kwargs.get("test_code",   [""] * len(completions))
        raw_problems = kwargs.get("raw_problem",  [""] * len(completions))
        rewards = []
        for completion, ep, tc, rp in zip(completions, entry_points, test_codes, raw_problems):
            try:
                task = {"entry_point": ep, "test": tc, "prompt": rp}
                code = extract_code(completion, ep, rp)
                ok, _ = run_humaneval_test(task, code)
                rewards.append(1.0 if ok else 0.0)
            except Exception:  # noqa: BLE001
                rewards.append(0.0)
        n_pass = sum(1 for r in rewards if r > 0)
        print(f"[grpo_reward] batch pass={n_pass}/{len(rewards)}", flush=True)
        return rewards

    return reward_fn


def _make_dapo_reward_fn(base_fn, n_generations: int):
    """Dynamic sampling wrapper (DAPO §3.2).

    When all K completions for a prompt share the same binary reward (all-pass
    or all-fail), the GRPO group-normalized advantage is identically zero and
    the batch contributes no gradient signal. DAPO skips such groups entirely.

    We approximate that here by zeroing the rewards for all-same groups, which
    drives their advantage to ~0 via (0-0)/(0+ε). This is gradient-equivalent
    to skipping, without requiring changes to TRL's sampler.
    """
    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        rewards = list(base_fn(completions, **kwargs))
        K = n_generations
        if K > 1 and len(rewards) >= K and len(rewards) % K == 0:
            n_prompts = len(rewards) // K
            filtered = 0
            for i in range(n_prompts):
                grp = rewards[i * K : (i + 1) * K]
                if len(set(grp)) == 1:      # all pass or all fail
                    for j in range(K):
                        rewards[i * K + j] = 0.0
                    filtered += 1
            if filtered:
                print(
                    f"[dapo] dynamic_sampling: zeroed {filtered}/{n_prompts} "
                    f"zero-variance groups  (informative={n_prompts - filtered}/{n_prompts})",
                    flush=True,
                )
        return rewards
    return reward_fn


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_model_and_tok(model_path: str):
    """Load model + tokenizer. Merges LoRA adapter if detected."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = Path(model_path)
    is_adapter = (adapter_dir / "adapter_config.json").exists()

    if is_adapter:
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        base = cfg.get("base_model_name_or_path",
                       os.environ.get("HE_SMALL_MODEL",
                                      "Qwen/Qwen2.5-Coder-1.5B-Instruct"))
        print(f"[grpo] loading LoRA adapter ({model_path}) on top of {base}", flush=True)
        tok   = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
        # Merge adapter weights before adding a fresh GRPO LoRA
        model = model.merge_and_unload()
        print("[grpo] LoRA merged", flush=True)
    else:
        print(f"[grpo] loading base model: {model_path}", flush=True)
        tok   = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


# ─────────────────────────────────────────────────────────────────────────────
# Multi-turn repair rollout (ReAct) GRPO/DAPO — aligned with the benchmark
# ─────────────────────────────────────────────────────────────────────────────

def _repair_rollout(model, tok, task, procedure, *, style, max_turns,
                    temperature, max_new_tokens):
    """One multi-turn repair trajectory, on-policy with ``model``.

    Mirrors the benchmark's HumanEval adapter ReAct repair: generate code, run
    the tests, and on failure feed the error back as the next user turn. Returns
    ``{"reward": 1.0/0.0, "turns": [{"prompt_ids", "completion_ids"}]}`` — the
    captured token ids feed the GRPO token loss directly (no re-tokenisation).
    """
    import torch
    from src.pipeline.benches.humaneval.adapter import _format_multiturn_prompt
    from src.models import extract_code, run_humaneval_test

    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0,
                  "pad_token_id": tok.pad_token_id, "eos_token_id": tok.eos_token_id}
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    if style == "qwen-chat":
        qend = tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(qend, int) and qend >= 0:
            gen_kwargs["eos_token_id"] = qend

    raw_problem = task["prompt"]
    first = f"{procedure}\n\n---\n\n{raw_problem}" if procedure else raw_problem
    messages = [{"role": "user", "content": first}]
    turns, ok = [], False
    for turn_idx in range(max_turns):
        prompt_str = _format_multiturn_prompt(messages, style)
        enc = tok(prompt_str, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        new_ids = out[0][enc["input_ids"].shape[-1]:]
        completion = tok.decode(new_ids, skip_special_tokens=True)
        turns.append({"prompt_ids": enc["input_ids"][0].tolist(),
                      "completion_ids": new_ids.tolist()})
        code = extract_code(completion, task["entry_point"], task["prompt"])
        ok, error = run_humaneval_test(task, code)
        if ok:
            break
        if turn_idx < max_turns - 1:
            messages.append({"role": "assistant", "content": code})
            messages.append({"role": "user", "content": (
                f"Your code failed with the following error:\n\n{error}\n\n"
                "Analyze the error and return a corrected version of the complete function.")})
    return {"reward": 1.0 if ok else 0.0, "turns": turns}


def _repair_rollout_batched(model, tok, task, procedure, *, n, style, max_turns,
                            temperature, max_new_tokens):
    """K on-policy repair trajectories for one task, generated in BATCHES.

    Same result shape as K calls to `_repair_rollout` (a list of n
    {"reward", "turns":[{"prompt_ids","completion_ids"}]} dicts), but each turn
    issues ONE batched `model.generate` over all still-failing trajectories
    instead of n sequential calls — ~Kx fewer generate calls, far higher GPU
    utilisation (the main HumanEval-GRPO speedup while vLLM is driver-blocked).
    """
    import torch
    from src.pipeline.benches.humaneval.adapter import _format_multiturn_prompt
    from src.models import extract_code, run_humaneval_test

    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0,
                  "pad_token_id": tok.pad_token_id, "eos_token_id": tok.eos_token_id}
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    if style == "qwen-chat":
        qend = tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(qend, int) and qend >= 0:
            gen_kwargs["eos_token_id"] = qend

    raw_problem = task["prompt"]
    first = f"{procedure}\n\n---\n\n{raw_problem}" if procedure else raw_problem
    convs = [[{"role": "user", "content": first}] for _ in range(n)]
    turns_rec: list = [[] for _ in range(n)]
    ok = [False] * n
    done = [False] * n

    for turn_idx in range(max_turns):
        active = [i for i in range(n) if not done[i]]
        if not active:
            break
        prompts = [_format_multiturn_prompt(convs[i], style) for i in active]
        # Left-pad so all completions start at the same column for batched decode.
        old_side = tok.padding_side
        tok.padding_side = "left"
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        tok.padding_side = old_side
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        gen = out[:, enc["input_ids"].shape[1]:]  # new tokens, aligned across rows

        for bi, i in enumerate(active):
            new_ids = gen[bi].tolist()
            while new_ids and new_ids[-1] == tok.pad_token_id:  # trim right pad
                new_ids.pop()
            completion = tok.decode(new_ids, skip_special_tokens=True)
            # clean (unpadded) prompt ids for this trajectory's loss example
            prompt_ids = tok(prompts[bi], return_tensors="pt")["input_ids"][0].tolist()
            turns_rec[i].append({"prompt_ids": prompt_ids, "completion_ids": new_ids})
            code = extract_code(completion, task["entry_point"], task["prompt"])
            passed, error = run_humaneval_test(task, code)
            if passed:
                ok[i] = True
                done[i] = True
            elif turn_idx < max_turns - 1:
                convs[i].append({"role": "assistant", "content": code})
                convs[i].append({"role": "user", "content": (
                    f"Your code failed with the following error:\n\n{error}\n\n"
                    "Analyze the error and return a corrected version of the complete function.")})
            else:
                done[i] = True

    return [{"reward": 1.0 if ok[i] else 0.0, "turns": turns_rec[i]} for i in range(n)]


def _repair_rollout_vllm(client, served_model, tok, task, procedure, *, n, max_turns,
                         temperature, max_new_tokens):
    """K multi-turn repair trajectories via a vLLM OpenAI endpoint (parallel).

    Valid because GRPO collects ALL rollouts BEFORE the gradient update — the
    rollout policy is STATIC (the served SFT checkpoint), so no weight sync is
    needed. Generation goes over HTTP (vLLM continuous-batches the concurrent
    requests → far higher throughput than in-process HF generate) while the
    multi-turn ReAct repair loop is preserved.

    The OpenAI chat API returns TEXT, so prompt/completion token ids for the GRPO
    loss are reconstructed with the SAME tokenizer + chat template vLLM used.
    """
    from concurrent.futures import ThreadPoolExecutor
    from src.models import extract_code, run_humaneval_test

    SYSTEM = ("You are a Python coding assistant. Return only valid Python code. "
              "Do not include Markdown fences, explanations, examples, or tests.")
    first = f"{procedure}\n\n---\n\n{task['prompt']}" if procedure else task["prompt"]
    convs = [[{"role": "user", "content": first}] for _ in range(n)]
    turns_rec: list = [[] for _ in range(n)]
    ok = [False] * n
    done = [False] * n

    def _gen(i):
        msgs = [{"role": "system", "content": SYSTEM}, *convs[i]]
        r = client.chat.completions.create(
            model=served_model, messages=msgs,
            temperature=temperature, max_tokens=max_new_tokens)
        return i, (r.choices[0].message.content or "")

    for turn_idx in range(max_turns):
        active = [i for i in range(n) if not done[i]]
        if not active:
            break
        with ThreadPoolExecutor(max_workers=len(active)) as pool:
            results = list(pool.map(_gen, active))
        for i, completion in results:
            msgs = [{"role": "system", "content": SYSTEM}, *convs[i]]
            # ids consistent with what vLLM saw (same template) for the GRPO loss
            prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                                 tokenize=True)
            completion_ids = tok(completion, add_special_tokens=False)["input_ids"]
            turns_rec[i].append({"prompt_ids": prompt_ids, "completion_ids": completion_ids})
            code = extract_code(completion, task["entry_point"], task["prompt"])
            passed, error = run_humaneval_test(task, code)
            if passed:
                ok[i] = True
                done[i] = True
            elif turn_idx < max_turns - 1:
                convs[i].append({"role": "assistant", "content": code})
                convs[i].append({"role": "user", "content": (
                    f"Your code failed with the following error:\n\n{error}\n\n"
                    "Analyze the error and return a corrected version of the complete function.")})
            else:
                done[i] = True

    return [{"reward": 1.0 if ok[i] else 0.0, "turns": turns_rec[i]} for i in range(n)]


def _repair_rollout_vllm_global(client, served_model, tok, tasks, get_procedure, *,
                                n, max_turns, temperature, max_new_tokens, concurrency):
    """ALL tasks × K rollouts concurrently against vLLM, turn by turn.

    Instead of one task at a time (K concurrent each), fire every still-active
    (task, rollout) unit at once — vLLM continuous-batches hundreds of requests,
    so a turn costs ~one batched forward regardless of task count. Returns the
    same `groups` shape as the per-task loop.
    """
    from concurrent.futures import ThreadPoolExecutor
    from src.models import extract_code, run_humaneval_test

    SYSTEM = ("You are a Python coding assistant. Return only valid Python code. "
              "Do not include Markdown fences, explanations, examples, or tests.")
    units = []
    for ti, task in enumerate(tasks):
        proc = get_procedure(task["prompt"]) or ""
        first = f"{proc}\n\n---\n\n{task['prompt']}" if proc else task["prompt"]
        for _ in range(n):
            units.append({"ti": ti, "conv": [{"role": "user", "content": first}],
                          "turns": [], "ok": False, "done": False})

    def _gen(u):
        msgs = [{"role": "system", "content": SYSTEM}, *u["conv"]]
        r = client.chat.completions.create(
            model=served_model, messages=msgs,
            temperature=temperature, max_tokens=max_new_tokens)
        return u, (r.choices[0].message.content or "")

    for turn_idx in range(max_turns):
        active = [u for u in units if not u["done"]]
        if not active:
            break
        with ThreadPoolExecutor(max_workers=min(concurrency, len(active))) as pool:
            results = list(pool.map(_gen, active))
        for u, completion in results:
            task = tasks[u["ti"]]
            msgs = [{"role": "system", "content": SYSTEM}, *u["conv"]]
            prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
            completion_ids = tok(completion, add_special_tokens=False)["input_ids"]
            u["turns"].append({"prompt_ids": prompt_ids, "completion_ids": completion_ids})
            code = extract_code(completion, task["entry_point"], task["prompt"])
            passed, error = run_humaneval_test(task, code)
            if passed:
                u["ok"] = True
                u["done"] = True
            elif turn_idx < max_turns - 1:
                u["conv"].append({"role": "assistant", "content": code})
                u["conv"].append({"role": "user", "content": (
                    f"Your code failed with the following error:\n\n{error}\n\n"
                    "Analyze the error and return a corrected version of the complete function.")})
            else:
                u["done"] = True
        n_done = sum(1 for u in units if u["done"])
        print(f"[grpo] vLLM rollout turn {turn_idx+1}/{max_turns}: "
              f"{len(active)} reqs, {n_done}/{len(units)} units done", flush=True)

    groups = []
    for ti, task in enumerate(tasks):
        rollouts = [{"reward": 1.0 if u["ok"] else 0.0, "turns": u["turns"]}
                    for u in units if u["ti"] == ti]
        groups.append({"task_id": task["task_id"], "rollouts": rollouts})
    return groups


def _run_repair_grpo(args, tasks, get_procedure) -> int:
    from src.pipeline.grpo_core import compute_advantages, grpo_update

    is_dapo = args.algo == "dapo"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_new_tokens = int(os.environ.get("HE_MAX_NEW_TOKENS", "768"))
    print(f"[grpo] repair rollout: algo={args.algo} K={args.n_generations} "
          f"max_turns={args.max_repair_turns} temp={args.temperature} "
          f"dynamic_sampling={is_dapo}", flush=True)

    if args.dry_run:
        # Repair rollouts need the policy (GPU); preview config + a sample prompt.
        from src.pipeline.benches.humaneval.adapter import _format_multiturn_prompt
        proc = get_procedure(tasks[0]["prompt"]) if tasks else ""
        first = f"{proc}\n\n---\n\n{tasks[0]['prompt']}" if proc and tasks else (
            tasks[0]["prompt"] if tasks else "")
        sample = _format_multiturn_prompt([{"role": "user", "content": first}], args.prompt_style)
        print(f"[grpo] dry-run ({args.algo}, repair): {len(tasks)} tasks × K={args.n_generations} "
              f"× ≤{args.max_repair_turns} turns. Sample turn-1 prompt[:300]:\n{sample[:300]}",
              flush=True)
        json.dump({"algo": args.algo, "rollout": "repair", "dry_run": True,
                   "n_tasks": len(tasks), "k": args.n_generations,
                   "max_turns": args.max_repair_turns},
                  open(out_dir / "grpo_info.json", "w"), indent=2)
        return 0

    from peft import LoraConfig, TaskType, get_peft_model
    model, tok = _load_model_and_tok(args.model)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none"))
    model.eval()

    # vLLM rollout: serve the (static) rollout policy and generate over HTTP.
    # GRPO_ROLLOUT_VLLM_URL points at a vLLM server hosting THIS checkpoint
    # (args.model); GRPO_ROLLOUT_VLLM_MODEL is its served name. The gradient
    # update still runs on the in-process `model`.
    vllm_url = os.environ.get("GRPO_ROLLOUT_VLLM_URL", "")
    vllm_client = None
    if vllm_url:
        from openai import OpenAI
        vllm_client = OpenAI(api_key=os.environ.get("HE_VLLM_API_KEY", "EMPTY"),
                             base_url=vllm_url)
        served = os.environ.get("GRPO_ROLLOUT_VLLM_MODEL", args.model)
        print(f"[grpo] rollout via vLLM @ {vllm_url} (model={served}); "
              f"update in-process. Multi-turn repair preserved.", flush=True)

    # ── ROLLOUT: K on-policy repair trajectories per task ────────────────────
    if vllm_client is not None:
        # vLLM: all tasks × K fired concurrently (server batches hundreds) — far
        # faster than one task at a time.
        concurrency = int(os.environ.get("GRPO_ROLLOUT_CONCURRENCY", "64"))
        print(f"[grpo] vLLM global rollout: {len(tasks)}×K={args.n_generations} "
              f"units, concurrency={concurrency}", flush=True)
        groups = _repair_rollout_vllm_global(
            vllm_client, served, tok, tasks, get_procedure,
            n=args.n_generations, max_turns=args.max_repair_turns,
            temperature=args.temperature, max_new_tokens=max_new_tokens,
            concurrency=concurrency)
        for g in groups:
            n_pass = sum(1 for r in g["rollouts"] if r["reward"] > 0)
            print(f"[grpo] {g['task_id']}  pass={n_pass}/{len(g['rollouts'])}", flush=True)
    else:
        groups = []
        for i, task in enumerate(tasks, 1):
            proc = get_procedure(task["prompt"]) or ""
            rollouts = _repair_rollout_batched(
                model, tok, task, proc, n=args.n_generations, style=args.prompt_style,
                max_turns=args.max_repair_turns, temperature=args.temperature,
                max_new_tokens=max_new_tokens)
            n_pass = sum(1 for r in rollouts if r["reward"] > 0)
            print(f"[grpo] task {i}/{len(tasks)} {task['task_id']}  "
                  f"pass={n_pass}/{len(rollouts)}  "
                  f"avg_turns={sum(len(r['turns']) for r in rollouts)/len(rollouts):.1f}", flush=True)
            groups.append({"task_id": task["task_id"], "rollouts": rollouts})

    kept, adv_stats = compute_advantages(groups, dynamic_sampling=is_dapo)
    print(f"[grpo] advantage: {adv_stats}", flush=True)
    if not kept:
        print("[grpo] no informative groups; nothing to train.", flush=True)
        (out_dir / "STATUS").write_text("no_informative_rollouts\n")
        json.dump({"algo": args.algo, "rollout": "repair", **adv_stats},
                  open(out_dir / "grpo_info.json", "w"), indent=2)
        return 0

    # ── EXAMPLES: each repair turn → advantage-weighted token-loss example ───
    examples = []
    for r in kept:
        adv = float(r["advantage"])
        for turn in r["turns"]:
            ids = turn["prompt_ids"] + turn["completion_ids"]
            mask = [0] * len(turn["prompt_ids"]) + [1] * len(turn["completion_ids"])
            if sum(mask) == 0:
                continue
            if len(ids) > args.max_len:          # keep the completion, drop prompt head
                ids, mask = ids[-args.max_len:], mask[-args.max_len:]
            examples.append({"input_ids": ids, "completion_mask": mask, "advantage": adv})
    print(f"[grpo] {len(examples)} repair-turn examples", flush=True)

    n_steps = grpo_update(model, tok, examples, args)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"[grpo] saved LoRA to {out_dir} (update steps={n_steps})", flush=True)
    json.dump({"algo": args.algo, "rollout": "repair", "model": args.model,
               "n_generations": args.n_generations, "max_turns": args.max_repair_turns,
               "epochs": args.epochs, "lr": args.lr, "beta": args.beta,
               "clip_low": args.clip_low, "clip_high": args.clip_high if is_dapo else args.clip_low,
               "dapo_dynamic_sampling": is_dapo, "update_steps": n_steps,
               "n_examples": len(examples), **adv_stats},
              open(out_dir / "grpo_info.json", "w"), indent=2)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="SFT checkpoint path or base HF model ID")
    ap.add_argument("--bench-data",
                    default=str(REPO_ROOT / "data" / "HumanEval.jsonl"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--skillbook", default=None)
    ap.add_argument("--n-generations", type=int,
                    default=int(os.environ.get("GRPO_N_GENERATIONS", "8")),
                    help="K: completions per prompt (group size for advantage estimation)")
    ap.add_argument("--epochs", type=int,
                    default=int(os.environ.get("GRPO_EPOCHS", "1")))
    ap.add_argument("--lr", type=float,
                    default=float(os.environ.get("GRPO_LR", "5e-6")))
    ap.add_argument("--batch-size", type=int,
                    default=int(os.environ.get("GRPO_BATCH_SIZE", "4")))
    ap.add_argument("--max-completion-len", type=int, default=768)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--beta", type=float,
                    default=float(os.environ.get("GRPO_BETA", "0.04")),
                    help="KL penalty coefficient")
    ap.add_argument("--prompt-style",
                    default=os.environ.get("HE_PROMPT_STYLE", "qwen-chat"))
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--algo", default=os.environ.get("GRPO_ALGO", "grpo"),
        choices=["grpo", "dapo"],
        help=(
            "Training algorithm. 'dapo' enables three improvements over vanilla GRPO: "
            "(1) dynamic sampling — zero-grad for all-same-reward groups; "
            "(2) clip-higher — asymmetric PPO ratio clip [1-ε_low, 1+ε_high]; "
            "(3) token-level loss (loss_type=token, if TRL supports it). "
            "Use GRPO_ALGO=dapo to enable from shell."
        ),
    )
    ap.add_argument("--clip-low",  type=float,
                    default=float(os.environ.get("DAPO_CLIP_LOW",  "0.2")),
                    help="PPO lower clip bound (both algos; standard ε=0.2)")
    ap.add_argument("--clip-high", type=float,
                    default=float(os.environ.get("DAPO_CLIP_HIGH", "0.5")),
                    help="DAPO upper clip bound for positive-advantage tokens")
    ap.add_argument("--use-vllm", action="store_true",
                    default=os.environ.get("GRPO_USE_VLLM", "0") == "1",
                    help="(single rollout only) use vLLM colocate for generation; "
                         "TRL auto-syncs the training weights into the vLLM engine "
                         "each step (the 'weight passing'). Needs vllm importable in "
                         "THIS env (run under .vllm_venv). Env: GRPO_USE_VLLM=1")
    ap.add_argument("--vllm-gpu-util", type=float,
                    default=float(os.environ.get("GRPO_VLLM_GPU_UTIL", "0.3")),
                    help="vLLM KV-cache fraction of the training GPU (colocate "
                         "shares the GPU with the trainer, so keep this modest)")
    ap.add_argument(
        "--rollout", default=os.environ.get("HE_GRPO_ROLLOUT", "repair"),
        choices=["repair", "single"],
        help=(
            "Rollout style. 'repair' (default) matches the benchmark: multi-turn "
            "ReAct repair — generate code, run tests, feed the error back, retry "
            "(K trajectories/task, reward = final pass); trajectory advantage is "
            "shared across the turn's tokens (uses grpo_core). 'single' is the "
            "legacy one-shot TRL path (one completion/sample, no repair loop)."
        ),
    )
    ap.add_argument("--max-repair-turns", type=int,
                    default=int(os.environ.get("HE_MAX_REPAIR_TURNS", "3")),
                    help="repair rollout: max attempts before giving up (turn 1 = first try)")
    ap.add_argument("--temperature", type=float,
                    default=float(os.environ.get("GRPO_TEMPERATURE", "0.8")),
                    help="repair rollout sampling temperature (needs >0 for intra-group variance)")
    ap.add_argument("--max-len", type=int, default=int(os.environ.get("GRPO_MAX_LEN", "4096")),
                    help="repair rollout: per-example token cap")
    ap.add_argument("--logging-steps", type=int, default=5,
                    help="repair rollout: GRPO update log interval")
    args = ap.parse_args()

    get_procedure = _load_procedures(args.skillbook)
    tasks = _load_humaneval_tasks(args.bench_data, split=args.split)
    print(f"[grpo] {len(tasks)} tasks (split={args.split})  rollout={args.rollout}", flush=True)

    if args.rollout == "repair":
        # Multi-turn repair GRPO/DAPO — aligned with the benchmark's ReAct repair.
        return _run_repair_grpo(args, tasks, get_procedure)

    dataset = _build_dataset(tasks, get_procedure, args.prompt_style)
    print(f"[grpo] dataset built: {len(dataset)} prompts", flush=True)

    if args.dry_run:
        print(f"[grpo] dry-run ({args.algo}): stopping after dataset build")
        print(f"  prompt[:300]:\n{dataset[0]['prompt'][:300]}")
        return 0

    from peft import LoraConfig, TaskType, get_peft_model
    from trl import GRPOConfig, GRPOTrainer

    is_dapo = args.algo == "dapo"
    print(f"[grpo] algo={args.algo}", flush=True)

    # Build reward function — DAPO wraps with dynamic sampling.
    base_reward_fn = _make_reward_fn()
    reward_fn = (
        _make_dapo_reward_fn(base_reward_fn, args.n_generations)
        if is_dapo else base_reward_fn
    )

    model, tok = _load_model_and_tok(args.model)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[grpo] trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)",
          flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── GRPOConfig: TRL 1.6 has native DAPO support ──────────────────────────
    # loss_type="grpo" : symmetric clip clamp(r, 1-ε, 1+ε), sequence-level norm
    # loss_type="dapo" : asymmetric clip clamp(r, 1-ε_low, 1+ε_high),
    #                    token-level global norm (better gradient signal)
    # Dynamic sampling (DAPO §3.2) is handled by _make_dapo_reward_fn above;
    # TRL normalises advantages within each group so zeroed rewards → zero grad.
    # ─────────────────────────────────────────────────────────────────────────
    if is_dapo:
        loss_type   = "dapo"
        epsilon_low  = args.clip_low
        epsilon_high = args.clip_high   # asymmetric upper bound (> epsilon_low)
        print(
            f"[dapo] loss_type=dapo  ε_low={epsilon_low}  ε_high={epsilon_high}  "
            f"dynamic_sampling=True",
            flush=True,
        )
    else:
        loss_type    = "grpo"
        epsilon_low  = args.clip_low
        epsilon_high = args.clip_low    # same as low → symmetric clip

    # vLLM colocate rollout: TRL spins up an in-process vLLM engine on the
    # training GPU and re-syncs the policy weights into it every step (the
    # "weight passing"). Big speedup for generation; needs vllm importable here.
    vllm_kwargs = {}
    if args.use_vllm:
        vllm_kwargs = {
            "use_vllm": True,
            "vllm_mode": "colocate",
            "vllm_gpu_memory_utilization": args.vllm_gpu_util,
        }
        print(f"[grpo] vLLM colocate rollout ON  gpu_util={args.vllm_gpu_util} "
              f"(weights auto-synced to engine each step)", flush=True)

    grpo_cfg = GRPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 8 // args.batch_size),
        learning_rate=args.lr,
        max_completion_length=args.max_completion_len,
        num_generations=args.n_generations,
        temperature=0.8,
        beta=args.beta,
        loss_type=loss_type,
        epsilon=epsilon_low,
        epsilon_high=epsilon_high,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        **vllm_kwargs,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_cfg,
        train_dataset=dataset,
        processing_class=tok,
    )

    print(
        f"[grpo] training  algo={args.algo}  K={args.n_generations}  "
        f"epochs={args.epochs}  lr={args.lr}  beta={args.beta}",
        flush=True,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"[grpo] saved to {out_dir}", flush=True)

    log_history = list(getattr(trainer.state, "log_history", []) or [])
    json.dump({
        "algo": args.algo,
        "trl_loss_type": loss_type,
        "model": args.model,
        "bench_data": args.bench_data,
        "n_generations": args.n_generations,
        "epochs": args.epochs,
        "lr": args.lr,
        "beta": args.beta,
        "epsilon_low": epsilon_low,
        "epsilon_high": epsilon_high,
        "dapo_dynamic_sampling": is_dapo,
        "rollout": "single",
        "use_vllm": args.use_vllm,
        "vllm_gpu_util": args.vllm_gpu_util if args.use_vllm else None,
        "split": args.split,
        "n_tasks": len(tasks),
        "skillbook": args.skillbook,
        "prompt_style": args.prompt_style,
        "log_history": log_history,  # per-step reward/kl/loss for inspection
    }, open(out_dir / "grpo_info.json", "w"), indent=2)

    # Training curve (reward / kl / loss vs step) for eyeballing.
    try:
        from src.train_plots import plot_training_curves
        plot_training_curves(log_history, out_dir / "training_curve.png",
                             title=f"{args.algo.upper()} — {Path(str(out_dir)).name}")
    except Exception as e:  # noqa: BLE001
        print(f"[grpo] WARN could not plot training curve: {e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
