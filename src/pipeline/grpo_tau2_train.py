#!/usr/bin/env python3
"""Phase 3b (tau-2): trajectory-level GRPO / DAPO on-policy RL.

HumanEval GRPO ([grpo_train_simple.py]) is single-turn: prompt -> code,
reward = test execution. tau-2 is **multi-turn agentic** (agent <-> user-sim <->
tools), so a "completion" is not one string but a whole trajectory, and the
reward is the env outcome (`passed`). TRL's GRPOTrainer cannot drive that rollout
itself, so this module does it explicitly:

    1. ROLLOUT   For each task, sample K full tau-2 rollouts through the adapter
                 (agent = policy under training, served via vLLM; user-sim +
                 NL judge = gpt-5.2). Each rollout yields reward = `passed` and
                 the structured per-agent-turn trajectory (input_messages ->
                 response). See Tau2 adapter `rollout()`.
    2. ADVANTAGE Group-normalise the K rewards: A_i = (r_i - mean)/(std + eps)
                 — the GRPO estimator (DeepSeekMath), no value network.
                 DAPO dynamic sampling: drop groups whose K rewards are all equal
                 (zero variance => zero advantage => zero gradient), so the batch
                 is filled only with informative groups.
    3. EXAMPLES  Decompose each rollout into per-step (context -> response)
                 supervised-shaped examples, each carrying its trajectory's
                 advantage. The agent response tokens are the only ones scored
                 (prompt/tool/user tokens are masked).
    4. UPDATE    GRPO / DAPO policy-gradient update on the response tokens:
                   per-token L = -min(ratio * A, clip(ratio, 1-e_lo, 1+e_hi) * A)
                 GRPO  : symmetric clip (e_hi == e_lo), sequence-level mean.
                 DAPO  : clip-higher (e_hi > e_lo), token-level global mean.
                 Optional KL(beta) penalty against the adapter-disabled ref.
                 LoRA adapter saved to --output-dir (pipeline prefers it over
                 the SFT checkpoint, same as HumanEval).

This is ONE GRPO iteration (rollout with the current policy -> update). Multiple
iterations require re-serving the updated weights to vLLM between rounds; the
pipeline owns that loop (--rounds is a convenience that re-rolls against the same
endpoint and is only truly on-policy if the caller resyncs vLLM).

Mock / dry-run
--------------
SCALING_MOCK=1 makes the adapter synthesise deterministic rollouts (no API/GPU),
and --dry-run stops after example building. Together they validate the full
rollout -> advantage -> example pipeline without a GPU or gpt-5.2 key.

Requires (real runs): torch, transformers, peft. trl is NOT required.
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
# Skillbook procedure prefix (same contract as HumanEval GRPO / SFT)
# ─────────────────────────────────────────────────────────────────────────────

def _load_procedures(skillbook_path: str | None):
    if not skillbook_path or not Path(skillbook_path).exists():
        return lambda _: ""
    try:
        from src.skills import SkillBook
        sb = SkillBook()
        sb.load(Path(skillbook_path))
        print(f"[grpo-tau2] loaded skillbook ({len(sb.skills)} clusters) from "
              f"{skillbook_path}", flush=True)
        return sb.get_procedure
    except Exception as e:  # noqa: BLE001
        print(f"[grpo-tau2] WARN could not load skillbook: {e}", flush=True)
        return lambda _: ""


# ─────────────────────────────────────────────────────────────────────────────
# 1. Rollout
# ─────────────────────────────────────────────────────────────────────────────

def collect_rollouts(adapter, tasks, rollout_model, k, temperature):
    """Return one group per task: ``{"task_id", "rollouts": [{passed,reward,steps}]}``."""
    groups = []
    for i, task in enumerate(tasks, 1):
        try:
            rollouts = adapter.rollout(task, rollout_model, n=k, temperature=temperature)
        except Exception as e:  # noqa: BLE001
            print(f"[grpo-tau2] rollout FAILED task={task.get('task_id')}: {e}",
                  file=sys.stderr, flush=True)
            continue
        rewards = [float(r.get("reward", 0.0)) for r in rollouts]
        n_pass = sum(1 for r in rollouts if r.get("passed"))
        print(f"[grpo-tau2] task {i}/{len(tasks)} {task.get('task_id')}  "
              f"pass={n_pass}/{len(rollouts)}  rewards={rewards}", flush=True)
        groups.append({"task_id": task.get("task_id"), "rollouts": rollouts})
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# 2. Advantage + DAPO dynamic sampling  → shared with HumanEval in grpo_core
# ─────────────────────────────────────────────────────────────────────────────

from src.pipeline.grpo_core import compute_advantages, grpo_update  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-step examples (tokenisation)
# ─────────────────────────────────────────────────────────────────────────────

def _assistant_message(response: dict) -> dict:
    """Reconstruct the assistant chat message from a StepData.response."""
    msg = {"role": "assistant", "content": response.get("content") or ""}
    tool_calls = response.get("tool_calls") or []
    norm = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = fn.get("name") or tc.get("name")
        args = fn.get("arguments") if fn else tc.get("arguments")
        if isinstance(args, (dict, list)):
            args = json.dumps(args, ensure_ascii=False)
        if name is None:
            continue
        norm.append({
            "id": tc.get("id", f"call_{len(norm)}"),
            "type": "function",
            "function": {"name": name, "arguments": args if args is not None else "{}"},
        })
    if norm:
        msg["tool_calls"] = norm
    return msg


def build_examples(rollouts, tokenizer, get_procedure, *, max_len: int):
    """Tokenise each agent step into (input_ids, completion_mask, advantage).

    The skillbook procedure (if any) is prepended to the first system message so
    the training context matches inference (the adapter prepends it at rollout).
    Steps that fail to tokenise (template/tool-call edge cases) are skipped.
    """
    examples, n_steps, n_skipped = [], 0, 0
    for r in rollouts:
        adv = float(r["advantage"])
        steps = r.get("steps") or []
        procedure = get_procedure(steps[0]["input_messages"][-1].get("content", "")
                                  if steps and steps[0].get("input_messages") else "")
        for step in steps:
            n_steps += 1
            in_msgs = list(step.get("input_messages") or [])
            functions = step.get("functions") or []
            if not in_msgs:
                n_skipped += 1
                continue
            if procedure:
                in_msgs = _prepend_procedure(in_msgs, procedure)
            try:
                kwargs = {"tokenize": True}
                if functions:
                    kwargs["tools"] = functions
                prompt_ids = tokenizer.apply_chat_template(
                    in_msgs, add_generation_prompt=True, **kwargs)
                full_ids = tokenizer.apply_chat_template(
                    in_msgs + [_assistant_message(step["response"])],
                    add_generation_prompt=False, **kwargs)
            except Exception as e:  # noqa: BLE001
                n_skipped += 1
                if n_skipped <= 3:
                    print(f"[grpo-tau2] WARN step tokenise failed (skipping): {e}",
                          file=sys.stderr)
                continue
            if len(full_ids) <= len(prompt_ids):
                n_skipped += 1
                continue
            full_ids = full_ids[:max_len]
            n_prompt = min(len(prompt_ids), len(full_ids))
            mask = [0] * n_prompt + [1] * (len(full_ids) - n_prompt)
            if sum(mask) == 0:
                n_skipped += 1
                continue
            examples.append({"input_ids": full_ids, "completion_mask": mask,
                             "advantage": adv})
    stats = {"n_steps_seen": n_steps, "n_examples": len(examples),
             "n_steps_skipped": n_skipped}
    return examples, stats


def _prepend_procedure(messages: list, procedure: str) -> list:
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "system":
            m["content"] = f"{procedure}\n\n---\n\n{m.get('content','')}"
            return out
    # No system message: inject one at the front.
    return [{"role": "system", "content": procedure}] + out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model loading + GRPO/DAPO update
# ─────────────────────────────────────────────────────────────────────────────

def _load_model_and_tok(model_path: str, base_env: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = Path(model_path)
    is_adapter = (adapter_dir / "adapter_config.json").exists()
    if is_adapter:
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        base = cfg.get("base_model_name_or_path", os.environ.get(base_env, model_path))
        print(f"[grpo-tau2] loading LoRA adapter ({model_path}) on {base}", flush=True)
        tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        print("[grpo-tau2] SFT LoRA merged into base", flush=True)
    else:
        print(f"[grpo-tau2] loading base model: {model_path}", flush=True)
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="SFT checkpoint path or base HF model id (the policy to update)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rollout-model",
                    default=os.environ.get("TAU2_GRPO_ROLLOUT_MODEL")
                    or f"openai/{os.environ.get('TAU2_LOCAL_SERVED_MODEL', 'evol-llm-student')}",
                    help="agent model id the adapter rolls out (the vLLM-served policy)")
    ap.add_argument("--n-tasks", type=int, default=int(os.environ.get("GRPO_N_TASKS", "100")))
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--skillbook", default=None)
    ap.add_argument("--n-generations", type=int,
                    default=int(os.environ.get("GRPO_N_GENERATIONS", "8")),
                    help="K: rollouts per task (group size for advantage)")
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("GRPO_EPOCHS", "1")))
    ap.add_argument("--lr", type=float, default=float(os.environ.get("GRPO_LR", "5e-6")))
    ap.add_argument("--batch-size", type=int,
                    default=int(os.environ.get("GRPO_BATCH_SIZE", "2")))
    ap.add_argument("--beta", type=float, default=float(os.environ.get("GRPO_BETA", "0.0")),
                    help="KL penalty vs the adapter-disabled reference (0 disables)")
    ap.add_argument("--temperature", type=float,
                    default=float(os.environ.get("GRPO_TEMPERATURE", "1.0")))
    ap.add_argument("--max-len", type=int, default=int(os.environ.get("GRPO_MAX_LEN", "8192")))
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--algo", default=os.environ.get("GRPO_ALGO", "grpo"),
                    choices=["grpo", "dapo"])
    ap.add_argument("--clip-low", type=float,
                    default=float(os.environ.get("DAPO_CLIP_LOW", "0.2")))
    ap.add_argument("--clip-high", type=float,
                    default=float(os.environ.get("DAPO_CLIP_HIGH", "0.5")))
    ap.add_argument("--dry-run", action="store_true",
                    help="stop after example building (validates rollout+advantage wiring)")
    args = ap.parse_args()

    from src.pipeline.benches import load_adapter
    adapter = load_adapter("tau2_bench")
    is_dapo = args.algo == "dapo"
    print(f"[grpo-tau2] algo={args.algo}  K={args.n_generations}  "
          f"dynamic_sampling={is_dapo}  clip=[{args.clip_low},"
          f"{args.clip_high if is_dapo else args.clip_low}]  "
          f"rollout_model={args.rollout_model}  domain={os.environ.get('TAU2_DOMAIN')}",
          flush=True)

    tasks = adapter.load_tasks(args.n_tasks, split=args.split)
    print(f"[grpo-tau2] {len(tasks)} tasks (split={args.split})", flush=True)

    groups = collect_rollouts(adapter, tasks, args.rollout_model,
                              args.n_generations, args.temperature)
    rollouts, adv_stats = compute_advantages(groups, dynamic_sampling=is_dapo)
    print(f"[grpo-tau2] advantage: {adv_stats}", flush=True)

    get_procedure = _load_procedures(args.skillbook)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not rollouts:
        print("[grpo-tau2] no informative rollouts (all groups zero-variance); "
              "nothing to train. Writing STATUS and exiting.", flush=True)
        (out_dir / "STATUS").write_text("no_informative_rollouts\n")
        json.dump({"algo": args.algo, **adv_stats}, open(out_dir / "grpo_info.json", "w"),
                  indent=2)
        return 0

    if args.dry_run:
        # Tokeniser-free preview so dry-run needs no model download.
        n_steps = sum(len(r.get("steps") or []) for r in rollouts)
        print(f"[grpo-tau2] dry-run ({args.algo}): {len(rollouts)} rollouts, "
              f"~{n_steps} agent steps ready; advantages "
              f"{[round(r['advantage'], 3) for r in rollouts[:8]]}...", flush=True)
        json.dump({"algo": args.algo, "dry_run": True, **adv_stats,
                   "n_agent_steps": n_steps},
                  open(out_dir / "grpo_info.json", "w"), indent=2)
        return 0

    base_env = "HE_SMALL_MODEL"  # reuse the same base-fallback env as HumanEval path
    model, tok = _load_model_and_tok(args.model, base_env)

    examples, ex_stats = build_examples(rollouts, tok, get_procedure, max_len=args.max_len)
    print(f"[grpo-tau2] examples: {ex_stats}", flush=True)
    if not examples:
        print("[grpo-tau2] no tokenisable examples; exiting.", flush=True)
        (out_dir / "STATUS").write_text("no_examples\n")
        return 0

    from peft import LoraConfig, TaskType, get_peft_model
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none")
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[grpo-tau2] trainable {trainable:,}/{total:,} "
          f"({100*trainable/total:.2f}%)", flush=True)

    if is_dapo:
        print(f"[dapo] loss_type=token-level  e_low={args.clip_low}  "
              f"e_high={args.clip_high}  dynamic_sampling=True", flush=True)
    else:
        print(f"[grpo] loss_type=sequence-level  symmetric_clip=±{args.clip_low}",
              flush=True)

    n_steps = grpo_update(model, tok, examples, args)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"[grpo-tau2] saved LoRA to {out_dir}  (update steps={n_steps})", flush=True)

    json.dump({
        "algo": args.algo, "model": args.model, "rollout_model": args.rollout_model,
        "n_generations": args.n_generations, "epochs": args.epochs, "lr": args.lr,
        "beta": args.beta, "clip_low": args.clip_low,
        "clip_high": args.clip_high if is_dapo else args.clip_low,
        "dapo_dynamic_sampling": is_dapo, "split": args.split,
        "domain": os.environ.get("TAU2_DOMAIN"), "skillbook": args.skillbook,
        "update_steps": n_steps, **adv_stats, **ex_stats,
    }, open(out_dir / "grpo_info.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
