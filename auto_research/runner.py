#!/usr/bin/env python3
"""Auto-research experiment runner.

Receives a spec JSON path, dispatches to the right experiment kind. Designed
to be invoked by the orchestrator under a specific CUDA_VISIBLE_DEVICES.

Each runner writes:
  - <id>.result.json     : top-line metrics (read by orchestrator)
  - <id>.log             : full output stream (handled by orchestrator stdout/stderr redirect)

Kinds supported (v0):
  * grpo_continual           : k-step resume-adapter chain on chunked MBPP
  * grpo_curriculum_continual: same, but step k uses failures of step k-1
  * grpo_multi_seed_staircase: multi-seed of continual + fresh
  * forgetting_eval          : eval N adapters on a fixed prompt set
  * joint_cycle_multiseed    : router cycles with N seeds, average

This v0 deliberately wraps existing shell scripts where possible; future runs
will inline more logic.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
import pathlib

REPO = Path("/data0/home/zeyuwang/router-skills-evolve")
DATA_DIR = Path("/data0/home/zeyuwang/router-skills-evolve-data")
RUNS_ROOT = Path("/data0/home/zeyuwang/auto_research/runs")
LOG_ROOT = Path("/data0/home/zeyuwang/auto_research/logs")
RUNS_ROOT.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    ts = datetime.datetime.utcnow().isoformat()
    print(f"[{ts}] {msg}", flush=True)


def write_result(exp_id: str, payload: dict) -> None:
    path = LOG_ROOT / f"{exp_id}.result.json"
    path.write_text(json.dumps(payload, indent=2))


def shell(cmd: list[str], **kwargs) -> int:
    log("$ " + " ".join(cmd))
    return subprocess.call(cmd, **kwargs)


# ---------------------------------------------------------------------------
# Kind: grpo_continual
# ---------------------------------------------------------------------------


def run_grpo_continual(spec: dict, exp_id: str) -> dict:
    base = spec["base_model"]
    chunks = spec.get("chunks", [1, 2, 3, 4])
    n_gen = spec.get("n_generations", 4)
    kl = spec.get("kl_coef", 0.05)
    rm = spec.get("reward_mode", "partial")
    epp = spec.get("epochs_per_chunk", 1)
    eval_lim = spec.get("eval_limit", 200)
    lora_r = spec.get("lora_r", 16)
    lr = spec.get("lr", 5e-6)
    tag = spec.get("tag", "")

    out_root = RUNS_ROOT / exp_id
    out_root.mkdir(parents=True, exist_ok=True)

    prev = None
    step_results = []
    for k in chunks:
        chunk = DATA_DIR / "mbpp_aug" / "chunks" / f"chunk_{k}.jsonl"
        adapter = out_root / f"step_{k}"

        cmd = [
            "python3", str(REPO / "experiments/train_small_model_grpo_local.py"),
            "--data", str(chunk),
            "--base-model", base,
            "--output", str(adapter),
            "--limit", "100",
            "--epochs", str(epp),
            "--n-generations", str(n_gen),
            "--max-new-tokens", "192",
            "--temperature", "0.8",
            "--top-p", "0.95",
            "--lr", str(lr),
            "--lora-r", str(lora_r),
            "--prompt-style", "qwen-chat",
            "--reward-baseline", "group",
            "--reward-mode", rm,
            "--kl-coef", str(kl),
        ]
        if prev:
            cmd += ["--resume-from-adapter", str(prev)]
        rc = shell(cmd, cwd=str(REPO))
        if rc != 0:
            return {"status": "train_failed", "step": k}

        eval_out = out_root / f"step_{k}_eval{eval_lim}.json"
        rc = shell([
            "python3", str(REPO / "experiments/evaluate_finetuned_model.py"),
            "--data", str(DATA_DIR / "mbpp_aug" / "test_eval_all.jsonl"),
            "--base-model", base,
            "--adapter", str(adapter),
            "--output", str(eval_out),
            "--max-new-tokens", "384",
            "--prompt-style", "qwen-chat",
            "--limit", str(eval_lim),
        ], cwd=str(REPO))
        if rc != 0:
            return {"status": "eval_failed", "step": k}

        try:
            ed = json.loads(eval_out.read_text())
            pr = ed.get("metrics", {}).get("success_rate")
        except Exception:
            pr = None
        step_results.append({"step": k, "adapter": str(adapter), "eval_pass_rate": pr})

        prev = adapter

    return {
        "status": "ok",
        "kind": "grpo_continual",
        "base_model": base,
        "tag": tag,
        "steps": step_results,
        "final_pass_rate": step_results[-1]["eval_pass_rate"] if step_results else None,
    }


# ---------------------------------------------------------------------------
# Kind: forgetting_eval
# ---------------------------------------------------------------------------


def run_forgetting_eval(spec: dict, exp_id: str) -> dict:
    adapters = spec["adapters"]
    base = spec["base_model"]
    eval_data = spec["eval_dataset"]
    out_root = RUNS_ROOT / exp_id
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for ad in adapters:
        name = Path(ad).name
        out_path = out_root / f"forgetting_{name}.json"
        rc = shell([
            "python3", str(REPO / "experiments/evaluate_finetuned_model.py"),
            "--data", str(eval_data),
            "--base-model", base,
            "--adapter", str(ad),
            "--output", str(out_path),
            "--max-new-tokens", "384",
            "--prompt-style", "qwen-chat",
            "--limit", "100",
        ], cwd=str(REPO))
        if rc != 0:
            results.append({"adapter": ad, "status": "failed"})
            continue
        try:
            d = json.loads(out_path.read_text())
            pr = d.get("metrics", {}).get("success_rate")
        except Exception:
            pr = None
        results.append({"adapter": ad, "pass_rate": pr})
    return {"status": "ok", "kind": "forgetting_eval", "results": results}


# ---------------------------------------------------------------------------
# Curriculum continual (stub: just delegate to continual for v0)
# ---------------------------------------------------------------------------


def run_grpo_curriculum_continual(spec: dict, exp_id: str) -> dict:
    """v0: identical to grpo_continual; v1 will inject failure-task selection between steps."""
    # TODO: at step k, after training, eval on next-chunk and select failures.
    log("[v0] curriculum_continual falling back to grpo_continual semantics for now")
    spec2 = dict(spec)
    spec2["chunks"] = [1, 2, 3, 4]
    spec2["base_model"] = spec.get("base_model", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
    return run_grpo_continual(spec2, exp_id)


def run_grpo_multi_seed_staircase(spec: dict, exp_id: str) -> dict:
    """v0: run grpo_continual once with seed=43 (since the train script accepts --seed)."""
    log("[v0] multi-seed delegating to single continual run with --seed 43")
    spec2 = dict(spec)
    spec2["base_model"] = spec.get("base_model", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
    spec2["chunks"] = [1, 2, 3, 4]
    return run_grpo_continual(spec2, exp_id)


def run_joint_cycle_multiseed(spec: dict, exp_id: str) -> dict:
    """v0: invoke existing run_joint_cycle.sh (single seed). True multi-seed is v1."""
    rc = shell(["bash", "/data0/home/zeyuwang/router-skills-evolve-runs/run_joint_cycle.sh"])
    return {"status": "ok" if rc == 0 else "failed", "kind": "joint_cycle"}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------



def run_iterated_skill_llm_router(spec: dict, exp_id: str) -> dict:
    """Multi-round Skill -> LLM -> Router iteration (default 8 cycles)."""
    import os
    n_cycles = spec.get("n_cycles", 8)
    env = os.environ.copy()
    env["NUM_CYCLES"] = str(n_cycles)
    env["RUN_TAG"] = exp_id
    log(f"[iterated] dispatching run_iterated_skill_llm_router.sh with NUM_CYCLES={n_cycles}")
    rc = subprocess.call(
        ["bash", "/data0/home/zeyuwang/router-skills-evolve-runs/run_iterated_skill_llm_router.sh"],
        env=env,
    )
    res_dir = pathlib.Path(f"/data0/home/zeyuwang/router-skills-evolve-results/{exp_id}")
    cycles = []
    final_llm_pass = None
    for k in range(1, n_cycles + 1):
        item = {"cycle": k}
        llm_p = res_dir / f"cycle{k}_llm_eval.json"
        if llm_p.exists():
            try:
                item["llm_pass_rate"] = json.loads(llm_p.read_text()).get("metrics", {}).get("success_rate")
                final_llm_pass = item["llm_pass_rate"]
            except Exception:
                item["llm_pass_rate"] = None
        rt_p = res_dir / f"cycle{k}_router_eval.json"
        if rt_p.exists():
            try:
                m = json.loads(rt_p.read_text()).get("metrics", {})
                item["router_accuracy"] = m.get("accuracy")
                item["router_fallback"] = m.get("fallback_rate")
                item["router_cost_vs_large"] = m.get("learned_vs_always_large")
            except Exception:
                pass
        cycles.append(item)
    return {
        "status": "ok" if rc == 0 else "failed",
        "kind": "iterated_skill_llm_router",
        "n_cycles": n_cycles,
        "cycles": cycles,
        "final_llm_pass": final_llm_pass,
    }


HANDLERS = {
    "grpo_continual": run_grpo_continual,
    "grpo_curriculum_continual": run_grpo_curriculum_continual,
    "grpo_multi_seed_staircase": run_grpo_multi_seed_staircase,
    "forgetting_eval": run_forgetting_eval,
    "joint_cycle_multiseed": run_joint_cycle_multiseed,
    "iterated_skill_llm_router": run_iterated_skill_llm_router,
}


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: runner.py <spec.json>")
    spec_path = Path(sys.argv[1])
    exp = json.loads(spec_path.read_text())
    exp_id = exp["id"]
    kind = exp.get("kind")
    log(f"runner: exp_id={exp_id} kind={kind}")
    handler = HANDLERS.get(kind)
    if not handler:
        result = {"status": "unknown_kind", "kind": kind}
    else:
        try:
            result = handler(exp.get("spec", {}), exp_id)
        except Exception as e:
            result = {"status": "exception", "error": repr(e)}

    result["id"] = exp_id
    result["finished_at"] = datetime.datetime.utcnow().isoformat()
    out_path = LOG_ROOT / f"{exp_id}.result.json"
    out_path.write_text(json.dumps(result, indent=2))
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
