#!/usr/bin/env python3
"""End-to-end runner for the joint evolver experiment.

This script wires together the three tracks that were previously run as
separate commands:

1. SkillBook trace collection / stats update.
2. Learnable router training, evaluation, and threshold tuning.
3. Small-code-model SFT and evaluation.

The runner is intentionally command-oriented: each stage calls the existing
single-purpose experiment scripts and records a manifest. Use ``--dry-run`` to
print the exact plan without launching training or LLM calls.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional


ROOT = Path(__file__).resolve().parent.parent


@dataclass
class StageResult:
    name: str
    command: List[str]
    status: str
    started_at: float
    finished_at: float
    returncode: Optional[int] = None
    output: Optional[str] = None
    error: Optional[str] = None

    @property
    def duration_s(self) -> float:
        return self.finished_at - self.started_at

    def to_dict(self) -> dict:
        data = asdict(self)
        data["duration_s"] = self.duration_s
        data["command_text"] = " ".join(shlex.quote(part) for part in self.command)
        return data


@dataclass
class CycleResult:
    cycle: int
    stages: List[StageResult] = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "stages": [stage.to_dict() for stage in self.stages],
            "artifacts": self.artifacts,
        }


class Runner:
    def __init__(self, *, dry_run: bool, continue_on_error: bool, env: dict):
        self.dry_run = dry_run
        self.continue_on_error = continue_on_error
        self.env = env

    def run(self, name: str, command: List[str], output: Optional[Path] = None) -> StageResult:
        started = time.time()
        print("\n" + "=" * 80)
        print(f"STAGE: {name}")
        print("=" * 80)
        print("$ " + " ".join(shlex.quote(part) for part in command))

        if self.dry_run:
            finished = time.time()
            return StageResult(
                name=name,
                command=command,
                status="dry-run",
                started_at=started,
                finished_at=finished,
                output=str(output) if output else None,
            )

        try:
            subprocess.run(command, cwd=ROOT, env=self.env, check=True)
            status = "ok"
            returncode = 0
            error = None
        except subprocess.CalledProcessError as exc:
            status = "failed"
            returncode = exc.returncode
            error = str(exc)
            if not self.continue_on_error:
                finished = time.time()
                result = StageResult(
                    name=name,
                    command=command,
                    status=status,
                    started_at=started,
                    finished_at=finished,
                    returncode=returncode,
                    output=str(output) if output else None,
                    error=error,
                )
                print(json.dumps(result.to_dict(), indent=2))
                raise

        finished = time.time()
        return StageResult(
            name=name,
            command=command,
            status=status,
            started_at=started,
            finished_at=finished,
            returncode=returncode,
            output=str(output) if output else None,
            error=error,
        )


def py(*parts: str | Path) -> List[str]:
    return [sys.executable, *[str(part) for part in parts]]


def extend(cmd: List[str], flag: str, values: Iterable[str | Path]) -> None:
    values = [str(value) for value in values if str(value)]
    if values:
        cmd.append(flag)
        cmd.extend(values)


def maybe_extend(cmd: List[str], flag: str, value: Optional[str | Path]) -> None:
    if value is not None and str(value) != "":
        cmd.extend([flag, str(value)])


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def write_manifest(path: Path, args: argparse.Namespace, cycles: List[CycleResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args),
        "cycles": [cycle.to_dict() for cycle in cycles],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nManifest saved: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the joint skills/router/LLM evolver loop")

    parser.add_argument("--cycles", type=int, default=1, help="Number of end-to-end cycles")
    parser.add_argument("--workdir", default="runs/joint_evolver", help="Output working directory")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--proxy", default="", help="Optional HTTP(S) proxy, e.g. http://127.0.0.1:1080")

    parser.add_argument("--skip-skills", action="store_true", help="Skip live SkillBook trace collection")
    parser.add_argument("--skip-router", action="store_true", help="Skip learnable-router training/tuning")
    parser.add_argument("--skip-llm", action="store_true", help="Skip small LLM train/eval")

    parser.add_argument("--skills-n", type=int, default=30)
    parser.add_argument("--skills-rounds", type=int, default=3)
    parser.add_argument("--skills-start", type=int, default=0)

    parser.add_argument("--traces", nargs="+", default=["data/traces/*.jsonl"])
    parser.add_argument("--tasks", default="data/HumanEval.jsonl")
    parser.add_argument("--router-bench-files", nargs="+", default=["bench/data/all.jsonl"])
    parser.add_argument("--router-base-model", default="google/bert_uncased_L-2_H-128_A-2")
    parser.add_argument("--router-epochs", type=float, default=8)
    parser.add_argument("--router-batch-size", type=int, default=32)
    parser.add_argument("--router-max-fallback-rate", type=float, default=0.02)
    parser.add_argument("--router-min-accuracy", type=float, default=0.90)

    parser.add_argument("--mbpp-train-output", default="", help="Optional prebuilt MBPP train JSONL")
    parser.add_argument("--mbpp-eval-output", default="", help="Optional prebuilt MBPP eval JSONL")
    parser.add_argument("--mbpp-max-train", type=int, default=200)
    parser.add_argument("--mbpp-max-eval", type=int, default=100)
    parser.add_argument("--llm-base-model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--llm-train-data", default="", help="Optional prebuilt SFT JSONL")
    parser.add_argument("--llm-eval-data", default="", help="Optional prebuilt eval JSONL")
    parser.add_argument("--llm-eval-limit", type=int, default=20)
    parser.add_argument("--llm-epochs", type=int, default=3)
    parser.add_argument("--llm-batch-size", type=int, default=2)
    parser.add_argument("--llm-grad-accum", type=int, default=4)
    parser.add_argument("--llm-lora-r", type=int, default=16)
    parser.add_argument("--llm-lr", type=float, default=1e-4)
    parser.add_argument("--llm-max-seq-len", type=int, default=1024)
    parser.add_argument("--llm-max-new-tokens", type=int, default=384)
    parser.add_argument("--llm-use-4bit", action="store_true")

    return parser


def run_cycle(args: argparse.Namespace, cycle_idx: int, runner: Runner, workdir: Path) -> CycleResult:
    cycle_dir = workdir / f"cycle_{cycle_idx:02d}"
    data_dir = cycle_dir / "data"
    router_dir = cycle_dir / "router"
    llm_dir = cycle_dir / "llm"
    results_dir = cycle_dir / "results"
    for directory in [data_dir, router_dir, llm_dir, results_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    cycle = CycleResult(cycle=cycle_idx)

    if not args.skip_skills:
        cmd = py(
            "experiments/run_evolve.py",
            "--n",
            str(args.skills_n),
            "--rounds",
            str(args.skills_rounds),
            "--start",
            str(args.skills_start),
        )
        cycle.stages.append(runner.run("skills_trace_collection", cmd))

    router_data = data_dir / "uncommonroute_bench.jsonl"
    router_model = router_dir / "learned-router"
    router_eval = results_dir / "learned_router_eval.json"
    router_thresholds = results_dir / "learned_router_thresholds.json"

    if not args.skip_router:
        cmd = py(
            "experiments/import_uncommonroute_bench.py",
            "--output",
            router_data,
        )
        extend(cmd, "--files", args.router_bench_files)
        cycle.stages.append(runner.run("import_router_bench", cmd, router_data))

        cmd = py(
            "experiments/train_learnable_router.py",
            "--tasks",
            args.tasks,
            "--router-data",
            router_data,
            "--output",
            router_model,
            "--base-model",
            args.router_base_model,
            "--epochs",
            str(args.router_epochs),
            "--batch-size",
            str(args.router_batch_size),
            "--class-weight",
            "balanced",
        )
        extend(cmd, "--traces", args.traces)
        cycle.stages.append(runner.run("train_learnable_router", cmd, router_model))

        cmd = py(
            "experiments/evaluate_learnable_router.py",
            "--model",
            router_model,
            "--tasks",
            args.tasks,
            "--router-data",
            router_data,
            "--output",
            router_eval,
        )
        extend(cmd, "--traces", args.traces)
        cycle.stages.append(runner.run("evaluate_learnable_router", cmd, router_eval))

        cmd = py(
            "experiments/tune_learnable_router_threshold.py",
            "--model",
            router_model,
            "--tasks",
            args.tasks,
            "--router-data",
            router_data,
            "--max-fallback-rate",
            str(args.router_max_fallback_rate),
            "--min-accuracy",
            str(args.router_min_accuracy),
            "--output",
            router_thresholds,
        )
        extend(cmd, "--traces", args.traces)
        cycle.stages.append(runner.run("tune_router_threshold", cmd, router_thresholds))

    mbpp_train = Path(args.mbpp_train_output) if args.mbpp_train_output else data_dir / "mbpp_train.jsonl"
    mbpp_eval = Path(args.mbpp_eval_output) if args.mbpp_eval_output else data_dir / "mbpp_eval.jsonl"
    llm_train_data = Path(args.llm_train_data) if args.llm_train_data else mbpp_train
    llm_eval_data = Path(args.llm_eval_data) if args.llm_eval_data else mbpp_eval
    base_eval = results_dir / "llm_base_eval.json"
    adapter_eval = results_dir / "llm_adapter_eval.json"
    adapter_dir = llm_dir / "small_model_lora"

    if not args.skip_llm:
        if not args.llm_train_data or not args.llm_eval_data:
            cmd = py(
                "experiments/import_mbpp_training_data.py",
                "--train-output",
                mbpp_train,
                "--eval-output",
                mbpp_eval,
                "--max-train",
                str(args.mbpp_max_train),
                "--max-eval",
                str(args.mbpp_max_eval),
            )
            cycle.stages.append(runner.run("import_mbpp_data", cmd))

        cmd = py(
            "experiments/evaluate_finetuned_model.py",
            "--data",
            llm_eval_data,
            "--base-model",
            args.llm_base_model,
            "--output",
            base_eval,
            "--max-new-tokens",
            str(args.llm_max_new_tokens),
        )
        if args.llm_eval_limit:
            cmd.extend(["--limit", str(args.llm_eval_limit)])
        cycle.stages.append(runner.run("evaluate_base_llm", cmd, base_eval))

        cmd = py(
            "experiments/train_small_model.py",
            "--data",
            llm_train_data,
            "--base-model",
            args.llm_base_model,
            "--output",
            adapter_dir,
            "--epochs",
            str(args.llm_epochs),
            "--batch-size",
            str(args.llm_batch_size),
            "--grad-accum",
            str(args.llm_grad_accum),
            "--lora-r",
            str(args.llm_lora_r),
            "--lr",
            str(args.llm_lr),
            "--max-seq-len",
            str(args.llm_max_seq_len),
        )
        if args.llm_use_4bit:
            cmd.append("--use-4bit")
        cycle.stages.append(runner.run("train_small_llm", cmd, adapter_dir))

        cmd = py(
            "experiments/evaluate_finetuned_model.py",
            "--data",
            llm_eval_data,
            "--base-model",
            args.llm_base_model,
            "--adapter",
            adapter_dir,
            "--output",
            adapter_eval,
            "--max-new-tokens",
            str(args.llm_max_new_tokens),
        )
        if args.llm_eval_limit:
            cmd.extend(["--limit", str(args.llm_eval_limit)])
        cycle.stages.append(runner.run("evaluate_adapter_llm", cmd, adapter_eval))

    cycle.artifacts = {
        "router_data": str(router_data),
        "router_model": str(router_model),
        "router_eval": str(router_eval),
        "router_thresholds": str(router_thresholds),
        "mbpp_train": str(mbpp_train),
        "mbpp_eval": str(mbpp_eval),
        "llm_adapter": str(adapter_dir),
        "llm_base_eval": str(base_eval),
        "llm_adapter_eval": str(adapter_eval),
    }

    router_threshold_payload = read_json(router_thresholds)
    if router_threshold_payload:
        cycle.artifacts["router_recommended_threshold"] = router_threshold_payload.get("recommended")
    base_payload = read_json(base_eval)
    adapter_payload = read_json(adapter_eval)
    if base_payload:
        cycle.artifacts["llm_base_metrics"] = base_payload.get("metrics")
    if adapter_payload:
        cycle.artifacts["llm_adapter_metrics"] = adapter_payload.get("metrics")

    return cycle


def main() -> None:
    args = build_parser().parse_args()

    workdir = Path(args.workdir)
    if not workdir.is_absolute():
        workdir = ROOT / workdir
    workdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.proxy:
        env["http_proxy"] = args.proxy
        env["https_proxy"] = args.proxy
        env["HTTP_PROXY"] = args.proxy
        env["HTTPS_PROXY"] = args.proxy

    runner = Runner(dry_run=args.dry_run, continue_on_error=args.continue_on_error, env=env)
    cycles: List[CycleResult] = []
    manifest_path = workdir / "joint_evolver_manifest.json"

    for cycle_idx in range(args.cycles):
        print("\n" + "#" * 80)
        print(f"JOINT EVOLVER CYCLE {cycle_idx}")
        print("#" * 80)
        cycle = run_cycle(args, cycle_idx, runner, workdir)
        cycles.append(cycle)
        write_manifest(manifest_path, args, cycles)

    print("\nDone.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
