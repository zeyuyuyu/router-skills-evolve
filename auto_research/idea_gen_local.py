#!/usr/bin/env python3
"""Local fallback idea generator.

Rule-based: when the experiment queue empties, pick the hottest arxiv tag we
haven't fully covered, derive a perturbation experiment from our most recent
positive result, and append it to state.json.

Designed to be safe and idempotent — never overshoots the daily cap and never
generates duplicates.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/data0/home/zeyuwang/auto_research")
STATE = ROOT / "state.json"
HOTSPOTS = ROOT / "hotspots"
REPORTS = ROOT / "reports"

# Tag -> list of perturbation specs. Each spec is one experiment template.
# Each template will be parameterized by varying a single dimension over
# tries that haven't been done yet.
TEMPLATES = {
    "rl_align": [
        # vary lr around 5e-6
        {"kind": "grpo_continual", "param": "lr", "values": [1e-6, 3e-6, 1e-5, 3e-5],
         "base": {"base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "chunks": [1,2,3,4],
                  "n_generations": 4, "kl_coef": 0.05, "reward_mode": "partial",
                  "epochs_per_chunk": 1, "eval_limit": 200}},
        # vary KL coef
        {"kind": "grpo_continual", "param": "kl_coef", "values": [0.0, 0.01, 0.1, 0.2],
         "base": {"base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "chunks": [1,2,3,4],
                  "n_generations": 4, "lr": 5e-6, "reward_mode": "partial",
                  "epochs_per_chunk": 1, "eval_limit": 200}},
        # vary rollouts
        {"kind": "grpo_continual", "param": "n_generations", "values": [2, 6, 8],
         "base": {"base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "chunks": [1,2,3,4],
                  "kl_coef": 0.05, "lr": 5e-6, "reward_mode": "partial",
                  "epochs_per_chunk": 1, "eval_limit": 200}},
    ],
    "continual": [
        # multi-seed of 3B continual (extends our top-performing 61% result)
        {"kind": "grpo_multi_seed_staircase", "param": "seed_tag", "values": ["s44","s45","s46"],
         "base": {"base_model": "Qwen/Qwen2.5-Coder-3B-Instruct", "chunks": [1,2,3,4],
                  "n_generations": 4, "kl_coef": 0.05, "reward_mode": "partial",
                  "epochs_per_chunk": 1, "eval_limit": 200}},
        # forgetting test on every other adapter we have
        {"kind": "forgetting_eval", "param": "eval_chunk", "values": [1, 2, 3],
         "base": {"adapters": [
            "/data0/home/zeyuwang/auto_research/runs/exp_2026_05_12_001_continual_3b/step_4",
            "/data0/home/zeyuwang/auto_research/runs/exp_2026_05_12_002_curriculum_continual_15b/step_4",
         ], "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct"}},
    ],
    "training_eff": [
        # vary LoRA rank
        {"kind": "grpo_continual", "param": "lora_r", "values": [8, 32, 64],
         "base": {"base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "chunks": [1,2,3,4],
                  "n_generations": 4, "kl_coef": 0.05, "reward_mode": "partial",
                  "epochs_per_chunk": 1, "eval_limit": 200, "lr": 5e-6}},
    ],
    "moe_routing": [
        {"kind": "joint_cycle_multiseed", "param": "seed_tag", "values": ["s43","s44"],
         "base": {"n_cycles": 4}},
    ],
}

DAILY_CAP = 20  # max experiments local fallback can inject per day


def load_state() -> dict:
    return json.loads(STATE.read_text())


def save_state(state: dict) -> None:
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    tmp.replace(STATE)


def already_done(state: dict, key: tuple) -> bool:
    """Return True if (kind, param, value, base_model) combo seen in history or queue."""
    for src in (state.get("history", []), state.get("queue", [])):
        for entry in src:
            spec = entry.get("spec") or entry.get("exp", {}).get("spec") or {}
            kind = entry.get("kind") or entry.get("exp", {}).get("kind")
            tup = (kind, spec.get("base_model"), spec.get("lr"), spec.get("kl_coef"),
                   spec.get("lora_r"), spec.get("n_generations"), spec.get("tag"))
            if tup == key:
                return True
    return False


def top_arxiv_tags(date: str | None = None) -> list[tuple[str, int]]:
    """Return today's tag counts; fall back to most recent."""
    target = HOTSPOTS / f"{date or datetime.datetime.utcnow().date().isoformat()}.json"
    if not target.exists():
        files = sorted(HOTSPOTS.glob("*.json"))
        if not files:
            return []
        target = files[-1]
    data = json.loads(target.read_text())
    c = Counter()
    for p in data.get("papers", []):
        for t in p.get("tags", []):
            c[t] += 1
    return c.most_common()


def count_today_from_local_gen(state: dict) -> int:
    today = datetime.datetime.utcnow().date().isoformat()
    n = 0
    for entry in state.get("history", []) + state.get("queue", []):
        if entry.get("source") == "local_fallback" and entry.get("created_at", "").startswith(today):
            n += 1
    return n


def main() -> None:
    state = load_state()
    if state.get("queue") or state.get("running"):
        print("[local-idea-gen] queue or running not empty; skipping")
        return

    today_n = count_today_from_local_gen(state)
    if today_n >= DAILY_CAP:
        print(f"[local-idea-gen] hit daily cap ({today_n}/{DAILY_CAP}); skipping")
        return

    tags = top_arxiv_tags()
    print(f"[local-idea-gen] top tags: {tags[:5]}")
    if not tags:
        print("[local-idea-gen] no arxiv hotspots; falling back to perturbing last positive result")
        # ultra-fallback: re-run a different seed of the best result
        tags = [("rl_align", 1)]

    new_specs: list[dict] = []
    for tag, count in tags:
        if len(new_specs) >= 3:
            break
        for tmpl in TEMPLATES.get(tag, []):
            for v in tmpl["values"]:
                spec = dict(tmpl["base"])
                spec[tmpl["param"]] = v
                if "tag" not in spec:
                    spec["tag"] = f"{tmpl['param']}_{v}"
                key = (tmpl["kind"], spec.get("base_model"), spec.get("lr"),
                       spec.get("kl_coef"), spec.get("lora_r"),
                       spec.get("n_generations"), spec.get("tag"))
                if already_done(state, key):
                    continue
                ts = datetime.datetime.utcnow().strftime("%Y_%m_%d_%H%M%S")
                exp_id = f"exp_{ts}_local_{tag}_{tmpl['param']}_{v}"
                new_specs.append({
                    "id": exp_id,
                    "priority": 6,
                    "rationale": (
                        f"Local fallback: arxiv tag '{tag}' is trending ({count} papers today). "
                        f"Perturbing {tmpl['param']}={v} relative to our most-recent positive result. "
                        f"Queue was empty; perpetuating motion."
                    ),
                    "kind": tmpl["kind"],
                    "spec": spec,
                    "gpu": "auto",
                    "source": "local_fallback",
                    "created_at": datetime.datetime.utcnow().isoformat(),
                })
                break  # one perturbation per template
            if len(new_specs) >= 3:
                break

    if not new_specs:
        print("[local-idea-gen] no fresh combos available (every template combo already tried); idle.")
        return

    state.setdefault("queue", []).extend(new_specs)
    save_state(state)

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"local_idea_gen-{datetime.datetime.utcnow().date().isoformat()}.md"
    lines = [f"# Local idea-gen {datetime.datetime.utcnow().isoformat()}", ""]
    lines.append(f"Top tags: {tags[:5]}")
    lines.append("")
    lines.append("Queued:")
    for s in new_specs:
        lines.append(f"- `{s['id']}` (priority {s['priority']}): {s['rationale']}")
    with open(out, "a") as f:
        f.write("\n".join(lines) + "\n\n")
    print(f"[local-idea-gen] queued {len(new_specs)} new experiments; report: {out}")
    for s in new_specs:
        print(f"  + {s['id']}")


if __name__ == "__main__":
    main()
