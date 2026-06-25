# Standalone (single-track) ablation

Isolate each MERA track — **skills**, **LLM (SFT)**, **router** — from a *shared*
initial trace set, and measure its before/after effect on the held-out test split.
Motivation for joint optimization (see paper Method §"Why Joint Optimization").

All three reuse the same cycle-0 traces (collect once with `--schedule S`, then the
other tracks reuse via `SCALING_TRACE_RESUME=1`), and evaluate on the merged
HumanEval+MBPP **eval** split (582 tasks, `data/he_mbpp.jsonl`), same harness
(`HE_MAX_REPAIR_TURNS=3`).

## Scripts

```bash
# 1) Skills: base small  vs  base + distilled procedure
bash scripts/standalone_test/skills_test.sh results/<run>/cycle_0/skillbook.json [GPU]

# 2) LLM: base small  vs  SFT-trained small (no skills)
bash scripts/standalone_test/llm_test.sh results/<run>/cycle_0/llm_adapter [GPU]

# 3) Router: always-small  vs  learned-router routing (offline, on held-out traces)
bash scripts/standalone_test/router_test.sh results/<run>/cycle_0/router results/<run>/heldout_eval/traces.jsonl [GPU]
```

The pipeline runs that produced the inputs (each isolates one stage, shares traces):

```bash
TS=$(date +%m%d_%H%M)
# exp1 collects the shared traces + skills
HE_DATA=$PWD/data/he_mbpp.jsonl EXPERIMENT_CONFIG=humaneval_dapo_gpt SCALING_FORCE_BOTH=1 \
HE_USE_VLLM=1 HE_VLLM_SMALL_GPU=4 N_TASKS=600 SKILL_DISTILL_MAX_TOKENS=2000 \
EXPERIMENT_NAME=e2e_skills_$TS bash scripts/run_full_pipeline.sh --bench humaneval --n-cycles 1 --schedule S
# exp2 (L) and exp3 (R) reuse the traces:  SCALING_TRACE_RESUME=1 ... --schedule L  / --schedule R
```

## Result (held-out test split, 582 tasks; base SLM ≈ 34%)

| Track (alone) | Before | After | Δ | Cost vs large |
|---|---:|---:|---:|---:|
| Skills (procedure) | 34.2% | 38.3% | +4.1 | 10% |
| LLM (SFT) | 34.2% | 48.6% | +14.4 | 10% |
| Router (learned) | 33.7% | 79.6% | +45.9 | 67% |
| always-large (ref.) | — | 92.4% | — | 100% |

Router uses the Qwen2.5-Coder embedding featurizer (beats TF-IDF; `train_router_simple.py`
auto-compares and records both in `router_meta.json`). No single track alone reaches
large-model quality cheaply → motivates joint co-evolution.
