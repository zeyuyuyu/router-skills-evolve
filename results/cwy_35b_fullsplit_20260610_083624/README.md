# tau2 Full Train/Test Split 35B Joint Evolution

This directory contains sanitized aggregate results for the tau2 full train/test split 35B joint evolution run.

Raw traces, logs, model weights, prompts, task text, and environment-specific runtime details are intentionally omitted.

## Setup

- Bench: `tau2_bench`
- Domains: `retail`, `telecom`, `airline`
- Train split: 178 tasks
- Held-out test/eval split: 100 tasks
- Train/test overlap: 0 tasks
- Schedule: `SLR`
- SFT epochs: 2 per cycle
- Cycles completed: 0, 1, 2, 3

## Held-out Test Result

Held-out test uses the full eval/test split: 100 tasks, with 40 retail, 40 telecom, and 20 airline tasks.

| Variant | Routing Acc | Large F1 | Fallback | Cost vs Always-Large | Task Pass |
|---|---:|---:|---:|---:|---:|
| Base (always-small) | 80.00% | 0.00% | 20.00% | 10.00% | 80.00% |
| Always-large | 20.00% | 33.33% | 0.00% | 100.00% | 71.00% |
| + Skills evolve | 80.00% | 0.00% | 20.00% | 10.00% | 80.00% |
| + Router training | 66.00% | 43.33% | 7.00% | 46.00% | 71.00% |
| Full (+ LLM training) | 66.00% | 43.33% | 7.00% | 46.00% | 71.00% |

## Included Files

- `final_ablation_table.md`: aggregate training-cycle ablation table.
- `cycle_*/e2e_ablation_summary.{json,md}`: per-cycle aggregate E2E ablations on the train split traces.
- `heldout_eval/e2e_ablation_summary.{json,md}`: aggregate E2E ablation on the held-out test split.
- `audit_summary.json`: sanitized completion audit with split, trace, SFT, and result checks.
