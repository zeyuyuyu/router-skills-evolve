# Final ablation table

Cycles: **4**  |  Bench: **humaneval**  |  Model: **05_qwen3_5_4b_273**  |  Schedule: **SLR**

Results are split into the **training split** (used to evolve skills/router/adapter; routing is evaluated in-sample so it is optimistic) and the **held-out test split** (the honest generalization estimate).

## Training split — final cycle (cycle 3)

| System variant | Routing Acc | Large F1 | Fallback | Cost vs Always-Large | Task Pass |
|---|---:|---:|---:|---:|---:|
| Always-large | 24.39% | 39.22% | 0.00% | 100.00% | 96.34% |
| Small + Skills (always-small + procedure) | 75.61% | 0.00% | 24.39% | 10.00% | 75.61% |
| + Router training | 92.68% | 83.33% | 6.10% | 27.56% | 92.68% |
| Full (+ LLM training) | 92.68% | 83.33% | 6.10% | 27.56% | 92.68% |

## Training split — per-cycle progression (Full variant)

| Cycle | Routing Acc | Fallback | Task Pass | Cost vs Large |
|---:|---:|---:|---:|---:|
| 0 | 92.68% | 6.10% | 91.46% | 31.95% |
| 1 | 90.24% | 4.88% | 92.68% | 40.73% |
| 2 | 93.90% | 6.10% | 91.46% | 28.66% |
| 3 | 92.68% | 6.10% | 92.68% | 27.56% |

## Held-out TEST split — final cycle (router trained on train, evaluated here)

| System variant | Routing Acc | Large F1 | Fallback | Cost vs Always-Large | Task Pass |
|---|---:|---:|---:|---:|---:|
| Always-large | 37.80% | 54.87% | 0.00% | 100.00% | 96.34% |
| Small + Skills (always-small + procedure) | 62.20% | 0.00% | 37.80% | 10.00% | 62.20% |
| + Router training | 60.98% | 5.88% | 36.59% | 13.29% | 63.41% |
| Full (+ LLM training) | 60.98% | 5.88% | 36.59% | 13.29% | 63.41% |

## Baseline reference (HumanEval × 1.5B, main branch 2026-05-09)

| System | Routing Acc | Task Pass |
|---|---:|---:|
| Base | 68.28%| 47%|
| + Skills | 69.46%| 47%|
| + Router | **93.04%**  | 47%|
| Full | **93.04%**  | **49%**  |
