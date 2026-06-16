# Final ablation table

Cycles: **2**  |  Bench: **humaneval**  |  Model: **05_qwen3_5_4b_273**  |  Schedule: **SLR**

## Final cycle (cycle 1)

| System variant | Routing Acc | Large F1 | Fallback | Cost vs Always-Large | Task Pass |
|---|---:|---:|---:|---:|---:|
| Base (always-small) | 65.85% | 0.00% | 34.15% | 10.00% | 65.85% |
| Always-large | 34.15% | 50.91% | 0.00% | 100.00% | 75.61% |
| + Skills evolve | 70.73% | 47.83% | 20.73% | 29.76% | 76.83% |
| + Router training | 87.80% | 81.48% | 7.32% | 38.54% | 78.05% |
| Full (+ LLM training) | 87.80% | 81.48% | 7.32% | 38.54% | 78.05% |

## Per-cycle progression (Full variant)

| Cycle | Routing Acc | Fallback | Task Pass | Cost vs Large |
|---:|---:|---:|---:|---:|
| 0 | 87.80% | 7.32% | 73.17% | 38.54% |
| 1 | 87.80% | 7.32% | 78.05% | 38.54% |

## Baseline reference (HumanEval × 1.5B, main branch 2026-05-09)

| System | Routing Acc | Task Pass |
|---|---:|---:|
| Base | 68.28%| 47%|
| + Skills | 69.46%| 47%|
| + Router | **93.04%**  | 47%|
| Full | **93.04%**  | **49%**  |
