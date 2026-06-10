# Final ablation table

Cycles: **4**  |  Bench: **tau2_bench**  |  Model: **08_qwen3_6_35b_a3b_273**  |  Schedule: **SLR**

## Final cycle (cycle 3)

| System variant | Routing Acc | Large F1 | Fallback | Cost vs Always-Large | Task Pass |
|---|---:|---:|---:|---:|---:|
| Base (always-small + fallback) | 68.92% | 0.00% | 31.08% | 10.00% | 68.92% |
| + Skills evolve | 68.92% | 0.00% | 31.08% | 10.00% | 68.92% |
| + Router training | 89.19% | 81.82% | 6.76% | 35.54% | 72.97% |
| Full (+ LLM training) | 89.19% | 81.82% | 6.76% | 35.54% | 72.97% |

## Per-cycle progression (Full variant)

| Cycle | Routing Acc | Fallback | Task Pass | Cost vs Large |
|---:|---:|---:|---:|---:|
| 0 | 93.24% | 4.05% | 78.38% | 47.70% |
| 1 | 93.24% | 4.05% | 89.19% | 22.16% |
| 2 | 82.43% | 5.41% | 70.27% | 48.92% |
| 3 | 89.19% | 6.76% | 72.97% | 35.54% |

## Baseline reference (HumanEval × 1.5B, main branch 2026-05-09)

| System | Routing Acc | Task Pass |
|---|---:|---:|
| Base | 68.28%| 47%|
| + Skills | 69.46%| 47%|
| + Router | **93.04%**  | 47%|
| Full | **93.04%**  | **49%**  |
