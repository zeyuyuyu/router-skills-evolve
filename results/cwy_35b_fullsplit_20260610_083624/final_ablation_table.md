# Final ablation table

Cycles: **4**  |  Bench: **tau2_bench**  |  Model: **08_qwen3_6_35b_a3b_273**  |  Schedule: **SLR**

## Final cycle (cycle 3)

| System variant | Routing Acc | Large F1 | Fallback | Cost vs Always-Large | Task Pass |
|---|---:|---:|---:|---:|---:|
| Base (always-small) | 76.97% | 0.00% | 23.03% | 10.00% | 76.97% |
| Always-large | 23.03% | 37.44% | 0.00% | 100.00% | 60.11% |
| + Skills evolve | 76.97% | 0.00% | 23.03% | 10.00% | 76.97% |
| + Router training | 79.78% | 68.42% | 1.12% | 46.91% | 69.66% |
| Full (+ LLM training) | 79.78% | 68.42% | 1.12% | 46.91% | 69.66% |

## Per-cycle progression (Full variant)

| Cycle | Routing Acc | Fallback | Task Pass | Cost vs Large |
|---:|---:|---:|---:|---:|
| 0 | 84.27% | 5.06% | 70.22% | 43.88% |
| 1 | 84.27% | 3.37% | 72.47% | 43.37% |
| 2 | 81.46% | 5.06% | 70.22% | 39.33% |
| 3 | 79.78% | 1.12% | 69.66% | 46.91% |

## Baseline reference (HumanEval × 1.5B, main branch 2026-05-09)

| System | Routing Acc | Task Pass |
|---|---:|---:|
| Base | 68.28%| 47%|
| + Skills | 69.46%| 47%|
| + Router | **93.04%**  | 47%|
| Full | **93.04%**  | **49%**  |
