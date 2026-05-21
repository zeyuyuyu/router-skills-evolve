# Long-horizon iterated Skill → LLM → Router pipeline (8 cycles)

Single 8-cycle execution of the `skill_first` ordering, completed
2026-05-20 21:13 UTC on a single A800 80 GB GPU.

Each cycle k = 1..8 ran, in strict order:
1. **Skill update** — SkillBook online frequency update from a 15-trace batch
   (batches loop modulo the trace pool).
2. **LLM training + eval** — GRPO continual training on MBPP chunk
   `((k - 1) mod 4) + 1`, resuming the previous cycle's LoRA adapter. Then
   evaluated on the first 200 tasks of `mbpp_aug/test_eval_all.jsonl`.
3. **Router training + eval** — BERT-tiny binary classifier trained from
   scratch on the cumulative UncommonRoute bench slice (seed = 42 + k for
   natural seed diversity), then evaluated on the 832-task fixed held-out
   bench split.

State persistence:

- `SkillBook` — accumulated across all 8 cycles.
- LLM `LoRA` adapter — chained, each cycle resumed from the previous.
- Router — fresh init each cycle (seed = 42 + k).

## Files

| File | Description |
| --- | --- |
| `cycle<k>_llm_eval.json` | MBPP eval200 results for the LLM adapter at end of cycle k |
| `cycle<k>_router_eval.json` | 832-task held-out evaluation of the router at end of cycle k |
| `cycle_timings.jsonl` | Per-cycle wall time (seconds) |
| `iterated.log` | Full stdout/stderr stream from the entire run |

## Headline numbers

| Cycle | Router Acc | Fallback | Cost vs Large | LLM Pass |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 57.6% | 0.4% | 85.6% | 47.0% |
| 2 | 71.1% | 4.3% | 68.6% | 47.0% |
| 3 | **87.8%** | 3.4% | 52.8% | 47.0% |
| 4 | 82.3% | 3.5% | 57.4% | 47.0% |
| 5 | 76.7% | 14.4% | 54.2% | 47.0% |
| 6 | 85.3% | 3.1% | 56.0% | 46.5% |
| 7 | 82.7% | 4.6% | 56.6% | 47.0% |
| 8 | 85.0% | 4.0% | 54.3% | **47.5%** |

Total wall time: 3.4 h (mean 25.7 min / cycle, range 24:11 – 26:49).
SkillBook end state: 34 signatures, 120 observations.

## Repro

```bash
NUM_CYCLES=8 CUDA_VISIBLE_DEVICES=0 \
  bash /data0/home/zeyuwang/router-skills-evolve-runs/run_iterated_skill_llm_router.sh
```

Discussion of these results lives in
[docs/E2E_ABLATION_RESULTS.md](../../docs/E2E_ABLATION_RESULTS.md) under
the "Long-horizon iterated pipeline" section.
