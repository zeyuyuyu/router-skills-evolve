# End-to-End Ablation Results

> ⚠️ **历史报告（2026-05-09，旧设计）**。下面的数字来自**多 cluster signature +
> Skills 参与路由 + 独立 Base 臂**的版本。当前代码已改为**单一全局 skill + Router 独占路由**，
> ablation 四臂变为 `large / skills(全 small+procedure) / router / full`，
> `skills` 臂取代了原来的独立 `Base`。下表保留作历史记录；新设计的数字需重跑实验后更新。

This report summarizes the first unified end-to-end ablation run for the
evolver project. The goal is to avoid reporting isolated "skills evolve",
"router training", and "LLM training" numbers, and instead compare the system
variants under one evaluation frame.

## Setup

Host: A800  
Run date: 2026-05-09  
Evaluation examples: 848 router-labelled prompts

Inputs:

- real traces from `data/traces/*.jsonl`
- UncommonRoute router weak labels
- learned router checkpoint from the mixed-router run
- Qwen-chat LLM base/adapter evaluation from A800

Full artifacts on A800:

```text
/data0/home/zeyuwang/router-skills-evolve-results/e2e_ablation/e2e_ablation_router_sft_qwenchat.json
/data0/home/zeyuwang/router-skills-evolve-results/e2e_ablation/e2e_ablation_router_sft_qwenchat.md
```

## Routing ablation

| Variant | Accuracy | Large F1 | Fallback Rate | Cost vs Always-Large | Predicted Large |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base: always-small + fallback | 68.28% | 0.00% | 31.72% | 33.54% | 0 |
| + SkillBook signature routing | 69.46% | 24.93% | 26.65% | 37.27% | 76 |
| + Learned router | **93.04%** | **89.48%** | **2.12%** | 37.75% | 292 |

Interpretation:

- Base is cheap but falls back on all hard examples.
- SkillBook alone helps slightly, but hard-route recall is still weak.
- Learned router is the strongest positive ablation: it sharply reduces
  fallback while preserving low cost.

## Component ablation table

This is the paper-facing version of the ablation. It asks what happens when we
remove each evolver component from the current end-to-end system. Routing
metrics are evaluated on the 848-example router split. LLM metrics use the
larger 1.5B code-model run on MBPP eval100 when available.

| System variant | Skills evolve | Router training | LLM training | Routing accuracy | Fallback rate | Cost vs always-large | Code pass rate |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Base: always-small + fallback | ✗ | ✗ | ✗ | 68.28% | 31.72% | **33.54%** | 47/100 = 47% |
| + Skills evolve | ✓ | ✗ | ✗ | 69.46% | 26.65% | 37.27% | 47/100 = 47% |
| + Router training | ✓ | ✓ | ✗ | **93.04%** | 2.12% | 37.75% | 47/100 = 47% |
| Full: + Router + LLM GRPO | ✓ | ✓ | ✓ | **93.04%** | 2.12% | 37.75% | **49/100 = 49%** |

Read this as an offline end-to-end proxy:

- **Removing router training** causes the largest drop: fallback rises from
  2.12% to 26.65% and large-class F1 drops sharply.
- **Removing skills evolve** leaves only the base fallback policy in this table;
  SkillBook gives a small but visible gain over base.
- **Removing LLM training** keeps routing unchanged but code pass rate drops
  from 49% to 47% in the 1.5B GRPO experiment.
- **Full system** is currently best on quality, while base is cheapest because
  it delays large-model calls until fallback. The cost/quality trade-off should
  be reported jointly.

## LLM training ablation

| Variant | Eval | Pass Rate |
| --- | --- | ---: |
| Qwen2.5-Coder-0.5B-Instruct base, qwen-chat prompt | MBPP eval20 | **10/20 = 50%** |
| SFT LoRA adapter, qwen-chat prompt | MBPP eval20 | 9/20 = 45% |
| Local GRPO LoRA smoke run | MBPP eval20 on 8x4090 | 9/20 = 45% |
| Qwen2.5-Coder-1.5B-Instruct base | MBPP eval100 | 47/100 = 47% |
| Qwen2.5-Coder-1.5B-Instruct + local GRPO | MBPP eval100 | **49/100 = 49%** |

Interpretation:

- The qwen-chat prompt substantially improves generation format.
- Naive SFT is still below the base model.
- Local GRPO has shown a small positive signal in a separate 8x4090 smoke run
  when compared against that run's measured base (9/20 vs 8/20), but it is not
  yet a strong model-evolution result.

## Current conclusion

The clean end-to-end ablation currently supports the following story:

1. **Router evolution works.** Learned routing gives the clearest improvement
   in cost-quality behavior.
2. **SkillBook is useful but insufficient alone.** It gives transparent online
   memory, but weak recall on large-route examples.
3. **LLM evolution needs stronger recipes.** SFT does not yet beat the base
   model; local GRPO is promising but still small-scale.

For a stronger EMNLP-style result, the next experiments should use the unified
Evolver Dataset splits and report:

- router ablations on router `test`;
- code-model SFT + GRPO/DAPO on code `test` (DPO dropped);
- end-to-end pass rate, fallback rate, cost, and small/large usage under the
  same evaluation set, reported as the four-arm `large / skills / router / full`.

