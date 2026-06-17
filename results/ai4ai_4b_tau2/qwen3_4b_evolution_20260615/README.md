# AI4AI Qwen3-4B Tau2 Run

This directory contains the AI4AI Qwen3-4B tau2 artifacts from 2026-06-17.

## Final Official Eval

Official tau2 CLI, test split, GPT-5.2 user simulator and GPT-5.2 NL judge:

```bash
tau2 run \
  --domain <airline|retail|telecom> \
  --agent-llm openai/qwen3-4b-sft \
  --user-llm openai/openai/gpt-5.2 \
  --task-split-name test \
  --num-trials 1 \
  --max-steps 200 \
  --seed 300
```

| model / run | airline | retail | telecom | total |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-4B base | 4/20 | 8/40 | 9/40 | 21/100 |
| old hard-trace SFT | 6/20 | 2/40 | 2/40 | 10/100 |
| fixed stage2-full SFT + parserfix | 6/20 | 14/40 | 15/40 | 35/100 |

Final result: `35/100` pass rate `0.350`.

Sanity checks for the final run:

| check | value |
| --- | ---: |
| assistant tool calls | 1329 |
| assistant `<user>` tag leaks | 0 |
| unexpected tool kwarg errors | 0 |

## Data Summary

Training trace collection used train split `100` tasks:

| artifact | rows |
| --- | ---: |
| raw train traces | 100 |
| purified train traces | 72 |
| hard SFT pairs | 52 |

Purification removed 28 rows where the GPT-5.2 teacher failed or was missing.

## Where To Look

- Final eval summary: `eval_stage2_full_parserfix/stage2_full_parserfix_summary.md`
- Final eval JSON summary: `eval_stage2_full_parserfix/stage2_full_parserfix_summary.json`
- Raw official tau2 eval outputs: `eval_stage2_full_parserfix/*.results.json`
- Train traces: `train_traces/`
- SFT data: `sft/`
- SkillBook: `skillbook/`
- Router: `router/`
- Training metadata/logs: `train/qwen3_4b_stage2_full_1epoch/`

The 30GB checkpoint weights are intentionally not committed to Git. The
checkpoint path used for this eval was:

```text
/root/router-skills-evolve/results/ai4ai_4b_tau2/qwen3_4b_evolution_20260615/train/qwen3_4b_stage2_full_1epoch/checkpoint-final
```

## Reproduce

Use:

```bash
COMMONSTACK_API_KEY=... bash scripts/ai4ai_qwen3_4b/run_stage2_full_official_eval.sh
```

The script serves the checkpoint on 8 GPUs, runs official tau2 CLI test100, and
writes a fresh summary back into this directory.
