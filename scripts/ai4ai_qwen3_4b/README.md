# AI4AI Qwen3-4B Tau2 Utilities

This folder contains the small serving/eval utilities used for the AI4AI
Qwen3-4B tau2 result.

Run the final official eval:

```bash
COMMONSTACK_API_KEY=... bash scripts/ai4ai_qwen3_4b/run_stage2_full_official_eval.sh
```

Modes:

```bash
MODE=smoke bash scripts/ai4ai_qwen3_4b/run_stage2_full_official_eval.sh
MODE=summarize bash scripts/ai4ai_qwen3_4b/run_stage2_full_official_eval.sh
```

Outputs land in:

```text
results/ai4ai_4b_tau2/qwen3_4b_evolution_20260615/eval_stage2_full_parserfix/
```

The eval script expects the trained checkpoint on the remote GPU node at:

```text
/root/router-skills-evolve/results/ai4ai_4b_tau2/qwen3_4b_evolution_20260615/train/qwen3_4b_stage2_full_1epoch/checkpoint-final
```

Override remote/checkpoint settings with environment variables:

```bash
REMOTE_HOST=...
REMOTE_PORT=...
REMOTE_REPO=...
SFT_CHECKPOINT=...
```
