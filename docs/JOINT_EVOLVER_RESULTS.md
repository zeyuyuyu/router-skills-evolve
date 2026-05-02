# Joint Evolver A800 Results

This report summarizes the first real end-to-end joint evolver runs on the
8xA800 host. Full model checkpoints and detailed manifests are kept on the A800
data disk; this repo stores the small summary only.

## Run 1: full joint pipeline

Manifest on A800:

```text
/data0/home/zeyuwang/router-skills-evolve-runs/joint_evolver_real_20260502_050032/joint_evolver_manifest.json
```

Pipeline stages:

1. Import UncommonRoute bench router data.
2. Train learnable router.
3. Evaluate router and tune threshold.
4. Import MBPP train/eval data.
5. Evaluate base small LLM.
6. Train LoRA adapter.
7. Evaluate adapter.

Key metrics:

| Track | Metric | Value |
| --- | --- | --- |
| Router | recommended threshold | 0.53 |
| Router | accuracy | 92.36% |
| Router | large F1 | 88.63% |
| Router | fallback rate | 1.98% |
| Router | cost vs always-large | 38.57% |
| LLM base | MBPP eval20 pass rate | 3/20 = 15% |
| LLM LoRA | MBPP eval20 pass rate | 2/20 = 10% |

## Run 2: code-style LLM prompt

Manifest on A800:

```text
/data0/home/zeyuwang/router-skills-evolve-runs/joint_evolver_real_codeprompt_20260502_051749/joint_evolver_manifest.json
```

This run skipped the router track and tested a stricter code-generation prompt
for the LLM track.

| Track | Metric | Value |
| --- | --- | --- |
| LLM base | MBPP eval20 pass rate | 7/20 = 35% |
| LLM LoRA | MBPP eval20 pass rate | 1/20 = 5% |

## Run 3: Qwen chat-style LLM prompt

Manifest on A800:

```text
/data0/home/zeyuwang/router-skills-evolve-runs/joint_evolver_real_qwenchat_20260502_084947/joint_evolver_manifest.json
```

This run used Qwen Instruct chat tokens for both SFT and evaluation:
`--llm-prompt-style qwen-chat`.

| Track | Metric | Value |
| --- | --- | --- |
| LLM base | MBPP eval20 pass rate | 10/20 = 50% |
| LLM LoRA, 3 epochs | MBPP eval20 pass rate | 9/20 = 45% |
| LLM LoRA, conservative 1 epoch | MBPP eval20 pass rate | 9/20 = 45% |

## Interpretation

- The end-to-end runner works and produces a manifest with all stage commands,
  artifact paths, and key metrics.
- The router track remains the positive signal.
- Prompt formatting matters: code-style prompting improved base-model evaluation
  from 15% to 35%, and Qwen chat-style prompting improved it to 50%.
- Qwen chat-style SFT avoids the catastrophic syntax-error regression seen with
  earlier prompts, but the current LoRA adapter still does not beat the base
  model on the held-out MBPP slice.
- Next iterations should focus on DPO/GRPO or better data filtering/augmentation
  rather than promoting the current SFT adapter.
