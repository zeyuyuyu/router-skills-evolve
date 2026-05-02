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

## Interpretation

- The end-to-end runner works and produces a manifest with all stage commands,
  artifact paths, and key metrics.
- The router track remains the positive signal.
- The stricter code prompt improves base-model evaluation substantially.
- The current LoRA SFT recipe still degrades adapter performance; next iterations
  should focus on DPO/GRPO or larger held-out SFT data with stronger output
  constraints.
