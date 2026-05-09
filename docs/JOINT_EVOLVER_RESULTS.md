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

## Run 4: larger-model GRPO on A800

This run moved beyond the 0.5B smoke setup and used a larger code model:

```text
Qwen/Qwen2.5-Coder-1.5B-Instruct
```

Training:

```text
local GRPO / REINFORCE
200 MBPP augmented train tasks
4 rollouts per prompt
1 epoch
LoRA rank 16
```

Held-out evaluation:

```text
MBPP test_eval_all first 100
```

| Track | Metric | Value |
| --- | --- | --- |
| LLM base 1.5B | MBPP eval100 pass rate | 47/100 = 47% |
| LLM GRPO 1.5B | MBPP eval100 pass rate | **49/100 = 49%** |

Artifact paths:

```text
/data0/home/zeyuwang/router-skills-evolve-results/rl_15b/qwen25_coder_15b_base_eval100.json
/data0/home/zeyuwang/router-skills-evolve-results/rl_15b/qwen25_coder_15b_grpo_eval100.json
/data0/home/zeyuwang/router-skills-evolve-runs/rl_15b/qwen25_coder_15b_grpo_200x4
```

### Additional scaling attempts

We also tried increasing training budget or model size. These runs did not
improve over their own base models:

| Model | Method | Base | Adapter | Result |
| --- | --- | ---: | ---: | --- |
| Qwen2.5-Coder-1.5B-Instruct | GRPO 400 tasks x 4 rollouts | 47/100 | 46/100 | degraded |
| Qwen2.5-Coder-3B-Instruct | GRPO 200 tasks x 4 rollouts | 62/100 | 60/100 | degraded |
| Qwen2.5-Coder-3B-Instruct | conservative SFT 400 tasks | 62/100 | 62/100 | tied |
| Qwen2.5-Coder-7B-Instruct | GRPO 100 tasks x 4 rollouts | 77/100 | 75/100 | degraded |

The current best positive LLM-training case is therefore the 1.5B GRPO 200x4
run. Scaling model size or training budget alone is not sufficient; the next
step should tune the RL objective, add KL/reference stabilization, and improve
reward/data selection.

## Interpretation

- The end-to-end runner works and produces a manifest with all stage commands,
  artifact paths, and key metrics.
- The router track remains the positive signal.
- Prompt formatting matters: code-style prompting improved base-model evaluation
  from 15% to 35%, and Qwen chat-style prompting improved it to 50%.
- Qwen chat-style SFT avoids the catastrophic syntax-error regression seen with
  earlier prompts, but the current LoRA adapter still does not beat the base
  model on the held-out MBPP slice.
- Larger-model GRPO gives a positive model-evolution signal: 1.5B GRPO improves
  from 47/100 to 49/100 on a 100-example held-out slice.
- Next iterations should focus on making the GRPO/DPO margin larger with more
  rollouts, KL/reference stabilization, and larger held-out evaluations.
