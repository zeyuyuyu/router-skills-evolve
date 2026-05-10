# LLM Training Results

This document summarizes the LLM model-evolution experiments run so far. The
main takeaway is that naive SFT is not enough, while local GRPO gives the first
positive signal on a larger held-out evaluation.

## Summary table

| Model | Training paradigm | Configuration | Eval | Base | Trained | Result |
| --- | --- | --- | --- | ---: | ---: | --- |
| Qwen2.5-Coder-0.5B-Instruct | SFT | 6 hard examples, LoRA r=16, 8 epochs | HumanEval hard 6 | 4/6 | 2/6 | degraded |
| Qwen2.5-Coder-0.5B-Instruct | SFT | MBPP train120, LoRA r=16, 3 epochs | MBPP eval20 | 3/20 | 2/20 | degraded |
| Qwen2.5-Coder-0.5B-Instruct | SFT + qwen-chat prompt | MBPP train120, LoRA r=16, 3 epochs | MBPP eval20 | 10/20 | 9/20 | close, below base |
| Qwen2.5-Coder-0.5B-Instruct | conservative SFT | MBPP train120, LoRA r=8, lr=2e-5, 1 epoch | MBPP eval20 | 10/20 | 9/20 | below base |
| Qwen2.5-Coder-0.5B-Instruct | expanded-data SFT | MBPP augmented 400, LoRA r=16, lr=5e-5, 2 epochs | MBPP eval20 | 10/20 | 9/20 | below base |
| Qwen2.5-Coder-0.5B-Instruct | local GRPO | MBPP train40, 4 rollouts, 1 epoch, LoRA r=8 | MBPP eval20 | 8/20 | 9/20 | small gain |
| Qwen2.5-Coder-0.5B-Instruct | local GRPO | MBPP train120, 4 rollouts, 1 epoch | MBPP eval20 | 8/20 | 9/20 | small gain replicated |
| Qwen2.5-Coder-1.5B-Instruct | local GRPO | MBPP train200, 4 rollouts, 1 epoch, LoRA r=16 | MBPP eval100 | 47/100 | 49/100 | best positive case |
| Qwen2.5-Coder-1.5B-Instruct | local GRPO | MBPP train400, 4 rollouts, 1 epoch | MBPP eval100 | 47/100 | 46/100 | degraded |
| Qwen2.5-Coder-3B-Instruct | local GRPO | MBPP train200, 4 rollouts, 1 epoch | MBPP eval100 | 62/100 | 60/100 | degraded |
| Qwen2.5-Coder-3B-Instruct | conservative SFT | MBPP train400, LoRA r=8, lr=1e-5, 1 epoch | MBPP eval100 | 62/100 | 62/100 | tied |
| Qwen2.5-Coder-7B-Instruct | local GRPO | MBPP train100, 4 rollouts, 1 epoch | MBPP eval100 | 77/100 | 75/100 | degraded |

## Training paradigms

### Supervised fine-tuning

SFT uses reference code as the target completion. It was tested on hard
HumanEval-style examples and MBPP-derived examples. Early Alpaca/code prompts
caused many output-format issues. Switching to `qwen-chat` improved formatting
and base evaluation, but SFT still did not exceed the base model.

Observed pattern:

- SFT can easily degrade a small instruction-tuned code model on small data.
- Prompt formatting matters a lot.
- More SFT data did not automatically improve results.

### Direct Preference Optimization

DPO data was prepared with:

```text
chosen   = reference code
rejected = base model failed output
```

The data builder produced 86 DPO pairs on A800. However, DPOTrainer did not run
in the then-current A800 environment because the installed TRL package expected
a newer/different torch FSDP API.

Status:

- DPO data construction: done.
- DPO training: not completed due to environment compatibility.

### Local GRPO / RL

Local GRPO avoids TRL and uses executable tests directly as rewards:

```text
sample multiple completions
run tests
reward = pass/fail
group-relative advantage
LoRA update
```

This is the first paradigm that produced a positive LLM-training result on a
larger held-out evaluation:

```text
Qwen2.5-Coder-1.5B-Instruct
base: 47/100
GRPO: 49/100
```

The improvement is small but positive. Scaling model size alone did not help:
3B and 7B GRPO runs degraded relative to their stronger base models.

## Current best result

The strongest positive LLM-training result is:

```text
Model: Qwen2.5-Coder-1.5B-Instruct
Method: local GRPO
Train: 200 MBPP augmented tasks, 4 rollouts per prompt, 1 epoch
Eval: MBPP eval100
Base: 47/100
Adapter: 49/100
```

Artifacts:

```text
/data0/home/zeyuwang/router-skills-evolve-results/rl_15b/qwen25_coder_15b_base_eval100.json
/data0/home/zeyuwang/router-skills-evolve-results/rl_15b/qwen25_coder_15b_grpo_eval100.json
/data0/home/zeyuwang/router-skills-evolve-runs/rl_15b/qwen25_coder_15b_grpo_200x4
```

## Takeaways

1. Router evolution is already strong and stable.
2. Naive SFT does not reliably improve small code models.
3. Local GRPO can beat base, but the margin is small.
4. Larger models have stronger bases, and the current lightweight RL recipe can
   degrade them.
5. Next steps should focus on better RL/DPO recipes, not just larger models.

## Recommended next experiments

- Use a compatible official DPO/GRPO stack with KL/reference penalty.
- Increase rollout budget with pass@k-style filtering.
- Train on the unified Evolver Dataset `code_tasks/train` and evaluate on
  `code_tasks/test`.
- Add reward shaping for syntax validity, correct function name, and test pass.
- Run multiple seeds for the 1.5B GRPO setup to verify robustness.

