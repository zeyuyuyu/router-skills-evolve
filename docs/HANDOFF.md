# Router Skills Evolve Handoff

This handoff is intended for anyone who needs to continue the router/model
evolver work without reading the full chat history.

## 0. Repository

```text
https://github.com/zeyuyuyu/router-skills-evolve
```

Main branch currently contains:

- learnable router pipeline
- unified dataset builder
- end-to-end joint runner
- end-to-end ablation script
- local GRPO trainer
- experiment summaries and paper draft

Important docs:

```text
docs/JOINT_EVOLVER.md
docs/E2E_ABLATION_RESULTS.md
docs/LLM_TRAINING_RESULTS.md
docs/EVOLVER_DATASET.md
docs/EVOLVER_DATASET_CARD_V1.md
docs/PAPER_DRAFT.md
```

Important result summaries:

```text
results/e2e_ablation_a800_20260509_summary.json
results/rl_15b_a800_20260509_summary.json
results/best_llm_training_case_20260509.json
results/llm_training_summary_20260510.json
results/datasets/evolver_dataset_v1_stats.json
```

Open/auxiliary branches that may still be useful:

```text
cursor/build-evolver-dataset-29f9
cursor/rl-llm-evolver-29f9
cursor/dpo-llm-evolver-29f9
```

## 1. Current conclusion

### Router evolution

Router evolution is the strongest positive result.

From the A800 end-to-end ablation:

| Variant | Routing Accuracy | Large F1 | Fallback Rate | Cost vs Always-Large |
| --- | ---: | ---: | ---: | ---: |
| Base: always-small + fallback | 68.28% | 0.00% | 31.72% | 33.54% |
| + SkillBook | 69.46% | 24.93% | 26.65% | 37.27% |
| + Learned router | **93.04%** | **89.48%** | **2.12%** | 37.75% |

### LLM evolution

Naive SFT did not beat base. Local GRPO produced the first positive result.

Best positive case:

```text
Model: Qwen2.5-Coder-1.5B-Instruct
Method: local GRPO
Train: 200 MBPP augmented tasks, 4 rollouts/prompt, 1 epoch
Eval: MBPP eval100
Base: 47/100
Adapter: 49/100
```

Scaling alone is not enough:

```text
1.5B GRPO 400x4: 47/100 -> 46/100
3B GRPO 200x4: 62/100 -> 60/100
3B conservative SFT: 62/100 -> 62/100
7B GRPO 100x4: 77/100 -> 75/100
```

## 2. Machines and paths

### A800 server

```text
ssh -p 50507 zeyuwang@117.74.66.181
password: Ls453u4f
```

Proxy for HuggingFace/GitHub:

```bash
export http_proxy=http://127.0.0.1:1080
export https_proxy=http://127.0.0.1:1080
```

Repo and artifacts:

```text
/data0/home/zeyuwang/router-skills-evolve
/data0/home/zeyuwang/router-skills-evolve-data
/data0/home/zeyuwang/router-skills-evolve-runs
/data0/home/zeyuwang/router-skills-evolve-results
```

Notable A800 artifacts:

```text
/data0/home/zeyuwang/router-skills-evolve-runs/learned-router-mixed-pretrained
/data0/home/zeyuwang/router-skills-evolve-results/e2e_ablation/e2e_ablation_router_sft_qwenchat.json
/data0/home/zeyuwang/router-skills-evolve-runs/rl_15b/qwen25_coder_15b_grpo_200x4
/data0/home/zeyuwang/router-skills-evolve-results/rl_15b/qwen25_coder_15b_grpo_eval100.json
/data0/home/zeyuwang/router-skills-evolve-results/rl_more/
```

### 8x4090 server

```text
ssh -p 7980 djh@111.2.199.31
password: jushendjh
```

Data disk:

```text
/data/djh/router-skills
```

Virtual environment:

```text
/data/djh/router-skills/venv
```

The 4090 host initially had no pip/torch. Installed:

```bash
sudo apt-get install -y python3-pip python3-venv
python3 -m venv /data/djh/router-skills/venv
/data/djh/router-skills/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
/data/djh/router-skills/venv/bin/pip install transformers datasets peft accelerate openai requests scikit-learn
```

Important: default PyTorch pulled CUDA13 and did not work with the driver.
Use CUDA 12.8 wheels.

### 6xA40 server

```text
ssh -p 62494 root@221.220.242.224
password: 33v0ZsHJAyCLHFVD
```

This host has no direct HuggingFace access. It has a SOCKS proxy at
`127.0.0.1:1080`, not an HTTP proxy:

```bash
export ALL_PROXY=socks5h://127.0.0.1:1080
export all_proxy=socks5h://127.0.0.1:1080
```

For Python requests/HF, install:

```bash
apt-get install -y python3-socks
```

It was used for some local GRPO smoke tests.

## 3. Data

### Unified Evolver Dataset v1

Built on 8x4090:

```text
/data/djh/router-skills/datasets/evolver_dataset_v1
```

Committed summaries:

```text
results/datasets/evolver_dataset_v1_stats.json
docs/EVOLVER_DATASET_CARD_V1.md
```

Stats:

```text
Code tasks: 590
- HumanEval: 164
- hard traces: 6
- MBPP: 420
- all have references and tests

Code splits:
- train: 472
- dev: 59
- test: 59

Router tasks: 3328
- UncommonRoute weak labels: 3328
- small: 2258
- large: 1070

Router splits:
- train: 2662
- dev: 332
- test: 334
```

Build command:

```bash
HF_ENDPOINT=https://hf-mirror.com \
python3 experiments/build_evolver_dataset.py \
  --output-dir /data/djh/router-skills/datasets/evolver_dataset_v1 \
  --uncommonroute-files bench/data/all.jsonl
```

## 4. Key scripts

### Dataset

```text
experiments/build_evolver_dataset.py
experiments/import_mbpp_training_data.py
experiments/import_uncommonroute_bench.py
```

### Router

```text
experiments/train_learnable_router.py
experiments/evaluate_learnable_router.py
experiments/tune_learnable_router_threshold.py
```

### End-to-end / ablation

```text
experiments/run_joint_evolver.py
experiments/run_e2e_ablation.py
```

### LLM training/eval

```text
experiments/train_small_model.py
experiments/evaluate_finetuned_model.py
experiments/build_mbpp_dpo_data.py
experiments/train_small_model_dpo.py
experiments/train_small_model_grpo_local.py
```

## 5. Reproduce key experiments

### Router ablation

On A800:

```bash
cd /data0/home/zeyuwang/router-skills-evolve
export http_proxy=http://127.0.0.1:1080
export https_proxy=http://127.0.0.1:1080

python3 experiments/run_e2e_ablation.py \
  --traces 'data/traces/*.jsonl' \
  --router-data /data0/home/zeyuwang/router-skills-evolve-data/uncommonroute_bench.jsonl \
  --tasks data/HumanEval.jsonl \
  --learned-router-model /data0/home/zeyuwang/router-skills-evolve-runs/learned-router-mixed-pretrained \
  --learned-threshold 0.57 \
  --llm-base-eval /data0/home/zeyuwang/router-skills-evolve-runs/joint_evolver_real_qwenchat_20260502_084947/cycle_00/results/llm_base_eval.json \
  --llm-adapter-eval /data0/home/zeyuwang/router-skills-evolve-runs/joint_evolver_real_qwenchat_20260502_084947/cycle_00/results/llm_adapter_eval.json \
  --output /data0/home/zeyuwang/router-skills-evolve-results/e2e_ablation/e2e_ablation_router_sft_qwenchat.json \
  --markdown-output /data0/home/zeyuwang/router-skills-evolve-results/e2e_ablation/e2e_ablation_router_sft_qwenchat.md
```

### Best current LLM case

On A800:

```bash
cd /data0/home/zeyuwang/router-skills-evolve
export http_proxy=http://127.0.0.1:1080
export https_proxy=http://127.0.0.1:1080

CUDA_VISIBLE_DEVICES=6 python3 experiments/train_small_model_grpo_local.py \
  --data /data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/train_aug_excluding_eval20.jsonl \
  --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --output /data0/home/zeyuwang/router-skills-evolve-runs/rl_15b/qwen25_coder_15b_grpo_200x4 \
  --limit 200 \
  --epochs 1 \
  --n-generations 4 \
  --max-new-tokens 192 \
  --temperature 0.8 \
  --top-p 0.95 \
  --lr 5e-6 \
  --lora-r 16 \
  --prompt-style qwen-chat \
  --reward-baseline zero

CUDA_VISIBLE_DEVICES=6 python3 experiments/evaluate_finetuned_model.py \
  --data /data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/test_eval_all.jsonl \
  --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --adapter /data0/home/zeyuwang/router-skills-evolve-runs/rl_15b/qwen25_coder_15b_grpo_200x4 \
  --output /data0/home/zeyuwang/router-skills-evolve-results/rl_15b/qwen25_coder_15b_grpo_eval100.json \
  --max-new-tokens 384 \
  --prompt-style qwen-chat \
  --limit 100
```

Expected result:

```text
base: 47/100
GRPO adapter: 49/100
```

## 6. Current open experimental questions

1. **Make LLM training margin larger.**
   - Current best is only +2 points.
   - More rollouts or more tasks did not automatically help.
   - 3B and 7B bases are stronger but current lightweight GRPO degraded them.

2. **Replace lightweight GRPO with more stable RL/DPO.**
   - Add KL/reference penalty.
   - Use advantage normalization with a reference model.
   - Try official DPO/GRPO stack once torch/TRL versions are compatible.

3. **Use fixed dataset splits.**
   - Use Evolver Dataset v1 `code_tasks/train/dev/test`.
   - Avoid 20-example smoke tests for paper claims.

4. **Improve reward shaping.**
   - Reward syntax validity.
   - Reward correct function name.
   - Reward partial test pass count, not just binary pass/fail.

## 7. Known pitfalls

- **A800 proxy:** use HTTP proxy vars:
  `http_proxy=http://127.0.0.1:1080`, `https_proxy=http://127.0.0.1:1080`.
- **A40 proxy:** use SOCKS proxy vars:
  `ALL_PROXY=socks5h://127.0.0.1:1080`; install `python3-socks`.
- **4090 PyTorch:** default torch CUDA13 wheel fails; use CUDA12.8 wheel.
- **SFT can degrade.** Do not assume reference-code SFT improves an
  instruction-tuned code model.
- **Large model != automatic gain.** 3B/7B bases were strong, but current GRPO
  recipe degraded them.
- **Do not commit checkpoints.** Only commit small JSON/Markdown summaries.

## 8. Recommended next steps

1. Merge open PRs if desired:
   - Dataset/paper draft PR #8
   - RL/GRPO PR #7
2. Convert `train_small_model_grpo_local.py` reward from binary to partial
   reward based on number of passing asserts.
3. Rerun 1.5B GRPO with partial rewards and KL penalty.
4. Run official DPO once environment compatibility is fixed.
5. Update `docs/PAPER_DRAFT.md` tables with final ablation numbers.

