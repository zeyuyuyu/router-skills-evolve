 # Learnable Router

This repo now has two different training paths:

1. `experiments/train_small_model*.py` trains the code-generation model.
2. `experiments/train_learnable_router.py` trains the router itself.

The learnable router is a BERT-style binary classifier:

- label `0` / `small`: try the cheap model first.
- label `1` / `large`: skip the cheap model and call the expensive model.

Labels are built from traces. If a trace shows that the small model passed, the
label is `small`. If the small model failed and the large model passed, the
label is `large`.

## Train

```bash
pip install torch transformers datasets scikit-learn

python3 experiments/import_uncommonroute_bench.py \
  --files bench/data/all.jsonl \
  --output data/router_training/uncommonroute_bench.jsonl

python3 experiments/train_learnable_router.py \
  --traces "data/traces/*.jsonl" \
  --router-data data/router_training/uncommonroute_bench.jsonl \
  --tasks data/HumanEval.jsonl \
  --base-model google/bert_uncased_L-2_H-128_A-2 \
  --output outputs/learned-router \
  --epochs 8
```

Use a larger encoder when enough traces are available:

```bash
python3 experiments/train_learnable_router.py \
  --base-model bert-base-uncased \
  --output outputs/learned-router-bert-base
```

## Offline evaluation

```bash
python3 experiments/evaluate_learnable_router.py \
  --model outputs/learned-router \
  --traces "data/traces/*.jsonl" \
  --router-data data/router_training/uncommonroute_bench.jsonl \
  --tasks data/HumanEval.jsonl \
  --output results/learned_router_eval.json
```

Offline evaluation replays labelled traces without calling LLMs. It estimates
route accuracy, fallback count, and cost compared with always using the large
model or always trying small with fallback.

## Threshold tuning

The classifier outputs `p_large`. The default threshold is `0.5`, but production
usually needs an explicit cost/quality trade-off:

```bash
python3 experiments/tune_learnable_router_threshold.py \
  --model outputs/learned-router \
  --traces "data/traces/*.jsonl" \
  --router-data data/router_training/uncommonroute_bench.jsonl \
  --max-fallback-rate 0.02 \
  --output results/learned_router_thresholds.json
```

The tuner scans thresholds and recommends the lowest-cost setting that satisfies
the requested fallback-rate cap.

## Joint evolver experiment

The current evolver loop can run three tracks side by side:

1. SkillBook statistics: existing signature-based routing.
2. Learnable router: `train_learnable_router.py` + threshold tuning.
3. Small LLM fine-tuning: `train_small_model.py` on hard examples, then
   `evaluate_finetuned_model.py` on the same HumanEval tasks.

Example fine-tuned model evaluation:

```bash
python3 experiments/evaluate_finetuned_model.py \
  --base-model MiniMaxAI/MiniMax-M2 \
  --adapter outputs/minimax-m2-finetuned \
  --data data/training_data.jsonl \
  --output results/finetuned_eval.json
```

Use this to compare base small-model pass rate against the fine-tuned adapter
before updating the router's small-model success assumptions.
