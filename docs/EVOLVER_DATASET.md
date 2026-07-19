# Evolver Dataset Bundle

`experiments/build_evolver_dataset.py` builds a larger, structured dataset for
router/model evolution experiments. The intent is to move beyond tiny smoke
tests such as "20 MBPP examples" and produce fixed train/dev/test splits for
paper-style experiments.

## Data sources

The builder can combine:

- **HumanEval**: executable code-generation tasks with reference solutions.
- **Hard traces**: `data/training_data.jsonl`, small but high-value hard cases
  where the large model produced a passing solution.
- **MBPP**: executable Python tasks with reference code and tests.
- **UncommonRoute bench**: prompt/tier labels used as weak supervision for
  router training, not for code-generation SFT/RL.

## Output layout

```text
data/evolver_dataset/
├── code_tasks/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
├── router_tasks/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
├── dataset_stats.json
├── sample.json
└── DATASET_CARD.md
```

## Code-task schema

Each row in `code_tasks/*.jsonl` contains:

```json
{
  "id": "MBPP/602",
  "source": "mbpp",
  "source_split": "train",
  "task_type": "code_generation",
  "prompt": "...",
  "instruction": "...",
  "input": "",
  "reference_solution": "def ...",
  "output": "def ...",
  "entry_point": "function_name",
  "test": "def check(candidate): ...",
  "has_tests": true,
  "has_reference": true,
  "signature": "M|list/num",
  "metadata": {}
}
```

## Router-task schema

Each row in `router_tasks/*.jsonl` contains:

```json
{
  "id": "uncommonroute/all/0",
  "source": "uncommonroute_bench",
  "source_split": "all",
  "task_type": "router_supervision",
  "prompt": "...",
  "router_label": 0,
  "router_label_name": "small",
  "tier": "MEDIUM",
  "category": "comparison",
  "lang": "en",
  "weak_label": true,
  "signature": "S|general"
}
```

## Build commands

Smoke test without external downloads:

```bash
python3 experiments/build_evolver_dataset.py \
  --skip-external \
  --dry-run
```

Full build with MBPP and UncommonRoute:

```bash
python3 experiments/build_evolver_dataset.py \
  --output-dir data/evolver_dataset \
  --mbpp-splits train validation test \
  --uncommonroute-files bench/data/all.jsonl
```

On machines without direct HuggingFace access, use a mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com \
python3 experiments/build_evolver_dataset.py \
  --output-dir data/evolver_dataset
```

## How to use

- Router training should use `router_tasks/train.jsonl` plus real trace data.
- LLM SFT/RL should use `code_tasks/train.jsonl` and evaluate on held-out
  `code_tasks/dev.jsonl` or `code_tasks/test.jsonl`.
- The dataset card and stats should be included in experiment reports.

