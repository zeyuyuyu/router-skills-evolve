# Evolver Dataset

This dataset bundle is built by `experiments/build_evolver_dataset.py`.

## Purpose

The bundle is meant for EMNLP-style evolver experiments where router learning
and model learning are evaluated together. It separates executable code tasks
from router-only weak supervision so experiments can report clearly what signal
is used.

## Schema

### Code tasks

- `id`
- `source`, `source_split`
- `task_type = code_generation`
- `prompt` / `instruction`
- `reference_solution` / `output`
- `entry_point`
- `test`
- `has_tests`, `has_reference`
- `signature`
- `metadata`

### Router tasks

- `id`
- `source`, `source_split`
- `task_type = router_supervision`
- `prompt`
- `router_label`, `router_label_name`
- `tier`, `category`, `lang`
- `weak_label`
- `signature`

## Statistics

```json
{
  "code_tasks": {
    "total": 590,
    "by_source": {
      "humaneval": 164,
      "hard_traces": 6,
      "mbpp": 420
    },
    "by_source_split": {
      "humaneval:all": 164,
      "hard_traces:all": 6,
      "mbpp:train": 120,
      "mbpp:validation": 43,
      "mbpp:test": 257
    },
    "by_task_type": {
      "code_generation": 590
    },
    "has_tests": 590,
    "has_reference": 590
  },
  "router_tasks": {
    "total": 3328,
    "by_source": {
      "uncommonroute_bench": 3328
    },
    "by_source_split": {
      "uncommonroute_bench:all": 3328
    },
    "by_task_type": {
      "router_supervision": 3328
    },
    "has_tests": 0,
    "has_reference": 0,
    "router_labels": {
      "small": 2258,
      "large": 1070
    }
  },
  "splits": {
    "code_tasks": {
      "train": 472,
      "dev": 59,
      "test": 59
    },
    "router_tasks": {
      "train": 2662,
      "dev": 332,
      "test": 334
    }
  }
}
```

## Build arguments

```json
{
  "output_dir": "/data/djh/router-skills/datasets/evolver_dataset_v1",
  "seed": 42,
  "train_ratio": 0.8,
  "dev_ratio": 0.1,
  "include_humaneval": true,
  "humaneval_path": "data/HumanEval.jsonl",
  "include_hard_traces": true,
  "hard_traces_path": "data/training_data.jsonl",
  "include_mbpp": true,
  "mbpp_dataset": "google-research-datasets/mbpp",
  "mbpp_config": "sanitized",
  "mbpp_splits": [
    "train",
    "validation",
    "test"
  ],
  "max_mbpp_per_split": null,
  "include_uncommonroute": true,
  "uncommonroute_repo": "CommonstackAI/UncommonRoute",
  "uncommonroute_files": [
    "bench/data/all.jsonl"
  ],
  "max_uncommonroute": null,
  "skip_external": false,
  "dry_run": false
}
```

## Notes

- UncommonRoute bench labels are weak router labels, not executable code tasks.
- MBPP and HumanEval contain executable tests and references.
- Existing `data/training_data.jsonl` hard traces are high-value but small.
