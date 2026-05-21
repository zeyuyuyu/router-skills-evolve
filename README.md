# router-skills-evolve

A self-improving LLM serving system that jointly evolves three orthogonal
components from production traces:

1. **Router** — a learnable BERT-style classifier that routes each prompt to
   a cheap or expensive LLM.
2. **SkillBook** — an online frequency table over prompt signatures that
   tracks each model's success rate on each prompt cluster.
3. **LLM adapter** — a continually-trained LoRA on the cheap LLM, hardening
   it on the tasks the router currently sends to the expensive one.

All three update in a cycle:

```
    SkillBook (online stats)
           │
           ▼
    LLM continual GRPO  ──── traces ──── Learnable router (BERT-tiny)
           │                                   │
           ▼                                   ▼
       per-task pass/fail              routing accuracy / fallback
                  │       │       │
                  ▼       ▼       ▼
                Joint evaluation
```

The full pipeline is driven by `experiments/run_joint_evolver.py` for one-shot
runs and by `auto_research/` for 24/7 autonomous operation.

## Quick start

```bash
# environment
pip install -r requirements.txt
export COMMONSTACK_API_KEY="..."        # for the router-skills-evolve LLM calls
export HF_ENDPOINT=https://hf-mirror.com  # optional, for hf-cache mirror

# clone and a smoke run
git clone https://github.com/zeyuyuyu/router-skills-evolve
cd router-skills-evolve
python3 experiments/run_evolve.py --n 30 --rounds 3
```

## Layout

```
router-skills-evolve/
├── src/                    # core: SkillBook, RouterWithSkills, models.py
├── experiments/            # one-shot training and evaluation scripts
│   ├── run_evolve.py                       # SkillBook + traces loop
│   ├── run_joint_evolver.py                # 1-cycle joint of all 3 tracks
│   ├── run_e2e_ablation.py                 # offline ablation reporting
│   ├── train_learnable_router.py           # BERT-tiny router
│   ├── train_small_model_grpo_local.py     # LoRA + GRPO + K3 KL
│   ├── train_small_model_dpo.py            # LoRA + DPO
│   └── evaluate_finetuned_model.py
├── scripts/                # multi-cycle ordering / iterated drivers
│   ├── run_iterated_skill_llm_router.sh    # N-cycle joint loop
│   ├── run_ordering_seq.sh                 # skill_first / llm_first
│   ├── run_ordering_exp.sh                 # serial / xiaojie / user / parallel
│   └── run_ordering_parallel.sh            # router ‖ LLM dual-GPU
├── auto_research/          # 24/7 autonomous orchestrator
│   ├── orchestrator.py                     # cron tick: collect + launch
│   ├── runner.py                           # per-kind experiment dispatcher
│   ├── idea_gen_local.py                   # rule-based fallback ideator
│   ├── arxiv_scrape.py                     # daily LLM-keyword arxiv pull
│   ├── cron_entry.sh                       # crontab target (every 30 min)
│   └── README.md                           # orchestrator docs
├── data/
│   ├── HumanEval.jsonl                     # 164 code tasks
│   ├── training_data.jsonl                 # extracted SFT pairs
│   ├── traces/                             # accumulated routing traces
│   └── skills/                             # persisted SkillBook snapshots
├── results/                # per-run JSON summaries
│   ├── iterated_skill_llm_router_8cycles/  # 8-cycle joint run, raw evals
│   └── *.json                              # historical headline numbers
├── docs/
│   ├── HIGHLIGHTS.md                       # ⭐ current best results
│   ├── E2E_ABLATION_RESULTS.md             # full ablation tables
│   ├── ARCHITECTURE.md, JOINT_EVOLVER.md, ...
└── requirements.txt
```

## Three ways to run

### 1. One-shot joint evolver

```bash
python3 experiments/run_joint_evolver.py \
  --cycles 1 \
  --traces "data/traces/*.jsonl" \
  --router-base-model google/bert_uncased_L-2_H-128_A-2 \
  --llm-train-data data/training_data.jsonl \
  --llm-base-model Qwen/Qwen2.5-Coder-1.5B-Instruct
```

Produces `joint_evolver_manifest.json` with all stage commands, metrics, and
artifact paths.

### 2. Multi-cycle iterated pipeline (Skill → LLM → Router, repeated)

```bash
NUM_CYCLES=8 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_iterated_skill_llm_router.sh
```

Each cycle does: SkillBook online update → continual-train LoRA on a fresh
MBPP chunk → retrain BERT router on the cumulative bench slice.

### 3. 24/7 autonomous mode (cron-driven)

```bash
# one-time setup on a GPU host with at least one A800-class card:
bash auto_research/cron_entry.sh   # smoke
crontab -e  # add: */30 * * * * /path/to/auto_research/cron_entry.sh
```

The orchestrator runs every 30 minutes: it `git pull`s, applies any pending
queue updates from cloud agents, launches the highest-priority queued
experiment on a free GPU, and collects finished experiments. Pair with cloud
Claude routines (idea-gen daily, shepherd every 2h) for fully autonomous
research.

See [`auto_research/README.md`](auto_research/README.md) for details.

## Current best results

See [`docs/HIGHLIGHTS.md`](docs/HIGHLIGHTS.md) for the complete summary. As of
2026-05-21:

- **Router**: cold-start 57.6% → cycle 3 peak 87.8% → settled 82–88%
  (8-cycle iterated, BERT-tiny, 832-task held-out)
- **Skills**: monotonic accumulation, 34 signatures from 8-cycle run
- **LLM 1.5B GRPO**: ~47% on MBPP eval200 (recipe-invariant across orderings
  and cycle counts; 4 candidate configurations show 50–51% on eval100, under
  verification)
- **LLM 3B GRPO**: 61.0% on MBPP eval200
- **Pipeline**: only `parallel` ordering saves wall time (~4 min/cycle from
  router ‖ LLM dual-GPU)

See [`docs/E2E_ABLATION_RESULTS.md`](docs/E2E_ABLATION_RESULTS.md) for the
full table.

## Concepts

### Router

A BERT-tiny binary classifier (`google/bert_uncased_L-2_H-128_A-2` by default)
that predicts whether to route a prompt to the cheap or expensive LLM. Trained
on a mix of UncommonRoute weak labels and accumulated traces. Threshold tuned
per cost/quality target.

### SkillBook

A signature → frequency table. The signature is a hand-coded folding of the
prompt:

```
length_bucket × {list, str, num, sort, theory, crypto, advanced, bool, ...}
```

Per signature, the SkillBook stores `(model_id) → (successes, total)`.
Routing decisions consult the book with Laplace-smoothed success rates.

### Joint cycle

Each cycle k:

1. SkillBook ← merge new traces (online, sub-second).
2. LLM adapter ← continual GRPO step on a fresh task chunk; LoRA is *resumed*
   from cycle k-1.
3. Router ← retrain BERT on cumulative router-supervision data.

Evaluation: MBPP eval200 for the LLM, 832-task held-out for the router.

## Languages / 语言

- **English (this file)**: README.md
- **中文**: [README.zh.md](README.zh.md)

## Citation

If you use this code or data, please cite via the GitHub release page:
https://github.com/zeyuyuyu/router-skills-evolve

## License

MIT.
