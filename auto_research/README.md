# Auto-research (v0)

24/7 experiment orchestrator for the router-skills-evolve LLM research project.

## Layout

```
~/auto_research/
├── state.json        # canonical queue + history + config
├── orchestrator.py   # single-shot driver (called by cron every 30 min)
├── runner.py         # dispatches experiment kinds to actual training/eval
├── report.py         # writes reports/daily-YYYY-MM-DD.md
├── cron_entry.sh     # crontab line target
├── logs/             # orchestrator + per-experiment logs
├── runs/             # per-experiment outputs (adapters, eval jsons)
└── reports/          # markdown reports
```

## Cron

```
*/30 * * * * /data0/home/zeyuwang/auto_research/cron_entry.sh >> /data0/home/zeyuwang/auto_research/logs/cron.log 2>&1
0 23 * * *  python3 /data0/home/zeyuwang/auto_research/report.py >> /data0/home/zeyuwang/auto_research/logs/cron.log 2>&1
```

## Adding experiments manually

Edit `state.json`, append to `queue`. The orchestrator's next tick picks
it up. Required fields: `id`, `kind`, `priority`, `spec`. Optional:
`rationale`, `gpu`.

## Kinds (v0)

| Kind | What it does |
| --- | --- |
| `grpo_continual` | Continual-train LoRA adapter through k chunks |
| `grpo_curriculum_continual` | Same but step k trains on failures of step k-1 (v1 will implement; v0 = grpo_continual) |
| `grpo_multi_seed_staircase` | Run continual with non-default seed (v1: true multi-seed) |
| `forgetting_eval` | Eval N adapters on a fixed prompt set |
| `joint_cycle_multiseed` | Invoke existing run_joint_cycle.sh |

## Coming in v1

- arxiv RSS daily fetch (cs.CL + cs.LG, LLM keywords)
- Claude API idea generator: read recent history + arxiv, propose 5-10 next experiments
- Paper drafter: weekly Claude agent that compiles top results into a paper section
- True multi-seed handler
- Real curriculum-continual (failure mining between steps)
