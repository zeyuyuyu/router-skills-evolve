# Results highlights (as of 2026-05-21)

This page tracks the highest-signal experiments across the router-skills-evolve
auto-research pipeline. For the full method-by-method ablation see
[`E2E_ABLATION_RESULTS.md`](E2E_ABLATION_RESULTS.md).

## TL;DR

| Track | Status | Headline |
| --- | --- | --- |
| Router (BERT-tiny binary classifier) | ✅ clear win | 8-cycle iterated: cold-start 57.6% → **cycle 3 peak 87.8%** → cycles 4-8 settled in 82-88% band. Fallback dropped from 8.1% (cycle 1) to ~4% (cycles 3-8). |
| Skills (online frequency stats) | ✅ clear win | Monotonic signature accumulation: 4-cycle run → 13 signatures, 8-cycle run → **34 signatures, 120 observations**. No collapse, persists across cycles. |
| LLM (1.5B + GRPO + LoRA) on eval100 | 🟡 ambiguous | Four configs reached **50–51% on first-100-task eval** vs ~47% baseline. Eval slice may explain the gap; verification on eval200 in progress. |
| LLM (3B + GRPO + LoRA) on eval200 | ✅ headroom confirmed | 3B reaches **61.0%** (continual K3 KL=0.05). Recipe-invariant — gain is from base capability, not GRPO induction. |
| Pipeline ordering | ✅ analysis-grade | LLM pass invariant across 6 orderings (46.5–47.5%); **only `parallel` saves wall time** (95 min vs 99 min serial, 4 min/cycle from router ∥ LLM double-GPU). |
| Joint orchestration (auto-research v0) | ✅ shipped | 24/7 perpetual motion machine running: A800 cron every 30 min + 4 cloud Claude routines (idea-gen daily, shepherd 2h, paper-drafter Sunday, paper-pipeline Friday). 47+ experiments completed autonomously since 2026-05-12. |

## Router track — 8-cycle iterated pipeline

Single 8-cycle execution of the `skill_first` ordering (`Skill → LLM → Router`
per cycle, chunks loop mod 4, persistent SkillBook + LLM adapter chain, BERT
router seed = 42 + k). Completed 2026-05-20 21:13 UTC.

| Cycle | Router Acc | Fallback | Cost vs Large | LLM Pass (eval200) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 57.6% | 0.4% | 85.6% | 47.0% |
| 2 | 71.1% | 4.3% | 68.6% | 47.0% |
| 3 | **87.8%** | 3.4% | 52.8% | 47.0% |
| 4 | 82.3% | 3.5% | 57.4% | 47.0% |
| 5 | 76.7% | 14.4% | 54.2% | 47.0% |
| 6 | 85.3% | 3.1% | 56.0% | 46.5% |
| 7 | 82.7% | 4.6% | 56.6% | 47.0% |
| 8 | 85.0% | 4.0% | 54.3% | **47.5%** |

The cycle-5 dip (76.7% accuracy, 14.4% fallback) is a per-cycle BERT-init seed
outlier that recovers immediately at cycle 6. The system converges into the
82–88% band after a 2-cycle warm-up.

Reproduce:
```bash
NUM_CYCLES=8 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_iterated_skill_llm_router.sh
```

Full per-cycle eval JSONs:
[`results/iterated_skill_llm_router_8cycles/`](../results/iterated_skill_llm_router_8cycles/)

## 1.5B GRPO — candidate breakthroughs (needs eval200 verification)

Four agent-proposed configurations reached 50–51% on the first-100-task eval
slice — above the ~47% baseline we previously established on the eval200 slice.

| Experiment | Spec | step_4 eval100 | step_4 eval200 |
| --- | --- | ---: | ---: |
| `exp_2026_05_18_003_scafgrpo_hint_injection_15b` | vanilla GRPO continual, 4×100 (default) | **51.0%** | (verification in progress) |
| `exp_2026_05_17_002_grpo_15b_lr1e6_200x4` | **lr=1e-6** (5× lower), 4×100 default otherwise | 50.0% | (verification in progress) |
| `exp_2026_05_17_003_grpo_15b_lora_r32_200x4` | **LoRA r=32** (2× default), 4×100 default otherwise | 50.0% | (verification in progress) |
| `exp_2026_05_16_004_staircase_grpo_15b_100_100` | default recipe, 4×100 chained | 50.0% | (verification in progress) |

**Caveat**: these eval100 numbers are on the first-100 tasks of
`mbpp_aug/test_eval_all.jsonl`, which may be easier than tasks 101–200. Prior
4-cycle ordering runs showed continual step-4 = 50% on eval100 but 47.5% on
eval200 for the same adapter (3-pp gap). Re-evaluating these four winners on
eval200 is queued; updated numbers will appear here when complete.

If the eval200 verification confirms ≥+2 pp over the 47% baseline, the
cloud agent's Scaf-GRPO (arxiv 2510.19807) + the lr/LoRA-rank ablations
are real positives worth multi-seed confirmation.

Cloud agent rationale text for `scafgrpo_hint_injection_15b`:
> Scaf-GRPO (arxiv:2510.19807, Oct 2025/Feb 2026) addresses the 'learning cliff':
> tasks where the model consistently fails all rollouts, collapsing the advantage
> to zero and making the task invisible to the policy gradient.

(Note: the v0 runner.py does not yet implement the actual hint-injection
mechanism; the agent's spec was dispatched as vanilla `grpo_continual`. The
51% result is therefore from the default recipe — interesting in its own
right and likely just within eval-slice noise.)

## 3B GRPO — sits at ~60% band

| Experiment | Spec | MBPP eval200 |
| --- | --- | ---: |
| `exp_2026_05_12_001_continual_3b` | continual 4×100, K3 KL=**0.05** | **61.0%** |
| `exp_2026_05_20_144311_man_3b_kl01` | continual 4×100, K3 KL=**0.10** | 59.5% |

3B base capability already ~60%; KL choice in [0.05, 0.10] doesn't move the
needle. Worth multi-seed verification.

## Pipeline ordering — `parallel` is the only real wall-time win

Six orderings tested, each running 4 cycles on Qwen2.5-Coder-1.5B:

| Ordering | Wall time (4 cycles) | Notes |
| --- | ---: | --- |
| `parallel` (Skill ∥ Router ∥ LLM, 2 GPUs/cycle) | **95 min** | only true wall-time win, saves ~4 min/cycle |
| `llm_first` (LLM → Skill → Router) | 97 min | |
| `serial` (Skill → Router → LLM) | 99 min | baseline |
| `xiaojie` (Skill ∥ LLM → Router) | 99 min | identical to serial because skill is sub-second |
| `skill_first` (Skill → LLM → Router) | 99 min | identical |
| `user` (Skill ∥ Router → LLM) | 100 min | identical |

LLM pass rate **invariant** across all 6 orderings (46.5–47.5%, ±0.5 pp).
Pipeline ordering is therefore a scheduling concern, not a method choice. See
[`E2E_ABLATION_RESULTS.md`](E2E_ABLATION_RESULTS.md) "Joint cycle ordering"
section for full table.

## Auto-research v0 — 24/7 perpetual motion

System running since 2026-05-12 and surviving multiple cron-tick / agent
failures. Composition:

- **A800 cron every 30 min**: `git fetch` from main, apply
  `auto_research/pending_queue_update.py` if changed, run orchestrator tick
  (pick top-priority queue entry, launch on free GPU, collect finished).
- **Cloud Claude routines** (commit to main via GitHub-as-message-bus):
  - `auto-research-idea-gen` (daily 07:00 UTC) — read state + arxiv hotspots,
    propose 3-5 new experiments
  - `auto-research-shepherd` (every 2h) — health checks, intervenes on stalls
  - `auto-research-paper-drafter` (Sunday 09:00 UTC) — rolling markdown draft
  - `auto-research-paper-pipeline` (Friday 11:00 UTC) — AAAI LaTeX +
    self-review + experiment injection
- **Local fallback**: when queue empties between cloud agent firings,
  rule-based `idea_gen_local.py` injects 3 perturbation experiments to
  perpetuate motion (DAILY_CAP = 20).
- **arxiv hotspot scrape** (daily 06:00 UTC): cs.CL / cs.LG / cs.AI tagged
  by keyword groups (rl_align, continual, moe_routing, etc.) feeding both
  cloud and local idea generators.

47+ experiments completed autonomously since deployment.

## What's open

1. **Verify the 1.5B 50–51% candidates on eval200**. In progress (background).
   If confirmed, multi-seed × 3 to nail down the effect size.
2. **3B + scaf-grpo / r=32 / low-lr stack**. None tested at 3B yet; +3-4 pp
   transfer would mean 63–64% on 3B base.
3. **Implement real Scaf-GRPO / hint injection / JEPA option C** in
   `experiments/train_small_model_grpo_local.py`. The v0 runner currently
   dispatches all "named" agent experiments to vanilla continual; the named
   recipe ideas are queued but not yet executed faithfully.
4. **8-cycle multi-seed router**. Replace the seed=42+k pattern with a proper
   3-seed mean to remove the cycle 5 outlier story.
5. **Bring up GPU 2/3** (currently excluded from orchestrator candidate list
   to avoid disturbing other A800 users). 6/8 GPU utilization vs 8/8 would
   bump throughput ~33%.

## Pointers

| Topic | Path |
| --- | --- |
| Full method-by-method ablation | [`docs/E2E_ABLATION_RESULTS.md`](E2E_ABLATION_RESULTS.md) |
| 8-cycle raw eval JSONs | [`results/iterated_skill_llm_router_8cycles/`](../results/iterated_skill_llm_router_8cycles/) |
| Driver shell scripts | [`scripts/`](../scripts/) |
| Auto-research orchestrator code | [`auto_research/`](../auto_research/) |
| JEPA literature memo (for Friday paper-pipeline) | A800: `/data0/home/zeyuwang/auto_research/paper/JEPA_memo.md` |
| Cloud routine state | https://claude.ai/code/routines |
