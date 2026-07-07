#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-07 (EXP-110, EXP-111).

A800 connectivity: offline since 2026-05-14 (day ~54). Apply when restored.

Prior patches to apply before this one (daily-patch chain):
    python3 auto_research/pending_queue_update.py
    python3 auto_research/pending_queue_update_2026_06_06.py            # EXP-054, EXP-055
    python3 auto_research/pending_queue_update_2026_06_07.py            # EXP-056, EXP-057
    python3 auto_research/pending_queue_update_2026_06_08.py            # EXP-058, EXP-059
    python3 auto_research/pending_queue_update_2026_06_10.py            # EXP-060, EXP-061
    python3 auto_research/pending_queue_update_2026_06_11.py            # EXP-062, EXP-063
    python3 auto_research/pending_queue_update_2026_06_12.py            # EXP-064, EXP-065
    python3 auto_research/pending_queue_update_2026_06_12_v2.py         # EXP-066, EXP-067
    python3 auto_research/pending_queue_update_2026_06_14.py            # EXP-068, EXP-069
    python3 auto_research/pending_queue_update_2026_06_15.py            # EXP-070, EXP-071
    python3 auto_research/pending_queue_update_2026_06_16.py            # EXP-072, EXP-073
    python3 auto_research/pending_queue_update_2026_06_17.py            # EXP-074, EXP-075
    python3 auto_research/pending_queue_update_2026_06_18.py            # EXP-076, EXP-077
    python3 auto_research/pending_queue_update_2026_06_19.py            # EXP-078, EXP-079
    python3 auto_research/pending_queue_update_2026_06_19_paper.py      # EXP-080, EXP-081
    python3 auto_research/pending_queue_update_2026_06_21.py            # EXP-082, EXP-083
    python3 auto_research/pending_queue_update_2026_06_22.py            # EXP-084, EXP-085
    python3 auto_research/pending_queue_update_2026_06_23.py            # EXP-086, EXP-087
    python3 auto_research/pending_queue_update_2026_06_25.py            # EXP-088, EXP-089
    python3 auto_research/pending_queue_update_2026_06_26.py            # EXP-090, EXP-091
    python3 auto_research/pending_queue_update_2026_06_27.py            # EXP-092, EXP-093
    python3 auto_research/pending_queue_update_2026_06_28.py            # EXP-094, EXP-095
    python3 auto_research/pending_queue_update_2026_06_29.py            # EXP-096, EXP-097
    python3 auto_research/pending_queue_update_2026_06_30.py            # EXP-098, EXP-099
    python3 auto_research/pending_queue_update_2026_07_01.py            # EXP-100, EXP-101
    python3 auto_research/pending_queue_update_2026_07_02.py            # EXP-102, EXP-103
    python3 auto_research/pending_queue_update_2026_07_03.py            # EXP-104, EXP-105
    python3 auto_research/pending_queue_update_2026_07_04.py            # EXP-106, EXP-107
    python3 auto_research/pending_queue_update_2026_07_05.py            # EXP-108, EXP-109

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_07.py            # EXP-110, EXP-111

Queue was ~113 pending on 2026-07-05 (+2 added: EXP-108, EXP-109).
Queue cap applied: >20 → max 2 new experiments.

Data motivating today's proposals
----------------------------------
results/e2e_4cyc_gpt55/cycle_3/e2e_ablation_summary.json (unchanged — A800 offline):
    HumanEval 4-cycle:
      large (always-large):        task_pass=96.34%, cost_vs_large=100%
      skills (always-small+proc):  task_pass=75.61%, cost_vs_large=10%
      router (logistic, cycle 3):  task_pass=92.68%, routing_acc=92.68%,
                                   cost_vs_large=27.56%, fallback=6.10%
      full (router+GRPO):          task_pass=92.68% ← SAME as router (Full=Router bottleneck)
    Root problem A: ACR=52.4% — 43/82 GRPO groups have zero within-group reward
    variance → zero gradient → Full = Router.
    Root problem B: Skills gap: 75.61% vs large 96.34% — 20.7pp gap.

arxiv:2607.01480 — Procedural Memory Distillation: Online Reflection for
    Self-Improving Language Models (July 1, 2026):
    RLVR (GRPO/SDPO) evaluates each rollout against a verifier and updates the policy
    from that episode-level signal. However, the richer procedural information in
    the rollout is rarely retained or reused. PMD converts cross-episode signals into
    reusable procedural memory and distills it into the policy's weights during training.
    The memory operates at three levels: raw trajectories, self-reflected strategies,
    and higher-level behavioral patterns that recur across problems. A memory-conditioned
    self-teacher (the same model, conditioned on accumulated procedure) supervises the
    student on its own rollouts, enabling the student to progressively internalize
    procedural knowledge within its parameters.
    Key principle: co-evolution — policy generates rollouts → updates memory →
    memory shapes the supervision signal → policy updates → next rollout.
    Results: PMD improves over SDPO by 3.8-5.5% on SCIKNOWEVAL and 7.9-13.6% on
    LIVECODEBENCH (Qwen3-8B and OLMo3-Instruct-7B). The code reasoning improvement
    (+7.9-13.6% LIVECODEBENCH) is the most directly relevant to our HumanEval setting.

    RELEVANCE TO OUR PIPELINE:
    Our SkillBook already embodies a procedure injection mechanism: the distilled
    procedure (extracted from cycle traces) is prepended to the small model's prompt
    during SFT and inference. However, this procedure is:
      (a) frozen between GRPO epochs — it does not update as GRPO improves the policy;
      (b) extracted only once per cycle (from the collected traces), not from the
          richer, iteratively improving GRPO rollouts mid-cycle.
    PMD suggests making procedure injection a LIVE TRAINING SIGNAL during GRPO:
      — After each mini-batch, identify passing rollouts (reward=1 under verifier).
      — Extract a "skill hint" from each passing rollout: the first 128 tokens of
        each unique passing solution body form an epoch-level procedure buffer.
      — For the next batch: prepend the buffer as `# Verified strategy:\n{hint}\n\n`
        to the task prompt, so GRPO rollouts benefit from accumulated procedure.
      — At the end of each GRPO epoch, use the procedure buffer to create an
        auxiliary teacher loss L_aux = mean_CE(π_θ(t | proc_buffer + problem))
        over the sampled passing rollout tokens, weighted by λ_pmd=0.05, added to
        the GRPO loss. This anchors the policy to its own successful strategies.
    The memory is ephemeral (within-cycle only), memory-free at inference (procedure
    is discarded after GRPO; the inference prompt uses the SkillBook from Phase 2).
    This avoids test-time memory overhead and is compatible with all other EXPs.

    DISTINCTION FROM EXISTING EXPs:
    - EXP-109 (MOPD): peer context injected into SFT training only, at fixed traces.
      PMD (EXP-110): procedure extracted live from GRPO rollouts; injected into GRPO
      training itself (not SFT); creates a feedback loop within Phase 3b.
    - EXP-107 (DemoPSD): data filtering for SFT (which traces to distill).
      PMD (EXP-110): procedure injection during GRPO (how to condition rollout prompts).
    - SkillBook (existing): procedure extracted post-cycle, used at inference.
      PMD (EXP-110): procedure extracted within-epoch, used during training.
    All three are orthogonal and could be combined.

    IMPLEMENTATION: ~60 lines in grpo_train_simple.py.
      (1) Maintain a procedure_buffer dict: {task_prefix: hint_tokens} (max 50 entries,
          LRU eviction).
      (2) Before sampling rollouts for each batch: for tasks in the batch that have an
          entry in procedure_buffer, prepend `# Verified strategy:\n{hint}\n---\n\n`
          to the input prompt. This is inference-time conditioning on the procedure.
      (3) After evaluating rollouts (scoring rewards): for each task with reward=1,
          extract the first 128 tokens of the rollout body and upsert into
          procedure_buffer[task_prefix].
      (4) At end of each epoch: compute L_aux loss over all tasks with procedure_buffer
          entries: L_aux = CE(π_θ(passing_rollout_tokens | proc_prompt + problem)).
          Add λ_pmd * L_aux to the total loss. (One forward pass over ~50 task examples.)
      (5) Track: procedure_buffer fill rate, L_aux over epochs, and whether adding the
          procedure prefix to rollout prompts raises the within-batch pass rate
          (i.e., fewer all-fail groups).
    No external API calls needed. The "self-teacher" uses the current policy's own
    successful rollouts as supervision, matching PMD's memory-conditioned self-teacher
    design.

arxiv:2606.04560 — Rollout-Level Advantage-Prioritized Experience Replay for GRPO
    (June 3, 2026; Seoul National University):
    GRPO discards each rollout after a single gradient update, even though generating
    rollouts (the slow part) vastly exceeds the cost of additional backward passes.
    Naive replay destabilizes training because LLM policies drift quickly, making
    stored rollouts stale. The paper proposes three safeguards:
      1. Rollout-level granularity: store individual rollouts, not whole groups, in
         the replay buffer, so fresh and replayed rollouts can be mixed within a batch.
      2. Age eviction: any rollout older than tau_max gradient steps is evicted,
         bounding the maximum staleness (default tau_max=4).
      3. Advantage-magnitude prioritization: replay rollouts with highest |advantage|
         more frequently, concentrating compute on the most informative examples.
      4. Off-policy correction: multiply replayed rollout loss by an importance weight
         clip(π_θ(rollout)/π_old(rollout), 1-ε, 1+ε) to correct for policy drift.
    The paper shows that the combination is safe under fast LLM policy drift, achieving
    throughput improvements (same-time higher pass rate) vs. standard GRPO by reusing
    compute on high-advantage rollouts.

    RELEVANCE TO OUR PIPELINE:
    Our ACR=52.4% means 43/82 GRPO groups produce zero-advantage rollouts in a typical
    batch. The 39/82 non-collapsed groups generate rollouts with non-zero advantages;
    these are currently used once and discarded. With advantage-prioritized replay:
      — Zero-advantage rollouts from collapsed groups: |advantage|=0 → lowest priority
        → effectively never replayed → their wasted compute is not amplified.
      — High-advantage rollouts from non-collapsed groups: |advantage|>0 → replayed
        up to tau_max=4 times → the informative gradient signal from the 47.6% of tasks
        is reused, compensating for the wasted compute from the 52.4% zero-variance tasks.
    Expected effect: even without reducing ACR, the informative groups contribute more
    gradient per wall-clock hour. Training becomes effectively denser without requiring
    more rollout generation (the expensive part).

    DISTINCTION FROM EXISTING EXPs:
    - EXP-104 (Selective-Group GRPO): MASKS zero-variance groups (skips their backward).
      RAPG (EXP-111): does not mask zero-variance groups; instead, AMPLIFIES non-zero
      groups by replaying their high-advantage rollouts.
    - EXP-108 (Sign Advantage): changes advantage FORMULA so zero-variance groups get
      A=±1 instead of A=0. Targets ACR via formula change.
      RAPG (EXP-111): leaves the advantage formula unchanged; targets throughput via
      replay. RAPG is orthogonal to EXP-108 and combinable: Sign advantage provides
      non-zero gradient from zero-variance groups; replay amplifies high-advantage
      rollouts from non-collapsed groups. Could be combined as EXP-112.
    - EXP-106 (SRPO): routes failed rollouts to an SDPO loss head (uses π_ref).
      RAPG (EXP-111): replays high-advantage rollouts (both passed and failed) with
      age eviction, no extra loss head, no π_ref needed.
    All four EXPs target the Full=Router bottleneck via different mechanisms; results
    across them reveal which bottleneck component (formula degeneracy, routing,
    information reuse) is the binding constraint.

    IMPLEMENTATION: ~80 lines in grpo_train_simple.py.
      (1) Define ReplayBuffer class: max-heap of (−|advantage|, step, rollout_dict).
          (Python heapq on a list of tuples; ~20 lines.)
      (2) After each batch's advantage computation: push all rollouts with |advantage|>0
          to the heap, tagging each with the current step number.
      (3) Eviction: on each push, pop entries where step < current_step − tau_max.
          (Simple loop over the heap's tail after heapify.)
      (4) Each batch composition:
          n_fresh = batch_size * (1 - TAU_REPLAY)   # e.g., TAU_REPLAY=0.3
          n_replay = batch_size * TAU_REPLAY
          Concatenate fresh rollouts with n_replay samples drawn from the heap
          (softmax-weighted by |advantage|, without replacement).
      (5) Off-policy weight: for each replayed rollout, token-level importance weight
          = clip(exp(log_π_θ(tokens|prompt) - log_π_old(tokens|prompt)), 0.8, 1.2).
          π_old log-probs are stored when the rollout is first added to the buffer.
      (6) Track: replay buffer size per step, mean advantage of replayed vs. fresh
          rollouts, training loss from fresh vs. replayed components, pass@1 curve.
      Constants: TAU_REPLAY=0.3, tau_max=4, importance_clip=(0.8, 1.2).

AAAI 2027 context (deadline 2026-08-15, 39 days):
    EXP-110 (PMD, today): skills arm improvement + GRPO improvement via procedure
    co-evolution. PMD is architecturally novel relative to all queued EXPs and directly
    extends the SkillBook narrative ("procedures co-evolve with the policy, not just the
    SFT data"). A 3.8-5.5% improvement on reasoning tasks translates to ~3pp on HumanEval
    — if it pushes skills arm to >78%, the skills-vs-large gap narrative strengthens.
    Most importantly, PMD is a natural connection between our SkillBook (§3) and the GRPO
    phase (§4): the paper is published 6 days ago, providing a fresh citation.
    EXP-111 (RAPG, today): efficiency-side ACR mitigation. The argument is simple and
    printable in 2 sentences: our zero-variance groups waste compute; replay reuses
    the non-zero groups proportionally. If it shows +1pp pass@1 with no other change,
    it motivates a combined EXP-112 (Sign+RAPG) as the next step.
"""
import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = (
    "/data0/home/zeyuwang/router-skills-evolve-data/humaneval/HumanEval.jsonl.gz"
)

NEW_EXPERIMENTS = [
    {
        "id": "exp_2026_07_07_001_pmd_procedural_memory_grpo_skills_humaneval",
        "priority": 7,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2607.01480 — Procedural Memory Distillation: Online Reflection "
            "for Self-Improving Language Models (July 1, 2026). "
            ""
            "CORE IDEA: PMD converts cross-episode RLVR signals into reusable "
            "procedural memory (strategy + lesson abstractions from passing rollouts) "
            "and distills it back into the policy via a memory-conditioned self-teacher "
            "loss during training. The key design is co-evolution: "
            "  rollouts → memory update → memory-conditioned prompts for next rollouts "
            "  → memory-conditioned teacher loss → policy update → next rollouts. "
            "No external model needed — the policy itself acts as both student and teacher. "
            "Results: +7.9-13.6% on LIVECODEBENCH vs SDPO (Qwen3-8B, OLMo3-7B). "
            ""
            "RELEVANCE TO OUR PIPELINE: "
            "Our SkillBook injects a fixed per-cycle procedure at SFT/inference time "
            "but does NOT update during GRPO training. PMD reveals a missed opportunity: "
            "the passing rollouts mid-GRPO encode the current best solving strategies, "
            "but this information is wasted rather than fed back as a training scaffold. "
            "EXP-110 adapts PMD to our grpo_continual kind: "
            "  (1) Maintain an epoch-level procedure_buffer populated from passing rollouts. "
            "  (2) Prepend the buffer as `# Verified strategy:\\n{hint}\\n---\\n\\n` "
            "      to prompts for subsequent rollout batches within the same epoch. "
            "  (3) Add λ_pmd=0.05 auxiliary CE loss over passing-rollout tokens conditioned "
            "      on the procedure buffer, as the memory-conditioned self-teacher signal. "
            "Expected: procedure_buffer prefix reduces ACR-equivalent groups mid-epoch "
            "(tasks that once failed may pass when prompted with a verified strategy), "
            "and the auxiliary loss anchors the policy to its own successful strategies "
            "without a reference model. "
            ""
            "DISTINCTION FROM EXISTING EXPs: "
            "EXP-109 (MOPD): peer context in SFT data; no feedback loop within GRPO. "
            "EXP-107 (DemoPSD): data filtering for SFT selection. "
            "EXP-110 (PMD): live procedure extraction+injection within GRPO Phase 3b. "
            "All three are orthogonal and could be stacked. "
            ""
            "AAAI NOTE: PMD (2607.01480) is a July 2026 preprint — fresh citation "
            "that directly extends the SkillBook narrative ('procedures co-evolve with "
            "the policy during GRPO, not just between cycles'). "
        ),
        "spec": {
            "pipeline": "humaneval_pmd_procedure_memory_grpo",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 1,
            "grpo_algo": "dapo_with_pmd_aux",
            "pmd_procedure_buffer_max_entries": 50,
            "pmd_hint_tokens_per_task": 128,
            "pmd_hint_prepend_format": "# Verified strategy:\n{hint}\n---\n\n",
            "pmd_aux_loss_weight": 0.05,
            "pmd_aux_loss_scope": "passing_rollout_tokens_conditioned_on_proc_prompt",
            "n_generations": 8,
            "grpo_temperature": 1.0,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (DAPO binary reference: 92.68% HE, ACR=52.4%)",
                "e2e_4cyc_gpt55 cycle_3 skills arm (75.61% HE)",
                "arxiv:2607.01480 (PMD +7.9-13.6% LIVECODEBENCH vs SDPO, Qwen3-8B)",
                "EXP-109 (MOPD: peer context in SFT — complementary, not competing)",
            ],
            "implementation_note": (
                "In grpo_train_simple.py: "
                "(1) Add procedure_buffer = {} (task_prefix → hint_str, max 50 LRU). "
                "(2) Before sampling each rollout batch: for any task present in "
                "    procedure_buffer, prepend "
                "    `# Verified strategy:\\n{buffer[task]}\\n---\\n\\n` to its prompt. "
                "(3) After reward evaluation: for tasks with reward=1, "
                "    extract first 128 decoded tokens of the rollout body and upsert "
                "    into procedure_buffer[task_prefix]. "
                "(4) Once per epoch, compute L_aux = mean cross-entropy over all "
                "    procedure_buffer entries: forward(proc_prompt + problem) → "
                "    CE loss on passing rollout tokens. Add 0.05 * L_aux to total loss. "
                "(5) Track procedure_buffer fill_rate (# entries / 50) and within-batch "
                "    pass_rate_with_hint vs pass_rate_without_hint. "
                "~60 lines total. No external API calls. No extra rollout generation. "
                "The auxiliary forward pass over ~50 tasks per epoch adds <5% GPU overhead."
            ),
            "primary_metrics": [
                "humaneval_pass_at_1_post_1cycle",
                "grpo_acr_equivalent",
                "grpo_procedure_buffer_fill_rate",
                "grpo_within_batch_pass_rate_with_hint",
                "grpo_within_batch_pass_rate_without_hint",
                "grpo_aux_loss_curve",
                "grpo_fraction_nonzero_advantage_groups",
                "training_loss_curve",
                "output_length_mean",
            ],
            "target_outcome": (
                "pass@1 >= 93.68% (Full > Router, +1pp) or skills arm >= 77% (+1.4pp). "
                "procedure_buffer fill_rate > 0.6 by epoch 2 (most tasks have a "
                "verified strategy hint available). "
                "within-batch pass_rate_with_hint > pass_rate_without_hint (procedure "
                "prefix reduces ACR-equivalent groups mid-epoch). "
                "L_aux converges within 3 epochs (policy internalizes procedural memory). "
            ),
            "gpu": "auto",
            "estimated_gpu_hours": 2.0,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_07_07_002_rapg_rollout_advantage_prioritized_replay_grpo_humaneval",
        "priority": 6,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2606.04560 — Rollout-Level Advantage-Prioritized Experience Replay "
            "for GRPO (June 3, 2026; Seoul National University). "
            ""
            "CORE IDEA: GRPO discards each rollout after one gradient update. Generating "
            "rollouts (autoregressive decode) is the bottleneck — backward passes are cheap. "
            "The paper proposes a replay buffer for individual rollouts with: "
            "  (a) advantage-magnitude prioritization (highest |A| replayed most often), "
            "  (b) age eviction (any rollout older than tau_max steps is dropped), "
            "  (c) off-policy importance-weight correction (clip π_θ/π_old to [0.8, 1.2]). "
            "This combination is safe under fast LLM policy drift. "
            ""
            "RELEVANCE TO OUR PIPELINE: "
            "ACR=52.4% means 43/82 groups have |A|=0 → bottom of the priority heap → "
            "effectively never replayed. The 39/82 non-collapsed groups produce high-|A| "
            "rollouts that are currently used once. With τ=0.3 replay fraction and "
            "tau_max=4: each high-advantage rollout is reused ~1.7x on average within "
            "its staleness window. Net effect: 47.6% * 1.7x ≈ 81% of a fully-non-collapsed "
            "GRPO run's compute efficiency, from rollout generation cost savings alone. "
            ""
            "DISTINCTION FROM OTHER ACR FIXES: "
            "EXP-104 (Selective-Group GRPO): hard-masks zero-A groups. "
            "EXP-108 (Sign Advantage): changes A formula to ±1 for all groups. "
            "EXP-111 (RAPG): leaves both the mask and formula unchanged; reuses the "
            "non-zero-A rollouts more. These are orthogonal: EXP-108 + EXP-111 together "
            "would have Sign advantage (non-zero A for all groups) AND replay (reuse "
            "the highest-A rollouts). If both show independent gains, EXP-112 = combined. "
            ""
            "AAAI NOTE: Separate from the ACR narrative, RAPG presents a practical "
            "training efficiency result ('same performance in 0.7x rollout generation "
            "budget') that is relevant for the 'practical deployment' subsection in §5. "
        ),
        "spec": {
            "pipeline": "humaneval_rapg_rollout_replay_grpo",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 1,
            "grpo_algo": "dapo_with_rollout_replay",
            "replay_tau": 0.3,
            "replay_tau_max_steps": 4,
            "replay_importance_clip": [0.8, 1.2],
            "replay_buffer_max_size": 200,
            "replay_priority": "abs_advantage",
            "n_generations": 8,
            "grpo_temperature": 1.0,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (DAPO binary reference: 92.68% HE, ACR=52.4%)",
                "arxiv:2606.04560 (RAPG: replay improves throughput under fast policy drift)",
                "EXP-104 (selective-group GRPO: orthogonal — masks zeros; RAPG amplifies non-zeros)",
                "EXP-108 (sign advantage: orthogonal — changes A formula; RAPG changes reuse)",
            ],
            "implementation_note": (
                "In grpo_train_simple.py: "
                "(1) Define ReplayBuffer: Python list as max-heap of "
                "    (-abs_advantage, step, log_probs_old, rollout_tokens, prompt_ids). "
                "    heapq.heappush / heapq.nsmallest for sampling. Max size 200. "
                "(2) After each batch's advantage computation: "
                "    for each rollout with abs(advantage)>0 and age<tau_max, heappush. "
                "    Evict: iterate heap and remove entries where step < current_step - tau_max. "
                "(3) Batch composition each step: "
                "    n_fresh = int(batch_size * (1 - 0.3)) fresh rollouts (standard GRPO). "
                "    n_replay = batch_size - n_fresh rollouts sampled from heap "
                "    (softmax over -heap[i][0], i.e. proportional to |advantage|). "
                "(4) For replayed rollouts, compute token-level importance weight: "
                "    w = clip(exp(log_π_θ - log_π_old), 0.8, 1.2). "
                "    Multiply replayed rollout's per-token loss by w before averaging. "
                "(5) Track: replay_buffer_size, advantage_distribution_replay vs_fresh, "
                "    rollout_generation_count_per_step, training_wall_clock_per_step. "
                "~80 lines total. No extra rollout generation. No new models. "
                "Store log_probs_old at rollout time for importance weight computation."
            ),
            "primary_metrics": [
                "humaneval_pass_at_1_post_1cycle",
                "grpo_acr_equivalent",
                "grpo_replay_buffer_mean_size",
                "grpo_replay_buffer_advantage_distribution",
                "grpo_fresh_vs_replay_loss_ratio",
                "grpo_rollout_generation_count_per_pass_at_1_gain",
                "training_wall_clock_minutes",
                "training_loss_curve",
                "output_length_mean",
            ],
            "target_outcome": (
                "pass@1 >= 92.68% with fewer rollout generation calls vs. DAPO baseline "
                "(same or better quality, lower rollout compute). "
                "replay_buffer_mean_size > 60 by step 50 (buffer fills with high-A rollouts). "
                "advantage_distribution_replay > advantage_distribution_fresh "
                "(replay successfully over-represents informative rollouts). "
                "Stretch: pass@1 >= 93.5% (+0.8pp) if high-A rollouts used 1.7x carry "
                "disproportionate gradient signal. "
            ),
            "gpu": "auto",
            "estimated_gpu_hours": 1.75,
        },
        "gpu": "auto",
    },
]


def main() -> None:
    if not STATE_PATH.exists():
        print(f"ERROR: {STATE_PATH} not found. Is the A800 mounted?")
        raise SystemExit(1)

    with STATE_PATH.open() as f:
        state = json.load(f)

    existing_ids = {e["id"] for e in state.get("queue", [])}
    existing_ids |= {e["id"] for e in state.get("history", [])}

    added = []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] in existing_ids:
            print(f"  SKIP (duplicate): {exp['id']}")
            continue
        state["queue"].append(exp)
        added.append(exp["id"])
        print(f"  QUEUED: {exp['id']}  (priority={exp['priority']}, "
              f"kind={exp['kind']})")

    if not added:
        print("No new experiments added (all duplicates).")
        return

    tmp = STATE_PATH.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(STATE_PATH)
    print(f"\nSaved {STATE_PATH}. Added {len(added)} experiment(s): {added}")


if __name__ == "__main__":
    main()
