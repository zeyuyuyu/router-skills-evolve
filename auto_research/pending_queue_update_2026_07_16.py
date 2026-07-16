#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-16 (EXP-118, EXP-119).

A800 connectivity: offline since 2026-05-14 (day ~63). Apply when restored.

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
    python3 auto_research/pending_queue_update_2026_07_03_paper.py      # EXP-106, EXP-107
    python3 auto_research/pending_queue_update_2026_07_05.py            # EXP-108, EXP-109
    python3 auto_research/pending_queue_update_2026_07_10_paper.py      # (paper-pipeline EXPs)
    python3 auto_research/pending_queue_update_2026_07_11.py            # EXP-110, EXP-111
    python3 auto_research/pending_queue_update_2026_07_12.py            # EXP-112, EXP-113
    python3 auto_research/pending_queue_update_2026_07_13.py            # EXP-114, EXP-115
    python3 auto_research/pending_queue_update_2026_07_14.py            # EXP-116, EXP-117

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_16.py            # EXP-118, EXP-119

Queue was ~127 pending on 2026-07-14 (+2 added: EXP-116, EXP-117).
Queue cap applied: >20 → max 2 new experiments.

Data motivating today's proposals
----------------------------------
results/e2e_4cyc_gpt55/cycle_3/e2e_ablation_summary.json (unchanged — A800 offline day 63):
    HumanEval 4-cycle:
      large (always-large):        task_pass=96.34%, cost_vs_large=100%
      skills (always-small+proc):  task_pass=75.61%, cost_vs_large=10%
      router (logistic, cycle 3):  task_pass=92.68%, routing_acc=92.68%,
                                   cost_vs_large=27.56%, fallback=6.10%
      full (router+GRPO):          task_pass=92.68% ← SAME as router
    Root problems:
      (A) ACR=52.4% — 43/82 GRPO groups have zero within-group reward variance
          → zero gradient → Full = Router.
      (B) Non-collapsed groups (47.6%) = ALL GRPO learning signal; each must be
          used as efficiently as possible.
      (C) Skills gap: 75.61% vs large 96.34% — 20.7pp gap.

AAAI 2027 deadline: 2026-08-15 (30 days from today).
A800 offline since 2026-05-14 (day 63).

arxiv:2606.04560 — Rollout-Level Advantage-Prioritized Experience Replay for GRPO (June 2026):

    PROBLEM THEY SOLVE:
    Standard GRPO discards all rollout data after each gradient step ("generate-
    and-discard" paradigm). When a training group has zero within-group reward
    variance (all K rollouts succeed or fail with the same reward), the entire
    group contributes zero gradient. The rollout data from MORE TRACTABLE earlier
    training steps — when the same prompt produced mixed pass/fail outcomes —
    is thrown away, even though it contained the highest-advantage signals for
    exactly those prompts.

    THEIR SOLUTION (RAPER):
    Maintain a priority replay buffer of INDIVIDUAL rollouts (not whole groups),
    keyed by (prompt_id, rollout_hash), with priority = |advantage| at the time
    of generation. When a current-step group has zero within-group variance
    (collapsed), inject top-K historical rollouts for that prompt from the buffer
    to form a mixed group that restores gradient signal. Staleness control:
    age eviction (rollout expires after max_age=2 training epochs); on-policy
    dominance: 70% fresh rollouts / 30% replayed rollouts in each mixed group.
    The buffer stores (rollout_tokens, logprob_at_generation, reward, age).
    For the group advantage, they normalize over the COMBINED fresh + replayed
    distribution (not separately), preserving comparability. Evaluated on Qwen2.5
    (1.5B, 7B) and Llama-3.1-8B on MATH and code: +2.1–4.3pp over GRPO baseline.

    RELEVANCE TO US:
    Our ACR=52.4% means 43/82 GRPO groups are collapsed AT EACH STEP. These 43
    groups produce ZERO gradient. For many of these prompts, earlier in training
    (cycle 0, or the first few steps of cycle k), the policy was less capable →
    some rollouts passed, some failed → non-zero advantage → information exists.
    RAPER retrieves exactly those high-|advantage| historical rollouts and injects
    them back into current collapsed groups. The mechanism is orthogonal to:
      - TACO (EXP-116): per-token credit suppression in NON-COLLAPSED winning
        rollouts — RAPER targets COLLAPSED groups.
      - AVSPO (EXP-094): generates SYNTHETIC rollouts for collapsed groups —
        RAPER uses REAL historical rollouts with known rewards.
      - Sign advantage (EXP-108): changes advantage FORMULA for all-fail groups —
        RAPER changes group COMPOSITION by injecting old data.
      - RLCSD (EXP-114): adds contrastive distillation loss — RAPER changes
        rollout sampling.
    Implementation: replay buffer as in-memory dict (prompt_id → deque of
    rollout records); ~50 lines in grpo_train_simple.py's training loop; no
    external storage needed for K=8, 82 prompts, max_age=2 epochs.

arxiv:2606.06058 — MDP-GRPO: Stabilized Group Relative Policy Optimization for
    Multi-Constraint Instruction Following (June 2026):

    PROBLEM THEY SOLVE:
    Z-score group normalization in GRPO has three pathologies formalized in this
    paper: (1) LOW-VARIANCE AMPLIFICATION — when within-group reward variance is
    small but non-zero, z-score inflates the advantage signal beyond its
    information content (noisy gradient); (2) MEAN-CENTERING BLINDNESS — when all
    rewards are the same positive value, the mean-centered advantage is zero even
    though the model is systematically correct (should reinforce); (3) ZERO-
    VARIANCE COLLAPSE — the main problem we face: all K rollouts have identical
    reward → advantage = 0 → zero gradient.

    THEIR SOLUTION (MDP-GRPO):
    Four components, three of which apply to us:
    (a) MULTI-TEMPERATURE SAMPLING: within each group of K rollouts, generate
        K/2 = 4 rollouts at T=0.7 (conservative / likely to pass) and K/2 = 4
        rollouts at T=1.2 (exploratory / diverse). The mixed-temperature group
        has higher within-group reward variance → lower ACR. For a task with
        pass_rate p, the probability of all-fail under mixed-temperature sampling
        is LOWER than under uniform T=1.0: at T=0.7, P(pass) ≈ p^0.85; at
        T=1.2, P(pass) ≈ p^1.15. For p=0.15, uniform K=8: P(all-fail) = 0.85^8
        ≈ 27%; mixed T: P(all-fail) = (0.85^0.85 * 0.85^1.15)^4... = lower.
        Practically: the mixed group is more likely to have ≥1 passing rollout.
    (b) DUAL-ANCHOR NORMALIZATION: for groups that are STILL all-fail or all-pass
        after multi-temperature sampling, instead of within-group mean/std (which
        are 0/0), use cross-batch moving-average mean/std as the anchor:
        A_i = (r_i - EMA_mean) / (EMA_std + ε), where EMA is over the last 32
        training steps. This provides a meaningful gradient direction even for
        uniform-reward groups.
    (c) ASYMMETRIC KL: upweight KL penalty for policy moves AWAY from the prior
        on groups with low-confidence rewards (all-fail or near-uniform).
    Evaluated on Qwen2.5 (1.5B, 7B) on instruction following + code: formalizes
    the pathologies and shows +1.8–3.1pp vs GRPO across tasks.

    RELEVANCE TO US:
    MDP-GRPO targets ACR=52.4% via TWO mechanisms: (a) multi-temperature sampling
    reduces the probability of all-fail groups at the SAMPLING stage; (b) dual-
    anchor normalization provides a gradient direction for groups that STILL
    collapse after (a). Together, they attack both ends of the collapsed-group
    problem. Distinct from RAPER (no replay buffer, changes sampling and
    normalization not group composition), AVSPO (no synthetic samples, uses
    within-group temperature variation), sign advantage (not formula change, uses
    cross-batch anchor). The key novelty of dual-anchor advantage (using EMA
    statistics across batches as the normalization anchor) is NOT present in any
    prior queued experiment. Implementation: ~30 lines — replace the single
    client.generate(..., temperature=GRPO_TEMPERATURE) with two calls at T=0.7
    and T=1.2, then replace the per-group mean/std with EMA-based stats for
    zero-variance groups.
"""

import json
import os
import tempfile
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")

_HE_EVAL = (
    "/data0/home/zeyuwang/router-skills-evolve-data/humaneval/HumanEval.jsonl.gz"
)
_CYCLE_DIR = "/data0/home/zeyuwang/router-skills-evolve-data/e2e_4cyc_gpt55"

NEW_EXPERIMENTS = [
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-118: RAPER — Rollout-Level Advantage-Prioritized Experience Replay
    #          for GRPO (HumanEval)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_16_001_raper_rollout_priority_replay_grpo_humaneval",
        "priority": 8,
        "gpu": "auto",
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2606.04560 — Rollout-Level Advantage-Prioritized Experience Replay "
            "for GRPO (RAPER, June 2026). Our primary blocking issue is ACR=52.4%: "
            "43/82 GRPO groups have zero within-group reward variance → zero gradient "
            "→ Full = Router (92.68% in both arms). Standard GRPO discards all rollout "
            "data after each step. Earlier in training, many of the currently-collapsed "
            "prompts produced non-zero advantages (some rollouts passed, some failed). "
            "RAPER retrieves those historical high-|advantage| rollouts from a priority "
            "buffer and injects them into current collapsed groups, restoring gradient "
            "signal. Key properties: rollout-level (not group-level) priority keyed by "
            "|advantage|; staleness bounded by max_age=2 epochs; fresh/replay ratio "
            "70/30 preserves on-policy dominance; combined group normalization maintains "
            "advantage comparability. Distinct from AVSPO (synthetic, not historical), "
            "sign advantage (formula, not composition), RLCSD (contrastive distillation, "
            "not replay), TACO (per-token credit, targets non-collapsed groups). "
            "Target: ACR 52.4% → ≤35%, Full arm 92.68% → ≥93.5%. AAAI 2027 deadline "
            "2026-08-15: 30 days."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 4,
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "raper_enabled": True,
            "raper_buffer_size_per_prompt": 16,
            "raper_max_age_epochs": 2,
            "raper_fresh_replay_ratio": 0.70,
            "raper_priority_metric": "abs_advantage",
            "raper_injection_condition": "zero_within_group_variance",
            "raper_combined_normalization": True,
            "n_generations": 8,
            "grpo_temperature": 1.0,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "analysis": [
                "acr_per_cycle_raper_vs_baseline",
                "full_arm_pass_at_1_improvement",
                "replay_injection_rate_per_epoch",
                "advantage_magnitude_dist_fresh_vs_replayed",
                "buffer_hit_rate_by_task_difficulty",
            ],
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (baseline): full 92.68%, router 92.68%",
                "EXP-094 (AVSPO): synthetic collapsed-group injection",
                "EXP-108 (sign advantage): formula fix for all-fail groups",
                "EXP-116 (TACO): per-token credit in non-collapsed groups",
                "arxiv:2606.04560 (RAPER): +2.1–4.3pp on math/code vs GRPO",
            ],
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
            ],
            "implementation_note": (
                "In grpo_train_simple.py: "
                "Initialize replay_buffer = defaultdict(deque) (key = prompt_id). "
                "After each group's rollout generation: store (tokens, logprob, reward, "
                "epoch, step) for each rollout in replay_buffer[prompt_id], capped at "
                "raper_buffer_size_per_prompt=16 by evicting oldest first. "
                "Before advantage computation: check if all K current rollouts have same "
                "reward (all-pass or all-fail). If yes, fetch top-K' rollouts from "
                "replay_buffer[prompt_id] with highest |stored_advantage|, filtering by "
                "age <= max_age_epochs. Compose mixed group: 0.7*K fresh + 0.3*K "
                "replayed (rounded). Re-compute advantage over the COMBINED group using "
                "standard z-score normalization. For replayed rollouts, clamp ratio "
                "π_θ(t|x) / π_old(t|x) using the stored logprob as π_old. ~50 lines "
                "total, no external storage. Compatible with existing LoRA infrastructure."
            ),
            "arxiv_ref": "2606.04560",
            "estimated_gpu_hours": 2.5,
        },
    },
    # ──────────────────────────────────────────────────────────────────────────
    # EXP-119: MDP-GRPO — Multi-Temperature Sampling + Cross-Batch Dual-Anchor
    #          Normalization for Zero-Variance Groups (HumanEval)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "exp_2026_07_16_002_mdp_grpo_multitemp_dualanchor_humaneval",
        "priority": 7,
        "gpu": "auto",
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2606.06058 — MDP-GRPO: Stabilized Group Relative Policy Optimization "
            "for Multi-Constraint Instruction Following (June 2026). Formalizes three "
            "pathologies of z-score group normalization: (1) low-variance amplification, "
            "(2) mean-centering blindness, (3) zero-variance collapse (our primary issue: "
            "ACR=52.4%). Proposes two orthogonal fixes applicable to our pipeline: "
            "(a) multi-temperature sampling — K/2=4 rollouts at T=0.7 (conservative) "
            "and K/2=4 at T=1.2 (exploratory) per group; this increases within-group "
            "reward dispersion → lower probability of all-fail groups at the sampling "
            "stage; for tasks at p=0.15 (our marginal tasks), mixed-temperature reduces "
            "P(all-fail) from 27% (uniform T=1.0, K=8) to an estimated 19%; "
            "(b) dual-anchor normalization — for groups still collapsed after multi-temp "
            "sampling, replace within-group mean/std with EMA cross-batch statistics "
            "(window=32 steps) as the normalization anchor: A_i = (r_i - EMA_mean) / "
            "(EMA_std + ε), providing a meaningful gradient direction rather than zero. "
            "Distinct from RAPER (no replay, changes sampling/normalization not "
            "composition), sign advantage (formula for all-fail only, no temperature "
            "variation), AVSPO (synthetic samples, not multi-temperature), TACO (per-"
            "token credit in winning rollouts, not group variance). Target: ACR 52.4% "
            "→ ≤38%, Full arm → ≥93.0%. AAAI 2027 deadline 2026-08-15: 30 days."
        ),
        "spec": {
            "bench": "humaneval",
            "n_cycles": 4,
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "large_model": "openai/gpt-5.5",
            "mdp_grpo_enabled": True,
            "mdp_multitemp_T_low": 0.7,
            "mdp_multitemp_T_high": 1.2,
            "mdp_multitemp_K_low": 4,
            "mdp_multitemp_K_high": 4,
            "mdp_dual_anchor_ema_window": 32,
            "mdp_dual_anchor_eps": 1e-4,
            "mdp_dual_anchor_condition": "zero_within_group_variance",
            "n_generations": 8,
            "grpo_temperature": None,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "analysis": [
                "acr_per_cycle_multitemp_vs_singlettemp",
                "within_group_reward_variance_distribution",
                "dual_anchor_activation_rate",
                "ema_stats_per_training_step",
                "full_arm_pass_at_1_improvement",
            ],
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (baseline): full 92.68%, router 92.68%",
                "EXP-108 (sign advantage): all-fail formula change only",
                "EXP-118 (RAPER): replay-based collapsed-group fix (complementary)",
                "arxiv:2606.06058 (MDP-GRPO): +1.8–3.1pp vs GRPO on code/instruction",
            ],
            "implementation_files": [
                "src/pipeline/grpo_train_simple.py",
            ],
            "implementation_note": (
                "In grpo_train_simple.py: "
                "Replace the single client.generate(..., temperature=GRPO_TEMPERATURE) "
                "call with TWO calls: K_low=4 rollouts at T=0.7 and K_high=4 rollouts "
                "at T=1.2, then concatenate into a combined group of K=8. "
                "Add an EMA tracker: ema_mean (initialized to 0.5, i.e., midpoint of "
                "[0,1] binary reward), ema_std (initialized to 0.5), updated each step "
                "with momentum=1-1/32. "
                "In advantage computation: after computing within-group mean/std, check "
                "if std == 0 (zero-variance collapse). If yes, substitute EMA stats: "
                "A_i = (r_i - ema_mean) / (ema_std + 1e-4). "
                "Note: GRPO_TEMPERATURE env var is overridden by this experiment; set "
                "mdp_multitemp_T_low/high directly in the spec above. "
                "~30 lines total, no external storage or buffer management."
            ),
            "arxiv_ref": "2606.06058",
            "estimated_gpu_hours": 2.5,
        },
    },
]


def atomic_save(path: Path, obj) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


def main():
    with open(STATE_PATH) as f:
        state = json.load(f)

    queue = state.get("queue", [])
    history = state.get("history", [])
    existing_ids = {e["id"] for e in queue} | {e["id"] for e in history}

    added = []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] in existing_ids:
            print(f"  SKIP (already queued): {exp['id']}")
            continue
        queue.append(exp)
        existing_ids.add(exp["id"])
        added.append(exp["id"])
        print(f"  ADDED: {exp['id']}")

    state["queue"] = queue
    atomic_save(STATE_PATH, state)
    print(f"\nDone. Added {len(added)} experiments. Total queue: {len(queue)} pending.")


if __name__ == "__main__":
    main()
