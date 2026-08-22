"""
Pending queue update — 2026-08-22
Run on A800 when connectivity is restored:
  python /data0/home/zeyuwang/auto_research/pending_queue_update_2026_08_22.py
Appends EXP-183 and EXP-184 to state["queue"] and saves atomically.

A800 offline since 2026-05-14 (day 100). Queue ~182 pending (>20 cap → 2 today).
AAAI 2027 camera-ready window (2026-08-16 to ~2026-08-29), day 7.
GPU window: CLOSED. EXP-183 (F-GRPO tail-miss) and EXP-184 (no-std-norm) queued for GPU return.

NEW PAPERS TODAY:
  EXP-183: F-GRPO tail-miss fix (arxiv:2602.06717) —
           GRPO all-fail groups (no correct rollout at K=8) contribute zero positive gradient;
           on hard HumanEval problems (pass@1 < 0.2 under Qwen3-4B) this is frequent and
           silently wastes training compute. F-GRPO detects all-fail groups and extends
           sampling to K_max=32 at higher temperature, guaranteeing >=1 correct rollout
           per prompt (when any correct solution exists). Distinct from Cue-GRPO (EXP-176):
           Cue-GRPO redistributes credit among already-sampled correct rollouts; F-GRPO
           ensures hard prompts are represented in the gradient at all. Priority 7.
  EXP-184: Dark Room GRPO no-std-norm for sparse binary reward (arxiv:2607.21273) —
           GRPO normalizes advantages by dividing by group std. In sparse binary-reward
           settings (pass@1), as the policy masters easy problems the group std -> 0,
           but normalized advantages remain +/-1, producing full-scale gradient pressure
           from trivially-solved groups that carry zero learning content. Removing std
           normalization (mean-only normalization) prevents this pathology. A 1-line
           change to src/pipeline/grpo_train_simple.py; 1 cycle from cycle-3 checkpoint.
           Priority 6.

Apply chain before this patch:
    python3 auto_research/pending_queue_update_2026_08_11.py  # EXP-158, EXP-159
    python3 auto_research/pending_queue_update_2026_08_12.py  # EXP-160, EXP-161
    python3 auto_research/pending_queue_update_2026_08_13.py  # EXP-162, EXP-163
    python3 auto_research/pending_queue_update_2026_08_14.py  # EXP-164, EXP-165
    python3 auto_research/pending_queue_update_2026_08_15.py  # EXP-166, EXP-167
    python3 auto_research/pending_queue_update_2026_08_16.py  # EXP-168, EXP-169
    python3 auto_research/pending_queue_update_2026_08_17.py  # EXP-170, EXP-171
    python3 auto_research/pending_queue_update_2026_08_18.py  # EXP-172, EXP-173
    python3 auto_research/pending_queue_patch_2026-08-19.py   # EXP-174..178 (block)
    python3 auto_research/pending_queue_update_2026_08_20.py  # EXP-179, EXP-180
    python3 auto_research/pending_queue_update_2026_08_21.py  # EXP-181, EXP-182
    python3 auto_research/pending_queue_update_2026_08_22.py  # EXP-183, EXP-184 (this file)
"""
import json, os, tempfile, shutil

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

new_experiments = [
    {
        "id": "exp_2026_08_22_001_fgrpo_finite_rollout_tail_miss_minority_oracle_sampling",
        "priority": 7,
        "kind": "grpo_continual",
        "gpu": "auto",
        "rationale": (
            "arxiv:2602.06717 ('F-GRPO: Don't Let Your Policy Learn the Obvious and Forget "
            "the Rare', February 2026, revised August 2026) identifies a structural failure "
            "mode in standard GRPO at finite group sizes: when K rollouts for a given prompt "
            "yield no correct solution ('all-fail group'), the group's positive advantage is "
            "zero across all rollouts, contributing no upward gradient for that prompt. The "
            "prompt wastes a training slot, and the policy's correct-solution probability for "
            "that prompt remains unchanged. This is especially damaging on hard problems where "
            "the small model's pass@1 is low (say 0.1): at K=8, the probability of an all-fail "
            "group is (0.9)^8 ~= 43%---nearly half of training slots for hard problems carry "
            "no positive signal. The effect is that easy problems (high pass@1, low all-fail "
            "frequency) dominate gradient updates while the hard tail receives minimal training "
            "pressure.\n\n"
            "F-GRPO's fix: at sampling time, detect all-fail groups after K initial rollouts. "
            "For each all-fail group, sample K' additional rollouts at higher temperature "
            "(T_explore = 1.2) until at least 1 correct solution is found or K_max = 32 total "
            "rollouts are exhausted. If at least 1 correct rollout is found, form the GRPO "
            "group from the (now mixed) K + K' rollouts normally. If none found (K_max "
            "exhausted), skip the prompt for this step (consistent with standard GRPO behavior "
            "for problems with no correct solution at any temperature). Results: F-GRPO at "
            "K=8 + K_max=32 achieves +3.5/+11.8% on AIME 2025 at 4x less compute than "
            "standard GRPO at K=32, because the added rollouts are targeted only at groups "
            "that would otherwise be wasted.\n\n"
            "MERA connection: By cycle 3, Qwen3-4B still fails ~30% of HumanEval hard tail "
            "at pass@1. With K=8, roughly 20-43% of training steps on these problems are "
            "all-fail groups --- no gradient for those prompts. F-GRPO's minority-oracle "
            "sampling recovers signal from these wasted steps at marginal cost (easy problems "
            "already hit correct at K=8 and incur no extra inference). This is DISTINCT from "
            "Cue-GRPO (EXP-176, 2608.03467): Cue-GRPO redistributes credit AMONG already-"
            "sampled correct rollouts; F-GRPO ensures hard prompts have correct rollouts in "
            "the group at all. The two are complementary: apply Cue-GRPO's rarity weighting "
            "AFTER F-GRPO's minority-oracle sampling. Also distinct from G2RPO-A (EXP-175, "
            "2508.13023, which injects procedural guidance into the prompt to raise pass@1) "
            "and hard-first curriculum (EXP-173, which reorders batches by difficulty without "
            "changing the sampling budget per group). Implementation: in "
            "src/pipeline/grpo_train_simple.py, after K initial rollouts, check if any pass "
            "the verifier; if not, sample K' more at T_explore=1.2 until >=1 correct or "
            "K_max=32 total. Track all_fail_rate per cycle as a diagnostic metric."
        ),
        "spec": {
            "base_model": "05_qwen3_5_4b_273",
            "from_checkpoint": "results/e2e_4cyc_gpt55/cycle_3/grpo_adapter",
            "bench": "humaneval",
            "n_cycles": 2,
            "grpo_temperature": 0.9,
            "grpo_batch_size": 1,
            "n_generations": 8,
            "algo": "dapo",
            "fgrpo_tail_miss_fix": True,
            "fgrpo_explore_temperature": 1.2,
            "fgrpo_k_max": 32,
            "cost_target": 0.35,
            "paper": "arxiv:2602.06717",
        },
    },
    {
        "id": "exp_2026_08_22_002_grpo_no_std_norm_sparse_binary_reward_dark_room_prevention",
        "priority": 6,
        "kind": "grpo_continual",
        "gpu": "auto",
        "rationale": (
            "arxiv:2607.21273 ('The Dark Room in the Reward Channel: Dense Prediction "
            "Rewards Collapse GRPO-Trained LLM Agents --- and What Actually Works', Wang, "
            "July 2026) diagnoses a subtle pathology in GRPO's advantage normalization under "
            "sparse rewards. GRPO computes normalized advantages as "
            "A_norm = (A - mean(A)) / std(A) per group, then clips and updates. In sparse "
            "binary-reward settings (e.g., pass@1), as the policy masters easy problems, "
            "groups for those problems move toward all-success (all rollouts correct). The "
            "group's std(A) -> 0, but the normalized advantages remain at +/-1 magnitude "
            "due to the division by std. This means all-success groups --- which carry ZERO "
            "learning information (every rollout is already correct, improving further is "
            "impossible) --- nonetheless exert full-scale gradient pressure in arbitrary "
            "directions determined by numerical noise. The effect is equivalent to training "
            "on noise for easy problems, which can degrade performance on harder problems or "
            "produce training instability.\n\n"
            "The paper identifies this as a masked pathology: the optimizer is reinforcing "
            "trivially-correct solutions as if they required attention, at the cost of gradient "
            "bandwidth for genuinely hard problems. The fix is straightforward: remove std "
            "normalization (mean-only normalization: A_norm = A - mean(A)) or floor std at a "
            "minimum value (e.g., max(std(A), eps=0.1)). Empirically, mean-only normalization "
            "matches or exceeds std-normalized GRPO on code generation tasks while reducing "
            "training variance, especially at cycles >=2 when the policy has already mastered "
            "the easy tail.\n\n"
            "MERA connection: By cycle 3, Qwen3-4B on HumanEval has already solved ~70% of "
            "problems (pass@1 ~= 0.7). At K=8, many groups for easy problems are all-success. "
            "Our current GRPO Phase 3b normalizes by std --- so these all-success groups "
            "contribute noisy gradient that could be destabilizing the remaining learning on "
            "hard problems. The test-split routing accuracy plateau (60.98% at cycle 3) may "
            "partly reflect this noise: the model oscillates under trivial-correct gradient "
            "pressure rather than converging on hard cases. Mean-only normalization (A_norm = "
            "A - mean(A)) is a 1-line change to grpo_train_simple.py and should reduce noise "
            "on the easy tail while keeping full gradient pressure on hard problems. "
            "DISTINCT from: EXP-183 (F-GRPO tail-miss addresses all-FAIL groups; this "
            "addresses all-SUCCESS groups --- complementary); EXP-176 (Cue-GRPO: inter-group "
            "credit distribution; this: intra-group normalization); EXP-172/173 (forgetting "
            "via replay/curriculum; this: normalization stability). "
            "Implementation: in grpo_train_simple.py replace the advantage std-norm step with "
            "mean-only: advantages = advantages - advantages.mean(dim=-1, keepdim=True). "
            "1 cycle from cycle-3 checkpoint. Compare train/test routing accuracy and pass@1 "
            "against baseline GRPO cycle-4."
        ),
        "spec": {
            "base_model": "05_qwen3_5_4b_273",
            "from_checkpoint": "results/e2e_4cyc_gpt55/cycle_3/grpo_adapter",
            "bench": "humaneval",
            "n_cycles": 1,
            "grpo_temperature": 0.9,
            "grpo_batch_size": 1,
            "n_generations": 8,
            "algo": "dapo",
            "grpo_std_norm": False,
            "grpo_advantage_norm_mode": "mean_only",
            "cost_target": 0.20,
            "paper": "arxiv:2607.21273",
        },
    },
]


def already_exists(state, exp_id):
    for item in state.get("queue", []) + state.get("history", []):
        if item.get("id") == exp_id:
            return True
    return False


with open(STATE_PATH, "r") as f:
    state = json.load(f)

added = []
for exp in new_experiments:
    if already_exists(state, exp["id"]):
        print(f"SKIP (already exists): {exp['id']}")
    else:
        state.setdefault("queue", []).append(exp)
        added.append(exp["id"])
        print(f"ADDED: {exp['id']} (priority={exp['priority']})")

if added:
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    shutil.move(tmp, STATE_PATH)
    print(f"\nSaved {STATE_PATH} with {len(added)} new experiment(s).")
else:
    print("\nNo new experiments added (all duplicates).")
