#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-05 (EXP-108, EXP-109).

A800 connectivity: offline since 2026-05-14 (day ~52). Apply when restored.

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

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_05.py            # EXP-108, EXP-109

Queue was ~111 pending on 2026-07-04 (+2 added: EXP-106, EXP-107).
Queue cap applied: >20 → max 2 new experiments.

Data motivating today's proposals
----------------------------------
results/e2e_4cyc_gpt55/cycle_3/e2e_ablation_summary.json (unchanged — A800 offline):
    HumanEval 4-cycle:
      large (always-large):        task_pass=96.34%, cost_vs_large=100%
      skills (always-small+proc):  task_pass=75.61%, cost_vs_large=10%
      router (logistic, cycle 3):  task_pass=92.68%, routing_acc=92.68%,
                                   cost_vs_large=27.56%, fallback=6.10%
      full (router+GRPO):          task_pass=92.68% ← SAME as router
    Root problem: ACR=52.4% — 43/82 GRPO groups have zero within-group reward
    variance → zero gradient → Full = Router.
    Skills gap: 75.61% vs large 96.34% — 20.7pp gap. SFT crowds on easy tasks.

arxiv:2605.07689 — Gradient Starvation in Binary-Reward GRPO (May 8, 2026):
    Group-mean centering in GRPO produces A=0 (zero advantage) for two cases:
      all-fail groups: mean(reward)=0, all rollouts get advantage=0-0=0 → no gradient
      all-pass groups: mean(reward)=1, all rollouts get advantage=1-1=0 → no gradient
    The paper PROVES degeneracy rate ≥ Bernoulli prediction (Jensen's inequality).
    Measured: 0.69 degeneracy rate at G=4 in Qwen3.5-9B GSM8K training logs.
    Our ACR=52.4% (43/82 all-fail groups at G=8) is consistent with this finding.

    THE FIX — Sign advantage: A = 2r - 1
      r=1 (correct):  A = +1 → reinforce correct tokens (same direction as standard GRPO)
      r=0 (wrong):    A = -1 → penalize wrong tokens (NEW: all-fail groups now get gradient)
      No all-zero advantage groups under Sign: A is never 0 for binary reward.
      No π_ref required (unlike SRPO/EXP-106) — just replace the advantage formula.
      Implementation: ~2 lines in grpo_train_simple.py.

    EMPIRICAL RESULTS (paper):
      Sign advantage: 73.8% GSM8K accuracy
      Standard normalized group-mean DrGRPO at G=4: 28.4% accuracy
      Difference: +45.4 percentage points, 7 seeds, p<0.0001.
      Effect is largest when degeneracy rate is highest (small G, hard tasks).

    RELATIONSHIP TO SRPO (EXP-106):
      SRPO routes failed rollouts to an SDPO loss head (logit-level correction
      toward π_ref). This is a NEW loss head in addition to GRPO.
      Sign advantage removes group-mean centering so A is never 0 for binary r.
      These are architecturally different: SRPO needs π_ref logits; Sign does not.
      Both target ACR=52.4%; they are orthogonal and potentially combinable.
      If both EXP-106 and EXP-108 succeed, a future EXP-110 could combine them:
      Sign advantage (no degenerate groups) + SDPO correction (stronger signal
      for the still-failed rollouts that have A=-1 under Sign).

    CLIP RATIO INTERACTION:
      DAPO uses clip_high/clip_low ratios. Sign advantage produces larger
      |A| values (±1 vs ±σ_normalized) — clip ratio may need adjustment.
      Suggested: run with DAPO's default clip first, then with clip_ratio=1.0
      (no clipping) as a variant to assess sensitivity.

arxiv:2605.12652 — Multi-Rollout On-Policy Distillation (MOPD, May 12 / v2 Jun 1, 2026):
    Standard OPD distills each (query, trace) pair independently. MOPD observes
    that G rollouts from the same query are generated together but never used
    together — a missed opportunity for richer supervision.
    MOPD conditions each training example on peer rollout context:
      Positive peer imitation: prepend one successful peer rollout as few-shot demo.
      Contrastive conditioning: prepend one success + one failure as contrast.
    Teacher signal incorporates peer context → learns what makes a peer succeed
    (or what makes a peer fail, to avoid).
    Results: +2-4% on reasoning benchmarks vs standard OPD, concentrated on hard tasks.

    RELEVANCE TO SKILLS ARM:
      Our SFT Phase 3a collects G=8 rollouts per query (SCALING_FORCE_BOTH=1).
      These 7 peer rollouts are currently DISCARDED during SFT data construction
      (traces_to_sft.py only uses the large-model traces, not peer context).
      MOPD extracts value from this discarded peer data.
      Complementary to DemoPSD (EXP-107): DemoPSD changes WHAT to distill
      (Category-A disagreement traces only); MOPD changes HOW to distill (peer
      context). Both can be applied simultaneously if both show positive results.
      Implementation: ~30 lines in traces_to_sft.py.

    SEQUENCE LENGTH CONCERN:
      Adding peer context increases sequence length. At Qwen2.5-Coder-1.5B with
      HumanEval (max solution ~256 tokens), one peer context adds ~256-512 tokens
      to training sequences. With 512-token peer context cap, sequences stay under
      2048 tokens for most tasks — manageable on A800 in ~1.75 GPU hours.
      If context causes OOM: reduce peer context to 256 tokens or use only
      the first 128 tokens of the peer rollout (the crucial reasoning-start).

AAAI 2027 context (deadline 2026-08-15, 41 days):
    EXP-108 (Sign advantage): priority-8 ACR fix. If it succeeds, the result
    "we eliminated gradient starvation with a 2-line change and achieved +Xpp"
    is a clean ablation for §4. Combinable with SRPO (EXP-106).
    EXP-109 (MOPD): skills arm improvement. If skills arm reaches ≥78% (vs
    75.61%), the 17-18pp skills-vs-large gap becomes a strong narrative for
    "distillation quality matters as much as distillation quantity."
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
        "id": "exp_2026_07_05_001_sign_advantage_grpo_gradient_starvation_humaneval",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2605.07689 — Gradient Starvation in Binary-Reward GRPO: "
            "Why Group-Mean Centering Fails and Why the Simplest Fix Works "
            "(May 8, 2026). "
            ""
            "ROOT CAUSE OF ACR=52.4%: "
            "DAPO uses group-mean-centered advantage: "
            "  A_i = (r_i - mean(r)) / std(r) "
            "When all G=8 rollouts in a group fail (r_i=0 for all i): "
            "  mean(r) = 0, std(r) = 0 → advantage = 0 for all rollouts → zero gradient. "
            "This is gradient starvation. It affects 43/82 groups (52.4% ACR) per cycle. "
            "Paper proves degeneracy rate always ≥ Bernoulli prediction (Jensen's). "
            "Empirical degeneracy rate: 0.69 at G=4 in Qwen3.5-9B GSM8K logs — "
            "our ACR=0.524 at G=8 is consistent (larger G reduces degeneracy). "
            ""
            "THE SIGN ADVANTAGE FIX: "
            "Replace group-mean centering with a fixed reference advantage: "
            "  A = 2r - 1    (where r ∈ {0, 1} is the binary reward) "
            "  r=1 (correct rollout): A = +1  → reinforce correct tokens "
            "  r=0 (failed rollout):  A = -1  → penalize wrong tokens "
            "Under Sign advantage: "
            "  All-fail group: A = -1 for all rollouts → gradient that decreases "
            "    the probability of each wrong token (failure descent). "
            "    43 previously wasted groups now contribute negative reinforcement. "
            "  All-pass group: A = +1 for all rollouts → gradient that increases "
            "    correct token probabilities (standard reinforcement). "
            "  Mixed group: A ∈ {-1, +1} → standard relative ordering preserved. "
            "No group ever has A=0 under binary Sign advantage. "
            "No π_ref needed (unlike SRPO/EXP-106): Sign advantage is self-contained. "
            "Implementation: replace advantage computation in grpo_train_simple.py. "
            "  BEFORE: advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8) "
            "  AFTER:  advantages = 2.0 * rewards - 1.0 "
            "This is a ~2 line change. "
            ""
            "EMPIRICAL RESULTS (paper, arxiv:2605.07689): "
            "Sign advantage reaches 73.8% GSM8K accuracy vs 28.4% for "
            "normalized group-mean DrGRPO at G=4 (45.4pp gain, 7 seeds, p<0.0001). "
            "Effect is largest when degeneracy rate is high (hard tasks, small G). "
            "Our setting: G=8 HumanEval, degeneracy rate=0.524 — expect meaningful gain. "
            ""
            "RELATIONSHIP TO SRPO (EXP-106, SDPO routing): "
            "SRPO adds a new SDPO loss head for failed rollouts (logit correction "
            "toward π_ref). Sign advantage replaces the group-mean-centered advantage "
            "formula so degenerate groups get A≠0. These are orthogonal mechanisms: "
            "  - SRPO: adds computation (SDPO head), requires π_ref logits "
            "  - Sign: changes advantage formula, no new computation or model "
            "Both address ACR=52.4% via different mechanisms. If both succeed, "
            "a future EXP-110 can combine them: Sign advantage for the GRPO path "
            "+ SDPO for failed rollouts. "
            ""
            "CLIP RATIO INTERACTION: "
            "DAPO's clipping uses a clip_high/clip_low ratio. Sign advantage produces "
            "|A|=1 constant vs. normalized advantage with std≈1 → similar scale. "
            "Test with default DAPO clip first; if gradient explosion occurs, reduce "
            "clip_high from default 0.2 to 0.1. "
        ),
        "spec": {
            "pipeline": "humaneval_sign_advantage_grpo",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 1,
            "grpo_algo": "sign_advantage",
            "sign_advantage_formula": "2 * reward - 1",
            "variants": [
                "sign_advantage_default_clip",
                "sign_advantage_clip_high_0.1",
                "dapo_baseline",
            ],
            "n_generations": 8,
            "grpo_temperature": 1.0,
            "scaling_force_both": True,
            "sft_include_success": True,
            "eval_data": _HE_EVAL,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (DAPO binary reference: 92.68% HE, ACR=52.4%)",
                "arxiv:2605.07689 (Sign advantage +45.4pp GSM8K vs group-mean DrGRPO)",
                "EXP-106 (SRPO: orthogonal ACR fix via SDPO routing)",
            ],
            "implementation_note": (
                "In grpo_train_simple.py, in the advantage computation step, "
                "replace: `advantages = (rewards - rewards.mean(-1, keepdim=True)) "
                "/ (rewards.std(-1, keepdim=True) + 1e-8)` "
                "with: `advantages = 2.0 * rewards - 1.0` "
                "Approximately 2 lines of change. "
                "All other DAPO/GRPO logic (clipping, KL penalty, token masking) unchanged. "
                "Variant B uses clip_high=0.1 in case |A|=1 causes instability with default clip."
            ),
            "primary_metrics": [
                "humaneval_pass_at_1_post_1cycle",
                "grpo_fraction_nonzero_advantage_groups",
                "grpo_acr_equivalent_dapo_baseline",
                "grpo_mean_abs_advantage",
                "grpo_gradient_norm_mean",
                "grpo_gradient_norm_std",
                "training_loss_curve",
                "output_length_mean",
                "output_length_std",
                "variant_delta_pass_at_1",
            ],
            "target_outcome": (
                "Sign advantage variant achieves pass@1 > 92.68% (DAPO baseline). "
                "Fraction of non-zero-advantage groups = 100% (vs 47.6% under DAPO). "
                "Gradient norm is stable (no explosion from |A|=1 constant signal). "
                "If pass@1 >= 93.68% (+1pp): Full > Router → AAAI headline. "
                "Output length stable or longer (Sign encourages exploration on hard tasks). "
            ),
            "gpu": "auto",
            "estimated_gpu_hours": 1.5,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_07_05_002_mopd_peer_conditioned_distillation_skills_humaneval",
        "priority": 6,
        "kind": "joint_cycle_multiseed",
        "rationale": (
            "arxiv:2605.12652 — Multi-Rollout On-Policy Distillation via "
            "Peer Successes and Failures (MOPD, CMU/Microsoft/Purdue, "
            "May 12, 2026, v2 June 1, 2026). "
            ""
            "PROBLEM: WASTED PEER ROLLOUT DATA IN SFT "
            "Our Phase 3a SFT (traces_to_sft.py) processes each "
            "(query, large_model_trace) pair independently. "
            "We collect G=8 small-model rollouts per query — 7 peer rollouts "
            "per training example are discarded after reward computation. "
            "MOPD shows these peers are informative: a failed rollout tells the "
            "model what NOT to do; a successful rollout provides positive evidence "
            "for valid patterns — both from the student model's own attempt space. "
            ""
            "MOPD APPROACH: "
            "For each SFT training example (query, large_trace): "
            "  1. Retrieve the G=8 small-model rollouts from traces.jsonl. "
            "  2. Positive peer: select one rollout with reward=1 (if any). "
            "  3. Negative peer: select one rollout with reward=0 (if any). "
            "  4. Construct augmented training prompt: "
            "       peer_success_block + peer_failure_block + procedure + problem "
            "     where procedure is the existing SkillBook procedure. "
            "  5. Train the SFT model on: augmented_prompt → large_trace. "
            "The model learns: 'given what I can and cannot do on this query, "
            "here is what the large model does' → richer supervision signal. "
            ""
            "TWO VARIANTS (from MOPD paper): "
            "  (A) Positive-only peer imitation: prepend one peer success only. "
            "      '# Here is an approach that works:\n{peer_success}\n\n---\n\n{problem}' "
            "  (B) Contrastive success-failure conditioning: prepend both. "
            "      '# Correct approach:\n{peer_success}\n# Avoid:\n{peer_failure}\n\n---\n\n{problem}' "
            "  (C) No peer context (baseline: 75.61% skills arm pass@1). "
            "MOPD paper finds variant B (contrastive) > variant A > no peer context. "
            ""
            "COMPLEMENTARITY WITH DemoPSD (EXP-107): "
            "DemoPSD (EXP-107) changes WHAT to distill: "
            "  → only distill from Category A (disagreement) traces; weight Category B at 0.1. "
            "MOPD changes HOW to distill: "
            "  → for each selected trace, enrich the context with peer rollouts. "
            "Both can be applied simultaneously: "
            "  MOPD+DemoPSD = only distill Category-A traces AND add peer context. "
            "  This combination could be EXP-110 if both EXP-107 and EXP-109 succeed. "
            ""
            "SEQUENCE LENGTH: "
            "HumanEval solutions are short (~100-256 tokens). "
            "With 512-token cap on peer context (each peer rollout truncated to 256 tokens), "
            "total training sequence stays under 2048 tokens for 95%+ of tasks. "
            "No OOM expected on A800 80GB for Qwen2.5-Coder-1.5B SFT. "
            "If edge cases OOM: cap peer context at 128 tokens. "
            ""
            "IMPLEMENTATION (~30 lines in traces_to_sft.py): "
            "1. Group traces by query_id to access peer rollouts for each query. "
            "2. For each selected trace, sample one peer_success (reward=1) and "
            "   one peer_failure (reward=0) from same query's rollouts (not the "
            "   trace being distilled). "
            "3. Augment prompt string before inserting into SFT DataLoader. "
            "4. Cap peer context at 512 tokens total (256 per peer rollout). "
            "5. Fallback: if no peer_success available (all-fail group), "
            "   use positive-only conditioning with an empty peer_success block. "
            ""
            "AAAI 2027 NARRATIVE VALUE: "
            "A skills arm improvement from 75.61% to ≥77.5% with MOPD would "
            "support the claim: 'our multi-rollout distillation extracts value "
            "from all G rollouts, not just the large-model trace.' "
            "This aligns with the paper's theme: the G rollouts generated per "
            "cycle are not only reward-labeled for router training, but also "
            "provide peer context for SFT — a new use for existing computation. "
        ),
        "spec": {
            "pipeline": "humaneval_mopd_peer_conditioned_sft",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 1,
            "sft_mode": "mopd_peer_conditioned",
            "mopd_peer_context_tokens": 512,
            "mopd_peer_context_per_rollout": 256,
            "mopd_variants": [
                "positive_only_peer",
                "contrastive_peer",
                "no_peer_baseline",
            ],
            "sft_include_success": True,
            "scaling_force_both": True,
            "eval_arm": "skills",
            "eval_data": _HE_EVAL,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 skills arm: 75.61% HE pass@1",
                "arxiv:2605.12652 (MOPD +2-4% on reasoning benchmarks vs standard OPD)",
                "EXP-107 (DemoPSD: complementary WHAT-to-distill filter)",
            ],
            "implementation_note": (
                "In traces_to_sft.py: "
                "After grouping traces by query_id, for each selected SFT trace: "
                "  peer_rollouts = [r for r in query_rollouts if r != current_trace] "
                "  peer_success = next((r for r in peer_rollouts if r.reward == 1), None) "
                "  peer_failure = next((r for r in peer_rollouts if r.reward == 0), None) "
                "Augment prompt: "
                "  Variant A (positive-only): "
                "    f'# Here is a working approach:\\n{peer_success[:256tokens]}\\n\\n---\\n\\n{problem}' "
                "  Variant B (contrastive): "
                "    f'# Working approach:\\n{peer_success[:256t]}\\n# Avoid:\\n{peer_failure[:256t]}\\n\\n---\\n\\n{problem}' "
                "  Variant C (baseline): existing format unchanged. "
                "Use tokenizer to count and cap at 512 tokens for peer context block. "
                "~30 lines additional code in traces_to_sft.py."
            ),
            "primary_metrics": [
                "humaneval_pass_at_1_skills_arm_post_sft",
                "sft_sequence_length_mean",
                "sft_sequence_length_p95",
                "sft_peer_success_availability_rate",
                "sft_peer_failure_availability_rate",
                "skills_arm_hard_task_pass_rate",
                "skills_arm_easy_task_pass_rate",
                "variant_delta_pass_at_1",
            ],
            "target_outcome": (
                "Best variant achieves skills arm pass@1 >= 77.5% (+1.9pp from 75.61%). "
                "Contrastive variant (B) outperforms positive-only (A) per MOPD paper. "
                "Hard-task improvement > easy-task improvement (peer context most "
                "informative where model is most uncertain). "
                "Sequence length stays under 2048 tokens for 95%+ of tasks (no OOM). "
                "Peer context availability: >70% of tasks have both a peer_success "
                "and peer_failure available (confirmed by small_rollout_results stats). "
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
