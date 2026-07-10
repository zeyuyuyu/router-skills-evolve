#!/usr/bin/env python3
"""
Daily queue patch — 2026-07-10 (EXP-110, EXP-111).

A800 connectivity: offline since 2026-05-14 (day ~57). Apply when restored.

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
    python3 auto_research/pending_queue_update_2026_07_03_paper.py      # EXP-106 (paper), EXP-107 (paper)
    python3 auto_research/pending_queue_update_2026_07_04.py            # EXP-106, EXP-107
    python3 auto_research/pending_queue_update_2026_07_05.py            # EXP-108, EXP-109

Then apply this patch:
    python3 auto_research/pending_queue_update_2026_07_10.py            # EXP-110, EXP-111

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
      full (router+GRPO):          task_pass=92.68% ← SAME as router
    Two open bottlenecks:
      (A) Full = Router: ACR=52.4% → 43/82 GRPO groups zero-variance → zero gradient
      (B) Skills gap: 75.61% vs large 96.34% — 20.7pp gap

arxiv:2607.00152 — "GRPO, Dr. GRPO, and DAPO Are Three Operations on One Number:
The Group-Standard-Deviation Identity" (July 2026, very fresh):
    Formal proof: GRPO/DrGRPO/DAPO all reduce to a single scalar function of the
    within-group reward standard deviation. The gradient magnitude is proportional
    to group-std. For binary rewards: when std = 0 (all-pass or all-fail group),
    gradient = 0 by algebraic identity — regardless of which GRPO variant is used.
    Key corollary: any algorithm that needs within-group variance for its gradient
    is fundamentally incapable of learning from degenerate groups. The solution must
    therefore: (A) change the reward structure to ensure std > 0 (e.g., partial
    credit, relative ranking), OR (B) change the baseline to not be per-group mean
    (e.g., global EMA, sign advantage), OR (C) skip degenerate groups and reallocate
    rollout compute elsewhere (e.g., AERO, arxiv:2602.14338).
    Our EXP-108 (Sign advantage) implements option B. EXP-110 implements option A
    (partial-credit rewards that create within-group variance for all-fail groups).
    The group-std identity (2607.00152) provides the tightest theoretical framing for
    why EXP-110 must work: partial credit → std > 0 → gradient > 0, provably.

arxiv:2606.06021 — "OPRD: On-Policy Representation Distillation" (June 4, 2026,
revised June 20, 2026):
    Standard on-policy distillation supervises the student only at the output layer
    (next-token KL to the teacher's token distribution). OPRD adds hidden-state
    alignment: for selected transformer layers (typically every 4th), it minimizes
    MSE between student and teacher hidden states on the SAME on-policy rollouts.
    The LM head is a low-rank bottleneck: teacher representations that encode
    planning, variable tracking, and algorithmic structure cannot all be expressed
    via single-token KL. Hidden-state supervision directly transfers this sub-word
    structure without compressing it through the LM head first.
    Results: OPRD is Pareto-dominant over output-space OPD across accuracy, training
    time, and GPU memory (same rollout budget, richer supervision).
    RELEVANCE: Our skills arm is stuck at 75.61% despite seeing teacher traces from
    GPT-5.5 on every HumanEval problem (SCALING_FORCE_BOTH=1). The 20.7pp gap is
    too large to close purely via token-level KL given the capacity gap between
    Qwen2.5-Coder-1.5B and GPT-5.5. OPRD's hidden-state alignment bypasses the
    output bottleneck — the small model's internal state is directly pulled toward
    the large model's internal representations on the same reasoning step, giving
    gradient signal for planning and algorithmic computation that output-KL cannot.
    Implementation: add a hidden_state_align_loss to train_small_model.py alongside
    standard CE loss. Requires the large model (teacher) to expose hidden states
    during the SFT data-collection pass — for GPT-5.5 this is not possible (API-only),
    so we proxy the teacher's hidden states with the cycle's best small-model checkpoint
    acting as a local pseudo-teacher (self-distillation variant of OPRD).
    Alternative: use a local Qwen3.5-4B (available on A800) as the teacher model for
    hidden-state alignment while keeping GPT-5.5 traces for output-level supervision.
    Estimated improvement: +2-4pp skills arm pass@1 per the OPRD paper benchmarks.

AAAI 2027 context (deadline 2026-08-15, 36 days from today):
    EXP-110 (partial-credit reward): plugs the group-std zero-gradient gap without
    needing a reference model or changing the GRPO architecture. Clean story for §5
    "gradient starvation analysis": sign advantage (EXP-108) and partial credit
    (EXP-110) are two orthogonal solutions to the same algebraic problem identified
    by arxiv:2607.00152. Table: ACR under DAPO vs Sign vs Partial-Credit.
    EXP-111 (OPRD local teacher): if skills arm reaches ≥78% (+2.4pp), the narrative
    becomes: "standard output-KL distillation plateaus at 75.6%; hidden-state
    alignment with a local pseudo-teacher adds 2.4pp without additional teacher API
    calls." This is a cost-neutral improvement story for §6.
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
        "id": "exp_2026_07_10_001_partial_credit_reward_grpo_group_std_identity_humaneval",
        "priority": 8,
        "kind": "grpo_continual",
        "rationale": (
            "arxiv:2607.00152 — 'GRPO, Dr. GRPO, and DAPO Are Three Operations on One "
            "Number: The Group-Standard-Deviation Identity' (July 2026). "
            ""
            "THEORETICAL BASIS — THE GROUP-STD IDENTITY: "
            "The paper proves that the GRPO family of algorithms all reduce to the "
            "same scalar function of within-group reward standard deviation (std). "
            "Specifically, the effective gradient magnitude is proportional to std(r_1..r_G). "
            "For binary rewards (r_i ∈ {0,1}) with an all-fail group (r_i=0 for all i): "
            "  std = sqrt((1/G) * sum((0-0)^2)) = 0 → gradient = 0. "
            "This is NOT an implementation bug and NOT specific to DAPO: it is an "
            "algebraic identity that holds for GRPO, DrGRPO, and DAPO equally. "
            "The identity implies THREE possible fixes: "
            "  (A) Change reward to ensure std > 0 in degenerate groups. ← THIS EXPERIMENT "
            "  (B) Change the baseline (e.g., global EMA, Sign advantage / EXP-108). "
            "  (C) Skip degenerate groups, reallocate rollout compute (AERO, arxiv:2602.14338). "
            "EXP-108 (Sign advantage, arxiv:2605.07689) implements option B. "
            "EXP-110 implements option A — partial-credit rewards ensure std > 0 for all groups. "
            ""
            "PARTIAL-CREDIT REWARD DESIGN: "
            "Current reward: r_i = 1.0 if all assertions in check() pass, else 0.0. "
            "Proposed partial-credit reward: "
            "  r_i = w_compile * compile_ok  +  w_run * runs_without_exception "
            "         +  w_pass * test_pass_fraction "
            "Where: "
            "  compile_ok       ∈ {0,1}: code is syntactically valid Python (py_compile check) "
            "  runs_without_exception ∈ {0,1}: code executes without error (excluding assert) "
            "  test_pass_fraction  ∈ [0,1]: fraction of individual assert statements that pass "
            "  w_compile = 0.05,  w_run = 0.10,  w_pass = 0.85 "
            "  → total r_i ∈ [0.0, 1.0], where r_i=1.0 iff all tests pass (same signal for "
            "    full-pass rollouts; richer signal for all-fail groups). "
            ""
            "EFFECT ON ALL-FAIL GROUPS: "
            "In a group where all G=8 rollouts fail the full test suite: "
            "  If some rollouts at least compile and/or partially pass tests, they get r > 0. "
            "  Example: rollout A compiles but crashes → r=0.05. "
            "           rollout B compiles and passes 1/3 tests → r = 0.05 + 0.10 + 0.85*(1/3) ≈ 0.43. "
            "           rollout C doesn't compile → r = 0.0. "
            "  Group std = std(0.43, 0.05, 0.0, ...) > 0 → non-zero gradient, provably. "
            "  The 43 currently wasted all-fail groups each produce SOME within-group variance "
            "  as long as rollouts differ in compilation or partial test results. "
            ""
            "IMPLEMENTATION — HOW TO ADD PARTIAL CREDIT: "
            "In grpo_train_simple.py (or collect_traces.py), modify the reward function: "
            "  def compute_partial_reward(code, test_cases): "
            "    # Stage 1: compile check "
            "    try: compile(code, '<string>', 'exec'); compile_ok = 1.0 "
            "    except SyntaxError: return 0.0  # compile_ok=0, rest meaningless "
            "    # Stage 2: execution check "
            "    try: exec(code_with_imports, env); runs_ok = 1.0 "
            "    except Exception: runs_ok = 0.0 "
            "    # Stage 3: partial test pass "
            "    passed = sum(run_single_assert(code, t) for t in test_cases) "
            "    frac = passed / len(test_cases) "
            "    return 0.05 * compile_ok + 0.10 * runs_ok + 0.85 * frac "
            "Note: HumanEval check() functions have extractable assert statements. "
            "The `humaneval_pass_at_k.py` utility already runs individual test cases; "
            "extracting individual assert pass rates requires ~20 additional lines. "
            ""
            "COMPLEMENTARITY WITH EXP-108 (Sign Advantage): "
            "EXP-108 (Sign advantage): changes baseline so all-fail groups get A=-1 "
            "  regardless of within-group variance. "
            "EXP-110 (Partial credit): changes reward so all-fail groups have std > 0 "
            "  and the standard DAPO advantage formulation itself produces non-zero gradient. "
            "These are ORTHOGONAL mechanisms — both address the group-std=0 problem via "
            "different sides of the identity. They can be combined in a future EXP-112: "
            "  Partial-credit reward (EXP-110) + Sign advantage (EXP-108) + SRPO (EXP-106). "
            "If all three succeed independently: three convergent ACR fixes → AAAI §5. "
            ""
            "VARIANTS TO COMPARE: "
            "  (A) Partial credit, DAPO advantage (this experiment's main variant) "
            "  (B) Partial credit, Sign advantage (combines EXP-108 + EXP-110) "
            "  (C) Binary reward, DAPO advantage (baseline — reproduce 92.68%) "
            "  (D) Binary reward, Sign advantage (EXP-108 baseline for ablation) "
            "  Note: if A800 restored before EXP-108 runs, variants C and D can be "
            "  run simultaneously in this experiment to save GPU time. "
            ""
            "RISK — REWARD SCALE CHANGE: "
            "Partial-credit r ∈ [0, 1] vs binary r ∈ {0, 1}: the scale is preserved. "
            "HOWEVER, the full-pass reward is r=1.0 in both cases, so no additional "
            "KL penalty adjustment is needed. The KL term is already calibrated to "
            "binary rewards in the range [0, 1]. "
            "Risk: introducing partial credit for partial-pass groups might encourage "
            "'partial compliance' (code that passes some tests but not all), reducing "
            "incentive to fully solve hard tasks. Mitigation: set w_pass=0.85 so the "
            "incentive to pass ALL tests (r=1.0) is still 7× stronger than compiling only. "
            "Monitor: check if full-test-pass rate improves or stays flat (would indicate "
            "model is optimizing partial credit instead of full solve). "
        ),
        "spec": {
            "pipeline": "humaneval_partial_credit_grpo_group_std_identity",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "n_cycles": 1,
            "reward_type": "partial_credit",
            "partial_credit_weights": {
                "compile_ok": 0.05,
                "runs_without_exception": 0.10,
                "test_pass_fraction": 0.85,
            },
            "grpo_algo": "dapo",
            "n_generations": 8,
            "grpo_temperature": 1.0,
            "scaling_force_both": True,
            "sft_include_success": True,
            "variants": [
                "partial_credit_dapo",
                "partial_credit_sign_advantage",
                "binary_dapo_baseline",
            ],
            "eval_data": _HE_EVAL,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 (binary DAPO baseline: 92.68% HE, ACR=52.4%)",
                "arxiv:2607.00152 (group-std identity: std=0 → gradient=0 provably)",
                "arxiv:2605.07689 / EXP-108 (Sign advantage: option B fix)",
                "arxiv:2601.23058 (RLRR: relative ranking, option A conceptual ancestor)",
            ],
            "implementation_note": (
                "In grpo_train_simple.py (or collect_traces.py reward function): "
                "Replace binary `reward = int(all_tests_pass)` with partial_credit(). "
                "partial_credit() must run:  "
                "  (1) py_compile check → compile_ok  "
                "  (2) exec() in a sandboxed env → runs_without_exception  "
                "  (3) individual assert-statement execution → pass_fraction  "
                "HumanEval check() functions use a fixed structure (`candidate(args) == expected`); "
                "individual assertions can be extracted via ast.parse on the check() body. "
                "Total implementation: ~25 lines in reward function + ~20 lines in test harness. "
                "The DAPO advantage remains unchanged — the group-std identity now applies to "
                "partial-credit rewards, guaranteeing std > 0 for groups with heterogeneous "
                "partial-credit scores."
            ),
            "primary_metrics": [
                "humaneval_full_pass_at_1_post_1cycle",
                "grpo_fraction_groups_with_nonzero_std",
                "grpo_mean_within_group_std",
                "grpo_acr_equivalent_binary_reward",
                "grpo_mean_partial_credit_reward",
                "partial_credit_compilation_rate",
                "partial_credit_partial_pass_rate",
                "full_solve_rate_per_rollout",
                "variant_delta_full_pass_at_1",
            ],
            "target_outcome": (
                "Primary: full-test pass@1 >= 93.68% (+1pp over binary DAPO baseline 92.68%). "
                "ACR metric: fraction of groups with std < 0.001 drops to < 10% "
                "(vs 52.4% with binary reward). "
                "Compilation rate >= 80% across rollouts (sanity check on reward decomposition). "
                "No reward-hacking: if full-solve rate per rollout DECREASES (model chases partial "
                "credit), reduce w_pass from 0.85 to 0.95 and rerun. "
                "Variant B (partial credit + Sign advantage) expected to outperform Variant A "
                "(partial credit + DAPO) since Sign advantage eliminates remaining std=0 edge cases."
            ),
            "gpu": "auto",
            "estimated_gpu_hours": 1.5,
        },
        "gpu": "auto",
    },
    {
        "id": "exp_2026_07_10_002_oprd_local_pseudo_teacher_hidden_state_align_skills_humaneval",
        "priority": 7,
        "kind": "joint_cycle_multiseed",
        "rationale": (
            "arxiv:2606.06021 — 'OPRD: On-Policy Representation Distillation' "
            "(June 4, 2026, revised June 20, 2026). "
            ""
            "THE SKILLS GAP — OUTPUT-SPACE DISTILLATION IS A BOTTLENECK: "
            "Skills arm: 75.61% pass@1. Large model (GPT-5.5): 96.34%. Gap: 20.7pp. "
            "We run SCALING_FORCE_BOTH=1, so GPT-5.5 traces are available for every "
            "HumanEval task — the distillation DATA is not the limit. "
            "OPRD's hypothesis: the LM head is an information bottleneck. "
            "The large model's hidden representations at each layer encode: "
            "  - Variable bindings and scope tracking "
            "  - Algorithmic structure (loops, recursion patterns) "
            "  - Error-correction planning (detecting syntactic issues mid-generation) "
            "Token-level KL only sees the FINAL OUTPUT of this computation — the next "
            "token distribution. It cannot directly supervise the hidden state that "
            "produced that distribution. If the 1.5B model's hidden state doesn't "
            "represent the same algorithmic structure at mid-generation, the correct "
            "final token may still appear but for the WRONG reasons — leading to "
            "systematic failure on novel problem variations. "
            ""
            "OPRD MECHANISM: "
            "For each SFT training step: "
            "  1. Generate rollouts from the CURRENT small model (on-policy). "
            "  2. Run the TEACHER model on the SAME input to get teacher hidden states. "
            "  3. Add hidden-state alignment loss alongside standard CE loss: "
            "       L_total = L_CE + λ_align * L_hidden "
            "     where L_hidden = (1/|selected_layers|) * sum_l MSE(h_student_l, P_l(h_teacher_l)) "
            "     P_l is a learnable projection (linear layer) from teacher_dim to student_dim. "
            "  4. Standard CE loss on the large-model traces (Phase 3a SFT output). "
            ""
            "LOCAL PSEUDO-TEACHER ADAPTATION: "
            "GPT-5.5 is an API-only model — hidden states are inaccessible. "
            "This experiment uses a LOCAL PSEUDO-TEACHER: Qwen3.5-4B (available on A800, "
            "9.7× more parameters than the 1.5B student). This is NOT the same as "
            "using GPT-5.5 hidden states, but: "
            "  (a) Qwen3.5-4B is from the same model family → hidden state spaces are "
            "      compatible with a linear projection. "
            "  (b) Qwen3.5-4B at 75%+ HumanEval accuracy has better algorithmic "
            "      representations than Qwen2.5-Coder-1.5B at 75.61%. "
            "  (c) The OPRD paper shows that hidden-state alignment with a same-family "
            "      model (4B → 1.5B in code) improves distillation quality even when "
            "      the teacher's OUTPUT accuracy is only marginally better than the student's. "
            "      The hidden representations are richer than the performance gap implies. "
            "  (d) No additional API cost: Qwen3.5-4B runs locally in inference mode "
            "      alongside the 1.5B student training; A800 80GB handles both simultaneously "
            "      (4B ≈ 8 GB in bf16 inference + 1.5B ≈ 3 GB training state = ~11 GB total, "
            "       well within 80 GB). "
            ""
            "TRAINING CONFIGURATION: "
            "  - Student: Qwen2.5-Coder-1.5B-Instruct (our small model). "
            "  - Pseudo-teacher: Qwen3.5-4B (local, inference only during training). "
            "  - Hidden-state alignment layers: select every 4th layer of the student "
            "    (layers 4, 8, 12, 16 of 28 total), project onto corresponding "
            "    teacher layers with linear projection P_l (d_student=2048 → d_teacher=3072, "
            "    then MSE in teacher space). "
            "  - λ_align = 0.1 initially; if training loss diverges, reduce to 0.05. "
            "  - Alignment computed on GPT-5.5 TRACES (same data used for standard SFT), "
            "    not on fresh student rollouts — this simplifies implementation and avoids "
            "    needing an additional rollout collection pass. "
            "  - Standard CE loss: Qwen2.5-Coder-1.5B output logits vs GPT-5.5 token labels "
            "    (same as Phase 3a SFT). "
            ""
            "COMPLEMENTARITY WITH MOPD (EXP-109) AND DemoPSD (EXP-107): "
            "EXP-107 (DemoPSD): changes WHAT to distill (Category-A disagreement traces). "
            "EXP-109 (MOPD): changes HOW the PROMPT context is presented (peer rollouts). "
            "EXP-111 (OPRD): changes WHERE supervision is applied (hidden states + outputs). "
            "All three are orthogonal and could be combined in a comprehensive EXP-112: "
            "  DemoPSD selection + MOPD peer context + OPRD hidden-state alignment "
            "  → up to 4-6pp cumulative improvement if effects are additive. "
            ""
            "VARIANTS: "
            "  (A) OPRD local pseudo-teacher (Qwen3.5-4B hidden states) + CE on GPT-5.5 traces. "
            "  (B) Standard CE on GPT-5.5 traces only (baseline: current 75.61%). "
            "  (C) OPRD + DemoPSD (Category-A filter): best of EXP-107 + hidden alignment. "
            ""
            "AAAI 2027 NARRATIVE: "
            "Skills arm was stuck at 75.61% despite using GPT-5.5 teacher traces on every "
            "problem. OPRD shows that a local 4B model's hidden states provide supervision "
            "that output-KL to GPT-5.5 cannot. Key result: 'the bottleneck is not data "
            "quantity or teacher quality — it is the information compression at the LM head.' "
            "If OPRD pushes skills arm from 75.61% to ≥77.5%, the skills-large gap "
            "narrows from 20.7pp to 18.8pp — still significant, but with a clear "
            "mechanistic explanation (hidden-state subspace alignment via OPRD) and "
            "a path forward (larger local pseudo-teacher in future cycles). "
        ),
        "spec": {
            "pipeline": "humaneval_oprd_local_pseudo_teacher_hidden_state_align",
            "bench": "humaneval",
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "pseudo_teacher_model": "Qwen/Qwen3.5-4B",
            "n_cycles": 1,
            "sft_mode": "oprd_hidden_state_align",
            "oprd_align_layers": [4, 8, 12, 16],
            "oprd_projection_type": "linear",
            "oprd_lambda_align": 0.1,
            "oprd_lambda_align_fallback": 0.05,
            "sft_include_success": True,
            "scaling_force_both": True,
            "eval_arm": "skills",
            "variants": [
                "oprd_local_qwen35_4b_teacher",
                "standard_ce_baseline",
                "oprd_plus_demopsd_catA_filter",
            ],
            "eval_data": _HE_EVAL,
            "compare_with": [
                "e2e_4cyc_gpt55 cycle_3 skills arm: 75.61% HE pass@1",
                "arxiv:2606.06021 (OPRD: Pareto-dominant over output-space OPD)",
                "EXP-107 (DemoPSD: complementary WHAT-to-distill filter)",
                "EXP-109 (MOPD: complementary HOW-to-present peer context)",
            ],
            "implementation_note": (
                "In train_small_model.py: "
                "1. Load pseudo-teacher (Qwen3.5-4B) in inference mode alongside student training. "
                "   teacher_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-4B', "
                "     device_map='cuda:0', torch_dtype=torch.bfloat16) "
                "   teacher_model.eval()  # inference only, no grad "
                "2. For each training batch, run teacher forward pass to get hidden states: "
                "   with torch.no_grad(): teacher_outputs = teacher_model(**batch, output_hidden_states=True) "
                "3. Define learnable projections per selected layer (d_student→d_teacher): "
                "   projections = nn.ModuleList([nn.Linear(2048, 3072) for _ in align_layers]) "
                "4. Compute OPRD alignment loss: "
                "   L_hidden = mean(MSE(projections[i](student_hs[l]), teacher_hs[l]) "
                "                   for i, l in enumerate(align_layers)) "
                "5. Total loss: L_total = L_CE + 0.1 * L_hidden "
                "Monitor: if ||L_hidden|| > 10 in first 100 steps, reduce lambda to 0.05. "
                "Estimated additional GPU memory: ~8 GB for Qwen3.5-4B in bf16 inference "
                "(total: ~11 GB for both models; A800 80 GB has ample headroom). "
                "Estimated additional wall time: +15-20% per step (one teacher forward pass "
                "per batch with no_grad — faster than a full backward pass). "
                "VARIANT C: combine with DemoPSD by applying the Category-A filter BEFORE "
                "OPRD alignment — train on Category-A traces only, with both CE and hidden "
                "state alignment losses."
            ),
            "primary_metrics": [
                "humaneval_pass_at_1_skills_arm_post_sft",
                "sft_total_loss_curve",
                "sft_ce_loss_component",
                "sft_hidden_align_loss_component",
                "projection_layer_grad_norm",
                "skills_arm_hard_task_pass_rate",
                "skills_arm_easy_task_pass_rate",
                "variant_delta_pass_at_1",
                "sft_training_steps_per_second",
                "peak_gpu_memory_used_gb",
            ],
            "target_outcome": (
                "Primary: skills arm pass@1 >= 77.5% (+1.9pp from 75.61% baseline). "
                "Hidden-state alignment loss converges within 200 SFT steps (else reduce lambda). "
                "GPU memory stays under 60 GB total (both models + training state). "
                "Hard-task improvement > easy-task improvement (OPRD most informative where "
                "output-KL is most bottlenecked — hard algorithmic tasks). "
                "Variant C (OPRD + DemoPSD) expected to outperform variant A (OPRD alone), "
                "since DemoPSD selects the Category-A tasks where hidden-state supervision "
                "is most informative (student disagrees with teacher at output level → "
                "hidden state representations differ most → alignment loss most informative). "
            ),
            "gpu": "auto",
            "estimated_gpu_hours": 2.0,
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
