"""
Pending queue update — 2026-09-16
Run on A800 when connectivity is restored:
  python /data0/home/zeyuwang/auto_research/pending_queue_update_2026_09_16.py
Appends EXP-214 and EXP-215 to state["queue"] and saves atomically.

A800 offline since 2026-05-14 (day ~124). SSH port 50507 unreachable from remote
execution environment (TCP timeout; proxy is HTTPS-only and cannot tunnel raw SSH).
Queue ~211 pending (>20 cap → 2 experiments today).
Next target: ICLR 2027 (~Oct 1 deadline, ~15 days out). URGENT.
Both experiments are OFFLINE / 0h GPU (paper positioning analyses for §4/§2).

Hotspot source: WebSearch fallback (A800 hotspot file unavailable — A800 offline).
Top new papers found (this run, 2026-09-16):
  arxiv:2609.01345  "Cheap Verifiers, Large Blind Spots: Measuring the Reliability
    Cost of Cost-Saving Cascades" (Sep 1, 2026, Rajput) — verifier blind-spot β
    grows 0.12 → 0.55 as student scales 0.5B → 32B. MERA's pytest verifier has
    β=0 by construction (deterministic pass/fail). Key ICLR reviewer risk: no
    mention in current paper. → EXP-214: pytest β=0 §4/§7 Positioning Audit.
  arxiv:2609.11446  "Calibration-Aware Uncertainty Cascades for Efficient
    Heterogeneous Model Collaboration" (Sep 10, 2026, Zhang et al.) — CAUC post-hoc
    calibration gives a common reliability scale; must be re-run after each model
    update. MERA contrast: oracle-label training gives end-to-end calibration that
    recalibrates each cycle. Tagged ★★ in Sep 12 hotspot; not yet acted on (queue
    cap hit). → EXP-215: CAUC §2 LLM Routing Positioning Audit.
  arxiv:2509.22984  "From Deferral to Learning: Online In-Context KD for LLM
    Cascades" (Sep 2025, Wu et al.) — zero-parameter ICL distillation; requires
    O(N) trace cache + retrieval overhead. MERA contrast: SFT bakes knowledge into
    weights (zero inference overhead, structural generalization). Background cite
    candidate; incorporated into EXP-215's audit as bib entry intercascade2025.

Apply chain before this patch:
    python3 auto_research/pending_queue_update_2026_09_11.py  # EXP-210, EXP-211
    python3 auto_research/pending_queue_update_2026_09_12.py  # EXP-212, EXP-213
"""

import json
import os
import shutil

STATE_PATH = "/data0/home/zeyuwang/auto_research/state.json"

NEW_EXPERIMENTS = [
    {
        "id": "EXP-214",
        "priority": 8,
        "title": (
            "Cheap Verifiers Blind Spots §4/§7 Positioning: MERA pytest β≈0 "
            "Advantage vs. LLM-Judge Cascades for ICLR 2027 (arxiv:2609.01345)"
        ),
        "paper": "arxiv:2609.01345",
        "paper_title": (
            "Cheap Verifiers, Large Blind Spots: Measuring the Reliability "
            "Cost of Cost-Saving Cascades"
        ),
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": (
                "paper auto_research/paper/paper.md — §4.1 Trace Collection "
                "   (run-both oracle, SCALING_FORCE_BOTH=1 design), §7 Limitations. "
                "src/pipeline/collect_traces.py — _policy_decision, oracle label "
                "   construction (label = small_fails AND large_succeeds; both evaluated "
                "   by running pytest on generated code). "
                "CLAUDE.md — 'SCALING_FORCE_BOTH=1 for the canonical run' gotcha; "
                "   pytest reward signal (binary pass/fail). "
                "results/e2e_4cyc_gpt55/ — routing oracle label distribution across "
                "   cycles 0–3; verify that oracle label flip rate (small_pass→label=1) "
                "   is near zero across cycles (i.e., gpt-5.5 almost never fails tasks "
                "   where Qwen3-4B passes — the zero-harm property)."
            ),
            "metric": (
                "arxiv:2609.01345 (Sep 1, 2026) documents the Cheap Verifier Trap in "
                "cost-saving LLM cascades: when the student is fine-tuned on the verifier's "
                "rejections, the verifier's blind spot (β = fraction of student wrong answers "
                "the verifier accepts as correct) grows adversarially. β scales from 0.12 at "
                "0.5B student to 0.55 at 32B student — the better the student, the larger the "
                "fraction of errors that slip past the verifier. A frontier verifier (GPT-4o) "
                "drives β to ~0.05 but then escalates on 46% of hard-MATH queries for a 39% "
                "true error rate — eliminating much of the cost savings. "
                "MERA intersection: "
                "(1) β=0 by construction. MERA's verifier is pytest — deterministic binary "
                "    execution. A code solution is correct iff all test cases pass; there is "
                "    no model-in-the-loop that can hallucinate acceptance of wrong code. "
                "    As Qwen3-4B improves over N cycles, the pytest oracle never develops a "
                "    blind spot: tests written for HumanEval are fixed, not generated. "
                "(2) SCALING_FORCE_BOTH=1 as a blind-spot avoidance strategy. The canonical "
                "    run runs gpt-5.5 on EVERY task to get the ground-truth oracle labels "
                "    (small_fails AND large_succeeds). Without SCALING_FORCE_BOTH, MERA only "
                "    runs the large model on tasks the small model FAILED — a cost-saving "
                "    shortcut that risks using small-model-accepted traces (potential blind "
                "    spots if the pytest runner has any flakiness) as positive examples. "
                "    SCALING_FORCE_BOTH eliminates this risk entirely. "
                "(3) Contrast with LLM-judge cascades (the paper's target). For open-ended "
                "    tasks where no executable verifier exists, LLM judges are unavoidable "
                "    and the trap applies. MERA's scope is intentionally restricted to tasks "
                "    with deterministic verifiers (HumanEval, tau2 code tasks). "
                "Audit tasks: "
                "  (a) Search paper.md §4.1 Trace Collection: does it mention the "
                "      deterministic nature of the pytest oracle? If not, add: "
                "      'Unlike LLM-judge cascade systems, where verifier blind spots grow "
                "      with student capability [cheapverifiers2026], MERA's pytest oracle is "
                "      deterministic — a solution is correct iff its unit tests pass — "
                "      ensuring β=0 across all training cycles.' "
                "  (b) Check §7 Limitations: is there a sentence noting the scope restriction "
                "      to executable-test domains? If not, add one: 'MERA assumes access to "
                "      deterministic verifiers (unit tests); tasks lacking auto-gradable test "
                "      suites would require LLM judges, reintroducing verifier blind spots [cite].' "
                "  (c) Check §4.1 SCALING_FORCE_BOTH=1 rationale: can cite 2609.01345 as "
                "      motivation for always running both models (avoids relying on small-model "
                "      acceptance as a learning signal). "
                "  (d) Add bib entry: cheapverifiers2026. "
                "  (e) As background, check if intercascade2025 (arxiv:2509.22984) is in §2 "
                "      Related Work; if not, add as a zero-parameter-update contrast cite. "
                "Output: §4.1 sentence + §7 limitation sentence + 1-2 bib entries."
            ),
            "estimated_gpu_hours": 0,
            "expected_paper_impact": (
                "Closes an ICLR reviewer risk: a reviewer familiar with the Cheap Verifier "
                "reliability literature (2609.01345 is a Sep 1 paper in a hot area) could ask "
                "why MERA's SFT loop doesn't suffer from blind-spot corruption. The answer "
                "(pytest is deterministic, β=0) is obvious to practitioners but currently "
                "absent from the paper. Adds theoretical grounding for two MERA design "
                "decisions (execution-based reward, SCALING_FORCE_BOTH=1). "
                "Soundness +0.1; Related-Work coverage +0.1; new Sep 2026 citation."
            ),
        },
        "gpu": "auto",
    },
    {
        "id": "EXP-215",
        "priority": 7,
        "title": (
            "CAUC Calibration §2 LLM Routing Positioning Audit: Post-Hoc Static "
            "Calibration vs. MERA Cycle-Wise Oracle Recalibration for ICLR 2027 "
            "(arxiv:2609.11446)"
        ),
        "paper": "arxiv:2609.11446",
        "paper_title": (
            "Calibration-Aware Uncertainty Cascades for Efficient Heterogeneous "
            "Model Collaboration"
        ),
        "kind": "forgetting_eval",
        "gpu": False,
        "spec": {
            "type": "offline_analysis",
            "data_source": (
                "paper auto_research/paper/paper.md — §2 LLM Routing; look for "
                "   existing cites around SRR (2609.07786), UCCI (2605.18796), "
                "   calibration-based routing, and uncertainty-based cascade routing. "
                "src/pipeline/train_router_simple.py — router training: logistic "
                "   regression on oracle labels (raw prompt features, no post-hoc "
                "   calibration step). Router re-trained each cycle on fresh oracle labels. "
                "results/e2e_4cyc_gpt55/ — routing accuracy per cycle (cycle-0: ~80%, "
                "   cycle-3: ~93.04%); shows router calibration improves cycle-wise "
                "   without explicit calibration procedure."
            ),
            "metric": (
                "arxiv:2609.11446 (CAUC, Sep 10 2026) proposes Calibration-Aware Uncertainty "
                "Cascades: each model in a heterogeneous pool is independently calibrated via "
                "post-hoc temperature scaling on held-out validation data. The calibrated "
                "confidence scores establish a common reliability scale, decoupled from any "
                "particular model pool or operating budget. Policy (accept early / invoke strong "
                "/ combine outputs) is then selected using calibrated thresholds on this common "
                "scale. Results on 6 LLM benchmarks + 3 image classification datasets show "
                "superior performance-cost trade-offs. "
                "MERA intersection: "
                "(1) End-to-end vs. post-hoc. MERA trains a logistic router on oracle labels "
                "    (ground-truth correctness from pytest runs), which ARE the calibration "
                "    signal — no separate post-hoc step needed. CAUC requires an additional "
                "    validation-data calibration step each time any model is updated. "
                "(2) Cycle-wise recalibration. MERA's router is retrained each cycle as the "
                "    student's capability distribution shifts (more tasks move from hard to "
                "    easy as Qwen3-4B improves). This is an online, end-to-end analog of "
                "    CAUC's offline re-calibration procedure. CAUC in a multi-cycle setting "
                "    would require re-running temperature scaling after every student SFT/GRPO "
                "    update — operationally costly. MERA absorbs this implicitly into the "
                "    oracle-label collection phase (Phase 1) and router training (Phase 4). "
                "(3) Common reliability scale. CAUC's value proposition is a unified "
                "    confidence scale across heterogeneous models. MERA achieves the same "
                "    via oracle labels: the label is 1 iff the task is beyond the current "
                "    small model's capability, regardless of model architecture — it is "
                "    calibrated to actual pass/fail, not predicted confidence. "
                "Tagged ★★ in the Sep 12 hotspot but not acted on (queue cap hit that run). "
                "Audit tasks: "
                "  (a) Find §2 LLM Routing in paper.md. Locate the cite cluster covering "
                "      uncertainty-based and calibration-based routing (check for UCCI "
                "      arxiv:2605.18796, SRR arxiv:2609.07786, RouteJudge arxiv:2606.18774). "
                "  (b) Draft a positioning sentence for §2: "
                "      'Calibration-based approaches [CAUC, arxiv:2609.11446; UCCI, "
                "      arxiv:2605.18796] post-hoc rescale model confidence to a common "
                "      reliability axis using held-out validation data; re-calibration is "
                "      required whenever any model in the pool changes. MERA instead trains "
                "      an end-to-end router on oracle correctness labels — the ground-truth "
                "      reliability signal — with automatic cycle-wise recalibration as the "
                "      student's capability distribution shifts.' "
                "  (c) Add bib entry cauc2026 (arxiv:2609.11446). "
                "  (d) Check if arxiv:2509.22984 (Inter-Cascade, Sep 2025) is in §2 "
                "      Related Work. If not, add as background cite intercascade2025 with "
                "      one-sentence description: 'Inter-Cascade [arxiv:2509.22984] enables "
                "      online in-context KD without parameter updates; MERA instead bakes "
                "      the large model's solutions into the student's weights via SFT+GRPO, "
                "      adding zero inference latency and generalizing to structurally new prompts.' "
                "  (e) Check §2 for CAUC companion UCCI (2605.18796): if not cited, add. "
                "Output: §2 positioning sentence + 1-2 bib entries (cauc2026, optionally "
                "intercascade2025, ucci2026)."
            ),
            "estimated_gpu_hours": 0,
            "expected_paper_impact": (
                "§2 LLM Routing gains a Sep 10 2026 citation (CAUC) and a principled "
                "distinction between post-hoc calibration approaches and MERA's end-to-end "
                "oracle-label routing. Closes a related-work gap reviewers are likely to "
                "flag given the active Sep 2026 calibration-cascade literature. "
                "Pairing with SRR (EXP-212) and UCCI gives MERA a coherent cluster of "
                "Sep 2026 routing citations that demonstrates engagement with the latest work. "
                "Soundness 0, Related-Work coverage +0.1; 1-2 new citations."
            ),
        },
        "gpu": "auto",
    },
]


def main():
    if not os.path.exists(STATE_PATH):
        print(f"ERROR: state.json not found at {STATE_PATH}")
        return

    # Atomic read-modify-write
    tmp_path = STATE_PATH + ".tmp_0916"
    with open(STATE_PATH, "r") as f:
        state = json.load(f)

    existing_ids = set()
    for item in state.get("queue", []):
        existing_ids.add(item.get("id"))
    for item in state.get("history", []):
        existing_ids.add(item.get("id"))

    added = []
    for exp in NEW_EXPERIMENTS:
        if exp["id"] in existing_ids:
            print(f"SKIP {exp['id']} — already in queue or history")
        else:
            state.setdefault("queue", []).append(exp)
            added.append(exp["id"])
            print(f"ADDED {exp['id']} (priority={exp['priority']}): {exp['title'][:60]}...")

    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    shutil.move(tmp_path, STATE_PATH)

    print(f"\nDone. Added {len(added)} experiments: {added}")
    print(f"Queue length: {len(state.get('queue', []))}")


if __name__ == "__main__":
    main()
