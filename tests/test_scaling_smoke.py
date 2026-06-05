"""Smoke tests for the scaling-pipeline review fixes (2026-05-21).

Covers traces_to_sft conversion and the procedural SkillBook. No GPU/API.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def test_traces_to_sft_extracts_hard_tasks(tmp_path):
    """small-fail + large-OK rows with a completion -> SFT pairs; others dropped."""
    traces = tmp_path / "traces.jsonl"
    rows = [
        # hard task: small failed, large succeeded, has completion -> kept
        {"task_id": "t1", "prompt": "do X", "small_success": False,
         "large_success": True, "large_completion": "def x(): return 1",
         "final_model": "large"},
        # small already OK -> not a hard task -> dropped
        {"task_id": "t2", "prompt": "do Y", "small_success": True,
         "large_success": False, "large_completion": "", "final_model": "small"},
        # hard task but no completion -> dropped (counted)
        {"task_id": "t3", "prompt": "do Z", "small_success": False,
         "large_success": True, "large_completion": "", "final_model": "large"},
    ]
    traces.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "sft.jsonl"
    rc = subprocess.call([sys.executable,
                          str(REPO / "experiments/scaling/traces_to_sft.py"),
                          "--traces", str(traces), "--output", str(out)])
    assert rc == 0
    pairs = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(pairs) == 1
    assert pairs[0]["task_id"] == "t1"
    assert pairs[0]["prompt"] == "do X"
    assert pairs[0]["completion"] == "def x(): return 1"
    # train_small_model.py compat keys present
    assert pairs[0]["instruction"] == "do X"
    assert pairs[0]["output"] == "def x(): return 1"


def test_skillbook_procedural_distillation(tmp_path):
    from src.skills import SkillBook

    sb = SkillBook()
    sb.update("Write a python function to sort a list", "large", True, "t1",
              completion="```python\ndef s(xs):\n    return sorted(xs)\n```")
    n = sb.distill_all()
    assert n >= 1
    # the matched cluster has an exemplar + a non-empty heuristic procedure
    sig = next(iter(sb.skills))
    sk = sb.skills[sig]
    assert sk.exemplars, "exemplar should be stored on success+completion"
    assert sk.procedure, "procedure should be distilled"
    assert sk.procedure_source == "heuristic"

    # round-trip preserves procedural fields
    p = tmp_path / "sb.json"
    sb.save(p)
    sb2 = SkillBook()
    sb2.load(p)
    assert all(s.procedure for s in sb2.skills.values())


def test_skillbook_update_backward_compatible():
    """update() without completion still works (stats-only path)."""
    from src.skills import SkillBook
    sb = SkillBook()
    sb.update("some prompt", "small", True, "t1")  # no completion kwarg
    sig = next(iter(sb.skills))
    assert sb.skills[sig].stats["small"] == [1, 1]
    assert sb.skills[sig].exemplars == []


def test_predict_skills_reads_real_schema(tmp_path):
    """tofix.md #5: predict_skills must parse {'skills':[...]} with stats roles."""
    from src.skills import SkillBook
    import importlib.util
    sb = SkillBook()
    # cluster where large clearly beats small -> needs_large True
    for _ in range(5):
        sb.update("translate this regex pattern please", "large", True)
    sb.update("translate this regex pattern please", "small", False)
    p = tmp_path / "skillbook.json"
    sb.save(p)
    # import the ablation module
    spec = importlib.util.spec_from_file_location(
        "rea", REPO / "experiments/scaling/run_e2e_ablation_simple.py")
    rea = importlib.util.module_from_spec(spec); spec.loader.exec_module(rea)
    sig = next(iter(sb.skills))
    preds = rea.predict_skills([{"signature": sig}], p)
    assert preds == [1], "large-dominant cluster should route large (not silently always-small)"


def test_skillbook_canonical_roles():
    """tofix.md #2: stats keyed by role survive across 'adapter path' changes."""
    from src.skills import SkillBook
    sb = SkillBook()
    # cycle 0 + cycle 1 both record under canonical 'small'
    sb.update("count the vowels in a string", "small", True, "t1")
    sb.update("count the vowels in a string", "small", True, "t2")
    sig = next(iter(sb.skills))
    assert sb.skills[sig].can_downgrade_to_small("small", min_rate=0.8, min_samples=1) is True


def test_traces_to_sft_injects_procedure(tmp_path):
    """tofix.md #3: a cluster procedure is prepended to the SFT prompt."""
    import subprocess
    from src.skills import SkillBook
    # build a skillbook with a procedure for the cluster of our hard task
    sb = SkillBook()
    sb.update("sort a list of numbers in python", "large", True, "x1",
              completion="```python\ndef s(xs):\n    return sorted(xs)\n```")
    sb.distill_all()
    skb = tmp_path / "skillbook.json"; sb.save(skb)
    # a hard task in the same cluster
    traces = tmp_path / "traces.jsonl"
    traces.write_text(json.dumps({
        "task_id": "h1", "prompt": "sort a list of numbers in python",
        "signature": next(iter(sb.skills)),
        "small_success": False, "large_success": True,
        "large_completion": "def s(xs): return sorted(xs)", "final_model": "large",
    }) + "\n")
    out = tmp_path / "sft.jsonl"
    rc = subprocess.call([sys.executable, str(REPO / "experiments/scaling/traces_to_sft.py"),
                          "--traces", str(traces), "--output", str(out), "--skillbook", str(skb)])
    assert rc == 0
    row = json.loads(out.read_text().splitlines()[0])
    assert row["has_procedure"] is True
    assert "Procedure for cluster" in row["prompt"]
    assert "sort a list of numbers in python" in row["prompt"]


def test_completion_from_steps_uses_stepdata_response():
    """review 2026-06-05: SFT completion must come from steps[].response
    (content + tool_calls), since TaskRunResult.messages is empty on live tau2."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tau2ad", REPO / "experiments/scaling/benches/tau2_bench/adapter.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

    class FakeStep:
        def __init__(self, response): self.response = response

    steps = [
        FakeStep({"content": "look up the order", "tool_calls": [
            {"function": {"name": "lookup_order", "arguments": '{"id": 42}'}}]}),
        FakeStep({"content": "", "tool_calls": [
            {"function": {"name": "issue_refund", "arguments": '{"order": 42}'}}]}),
        FakeStep({"content": "done", "tool_calls": []}),
    ]
    out = m._completion_from_steps(steps)
    assert "look up the order" in out
    assert "<tool_call>lookup_order(" in out
    assert "<tool_call>issue_refund(" in out
    assert "done" in out
    # JSON dict-form steps + empty input
    assert m._completion_from_steps([{"response": {"content": "hi", "tool_calls": []}}]) == "hi"
    assert m._completion_from_steps([]) == ""
