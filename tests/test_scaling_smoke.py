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
