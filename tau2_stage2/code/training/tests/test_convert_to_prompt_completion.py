"""Tests for stage-2 row → TRL prompt/completion conversion (Spec §5.1)."""
import json
from pathlib import Path

from training.data.convert_to_prompt_completion import (
    convert_row,
    convert_dataset,
    load_domain_assets,
)


SAMPLE_ROW = {
    "_p": {"row_id": "airline_0_300_cs__step_1__locked",
           "domain": "airline", "task_id": "0", "step_1based": 1, "phase": "locked",
           "step_depth_frac": 0.0833, "epoch_alias": "cs", "seed": 300},
    "_system_ref": "domain_assets/airline_system.txt",
    "_tools_ref": "domain_assets/airline_tools.json",
    "_target_index": 0,
    "messages": [{"role": "assistant", "content": "Hi! How can I help you today?"}],
}

SAMPLE_ROW_DEEP = {
    "_p": {"row_id": "telecom_5_300_cs__step_3__locked",
           "domain": "telecom", "task_id": "5", "step_1based": 3, "phase": "locked",
           "step_depth_frac": 0.5, "epoch_alias": "cs", "seed": 300},
    "_system_ref": "domain_assets/telecom_system.txt",
    "_tools_ref": "domain_assets/telecom_tools.json",
    "_target_index": 4,
    "messages": [
        {"role": "assistant", "content": "Hello, how can I help?"},
        {"role": "user", "content": "I need help with my plan."},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "tc1", "name": "get_plan", "arguments": {"user_id": "u1"}}]},
        {"role": "tool", "tool_call_id": "tc1", "content": '{"plan": "premium"}'},
        {"role": "assistant", "content": "Your plan is Premium."},
    ],
}


def test_convert_row_step1():
    """Step 1 has empty prompt prefix; completion is the single message."""
    out = convert_row(SAMPLE_ROW, system_text="SYS", tools=[])
    assert out["prompt"][0]["role"] == "system"
    assert out["prompt"][0]["content"] == "SYS"
    assert len(out["prompt"]) == 1
    assert len(out["completion"]) == 1
    assert out["completion"][0]["role"] == "assistant"
    assert out["completion"][0]["content"] == "Hi! How can I help you today?"
    assert out["tools"] == []
    assert out["_meta"]["row_id"] == SAMPLE_ROW["_p"]["row_id"]


def test_convert_row_deep():
    """Deeper step has prior history as prompt, target as completion."""
    out = convert_row(SAMPLE_ROW_DEEP, system_text="SYS", tools=[{"x": 1}])
    assert len(out["prompt"]) == 5  # system + 4 prior
    assert out["prompt"][0]["content"] == "SYS"
    assert out["prompt"][-1]["role"] == "tool"
    assert len(out["completion"]) == 1
    assert out["completion"][0]["content"] == "Your plan is Premium."
    assert out["tools"] == [{"x": 1}]


def test_convert_row_target_with_tool_calls():
    """Tool-call target preserves tool_calls in completion (real τ²-bench flat shape)."""
    row = {
        "_p": {"row_id": "x", "domain": "airline", "task_id": "1",
               "step_1based": 1, "phase": "locked",
               "step_depth_frac": 0.1, "epoch_alias": "cs", "seed": 300},
        "_system_ref": "x", "_tools_ref": "x", "_target_index": 0,
        "messages": [
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "t1", "name": "foo", "arguments": {"a": 1}}]},
        ],
    }
    out = convert_row(row, system_text="SYS", tools=[])
    assert "tool_calls" in out["completion"][0]
    assert out["completion"][0]["tool_calls"][0]["id"] == "t1"


def test_convert_dataset_end_to_end(tmp_path: Path):
    """End-to-end: write input jsonl with a valid (user-prefixed) deep row, convert, read output."""
    src = tmp_path / "src.jsonl"
    da = tmp_path / "domain_assets"
    da.mkdir()
    (da / "telecom_system.txt").write_text("TELECOM SYS")
    (da / "telecom_tools.json").write_text(json.dumps([{"name": "tool_a"}]))

    # SAMPLE_ROW is the user-less greeting pattern (the 257-row case the
    # converter now filters); use SAMPLE_ROW_DEEP for the kept-row path.
    src.write_text(json.dumps(SAMPLE_ROW_DEEP) + "\n")
    dst = tmp_path / "out.jsonl"

    stats = convert_dataset(src_jsonl=src, dst_jsonl=dst, domain_assets_dir=da)
    assert stats["n_kept"] == 1
    assert stats["n_dropped_no_user"] == 0
    out_rows = [json.loads(l) for l in dst.read_text().splitlines()]
    assert len(out_rows) == 1
    assert out_rows[0]["prompt"][0]["content"] == "TELECOM SYS"
    assert out_rows[0]["tools"] == [{"name": "tool_a"}]


def test_convert_dataset_drops_user_less_row(tmp_path: Path):
    """Rows whose prompt-prefix has no user message are dropped + logged."""
    src = tmp_path / "src.jsonl"
    da = tmp_path / "domain_assets"
    da.mkdir()
    (da / "airline_system.txt").write_text("AIRLINE SYS")
    (da / "airline_tools.json").write_text(json.dumps([{"name": "tool_a"}]))
    (da / "telecom_system.txt").write_text("TELECOM SYS")
    (da / "telecom_tools.json").write_text(json.dumps([{"name": "tool_t"}]))

    # SAMPLE_ROW: ti=0, prefix=[], no user → dropped.
    # SAMPLE_ROW_DEEP: ti=4, prefix has a user → kept.
    src.write_text(
        json.dumps(SAMPLE_ROW) + "\n" + json.dumps(SAMPLE_ROW_DEEP) + "\n"
    )
    dst = tmp_path / "out.jsonl"
    drop_log = tmp_path / "dropped.jsonl"
    stats = convert_dataset(
        src_jsonl=src,
        dst_jsonl=dst,
        domain_assets_dir=da,
        drop_log=drop_log,
    )
    assert stats == {"n_kept": 1, "n_dropped_no_user": 1}
    out_rows = [json.loads(l) for l in dst.read_text().splitlines()]
    assert len(out_rows) == 1
    assert out_rows[0]["completion"][0]["content"] == "Your plan is Premium."
    dropped = [json.loads(l) for l in drop_log.read_text().splitlines()]
    assert len(dropped) == 1
    assert dropped[0]["reason"] == "no_user_in_prefix"
    assert dropped[0]["_target_index"] == 0
    assert dropped[0]["prefix_roles"] == []


def test_load_domain_assets(tmp_path: Path):
    da = tmp_path
    (da / "airline_system.txt").write_text("AIR")
    (da / "airline_tools.json").write_text(json.dumps([{"name": "t"}]))
    sys_texts, tools = load_domain_assets(da)
    assert sys_texts["airline"] == "AIR"
    assert tools["airline"] == [{"name": "t"}]
