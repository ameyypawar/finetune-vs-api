"""The chat-JSONL validator: accepts good files, flags invalid and unsupported ones."""

from __future__ import annotations

import json

import pytest

from conftest import ex, load_script
from finetune_vs_api.data import (
    chat_to_prompt_completion,
    read_ids,
    sidecar_path,
    validate_chat_jsonl,
    write_chat_jsonl,
)
from finetune_vs_api.prompts import finetune_record

SYSTEM = {"role": "system", "content": "Convert the request."}
USER = {"role": "user", "content": "wake me at six"}
ASSISTANT = {"role": "assistant", "content": '{"intent":"alarm_set","slots":[]}'}


def write(tmp_path, *records, raw=None, name="chat.jsonl"):
    path = tmp_path / name
    lines = [json.dumps(r) if not isinstance(r, str) else r for r in records]
    path.write_text("\n".join(lines) + ("\n" if lines else "") if raw is None else raw, encoding="utf-8")
    return path


def kinds(report):
    return sorted({i.kind for i in report.issues})


def messages_of(report):
    return " | ".join(i.message for i in report.issues)


# --- accepted ------------------------------------------------------------------------------


def test_a_good_file_is_accepted(tmp_path):
    path = write(tmp_path, {"messages": [SYSTEM, USER, ASSISTANT]}, {"messages": [USER, ASSISTANT]})
    report = validate_chat_jsonl(path)
    assert report.ok
    assert report.n_records == 2
    assert report.issues == []


def test_blank_lines_are_skipped_and_line_numbers_are_real(tmp_path):
    good = json.dumps({"messages": [USER, ASSISTANT]})
    path = write(tmp_path, raw=f"{good}\n\n{good}\nnot json\n")
    report = validate_chat_jsonl(path)
    assert report.n_records == 3
    assert [(i.line, i.kind) for i in report.issues] == [(4, "error")]


def test_the_files_this_repo_writes_validate(tmp_path):
    records = [finetune_record(ex(i, "train", "alarm_set", f"wake me at {i}", [("time", str(i))])) for i in range(5)]
    path = tmp_path / "sft.jsonl"
    write_chat_jsonl(path, records, ids=[str(i) for i in range(5)])
    assert validate_chat_jsonl(path).ok
    assert read_ids(sidecar_path(path)) == ["0", "1", "2", "3", "4"]


# --- invalid -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("record", "fragment"),
    [
        ("not json at all", "invalid JSON"),
        ([1, 2], "not a JSON object"),
        ({"foo": 1}, "'messages' must be a non-empty list"),
        ({"messages": []}, "'messages' must be a non-empty list"),
        ({"messages": "hi"}, "'messages' must be a non-empty list"),
        ({"messages": [USER, {"role": "robot", "content": "x"}]}, "invalid role"),
        ({"messages": [USER, {"role": "assistant", "content": 42}]}, "content must be a string"),
        ({"messages": [USER, {"role": "assistant"}]}, "content must be a string"),
        ({"messages": [USER, {"role": "assistant", "content": "  "}]}, "empty content"),
        ({"messages": [ASSISTANT, USER]}, "last message must be from the assistant"),
        ({"messages": [USER]}, "last message must be from the assistant"),
        ({"messages": [ASSISTANT]}, "no user message"),
        ({"messages": [USER, SYSTEM, ASSISTANT]}, "system message may only come first"),
        ({"messages": ["just a string", ASSISTANT]}, "is not an object"),
    ],
)
def test_invalid_records_are_errors(tmp_path, record, fragment):
    report = validate_chat_jsonl(write(tmp_path, record))
    assert not report.ok
    assert fragment in messages_of(report)
    assert report.errors


def test_an_empty_file_is_an_error(tmp_path):
    report = validate_chat_jsonl(write(tmp_path, raw=""))
    assert not report.ok
    assert "no records" in messages_of(report)


# --- unsupported in v1 ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("record", "fragment"),
    [
        ({"messages": [USER, ASSISTANT], "tools": [{"type": "function"}]}, "'tools'"),
        ({"messages": [USER, ASSISTANT], "functions": []}, "'functions'"),
        ({"messages": [USER, ASSISTANT], "parallel_tool_calls": True}, "'parallel_tool_calls'"),
        (
            {"messages": [USER, {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]}]},
            "'tool_calls'",
        ),
        ({"messages": [USER, {"role": "tool", "content": "x", "tool_call_id": "1"}, ASSISTANT]}, "role 'tool'"),
        (
            {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}, ASSISTANT]},
            "multimodal",
        ),
        (
            {"messages": [USER, {"role": "assistant", "content": "x", "weight": 0}]},
            "'weight'",
        ),
        ({"messages": [USER, ASSISTANT, USER, ASSISTANT]}, "multi-turn"),
    ],
)
def test_valid_but_unsupported_features_are_flagged(tmp_path, record, fragment):
    report = validate_chat_jsonl(write(tmp_path, record))
    assert not report.ok  # a file v1 cannot use is not "OK"
    assert fragment in messages_of(report)
    assert report.unsupported
    assert not report.errors  # valid OpenAI data that v1 cannot use, not malformed data


def test_multi_turn_is_unsupported_rather_than_invalid(tmp_path):
    report = validate_chat_jsonl(write(tmp_path, {"messages": [USER, ASSISTANT, USER, ASSISTANT]}))
    assert kinds(report) == ["unsupported"]


def test_extra_top_level_keys_only_warn(tmp_path):
    report = validate_chat_jsonl(write(tmp_path, {"id": "7", "messages": [USER, ASSISTANT]}))
    assert report.ok
    assert kinds(report) == ["warning"]
    assert "'id'" in messages_of(report)


# --- conversion to TRL's prompt/completion format --------------------------------------------


def test_chat_to_prompt_completion_splits_off_the_assistant_turn():
    out = chat_to_prompt_completion({"messages": [SYSTEM, USER, ASSISTANT]})
    assert out == {"prompt": [SYSTEM, USER], "completion": [ASSISTANT]}


def test_chat_to_prompt_completion_without_a_system_message():
    assert chat_to_prompt_completion({"messages": [USER, ASSISTANT]}) == {
        "prompt": [USER],
        "completion": [ASSISTANT],
    }


@pytest.mark.parametrize(
    "record",
    [
        {"messages": [USER, ASSISTANT, USER, ASSISTANT]},
        {"messages": [USER]},
        {"messages": [USER, ASSISTANT], "tools": []},
        {"messages": [USER, {"role": "assistant", "content": "x", "weight": 1}]},
    ],
)
def test_chat_to_prompt_completion_refuses_what_the_validator_flags(record):
    with pytest.raises(ValueError, match="cannot convert"):
        chat_to_prompt_completion(record)


def test_chat_to_prompt_completion_drops_extra_message_keys():
    record = {"messages": [{"role": "user", "content": "hi", "name": "bob"}, ASSISTANT]}
    assert chat_to_prompt_completion(record)["prompt"] == [{"role": "user", "content": "hi"}]


# --- writing ---------------------------------------------------------------------------------


def test_write_chat_jsonl_checks_the_ids_length(tmp_path):
    with pytest.raises(ValueError, match="ids"):
        write_chat_jsonl(tmp_path / "x.jsonl", [{"messages": [USER, ASSISTANT]}], ids=["1", "2"])


# --- the script ------------------------------------------------------------------------------


def test_script_exit_codes_and_report(tmp_path):
    script = load_script("validate_chat_jsonl")
    lines: list[str] = []
    good = write(tmp_path, {"messages": [USER, ASSISTANT]}, name="good.jsonl")
    assert script.run(good, out=lines.append) == 0
    assert lines[-1] == "OK"

    lines.clear()
    bad = write(tmp_path, {"messages": [USER, ASSISTANT, USER, ASSISTANT]}, {"foo": 1}, name="bad.jsonl")
    assert script.run(bad, out=lines.append) == 1
    text = "\n".join(lines)
    assert "unsupported in v1" in text and "errors" in text and "NOT USABLE" in text

    lines.clear()
    assert script.run(tmp_path / "missing.jsonl", out=lines.append) == 2
    assert script.main([str(good)]) == 0
