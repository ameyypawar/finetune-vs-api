"""The output contract: inventory, strict JSON Schema, serialization and parsing."""

from __future__ import annotations

import json

import pytest

from conftest import ex
from finetune_vs_api.schema import (
    LabelInventory,
    is_schema_valid,
    json_schema,
    label_inventory,
    parse_output,
    response_format,
    target_json,
)

INTENTS = ["alarm_set", "play_music"]
SLOTS = ["artist_name", "time"]
INV = LabelInventory(tuple(INTENTS), tuple(SLOTS))


def test_label_inventory_is_sorted_and_unique():
    train = [
        ex(1, "train", "play_music", "play x", [("artist_name", "x")]),
        ex(2, "train", "alarm_set", "wake me at six", [("time", "six")]),
        ex(3, "train", "alarm_set", "alarm at seven", [("time", "seven")]),
    ]
    inv = label_inventory(train)
    assert inv.intents == ("alarm_set", "play_music")
    assert inv.slot_types == ("artist_name", "time")


def test_label_inventory_refuses_anything_but_train():
    with pytest.raises(ValueError, match="train examples only"):
        label_inventory([ex(1, "train", "a", "x"), ex(2, "dev", "b", "y")])


def _walk_objects(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            yield schema
        for value in schema.values():
            yield from _walk_objects(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from _walk_objects(value)


def test_schema_satisfies_openai_strict_mode_rules():
    schema = json_schema(INTENTS, SLOTS)
    objects = list(_walk_objects(schema))
    assert len(objects) == 2  # the call and each slot
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert sorted(obj["required"]) == sorted(obj["properties"])  # every field required


def test_schema_constrains_labels_with_enums():
    schema = json_schema(INTENTS, SLOTS)
    assert schema["properties"]["intent"]["enum"] == INTENTS
    assert schema["properties"]["slots"]["items"]["properties"]["type"]["enum"] == SLOTS
    json.dumps(schema)  # serializable


def test_response_format_wraps_the_schema_for_chat_completions():
    fmt = response_format(INTENTS, SLOTS)
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] == json_schema(INTENTS, SLOTS)


def test_target_json_is_compact_ordered_and_keeps_slot_order():
    example = ex(1, "train", "alarm_set", "wake me at six on friday", [("time", "six"), ("date", "friday")])
    text = target_json(example)
    assert text == '{"intent":"alarm_set","slots":[{"type":"time","value":"six"},{"type":"date","value":"friday"}]}'
    assert " " not in text.replace("six on", "")  # no padding whitespace


def test_target_json_for_no_slots_and_round_trip():
    example = ex(1, "train", "general_greet", "hello there")
    assert target_json(example) == '{"intent":"general_greet","slots":[]}'
    assert parse_output(target_json(example)) == {"intent": "general_greet", "slots": []}


def test_target_json_keeps_non_ascii():
    assert "café" in target_json(ex(1, "train", "x", "café", [("place_name", "café")]))


@pytest.mark.parametrize(
    "text",
    [
        '{"intent": "a", "slots": []}',
        '  \n{"intent": "a", "slots": []}\n ',
        '```json\n{"intent": "a", "slots": []}\n```',
        '```\n{"intent": "a", "slots": []}\n```',
    ],
)
def test_parse_output_accepts_json_whitespace_and_one_code_fence(text):
    assert parse_output(text) == {"intent": "a", "slots": []}


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "not json",
        'Sure! {"intent": "a", "slots": []}',  # prose before
        '{"intent": "a", "slots": []} hope that helps',  # prose after
        '{"intent": "a", "slots": []}{"intent": "b", "slots": []}',  # two objects
        "[1, 2, 3]",  # not an object
        '"a string"',
        '{"intent": "a", "slots": [}',  # broken
    ],
)
def test_parse_output_does_not_repair_format_failures(text):
    assert parse_output(text) is None


GOOD = {"intent": "alarm_set", "slots": [{"type": "time", "value": "six"}]}


def test_valid_object_passes_with_and_without_inventory():
    assert is_schema_valid(GOOD)
    assert is_schema_valid(GOOD, INV)
    assert is_schema_valid({"intent": "alarm_set", "slots": []}, INV)


@pytest.mark.parametrize(
    "obj",
    [
        None,
        [],
        {},
        {"intent": "alarm_set"},  # slots missing
        {"slots": []},  # intent missing
        {**GOOD, "confidence": 0.9},  # extra key
        {"intent": 5, "slots": []},
        {"intent": "alarm_set", "slots": "none"},
        {"intent": "alarm_set", "slots": [{"type": "time"}]},  # slot missing value
        {"intent": "alarm_set", "slots": [{"type": "time", "value": "six", "x": 1}]},
        {"intent": "alarm_set", "slots": [{"type": "time", "value": 6}]},
        {"intent": "alarm_set", "slots": ["time"]},
    ],
)
def test_structurally_invalid_objects_fail(obj):
    assert not is_schema_valid(obj)


def test_inventory_enforces_the_enums():
    assert not is_schema_valid({"intent": "alarm_create", "slots": []}, INV)
    assert not is_schema_valid({"intent": "alarm_set", "slots": [{"type": "hour", "value": "6"}]}, INV)
    assert is_schema_valid({"intent": "alarm_create", "slots": []})  # no inventory: structure only
