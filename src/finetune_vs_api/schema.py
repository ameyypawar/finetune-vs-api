"""The output contract: one JSON function call.

    {"intent": "<one of 60>", "slots": [{"type": "<one of 55>", "value": "<text from the request>"}]}

This module owns the label inventory (derived from the TRAIN split only), the JSON Schema
for it in OpenAI strict mode, the canonical serialization of a labelled example, and the
rule for turning model output text into a parsed object.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import Example, read_examples

SCHEMA_NAME = "function_call"


@dataclass(frozen=True)
class LabelInventory:
    intents: tuple[str, ...]
    slot_types: tuple[str, ...]

    def to_dict(self) -> dict[str, list[str]]:
        return {"intents": list(self.intents), "slot_types": list(self.slot_types)}

    @classmethod
    def from_dict(cls, obj: dict[str, Sequence[str]]) -> LabelInventory:
        return cls(tuple(obj["intents"]), tuple(obj["slot_types"]))


def label_inventory(train: Iterable[Example]) -> LabelInventory:
    """Sorted intents and slot types seen in the training split.

    Refuses anything that is not a train example: the label set that constrains the
    schema and appears in prompts must not be read off dev or test.
    """
    intents: set[str] = set()
    slot_types: set[str] = set()
    for example in train:
        if example.split != "train":
            raise ValueError(f"label_inventory takes train examples only, got {example.split!r} ({example.id})")
        intents.add(example.intent)
        slot_types.update(s.type for s in example.slots)
    return LabelInventory(tuple(sorted(intents)), tuple(sorted(slot_types)))


def load_inventory(processed_dir: Path) -> LabelInventory:
    """The inventory from `<processed_dir>/train.jsonl` (run scripts/prepare_data.py first)."""
    path = Path(processed_dir) / "train.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/prepare_data.py first")
    return label_inventory(read_examples(path))


def json_schema(intents: Sequence[str], slot_types: Sequence[str]) -> dict[str, Any]:
    """The call as a JSON Schema that satisfies OpenAI strict structured outputs.

    Strict mode requires every property to be listed in `required` and
    `additionalProperties` to be false on every object; labels are constrained with enums.
    """
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": list(intents)},
            "slots": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": list(slot_types)},
                        "value": {"type": "string"},
                    },
                    "required": ["type", "value"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["intent", "slots"],
        "additionalProperties": False,
    }


def response_format(intents: Sequence[str], slot_types: Sequence[str]) -> dict[str, Any]:
    """`json_schema()` wrapped as a Chat Completions `response_format` value."""
    return {
        "type": "json_schema",
        "json_schema": {"name": SCHEMA_NAME, "strict": True, "schema": json_schema(intents, slot_types)},
    }


def target_obj(example: Example) -> dict[str, Any]:
    """The labelled call for an example; slots keep their order of appearance."""
    return {
        "intent": example.intent,
        "slots": [{"type": s.type, "value": s.value} for s in example.slots],
    }


def target_json(example: Example) -> str:
    """Canonical serialization: compact, key order intent then slots, type then value."""
    return json.dumps(target_obj(example), ensure_ascii=False, separators=(",", ":"))


_FENCE = re.compile(r"^```[A-Za-z0-9_-]*\s*\n?(.*?)\n?```$", re.DOTALL)


def parse_output(text: str | None) -> dict[str, Any] | None:
    """Parse model output as a single JSON object, or None.

    The only leniency is surrounding whitespace and one Markdown code fence around the
    object. Prose before or after the JSON, several objects, or anything that is not a JSON
    object is a failure to follow the format and returns None; it is not repaired.
    """
    if text is None:
        return None
    body = text.strip()
    fenced = _FENCE.match(body)
    if fenced:
        body = fenced.group(1).strip()
    try:
        obj = json.loads(body)
    except (json.JSONDecodeError, RecursionError):
        return None
    return obj if isinstance(obj, dict) else None


def is_schema_valid(obj: Any, inventory: LabelInventory | None = None) -> bool:
    """Does `obj` have exactly the shape of the schema?

    Structure is always checked: exactly the keys intent and slots, a string intent, and a
    list of objects with exactly the string keys type and value. Pass `inventory` to also
    enforce the enums (known intent, known slot types).
    """
    if not isinstance(obj, dict) or set(obj) != {"intent", "slots"}:
        return False
    intent, slots = obj["intent"], obj["slots"]
    if not isinstance(intent, str) or not isinstance(slots, list):
        return False
    if inventory is not None and intent not in inventory.intents:
        return False
    for slot in slots:
        if not isinstance(slot, dict) or set(slot) != {"type", "value"}:
            return False
        if not isinstance(slot["type"], str) or not isinstance(slot["value"], str):
            return False
        if inventory is not None and slot["type"] not in inventory.slot_types:
            return False
    return True
