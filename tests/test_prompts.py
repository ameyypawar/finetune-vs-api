"""The prompt registry: layout, versioning and fingerprints."""

from __future__ import annotations

import json

import pytest

from conftest import ex
from finetune_vs_api import prompts
from finetune_vs_api.prompts import (
    FINETUNED_INSTRUCTION,
    PROMPTS,
    finetune_record,
    get_prompt,
    prompt_hash,
    render_messages,
    static_prefix,
)
from finetune_vs_api.schema import LabelInventory, parse_output, target_json

INV = LabelInventory(("alarm_set", "play_music", "weather_query"), ("artist_name", "date", "time"))


def shots(n):
    return [ex(i, "train", "alarm_set", f"example request {i}", [("time", f"{i}")]) for i in range(n)]


def test_the_registry_has_the_three_planned_prompts():
    assert set(PROMPTS) == {"finetuned_v1", "zeroshot_v1", "fewshot_k10_v1"}
    assert [PROMPTS[n].k for n in ("finetuned_v1", "zeroshot_v1", "fewshot_k10_v1")] == [0, 0, 10]


def test_an_unknown_prompt_lists_the_known_ones():
    with pytest.raises(KeyError, match="fewshot_k10_v1"):
        get_prompt("fewshot_k10_v2")


# --- finetuned_v1 ----------------------------------------------------------------------------------


def test_finetuned_is_a_one_line_instruction_plus_the_request():
    messages = render_messages("finetuned_v1", "wake me at six")
    assert messages == [
        {"role": "system", "content": FINETUNED_INSTRUCTION},
        {"role": "user", "content": "wake me at six"},
    ]
    assert "\n" not in FINETUNED_INSTRUCTION


def test_finetuned_needs_no_label_inventory_and_takes_no_shots():
    assert static_prefix("finetuned_v1") == FINETUNED_INSTRUCTION
    with pytest.raises(ValueError, match="exactly 0"):
        render_messages("finetuned_v1", "x", shots=shots(1))


def test_finetune_record_is_the_prompt_followed_by_the_gold_call():
    example = ex(1, "train", "alarm_set", "wake me at six", [("time", "six")])
    record = finetune_record(example)
    assert record["messages"][:-1] == render_messages("finetuned_v1", "wake me at six")
    assert record["messages"][-1] == {"role": "assistant", "content": target_json(example)}
    assert parse_output(record["messages"][-1]["content"]) == {"intent": "alarm_set", "slots": [{"type": "time", "value": "six"}]}
    assert set(record) == {"messages"}  # nothing but messages, so it is also a valid OpenAI fine-tuning line


# --- zeroshot_v1 ----------------------------------------------------------------------------------------


def test_zeroshot_lists_every_label_and_says_to_copy_spans_verbatim():
    system = render_messages("zeroshot_v1", "wake me at six", inventory=INV)[0]["content"]
    for label in (*INV.intents, *INV.slot_types):
        assert label in system
    assert "copied verbatim from the request" in system
    assert "exactly one JSON function call" in system
    assert '"slots": []' in system  # what to do with no slots
    assert "{{" not in system and "}}" not in system  # the template was formatted, not left raw


def test_zeroshot_is_two_messages_and_needs_the_inventory():
    messages = render_messages("zeroshot_v1", "wake me at six", inventory=INV)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"] == "wake me at six"  # the bare request, like every prompt
    with pytest.raises(ValueError, match="inventory"):
        render_messages("zeroshot_v1", "x")


# --- fewshot_k10_v1 --------------------------------------------------------------------------------------


def test_fewshot_adds_ten_examples_as_prior_turns_before_the_request():
    messages = render_messages("fewshot_k10_v1", "wake me at six", inventory=INV, shots=shots(10))
    assert [m["role"] for m in messages] == ["system"] + ["user", "assistant"] * 10 + ["user"]
    assert messages[-1] == {"role": "user", "content": "wake me at six"}
    for user, assistant in zip(messages[1:-1:2], messages[2:-1:2], strict=True):
        assert parse_output(assistant["content"]) is not None
        assert user["content"].startswith("example request")


def test_the_most_similar_example_sits_next_to_the_request():
    ranked = shots(10)  # as RetrievalIndex.topk returns them: most similar first
    messages = render_messages("fewshot_k10_v1", "q", inventory=INV, shots=ranked)
    assert messages[-2]["content"] == target_json(ranked[0])  # last assistant turn is the closest example
    assert messages[-3]["content"] == ranked[0].text
    assert messages[1]["content"] == ranked[-1].text  # the least similar comes first


def test_fewshot_takes_exactly_k_examples():
    for count in (0, 9, 11):
        with pytest.raises(ValueError, match="exactly 10"):
            render_messages("fewshot_k10_v1", "x", inventory=INV, shots=shots(count))


def test_the_examples_use_the_same_serialization_as_training():
    example = ex(1, "train", "play_music", "play x by y", [("artist_name", "y")])
    messages = render_messages("fewshot_k10_v1", "q", inventory=INV, shots=[example] * 10)
    assert messages[2]["content"] == target_json(example) == finetune_record(example)["messages"][-1]["content"]


# --- the static prefix comes first ----------------------------------------------------------------------------


def test_everything_that_does_not_vary_comes_first():
    a = render_messages("fewshot_k10_v1", "request a", inventory=INV, shots=shots(10))
    b = render_messages("fewshot_k10_v1", "request b", inventory=INV, shots=list(reversed(shots(10))))
    assert a[0] == b[0]  # the system message is identical whatever the request and examples
    assert a[0]["content"] == static_prefix("fewshot_k10_v1", INV)
    assert "request a" not in a[0]["content"] and "example request" not in a[0]["content"]
    zero = render_messages("zeroshot_v1", "request a", inventory=INV)
    assert zero[0] == a[0]  # zero-shot and few-shot share the same cacheable prefix


# --- fingerprints --------------------------------------------------------------------------------------------------


def test_hashes_are_stable_and_distinct_per_prompt():
    first = {n: prompt_hash(n, INV) for n in PROMPTS}
    assert first == {n: prompt_hash(n, INV) for n in PROMPTS}
    assert len(set(first.values())) == 3 and all(len(h) == 64 for h in first.values())
    assert prompt_hash("finetuned_v1") == prompt_hash("finetuned_v1", INV)  # that prompt has no labels in it


def test_the_hash_moves_when_the_text_the_labels_or_the_rendering_change(monkeypatch):
    before = prompt_hash("fewshot_k10_v1", INV)
    other = LabelInventory(INV.intents + ("new_intent",), INV.slot_types)
    assert prompt_hash("fewshot_k10_v1", other) != before
    monkeypatch.setattr(prompts, "_ZEROSHOT_TEMPLATE", prompts._ZEROSHOT_TEMPLATE.replace("Do not add any other keys.", "Add no other keys."))
    assert prompt_hash("fewshot_k10_v1", INV) != before
    assert prompt_hash("zeroshot_v1", INV) != prompt_hash("finetuned_v1")
    monkeypatch.setattr(prompts, "FINETUNED_INSTRUCTION", "Different.")
    assert prompt_hash("finetuned_v1") != prompt_hash("zeroshot_v1", INV)


def test_a_label_using_prompt_cannot_be_hashed_without_the_labels():
    with pytest.raises(ValueError, match="inventory"):
        prompt_hash("zeroshot_v1")


def test_the_hash_input_is_json_serializable_and_includes_the_probe():
    spec = get_prompt("fewshot_k10_v1")
    probe = render_messages("fewshot_k10_v1", "probe request", inventory=INV, shots=prompts._probe_shots(spec.k, INV))
    json.dumps(probe)
    assert len(probe) == 22
