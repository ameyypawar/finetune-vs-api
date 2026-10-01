"""MASSIVE's `[label : value]` markup, and turning rows into examples."""

from __future__ import annotations

import pytest

from conftest import massive_row
from finetune_vs_api.data import Slot, parse_annot_utt, to_example


def slots(parsed):
    return [(s.type, s.value) for s in parsed.slots]


def test_plain_markup_gives_text_and_slots_in_order():
    parsed = parse_annot_utt("wake me up at [time : nine am] on [date : friday]")
    assert parsed.text == "wake me up at nine am on friday"
    assert slots(parsed) == [("time", "nine am"), ("date", "friday")]


def test_slots_keep_order_of_appearance_not_alphabetical_order():
    parsed = parse_annot_utt("[time : noon] on [date : friday] with [person : sam]")
    assert [s.type for s in parsed.slots] == ["time", "date", "person"]


def test_a_colon_inside_a_value_survives():
    parsed = parse_annot_utt("set an alarm for [time : 5:30] today")
    assert slots(parsed) == [("time", "5:30")]
    assert parsed.text == "set an alarm for 5:30 today"


def test_only_the_first_space_colon_space_splits_label_from_value():
    parsed = parse_annot_utt("remind me at [time : 5 : 30 : 15]")
    assert slots(parsed) == [("time", "5 : 30 : 15")]


def test_repeated_slot_types_are_all_kept():
    parsed = parse_annot_utt("from [date : monday] to [date : friday]")
    assert slots(parsed) == [("date", "monday"), ("date", "friday")]


def test_the_same_type_and_value_twice_is_two_slots():
    parsed = parse_annot_utt("[place_name : paris] or [place_name : paris]")
    assert len(parsed.slots) == 2
    assert parsed.slots[0] == parsed.slots[1] == Slot("place_name", "paris")


@pytest.mark.parametrize("annot", ["good night", "", "what time is it"])
def test_a_request_without_markup_has_no_slots(annot):
    parsed = parse_annot_utt(annot)
    assert parsed.slots == ()
    assert parsed.text == annot
    assert parsed.empty_dropped == 0


def test_a_slot_with_an_empty_value_is_dropped_and_counted():
    parsed = parse_annot_utt("remind me [date : ] to call")
    assert parsed.slots == ()
    assert parsed.empty_dropped == 1
    assert parsed.text == "remind me  to call"


def test_a_whitespace_only_value_counts_as_empty():
    parsed = parse_annot_utt("a [date :  ] b [time : noon]")
    assert slots(parsed) == [("time", "noon")]
    assert parsed.empty_dropped == 1


def test_apostrophes_and_unicode_are_untouched():
    parsed = parse_annot_utt("what's on my [list_name : grocery] list, café")
    assert parsed.text == "what's on my grocery list, café"


@pytest.mark.parametrize(
    "annot",
    [
        "set [time five am",  # unclosed
        "[time five am]",  # no ' : ' separator
        "[a : [b : c]]",  # nested
        "stray ] bracket",
        "[time : 5am] and [",
    ],
)
def test_malformed_markup_raises_instead_of_guessing(annot):
    with pytest.raises(ValueError, match="malformed"):
        parse_annot_utt(annot)


def test_to_example_maps_the_row():
    row = massive_row(7, "dev", "alarm_set", "wake me at [time : six]", scenario="alarm")
    example = to_example(row)
    assert (example.id, example.split, example.scenario, example.intent) == ("7", "dev", "alarm", "alarm_set")
    assert example.text == "wake me at six"
    assert example.slots == (Slot("time", "six"),)


def test_to_example_keeps_ids_as_strings():
    assert to_example(massive_row(12, "train", "x", "hello")).id == "12"


def test_to_example_rejects_an_unknown_partition():
    with pytest.raises(ValueError, match="partition"):
        to_example(massive_row(1, "validation", "x", "hello"))


def test_slot_values_are_substrings_of_the_text():
    example = to_example(massive_row(1, "train", "x", "play [song_name : be warned] by [artist_name : tech nine]"))
    assert all(s.value in example.text for s in example.slots)
