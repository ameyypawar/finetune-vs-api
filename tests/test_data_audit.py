"""Download and checksum, reading the archive, the audit, and the prepare_data pipeline."""

from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from conftest import ROWS, ex, make_archive, massive_row
from finetune_vs_api import data
from finetune_vs_api.data import ChecksumError, audit, download_massive, read_locale, to_example

URL = "https://stub.example/massive.tar.gz"
PAYLOAD = b"not really a tarball" * 500
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


def serve(payload=PAYLOAD, status=200):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, content=payload)

    return httpx.MockTransport(handler), calls


# --- download -------------------------------------------------------------------------------------------


def test_download_verifies_and_returns_the_sha256(tmp_path):
    transport, calls = serve()
    dest = tmp_path / "raw" / "a.tar.gz"
    assert download_massive(URL, dest, PAYLOAD_SHA, transport=transport) == PAYLOAD_SHA
    assert dest.read_bytes() == PAYLOAD and len(calls) == 1
    assert not dest.with_name("a.tar.gz.part").exists()


def test_without_a_pin_nothing_is_verified_and_the_hash_comes_back_for_pinning(tmp_path):
    transport, _ = serve()
    assert download_massive(URL, tmp_path / "a", None, transport=transport) == PAYLOAD_SHA


def test_a_wrong_pin_fails_and_leaves_nothing_behind(tmp_path):
    transport, _ = serve()
    dest = tmp_path / "a.tar.gz"
    with pytest.raises(ChecksumError, match="discarded"):
        download_massive(URL, dest, "0" * 64, transport=transport)
    assert not dest.exists() and not list(tmp_path.iterdir())


def test_the_pin_is_case_insensitive(tmp_path):
    transport, _ = serve()
    assert download_massive(URL, tmp_path / "a", PAYLOAD_SHA.upper(), transport=transport) == PAYLOAD_SHA


def test_an_existing_good_file_is_reused_without_a_request(tmp_path):
    dest = tmp_path / "a"
    dest.write_bytes(PAYLOAD)
    transport, calls = serve()
    assert download_massive(URL, dest, PAYLOAD_SHA, transport=transport) == PAYLOAD_SHA
    assert calls == []


def test_an_existing_file_with_the_wrong_hash_is_an_error_not_a_silent_redownload(tmp_path):
    dest = tmp_path / "a"
    dest.write_bytes(b"corrupt")
    transport, calls = serve()
    with pytest.raises(ChecksumError, match="--refresh"):
        download_massive(URL, dest, PAYLOAD_SHA, transport=transport)
    assert calls == [] and dest.read_bytes() == b"corrupt"


def test_refresh_downloads_again(tmp_path):
    dest = tmp_path / "a"
    dest.write_bytes(b"old")
    transport, calls = serve()
    assert download_massive(URL, dest, PAYLOAD_SHA, refresh=True, transport=transport) == PAYLOAD_SHA
    assert dest.read_bytes() == PAYLOAD and len(calls) == 1


def test_an_http_error_raises_and_cleans_up(tmp_path):
    transport, _ = serve(b"nope", status=503)
    dest = tmp_path / "a"
    with pytest.raises(httpx.HTTPStatusError):
        download_massive(URL, dest, PAYLOAD_SHA, transport=transport)
    assert not dest.exists() and not list(tmp_path.iterdir())


# --- reading the archive ----------------------------------------------------------------------------------


def test_read_locale_returns_the_rows_of_that_locale_only(tmp_path):
    rows = [massive_row(1, "train", "a", "hello [x : there]"), massive_row(2, "dev", "b", "bye")]
    path = tmp_path / "m.tar.gz"
    make_archive(path, rows)
    assert read_locale(path, "en-US") == rows


def test_read_locale_does_not_match_a_longer_file_name(tmp_path):
    path = tmp_path / "m.tar.gz"
    make_archive(path, [massive_row(1, "train", "a", "x", locale="xx-en-US")], locale="xx-en-US")
    with pytest.raises(FileNotFoundError, match="en-US.jsonl"):
        read_locale(path, "en-US")


def test_read_locale_reports_a_missing_locale(tmp_path):
    path = tmp_path / "m.tar.gz"
    make_archive(path, [massive_row(1, "train", "a", "x")])
    with pytest.raises(FileNotFoundError):
        read_locale(path, "de-DE")


# --- the audit ----------------------------------------------------------------------------------------------


def corpus():
    """A small corpus with every kind of finding in it, and known counts."""
    return [
        # train
        ex(1, "train", "alarm_set", "wake me at six", [("time", "six")], "alarm"),
        ex(2, "train", "alarm_set", "wake me at six", [("time", "six")], "alarm"),  # exact duplicate of 1
        ex(3, "train", "audio_volume_mute", "good night", [], "audio"),
        ex(4, "train", "iot_hue_lightoff", "Good  Night", [], "iot"),  # same text as 3, conflicting intent
        ex(5, "train", "alarm_set", "alarm at five", [("time", "five")], "alarm"),
        ex(6, "train", "alarm_set", "alarm at five", [("date", "five")], "alarm"),  # same intent, different slots
        ex(7, "train", "play_music", "play some jazz", [("music_genre", "jazz")], "play"),
        ex(8, "train", "weather_query", "what's the weather", [], "weather"),
        # dev
        ex(20, "dev", "alarm_set", "wake me at six", [("time", "six")], "alarm"),  # in train, same label
        ex(21, "dev", "play_music", "play the beatles", [("artist_name", "the beatles")], "play"),
        # test
        ex(30, "test", "audio_volume_mute", "good night", [], "audio"),  # in train; same label as 3
        ex(31, "test", "weather_query", "whats the weather", [], "weather"),  # matches 8 only without punctuation
        ex(32, "test", "alarm_set", "wake me at six", [("time", "six")], "alarm"),  # in train and dev
    ]


def test_split_sizes_and_label_coverage():
    a = audit(corpus())
    assert a["splits"] == {"train": 8, "dev": 2, "test": 3, "total": 13}
    intents = a["labels"]["intents"]
    assert intents["n_total"] == 5 and intents["per_split"] == {"train": 5, "dev": 2, "test": 3}
    assert intents["missing_from_split"]["dev"] == ["audio_volume_mute", "iot_hue_lightoff", "weather_query"]
    assert intents["not_in_train"] == {"dev": [], "test": []}
    slot_types = a["labels"]["slot_types"]
    assert slot_types["n_total"] == 4 and slot_types["per_split"]["train"] == 3
    assert slot_types["missing_from_split"]["train"] == ["artist_name"]
    assert slot_types["not_in_train"]["dev"] == ["artist_name"]  # a label dev uses that train never saw


def test_scenarios_and_slot_statistics():
    a = audit(corpus())
    assert a["scenarios"]["n_total"] == 5 and a["scenarios"]["per_split"]["train"]["alarm"] == 4
    slots = a["slots"]
    assert slots["total"] == 8 and slots["max_per_example"] == 1
    assert slots["examples_without_slots"] == {"train": 3, "dev": 0, "test": 2}
    assert slots["values_not_found_in_text"] == 0
    broken = corpus() + [ex(99, "train", "x", "hello", [("time", "midnight")])]
    assert audit(broken)["slots"]["values_not_found_in_text"] == 1


def test_duplicates_within_train_are_grouped_by_what_kind_of_duplicate_they_are():
    d = audit(corpus())["duplicates_within_train"]
    assert d["duplicate_text_groups"] == 3 and d["rows_in_groups"] == 6
    assert d["groups_with_identical_label"] == 1  # 1 and 2
    assert d["groups_with_conflicting_intent"] == 1  # 3 and 4, found despite the case and spacing
    assert d["groups_with_same_intent_different_slots"] == 1  # 5 and 6
    assert {tuple(g["ids"]) for g in d["first_groups"]} == {("1", "2"), ("3", "4"), ("5", "6")}


def test_text_shared_between_splits_is_counted_both_ways():
    o = audit(corpus())["overlap"]
    assert o["train_dev"] == {"shared_texts": 1, "train_items": 2, "dev_items": 1}
    assert o["train_test"] == {"shared_texts": 2, "train_items": 4, "test_items": 2}  # train 1, 2, 3 and 4
    assert o["dev_test"] == {"shared_texts": 1, "dev_items": 1, "test_items": 1}
    assert o["dev_item_ids_in_train"] == ["20"] and o["test_item_ids_in_train"] == ["30", "32"]
    assert o["dev_items_with_identical_label_in_train"] == 1 and o["test_items_with_identical_label_in_train"] == 2


def test_the_punctuation_insensitive_view_finds_more():
    o = audit(corpus())["overlap"]
    assert o["train_test"]["test_items"] == 2
    loose = o["ignoring_punctuation"]["train_test"]
    assert (loose["shared_texts"], loose["test_items"]) == (3, 3)  # "whats the weather" matches "what's the weather"


def test_the_audit_reports_whether_the_data_matches_what_was_expected():
    got = audit(corpus(), expected={"train": 8, "dev": 2, "test": 3, "intents": 5, "slot_types": 3})
    assert got["expected"] == {"values": {"train": 8, "dev": 2, "test": 3, "intents": 5, "slot_types": 3}, "matches": True, "differences": []}
    off = audit(corpus(), expected={"train": 9, "dev": 2, "intents": 60})["expected"]
    assert off["matches"] is False and off["differences"] == ["train: expected 9, got 8", "intents: expected 60, got 5"]
    assert "expected" not in audit(corpus())


def test_an_unknown_split_is_refused():
    with pytest.raises(ValueError, match="unknown split"):
        audit([ex(1, "validation", "a", "b")])


def test_the_audit_is_json_serializable():
    json.dumps(audit(corpus()))


# --- the whole prepare_data pipeline on a synthetic archive (fixture `prep` in conftest.py) ---------------------


def test_prepare_data_writes_everything_and_matches_the_expected_counts(prep):
    assert prep.run() == 0
    processed, results = prep.root / "processed", prep.root / "results"
    assert sorted(p.name for p in processed.iterdir()) == [
        "dev.jsonl", "sft_dev.ids.txt", "sft_dev.jsonl", "sft_train.ids.txt", "sft_train.jsonl", "test.jsonl", "train.jsonl",
    ]
    train = data.read_examples(processed / "train.jsonl")
    assert [e.id for e in train] == ["1", "2", "3"] and train[1].slots[0].value == "5:30"
    assert all(e.split == "train" for e in train)
    text = "\n".join(prep.lines)
    assert f"sha256: {prep.sha}" in text and "pinned in configs/data.yaml: matches" in text
    assert "audit MATCHES the expected counts" in text
    doc = json.loads((results / "data_audit.json").read_text())
    assert doc["dataset"]["archive_sha256"] == prep.sha and doc["dataset"]["license"] == "CC-BY-4.0"
    assert doc["splits"] == {"train": 3, "dev": 2, "test": 2, "total": 7}
    assert doc["integrity"] == {"markup_text_differs_from_utt": 0, "empty_value_slots_dropped": 0, "slot_values_not_found_in_text": 0}
    assert doc["inventory"] == {"intents": ["alarm_set", "general_greet"], "slot_types": ["date", "time"]}
    assert doc["overlap"]["train_test"]["test_items"] == 1  # "hello there"


def test_an_unpinned_run_says_so_and_still_prints_the_hash(prep):
    prep.write_config(sha256=None)
    assert prep.run() == 0
    text = "\n".join(prep.lines)
    assert f"sha256: {prep.sha}" in text and "NOT PINNED" in text


def test_a_wrong_pin_stops_with_exit_2_and_writes_no_data(prep):
    prep.write_config(sha256="0" * 64)
    assert prep.run() == 2
    assert "CHECKSUM ERROR" in "\n".join(prep.lines)
    assert not (prep.root / "processed").exists() and not (prep.root / "results").exists()


def test_unexpected_counts_are_reported_with_exit_3(prep):
    prep.write_config(expected={"train": 99, "dev": 2, "test": 2, "intents": 2, "slot_types": 2})
    assert prep.run() == 3
    text = "\n".join(prep.lines)
    assert "audit DIFFERS" in text and "train: expected 99, got 3" in text
    assert (prep.root / "results" / "data_audit.json").exists()  # the audit is still written so it can be read


def test_a_second_run_reuses_the_archive_and_refresh_downloads_again(prep):
    calls = []
    original = prep.paths["transport"]
    prep.paths["transport"] = httpx.MockTransport(lambda r: (calls.append(1), original.handler(r))[1])
    assert prep.run() == 0 and len(calls) == 1
    assert prep.run() == 0 and len(calls) == 1  # reused
    assert prep.run(refresh=True) == 0 and len(calls) == 2


def test_sft_files_are_chat_records_with_a_matching_id_sidecar(prep):
    prep.run()
    processed = prep.root / "processed"
    records = list(data.iter_chat_records(processed / "sft_train.jsonl"))
    ids = data.read_ids(processed / "sft_train.ids.txt")
    assert ids == ["1", "2", "3"] and len(records) == 3
    assert data.validate_chat_jsonl(processed / "sft_train.jsonl").ok
    assert json.loads(records[1]["messages"][-1]["content"]) == {"intent": "alarm_set", "slots": [{"type": "time", "value": "5:30"}]}


def test_to_example_of_the_archive_rows_round_trips_through_the_files(prep):
    prep.run()
    processed = prep.root / "processed"
    expected = [to_example(r) for r in ROWS if r["partition"] == "dev"]
    assert data.read_examples(processed / "dev.jsonl") == expected
