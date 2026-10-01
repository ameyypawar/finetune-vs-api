"""Nothing from dev or test may reach training, retrieval, or the label inventory.

Checked on a synthetic archive pushed through the real prepare_data pipeline, and again on the
real prepared files when they are present.
"""

from __future__ import annotations

import json

import pytest

from conftest import ex, fake_embed
from finetune_vs_api import config, data, prompts
from finetune_vs_api.metrics import normalize_value
from finetune_vs_api.retrieval import LeakageError, RetrievalIndex
from finetune_vs_api.schema import label_inventory, target_json

# --- the retrieval index is train only -----------------------------------------------------------------


def test_the_retrieval_index_only_accepts_train_examples():
    train = [ex(i, "train", "a", f"train text {i}") for i in range(3)]
    for split in ("dev", "test"):
        with pytest.raises(LeakageError):
            RetrievalIndex.build([*train, ex(99, split, "a", "leaky")], embedder=fake_embed)


def test_a_dev_or_test_request_retrieves_train_examples_and_nothing_else():
    train = [ex(i, "train", "a", f"alarm number {i}") for i in range(20)]
    dev = [ex(100 + i, "dev", "a", f"alarm number {i} dev") for i in range(5)]
    test = [ex(200 + i, "test", "a", f"alarm number {i} test") for i in range(5)]
    index = RetrievalIndex.build(train, embedder=fake_embed)
    for query in dev + test:
        got = index.topk(query.text, 10)
        assert {e.split for e in got} == {"train"}
        assert query.id not in {e.id for e in got}


def test_the_label_inventory_is_read_from_train_only():
    train = [ex(1, "train", "alarm_set", "x", [("time", "x")])]
    dev_only = ex(2, "dev", "dev_only_intent", "y", [("dev_only_slot", "y")])
    inventory = label_inventory(train)
    assert "dev_only_intent" not in inventory.intents and "dev_only_slot" not in inventory.slot_types
    with pytest.raises(ValueError, match="train examples only"):
        label_inventory([*train, dev_only])


# --- the training files, from the pipeline -----------------------------------------------------------------


def split_ids(processed):
    return {s: [e.id for e in data.read_examples(processed / f"{s}.jsonl")] for s in ("train", "dev", "test")}


def test_every_sft_train_id_is_in_train(prep):
    assert prep.run() == 0
    processed = prep.root / "processed"
    ids = split_ids(processed)
    sft_train = data.read_ids(processed / "sft_train.ids.txt")
    assert set(sft_train) <= set(ids["train"])
    assert not set(sft_train) & (set(ids["dev"]) | set(ids["test"]))
    assert sft_train == ids["train"]  # all of train, in order, and nothing else


def test_sft_dev_is_dev_only(prep):
    prep.run()
    processed = prep.root / "processed"
    ids = split_ids(processed)
    sft_dev = data.read_ids(processed / "sft_dev.ids.txt")
    assert sft_dev == ids["dev"]
    assert not set(sft_dev) & (set(ids["train"]) | set(ids["test"]))


def test_each_sft_record_is_exactly_its_example_so_the_sidecar_can_be_trusted(prep):
    prep.run()
    processed = prep.root / "processed"
    by_id = {e.id: e for s in ("train", "dev") for e in data.read_examples(processed / f"{s}.jsonl")}
    for name in ("sft_train", "sft_dev"):
        records = list(data.iter_chat_records(processed / f"{name}.jsonl"))
        ids = data.read_ids(processed / f"{name}.ids.txt")
        assert len(records) == len(ids)
        for record, item_id in zip(records, ids, strict=True):
            assert record == prompts.finetune_record(by_id[item_id])


def test_the_test_split_never_appears_in_any_training_artifact(prep):
    prep.run()
    processed = prep.root / "processed"
    test_ids = set(split_ids(processed)["test"])
    for name in ("sft_train.ids.txt", "sft_dev.ids.txt"):
        assert not set(data.read_ids(processed / name)) & test_ids


# --- the real prepared files, when present ------------------------------------------------------------------

real = config.PROCESSED_DIR
have_real = (real / "sft_train.ids.txt").exists() and (real / "test.jsonl").exists()
skip_real = pytest.mark.skipif(not have_real, reason="data/processed not prepared (run scripts/prepare_data.py)")


@skip_real
def test_real_sft_train_ids_are_exactly_the_train_ids():
    ids = split_ids(real)
    assert data.read_ids(real / "sft_train.ids.txt") == ids["train"]
    assert data.read_ids(real / "sft_dev.ids.txt") == ids["dev"]
    assert not set(ids["train"]) & set(ids["dev"]) and not set(ids["train"]) & set(ids["test"]) and not set(ids["dev"]) & set(ids["test"])


@skip_real
def test_real_sft_records_are_their_train_examples():
    by_id = {e.id: e for e in data.read_examples(real / "train.jsonl")}
    ids = data.read_ids(real / "sft_train.ids.txt")
    for record, item_id in zip(data.iter_chat_records(real / "sft_train.jsonl"), ids, strict=True):
        example = by_id[item_id]
        assert record["messages"][1]["content"] == example.text
        assert record["messages"][2]["content"] == target_json(example)


@skip_real
def test_test_text_in_the_training_data_is_only_what_the_audit_already_reports():
    # MASSIVE repeats some requests across splits. The audit counts them; the training file must
    # contain exactly that many, which shows that nothing else leaked in.
    audit = json.loads((config.RESULTS_DIR / "data_audit.json").read_text())
    train_texts = {
        normalize_value(record["messages"][1]["content"]) for record in data.iter_chat_records(real / "sft_train.jsonl")
    }
    for split, key in (("test", "train_test"), ("dev", "train_dev")):
        split_texts = {normalize_value(e.text) for e in data.read_examples(real / f"{split}.jsonl")}
        assert len(split_texts & train_texts) == audit["overlap"][key]["shared_texts"], split


@skip_real
def test_real_subsets_only_name_ids_from_their_own_split():
    ids = split_ids(real)
    stored = json.loads((real / "subsets.json").read_text())["subsets"]
    for name, entry in stored.items():
        assert set(entry["ids"]) <= set(ids[entry["split"]]), name
        assert not set(entry["ids"]) & set(ids["train"]), name
