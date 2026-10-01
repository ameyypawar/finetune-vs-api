"""Scenario-stratified, nested, seeded evaluation subsets."""

from __future__ import annotations

import json
import random
from collections import Counter

import pytest

from conftest import ex, load_script
from finetune_vs_api import config, data
from finetune_vs_api.subsets import (
    SUBSET_NAMES,
    allocate,
    make_subsets,
    resolve_subset,
    subset_hash,
)

SCENARIOS = [f"scenario_{i:02d}" for i in range(18)]


def corpus(split: str, per_scenario: list[int], start: int = 0):
    """Examples for one split; scenario i gets per_scenario[i] items. Ids are unique per call."""
    items, n = [], start
    for scenario, count in zip(SCENARIOS, per_scenario, strict=True):
        for _ in range(count):
            items.append(ex(n, split, "intent", f"text {n}", scenario=scenario))
            n += 1
    return items


DEV = corpus("dev", [20 + 5 * i for i in range(18)], start=0)  # 20..105 per scenario
TEST = corpus("test", [40 + 9 * i for i in range(18)], start=100_000)
BUILT = make_subsets(DEV, TEST)


def ids(name):
    return set(BUILT["subsets"][name]["ids"])


def per_scenario(name, examples):
    by_id = {e.id: e.scenario for e in examples}
    return Counter(by_id[i] for i in ids(name))


def test_sizes_and_split_purity():
    assert {n: BUILT["subsets"][n]["n"] for n in BUILT["subsets"]} == {"D100": 100, "D50": 50, "S500": 500, "S300": 300}
    dev_ids, test_ids = {e.id for e in DEV}, {e.id for e in TEST}
    assert ids("D100") <= dev_ids and ids("D50") <= dev_ids
    assert ids("S500") <= test_ids and ids("S300") <= test_ids
    assert {BUILT["subsets"][n]["split"] for n in ("D50", "D100")} == {"dev"}


def test_subsets_are_nested():
    assert ids("D50") < ids("D100")
    assert ids("S300") < ids("S500")


def test_every_scenario_meets_the_floor():
    for name, examples, floor in (("D100", DEV, 5), ("S300", TEST, 5), ("S500", TEST, 5), ("D50", DEV, 2)):
        counts = per_scenario(name, examples)
        assert len(counts) == 18
        assert min(counts.values()) >= floor, name
        assert BUILT["subsets"][name]["min_per_scenario_applied"] == floor


def test_d50_gets_a_floor_of_two_because_five_per_scenario_does_not_fit():
    assert 18 * 5 > 50
    assert BUILT["subsets"]["D50"]["min_per_scenario_applied"] == 2
    assert BUILT["subsets"]["D100"]["min_per_scenario_applied"] == 5


def test_allocation_is_proportional_where_the_floor_does_not_bind():
    counts = per_scenario("S500", TEST)
    population = len(TEST)
    for scenario in SCENARIOS:
        size = sum(1 for e in TEST if e.scenario == scenario)
        ideal = 500 * size / population
        assert ideal > 5  # the floor of 5 is not what decides this subset
        assert abs(counts[scenario] - ideal) <= 1.0 + 1e-9, scenario


def test_recorded_per_scenario_counts_match_the_ids():
    for name, examples in (("D100", DEV), ("D50", DEV), ("S500", TEST), ("S300", TEST)):
        assert BUILT["subsets"][name]["per_scenario"] == dict(sorted(per_scenario(name, examples).items()))


def test_determinism_and_independence_from_input_order():
    again = make_subsets(DEV, TEST)
    assert again == BUILT
    shuffled_dev, shuffled_test = DEV[:], TEST[:]
    random.Random(0).shuffle(shuffled_dev)
    random.Random(1).shuffle(shuffled_test)
    assert make_subsets(shuffled_dev, shuffled_test) == BUILT


def test_a_different_seed_gives_different_subsets():
    other = make_subsets(DEV, TEST, seed=1)
    assert other["subsets"]["S500"]["ids"] != BUILT["subsets"]["S500"]["ids"]
    assert other["subsets"]["S500"]["hash"] != BUILT["subsets"]["S500"]["hash"]


def test_the_default_seed_is_the_planned_one():
    assert BUILT["seed"] == 20261001


def test_each_subset_records_its_hash_and_full_hashes_cover_the_whole_split():
    for entry in BUILT["subsets"].values():
        assert entry["hash"] == subset_hash(entry["ids"])
    assert BUILT["full"]["dev"] == {"n": len(DEV), "hash": subset_hash(e.id for e in DEV)}
    assert BUILT["full"]["test"]["n"] == len(TEST)


def test_the_result_is_json_serializable():
    assert json.loads(json.dumps(BUILT)) == BUILT


def test_make_subsets_rejects_examples_in_the_wrong_split():
    with pytest.raises(ValueError, match="include"):
        make_subsets([ex(1, "train", "i", "t")], TEST)


def test_a_tiny_scenario_contributes_what_it_has_and_others_make_up_the_rest():
    sizes = [3] + [100] * 17  # scenario_00 has only 3 items, fewer than the floor of 5
    dev = corpus("dev", sizes)
    built = make_subsets(dev, TEST)
    counts = Counter(e.scenario for e in dev if e.id in set(built["subsets"]["D100"]["ids"]))
    assert counts["scenario_00"] <= 3
    assert built["subsets"]["D100"]["n"] == 100


# --- the hash ------------------------------------------------------------------------------


def test_subset_hash_ignores_order_and_detects_changes():
    assert subset_hash(["1", "2", "10"]) == subset_hash(["10", "1", "2"])
    assert subset_hash(["1", "2"]) != subset_hash(["1", "3"])
    assert subset_hash(["1", "2"]) != subset_hash(["1", "2", "3"])
    assert subset_hash([1, 2]) == subset_hash(["1", "2"])  # ids are compared as text


# --- allocate --------------------------------------------------------------------------------


def test_allocate_hits_the_total_exactly():
    counts = {"a": 500, "b": 300, "c": 120, "d": 7}
    for total in (10, 33, 100, 927):
        assert sum(allocate(counts, total, 2).values()) == total


def test_allocate_respects_floor_and_capacity():
    got = allocate({"a": 1000, "b": 10, "c": 3}, 50, 5)
    assert got["c"] == 3  # smaller than the floor: takes all it has
    assert got["b"] >= 5
    assert all(got[s] <= c for s, c in {"a": 1000, "b": 10, "c": 3}.items())


def test_allocate_is_deterministic_on_ties():
    counts = {"a": 10, "b": 10, "c": 10}
    assert allocate(counts, 10, 0) == allocate(dict(reversed(list(counts.items()))), 10, 0)
    assert sum(allocate(counts, 10, 0).values()) == 10


def test_allocate_errors():
    with pytest.raises(ValueError, match="cannot pick"):
        allocate({"a": 3}, 5, 0)
    with pytest.raises(ValueError, match="minimum"):
        allocate({"a": 50, "b": 50}, 5, 5)


# --- resolving a subset against a split ------------------------------------------------------


def test_resolve_full_and_named_subsets():
    dev_ids = [e.id for e in DEV]
    full = resolve_subset(BUILT, "full", "dev", dev_ids)
    assert full.ids == tuple(dev_ids) and full.hash == BUILT["full"]["dev"]["hash"]
    d50 = resolve_subset(BUILT, "D50", "dev", dev_ids)
    assert len(d50.ids) == 50 and d50.hash == BUILT["subsets"]["D50"]["hash"]


def test_a_subset_cannot_be_used_with_the_other_split():
    with pytest.raises(ValueError, match="dev subset"):
        resolve_subset(BUILT, "D100", "test", [e.id for e in TEST])
    with pytest.raises(ValueError, match="test subset"):
        resolve_subset(BUILT, "S300", "dev", [e.id for e in DEV])


def test_unknown_subset_is_rejected():
    with pytest.raises(ValueError, match="unknown subset"):
        resolve_subset(BUILT, "S1000", "test", [])
    assert SUBSET_NAMES == ("full", "D100", "D50", "S500", "S300")


def test_a_tampered_or_stale_subset_is_caught():
    tampered = json.loads(json.dumps(BUILT))
    tampered["subsets"]["S300"]["ids"][0] = "999999999"
    with pytest.raises(ValueError, match="recorded hash"):
        resolve_subset(tampered, "S300", "test", [e.id for e in TEST])
    with pytest.raises(ValueError, match="no longer matches"):
        resolve_subset(BUILT, "full", "test", [e.id for e in TEST][:-1])
    with pytest.raises(ValueError, match="not in the test split"):
        resolve_subset(BUILT, "S300", "test", [e.id for e in TEST][:10])


# --- the script -------------------------------------------------------------------------------


def test_script_writes_both_copies_and_check_detects_drift(tmp_path):
    processed, results = tmp_path / "processed", tmp_path / "results"
    data.write_examples(processed / "dev.jsonl", DEV)
    data.write_examples(processed / "test.jsonl", TEST)
    script = load_script("make_subsets")
    lines: list[str] = []
    assert script.run(processed_dir=processed, results_dir=results, out=lines.append) == 0
    a, b = (processed / "subsets.json").read_text(), (results / "subsets.json").read_text()
    assert a == b and json.loads(a) == BUILT
    assert script.run(check=True, processed_dir=processed, results_dir=results, out=lines.append) == 0
    (results / "subsets.json").write_text(b.replace('"seed": 20261001', '"seed": 1'))
    assert script.run(check=True, processed_dir=processed, results_dir=results, out=lines.append) == 1
    assert script.run(processed_dir=tmp_path / "missing", results_dir=results, out=lines.append) == 2


# --- the real files, when present ---------------------------------------------------------------

real_subsets = config.PROCESSED_DIR / "subsets.json"


@pytest.mark.skipif(not real_subsets.exists(), reason="data/processed not prepared (run scripts/)")
def test_real_subsets_are_nested_stratified_and_consistent_with_the_splits():
    stored = json.loads(real_subsets.read_text())
    dev = data.read_examples(config.PROCESSED_DIR / "dev.jsonl")
    test = data.read_examples(config.PROCESSED_DIR / "test.jsonl")
    assert stored == make_subsets(dev, test)
    sub = {k: set(v["ids"]) for k, v in stored["subsets"].items()}
    assert sub["D50"] < sub["D100"] and sub["S300"] < sub["S500"]
    assert stored["n_scenarios"] == 18
