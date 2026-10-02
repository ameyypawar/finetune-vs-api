"""scripts/compare.py: pairing, intervals, the McNemar test, break-even, the wording rule, and what
happens when results are missing. All on synthetic results (tests/results_fixtures.py)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import yaml

from conftest import load_script
from finetune_vs_api import config, metrics
from results_fixtures import (
    API_SYSTEMS,
    BASE,
    FT,
    GEMINI,
    GPT_OSS_20B,
    GPT_OSS_120B,
    QWEN_27B,
    STUB_MODEL,
    SYSTEMS,
    Lab,
)

compare = load_script("compare")
N = 200  # bootstrap resamples: enough to be stable, small enough to be quick
GOOD_LEVEL = {"concurrency": 1, "requests_per_s": 1, "latency_s": {"p50": 0.1, "p95": 0.2}}


def quiet_git(monkeypatch):
    monkeypatch.setattr(config, "git_commit", lambda cwd=None: None)
    monkeypatch.setattr(config, "git_dirty", lambda cwd=None: False)


def build(lab: Lab, n_resamples: int = N, **kwargs):
    return compare.build_comparison(
        results_dir=lab.results, processed_dir=lab.processed, config_dir=lab.config_dir, n_resamples=n_resamples, **kwargs
    )


@pytest.fixture(scope="module")
def standard(tmp_path_factory):
    """The six systems, the audit and a benchmark; built and compared once for the whole module."""
    with pytest.MonkeyPatch.context() as mp:
        quiet_git(mp)
        lab = Lab(tmp_path_factory.mktemp("standard")).populate()
        doc = build(lab)
    return lab, doc


@pytest.fixture(scope="module")
def mixed(tmp_path_factory):
    """The standard set with one API row moved to S300, the pre-registered subset no real row uses."""
    with pytest.MonkeyPatch.context() as mp:
        quiet_git(mp)
        lab = Lab(tmp_path_factory.mktemp("mixed"))
        lab.move_to_subset(GEMINI, "S300")
        lab.populate()
        doc = build(lab)
    return lab, doc


def system(doc, name):
    return next(s for s in doc["systems"] if s["name"] == name)


def em_vector(lab: Lab, name: str, ids) -> np.ndarray:
    return np.array([0.0 if i in lab.wrong[name] else 1.0 for i in ids])


# --- pairing ------------------------------------------------------------------------------------------


def test_every_system_is_compared_on_its_own_subset(standard):
    lab, doc = standard
    assert doc["reference"] == FT and doc["subsets"]["headline"] == "S500"
    assert [s["name"] for s in doc["systems"]] == list(SYSTEMS)
    for name in SYSTEMS:  # every API row runs on S500, and the self-hosted rows are compared there too
        s = system(doc, name)
        assert (s["comparison_subset"], s["n_subset"], s["n_scored"], s["status"]) == ("S500", 500, 500, "complete")
        assert s["metrics"]["n"] == 500


def test_the_pairing_uses_the_same_items_for_both_systems(standard):
    lab, doc = standard
    ids = lab.ids("S500")
    for name in (BASE, *API_SYSTEMS):
        pair = system(doc, name)["vs_reference"]
        assert (pair["subset"], pair["n"], pair["complete"]) == ("S500", len(ids), True)
        # both exact-match values were taken over exactly those items
        assert pair["system_exact_match"]["value"] == pytest.approx(em_vector(lab, name, ids).mean())
        assert pair["reference_exact_match"]["value"] == pytest.approx(em_vector(lab, FT, ids).mean())
    assert system(doc, FT)["vs_reference"] is None  # the reference is not compared with itself


def test_a_row_on_a_smaller_subset_is_compared_with_the_fine_tune_on_that_subset(mixed):
    """No real row uses S300 now, but the pairing still follows each system's own test subset."""
    lab, doc = mixed
    s = system(doc, GEMINI)
    assert (s["comparison_subset"], s["n_subset"], s["n_scored"], s["status"]) == ("S300", 300, 300, "complete")
    assert doc["subsets"]["headline"] == "S500" and doc["subsets"]["info"]["S300"]["n"] == 300
    for name in (FT, BASE, GPT_OSS_20B, GPT_OSS_120B, QWEN_27B):  # the others are untouched
        assert system(doc, name)["comparison_subset"] == "S500"
    ids = lab.ids("S300")
    pair = s["vs_reference"]
    assert (pair["subset"], pair["n"], pair["complete"]) == ("S300", 300, True)
    on_300 = em_vector(lab, FT, ids).mean()
    assert on_300 != pytest.approx(em_vector(lab, FT, lab.ids("S500")).mean())  # the fixture's wrong sets make the two differ
    assert pair["reference_exact_match"]["value"] == pytest.approx(on_300)  # the fine-tune, scored on the same 300 items
    assert pair["system_exact_match"]["value"] == pytest.approx(em_vector(lab, GEMINI, ids).mean())
    expected = metrics.paired_bootstrap(em_vector(lab, GEMINI, ids), em_vector(lab, FT, ids), n_resamples=N)
    assert pair["difference"]["ci95"] == [expected.ci_low, expected.ci_high]
    assert s["cost"]["calls"] == 300 and s["full_test"] is None


def test_the_full_split_is_a_secondary_column_for_the_self_hosted_rows_only(standard):
    lab, doc = standard
    for name in (FT, BASE):
        s = system(doc, name)
        assert s["kind"] == "self-hosted" and s["metrics"]["n"] == 500  # the headline is still S500
        full = s["full_test"]
        assert (full["n_expected"], full["n_scored"], full["complete"]) == (len(lab.test), len(lab.test), True)
        assert full["metrics"]["n"] == len(lab.test)
        assert full["metrics"]["exact_match"]["value"] == pytest.approx(em_vector(lab, name, lab.all_ids).mean())
    for name in API_SYSTEMS:
        assert system(doc, name)["kind"] == "api" and system(doc, name)["full_test"] is None


def test_scoring_reuses_the_runner_so_failed_calls_count_as_wrong(tmp_path, monkeypatch):
    quiet_git(monkeypatch)
    lab = Lab(tmp_path)
    ids = lab.ids("S500")
    lab.write_run(FT, wrong=ids[:10], errors=ids[10:15])
    lab.write_run(GPT_OSS_20B, ids=ids)
    s = system(build(lab), FT)
    assert s["metrics"]["exact_match"]["value"] == pytest.approx(1 - 15 / 500)
    assert s["metrics"]["schema_valid_rate"]["value"] == pytest.approx(1 - 5 / 500)  # failed calls are not valid
    assert s["n_calls_failed"] == 5


# --- intervals --------------------------------------------------------------------------------------------


def test_exact_match_intervals_are_the_seeded_bootstrap_of_the_per_item_scores(standard):
    lab, doc = standard
    for name in (FT, *API_SYSTEMS):
        s = system(doc, name)
        vector = em_vector(lab, name, lab.ids(s["comparison_subset"]))
        low, high = metrics.bootstrap_ci(vector, n_resamples=N)
        assert s["metrics"]["exact_match"]["value"] == pytest.approx(vector.mean())
        assert s["metrics"]["exact_match"]["ci95"] == [low, high]
        assert low < vector.mean() < high


def test_the_paired_difference_is_the_paired_bootstrap_of_system_minus_reference(standard):
    lab, doc = standard
    ids = lab.ids("S500")
    for name in (BASE, *API_SYSTEMS):
        expected = metrics.paired_bootstrap(em_vector(lab, name, ids), em_vector(lab, FT, ids), n_resamples=N)
        d = system(doc, name)["vs_reference"]["difference"]
        assert d["direction"] == "system minus reference"
        assert d["value"] == pytest.approx(expected.diff)
        assert d["ci95"] == [expected.ci_low, expected.ci_high]
        assert d["bootstrap_p"] == pytest.approx(expected.p_value)


def test_the_output_is_deterministic(standard):
    lab, doc = standard
    again = build(lab)
    assert json.dumps(again, sort_keys=True) == json.dumps(doc, sort_keys=True)


def test_the_bootstrap_settings_are_recorded(standard):
    _, doc = standard
    assert doc["bootstrap"] == {
        "resamples": N, "seed": metrics.BOOTSTRAP_SEED, "confidence": 0.95,
        "interval": "percentile; the paired bootstrap resamples the same items for both systems",
    }


# --- McNemar -------------------------------------------------------------------------------------------------


def two_systems(tmp_path, monkeypatch, *, ref_only: int, system_only: int, both: int):
    """The fine-tune and groq-gpt-oss-20b on S500 with exactly this many discordant and shared errors.

    `ref_only` items the fine-tune gets wrong and the system right, `system_only` the reverse.
    """
    quiet_git(monkeypatch)
    lab = Lab(tmp_path)
    ids = list(lab.ids("S500"))
    a, b = ref_only, ref_only + system_only
    ft_wrong = ids[:a] + ids[b : b + both]
    mini_wrong = ids[a:b] + ids[b : b + both]
    lab.write_run(FT, wrong=ft_wrong)
    lab.write_run(GPT_OSS_20B, wrong=mini_wrong)
    return lab


def mcnemar_p(a: int, b: int) -> float:
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def test_mcnemar_counts_are_oriented_system_over_reference(tmp_path, monkeypatch):
    lab = two_systems(tmp_path, monkeypatch, ref_only=30, system_only=10, both=20)
    pair = system(build(lab, 1000), GPT_OSS_20B)["vs_reference"]
    # the system is right where the fine-tune is wrong on 30 items, and wrong where it is right on 10
    assert pair["mcnemar"]["system_only"] == 30 and pair["mcnemar"]["reference_only"] == 10
    assert pair["mcnemar"]["p_value"] == pytest.approx(mcnemar_p(30, 10))
    assert 0 < pair["mcnemar"]["p_value"] < 0.01
    assert pair["difference"]["value"] == pytest.approx((30 - 10) / 500)


def test_mcnemar_with_no_discordant_items_is_one(tmp_path, monkeypatch):
    lab = two_systems(tmp_path, monkeypatch, ref_only=0, system_only=0, both=25)
    pair = system(build(lab, 300), GPT_OSS_20B)["vs_reference"]
    assert (pair["mcnemar"]["system_only"], pair["mcnemar"]["reference_only"], pair["mcnemar"]["p_value"]) == (0, 0, 1.0)
    assert pair["difference"]["value"] == 0.0


def test_the_standard_set_agrees_with_the_mcnemar_test_on_the_same_discordant_counts(standard):
    lab, doc = standard
    ids = lab.ids("S500")
    system_only = sum(1 for i in ids if i in lab.wrong[FT] and i not in lab.wrong[GPT_OSS_20B])
    reference_only = sum(1 for i in ids if i in lab.wrong[GPT_OSS_20B] and i not in lab.wrong[FT])
    mc = system(doc, GPT_OSS_20B)["vs_reference"]["mcnemar"]
    assert (mc["system_only"], mc["reference_only"]) == (system_only, reference_only)
    assert mc["p_value"] == pytest.approx(mcnemar_p(system_only, reference_only))


# --- the wording rule ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("low", "high", "relation"),
    [
        (0.01, 0.05, "system_beats_reference"),
        (-0.05, -0.01, "reference_beats_system"),
        (0.0, 0.05, "no_significant_difference"),  # the interval touches 0: it does not exclude it
        (-0.05, 0.0, "no_significant_difference"),
        (-0.02, 0.03, "no_significant_difference"),
        (0.0, 0.0, "no_significant_difference"),
    ],
)
def test_beats_only_when_the_interval_excludes_zero(low, high, relation):
    v = compare.verdict("sys", "ref", low, high)
    assert v["relation"] == relation
    assert ("beats" in v["text"]) == (relation != "no_significant_difference")
    if relation == "no_significant_difference":
        assert v["text"] == "no significant difference between sys and ref"
    elif relation == "system_beats_reference":
        assert v["text"] == "sys beats ref"
    else:
        assert v["text"] == "ref beats sys"


def test_the_wording_follows_the_interval_in_the_document(standard):
    _, doc = standard
    for s in doc["systems"]:
        pair = s["vs_reference"]
        if pair is None:
            continue
        low, high = pair["difference"]["ci95"]
        excludes_zero = low > 0 or high < 0
        assert ("beats" in pair["verdict"]["text"]) == excludes_zero, s["name"]


@pytest.mark.parametrize(
    ("ref_only", "system_only", "relation", "text"),
    [
        (30, 10, "system_beats_reference", f"{GPT_OSS_20B} beats {FT}"),
        (10, 30, "reference_beats_system", f"{FT} beats {GPT_OSS_20B}"),
        (12, 10, "no_significant_difference", f"no significant difference between {GPT_OSS_20B} and {FT}"),
    ],
)
def test_the_three_wordings_end_to_end(tmp_path, monkeypatch, ref_only, system_only, relation, text):
    lab = two_systems(tmp_path, monkeypatch, ref_only=ref_only, system_only=system_only, both=20)
    pair = system(build(lab, 1000), GPT_OSS_20B)["vs_reference"]
    # ref_only counts items the fine-tune gets wrong: that favours the system
    assert pair["verdict"] == {"relation": relation, "text": text}


def test_an_interval_and_a_mcnemar_test_that_disagree_are_flagged(tmp_path, monkeypatch):
    # 12 items won against 4 lost: the exact McNemar p is 0.077 (not significant). Force an interval
    # that just excludes 0, which a real resample can produce on an edge case like this.
    lab = two_systems(tmp_path, monkeypatch, ref_only=12, system_only=4, both=0)
    monkeypatch.setattr(metrics, "paired_bootstrap", lambda *a, **k: metrics.PairedResult(0.016, 0.001, 0.031, 0.04, 500))
    doc = build(lab)
    pair = system(doc, GPT_OSS_20B)["vs_reference"]
    assert pair["mcnemar"]["p_value"] == pytest.approx(mcnemar_p(12, 4)) and pair["mcnemar"]["p_value"] > 0.05
    assert pair["verdict"]["relation"] == "system_beats_reference"  # the wording follows the interval
    flagged = [w for w in doc["warnings"] if "disagree" in w]
    assert len(flagged) == 1 and GPT_OSS_20B in flagged[0] and "the wording follows the interval" in flagged[0]


def test_the_converse_disagreement_is_flagged_too(tmp_path, monkeypatch):
    lab = two_systems(tmp_path, monkeypatch, ref_only=25, system_only=5, both=0)  # McNemar p is far below 0.05
    monkeypatch.setattr(metrics, "paired_bootstrap", lambda *a, **k: metrics.PairedResult(0.04, -0.001, 0.08, 0.04, 500))
    doc = build(lab)
    assert system(doc, GPT_OSS_20B)["vs_reference"]["verdict"]["relation"] == "no_significant_difference"
    assert len([w for w in doc["warnings"] if "disagree" in w]) == 1


def test_agreement_raises_no_flag(tmp_path, monkeypatch):
    lab = two_systems(tmp_path, monkeypatch, ref_only=30, system_only=10, both=20)
    doc = build(lab, 1000)
    assert system(doc, GPT_OSS_20B)["vs_reference"]["verdict"]["relation"] == "system_beats_reference"
    assert not [w for w in doc["warnings"] if "disagree" in w]


# --- unseen text and scenarios ------------------------------------------------------------------------------------


def test_exact_match_on_items_whose_text_is_not_in_train(standard):
    lab, doc = standard
    in_train = set(json.loads((lab.results / "data_audit.json").read_text())["overlap"]["test_item_ids_in_train"])
    assert in_train == set(lab.planted) and len(in_train) == 3
    assert doc["data"]["audit"]["test_items_with_text_in_train"] == 3
    s500 = lab.ids("S500")
    for name in (FT, *API_SYSTEMS):
        s = system(doc, name)
        ids = [i for i in lab.ids(s["comparison_subset"]) if i not in in_train]
        block = s["unseen_text"]
        assert block["n"] == len(ids) and block["excluded"] == s["n_subset"] - len(ids)
        assert block["value"] == pytest.approx(em_vector(lab, name, ids).mean())
        assert block["ci95"] == list(metrics.bootstrap_ci(em_vector(lab, name, ids), n_resamples=N))
    assert system(doc, FT)["unseen_text"]["excluded"] == len(in_train & set(s500)) == 2
    assert system(doc, FT)["full_test"]["unseen_text"]["excluded"] == 3  # the planted item outside S500 counts here


def test_the_unseen_text_figure_follows_a_smaller_subset_too(mixed):
    lab, doc = mixed
    in_train = set(lab.planted)
    ids = [i for i in lab.ids("S300") if i not in in_train]
    block = system(doc, GEMINI)["unseen_text"]
    assert block["n"] == len(ids) and block["excluded"] == 300 - len(ids) == len(in_train & set(lab.ids("S300")))
    assert block["value"] == pytest.approx(em_vector(lab, GEMINI, ids).mean())


def test_exact_match_per_scenario(standard):
    lab, doc = standard
    s = system(doc, GPT_OSS_20B)
    ids = lab.ids("S500")
    assert sum(v["n"] for v in s["per_scenario"].values()) == len(ids)
    for scenario, cell in s["per_scenario"].items():
        in_scenario = [i for i in ids if lab.by_id[i].scenario == scenario]
        assert cell["n"] == len(in_scenario)
        assert cell["exact_match"] == pytest.approx(em_vector(lab, GPT_OSS_20B, in_scenario).mean())
    assert len(s["per_scenario"]) == 18


# --- cost, break-even, self-hosted ---------------------------------------------------------------------------------


def test_api_cost_per_1k_calls_has_both_cache_bounds(standard):
    lab, doc = standard
    # 1,000 prompt tokens (800 of them the static prefix) and 100 completion tokens per call, prices per million
    expected = {
        GPT_OSS_20B: {"upper": 1.2, "lower": 0.6},  # (1000 * 1.0 + 100 * 2.0) and (200 * 1.0 + 800 * 0.25 + 100 * 2.0)
        GPT_OSS_120B: {"upper": 0.6, "lower": 0.4},
        QWEN_27B: {"upper": 2.4, "lower": 2.4},  # (1000 * 2.0 + 100 * 4.0) twice: no cached-input price, nothing is discounted
        GEMINI: {"upper": 1.4, "lower": 1.0},  # (1000 * 1.0 + 100 * 4.0) and (200 * 1.0 + 800 * 0.5 + 100 * 4.0)
    }
    assert set(expected) == set(API_SYSTEMS)
    for name, bounds in expected.items():
        c = system(doc, name)["cost"]
        assert c["per_1k_calls_usd"]["upper"] == pytest.approx(bounds["upper"])
        assert c["per_1k_calls_usd"]["lower"] == pytest.approx(bounds["lower"])
        assert c["billing_basis"] == "free tier; priced at paid list price"
        entry = lab.sources["prices"][config.resolve_system(name, lab.config_dir)["price_id"]]
        assert c["price"]["url"].startswith("https://")
        assert (c["price"]["url"], c["price"]["retrieved_on"]) == (entry["url"], entry["retrieved_on"])  # the entry the run used
        assert c["calls"] == 500
    assert "cached_input" not in system(doc, QWEN_27B)["cost"]["price"]["usd_per_mtok"]


def test_self_hosted_cost_is_the_gpu_price_at_the_operating_point(standard):
    _, doc = standard
    # $0.50 an hour, kept busy at 10 requests a second
    c = system(doc, FT)["cost"]
    assert c["per_1k_calls_usd"] == pytest.approx(0.5 / 3600 / 10 * 1000)
    assert c["requests_per_s"] == 10.0 and c["concurrency"] == 8
    assert c["capacity_calls_per_month"] == pytest.approx(10 * 3600 * 730)
    assert system(doc, BASE)["cost"] is None  # the benchmark file covers the fine-tune only


def test_break_even_is_the_monthly_gpu_bill_over_the_api_cost_per_call(standard):
    _, doc = standard
    be = doc["break_even"]
    assert be["gpu"]["usd_per_hour"] == 0.5 and be["gpu"]["hours_per_month"] == 730
    assert be["gpu"]["monthly_usd"] == pytest.approx(365.0)
    assert be["gpu"]["url"] == "https://instances.vantage.sh/aws/ec2/g4dn.xlarge"
    assert set(be["apis"]) == set(API_SYSTEMS)
    # $365 a month over $0.0012 / $0.0006 a call
    small = be["apis"][GPT_OSS_20B]["calls_per_month"]
    assert small["no_caching"] == pytest.approx(365.0 / 0.0012) and small["cached_prefix"] == pytest.approx(365.0 / 0.0006)
    assert be["apis"][GPT_OSS_120B]["calls_per_month"]["cached_prefix"] == pytest.approx(365.0 / 0.0004)
    assert be["apis"][GEMINI]["calls_per_month"]["no_caching"] == pytest.approx(365.0 / 0.0014)
    assert be["apis"][GEMINI]["calls_per_month"]["cached_prefix"] == pytest.approx(365.0 / 0.0010)
    assert be["apis"][GPT_OSS_20B]["requests_per_s"]["no_caching"] == pytest.approx(365.0 / 0.0012 / (730 * 3600))
    for name, entry in be["apis"].items():
        no_caching, cached_prefix = entry["calls_per_month"]["no_caching"], entry["calls_per_month"]["cached_prefix"]
        if name == QWEN_27B:  # no cached-input price: caching saves nothing, so the two volumes coincide
            assert no_caching == pytest.approx(365.0 / 0.0024) and cached_prefix == pytest.approx(no_caching)
        else:  # the smaller volume is the dearer API bound
            assert no_caching < cached_prefix, name
    assert be["capacity_calls_per_month"] == pytest.approx(26_280_000)
    assert all(all(v is True for v in e["within_capacity"].values()) for e in be["apis"].values())


def test_a_break_even_above_what_one_gpu_can_serve_is_marked(tmp_path, monkeypatch):
    quiet_git(monkeypatch)
    lab = Lab(tmp_path)
    lab.write_run(FT)
    lab.write_run(GPT_OSS_20B)
    slow = {"gpu": "Tesla T4", "system": FT, "levels": [{"concurrency": 1, "requests_per_s": 0.001, "latency_s": {"p50": 9.0, "p95": 9.5}}], "operating_point": 1}
    lab.write_serving(doc=slow)
    be = build(lab)["break_even"]
    assert be["capacity_calls_per_month"] == pytest.approx(0.001 * 3600 * 730)  # 2,628 calls a month
    assert be["apis"][GPT_OSS_20B]["within_capacity"] == {"no_caching": False, "cached_prefix": False}


def test_without_a_benchmark_the_break_even_stands_but_capacity_and_self_hosted_cost_do_not(tmp_path, monkeypatch):
    quiet_git(monkeypatch)
    lab = Lab(tmp_path)
    lab.write_run(FT)
    lab.write_run(GPT_OSS_20B)
    doc = build(lab)
    be = doc["break_even"]
    assert be["capacity_calls_per_month"] is None
    assert be["apis"][GPT_OSS_20B]["calls_per_month"]["no_caching"] == pytest.approx(365.0 / 0.0012)
    assert be["apis"][GPT_OSS_20B]["within_capacity"] == {"no_caching": None, "cached_prefix": None}
    assert system(doc, FT)["cost"] is None and doc["self_hosted"]["benchmark"] is None
    assert any("no throughput benchmark" in w for w in doc["warnings"])


def test_a_free_api_row_has_no_break_even_volume():
    breakeven = compare.api_break_even({"upper": 0.0, "lower": 0.0}, {"usd_per_hour": 0.5}, None)
    assert breakeven["calls_per_month"] == {"no_caching": None, "cached_prefix": None}


# --- latency --------------------------------------------------------------------------------------------------------


def test_the_headline_latency_is_self_hosted_and_on_the_box(standard):
    _, doc = standard
    latency = doc["latency"]
    assert latency["self_hosted"]["basis"] == "on the box (no network path)"
    assert set(latency["self_hosted"]["systems"]) == {FT}
    ft = latency["self_hosted"]["systems"][FT]
    assert (ft["concurrency_1"]["p50_s"], ft["concurrency_1"]["p95_s"]) == (0.4, 0.6)
    assert (ft["operating_point"]["concurrency"], ft["operating_point"]["p95_s"]) == (8, 1.2)
    assert latency["self_hosted"]["gpu"] == "Tesla T4"


def test_api_latency_is_an_appendix_with_its_label(standard):
    _, doc = standard
    appendix = doc["latency"]["api_appendix"]
    assert appendix["label"] == "observed on free tiers from India; not representative of paid tiers"
    assert set(appendix["systems"]) == set(API_SYSTEMS)
    for entry in appendix["systems"].values():
        assert entry["p50"] <= entry["p95"] and entry["method"] == "nearest-rank" and entry["n"] > 0
    assert "Headline latency is the self-hosted rows" in doc["latency"]["policy"]


# --- the model name changing during a run ---------------------------------------------------------------------------------


def test_a_model_name_that_changes_partway_through_is_flagged(tmp_path, monkeypatch):
    quiet_git(monkeypatch)
    lab = Lab(tmp_path)
    lab.write_run(FT)
    lab.write_run(GEMINI, models=lambda position: "gemini-3.8-flash-001" if position <= 120 else "gemini-3.8-flash-002")
    lab.write_run(GPT_OSS_120B)
    doc = build(lab)
    names = system(doc, GEMINI)["model_names"]
    assert names["changed"] is True
    assert names["returned"] == ["gemini-3.8-flash-001", "gemini-3.8-flash-002"]
    assert [(s["rows"], s["first_row"], s["last_row"]) for s in names["segments"]] == [(120, 1, 120), (380, 121, 500)]
    assert names["requested"] == "gemini-3.8-flash"
    flagged = [w for w in doc["warnings"] if "model name" in w]
    assert len(flagged) == 1 and GEMINI in flagged[0] and "answer 120 of 500" in flagged[0]
    assert system(doc, GPT_OSS_120B)["model_names"]["changed"] is False


def test_a_name_that_flips_back_is_still_a_change(tmp_path, monkeypatch):
    quiet_git(monkeypatch)
    lab = Lab(tmp_path)
    lab.write_run(FT)
    lab.write_run(GPT_OSS_20B, models=lambda position: "b" if 100 < position <= 200 else "a")
    names = system(build(lab), GPT_OSS_20B)["model_names"]
    assert names["changed"] is True and names["returned"] == ["a", "b"] and len(names["segments"]) == 3


def test_a_steady_model_name_raises_no_flag(standard):
    _, doc = standard
    assert [s["model_names"]["changed"] for s in doc["systems"]] == [False] * len(SYSTEMS)
    assert not [w for w in doc["warnings"] if "model name" in w]
    assert system(doc, GPT_OSS_20B)["model_names"]["returned"] == [STUB_MODEL]


# --- missing and damaged files --------------------------------------------------------------------------------------------


def test_with_no_results_at_all_every_system_is_missing_and_nothing_is_needed(tmp_path):
    # the repository's own configs, and no data directory: nothing has been run, so nothing is read
    doc = compare.build_comparison(results_dir=tmp_path / "results", processed_dir=tmp_path / "no-data", n_resamples=N)
    assert doc["has_results"] is False
    assert [s["status"] for s in doc["systems"]] == ["missing"] * len(SYSTEMS)
    assert all(s["metrics"] is None and s["vs_reference"] is None and s["cost"] is None for s in doc["systems"])
    assert doc["break_even"]["apis"] == {} and doc["latency"]["api_appendix"]["systems"] == {}
    assert doc["warnings"] == []  # nothing is wrong: nothing has been run yet
    json.dumps(doc, allow_nan=False)


def test_run_writes_the_file_and_exits_zero_even_with_no_results(tmp_path):
    lines: list[str] = []
    code = compare.run(results_dir=tmp_path / "results", processed_dir=tmp_path / "nodata", out=lines.append)
    assert code == 0
    written = json.loads((tmp_path / "results" / "comparison.json").read_text())
    assert written["has_results"] is False and written["schema_version"] == 1
    assert any("no test results" in line for line in lines)


def test_missing_test_labels_stop_the_run_when_there_are_results_to_score(tmp_path, monkeypatch):
    quiet_git(monkeypatch)
    lab = Lab(tmp_path)
    lab.write_run(FT)
    lines: list[str] = []
    code = compare.run(results_dir=lab.results, processed_dir=tmp_path / "gone", config_dir=lab.config_dir, out=lines.append)
    assert code == 2 and "test.jsonl not found" in lines[-1]
    assert not (lab.results / "comparison.json").exists()


def test_a_subset_that_no_longer_matches_stops_the_run(tmp_path, monkeypatch):
    quiet_git(monkeypatch)
    lab = Lab(tmp_path)
    lab.write_run(FT)
    path = lab.processed / "subsets.json"
    doc = json.loads(path.read_text())
    doc["subsets"]["S500"]["hash"] = "0" * 64
    path.write_text(json.dumps(doc))
    with pytest.raises(compare.CompareError, match="subsets"):
        build(lab)


def test_a_missing_reference_leaves_the_others_unpaired_and_says_so(tmp_path, monkeypatch):
    quiet_git(monkeypatch)
    lab = Lab(tmp_path)
    lab.write_run(GPT_OSS_20B)
    doc = build(lab)
    assert system(doc, GPT_OSS_20B)["status"] == "complete" and system(doc, GPT_OSS_20B)["vs_reference"] is None
    assert system(doc, FT)["status"] == "missing"
    assert any("reference system" in w and FT in w for w in doc["warnings"])


def test_an_unknown_reference_is_an_error(tmp_path):
    with pytest.raises(compare.CompareError, match="nope"):
        compare.build_comparison(results_dir=tmp_path, processed_dir=tmp_path, reference="nope")


def test_a_missing_summary_costs_the_cost_and_the_provenance_and_nothing_else(tmp_path, monkeypatch):
    quiet_git(monkeypatch)
    lab = Lab(tmp_path)
    lab.write_run(FT)
    lab.write_run(GPT_OSS_20B, summary=False)
    doc = build(lab)
    s = system(doc, GPT_OSS_20B)
    assert s["status"] == "complete" and s["metrics"]["exact_match"]["value"] == 1.0
    assert s["cost"] is None and s["run"] is None and s["latency_observed"] is None
    assert any(GPT_OSS_20B in w and "no summary file" in w for w in doc["warnings"])
    assert GPT_OSS_20B not in doc["break_even"]["apis"]


def test_a_missing_audit_leaves_out_the_unseen_text_figure(tmp_path, monkeypatch):
    quiet_git(monkeypatch)
    lab = Lab(tmp_path)
    lab.write_run(FT)
    doc = build(lab)
    assert system(doc, FT)["unseen_text"] is None and doc["data"]["audit"] is None
    assert any("data_audit.json not found" in w for w in doc["warnings"])


def test_a_damaged_predictions_file_marks_that_system_unreadable_and_keeps_going(tmp_path, monkeypatch):
    quiet_git(monkeypatch)
    lab = Lab(tmp_path)
    lab.write_run(FT)
    run_dir = lab.write_run(GPT_OSS_20B)
    (run_dir / "predictions.jsonl").write_text('{"id": "1", "error": null}\n{"id": "2", "tex')
    doc = build(lab)
    assert system(doc, GPT_OSS_20B)["status"] == "unreadable" and system(doc, FT)["status"] == "complete"
    assert any("could not read predictions.jsonl" in w and GPT_OSS_20B in w for w in doc["warnings"])


def test_a_partial_run_is_scored_on_what_it_has_and_flagged(tmp_path, monkeypatch):
    quiet_git(monkeypatch)
    lab = Lab(tmp_path)
    ids = lab.ids("S500")
    lab.write_run(FT, wrong=ids[:50])
    lab.write_run(GPT_OSS_20B, ids=ids[:200], wrong=ids[:20])
    doc = build(lab)
    s = system(doc, GPT_OSS_20B)
    assert (s["status"], s["n_subset"], s["n_scored"]) == ("partial", 500, 200)
    assert s["vs_reference"]["n"] == 200 and s["vs_reference"]["complete"] is False
    assert any(GPT_OSS_20B in w and "200 of the 500" in w for w in doc["warnings"])
    assert any("partial summary" in w and GPT_OSS_20B in w for w in doc["warnings"])
    assert s["run"]["summary_is_partial"] is True


def test_a_run_outside_the_test_lock_is_flagged(tmp_path, monkeypatch):
    quiet_git(monkeypatch)
    lab = Lab(tmp_path)
    lab.write_run(FT)
    lab.write_run(GPT_OSS_20B, locked=False)
    doc = build(lab)
    assert any(GPT_OSS_20B in w and "test lock" in w for w in doc["warnings"])
    assert not any(FT in w and "test lock" in w for w in doc["warnings"])
    assert system(doc, FT)["run"]["lock"]["reason"] == "frozen for the test run"


def test_estimated_token_counts_are_flagged(tmp_path, monkeypatch):
    quiet_git(monkeypatch)
    lab = Lab(tmp_path)
    lab.write_run(FT)
    run_dir = lab.write_run(GPT_OSS_20B)
    path = run_dir / "summary.S500.json"
    summary = json.loads(path.read_text())
    summary["tokens"]["calls_with_estimated_usage"] = 7
    path.write_text(json.dumps(summary))
    assert any("7 calls had no token usage" in w for w in build(lab)["warnings"])


def test_a_changed_price_entry_is_flagged(tmp_path, monkeypatch):
    quiet_git(monkeypatch)
    lab = Lab(tmp_path)
    lab.write_run(FT)
    lab.write_run(GPT_OSS_20B)
    lab.sources["prices"]["groq-gpt-oss-20b"]["usd_per_mtok"]["input"] = 9.0
    (lab.config_dir / "sources.yaml").write_text(yaml.safe_dump(lab.sources, sort_keys=False))
    doc = build(lab)
    assert any("price entry" in w and GPT_OSS_20B in w for w in doc["warnings"])
    assert system(doc, GPT_OSS_20B)["cost"]["per_1k_calls_usd"]["upper"] == pytest.approx(1.2)  # the entry the run recorded


# --- the benchmark file --------------------------------------------------------------------------------------------------


def test_a_benchmark_for_another_gpu_is_listed_but_not_used(tmp_path, monkeypatch):
    quiet_git(monkeypatch)
    lab = Lab(tmp_path)
    lab.write_run(FT)
    lab.write_serving(name="l4", doc={"gpu": "NVIDIA L4", "system": FT, "levels": [{"concurrency": 1, "requests_per_s": 5, "latency_s": {"p50": 0.1, "p95": 0.2}}], "operating_point": 1})
    doc = build(lab)
    assert system(doc, FT)["cost"] is None and doc["self_hosted"]["benchmark"] is None
    assert any("results/serving/l4.json" in w and "none for the rented GPU" in w for w in doc["warnings"])


def test_the_file_for_the_rented_gpu_is_chosen_among_several(tmp_path, monkeypatch):
    quiet_git(monkeypatch)
    lab = Lab(tmp_path)
    lab.write_run(FT)
    lab.write_serving(name="a100", doc={"gpu": "A100", "system": FT, "levels": [{"concurrency": 1, "requests_per_s": 50, "latency_s": {"p50": 0.1, "p95": 0.2}}], "operating_point": 1})
    lab.write_serving(name="kaggle-t4")
    doc = build(lab)
    assert doc["self_hosted"]["benchmark"]["path"] == "results/serving/kaggle-t4.json"
    assert doc["self_hosted"]["benchmark"]["other_files"] == ["results/serving/a100.json"]
    assert system(doc, FT)["cost"]["requests_per_s"] == 10.0


@pytest.mark.parametrize(
    ("block", "p50", "p95", "rate", "concurrency"),
    [
        # levels keyed by concurrency, flat p50_s/p95_s, operating point given as a number
        ({"levels": {"1": {"p50_s": 0.3, "p95_s": 0.5, "req_per_s": 3.0}, "4": {"p50_s": 0.6, "p95_s": 0.9, "rps": 9.0}}, "operating_point": 4}, 0.3, 0.9, 9.0, 4),
        # a list, the nested spelling, the operating point as an object
        ({"levels": [{"concurrency": 1, "latency_s": {"p50": 0.3, "p95": 0.5}, "requests_per_s": 3.0}, {"concurrency": 2, "latency_s": {"p50": 0.4, "p95": 0.7}, "requests_per_s": 5.0}], "operating_point": {"concurrency": 2}}, 0.3, 0.7, 5.0, 2),
    ],
)
def test_the_accepted_benchmark_shapes(block, p50, p95, rate, concurrency):
    entry = compare.normalize_benchmark({"gpu": "T4", "system": FT, **block}, FT)[FT]
    assert entry["problems"] == []
    assert entry["single_stream"]["p50_s"] == p50
    assert entry["operating_point"]["p95_s"] == p95 and entry["operating_point"]["requests_per_s"] == rate
    assert entry["operating_point"]["concurrency"] == concurrency


def test_a_benchmark_that_found_no_operating_point_gives_its_reason(tmp_path, monkeypatch):
    quiet_git(monkeypatch)
    lab = Lab(tmp_path)
    lab.write_run(FT)
    note = "no concurrency level had p95 latency <= 1 s with no failed request"
    lab.write_serving(doc={"gpu": "Tesla T4", "system": FT, "levels": [GOOD_LEVEL], "operating_point": None, "operating_point_note": note})
    doc = build(lab)
    assert any(f"no operating point ({note})" in w for w in doc["warnings"])
    assert system(doc, FT)["cost"] is None  # nothing is invented in its place
    assert doc["latency"]["self_hosted"]["systems"][FT]["operating_point"] is None
    assert doc["latency"]["self_hosted"]["systems"][FT]["concurrency_1"]["p95_s"] == 0.2  # concurrency 1 is still reported


def test_a_benchmark_with_one_block_per_system(tmp_path, monkeypatch):
    quiet_git(monkeypatch)
    lab = Lab(tmp_path)
    lab.write_run(FT)
    lab.write_run(BASE)

    def levels(rate):
        return [
            {"concurrency": 1, "requests_per_s": rate / 4, "latency_s": {"p50": 0.2, "p95": 0.3}},
            {"concurrency": 4, "requests_per_s": rate, "latency_s": {"p50": 0.5, "p95": 0.8}},
        ]

    lab.write_serving(doc={"gpu": "T4", "systems": {FT: {"levels": levels(8.0), "operating_point": 4}, BASE: {"levels": levels(2.0), "operating_point": 4}}})
    doc = build(lab)
    assert system(doc, FT)["cost"]["requests_per_s"] == 8.0 and system(doc, BASE)["cost"]["requests_per_s"] == 2.0
    assert system(doc, BASE)["cost"]["per_1k_calls_usd"] == pytest.approx(4 * system(doc, FT)["cost"]["per_1k_calls_usd"])


@pytest.mark.parametrize(
    ("doc", "problem", "priced"),
    [
        ({"levels": []}, "no usable levels", False),
        ({"levels": [GOOD_LEVEL]}, "no operating point declared", False),
        ({"levels": [{**GOOD_LEVEL, "concurrency": 2}], "operating_point": 2}, "no level at concurrency 1", True),
        ({"levels": [GOOD_LEVEL], "operating_point": 16}, "is not one of the measured", False),
        # milliseconds are not rescaled: the level is reported as unusable, while its throughput still prices
        ({"levels": [{"concurrency": 1, "requests_per_s": 1, "latency_ms": {"p50": 100, "p95": 200}}], "operating_point": 1}, "lacks latency_s.p50", True),
    ],
)
def test_an_unusable_benchmark_is_reported_not_guessed(tmp_path, monkeypatch, doc, problem, priced):
    quiet_git(monkeypatch)
    lab = Lab(tmp_path)
    lab.write_run(FT)
    lab.write_serving(doc={"gpu": "T4", "system": FT, **doc})
    out = build(lab)
    assert [w for w in out["warnings"] if problem in w and "results/serving/t4.json" in w]
    assert (system(out, FT)["cost"] is not None) == priced


# --- the command line ----------------------------------------------------------------------------------------------------------


def test_main_writes_comparison_json_and_prints_the_findings(tmp_path, monkeypatch):
    quiet_git(monkeypatch)
    lab = Lab(tmp_path).populate()
    lines: list[str] = []
    code = compare.main(
        ["--n-resamples", "50"], results_dir=lab.results, processed_dir=lab.processed, config_dir=lab.config_dir, out=lines.append
    )
    assert code == 0
    written = json.loads((lab.results / "comparison.json").read_text())
    assert written["has_results"] is True and written["bootstrap"]["resamples"] == 50
    assert any(line.strip().startswith(GPT_OSS_20B) and "exact match" in line for line in lines)
    assert (lab.results / "comparison.json").read_text().endswith("}\n")


def test_the_reference_can_be_changed(tmp_path, monkeypatch):
    quiet_git(monkeypatch)
    lab = Lab(tmp_path).populate()
    doc = build(lab, reference=BASE)
    assert doc["reference"] == BASE
    assert system(doc, BASE)["vs_reference"] is None and system(doc, FT)["vs_reference"]["reference"] == BASE


def test_the_document_is_plain_json_with_no_nan(standard):
    _, doc = standard
    text = json.dumps(doc, allow_nan=False)
    assert json.loads(text) == doc
    assert doc["has_results"] is True and doc["billing_basis"] == "free tier; priced at paid list price"
