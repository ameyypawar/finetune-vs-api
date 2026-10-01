"""Exact match, slot counts, F1 edge cases, summary, and the statistics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from finetune_vs_api.metrics import (
    BOOTSTRAP_SEED,
    Item,
    bootstrap_ci,
    exact_match,
    f1_stat,
    item_scores,
    mcnemar_exact,
    normalize_value,
    paired_bootstrap,
    percentile,
    prf,
    slot_counts,
    summarize,
)


def call(intent, *slots):
    return {"intent": intent, "slots": [{"type": t, "value": v} for t, v in slots]}


# --- normalization ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Five AM", "five am"),
        ("  five   am  ", "five am"),
        ("five\tam\n", "five am"),
        ("", ""),
        ("5:30", "5:30"),  # punctuation is left alone
        ("CafÉ", "café"),
    ],
)
def test_normalize_value(raw, expected):
    assert normalize_value(raw) == expected


# --- exact match -----------------------------------------------------------------------


def test_exact_match_ignores_slot_order():
    gold = call("alarm_set", ("time", "six"), ("date", "friday"))
    pred = call("alarm_set", ("date", "friday"), ("time", "six"))
    assert exact_match(pred, gold)


def test_exact_match_normalizes_case_and_whitespace_in_values():
    assert exact_match(call("a", ("time", "Six  AM")), call("a", ("time", "six am")))


def test_exact_match_needs_the_same_slot_type():
    assert not exact_match(call("a", ("date", "friday")), call("a", ("time", "friday")))


def test_exact_match_needs_the_same_intent():
    assert not exact_match(call("b", ("time", "six")), call("a", ("time", "six")))


def test_exact_match_counts_duplicates():
    once = call("a", ("date", "monday"))
    twice = call("a", ("date", "monday"), ("date", "monday"))
    assert not exact_match(twice, once)
    assert not exact_match(once, twice)
    assert exact_match(twice, twice)


def test_exact_match_with_empty_slot_lists():
    assert exact_match(call("general_greet"), call("general_greet"))
    assert not exact_match(call("general_greet"), call("general_greet", ("time", "six")))
    assert not exact_match(call("general_greet", ("time", "six")), call("general_greet"))


def test_an_unparseable_prediction_never_matches():
    assert not exact_match(None, call("a"))


# --- slot counts ---------------------------------------------------------------------------


def test_slot_counts_basic():
    gold = [("time", "six"), ("date", "friday")]
    pred = [("time", "six"), ("date", "monday"), ("place_name", "paris")]
    assert slot_counts(pred, gold) == (1, 2, 1)


def test_slot_counts_duplicates_are_a_multiset():
    assert slot_counts([("d", "x"), ("d", "x")], [("d", "x")]) == (1, 1, 0)
    assert slot_counts([("d", "x")], [("d", "x"), ("d", "x")]) == (1, 0, 1)
    assert slot_counts([("d", "x"), ("d", "x")], [("d", "x"), ("d", "x")]) == (2, 0, 0)


def test_slot_counts_empty_lists():
    assert slot_counts([], []) == (0, 0, 0)
    assert slot_counts(None, None) == (0, 0, 0)
    assert slot_counts([], [("d", "x")]) == (0, 0, 1)
    assert slot_counts([("d", "x")], []) == (0, 1, 0)


def test_slot_counts_accepts_dicts_and_normalizes_values():
    assert slot_counts([{"type": "t", "value": "Six"}], [("t", "six")]) == (1, 0, 0)


def test_prf_zero_denominators_give_zero_not_nan():
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)
    assert prf(0, 3, 0) == (0.0, 0.0, 0.0)
    assert prf(0, 0, 3) == (0.0, 0.0, 0.0)


def test_prf_values():
    p, r, f1 = prf(2, 1, 3)
    assert (p, r) == (pytest.approx(2 / 3), pytest.approx(2 / 5))
    assert f1 == pytest.approx(2 * p * r / (p + r))


# --- summarize ---------------------------------------------------------------------------


def item(text, gold, pred, valid=True):
    return Item(text, gold, pred, valid)


def test_summarize_perfect_predictions():
    gold = call("a", ("time", "six"))
    s = summarize([item("at six", gold, gold)] * 4, with_ci=False)
    assert s["exact_match"]["value"] == 1.0
    assert s["slot_f1"]["value"] == 1.0
    assert s["schema_valid_rate"]["value"] == 1.0
    assert s["unfound_value_rate"]["value"] == 0.0


def test_summarize_micro_averages_over_all_slots():
    # Example 1: 2 gold slots, both right.  Example 2: 1 gold slot, 1 wrong, 1 spurious.
    g1, g2 = call("a", ("t", "x"), ("t", "y")), call("a", ("t", "z"))
    p2 = call("a", ("t", "w"), ("u", "v"))
    s = summarize([item("x y", g1, g1), item("z", g2, p2)], with_ci=False)
    assert s["slot_counts"] == {"tp": 2, "fp": 2, "fn": 1}
    assert s["slot_precision"]["value"] == pytest.approx(2 / 4)
    assert s["slot_recall"]["value"] == pytest.approx(2 / 3)
    assert s["exact_match"]["value"] == 0.5
    assert s["intent_accuracy"]["value"] == 1.0


def test_an_invalid_output_scores_zero_and_its_gold_slots_are_false_negatives():
    gold = call("a", ("t", "x"), ("t", "y"))
    s = summarize([item("x y", gold, call("a", ("t", "x"), ("t", "y")), valid=False)], with_ci=False)
    assert s["exact_match"]["value"] == 0.0
    assert s["intent_accuracy"]["value"] == 0.0
    assert s["schema_valid_rate"]["value"] == 0.0
    assert s["slot_counts"] == {"tp": 0, "fp": 0, "fn": 2}


def test_a_missing_prediction_is_invalid_too():
    gold = call("a", ("t", "x"))
    s = summarize([item("x", gold, None, valid=False)], with_ci=False)
    assert s["slot_counts"] == {"tp": 0, "fp": 0, "fn": 1}


def test_unfound_value_rate_is_the_share_of_predicted_values_missing_from_the_request():
    gold = call("a", ("t", "six"))
    pred = call("a", ("t", "six"), ("t", "6 am"), ("t", "SIX"))  # "6 am" is not in the request; "SIX" is
    s = summarize([item("wake me at six", gold, pred)], with_ci=False)
    assert s["unfound_value_rate"]["value"] == pytest.approx(1 / 3)
    assert s["predicted_slots"] == 3


def test_all_empty_slot_lists_do_not_divide_by_zero():
    gold = call("general_greet")
    s = summarize([item("hello", gold, gold)] * 3, with_ci=False)
    assert s["exact_match"]["value"] == 1.0
    assert s["slot_f1"]["value"] == 0.0  # nothing to extract: F1 is defined as 0.0
    assert s["unfound_value_rate"]["value"] == 0.0


def test_summarize_rejects_no_items():
    with pytest.raises(ValueError):
        summarize([])


def test_summarize_with_intervals_is_reproducible_and_brackets_the_value():
    gold = call("a", ("t", "x"))
    items = [item("x", gold, gold if i % 4 else call("b"), True) for i in range(80)]
    first = summarize(items, n_resamples=500)
    second = summarize(items, n_resamples=500)
    assert first == second
    for key in ("exact_match", "intent_accuracy", "slot_f1", "schema_valid_rate"):
        low, high = first[key]["ci95"]
        assert low <= first[key]["value"] <= high


def test_item_scores_alignment():
    gold = call("a", ("t", "x"))
    s = item_scores([item("x", gold, gold), item("x", gold, call("b"), True)])
    assert s["em"].tolist() == [1.0, 0.0]
    assert s["fn"].tolist() == [0.0, 1.0]


# --- bootstrap -----------------------------------------------------------------------------


def test_bootstrap_ci_is_reproducible_for_a_fixed_seed():
    values = np.random.default_rng(0).integers(0, 2, 200).astype(float)
    assert bootstrap_ci(values, n_resamples=2000) == bootstrap_ci(values, n_resamples=2000)
    assert bootstrap_ci(values, n_resamples=2000, seed=1) == bootstrap_ci(values, n_resamples=2000, seed=1)


def test_bootstrap_ci_changes_with_the_seed():
    values = np.random.default_rng(0).integers(0, 2, 200).astype(float)
    assert bootstrap_ci(values, n_resamples=2000, seed=1) != bootstrap_ci(values, n_resamples=2000, seed=2)


def test_the_default_seed_and_resample_count_are_the_documented_ones():
    assert BOOTSTRAP_SEED == 20261001
    values = np.random.default_rng(0).integers(0, 2, 60).astype(float)
    assert bootstrap_ci(values) == bootstrap_ci(values, n_resamples=10_000, seed=BOOTSTRAP_SEED)


def test_bootstrap_ci_brackets_the_mean_and_narrows_with_more_data():
    rng = np.random.default_rng(3)
    small = rng.integers(0, 2, 50).astype(float)
    large = rng.integers(0, 2, 5000).astype(float)
    lo_s, hi_s = bootstrap_ci(small, n_resamples=2000)
    lo_l, hi_l = bootstrap_ci(large, n_resamples=2000)
    assert lo_s <= small.mean() <= hi_s
    assert lo_l <= large.mean() <= hi_l
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_bootstrap_ci_of_constant_data_is_a_point():
    assert bootstrap_ci(np.ones(30), n_resamples=200) == (1.0, 1.0)


def test_bootstrap_ci_rejects_empty_input():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_bootstrap_ci_supports_a_multi_column_statistic():
    # per-item (tp, fp, fn); the interval for micro-F1 must bracket the observed F1.
    cols = np.array([[1, 0, 0]] * 30 + [[0, 1, 1]] * 10 + [[1, 1, 0]] * 10, dtype=float)
    tp, fp, fn = cols.sum(axis=0)
    observed = prf(tp, fp, fn)[2]
    low, high = bootstrap_ci(cols, f1_stat, n_resamples=1000)
    assert low <= observed <= high


# --- paired bootstrap ------------------------------------------------------------------------


def test_paired_bootstrap_identical_systems_show_no_difference():
    a = np.random.default_rng(1).integers(0, 2, 120).astype(float)
    result = paired_bootstrap(a, a.copy(), n_resamples=1000)
    assert result.diff == 0.0
    assert (result.ci_low, result.ci_high) == (0.0, 0.0)
    assert result.p_value == 1.0
    assert result.n == 120


def test_paired_bootstrap_detects_a_clearly_better_system():
    rng = np.random.default_rng(2)
    b = (rng.random(300) < 0.5).astype(float)
    a = np.maximum(b, (rng.random(300) < 0.5).astype(float))  # a is b plus extra hits
    result = paired_bootstrap(a, b, n_resamples=2000)
    assert result.diff > 0.1
    assert result.ci_low > 0
    assert result.p_value < 0.01


def test_paired_bootstrap_is_reproducible_and_checks_shapes():
    a, b = np.array([1.0, 0, 1, 1]), np.array([0.0, 0, 1, 0])
    assert paired_bootstrap(a, b, n_resamples=300) == paired_bootstrap(a, b, n_resamples=300)
    with pytest.raises(ValueError, match="same shape"):
        paired_bootstrap([1.0, 0.0], [1.0, 0.0, 1.0])


# --- McNemar -----------------------------------------------------------------------------


def test_mcnemar_exact_known_values():
    # 10 discordant pairs, all favouring one system: p = 2 / 2**10
    result = mcnemar_exact([True] * 10, [False] * 10)
    assert (result.a_only, result.b_only) == (10, 0)
    assert result.p_value == pytest.approx(2 / 1024)
    # 8 vs 2 of 10 discordant: p = 2 * (1 + 10 + 45) / 1024
    a = [True] * 8 + [False] * 2
    b = [False] * 8 + [True] * 2
    assert mcnemar_exact(a, b).p_value == pytest.approx(0.109375)


def test_mcnemar_exact_is_symmetric_and_caps_at_one():
    a = [True] * 6 + [False] * 6
    b = [False] * 6 + [True] * 6
    assert mcnemar_exact(a, b).p_value == 1.0
    assert mcnemar_exact(a, b).p_value == mcnemar_exact(b, a).p_value


def test_mcnemar_exact_ignores_concordant_pairs_and_handles_none_discordant():
    assert mcnemar_exact([True, False, True], [True, False, True]).p_value == 1.0
    assert mcnemar_exact([], []).p_value == 1.0
    padded = mcnemar_exact([True] * 8 + [True] * 50, [False] * 8 + [True] * 50)
    assert padded.p_value == mcnemar_exact([True] * 8, [False] * 8).p_value


def test_mcnemar_exact_handles_large_counts():
    result = mcnemar_exact([True] * 1500 + [False] * 1400, [False] * 1500 + [True] * 1400)
    assert 0.0 < result.p_value <= 1.0
    assert not math.isnan(result.p_value)


def test_mcnemar_exact_checks_lengths():
    with pytest.raises(ValueError):
        mcnemar_exact([True], [True, False])


# --- percentile ----------------------------------------------------------------------------


def test_percentile_nearest_rank_textbook_values():
    data = [15, 20, 35, 40, 50]
    assert percentile(data, 30) == 20
    assert percentile(data, 40) == 20
    assert percentile(data, 50) == 35
    assert percentile(data, 75) == 40
    assert percentile(data, 100) == 50
    assert percentile(data, 0) == 15


def test_percentile_sorts_its_input_and_returns_an_observed_value():
    data = [0.9, 0.1, 0.5, 0.3]
    assert percentile(data, 50) == 0.3
    assert percentile(data, 95) in data


def test_percentile_p95_of_twenty_values_is_the_nineteenth():
    assert percentile(range(1, 21), 95) == 19


def test_percentile_validation():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1, 2], 101)
    with pytest.raises(ValueError):
        percentile([1, 2], -1)
