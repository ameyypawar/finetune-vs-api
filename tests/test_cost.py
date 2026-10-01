"""Cost arithmetic: cached tokens, reasoning tokens, bounds, break-even, token estimates."""

from __future__ import annotations

import math

import pytest

from finetune_vs_api import cost
from finetune_vs_api.cost import (
    BILLING_BASIS,
    Price,
    Usage,
    api_call_cost,
    api_cost_bounds,
    breakeven_calls_per_month,
    cost_summary,
    count_message_tokens,
    estimate_usage,
    per_1k,
    selfhost_per_1k,
)

# gpt-4.1 list prices, USD per million tokens
GPT41 = Price(input_per_mtok=2.00, output_per_mtok=8.00, cached_input_per_mtok=0.50)
MINI = Price(input_per_mtok=0.40, output_per_mtok=1.60, cached_input_per_mtok=0.10)


# --- one call --------------------------------------------------------------------------------


def test_cached_tokens_are_priced_at_the_cached_rate():
    usage = Usage(prompt_tokens=1000, cached_tokens=400, completion_tokens=200)
    # 600 uncached * 2.00 + 400 cached * 0.50 + 200 output * 8.00, per million
    assert api_call_cost(usage, GPT41) == pytest.approx((600 * 2.00 + 400 * 0.50 + 200 * 8.00) / 1e6)
    assert api_call_cost(usage, GPT41) == pytest.approx(0.003)


def test_no_cached_tokens_means_everything_is_uncached():
    assert api_call_cost(Usage(prompt_tokens=1000, completion_tokens=200), GPT41) == pytest.approx(0.0036)
    assert api_call_cost(Usage(1000, 200, cached_tokens=0), GPT41) == pytest.approx(0.0036)


def test_cached_tokens_cannot_exceed_the_prompt():
    usage = Usage(prompt_tokens=100, cached_tokens=900, completion_tokens=0)
    assert api_call_cost(usage, GPT41) == pytest.approx(100 * 0.50 / 1e6)


def test_a_provider_without_a_cached_rate_bills_cached_tokens_as_input():
    flat = Price(input_per_mtok=1.0, output_per_mtok=2.0)
    usage = Usage(prompt_tokens=1000, cached_tokens=600, completion_tokens=100)
    assert api_call_cost(usage, flat) == pytest.approx((1000 * 1.0 + 100 * 2.0) / 1e6)


def test_reasoning_tokens_are_part_of_the_output_and_not_counted_twice():
    # completion_tokens already includes the 450 reasoning tokens
    usage = Usage(prompt_tokens=100, completion_tokens=500, reasoning_tokens=450)
    assert api_call_cost(usage, MINI) == pytest.approx((100 * 0.40 + 500 * 1.60) / 1e6)


def test_zero_tokens_cost_nothing():
    assert api_call_cost(Usage(0, 0), GPT41) == 0.0


def test_a_call_without_token_counts_cannot_be_priced():
    with pytest.raises(ValueError, match="estimate"):
        api_call_cost(Usage(reported=False), GPT41)
    with pytest.raises(ValueError):
        api_call_cost(Usage(prompt_tokens=10), GPT41)


# --- bounds ----------------------------------------------------------------------------------


def test_bounds_price_the_prefix_at_the_cached_rate_for_the_lower_bound():
    usage = Usage(prompt_tokens=2000, completion_tokens=100)
    bounds = api_cost_bounds(usage, MINI, cacheable_prefix_tokens=1500)
    assert bounds.upper == pytest.approx((2000 * 0.40 + 100 * 1.60) / 1e6)  # no caching
    assert bounds.lower == pytest.approx((500 * 0.40 + 1500 * 0.10 + 100 * 1.60) / 1e6)
    assert bounds.lower < bounds.upper


def test_bounds_are_equal_without_a_prefix_or_without_a_cached_discount():
    usage = Usage(prompt_tokens=2000, completion_tokens=100)
    assert api_cost_bounds(usage, MINI, 0).lower == api_cost_bounds(usage, MINI, 0).upper
    flat = Price(1.0, 2.0)
    assert api_cost_bounds(usage, flat, 1500).lower == api_cost_bounds(usage, flat, 1500).upper


def test_a_prefix_longer_than_the_prompt_is_capped():
    usage = Usage(prompt_tokens=100, completion_tokens=0)
    capped = api_cost_bounds(usage, MINI, cacheable_prefix_tokens=10_000)
    assert capped.lower == pytest.approx(100 * 0.10 / 1e6)


def test_bounds_do_not_depend_on_what_the_provider_reported_as_cached():
    base = Usage(prompt_tokens=2000, completion_tokens=100)
    reported = Usage(prompt_tokens=2000, completion_tokens=100, cached_tokens=1900)
    assert api_cost_bounds(base, MINI, 1500) == api_cost_bounds(reported, MINI, 1500)


def test_bounds_fields_are_named_upper_then_lower():
    assert api_cost_bounds(Usage(1000, 10), MINI, 500)._fields == ("upper", "lower")


# --- per 1,000 / self-host / break-even -------------------------------------------------------------


def test_per_1k():
    assert per_1k(0.5, 250) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        per_1k(1.0, 0)


def test_selfhost_per_1k_at_full_utilization():
    # $0.526/h at 2 requests/s: 7200 requests an hour
    assert selfhost_per_1k(0.526, 2.0) == pytest.approx(0.526 / 7200 * 1000)
    assert selfhost_per_1k(3600, 1.0) == pytest.approx(1000.0)
    with pytest.raises(ValueError):
        selfhost_per_1k(0.5, 0)


def test_breakeven_calls_per_month():
    assert breakeven_calls_per_month(0.526, 0.0005) == pytest.approx(0.526 * 730 / 0.0005)
    assert breakeven_calls_per_month(1.0, 0.001, hours=100) == pytest.approx(100_000)


def test_breakeven_with_a_free_api_is_never():
    assert breakeven_calls_per_month(0.526, 0.0) == math.inf


def test_breakeven_rejects_negative_costs():
    with pytest.raises(ValueError):
        breakeven_calls_per_month(0.5, -0.001)


# --- Price and Usage plumbing -------------------------------------------------------------------


def test_price_from_a_sources_entry():
    entry = {"usd_per_mtok": {"input": 0.40, "cached_input": 0.10, "output": 1.60}}
    assert Price.from_entry(entry) == MINI
    assert Price.from_entry({"usd_per_mtok": {"input": 1, "output": 2}}).cached == 1


def test_usage_round_trips_and_a_missing_object_means_unreported():
    usage = Usage(10, 5, cached_tokens=4, reasoning_tokens=3, estimated=True, estimate_method="m")
    assert Usage.from_dict(usage.to_dict()) == usage
    assert Usage.from_dict(None) == Usage(reported=False)
    assert Usage.from_dict({}) == Usage(reported=False)
    assert Usage.from_dict({"prompt_tokens": 7, "completion_tokens": 1}).total_tokens == 8


# --- local token estimates ---------------------------------------------------------------------


def words(text: str) -> int:
    return len(text.split())


def test_estimate_usage_is_flagged_and_counts_framing():
    messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "wake me at six"}]
    usage = estimate_usage(messages, "ok sure", counter=words, method="words")
    assert usage.estimated and not usage.reported
    assert usage.estimate_method == "words"
    # (2 + 3) + (4 + 3) per message, + 3 for priming the reply
    assert usage.prompt_tokens == 5 + 7 + 3 == count_message_tokens(messages, words)
    assert usage.completion_tokens == 2


def test_estimate_usage_with_no_output():
    usage = estimate_usage([{"role": "user", "content": "hi"}], None, counter=words)
    assert usage.completion_tokens == 0
    assert usage.estimate_method == "custom counter"
    assert usage.is_complete and api_call_cost(usage, MINI) > 0


def test_default_counter_uses_tiktoken_o200k_when_it_loads(monkeypatch):
    class FakeEncoding:
        def encode(self, text, disallowed_special=()):
            return text.split()

    seen = {}
    import tiktoken

    def fake_get(name):
        seen["name"] = name
        return FakeEncoding()

    monkeypatch.setattr(tiktoken, "get_encoding", fake_get)
    cost.default_counter.cache_clear()
    counter, method = cost.default_counter()
    assert seen["name"] == "o200k_base" and method == "tiktoken:o200k_base"
    assert counter("one two three") == 3
    cost.default_counter.cache_clear()


def test_default_counter_falls_back_and_says_so_when_tiktoken_cannot_load(monkeypatch):
    import tiktoken

    def broken(name):
        raise OSError("offline")

    monkeypatch.setattr(tiktoken, "get_encoding", broken)
    cost.default_counter.cache_clear()
    counter, method = cost.default_counter()
    assert "chars/4" in method
    assert counter("a" * 40) == 10 and counter("") == 0
    usage = estimate_usage([{"role": "user", "content": "a" * 40}], "b" * 8)
    assert usage.estimate_method == method and usage.completion_tokens == 2
    cost.default_counter.cache_clear()


# --- a whole run ----------------------------------------------------------------------------------


def test_cost_summary_totals_bounds_and_provenance():
    usages = [
        Usage(2000, 100),
        Usage(1000, 50, cached_tokens=0),
        estimate_usage([{"role": "user", "content": "x"}], "y", counter=words),
    ]
    summary = cost_summary(usages, MINI, cacheable_prefix_tokens=500)
    assert summary["billing_basis"] == BILLING_BASIS == "free tier; priced at paid list price"
    assert summary["calls"] == 3
    assert summary["calls_with_reported_usage"] == 2
    assert summary["calls_with_estimated_usage"] == 1
    assert summary["total_usd"]["lower"] < summary["total_usd"]["upper"]
    assert summary["per_1k_calls_usd"]["upper"] == pytest.approx(summary["total_usd"]["upper"] / 3 * 1000)
    assert summary["prompt_tokens"] == sum(u.prompt_tokens for u in usages)


def test_cost_summary_needs_calls():
    with pytest.raises(ValueError):
        cost_summary([], MINI, 0)
