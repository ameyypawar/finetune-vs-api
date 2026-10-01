"""run_batch: resumable append-only output, clean stops, the spend guard, circuit breakers."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from finetune_vs_api.client import (
    BatchItem,
    BudgetExceeded,
    ConfigMismatch,
    complete_usage,
    is_final,
    latest_rows,
    read_rows,
    run_batch,
)
from finetune_vs_api.cost import Price, Usage
from stubs import FakeTime, Script, chat_response, endpoint, make_client, run, words


def items(n, text="request"):
    return [BatchItem(str(i), [{"role": "user", "content": f"{text} {i}"}]) for i in range(n)]


def batch(client, its, out, **kw):
    kw.setdefault("concurrency", 1)

    async def go():
        async with client:
            return await run_batch(client, its, out, **kw)

    return run(go())


def ids_by_status(path):
    rows = latest_rows(read_rows(path))
    return {
        "ok": sorted(i for i, r in rows.items() if r["error"] is None),
        "failed": sorted(i for i, r in rows.items() if r["error"] is not None),
    }


# --- output ------------------------------------------------------------------------------------


def test_one_row_per_item_with_provenance(tmp_path):
    out = tmp_path / "predictions.jsonl"
    handler = Script(chat_response(prompt=50, completion=9, cached=20, headers={"x-ratelimit-remaining-requests": "7"}))
    client, _ = make_client(handler)
    result = batch(client, items(3), out, config_hash="abc123")
    assert (result.status, result.completed, result.errors, result.remaining) == ("complete", 3, 0, 0)
    rows = read_rows(out)
    assert [r["id"] for r in rows] == ["0", "1", "2"]
    row = rows[0]
    assert row["config_hash"] == "abc123" and row["error"] is None and row["retriable"] is None
    assert row["text"] == '{"intent":"alarm_set","slots":[]}'
    assert row["usage"]["prompt_tokens"] == 50 and row["usage"]["cached_tokens"] == 20
    assert row["model_returned"] == "stub-model-2026-09-01" and row["finish_reason"] == "stop"
    assert row["ratelimit_headers"] == {"x-ratelimit-remaining-requests": "7"}
    assert {"latency_s", "wall_s", "retries", "finished_at"} <= set(row)


def test_each_row_is_flushed_as_it_finishes(tmp_path):
    out = tmp_path / "p.jsonl"
    seen = []

    def progress(done, total, row):
        seen.append((done, total, len(read_rows(out))))

    batch(make_client(Script(chat_response()))[0], items(3), out, progress=progress)
    assert seen == [(1, 3, 1), (2, 3, 2), (3, 3, 3)]  # on disk before the callback fires


# --- resume --------------------------------------------------------------------------------------


def test_an_existing_file_is_never_overwritten_without_resume(tmp_path):
    out = tmp_path / "p.jsonl"
    batch(make_client(Script(chat_response()))[0], items(2), out)
    handler = Script(chat_response())
    with pytest.raises(FileExistsError, match="--resume"):
        batch(make_client(handler)[0], items(2), out)
    assert handler.requests == [] and len(read_rows(out)) == 2


def test_resume_skips_finished_items_and_retries_only_what_can_succeed(tmp_path):
    out = tmp_path / "p.jsonl"
    # 0 ok, 1 ok, 2 fails with a 503 every time (retriable), 3 fails with a 400 (final)
    def handler(request):
        body = json.loads(request.content)["messages"][-1]["content"]
        index = body.rsplit(" ", 1)[1]
        return {"0": chat_response(), "1": chat_response(), "2": httpx.Response(503, text="down"), "3": httpx.Response(400, text="bad")}[index]

    client, _ = make_client(handler, endpoint(max_retries=1))
    result = batch(client, items(4), out, config_hash="h")
    assert (result.completed, result.errors) == (2, 2)
    assert ids_by_status(out) == {"ok": ["0", "1"], "failed": ["2", "3"]}
    rows = {r["id"]: r for r in read_rows(out)}
    assert rows["2"]["retriable"] is True and rows["2"]["error_kind"] == "retries_exhausted"
    assert rows["3"]["retriable"] is False and rows["3"]["error_kind"] == "http_400"

    second = Script(chat_response())
    result = batch(make_client(second)[0], items(4), out, resume=True, config_hash="h")
    assert result.skipped == 3  # 0 and 1 done, 3 failed for good
    assert len(second.requests) == 1  # only item 2 was tried again
    assert ids_by_status(out) == {"ok": ["0", "1", "2"], "failed": ["3"]}
    assert len(read_rows(out)) == 5  # append-only: the failed row for 2 is still there, then the good one
    assert is_final(latest_rows(read_rows(out))["2"])


def test_resume_refuses_rows_from_a_different_configuration(tmp_path):
    out = tmp_path / "p.jsonl"
    batch(make_client(Script(chat_response()))[0], items(2), out, config_hash="aaaaaaaaaaaaaaa")
    handler = Script(chat_response())
    with pytest.raises(ConfigMismatch, match="different configuration"):
        batch(make_client(handler)[0], items(3), out, resume=True, config_hash="bbbbbbbbbbbbbbb")
    assert handler.requests == []


def test_resume_with_nothing_left_sends_nothing(tmp_path):
    out = tmp_path / "p.jsonl"
    batch(make_client(Script(chat_response()))[0], items(2), out)
    handler = Script(chat_response())
    result = batch(make_client(handler)[0], items(2), out, resume=True)
    assert result.status == "complete" and result.skipped == 2 and handler.requests == []


# --- quotas: stop cleanly, resume later ----------------------------------------------------------------


def quota_handler(ok_calls):
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] <= ok_calls:
            return chat_response()
        return httpx.Response(429, headers={"Retry-After": "7200"}, text="daily limit")

    return handler


def test_a_daily_quota_stops_the_run_cleanly_and_resume_finishes_it(tmp_path):
    out = tmp_path / "p.jsonl"
    ft = FakeTime()
    client, _ = make_client(quota_handler(3), ft=ft)
    result = batch(client, items(8), out, config_hash="h")
    assert result.status == "quota_exhausted"
    assert (result.completed, result.errors, result.remaining) == (3, 0, 5)
    assert result.reset_at == pytest.approx(ft.now + 7200)
    assert "resets at" in result.message
    assert ids_by_status(out)["ok"] == ["0", "1", "2"]
    assert len(read_rows(out)) == 3  # the item that hit the quota was not recorded as a failure

    ft.now += 7300
    later = Script(chat_response())
    resumed = batch(make_client(later, ft=ft)[0], items(8), out, resume=True, config_hash="h")
    assert (resumed.status, resumed.skipped, resumed.completed) == ("complete", 3, 5)
    assert ids_by_status(out)["ok"] == [str(i) for i in range(8)]


def test_with_several_workers_a_quota_still_leaves_a_consistent_file(tmp_path):
    out = tmp_path / "p.jsonl"
    client, _ = make_client(quota_handler(5))
    result = batch(client, items(20), out, concurrency=4)
    assert result.status == "quota_exhausted"
    rows = latest_rows(read_rows(out))
    assert len(rows) == result.completed + result.errors == 5
    assert result.completed + result.errors + result.remaining == 20  # nothing lost, nothing double counted
    assert all(r["error"] is None for r in rows.values())


def test_the_limiters_own_daily_cap_stops_the_run_too(tmp_path):
    out = tmp_path / "p.jsonl"
    client, ft = make_client(Script(chat_response()), endpoint(rpd=4), tmp_path=tmp_path / "state")
    result = batch(client, items(10), out)
    assert (result.status, result.completed, result.remaining) == ("quota_exhausted", 4, 6)
    assert result.reset_at == pytest.approx(ft.now + 86_400, abs=5)


# --- spend guard -------------------------------------------------------------------------------------


PRICE = Price(input_per_mtok=1000.0, output_per_mtok=1000.0)  # deliberately expensive: $0.001 per 1k tokens


def test_max_usd_needs_a_price(tmp_path):
    with pytest.raises(ValueError, match="needs a price"):
        batch(make_client(Script(chat_response()))[0], items(2), tmp_path / "p.jsonl", max_usd=1.0)


def test_a_run_projected_over_the_cap_is_refused_before_anything_is_sent(tmp_path):
    out = tmp_path / "p.jsonl"
    handler = Script(chat_response())
    with pytest.raises(BudgetExceeded, match="exceeds --max-usd"):
        batch(make_client(handler)[0], items(50), out, price=PRICE, max_usd=0.0001)
    assert handler.requests == [] and not out.exists()


def test_a_run_whose_real_usage_overshoots_the_projection_is_stopped_midway(tmp_path):
    out = tmp_path / "p.jsonl"
    # The estimate from words is tiny, so the preflight passes; the replies report big usage.
    handler = Script(chat_response(prompt=1000, completion=200))  # $1.20 a call at PRICE
    result = batch(make_client(handler)[0], items(30), out, price=PRICE, max_usd=10.0, warmup=3)
    assert result.status == "budget"
    assert result.completed < 30 and result.remaining == 30 - result.completed
    assert result.spent_usd <= 10.0 + 1.2  # at most one call over
    assert "would pass --max-usd" in result.message
    assert len(read_rows(out)) == result.completed  # what finished is saved


def test_a_run_comfortably_under_the_cap_completes_and_reports_spend(tmp_path):
    handler = Script(chat_response(prompt=100, completion=10))
    result = batch(make_client(handler)[0], items(5), tmp_path / "p.jsonl", price=PRICE, max_usd=5.0)
    assert result.status == "complete"
    assert result.spent_usd == pytest.approx(5 * 110 * 1000.0 / 1e6)


# --- circuit breakers ---------------------------------------------------------------------------------


def test_an_authentication_failure_stops_at_once(tmp_path):
    out = tmp_path / "p.jsonl"
    handler = Script(httpx.Response(401, text="bad token"))
    result = batch(make_client(handler)[0], items(10), out)
    assert result.status == "auth_error" and "API key" in result.message
    assert len(handler.requests) == 1
    assert read_rows(out)[0]["error_kind"] == "http_401" and read_rows(out)[0]["retriable"] is False


def test_a_streak_of_failures_trips_the_breaker(tmp_path):
    handler = Script(httpx.Response(400, text="bad request"))
    result = batch(make_client(handler)[0], items(20), tmp_path / "p.jsonl", max_fail_streak=3)
    assert result.status == "too_many_errors" and result.errors == 3 and len(handler.requests) == 3


def test_a_success_resets_the_failure_streak(tmp_path):
    seq = [httpx.Response(400), httpx.Response(400), chat_response()] * 4
    handler = Script(*seq)
    result = batch(make_client(handler)[0], items(12), tmp_path / "p.jsonl", max_fail_streak=3)
    assert result.status == "complete" and (result.completed, result.errors) == (4, 8)


def test_an_over_long_item_is_recorded_and_skipped_without_being_sent(tmp_path):
    out = tmp_path / "p.jsonl"
    handler = Script(chat_response())
    long = [BatchItem("long", [{"role": "user", "content": "word " * 200}])] + items(2)
    client, _ = make_client(handler, endpoint(max_input_tokens=50))
    result = batch(client, long, out)
    assert (result.completed, result.errors) == (2, 1)
    assert len(handler.requests) == 2
    row = latest_rows(read_rows(out))["long"]
    assert row["error_kind"] == "input_too_long" and row["retriable"] is False and row["text"] is None


# --- usage that the provider did not send -------------------------------------------------------------


def test_missing_usage_is_estimated_flagged_and_still_priced(tmp_path):
    out = tmp_path / "p.jsonl"
    handler = Script(chat_response(text="one two three", usage=False))
    result = batch(make_client(handler)[0], items(2), out, price=PRICE, max_usd=100.0)
    row = read_rows(out)[0]
    assert row["usage"]["reported"] is False and row["usage"]["estimated"] is True
    assert row["usage"]["estimate_method"] == "custom counter"
    assert row["usage"]["completion_tokens"] == 3 and row["usage"]["prompt_tokens"] > 0
    assert result.spent_usd > 0


def test_complete_usage_keeps_what_was_reported_and_fills_the_rest():
    messages = [{"role": "user", "content": "a b c"}]
    whole = Usage(10, 5)
    assert complete_usage(whole, messages, "x", words) is whole
    partial = complete_usage(Usage(prompt_tokens=77, reported=True), messages, "x y", words)
    assert partial.prompt_tokens == 77 and partial.completion_tokens == 2
    assert partial.estimated and partial.reported
    none = complete_usage(Usage(reported=False), messages, None, words)
    assert none.estimated and not none.reported and none.completion_tokens == 0


# --- concurrency -----------------------------------------------------------------------------------------


def test_the_batch_never_exceeds_the_endpoints_concurrency_cap(tmp_path):
    in_flight = peak = 0

    async def handler(request):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.005)
        in_flight -= 1
        return chat_response()

    client, _ = make_client(handler, endpoint(max_concurrency=2))
    result = batch(client, items(12), tmp_path / "p.jsonl", concurrency=8)
    assert result.completed == 12 and peak == 2


def test_the_batch_uses_the_requested_concurrency_when_there_is_no_cap(tmp_path):
    in_flight = peak = 0

    async def handler(request):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.005)
        in_flight -= 1
        return chat_response()

    client, _ = make_client(handler)
    batch(client, items(12), tmp_path / "p.jsonl", concurrency=4)
    assert peak == 4
