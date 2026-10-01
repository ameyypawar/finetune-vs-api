"""The chat client against stubs: parsing, retries, Retry-After, quotas, limiter, caps.

Nothing here reaches a real API. Transports are httpx.MockTransport, plus one test that
talks to a stdlib HTTP server on localhost. Time is faked, so no test sleeps.
"""

from __future__ import annotations

import asyncio
import email.utils
import json
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from finetune_vs_api.client import (
    ApiStatusError,
    ChatClient,
    InputTooLong,
    MalformedResponse,
    MissingApiKey,
    QuotaExhausted,
    RateLimiter,
    RequestTooLarge,
    RetriesExhausted,
    build_request_body,
    parse_duration_s,
    parse_retry_after,
    parse_usage,
    probe_rtt,
    ratelimit_headers,
)
from finetune_vs_api.cost import Usage, count_message_tokens
from stubs import (
    MESSAGES,
    SCHEMA,
    FakeTime,
    Script,
    chat_response,
    endpoint,
    make_client,
    one,
    run,
    words,
)

# --- a successful call ------------------------------------------------------------------------


def test_a_successful_call_is_parsed_in_full():
    handler = Script(
        chat_response(
            prompt=120,
            completion=30,
            cached=80,
            reasoning=12,
            headers={"x-ratelimit-limit-requests": "150", "x-ratelimit-remaining-requests": "149", "x-request-id": "abc"},
        )
    )
    client, _ = make_client(handler)
    out = run(one(client))
    assert out.text == '{"intent":"alarm_set","slots":[]}'
    assert out.usage == Usage(120, 30, cached_tokens=80, reasoning_tokens=12, reported=True)
    assert out.finish_reason == "stop" and out.retries == 0 and out.status_code == 200
    assert out.model_returned == "stub-model-2026-09-01"  # what the server says it ran, not what we asked for
    assert out.ratelimit_headers == {"x-ratelimit-limit-requests": "150", "x-ratelimit-remaining-requests": "149"}
    assert out.latency_s >= 0 and out.wall_s >= out.latency_s


def test_the_request_has_the_right_url_auth_and_body():
    handler = Script(chat_response())
    client, _ = make_client(handler)
    run(one(client))
    request = handler.requests[0]
    assert request.method == "POST" and str(request.url) == "http://stub.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer s3cret"
    assert handler.bodies()[0] == {"model": "stub-model", "messages": MESSAGES, "temperature": 0, "max_tokens": 100}


def test_a_trailing_slash_on_the_base_url_is_harmless():
    handler = Script(chat_response())
    client, _ = make_client(handler, endpoint(base_url="http://stub.test/v1/"))
    run(one(client))
    assert str(handler.requests[0].url) == "http://stub.test/v1/chat/completions"


def test_a_keyless_endpoint_sends_no_authorization_header():
    handler = Script(chat_response())
    client, _ = make_client(handler, endpoint(api_key_env=None))
    run(one(client))
    assert "authorization" not in handler.requests[0].headers


def test_a_missing_key_is_a_clear_error_that_never_prints_a_key():
    handler = Script(chat_response())
    client = ChatClient(endpoint(), transport=httpx.MockTransport(handler), environ={})
    with pytest.raises(MissingApiKey, match="STUB_KEY"):
        run(one(client))
    assert handler.requests == []
    blank = ChatClient(endpoint(), transport=httpx.MockTransport(handler), environ={"STUB_KEY": "  "})
    with pytest.raises(MissingApiKey):
        run(one(blank))


def test_response_format_is_sent_only_when_the_endpoint_supports_it():
    on, off = Script(chat_response()), Script(chat_response())
    run(one(make_client(on, endpoint(supports_json_schema=True))[0], response_format=SCHEMA))
    run(one(make_client(off, endpoint(supports_json_schema=False))[0], response_format=SCHEMA))
    assert on.bodies()[0]["response_format"] == SCHEMA
    assert "response_format" not in off.bodies()[0]


def test_drop_params_removes_what_an_endpoint_rejects():
    body = build_request_body(
        endpoint(drop_params=("temperature", "response_format"), supports_json_schema=True), MESSAGES, response_format=SCHEMA
    )
    assert "temperature" not in body and "response_format" not in body and body["max_tokens"] == 100


def test_extra_params_override_the_configured_ones():
    body = build_request_body(endpoint(), MESSAGES, extra_params={"max_tokens": 5})
    assert body["max_tokens"] == 5


# --- retries ---------------------------------------------------------------------------------------


def test_429_is_retried_and_retry_after_is_respected():
    handler = Script(httpx.Response(429, headers={"Retry-After": "7"}, text="slow down"), chat_response())
    client, ft = make_client(handler)
    out = run(one(client))
    assert out.retries == 1 and len(handler.requests) == 2
    assert ft.slept == [7.0]  # longer than the 1 s backoff, so Retry-After wins


def test_retry_after_as_an_http_date():
    ft = FakeTime()
    when = email.utils.format_datetime(datetime.fromtimestamp(ft.now + 30, UTC), usegmt=True)
    handler = Script(httpx.Response(503, headers={"Retry-After": when}), chat_response())
    client, _ = make_client(handler, ft=ft)
    run(one(client))
    assert ft.slept == [pytest.approx(30.0, abs=1.0)]


def test_5xx_backoff_is_exponential():
    handler = Script(httpx.Response(503), httpx.Response(502), httpx.Response(500), chat_response())
    client, ft = make_client(handler)
    assert run(one(client)).retries == 3
    assert ft.slept == [1.0, 2.0, 4.0]


def test_jitter_scales_the_delay_between_half_and_all_of_the_backoff():
    handler = Script(httpx.Response(503), httpx.Response(503), chat_response())
    client = ChatClient(
        endpoint(), transport=httpx.MockTransport(handler), sleep=(ft := FakeTime()).sleep, clock=ft.clock,
        rng=lambda: 0.0, counter=words, environ={"STUB_KEY": "k"},
    )
    run(one(client))
    assert ft.slept == [0.5, 1.0]


def test_backoff_is_capped():
    handler = Script(*[httpx.Response(503)] * 5, chat_response())
    client, ft = make_client(handler, endpoint(backoff_max_s=3.0, max_retries=5))
    run(one(client))
    assert ft.slept == [1.0, 2.0, 3.0, 3.0, 3.0]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
def test_other_4xx_are_not_retried(status):
    handler = Script(httpx.Response(status, text="nope"))
    client, ft = make_client(handler)
    with pytest.raises(ApiStatusError) as caught:
        run(one(client))
    assert caught.value.status == status and "nope" in caught.value.body
    assert len(handler.requests) == 1 and ft.slept == []


def test_retries_are_bounded():
    handler = Script(httpx.Response(503, text="down"))
    client, ft = make_client(handler, endpoint(max_retries=2))
    with pytest.raises(RetriesExhausted) as caught:
        run(one(client))
    assert caught.value.attempts == 3 and caught.value.last_status == 503
    assert len(handler.requests) == 3 and len(ft.slept) == 2


def test_transport_errors_and_timeouts_are_retried():
    handler = Script(httpx.ConnectError("boom"), httpx.ReadTimeout("slow"), chat_response())
    client, _ = make_client(handler)
    assert run(one(client)).retries == 2


def test_a_malformed_200_is_an_error_not_a_crash():
    for response in (httpx.Response(200, text="not json"), httpx.Response(200, json={"choices": []}), httpx.Response(200, json={"error": "x"})):
        client, _ = make_client(Script(response))
        with pytest.raises(MalformedResponse):
            run(one(client))


# --- quotas ------------------------------------------------------------------------------------------


def test_a_long_retry_after_on_429_is_a_quota_not_a_retry_loop():
    handler = Script(httpx.Response(429, headers={"Retry-After": "3600"}, text="limit"))
    client, ft = make_client(handler)
    with pytest.raises(QuotaExhausted) as caught:
        run(one(client))
    assert caught.value.reset_at == pytest.approx(ft.now + 3600)
    assert len(handler.requests) == 1 and ft.slept == []
    assert "resets at" in str(caught.value)


def test_once_blocked_the_client_does_not_ask_again_before_the_reset():
    ft = FakeTime()
    handler = Script(httpx.Response(429, headers={"Retry-After": "3600"}), chat_response())
    client, _ = make_client(handler, ft=ft)

    async def scenario():
        async with client:
            with pytest.raises(QuotaExhausted):
                await client.complete(MESSAGES)
            with pytest.raises(QuotaExhausted):
                await client.complete(MESSAGES)  # no second request
            assert len(handler.requests) == 1
            ft.now += 3601
            return await client.complete(MESSAGES)  # reset passed: sends again

    assert run(scenario()).text
    assert len(handler.requests) == 2


def test_a_daily_quota_survives_a_restart_when_the_endpoint_has_day_limits(tmp_path):
    ft = FakeTime()
    ep = endpoint(rpd=100)
    first, _ = make_client(Script(httpx.Response(429, headers={"Retry-After": "36000"})), ep, ft, tmp_path)
    with pytest.raises(QuotaExhausted):
        run(one(first))
    handler = Script(chat_response())
    restarted, _ = make_client(handler, ep, ft, tmp_path)  # a new process: new limiter, same state dir
    with pytest.raises(QuotaExhausted):
        run(one(restarted))
    assert handler.requests == []
    ft.now += 36001
    assert run(one(make_client(handler, ep, ft, tmp_path)[0])).text
    assert len(handler.requests) == 1


def test_a_daily_429_with_no_retry_after_is_a_quota_with_an_unknown_reset():
    body = {"error": {"message": "Rate limit reached for requests per day (RPD): Limit 1000"}}
    client, ft = make_client(Script(httpx.Response(429, json=body)))
    with pytest.raises(QuotaExhausted) as caught:
        run(one(client))
    assert caught.value.reset_at is None and "unknown" in str(caught.value)
    assert ft.slept == []


def test_a_plain_429_with_no_hints_is_just_backed_off_and_retried():
    handler = Script(httpx.Response(429, text="slow down"), chat_response())
    client, ft = make_client(handler)
    assert run(one(client)).retries == 1 and ft.slept == [1.0]


def test_x_ratelimit_reset_headers_are_used_when_there_is_no_retry_after():
    long = httpx.Response(429, headers={"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "7m12s"})
    client, ft = make_client(Script(long))
    with pytest.raises(QuotaExhausted) as caught:
        run(one(client))
    assert caught.value.reset_at == pytest.approx(ft.now + 432)

    short = httpx.Response(429, headers={"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "2.5s"})
    client, ft = make_client(Script(short, chat_response()))
    assert run(one(client)).retries == 1
    assert ft.slept == [2.5]


# --- header parsing --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("raw", "expected"), [("5", 5.0), ("0", 0.0), ("2.5", 2.5), ("-3", 0.0), (None, None), ("", None), ("soon", None)])
def test_parse_retry_after_seconds(raw, expected):
    assert parse_retry_after(raw, now=0.0) == expected


def test_parse_retry_after_http_date_is_relative_to_now():
    when = email.utils.format_datetime(datetime(2026, 10, 1, 12, 0, 30, tzinfo=UTC), usegmt=True)
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC).timestamp()
    assert parse_retry_after(when, now) == pytest.approx(30.0)
    assert parse_retry_after(when, now + 100) == 0.0  # already past


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1m30.5s", 90.5), ("250ms", 0.25), ("2h", 7200.0), ("1h2m3s", 3723.0), ("17", 17.0), ("6m0s", 360.0), ("garbage", None), ("5x", None), ("", None), (None, None)],
)
def test_parse_duration(raw, expected):
    assert parse_duration_s(raw, now=0.0) == expected


def test_parse_duration_reads_a_big_number_as_an_epoch_reset_time():
    assert parse_duration_s("1800000100", now=1_800_000_000.0) == pytest.approx(100.0)


def test_ratelimit_headers_keeps_only_ratelimit_and_retry_after():
    got = ratelimit_headers({"X-RateLimit-Limit": "10", "Retry-After": "3", "Content-Type": "x", "x-ratelimit-reset": "9"})
    assert got == {"x-ratelimit-limit": "10", "retry-after": "3", "x-ratelimit-reset": "9"}


# --- usage parsing --------------------------------------------------------------------------------------


def test_missing_usage_is_unreported_not_an_error():
    out = run(one(make_client(Script(chat_response(usage=False)))[0]))
    assert out.usage == Usage(reported=False) and out.text


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"usage": None}, Usage(reported=False)),
        ({}, Usage(reported=False)),
        ({"usage": {}}, Usage(reported=True)),
        ({"usage": {"prompt_tokens": 7}}, Usage(7, None, reported=True)),
        ({"usage": {"prompt_tokens": "12", "completion_tokens": 3.0}}, Usage(12, 3, reported=True)),
        ({"usage": {"prompt_tokens": True, "completion_tokens": 3}}, Usage(None, 3, reported=True)),
        ({"usage": {"prompt_tokens": 10, "completion_tokens": 2, "prompt_cache_hit_tokens": 6}}, Usage(10, 2, cached_tokens=6, reported=True)),
        ({"usage": {"prompt_tokens": 10, "completion_tokens": 2, "prompt_tokens_details": None}}, Usage(10, 2, reported=True)),
        ({"usage": {"prompt_tokens": 10, "completion_tokens": 2, "completion_tokens_details": {"reasoning_tokens": None}}}, Usage(10, 2, reported=True)),
    ],
)
def test_parse_usage_tolerates_missing_and_odd_fields(payload, expected):
    assert parse_usage(payload) == expected


def test_reasoning_tokens_are_part_of_completion_tokens_by_default():
    usage = parse_usage({"usage": {"prompt_tokens": 100, "completion_tokens": 900, "completion_tokens_details": {"reasoning_tokens": 850}}})
    assert (usage.completion_tokens, usage.reasoning_tokens) == (900, 850)


def test_a_provider_that_reports_reasoning_separately_is_normalized():
    payload = {"usage": {"prompt_tokens": 100, "completion_tokens": 50, "completion_tokens_details": {"reasoning_tokens": 850}}}
    assert parse_usage(payload, reasoning_in_completion=False).completion_tokens == 900


def test_empty_content_from_a_reasoning_model_and_content_parts():
    empty = run(one(make_client(Script(chat_response(text=None, finish="length")))[0]))
    assert empty.text is None and empty.finish_reason == "length"
    parts = httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}, "finish_reason": "stop"}]})
    assert run(one(make_client(Script(parts))[0])).text == "ab"


# --- input length -----------------------------------------------------------------------------------------


def test_a_request_over_max_input_tokens_is_never_sent():
    need = count_message_tokens(MESSAGES, words)
    handler = Script(chat_response())
    with pytest.raises(InputTooLong) as caught:
        run(one(make_client(handler, endpoint(max_input_tokens=need - 1))[0]))
    assert caught.value.estimated == need and caught.value.limit == need - 1
    assert handler.requests == []
    assert run(one(make_client(handler, endpoint(max_input_tokens=need))[0])).text  # exactly at the limit is fine


# --- the limiter -----------------------------------------------------------------------------------------


def test_requests_per_minute_are_paced():
    handler = Script(chat_response())
    client, ft = make_client(handler, endpoint(rpm=2))

    async def scenario():
        async with client:
            for _ in range(3):
                await client.complete(MESSAGES)

    run(scenario())
    assert len(handler.requests) == 3
    assert len(ft.slept) == 1 and ft.slept[0] == pytest.approx(60.0, abs=0.01)  # the third waits for the window


def test_requests_per_day_raise_with_the_time_the_oldest_ages_out_and_persist(tmp_path):
    ft = FakeTime()
    ep = endpoint(rpd=2)
    start = ft.now
    handler = Script(chat_response())
    client, _ = make_client(handler, ep, ft, tmp_path)

    async def scenario():
        async with client:
            await client.complete(MESSAGES)
            await client.complete(MESSAGES)
            with pytest.raises(QuotaExhausted) as caught:
                await client.complete(MESSAGES)
            return caught.value

    exc = run(scenario())
    assert exc.reset_at == pytest.approx(start + 86_400)
    assert len(handler.requests) == 2

    restarted, _ = make_client(handler, ep, ft, tmp_path)
    with pytest.raises(QuotaExhausted):
        run(one(restarted))
    assert len(handler.requests) == 2  # the count survived the restart
    ft.now = start + 86_401
    assert run(one(make_client(handler, ep, ft, tmp_path)[0])).text


def test_reasoning_tokens_count_against_the_per_minute_token_budget():
    # The first reply used 950 tokens, 800 of them reasoning. With a 1000-token budget the
    # second request (reserving about 100) has to wait for that usage to age out.
    handler = Script(chat_response(prompt=100, completion=850, reasoning=800), chat_response())
    client, ft = make_client(handler, endpoint(tpm=1000))

    async def scenario():
        async with client:
            await client.complete(MESSAGES)
            await client.complete(MESSAGES)

    run(scenario())
    assert len(ft.slept) == 1 and ft.slept[0] == pytest.approx(60.0, abs=0.01)


def test_reasoning_reported_separately_counts_too():
    handler = Script(chat_response(prompt=100, completion=50, reasoning=800), chat_response())
    client, ft = make_client(handler, endpoint(tpm=1000, reasoning_in_completion=False))

    async def scenario():
        async with client:
            await client.complete(MESSAGES)
            await client.complete(MESSAGES)

    run(scenario())
    assert len(ft.slept) == 1


def test_a_small_reply_does_not_make_the_next_request_wait():
    # the reservation (about 100 for the reply) is replaced by the real count (20)
    handler = Script(chat_response(prompt=100, completion=20))
    client, ft = make_client(handler, endpoint(tpm=1000))

    async def scenario():
        async with client:
            for _ in range(5):
                await client.complete(MESSAGES)

    run(scenario())
    assert ft.slept == []


def test_tokens_per_day_raise_quota_exhausted_with_a_reset_time(tmp_path):
    ft = FakeTime()
    start = ft.now
    handler = Script(chat_response(prompt=100, completion=300))
    client, _ = make_client(handler, endpoint(tpd=900), ft, tmp_path)

    async def scenario():
        async with client:
            await client.complete(MESSAGES)  # 400 used
            await client.complete(MESSAGES)  # 800 used
            with pytest.raises(QuotaExhausted) as caught:
                await client.complete(MESSAGES)  # would pass 900
            return caught.value

    exc = run(scenario())
    assert exc.reset_at == pytest.approx(start + 86_400)
    assert "tokens per day" in exc.reason


def test_a_request_that_can_never_fit_a_token_budget_is_an_error():
    client, _ = make_client(Script(chat_response()), endpoint(tpm=10))
    with pytest.raises(RequestTooLarge):
        run(one(client))


def test_the_limiter_unit_without_a_client(tmp_path):
    ft = FakeTime()
    limiter = RateLimiter(rpm=1, clock=ft.clock, sleep=ft.sleep)

    async def scenario():
        await limiter.acquire()
        await limiter.acquire()

    run(scenario())
    assert ft.slept == [pytest.approx(60.001)]
    assert limiter.usage_today() == (2, 0)


def test_a_restart_straight_after_a_stop_still_respects_the_per_minute_cap(tmp_path):
    ft = FakeTime()
    state = tmp_path / "state.json"

    async def two_requests(limiter):
        await limiter.acquire()
        await limiter.acquire()

    run(two_requests(RateLimiter(rpm=2, rpd=100, state_path=state, clock=ft.clock, sleep=ft.sleep)))
    assert ft.slept == []
    restarted = RateLimiter(rpm=2, rpd=100, state_path=state, clock=ft.clock, sleep=ft.sleep)
    run(restarted.acquire())  # a third request inside the same minute: it has to wait
    assert len(ft.slept) == 1 and ft.slept[0] == pytest.approx(60.0, abs=0.01)


def test_an_unreadable_state_file_is_ignored(tmp_path):
    state = tmp_path / "state.json"
    state.write_text("{not json")
    limiter = RateLimiter(rpd=5, state_path=state, clock=FakeTime().clock)
    assert limiter.usage_today() == (0, 0)


# --- concurrency -----------------------------------------------------------------------------------------


def test_the_concurrency_cap_is_enforced_per_endpoint():
    in_flight = peak = 0

    async def handler(request):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return chat_response()

    client, _ = make_client(handler, endpoint(max_concurrency=2))

    async def scenario():
        async with client:
            await asyncio.gather(*(client.complete(MESSAGES) for _ in range(8)))

    run(scenario())
    assert peak == 2


def test_without_a_cap_calls_run_concurrently():
    in_flight = peak = 0

    async def handler(request):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return chat_response()

    client, _ = make_client(handler)

    async def scenario():
        async with client:
            await asyncio.gather(*(client.complete(MESSAGES) for _ in range(6)))

    run(scenario())
    assert peak == 6


# --- probe_rtt -----------------------------------------------------------------------------------------------


def test_probe_rtt_measures_get_models_and_records_failures():
    handler = Script(httpx.Response(200, json={"data": []}), httpx.Response(200, json={}), httpx.Response(500), httpx.ConnectError("down"))

    out = run(probe_rtt(endpoint(), n=4, transport=httpx.MockTransport(handler), environ={"STUB_KEY": "k"}))
    assert out["url"] == "http://stub.test/v1/models" and out["n"] == 4
    assert out["statuses"] == [200, 200, 500, "ConnectError"]
    assert out["ok"] == 3 and out["min_s"] <= out["median_s"] <= out["max_s"]
    assert handler.requests[0].headers["authorization"] == "Bearer k"


def test_probe_rtt_with_everything_failing():
    out = run(probe_rtt(endpoint(api_key_env=None), n=2, transport=httpx.MockTransport(Script(httpx.ConnectError("x")))))
    assert out["ok"] == 0 and out["median_s"] is None


# --- over a real socket ------------------------------------------------------------------------------------


class StubServer(BaseHTTPRequestHandler):
    seen: list[dict] = []
    calls = 0

    def log_message(self, *args):  # keep test output quiet
        pass

    def _send(self, status, payload, headers=None):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - http.server naming
        self._send(200, {"data": [{"id": "stub-model"}]})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        StubServer.seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": json.loads(self.rfile.read(length))})
        StubServer.calls += 1
        if StubServer.calls == 1:
            return self._send(429, {"error": "slow down"}, {"Retry-After": "0"})
        self._send(
            200,
            {"model": "stub-model-live", "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 9, "completion_tokens": 2}},
            {"x-ratelimit-remaining-requests": "41"},
        )


def test_the_client_works_over_a_real_local_socket():
    StubServer.seen, StubServer.calls = [], 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        ft = FakeTime()
        ep = endpoint(base_url=f"http://127.0.0.1:{port}/v1")
        client = ChatClient(ep, sleep=ft.sleep, clock=ft.clock, rng=lambda: 1.0, counter=words, environ={"STUB_KEY": "k"})
        out = run(one(client))
        assert out.retries == 1 and out.text == "{}" and out.model_returned == "stub-model-live"
        assert out.ratelimit_headers == {"x-ratelimit-remaining-requests": "41"}
        assert [s["path"] for s in StubServer.seen] == ["/v1/chat/completions"] * 2
        assert StubServer.seen[0]["auth"] == "Bearer k" and StubServer.seen[0]["body"]["model"] == "stub-model"
        rtt = run(probe_rtt(ep, n=2, environ={"STUB_KEY": "k"}))
        assert rtt["ok"] == 2 and rtt["statuses"] == [200, 200]
    finally:
        server.shutdown()
        server.server_close()
