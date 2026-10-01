"""An async Chat Completions client for any OpenAI-compatible server, built for free tiers.

Free tiers cap requests (and sometimes tokens) per minute and per day, so this client does
more than send requests:

* A per-endpoint limiter paces requests and token budgets per minute, and counts the
  per-day budgets in a small state file so they survive a restart. Reasoning tokens count
  against the token budgets, because they are part of `completion_tokens`.
* A per-endpoint concurrency cap, and a `max_input_tokens` check that refuses to send a
  request estimated to be over the endpoint's input limit.
* Exponential backoff with jitter on 429 and 5xx that respects `Retry-After`; no retry on
  any other 4xx.
* When a daily quota runs out, whether the limiter notices or the server says so with a
  long `Retry-After`, the client raises `QuotaExhausted` carrying the reset time instead
  of sleeping or retrying in a loop. `run_batch` turns that into a clean stop: progress is
  already on disk, and a later `--resume` carries on.

Nothing here knows about the task; `evaluate.py` builds the prompts and scores the output.
"""

from __future__ import annotations

import asyncio
import email.utils
import json
import math
import os
import random
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from . import config
from .cost import (
    Counter,
    Price,
    Usage,
    api_cost_bounds,
    count_message_tokens,
    estimate_usage,
)

DAY_S = 86_400.0
MINUTE_S = 60.0
#: Words in a 429 body that say the limit hit was a daily one.
_DAILY_HINT = re.compile(r"per[\s_-]*day|daily|\bRPD\b|\bTPD\b", re.IGNORECASE)

# --- errors ----------------------------------------------------------------------------------


class ClientError(Exception):
    """Base class for everything this module raises on purpose."""


class MissingApiKey(ClientError):
    def __init__(self, env_var: str):
        super().__init__(f"environment variable {env_var} is not set (see .env.example)")
        self.env_var = env_var


class InputTooLong(ClientError):
    def __init__(self, estimated: int, limit: int):
        super().__init__(f"request estimated at {estimated} input tokens, over the limit of {limit}")
        self.estimated, self.limit = estimated, limit


class RequestTooLarge(ClientError):
    """A single request can never fit inside a token budget."""


class ApiStatusError(ClientError):
    """A non-retriable HTTP error (a 4xx other than 429)."""

    def __init__(self, status: int, body: str, headers: Mapping[str, str] | None = None):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status, self.body, self.headers = status, body, dict(headers or {})


class MalformedResponse(ClientError):
    """A 200 whose body is not a usable chat completion."""


class RetriesExhausted(ClientError):
    def __init__(self, attempts: int, last_status: int | None, last_error: str, headers: Mapping[str, str]):
        super().__init__(f"gave up after {attempts} attempts; last: {last_status or 'transport error'} {last_error[:200]}")
        self.attempts, self.last_status, self.last_error = attempts, last_status, last_error
        self.headers = dict(headers)


class QuotaExhausted(ClientError):
    """A daily (or otherwise long) quota is used up. `reset_at` is an epoch time, or None if unknown."""

    def __init__(self, reset_at: float | None, reason: str, endpoint: str = ""):
        when = f"resets at {format_time(reset_at)}" if reset_at else "reset time unknown"
        super().__init__(f"quota exhausted for {endpoint or 'endpoint'}: {reason}; {when}")
        self.reset_at, self.reason, self.endpoint = reset_at, reason, endpoint


class BudgetExceeded(ClientError):
    pass


class ConfigMismatch(ClientError):
    pass


def format_time(epoch: float | None) -> str:
    if epoch is None:
        return "unknown"
    return datetime.fromtimestamp(epoch, UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


# --- header parsing ----------------------------------------------------------------------------

_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h|d)")
_UNIT_S = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def parse_duration_s(value: str | None, now: float | None = None) -> float | None:
    """Seconds from a duration such as "1m30.5s" or "250ms", a plain number of seconds, or an
    epoch time (a number above 1e9, read as a reset instant). None if it is not understood."""
    if value is None:
        return None
    text = value.strip().lower()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        parts = _DURATION_PART.findall(text)
        if not parts or "".join(n + u for n, u in parts) != text.replace(" ", ""):
            return None
        return sum(float(n) * _UNIT_S[u] for n, u in parts)
    if number > 1e9:
        return max(0.0, number - (time.time() if now is None else now))
    return max(0.0, number)


def parse_retry_after(value: str | None, now: float | None = None) -> float | None:
    """`Retry-After` as seconds from now: delta-seconds or an HTTP-date. None if absent or unreadable."""
    if value is None or not value.strip():
        return None
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value.strip())
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - (time.time() if now is None else now))


def ratelimit_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """The `x-ratelimit-*` headers and `retry-after`, lower-cased, exactly as the server sent them."""
    return {k.lower(): v for k, v in headers.items() if k.lower().startswith("x-ratelimit") or k.lower() == "retry-after"}


def reset_after_s(headers: Mapping[str, str], now: float) -> float | None:
    """How long until the server says a limit resets: Retry-After, else the longest
    `x-ratelimit-reset*` value among limits whose `remaining` is zero (or all, if none say)."""
    direct = parse_retry_after(headers.get("retry-after"), now)
    if direct is not None:
        return direct
    candidates: list[float] = []
    for key, value in headers.items():
        if key.startswith("x-ratelimit-reset"):
            suffix = key[len("x-ratelimit-reset") :]
            remaining = headers.get(f"x-ratelimit-remaining{suffix}")
            seconds = parse_duration_s(value, now)
            if seconds is not None and (remaining in (None, "0", "0.0")):
                candidates.append(seconds)
    return max(candidates) if candidates else None


def _looks_like_daily_quota(body: str) -> bool:
    return bool(_DAILY_HINT.search(body or ""))


# --- endpoint ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Endpoint:
    """Everything needed to talk to one model on one OpenAI-compatible server."""

    name: str
    base_url: str
    model: str
    api_key_env: str | None = None
    supports_json_schema: bool = False
    params: Mapping[str, Any] = field(default_factory=dict)
    drop_params: tuple[str, ...] = ()
    reasoning_in_completion: bool = True
    headers: Mapping[str, str] = field(default_factory=dict)
    rpm: int | None = None
    rpd: int | None = None
    tpm: int | None = None
    tpd: int | None = None
    max_concurrency: int | None = None
    max_input_tokens: int | None = None
    timeout_s: float = 120.0
    max_retries: int = 5
    backoff_base_s: float = 1.0
    backoff_max_s: float = 60.0
    #: A Retry-After longer than this is treated as a quota running out, not a blip.
    quota_wait_threshold_s: float = 120.0
    catalog_url: str | None = None

    @property
    def limit_group(self) -> str:
        return f"{self.name}/{self.model}"

    @property
    def max_output_tokens(self) -> int:
        return int(self.params.get("max_completion_tokens") or self.params.get("max_tokens") or 256)

    @classmethod
    def from_system(cls, spec: Mapping[str, Any]) -> Endpoint:
        """Build from `config.resolve_system()` output."""
        limits = spec["limits"]
        return cls(
            name=spec["endpoint"],
            base_url=spec["base_url"],
            model=spec["model"],
            api_key_env=spec["api_key_env"],
            supports_json_schema=spec["supports_json_schema"],
            params=dict(spec["params"]),
            drop_params=tuple(spec["drop_params"]),
            reasoning_in_completion=spec["reasoning_in_completion"],
            rpm=limits.get("rpm"),
            rpd=limits.get("rpd"),
            tpm=limits.get("tpm"),
            tpd=limits.get("tpd"),
            max_concurrency=limits.get("max_concurrency"),
            max_input_tokens=limits.get("max_input_tokens"),
            catalog_url=spec.get("catalog_url"),
        )

    def api_key(self, environ: Mapping[str, str] | None = None) -> str | None:
        """The key from the environment, None for a keyless endpoint, or MissingApiKey."""
        if not self.api_key_env:
            return None
        value = (os.environ if environ is None else environ).get(self.api_key_env, "").strip()
        if not value:
            raise MissingApiKey(self.api_key_env)
        return value


def build_request_body(
    endpoint: Endpoint,
    messages: Sequence[Mapping[str, str]],
    *,
    response_format: Mapping[str, Any] | None = None,
    extra_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The JSON body for one request: model, messages, configured params, optional schema.

    `response_format` is only sent when the endpoint supports it. Anything in `drop_params`
    is removed last, for endpoints that reject a parameter they would otherwise be sent.
    """
    body: dict[str, Any] = {"model": endpoint.model, "messages": list(messages), **endpoint.params}
    if extra_params:
        body.update(extra_params)
    if response_format is not None and endpoint.supports_json_schema:
        body["response_format"] = response_format
    for name in endpoint.drop_params:
        body.pop(name, None)
    return body


# --- usage and response parsing --------------------------------------------------------------


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_usage(payload: Mapping[str, Any], reasoning_in_completion: bool = True) -> Usage:
    """Normalize a response's `usage` into `Usage`, tolerating anything missing.

    `completion_tokens` in the result always includes reasoning tokens: providers that
    report them separately (`reasoning_in_completion=False`) have them added.
    """
    raw = payload.get("usage")
    if not isinstance(raw, Mapping):
        return Usage(reported=False)
    prompt_details = raw.get("prompt_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or {}
    cached = (
        _int(prompt_details.get("cached_tokens")) if isinstance(prompt_details, Mapping) else None
    )
    if cached is None:
        cached = _int(raw.get("prompt_cache_hit_tokens"))
    if cached is None:
        cached = _int(raw.get("cached_tokens"))
    reasoning = _int(completion_details.get("reasoning_tokens")) if isinstance(completion_details, Mapping) else None
    completion = _int(raw.get("completion_tokens"))
    if completion is not None and reasoning and not reasoning_in_completion:
        completion += reasoning
    return Usage(
        prompt_tokens=_int(raw.get("prompt_tokens")),
        completion_tokens=completion,
        cached_tokens=cached,
        reasoning_tokens=reasoning,
        reported=True,
    )


def _message_text(content: Any) -> str | None:
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):  # some providers return content parts
        return "".join(p.get("text", "") for p in content if isinstance(p, Mapping))
    return None


def complete_usage(
    usage: Usage, messages: Sequence[Mapping[str, str]], text: str | None, counter: Counter | None = None
) -> Usage:
    """`usage` with any missing token count filled in from a local estimate, flagged `estimated`.

    Counts the provider did report are kept as they are. A usage that is already complete is
    returned unchanged.
    """
    if usage.is_complete:
        return usage
    guess = estimate_usage(messages, text, counter)
    return Usage(
        prompt_tokens=usage.prompt_tokens if usage.prompt_tokens is not None else guess.prompt_tokens,
        completion_tokens=usage.completion_tokens if usage.completion_tokens is not None else guess.completion_tokens,
        cached_tokens=usage.cached_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        reported=usage.reported,
        estimated=True,
        estimate_method=guess.estimate_method,
    )


@dataclass
class Completion:
    text: str | None
    usage: Usage
    latency_s: float  # the successful attempt only
    wall_s: float  # the whole call, including limiter waits and backoff
    finish_reason: str | None
    retries: int
    model_returned: str | None
    ratelimit_headers: dict[str, str]
    status_code: int = 200


# --- rate limiter ------------------------------------------------------------------------------


@dataclass
class Reservation:
    entry: list  # [timestamp, tokens]; shared by the minute and day windows


class RateLimiter:
    """Request and token budgets per minute and per day for one endpoint.

    Minute budgets are sliding 60-second windows: a request that does not fit waits for the
    window to free up. Day budgets are rolling 24-hour windows kept in a state file; a request
    that does not fit raises `QuotaExhausted` with the time the oldest usage ages out, because
    waiting hours inside a worker helps nobody. The rolling window is conservative for
    providers that reset at a fixed time of day: it never lets a run exceed the cap.

    Tokens are reserved before a request (estimated input plus expected output) and
    reconciled with the real count afterwards.
    """

    def __init__(
        self,
        *,
        rpm: int | None = None,
        rpd: int | None = None,
        tpm: int | None = None,
        tpd: int | None = None,
        state_path: Path | None = None,
        label: str = "",
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.rpm, self.rpd, self.tpm, self.tpd = rpm, rpd, tpm, tpd
        self.label = label
        self._clock, self._sleep = clock, sleep
        self._state_path = Path(state_path) if state_path else None
        self._minute: deque[list] = deque()
        self._day: deque[list] = deque()
        self._blocked_until: float | None = None
        self._blocked_reason = ""
        self._lock = asyncio.Lock()
        self._load()

    # state --------------------------------------------------------------------------------
    def _load(self) -> None:
        if not self._state_path or not self._state_path.exists():
            return
        try:
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return  # an unreadable state file is treated as no history rather than a crash
        self._day = deque([float(t), int(n)] for t, n in state.get("day", []))
        # The last minute's requests are part of the same entries, so a restart straight after a
        # stop cannot briefly exceed the per-minute cap.
        self._minute = deque(e for e in self._day if e[0] > self._clock() - MINUTE_S)
        self._blocked_until = state.get("blocked_until")
        self._blocked_reason = state.get("blocked_reason", "")

    def _save(self) -> None:
        if not self._state_path:
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {"day": [list(e) for e in self._day], "blocked_until": self._blocked_until, "blocked_reason": self._blocked_reason}
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, self._state_path)

    def _prune(self, now: float) -> None:
        while self._minute and self._minute[0][0] <= now - MINUTE_S:
            self._minute.popleft()
        while self._day and self._day[0][0] <= now - DAY_S:
            self._day.popleft()
        if self._blocked_until is not None and now >= self._blocked_until:
            self._blocked_until, self._blocked_reason = None, ""

    # budgets ------------------------------------------------------------------------------
    @staticmethod
    def _frees_at(window: deque, need: float, span: float) -> float:
        """When enough of the window's oldest tokens will have aged out to free `need` tokens."""
        freed = 0.0
        for stamp, tokens in window:
            freed += tokens
            if freed >= need:
                return stamp + span
        return window[-1][0] + span if window else 0.0

    def block_until(self, reset_at: float, reason: str) -> None:
        """Remember that the server said to stop until `reset_at` (persisted)."""
        self._blocked_until, self._blocked_reason = reset_at, reason
        self._save()

    @property
    def blocked_until(self) -> float | None:
        self._prune(self._clock())
        return self._blocked_until

    def usage_today(self) -> tuple[int, int]:
        """(requests, tokens) in the rolling 24-hour window."""
        self._prune(self._clock())
        return len(self._day), int(sum(e[1] for e in self._day))

    async def acquire(self, est_tokens: int = 0) -> Reservation:
        """Wait for room in the minute budgets, or raise QuotaExhausted if a day budget is spent."""
        async with self._lock:
            while True:
                now = self._clock()
                self._prune(now)
                if self._blocked_until is not None:
                    raise QuotaExhausted(self._blocked_until, self._blocked_reason or "blocked by an earlier rate-limit response", self.label)
                if self.rpd is not None and len(self._day) >= self.rpd:
                    raise QuotaExhausted(self._day[0][0] + DAY_S, f"{self.rpd} requests per day used", self.label)
                if self.tpd is not None:
                    if est_tokens > self.tpd:
                        raise RequestTooLarge(f"a request of ~{est_tokens} tokens exceeds the daily budget of {self.tpd}")
                    used = sum(e[1] for e in self._day)
                    if used + est_tokens > self.tpd:
                        reset = self._frees_at(self._day, used + est_tokens - self.tpd, DAY_S)
                        raise QuotaExhausted(reset, f"{self.tpd} tokens per day used ({int(used)} used)", self.label)
                wait = 0.0
                if self.rpm is not None and len(self._minute) >= self.rpm:
                    wait = max(wait, self._minute[0][0] + MINUTE_S - now)
                if self.tpm is not None:
                    if est_tokens > self.tpm:
                        raise RequestTooLarge(f"a request of ~{est_tokens} tokens exceeds the per-minute budget of {self.tpm}")
                    used_m = sum(e[1] for e in self._minute)
                    if used_m + est_tokens > self.tpm:
                        wait = max(wait, self._frees_at(self._minute, used_m + est_tokens - self.tpm, MINUTE_S) - now)
                if wait <= 0:
                    entry = [now, est_tokens]
                    self._minute.append(entry)
                    self._day.append(entry)
                    self._save()
                    return Reservation(entry)
                await self._sleep(wait + 0.001)

    def commit(self, reservation: Reservation, actual_tokens: int | None) -> None:
        """Replace the reserved token count with the real one (None keeps the reservation)."""
        if actual_tokens is not None:
            reservation.entry[1] = actual_tokens
            self._save()


# --- the client ----------------------------------------------------------------------------------


class ChatClient:
    """Sends chat requests to one endpoint, one `complete()` call at a time per request."""

    def __init__(
        self,
        endpoint: Endpoint,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        limiter: RateLimiter | None = None,
        state_dir: Path | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: Callable[[], float] = random.random,
        counter: Counter | None = None,
        environ: Mapping[str, str] | None = None,
    ):
        self.endpoint = endpoint
        self._transport = transport
        self._clock, self._sleep, self._rng = clock, sleep, rng
        self._counter = counter
        self._environ = environ
        if limiter is None:
            day_limited = endpoint.rpd is not None or endpoint.tpd is not None
            state_path = None
            if day_limited:
                safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", endpoint.limit_group)
                state_path = (state_dir or config.CACHE_DIR / "ratelimit") / f"{safe}.json"
            limiter = RateLimiter(
                rpm=endpoint.rpm, rpd=endpoint.rpd, tpm=endpoint.tpm, tpd=endpoint.tpd,
                state_path=state_path, label=endpoint.limit_group, clock=clock, sleep=sleep,
            )
        self.limiter = limiter
        self._http: httpx.AsyncClient | None = None
        self._sem: asyncio.Semaphore | None = None
        self._out_ema = float(min(endpoint.max_output_tokens, 512))

    # lifecycle ----------------------------------------------------------------------------
    async def __aenter__(self) -> ChatClient:
        self._ensure()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def _ensure(self) -> httpx.AsyncClient:
        if self._http is None:
            headers = {"Content-Type": "application/json", "Accept": "application/json", **self.endpoint.headers}
            key = self.endpoint.api_key(self._environ)
            if key:
                headers["Authorization"] = f"Bearer {key}"
            self._http = httpx.AsyncClient(
                headers=headers,
                timeout=httpx.Timeout(self.endpoint.timeout_s, connect=15.0),
                transport=self._transport,
            )
            if self.endpoint.max_concurrency:
                self._sem = asyncio.Semaphore(self.endpoint.max_concurrency)
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # token estimates ------------------------------------------------------------------------
    @property
    def counter(self) -> Counter | None:
        """The token counter in use, or None for the default (tiktoken, else chars/4)."""
        return self._counter

    @property
    def needs_token_estimates(self) -> bool:
        e = self.endpoint
        return any(v is not None for v in (e.max_input_tokens, e.tpm, e.tpd))

    def estimate_input_tokens(self, messages: Sequence[Mapping[str, str]]) -> int:
        return count_message_tokens(messages, self._counter)

    @property
    def expected_output_tokens(self) -> int:
        """What to reserve for the reply: a moving average of real replies, capped by max tokens."""
        return max(16, min(self.endpoint.max_output_tokens, math.ceil(self._out_ema * 1.5)))

    # the call -------------------------------------------------------------------------------
    async def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        response_format: Mapping[str, Any] | None = None,
        extra_params: Mapping[str, Any] | None = None,
    ) -> Completion:
        """One chat completion, with limiter, retries and quota handling.

        Raises InputTooLong (nothing sent), ApiStatusError (4xx other than 429),
        RetriesExhausted, MalformedResponse, or QuotaExhausted.
        """
        endpoint = self.endpoint
        http = self._ensure()
        est_in = self.estimate_input_tokens(messages) if self.needs_token_estimates else 0
        if endpoint.max_input_tokens and est_in > endpoint.max_input_tokens:
            raise InputTooLong(est_in, endpoint.max_input_tokens)
        body = build_request_body(endpoint, messages, response_format=response_format, extra_params=extra_params)
        url = endpoint.base_url.rstrip("/") + "/chat/completions"
        started = time.perf_counter()
        if self._sem is not None:
            async with self._sem:
                return await self._call(http, url, body, est_in, started)
        return await self._call(http, url, body, est_in, started)

    async def _call(self, http: httpx.AsyncClient, url: str, body: dict[str, Any], est_in: int, started: float) -> Completion:
        endpoint = self.endpoint
        retries = 0
        while True:
            reservation = await self.limiter.acquire(est_in + self.expected_output_tokens)
            attempt_start = time.perf_counter()
            status: int | None = None
            headers: dict[str, str] = {}
            error_text = ""
            try:
                response = await http.post(url, json=body)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                self.limiter.commit(reservation, 0)
                error_text = f"{type(exc).__name__}: {exc}"
            else:
                latency = time.perf_counter() - attempt_start
                status = response.status_code
                headers = ratelimit_headers(response.headers)
                if status == 200:
                    return self._finish(response, reservation, est_in, latency, started, retries, headers)
                self.limiter.commit(reservation, 0)  # a rejected request does not spend the token budget
                error_text = response.text
                if status != 429 and status < 500:
                    raise ApiStatusError(status, error_text, headers)

            retries += 1
            now = self._clock()
            wait_hint = reset_after_s(headers, now) if headers else None
            if status == 429:
                daily_hint = _looks_like_daily_quota(error_text)
                if (wait_hint is not None and wait_hint > endpoint.quota_wait_threshold_s) or (daily_hint and wait_hint is None):
                    reset_at = now + wait_hint if wait_hint is not None else None
                    reason = "the server reports a rate limit that resets later than this run should wait"
                    if reset_at is not None:
                        self.limiter.block_until(reset_at, reason)
                    raise QuotaExhausted(reset_at, reason, endpoint.limit_group)
            if retries > endpoint.max_retries:
                raise RetriesExhausted(retries, status, error_text, headers)
            backoff = min(endpoint.backoff_max_s, endpoint.backoff_base_s * 2 ** (retries - 1))
            delay = backoff * (0.5 + self._rng() / 2)
            if wait_hint is not None:
                delay = max(delay, min(wait_hint, endpoint.quota_wait_threshold_s))
            await self._sleep(delay)

    def _finish(self, response: httpx.Response, reservation: Reservation, est_in: int, latency: float, started: float, retries: int, headers: dict[str, str]) -> Completion:
        try:
            payload = response.json()
            choice = payload["choices"][0]
            text = _message_text(choice["message"].get("content"))
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            self.limiter.commit(reservation, None)
            raise MalformedResponse(f"not a chat completion: {response.text[:200]!r}") from exc
        usage = parse_usage(payload, self.endpoint.reasoning_in_completion)
        if usage.is_complete:
            self.limiter.commit(reservation, usage.total_tokens)
            self._out_ema = 0.8 * self._out_ema + 0.2 * usage.completion_tokens
        else:
            self.limiter.commit(reservation, None)  # keep the reservation: be conservative
        return Completion(
            text=text,
            usage=usage,
            latency_s=latency,
            wall_s=time.perf_counter() - started,
            finish_reason=choice.get("finish_reason"),
            retries=retries,
            model_returned=payload.get("model"),
            ratelimit_headers=headers,
        )


async def probe_rtt(
    endpoint: Endpoint,
    *,
    n: int = 5,
    path: str = "/models",
    transport: httpx.AsyncBaseTransport | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Round-trip time to an endpoint, from n GET requests to `base_url + path`.

    Meant for the network baseline of a local or rented server (it bills no tokens). It still
    counts as n requests to the provider and bypasses the limiter, so do not point it at a
    free-tier endpoint with a tight request cap.
    """
    headers = {"Accept": "application/json", **endpoint.headers}
    key = endpoint.api_key(environ)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    samples: list[float] = []
    statuses: list[int | str] = []
    async with httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(15.0), transport=transport) as http:
        for _ in range(n):
            start = time.perf_counter()
            try:
                response = await http.get(endpoint.base_url.rstrip("/") + path)
            except httpx.HTTPError as exc:
                statuses.append(type(exc).__name__)
                continue
            samples.append(time.perf_counter() - start)
            statuses.append(response.status_code)
    ordered = sorted(samples)
    return {
        "url": endpoint.base_url.rstrip("/") + path,
        "n": n,
        "ok": len(samples),
        "statuses": statuses,
        "min_s": ordered[0] if ordered else None,
        "median_s": ordered[len(ordered) // 2] if ordered else None,
        "max_s": ordered[-1] if ordered else None,
        "samples_s": samples,
    }


# --- batches --------------------------------------------------------------------------------------


@dataclass
class BatchItem:
    id: str
    messages: list[dict[str, str]]
    response_format: dict[str, Any] | None = None


@dataclass
class BatchResult:
    status: str  # complete | quota_exhausted | budget | auth_error | too_many_errors
    attempted: int = 0
    completed: int = 0
    errors: int = 0
    skipped: int = 0  # already done in an earlier run
    remaining: int = 0  # not attempted because the run stopped
    spent_usd: float = 0.0  # at list price, no caching (the upper bound)
    projected_usd: float | None = None
    reset_at: float | None = None
    message: str = ""


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def latest_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """The last row written for each id (an id can have an error row followed by a good one)."""
    return {row["id"]: row for row in rows}


def is_final(row: Mapping[str, Any]) -> bool:
    """A finished item: it succeeded, or failed in a way that retrying cannot fix."""
    return row.get("error") is None or row.get("retriable") is False


async def run_batch(
    client: ChatClient,
    items: Sequence[BatchItem],
    out_path: Path,
    *,
    concurrency: int = 4,
    resume: bool = False,
    config_hash: str | None = None,
    price: Price | None = None,
    max_usd: float | None = None,
    prefix_tokens: int = 0,
    progress: Callable[[int, int, Mapping[str, Any]], None] | None = None,
    max_fail_streak: int = 8,
    warmup: int = 5,
) -> BatchResult:
    """Run `items` through `client`, appending one JSON row per item to `out_path`.

    Resumable: with `resume=True`, items that already have a final row are skipped and the
    rest are attempted; without it an existing non-empty file is an error, so results are
    never silently overwritten or mixed. Rows from a different `config_hash` are refused.

    Stops cleanly (progress is already on disk) when a daily quota runs out, when projected
    spend would pass `max_usd`, on an authentication failure, or after `max_fail_streak`
    consecutive failures. `max_usd` needs a `price`; spend is at list price with no caching.
    """
    out_path = Path(out_path)
    existing = read_rows(out_path)
    if existing and not resume:
        raise FileExistsError(f"{out_path} already has {len(existing)} rows; pass resume=True (--resume) to continue it")
    if config_hash is not None:
        stale = {r.get("config_hash") for r in existing} - {config_hash, None}
        if stale:
            raise ConfigMismatch(
                f"{out_path} holds rows from a different configuration ({sorted(stale)[0][:12]}...). "
                "Move or delete it to start again."
            )
    done = {i for i, r in latest_rows(existing).items() if is_final(r)}
    pending = [it for it in items if it.id not in done]
    result = BatchResult(status="complete", skipped=len(items) - len(pending), remaining=len(pending))
    if not pending:
        return result
    if max_usd is not None and price is None:
        raise ValueError("--max-usd needs a price to project spend; this system has none")

    est_out = client.expected_output_tokens
    per_call: list[float] = []
    if price is not None:
        for it in pending:
            usage = Usage(client.estimate_input_tokens(it.messages), est_out)
            per_call.append(api_cost_bounds(usage, price, prefix_tokens).upper)
    if max_usd is not None:
        result.projected_usd = sum(per_call)
        if result.projected_usd > max_usd:
            raise BudgetExceeded(
                f"projected ${result.projected_usd:.4f} for {len(pending)} calls exceeds --max-usd ${max_usd:.4f}; "
                "lower --limit, use a cheaper system, or raise --max-usd"
            )

    queue: asyncio.Queue[BatchItem] = asyncio.Queue()
    for it in pending:
        queue.put_nowait(it)
    stop = asyncio.Event()
    write_lock = asyncio.Lock()
    state = {"streak": 0, "in_flight": 0, "reset_at": None}
    spent: list[float] = []
    workers = max(1, min(concurrency, client.endpoint.max_concurrency or concurrency))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def stop_with(status: str, message: str, reset_at: float | None = None) -> None:
        if not stop.is_set():
            result.status, result.message = status, message
        if reset_at is not None:
            state["reset_at"] = max(reset_at, state["reset_at"] or 0.0)
        stop.set()

    async def write(handle, row: dict[str, Any]) -> None:
        async with write_lock:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if progress:
                progress(result.completed + result.errors, len(pending), row)

    def error_row(item: BatchItem, kind: str, message: str, retriable: bool, status: int | None = None, headers: Mapping[str, str] | None = None, retries: int = 0) -> dict[str, Any]:
        return {"id": item.id, "config_hash": config_hash, "text": None, "usage": None, "error": message[:500],
                "error_kind": kind, "status_code": status, "retriable": retriable, "retries": retries,
                "ratelimit_headers": dict(headers or {}), "finished_at": _now()}

    async def worker(handle) -> None:
        while not stop.is_set():
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            result.attempted += 1
            state["in_flight"] += 1
            try:
                completion = await client.complete(item.messages, response_format=item.response_format)
            except QuotaExhausted as exc:
                result.attempted -= 1  # nothing was sent and nothing is recorded: resume retries it
                queue.put_nowait(item)
                stop_with("quota_exhausted", str(exc), exc.reset_at)
                return
            except InputTooLong as exc:
                row = error_row(item, "input_too_long", str(exc), retriable=False)
            except ApiStatusError as exc:
                row = error_row(item, f"http_{exc.status}", str(exc), retriable=False, status=exc.status, headers=exc.headers)
                if exc.status in (401, 403):
                    stop_with("auth_error", f"{exc}. Check the API key in .env.")
            except RetriesExhausted as exc:
                row = error_row(item, "retries_exhausted", str(exc), retriable=True, status=exc.last_status, headers=exc.headers, retries=exc.attempts)
            except MalformedResponse as exc:
                row = error_row(item, "malformed_response", str(exc), retriable=True)
            else:
                usage = complete_usage(completion.usage, item.messages, completion.text, client.counter)
                row = {"id": item.id, "config_hash": config_hash, "text": completion.text,
                       "usage": usage.to_dict(), "latency_s": completion.latency_s, "wall_s": completion.wall_s,
                       "finish_reason": completion.finish_reason, "retries": completion.retries,
                       "model_returned": completion.model_returned, "status_code": completion.status_code,
                       "ratelimit_headers": completion.ratelimit_headers, "error": None, "retriable": None,
                       "finished_at": _now()}
                if price is not None:
                    spent.append(api_cost_bounds(usage, price, prefix_tokens).upper)
            finally:
                state["in_flight"] -= 1

            if row["error"] is None:
                result.completed += 1
                state["streak"] = 0
            else:
                result.errors += 1
                state["streak"] += 1
                if state["streak"] >= max_fail_streak:
                    stop_with("too_many_errors", f"{max_fail_streak} consecutive failures; last: {row['error'][:120]}")
            await write(handle, row)
            if max_usd is not None and spent:
                total = sum(spent)
                mean = total / len(spent)
                remaining = queue.qsize() + state["in_flight"]
                projected = total + mean * remaining
                result.projected_usd = projected
                if total >= max_usd or (len(spent) >= warmup and projected > max_usd):
                    stop_with("budget", f"projected spend ${projected:.4f} would pass --max-usd ${max_usd:.4f}")

    with open(out_path, "a", encoding="utf-8", newline="\n") as handle:
        await asyncio.gather(*(worker(handle) for _ in range(workers)))
    result.remaining = queue.qsize()
    result.spent_usd = sum(spent)
    result.reset_at = state["reset_at"]
    return result


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
