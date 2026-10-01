"""Stub helpers shared by the client and batch tests: a fake clock, canned chat completions,
a scripted HTTP handler, and a client wired to them. Nothing here touches the network."""

from __future__ import annotations

import asyncio
import json

import httpx

from finetune_vs_api.client import ChatClient, Endpoint

MESSAGES = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "wake me at six"}]
SCHEMA = {"type": "json_schema", "json_schema": {"name": "function_call", "strict": True, "schema": {}}}


def words(text: str) -> int:
    return len(text.split())


class FakeTime:
    """A clock that only moves when something sleeps."""

    def __init__(self, start: float = 1_800_000_000.0):
        self.now = start
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def chat_response(
    text="{\"intent\":\"alarm_set\",\"slots\":[]}",
    *,
    prompt=100,
    completion=20,
    cached=None,
    reasoning=None,
    model="stub-model-2026-09-01",
    finish="stop",
    usage=True,
    headers=None,
    status=200,
):
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
    }
    if usage:
        body["usage"] = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
        if cached is not None:
            body["usage"]["prompt_tokens_details"] = {"cached_tokens": cached}
        if reasoning is not None:
            body["usage"]["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    return httpx.Response(status, json=body, headers=headers)


class Script:
    """A handler that replays responses in order (repeating the last) and records requests."""

    def __init__(self, *items):
        self.items = list(items)
        self.requests: list[httpx.Request] = []

    def __call__(self, request):
        self.requests.append(request)
        item = self.items.pop(0) if len(self.items) > 1 else self.items[0]
        if isinstance(item, Exception):
            raise item
        return item

    def bodies(self):
        return [json.loads(r.content) for r in self.requests]


def endpoint(**overrides) -> Endpoint:
    base = {
        "name": "stub",
        "base_url": "http://stub.test/v1",
        "model": "stub-model",
        "api_key_env": "STUB_KEY",
        "params": {"temperature": 0, "max_tokens": 100},
    }
    return Endpoint(**{**base, **overrides})


def make_client(handler, ep=None, ft=None, tmp_path=None, **kw):
    ft = ft or FakeTime()
    client = ChatClient(
        ep or endpoint(),
        transport=httpx.MockTransport(handler),
        clock=ft.clock,
        sleep=ft.sleep,
        rng=lambda: 1.0,  # jitter factor 1.0: the delay is exactly the backoff
        counter=words,
        environ={"STUB_KEY": "s3cret"},
        state_dir=tmp_path,
        **kw,
    )
    return client, ft


def run(coro):
    return asyncio.run(coro)


async def one(client, messages=MESSAGES, **kw):
    async with client:
        return await client.complete(messages, **kw)
