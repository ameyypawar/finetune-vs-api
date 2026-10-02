"""finetune_vs_api.hf_server: the stdlib chat server of the accuracy-only path, with a stub in place of the
model. It is exercised over a real local socket by the repository's own ChatClient, so what the evaluation
reads (text, usage, finish reason, the model name) is what the server writes."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.error
import urllib.request

import httpx
import pytest

from finetune_vs_api import hf_server
from finetune_vs_api.client import ChatClient, Endpoint

REPLY = '{"intent":"alarm_set","slots":[]}'


class Model:
    """A stand-in for the model: records what it was asked and how many calls overlapped."""

    def __init__(self, delay=0.0, fail=None):
        self.calls, self.delay, self.fail, self.inflight, self.max_inflight = [], delay, fail, 0, 0

    def __call__(self, messages, max_tokens):
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            self.calls.append((messages, max_tokens))
            if self.delay:
                time.sleep(self.delay)
            if self.fail:
                raise RuntimeError(self.fail)
            return REPLY, 11, 7, "stop"
        finally:
            self.inflight -= 1


@pytest.fixture
def serve():
    servers = []

    def start(model, name="ft-qwen3-4b-lora"):
        server = hf_server.make_server(model, name)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def post(url, body, raw=None):
    request = urllib.request.Request(url + "/v1/chat/completions", data=raw if raw is not None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


MESSAGES = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "wake me at six"}]


def test_the_repositorys_client_gets_text_usage_finish_reason_and_the_model_name(serve):
    model = Model()
    url = serve(model)
    endpoint = Endpoint(name="local", base_url=url + "/v1", model="ft-qwen3-4b-lora", params={"temperature": 0, "max_tokens": 64})

    async def go():
        async with ChatClient(endpoint) as client:
            return await client.complete(MESSAGES)

    completion = asyncio.run(go())
    assert completion.text == REPLY and completion.finish_reason == "stop" and completion.model_returned == "ft-qwen3-4b-lora"
    assert (completion.usage.prompt_tokens, completion.usage.completion_tokens, completion.usage.reported) == (11, 7, True)
    assert model.calls == [(MESSAGES, 64)]


def test_max_completion_tokens_is_understood_and_a_default_applies(serve):
    model = Model()
    url = serve(model)
    post(url, {"model": "ft-qwen3-4b-lora", "messages": MESSAGES, "max_completion_tokens": 33})
    post(url, {"model": "ft-qwen3-4b-lora", "messages": MESSAGES})
    assert [limit for _, limit in model.calls] == [33, 256]


def test_health_and_model_list_answer_like_a_real_server(serve):
    url = serve(Model(), name="Qwen/Qwen3-4B-Instruct-2507")
    assert httpx.get(url + "/health").status_code == 200
    assert httpx.get(url + "/v1/models").json() == {"object": "list", "data": [{"id": "Qwen/Qwen3-4B-Instruct-2507", "object": "model"}]}
    assert httpx.get(url + "/nowhere").status_code == 404 and httpx.post(url + "/v1/other", json={}).status_code == 404


def test_only_the_served_model_name_is_answered(serve):
    model = Model()
    url = serve(model)
    status, body = post(url, {"model": "ft-qwen3-4b-lora-epoch-1", "messages": MESSAGES})
    assert status == 404 and "unknown model" in body["error"]["message"] and "ft-qwen3-4b-lora" in body["error"]["message"]
    assert model.calls == []  # a request for another model never reaches this one


@pytest.mark.parametrize(
    "body",
    [{"model": "ft-qwen3-4b-lora"}, {"model": "ft-qwen3-4b-lora", "messages": []}, {"model": "ft-qwen3-4b-lora", "messages": [{"role": "user"}]},
     {"model": "ft-qwen3-4b-lora", "messages": "hi"}, {"model": "ft-qwen3-4b-lora", "messages": MESSAGES, "max_tokens": -5},
     {"model": "ft-qwen3-4b-lora", "messages": MESSAGES, "max_tokens": "many"}],
)
def test_a_bad_request_is_a_400_and_never_reaches_the_model(serve, body):
    model = Model()
    status, answer = post(serve(model), body)
    assert status == 400 and answer["error"]["message"].startswith("bad request") and model.calls == []


def test_unparseable_json_is_a_400(serve):
    status, answer = post(serve(Model()), None, raw=b"{not json")
    assert status == 400 and "bad request" in answer["error"]["message"]


def test_a_model_error_is_reported_and_the_server_keeps_serving(serve):
    model = Model(fail="CUDA out of memory")
    url = serve(model)
    status, body = post(url, {"model": "ft-qwen3-4b-lora", "messages": MESSAGES})
    assert status == 500 and body["error"]["message"] == "RuntimeError: CUDA out of memory"
    model.fail = None
    assert post(url, {"model": "ft-qwen3-4b-lora", "messages": MESSAGES})[0] == 200


def test_concurrent_requests_are_answered_one_at_a_time(serve):
    model = Model(delay=0.02)
    url = serve(model)
    results = []
    threads = [threading.Thread(target=lambda: results.append(post(url, {"model": "ft-qwen3-4b-lora", "messages": MESSAGES})[0])) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert results == [200] * 6 and len(model.calls) == 6 and model.max_inflight == 1


def test_main_loads_the_model_then_serves_it_under_the_given_name():
    loaded = []

    def loader(model_dir, revision):
        loaded.append((model_dir, revision))
        return Model()

    server = hf_server.main(["--model-dir", "/tmp/merged/ft", "--served-name", "ft-qwen3-4b-lora", "--port", "0", "--revision", "abc"], loader=loader, block=False)
    try:
        assert loaded == [("/tmp/merged/ft", "abc")]
        url = f"http://127.0.0.1:{server.server_address[1]}"
        assert post(url, {"model": "ft-qwen3-4b-lora", "messages": MESSAGES})[0] == 200
    finally:
        server.shutdown()
        server.server_close()


def test_importing_the_module_pulls_in_no_heavy_library():
    import subprocess
    import sys

    code = "import sys, finetune_vs_api.hf_server; print(sorted({'torch', 'transformers', 'numpy', 'httpx'} & set(sys.modules)))"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "[]"
