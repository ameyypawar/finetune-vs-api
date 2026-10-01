"""check_free_tiers against a stub. No real endpoint is contacted and no key is needed."""

from __future__ import annotations

import json
import shutil

import httpx
import pytest

from conftest import ROOT, build_processed, load_script
from finetune_vs_api import prompts, schema

KEYS = {"GITHUB_MODELS_TOKEN": "gh-token", "GROQ_API_KEY": "groq-token"}
CATALOG = [
    {"id": "openai/gpt-4.1-mini", "name": "GPT-4.1 mini", "publisher": "OpenAI", "rate_limit_tier": "low", "limits": {"max_input_tokens": 8000}},
    {"id": "openai/gpt-4.1", "name": "GPT-4.1", "publisher": "OpenAI", "rate_limit_tier": "high"},
    {"id": "deepseek/DeepSeek-R1", "name": "DeepSeek-R1", "publisher": "DeepSeek", "rate_limit_tier": "custom"},
]


class Stub:
    """Routes by host and path, and records every request that reaches it."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.chat = self.default_chat
        self.catalog = lambda request: httpx.Response(200, json=CATALOG)

    @staticmethod
    def default_chat(request, body):
        return httpx.Response(
            200,
            json={
                "model": body["model"] + "-2026-09-01",
                "choices": [{"message": {"content": '{"intent":"alarm_set","slots":[]}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 700, "completion_tokens": 12, "prompt_tokens_details": {"cached_tokens": 0}},
            },
            headers={"x-ratelimit-limit-requests": "150", "x-ratelimit-remaining-requests": "149"},
        )

    def __call__(self, request):
        self.requests.append(request)
        if request.url.path.endswith("/chat/completions"):
            return self.chat(request, json.loads(request.content))
        if request.url.path.endswith("/catalog/models"):
            return self.catalog(request)
        return httpx.Response(404, text="unexpected path")

    @property
    def posts(self):
        return [r for r in self.requests if r.method == "POST"]

    def bodies(self, host):
        return [json.loads(r.content) for r in self.posts if r.url.host == host]


@pytest.fixture
def env(tmp_path):
    class E:
        pass

    e = E()
    e.script = load_script("check_free_tiers")
    e.config_dir = tmp_path / "configs"
    shutil.copytree(ROOT / "configs", e.config_dir)
    e.processed = build_processed(tmp_path)
    e.results = tmp_path / "results"
    e.stub = Stub()
    e.lines: list[str] = []

    def run(**kw):
        defaults = dict(
            config_dir=e.config_dir, processed_dir=e.processed, results_dir=e.results,
            transport=httpx.MockTransport(e.stub), environ=KEYS, out=e.lines.append,
        )
        return e.script.run(**{**defaults, **kw})

    e.run = run
    e.read = lambda name: json.loads((e.results / "free_tiers" / f"{name}.json").read_text())
    return e


def test_one_small_request_per_model_with_the_real_strict_schema(env):
    assert env.run() == 0
    assert len(env.stub.posts) == 3  # gpt-4.1-mini and gpt-4.1 on GitHub Models, gpt-oss-120b on Groq
    assert not [r for r in env.stub.requests if r.url.host == "127.0.0.1"]  # the local endpoint is not a free tier
    body = env.stub.bodies("models.github.ai")[0]
    fmt = body["response_format"]
    inventory = schema.load_inventory(env.processed)
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["properties"]["intent"]["enum"] == list(inventory.intents)
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][1]["content"] == "wake me up at five am this week"
    system = body["messages"][0]["content"]
    assert system == prompts.static_prefix("zeroshot_v1", inventory)  # the zero-shot prompt: rules and label lists
    assert "copied verbatim" in system and "Intents:" in system
    by_host = {r.url.host: r.headers["authorization"] for r in env.stub.posts}
    assert by_host == {"models.github.ai": "Bearer gh-token", "api.groq.com": "Bearer groq-token"}


def test_what_is_recorded_for_each_model(env):
    env.run()
    github = env.read("github-models")
    assert set(github["models"]) == {"openai/gpt-4.1-mini", "openai/gpt-4.1"}
    entry = github["models"]["openai/gpt-4.1-mini"]
    assert entry["answered"] is True and entry["strict_json_schema_accepted"] is True
    assert entry["model_requested"] == "openai/gpt-4.1-mini"
    assert entry["model_returned"] == "openai/gpt-4.1-mini-2026-09-01"  # what the server says it ran
    assert entry["usage_reported"] is True
    assert entry["usage_fields_present"] == ["prompt_tokens", "completion_tokens", "cached_tokens"]
    assert entry["ratelimit_headers"] == {"x-ratelimit-limit-requests": "150", "x-ratelimit-remaining-requests": "149"}
    assert entry["tier"] == "low" and github["models"]["openai/gpt-4.1"]["tier"] == "high"
    [attempt] = entry["attempts"]
    assert attempt["kind"] == "strict_json_schema" and attempt["status"] == 200
    assert attempt["answer_valid_for_schema"] is True and attempt["latency_s"] >= 0
    assert env.read("groq")["models"]["openai/gpt-oss-120b"]["answered"] is True
    assert "local" not in {p.stem for p in (env.results / "free_tiers").iterdir()}


def test_github_models_also_lists_the_catalog_with_tiers(env):
    env.run()
    catalog = env.read("github-models")["catalog"]
    assert catalog["error"] is None and catalog["n_models"] == 3
    assert catalog["models_per_tier"] == {"low": 1, "high": 1, "custom": 1}
    assert {m["id"]: m["tier"] for m in catalog["models"]}["openai/gpt-4.1"] == "high"
    assert "catalog" not in env.read("groq")  # no catalog_url configured for Groq
    get = [r for r in env.stub.requests if r.method == "GET"][0]
    assert get.headers["authorization"] == "Bearer gh-token"


def test_a_rejected_strict_schema_is_recorded_and_a_plain_request_follows(env):
    def chat(request, body):
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "response_format json_schema is not supported"}})
        return Stub.default_chat(request, body)

    env.stub.chat = chat
    assert env.run(endpoints=["groq"]) == 0
    entry = env.read("groq")["models"]["openai/gpt-oss-120b"]
    assert entry["strict_json_schema_accepted"] is False and entry["answered"] is True
    assert [a["kind"] for a in entry["attempts"]] == ["strict_json_schema", "plain"]
    assert [a["status"] for a in entry["attempts"]] == [400, 200]
    assert "not supported" in entry["attempts"][0]["error"]
    assert entry["usage_reported"] is True  # taken from the plain request that did answer
    assert "response_format" not in env.stub.bodies("api.groq.com")[1]


def test_usage_that_the_provider_does_not_send_is_recorded_as_missing(env):
    def chat(request, body):
        return httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": "not json"}, "finish_reason": "stop"}]})

    env.stub.chat = chat
    env.run(endpoints=["groq"])
    entry = env.read("groq")["models"]["openai/gpt-oss-120b"]
    assert entry["usage_reported"] is False and entry["usage_fields_present"] == []
    assert entry["attempts"][0]["answer_valid_for_schema"] is False
    assert entry["ratelimit_headers"] == {}


def test_a_refused_key_stops_that_provider_and_exits_nonzero(env):
    env.stub.chat = lambda request, body: httpx.Response(401, json={"error": "bad credentials"})
    assert env.run(endpoints=["github-models"]) == 1
    assert len(env.stub.bodies("models.github.ai")) == 1  # the second model was not even tried
    entry = env.read("github-models")["models"]["openai/gpt-4.1-mini"]
    assert entry["answered"] is False and entry["strict_json_schema_accepted"] is None
    assert entry["attempts"][0]["status"] == 401
    assert any("key was refused" in line for line in env.lines)


def test_a_rate_limit_is_recorded_with_its_reset_and_never_retried(env):
    env.stub.chat = lambda request, body: httpx.Response(
        429, headers={"Retry-After": "3600", "x-ratelimit-remaining-requests": "0"}, text="limit"
    )
    assert env.run(endpoints=["groq"]) == 1
    assert len(env.stub.posts) == 1
    attempt = env.read("groq")["models"]["openai/gpt-oss-120b"]["attempts"][0]
    assert attempt["status"] == 429 and attempt["quota_reset_at"].endswith("+00:00")


def test_a_short_429_is_recorded_without_a_retry(env):
    env.stub.chat = lambda request, body: httpx.Response(429, headers={"Retry-After": "2", "x-ratelimit-limit-requests": "30"}, text="slow")
    env.run(endpoints=["groq"])
    attempt = env.read("groq")["models"]["openai/gpt-oss-120b"]["attempts"][0]
    assert len(env.stub.posts) == 1 and attempt["status"] == 429
    assert attempt["ratelimit_headers"]["retry-after"] == "2"


def test_a_missing_key_skips_that_endpoint_and_sends_nothing_to_it(env):
    assert env.run(environ={"GITHUB_MODELS_TOKEN": "gh-token"}) == 0
    assert {r.url.host for r in env.stub.requests} == {"models.github.ai"}
    assert any("GROQ_API_KEY is not set" in line for line in env.lines)
    assert not (env.results / "free_tiers" / "groq.json").exists()


def test_with_no_keys_at_all_nothing_is_sent(env):
    assert env.run(environ={}) == 2
    assert env.stub.requests == []


def test_a_catalog_failure_does_not_stop_the_model_checks(env):
    env.stub.catalog = lambda request: httpx.Response(500, text="catalog down")
    assert env.run(endpoints=["github-models"]) == 0
    doc = env.read("github-models")
    assert doc["catalog"]["status"] == 500 and "down" in doc["catalog"]["error"]
    assert doc["models"]["openai/gpt-4.1"]["answered"] is True


def test_an_unrecognized_catalog_shape_is_recorded_not_fatal(env):
    env.stub.catalog = lambda request: httpx.Response(200, json={"weird": True})
    env.run(endpoints=["github-models"])
    assert env.read("github-models")["catalog"]["error"] == "unrecognized catalog format"


def test_results_are_merged_by_model_across_runs(env, tmp_path):
    env.run(endpoints=["groq"])
    first = env.read("groq")["models"]["openai/gpt-oss-120b"]["checked_at"]
    env.run(endpoints=["groq"], model="openai/gpt-oss-20b")
    doc = env.read("groq")
    assert set(doc["models"]) == {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}
    # re-checking one model replaces its entry and keeps the other
    env.run(endpoints=["groq"])
    again = env.read("groq")
    assert set(again["models"]) == {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}
    assert again["models"]["openai/gpt-oss-20b"] == doc["models"]["openai/gpt-oss-20b"]
    assert first  # the first entry existed


def test_merge_into_keeps_models_it_does_not_mention(tmp_path):
    script = load_script("check_free_tiers")
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"endpoint": "e", "checked_at": "old", "models": {"a": {"v": 1}}}))
    merged = script.merge_into(path, {"endpoint": "e", "checked_at": "new", "models": {"b": {"v": 2}}})
    assert merged["models"] == {"a": {"v": 1}, "b": {"v": 2}} and merged["checked_at"] == "new"
    path.write_text("{broken")
    assert script.merge_into(path, {"models": {"c": {}}})["models"] == {"c": {}}  # a corrupt file is replaced, not fatal


def test_dry_run_sends_and_writes_nothing(env):
    assert env.run(dry_run=True) == 0
    assert env.stub.requests == [] and not (env.results / "free_tiers").exists()
    text = "\n".join(env.lines)
    assert "would send 1 request to https://models.github.ai/inference/chat/completions" in text
    assert "would list the catalog" in text and "dry run: nothing was sent" in text


def test_the_extra_model_option_needs_one_endpoint(env):
    assert env.run(model="x/y") == 2
    assert env.run(endpoints=["groq", "github-models"], model="x/y") == 2
    assert env.run(endpoints=["local"], model="x/y") == 2
    assert env.stub.requests == []


def test_plan_covers_api_endpoints_once_per_model(env):
    wanted = env.script.plan(env.config_dir, None, None)
    assert set(wanted) == {"github-models", "groq"}  # local has no API key: not a free tier
    assert [s["model"] for s in wanted["github-models"]] == ["openai/gpt-4.1-mini", "openai/gpt-4.1"]
    assert [s["model"] for s in wanted["groq"]] == ["openai/gpt-oss-120b"]


def test_parse_catalog_shapes(env):
    parse = env.script.parse_catalog
    assert parse([{"id": "a", "rate_limit_tier": "low"}])[0]["tier"] == "low"
    assert parse({"models": [{"id": "a", "tier": "high"}]})[0]["tier"] == "high"
    assert parse([{"id": "a"}])[0]["tier"] is None
    assert parse({"models": []}) == []
    assert parse("nope") is None and parse({"x": 1}) is None
    assert parse([1, "a", {"id": "ok"}]) == [{"id": "ok", "tier": None}]


def test_main_passes_the_exit_code_through(env):
    code = env.script.main(
        ["--endpoint", "groq"], config_dir=env.config_dir, processed_dir=env.processed, results_dir=env.results,
        transport=httpx.MockTransport(env.stub), environ=KEYS, out=env.lines.append,
    )
    assert code == 0 and (env.results / "free_tiers" / "groq.json").exists()
