"""check_free_tiers against a stub. No real endpoint is contacted and no key is needed."""

from __future__ import annotations

import asyncio
import json
import shutil

import httpx
import pytest
import yaml

from conftest import ROOT, build_processed, load_script
from finetune_vs_api import prompts, schema

KEYS = {"GEMINI_API_KEY": "gemini-token", "GROQ_API_KEY": "groq-token"}
GROQ_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]  # one probe each, in the order of systems.yaml
GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
#: No endpoint in the real configs has a catalog_url now, so the tests that cover the catalog give one to their own copy.
CATALOG_URL = "https://catalog.example.test/catalog/models"
CATALOG = [
    {"id": "openai/gpt-oss-20b", "name": "GPT OSS 20B", "publisher": "OpenAI", "rate_limit_tier": "low", "limits": {"max_input_tokens": 8000}},
    {"id": "openai/gpt-oss-120b", "name": "GPT OSS 120B", "publisher": "OpenAI", "rate_limit_tier": "high"},
    {"id": "qwen/qwen3.8-27b", "name": "Qwen3.8 27B", "publisher": "Qwen", "rate_limit_tier": "custom"},
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
            headers={"x-ratelimit-limit-requests": "1000", "x-ratelimit-remaining-requests": "999"},
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

    @property
    def gets(self):
        return [r for r in self.requests if r.method == "GET"]

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

    def edit_systems(change):
        """Change this test's copy of configs/systems.yaml: `change` gets the parsed document."""
        path = e.config_dir / "systems.yaml"
        doc = yaml.safe_load(path.read_text())
        change(doc)
        path.write_text(yaml.safe_dump(doc, sort_keys=False))

    e.run = run
    e.edit_systems = edit_systems
    e.give_catalog_url = lambda endpoint: edit_systems(lambda doc: doc["endpoints"][endpoint].update(catalog_url=CATALOG_URL))
    e.read = lambda name: json.loads((e.results / "free_tiers" / f"{name}.json").read_text())
    return e


def test_one_small_request_per_model_with_the_real_strict_schema(env):
    assert env.run() == 0
    assert len(env.stub.posts) == 4  # the three Groq models and gemini-3.5-flash-lite
    assert not [r for r in env.stub.requests if r.url.host == "127.0.0.1"]  # the local endpoint is not a free tier
    groq_bodies = env.stub.bodies("api.groq.com")
    assert [b["model"] for b in groq_bodies] == GROQ_MODELS  # one probe per model, none repeated
    body = groq_bodies[0]
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
    assert by_host == {"api.groq.com": "Bearer groq-token", "generativelanguage.googleapis.com": "Bearer gemini-token"}


def test_each_probe_carries_the_decoding_parameters_of_its_row(env):
    """A parameter an endpoint rejects (reasoning_effort "none" on the Qwen row, say) must show up in the probe."""
    env.run()
    assert {b["model"]: (b["temperature"], b["reasoning_effort"], b["max_completion_tokens"]) for b in env.stub.bodies("api.groq.com")} == {
        "openai/gpt-oss-20b": (0, "low", 1024),
        "openai/gpt-oss-120b": (0, "low", 1024),
        "qwen/qwen3.8-27b": (0, "none", 256),
    }
    [gemini_post] = [r for r in env.stub.posts if r.url.host == "generativelanguage.googleapis.com"]
    gemini = json.loads(gemini_post.content)
    assert str(gemini_post.url) == f"{GEMINI_URL}/chat/completions" and gemini["model"] == GEMINI_MODEL
    assert (gemini["temperature"], gemini["reasoning_effort"], gemini["max_completion_tokens"]) == (0, "minimal", 256)
    # the probe always asks whether the strict schema is accepted, whatever the config says
    assert gemini["response_format"]["json_schema"]["strict"] is True


def test_what_is_recorded_for_each_model(env):
    env.run()
    groq = env.read("groq")
    assert set(groq["models"]) == set(GROQ_MODELS) and all(m["answered"] for m in groq["models"].values())
    entry = groq["models"]["openai/gpt-oss-20b"]
    assert entry["answered"] is True and entry["strict_json_schema_accepted"] is True
    assert entry["model_requested"] == "openai/gpt-oss-20b"
    assert entry["model_returned"] == "openai/gpt-oss-20b-2026-09-01"  # what the server says it ran
    assert entry["usage_reported"] is True
    assert entry["usage_fields_present"] == ["prompt_tokens", "completion_tokens", "cached_tokens"]
    assert entry["ratelimit_headers"] == {"x-ratelimit-limit-requests": "1000", "x-ratelimit-remaining-requests": "999"}
    assert entry["tier"] is None  # no row sets one now
    [attempt] = entry["attempts"]
    assert attempt["kind"] == "strict_json_schema" and attempt["status"] == 200
    assert attempt["answer_valid_for_schema"] is True and attempt["latency_s"] >= 0
    gemini = env.read("gemini")
    assert gemini["base_url"] == GEMINI_URL and set(gemini["models"]) == {GEMINI_MODEL}
    assert gemini["models"][GEMINI_MODEL]["answered"] is True
    assert gemini["models"][GEMINI_MODEL]["model_returned"] == "gemini-3.5-flash-lite-2026-09-01"
    assert {p.stem for p in (env.results / "free_tiers").iterdir()} == {"groq", "gemini"}  # not local


def test_a_tier_set_on_a_row_is_recorded_with_its_probe(env):
    """No row in the real configs sets a tier now (it was a GitHub Models idea); one that does is still recorded."""
    env.edit_systems(lambda doc: doc["systems"]["groq-gpt-oss-120b-k10"].update(tier="high"))
    env.run(endpoints=["groq"])
    models = env.read("groq")["models"]
    assert models["openai/gpt-oss-120b"]["tier"] == "high" and models["openai/gpt-oss-20b"]["tier"] is None


def test_no_endpoint_in_the_configs_has_a_catalog_url_so_none_is_requested(env):
    assert env.run() == 0
    assert env.stub.gets == []
    assert "catalog" not in env.read("groq") and "catalog" not in env.read("gemini")


def test_an_endpoint_with_a_catalog_url_also_lists_the_catalog_with_tiers(env):
    env.give_catalog_url("groq")
    assert env.run() == 0
    catalog = env.read("groq")["catalog"]
    assert catalog["url"] == CATALOG_URL and catalog["error"] is None and catalog["n_models"] == 3
    assert catalog["models_per_tier"] == {"low": 1, "high": 1, "custom": 1}
    assert {m["id"]: m["tier"] for m in catalog["models"]}["openai/gpt-oss-120b"] == "high"
    assert "catalog" not in env.read("gemini")  # no catalog_url configured for it
    [get] = env.stub.gets
    assert str(get.url) == CATALOG_URL and get.headers["authorization"] == "Bearer groq-token"


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
    assert "response_format" not in env.stub.bodies("api.groq.com")[1]  # the plain request that follows the first model's


def test_a_200_that_is_not_a_chat_completion_is_recorded_and_a_plain_request_follows(env):
    def chat(request, body):
        if "response_format" in body:
            return httpx.Response(200, json={"error": {"message": "something unexpected"}})
        return Stub.default_chat(request, body)

    env.stub.chat = chat
    assert env.run(endpoints=["groq"]) == 0
    entry = env.read("groq")["models"]["openai/gpt-oss-120b"]
    assert entry["strict_json_schema_accepted"] is None and entry["answered"] is True
    assert [a["kind"] for a in entry["attempts"]] == ["strict_json_schema", "plain"]
    assert entry["attempts"][0]["error"].startswith("malformed response")
    assert entry["model_returned"] == "openai/gpt-oss-120b-2026-09-01"  # from the plain request


def test_two_unreadable_answers_are_recorded_without_crashing(env):
    env.stub.chat = lambda request, body: httpx.Response(200, json={"unexpected": True})
    assert env.run(endpoints=["groq"]) == 1
    entry = env.read("groq")["models"]["openai/gpt-oss-120b"]
    assert entry["answered"] is False and entry["strict_json_schema_accepted"] is None
    assert all(a["error"].startswith("malformed response") for a in entry["attempts"])


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
    def chat(request, body):
        if request.url.host == "api.groq.com":
            return httpx.Response(401, json={"error": "bad credentials"})
        return Stub.default_chat(request, body)

    env.stub.chat = chat
    assert env.run() == 1
    assert len(env.stub.bodies("api.groq.com")) == 1  # the other two Groq models were not even tried
    models = env.read("groq")["models"]
    assert set(models) == {"openai/gpt-oss-20b"}
    entry = models["openai/gpt-oss-20b"]
    assert entry["answered"] is False and entry["strict_json_schema_accepted"] is None
    assert entry["attempts"][0]["status"] == 401
    assert any("groq: the key was refused" in line for line in env.lines)
    assert env.read("gemini")["models"][GEMINI_MODEL]["answered"] is True  # the other provider is still probed


def test_a_rate_limit_is_recorded_with_its_reset_and_never_retried(env):
    env.stub.chat = lambda request, body: httpx.Response(
        429, headers={"Retry-After": "3600", "x-ratelimit-remaining-requests": "0"}, text="limit"
    )
    assert env.run(endpoints=["groq"]) == 1
    assert [b["model"] for b in env.stub.bodies("api.groq.com")] == GROQ_MODELS  # each model once, nothing repeated
    attempt = env.read("groq")["models"]["openai/gpt-oss-20b"]["attempts"][0]
    assert attempt["status"] == 429 and attempt["quota_reset_at"].endswith("+00:00")


def test_a_short_429_is_recorded_without_a_retry(env):
    env.stub.chat = lambda request, body: httpx.Response(429, headers={"Retry-After": "2", "x-ratelimit-limit-requests": "30"}, text="slow")
    env.run(endpoints=["groq"])
    attempt = env.read("groq")["models"]["openai/gpt-oss-20b"]["attempts"][0]
    assert len(env.stub.posts) == len(GROQ_MODELS) and attempt["status"] == 429
    assert attempt["ratelimit_headers"]["retry-after"] == "2"


def test_a_missing_key_skips_that_endpoint_and_sends_nothing_to_it(env):
    assert env.run(environ={"GEMINI_API_KEY": "gemini-token"}) == 0
    assert {r.url.host for r in env.stub.requests} == {"generativelanguage.googleapis.com"}
    assert any("GROQ_API_KEY is not set" in line for line in env.lines)
    assert not (env.results / "free_tiers" / "groq.json").exists()


def test_with_no_keys_at_all_nothing_is_sent(env):
    assert env.run(environ={}) == 2
    assert env.stub.requests == []


def test_a_catalog_failure_does_not_stop_the_model_checks(env):
    env.give_catalog_url("groq")
    env.stub.catalog = lambda request: httpx.Response(500, text="catalog down")
    assert env.run(endpoints=["groq"]) == 0
    doc = env.read("groq")
    assert doc["catalog"]["status"] == 500 and "down" in doc["catalog"]["error"]
    assert doc["models"]["openai/gpt-oss-120b"]["answered"] is True


def test_an_unrecognized_catalog_shape_is_recorded_not_fatal(env):
    env.give_catalog_url("groq")
    env.stub.catalog = lambda request: httpx.Response(200, json={"weird": True})
    env.run(endpoints=["groq"])
    assert env.read("groq")["catalog"]["error"] == "unrecognized catalog format"


def test_results_are_merged_by_model_across_runs(env):
    env.run(endpoints=["groq"])
    assert set(env.read("groq")["models"]) == set(GROQ_MODELS)
    env.run(endpoints=["groq"], model="extra/model")  # a model no system uses, probed as well
    doc = env.read("groq")
    assert set(doc["models"]) == {*GROQ_MODELS, "extra/model"}
    # re-checking the configured models replaces their entries and keeps the extra one it did not re-check
    env.run(endpoints=["groq"])
    again = env.read("groq")
    assert set(again["models"]) == {*GROQ_MODELS, "extra/model"}
    assert again["models"]["extra/model"] == doc["models"]["extra/model"]


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
    for model in GROQ_MODELS:
        assert f"would send 1 request to https://api.groq.com/openai/v1/chat/completions, model {model}" in text
    assert f"would send 1 request to {GEMINI_URL}/chat/completions, model {GEMINI_MODEL}" in text
    assert "would list the catalog" not in text and "dry run: nothing was sent" in text


def test_a_dry_run_names_the_catalog_it_would_list(env):
    env.give_catalog_url("groq")
    assert env.run(dry_run=True, endpoints=["groq"]) == 0
    assert f"groq: would list the catalog at {CATALOG_URL}" in env.lines
    assert env.stub.requests == []


def test_the_extra_model_option_needs_one_endpoint(env):
    assert env.run(model="x/y") == 2
    assert env.run(endpoints=["groq", "gemini"], model="x/y") == 2
    assert env.run(endpoints=["local"], model="x/y") == 2
    assert env.stub.requests == []


def test_plan_covers_api_endpoints_once_per_model(env):
    wanted = env.script.plan(env.config_dir, None, None)
    assert set(wanted) == {"groq", "gemini"}  # local has no API key: not a free tier
    assert [s["model"] for s in wanted["groq"]] == GROQ_MODELS
    assert [s["model"] for s in wanted["gemini"]] == [GEMINI_MODEL]


def test_parse_catalog_shapes(env):
    parse = env.script.parse_catalog
    assert parse([{"id": "a", "rate_limit_tier": "low"}])[0]["tier"] == "low"
    assert parse({"models": [{"id": "a", "tier": "high"}]})[0]["tier"] == "high"
    assert parse([{"id": "a"}])[0]["tier"] is None
    assert parse({"models": []}) == []
    assert parse("nope") is None and parse({"x": 1}) is None
    assert parse([1, "a", {"id": "ok"}]) == [{"id": "ok", "tier": None}]


def test_the_catalog_functions_work_on_their_own(env):
    """fetch_catalog called directly, with no endpoint configured for it."""
    transport = httpx.MockTransport(env.stub)
    done = asyncio.run(env.script.fetch_catalog(CATALOG_URL, "a-token", transport))
    assert done["error"] is None and done["status"] == 200 and done["n_models"] == 3
    assert done["models_per_tier"] == {"low": 1, "high": 1, "custom": 1}
    assert env.stub.gets[0].headers["authorization"] == "Bearer a-token"
    env.stub.catalog = lambda request: httpx.Response(404, text="no such catalog")
    assert asyncio.run(env.script.fetch_catalog(CATALOG_URL, None, transport))["status"] == 404
    assert "authorization" not in env.stub.gets[-1].headers  # no token, no header


def test_main_passes_the_exit_code_through(env):
    code = env.script.main(
        ["--endpoint", "groq"], config_dir=env.config_dir, processed_dir=env.processed, results_dir=env.results,
        transport=httpx.MockTransport(env.stub), environ=KEYS, out=env.lines.append,
    )
    assert code == 0 and (env.results / "free_tiers" / "groq.json").exists()
