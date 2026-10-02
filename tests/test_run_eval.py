"""The evaluation runner end to end, against a stub that answers from the gold labels.

The stub is an httpx MockTransport: no real endpoint is ever contacted. It reads the request,
finds which example it is about, and replies with that example's gold call, optionally
corrupted, so every metric in the summary has a known right answer.
"""

from __future__ import annotations

import json
import shutil

import httpx
import pytest
import yaml

from conftest import ROOT, build_processed, fake_embed, load_script
from finetune_vs_api import config, data, evaluate, prompts, schema
from finetune_vs_api.client import BudgetExceeded, ConfigMismatch, read_rows
from finetune_vs_api.config import LockError
from finetune_vs_api.schema import target_json
from stubs import FakeTime, chat_response, words

GPT_OSS_20B = "groq-gpt-oss-20b-k10"  # groq, S500, rpm 30 / rpd 1000 / tpm 8000 / tpd 200000, strict schema sent, price groq-gpt-oss-20b
QWEN_27B = "groq-qwen3.8-27b-k10"  # groq, S500, a price with no cached-input rate
GEMINI = "gemini-3.5-flash-lite-k10"  # gemini, S500, rpm 10 / rpd 500 and no token limits, strict schema not sent
LOCAL_BASE = "base-qwen3-4b-k10"
LOCAL_FT = "ft-qwen3-4b-lora"
KEYS = {"GROQ_API_KEY": "g", "GEMINI_API_KEY": "k"}


class World:
    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.processed = build_processed(tmp_path)
        self.config_dir = tmp_path / "configs"
        shutil.copytree(ROOT / "configs", self.config_dir)
        self.results = tmp_path / "results"
        self.ft = FakeTime()
        self.splits = {s: data.read_examples(self.processed / f"{s}.jsonl") for s in ("train", "dev", "test")}
        self.by_text = {e.text: e for rows in self.splits.values() for e in rows}
        self.bodies: list[dict] = []
        self.corrupt: dict[str, str] = {}
        self.status_for: dict[str, int] = {}
        self.usage = {"prompt": 1500, "completion": 30}

    # the stub endpoint ---------------------------------------------------------------------
    def handler(self, request):
        body = json.loads(request.content)
        self.bodies.append(body)
        example = self.by_text[body["messages"][-1]["content"]]
        if example.id in self.status_for:
            return httpx.Response(self.status_for[example.id], text="stub error")
        gold = target_json(example)
        wrong_intent = json.dumps({"intent": "weather_query" if example.intent != "weather_query" else "alarm_set", "slots": []})
        outputs = {
            "perfect": gold,
            "fenced": f"```json\n{gold}\n```",
            "prose": f"Sure! Here is the call: {gold}",
            "wrong_intent": wrong_intent,
            "extra_key": gold[:-1] + ',"confidence":0.9}',
            "invented_value": json.dumps({"intent": example.intent, "slots": [{"type": "time", "value": "midnight tomorrow"}]}),
        }
        text = outputs[self.corrupt.get(example.id, "perfect")]
        return chat_response(text, prompt=self.usage["prompt"], completion=self.usage["completion"], headers={"x-ratelimit-remaining-requests": "99"})

    # running ------------------------------------------------------------------------------------
    def kwargs(self, **extra):
        base = dict(
            transport=httpx.MockTransport(self.handler),
            embedder=fake_embed,
            counter=words,
            environ=KEYS,
            state_dir=self.tmp / "ratelimit",
            cache_dir=self.tmp / "cache",
            clock=self.ft.clock,
            sleep=self.ft.sleep,
            rng=lambda: 1.0,
            config_dir=self.config_dir,
            processed_dir=self.processed,
            results_dir=self.results,
            with_ci=False,
        )
        return {**base, **extra}

    def run(self, system, split, **extra):
        return evaluate.run_eval(system, split, **self.kwargs(**extra))

    def lock(self, system, subset=None, reason="frozen for the test run"):
        return config.write_test_lock(
            system, subset, reason, config_dir=self.config_dir, processed_dir=self.processed, lock_path=self.results / "test_lock.jsonl"
        )

    def edit_systems(self, old, new):
        path = self.config_dir / "systems.yaml"
        text = path.read_text()
        assert old in text
        path.write_text(text.replace(old, new, 1))

    def set_limits(self, system, **limits):
        """Change some of a row's limits in this test's copy of configs/systems.yaml."""
        path = self.config_dir / "systems.yaml"
        doc = yaml.safe_load(path.read_text())
        doc["systems"][system]["limits"].update(limits)
        path.write_text(yaml.safe_dump(doc, sort_keys=False))

    @property
    def train_texts(self):
        return {e.text for e in self.splits["train"]}


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "git_commit", lambda cwd=None: None)
    monkeypatch.setattr(config, "git_dirty", lambda cwd=None: True)
    return World(tmp_path)


# --- a dev run, start to finish ------------------------------------------------------------------------


def test_a_dev_run_writes_predictions_and_a_complete_summary(world):
    outcome = world.run(GPT_OSS_20B, "dev", subset="D50", n_resamples=200, with_ci=True)
    assert outcome.status == "complete" and len(world.bodies) == 50
    assert outcome.summary_path == world.results / "runs" / f"{GPT_OSS_20B}__dev" / "summary.D50.json"
    assert len(read_rows(outcome.predictions_path)) == 50

    s = outcome.summary
    assert (s["system"], s["split"], s["status"], s["n_scored"]) == (GPT_OSS_20B, "dev", "complete", 50)
    assert s["subset"]["name"] == "D50" and s["subset"]["n"] == 50 and len(s["subset"]["hash"]) == 64
    assert s["billing_basis"] == "free tier; priced at paid list price"
    assert len(s["config_hash"]) == 64 and set(s["config_components"]) == {"prompt", "schema", "system", "decoding", "checkpoint", "prices", "dataset"}
    assert s["git_commit"] is None and s["git_dirty"] is True  # a repo with no commits yet
    assert s["lock"] is None and s["stop"] is None
    assert s["endpoint"]["models_returned"] == {"stub-model-2026-09-01": 50}
    assert s["endpoint"]["model_requested"] == "openai/gpt-oss-20b"
    m = s["metrics"]
    for key in ("exact_match", "intent_accuracy", "slot_f1", "schema_valid_rate"):
        assert m[key]["value"] == 1.0 and len(m[key]["ci95"]) == 2
    assert m["unfound_value_rate"]["value"] == 0.0
    assert s["latency_s"]["method"] == "nearest-rank" and s["latency_s"]["p50"] <= s["latency_s"]["p95"]
    assert s["calls"]["last_ratelimit_headers"] == {"x-ratelimit-remaining-requests": "99"}


def test_the_summary_prices_the_calls_and_carries_the_price_snapshot(world):
    s = world.run(GPT_OSS_20B, "dev", subset="D50").summary
    c = s["cost"]
    assert c["billing_basis"] == "free tier; priced at paid list price"
    entry = yaml.safe_load((world.config_dir / "sources.yaml").read_text())["prices"]["groq-gpt-oss-20b"]
    assert c["price"] == {"id": "groq-gpt-oss-20b", **entry}  # the whole entry, with the URL and the date it was read
    assert c["price"]["url"].startswith("https://") and c["price"]["retrieved_on"]
    upper = (1500 * 0.075 + 30 * 0.30) / 1e6
    assert c["per_1k_calls_usd"]["upper"] == pytest.approx(upper * 1000)
    assert c["per_1k_calls_usd"]["lower"] < c["per_1k_calls_usd"]["upper"]  # the static prefix could be cached
    assert c["cacheable_prefix_tokens"] > 100 and c["calls"] == 50
    assert s["tokens"] == {"calls_with_reported_usage": 50, "calls_with_estimated_usage": 0, "calls_without_usage": 0}


def test_a_price_without_a_cached_rate_gives_the_same_lower_and_upper_cost(world):
    c = world.run(QWEN_27B, "dev", subset="D50").summary["cost"]
    assert c["price"]["id"] == "groq-qwen3.8-27b" and "cached_input" not in c["price"]["usd_per_mtok"]
    upper = (1500 * 0.80 + 30 * 4.00) / 1e6 * 1000  # no caching discount is assumed
    assert c["per_1k_calls_usd"]["upper"] == pytest.approx(upper)
    assert c["per_1k_calls_usd"]["lower"] == pytest.approx(upper)


def test_a_local_system_has_no_price_but_still_states_the_billing_basis(world):
    s = world.run(LOCAL_BASE, "dev", subset="D50").summary
    assert s["billing_basis"] == "free tier; priced at paid list price"
    assert "price" not in s["cost"] and "per_1k_calls_usd" not in s["cost"]
    assert "throughput benchmark" in s["cost"]["note"]
    assert s["endpoint"]["base_url"] == "http://127.0.0.1:8000/v1"


def test_the_request_is_built_from_the_chosen_prompt_with_train_only_examples(world):
    world.run(GPT_OSS_20B, "dev", subset="D50")
    first = world.bodies[0]
    assert first["model"] == "openai/gpt-oss-20b" and first["temperature"] == 0
    assert first["reasoning_effort"] == "low" and first["max_completion_tokens"] == 1024 and "max_tokens" not in first
    messages = first["messages"]
    assert [m["role"] for m in messages] == ["system"] + ["user", "assistant"] * 10 + ["user"]
    shots = [m["content"] for m in messages[1:-1:2]]
    assert len(shots) == 10 and set(shots) <= world.train_texts  # every retrieved example is a train example
    query = messages[-1]["content"]
    assert query not in world.train_texts and query not in shots
    all_eval_texts = {e.text for s in ("dev", "test") for e in world.splits[s]}
    assert not set(shots) & all_eval_texts  # no dev or test text ever appears as an example
    inventory = schema.load_inventory(world.processed)
    assert messages[0]["content"] == prompts.static_prefix("fewshot_k10_v1", inventory)
    assert len({json.dumps(b["messages"][0]) for b in world.bodies}) == 1  # identical static prefix first, for caching


def test_the_fine_tuned_prompt_is_two_messages_and_sends_no_schema(world):
    outcome = world.run(LOCAL_FT, "dev", subset="D50")
    body = world.bodies[0]
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == prompts.FINETUNED_INSTRUCTION
    assert "response_format" not in body and outcome.summary["decoding"]["strict_json_schema_sent"] is False


def test_the_strict_schema_is_sent_when_the_endpoint_is_configured_for_it(world):
    outcome = world.run(GPT_OSS_20B, "dev", subset="D50", limit=2)  # groq: all three models accepted it (check_free_tiers, 2026-10-02)
    fmt = world.bodies[0]["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["properties"]["intent"]["enum"] == list(schema.load_inventory(world.processed).intents)
    assert outcome.summary["decoding"]["strict_json_schema_sent"] is True


def test_the_strict_schema_is_not_sent_while_the_endpoint_is_unconfirmed(world):
    # an endpoint whose strict-schema support is not confirmed yet
    world.edit_systems("    api_key_env: GEMINI_API_KEY\n    supports_json_schema: true", "    api_key_env: GEMINI_API_KEY\n    supports_json_schema: false")
    outcome = world.run(GEMINI, "dev", subset="D50", limit=2)
    assert "response_format" not in world.bodies[0]
    assert outcome.summary["decoding"]["strict_json_schema_sent"] is False


def test_a_confirmed_endpoint_is_sent_the_schema(world):
    outcome = world.run(GEMINI, "dev", subset="D50", limit=2)  # gemini as shipped: confirmed on 2026-10-02
    assert world.bodies[0]["response_format"]["json_schema"]["strict"] is True
    assert outcome.summary["decoding"]["strict_json_schema_sent"] is True


# --- scoring --------------------------------------------------------------------------------------------


def test_bad_output_is_scored_as_failure_and_good_variants_are_accepted(world):
    outcome_ids = [e.id for e in data.read_examples(world.processed / "dev.jsonl")]
    d50 = json.loads((world.processed / "subsets.json").read_text())["subsets"]["D50"]["ids"]
    assert set(d50) <= set(outcome_ids)
    for i, mode in zip(d50[:20], ["prose"] * 4 + ["wrong_intent"] * 4 + ["extra_key"] * 4 + ["invented_value"] * 4 + ["fenced"] * 4, strict=True):
        world.corrupt[i] = mode
    m = world.run(GPT_OSS_20B, "dev", subset="D50").summary["metrics"]
    assert m["schema_valid_rate"]["value"] == pytest.approx(42 / 50)  # prose and extra_key are not valid; the rest are
    # prose and extra_key are invalid and wrong_intent is wrong; invented_value keeps the right intent
    assert m["intent_accuracy"]["value"] == pytest.approx((50 - 4 - 4 - 4) / 50)
    assert 0.0 < m["exact_match"]["value"] < 1.0
    # invented values are hallucinations: valid JSON, but the value is not in the request
    assert m["unfound_value_rate"]["value"] > 0.0
    assert m["slot_precision"]["value"] < 1.0 and m["slot_recall"]["value"] < 1.0


def test_failed_calls_count_against_the_system_and_are_reported(world):
    d50 = json.loads((world.processed / "subsets.json").read_text())["subsets"]["D50"]["ids"]
    world.status_for = {d50[0]: 400, d50[1]: 400}
    s = world.run(GPT_OSS_20B, "dev", subset="D50").summary
    assert s["status"] == "complete" and s["n_scored"] == 50 and s["n_calls_failed"] == 2
    assert s["metrics"]["schema_valid_rate"]["value"] == pytest.approx(48 / 50)
    assert s["calls"]["errors_by_kind"] == {"http_400": 2}


# --- prompts and options -----------------------------------------------------------------------------------


def test_zeroshot_is_allowed_on_dev_and_gets_its_own_run_directory(world):
    outcome = world.run(GPT_OSS_20B, "dev", subset="D50", prompt="zeroshot_v1")
    assert outcome.run_dir.name == f"{GPT_OSS_20B}__dev__zeroshot_v1"
    assert [m["role"] for m in world.bodies[0]["messages"]] == ["system", "user"]
    assert outcome.summary["prompt"]["name"] == "zeroshot_v1"
    world.bodies.clear()
    default = world.run(GPT_OSS_20B, "dev", subset="D50")
    assert default.run_dir.name == f"{GPT_OSS_20B}__dev"  # a separate directory, so nothing is mixed
    assert default.summary["config_hash"] != outcome.summary["config_hash"]


def test_prompt_rules(world):
    with pytest.raises(evaluate.EvalError, match="not allowed"):
        world.run(GPT_OSS_20B, "dev", subset="D50", prompt="finetuned_v1")
    world.lock(GPT_OSS_20B)
    with pytest.raises(evaluate.EvalError, match="not allowed"):
        world.run(GPT_OSS_20B, "test", prompt="zeroshot_v1")  # test always uses the system's own prompt
    assert world.bodies == []


def test_option_errors(world):
    with pytest.raises(evaluate.EvalError, match="--split"):
        world.run(GPT_OSS_20B, "train")
    with pytest.raises(evaluate.EvalError, match="--limit"):
        world.run(GPT_OSS_20B, "dev", subset="D50", limit=0)
    with pytest.raises(ValueError, match="test subset"):
        world.run(GPT_OSS_20B, "dev", subset="S300")  # a test subset cannot be used on dev
    with pytest.raises(config.ConfigError, match="unknown system"):
        world.run("nope", "dev")
    assert world.bodies == []


def test_limit_gives_a_partial_summary_never_a_final_one(world):
    outcome = world.run(GPT_OSS_20B, "dev", subset="D50", limit=7)
    assert outcome.status == "complete"  # the run itself finished what it was asked to do
    assert outcome.summary["status"] == "partial" and outcome.summary["n_scored"] == 7 and outcome.summary["limit"] == 7
    assert outcome.summary_path.name == "summary.D50.partial.json"
    assert not (outcome.run_dir / "summary.D50.json").exists()


# --- subsets share predictions ------------------------------------------------------------------------------


def test_the_nested_dev_subsets_never_spend_a_request_twice(world):
    world.run(GPT_OSS_20B, "dev", subset="D50")
    assert len(world.bodies) == 50
    second = world.run(GPT_OSS_20B, "dev", subset="D100", resume=True)
    assert len(world.bodies) == 100  # only the 50 new items were sent
    assert second.summary["n_scored"] == 100 and second.summary["subset"]["name"] == "D100"
    run_dir = world.results / "runs" / f"{GPT_OSS_20B}__dev"
    assert (run_dir / "summary.D50.json").exists() and (run_dir / "summary.D100.json").exists()
    ids = [r["id"] for r in read_rows(run_dir / "predictions.jsonl")]
    assert len(ids) == len(set(ids)) == 100


def test_rerunning_without_resume_refuses_to_overwrite(world):
    world.run(GPT_OSS_20B, "dev", subset="D50", limit=3)
    with pytest.raises(FileExistsError, match="--resume"):
        world.run(GPT_OSS_20B, "dev", subset="D50", limit=3)


def test_resume_refuses_predictions_made_under_a_different_configuration(world):
    world.run(GPT_OSS_20B, "dev", subset="D50", limit=3)
    world.edit_systems("    price_id: groq-gpt-oss-20b\n    params:\n      temperature: 0\n", "    price_id: groq-gpt-oss-20b\n    params:\n      temperature: 0.4\n")
    with pytest.raises(ConfigMismatch):
        world.run(GPT_OSS_20B, "dev", subset="D50", resume=True)


# --- the test lock ------------------------------------------------------------------------------------------------


def test_the_test_split_is_refused_before_any_test_data_is_read_or_any_request_sent(world, monkeypatch):
    read = []
    original = data.read_examples
    monkeypatch.setattr(data, "read_examples", lambda path: (read.append(path.name), original(path))[1])
    with pytest.raises(LockError, match="never been locked"):
        world.run(GPT_OSS_20B, "test")
    assert world.bodies == [] and "test.jsonl" not in read
    assert not (world.results / "runs").exists()


def test_a_locked_test_run_goes_through_and_records_the_lock(world):
    entry = world.lock(GEMINI, reason="GPT_OSS_20B rows lock before the fine-tune exists")
    # Raising the daily cap changes how fast a run goes, not what it measures, so the lock still holds.
    world.set_limits(GEMINI, rpd=5000)
    outcome = world.run(GEMINI, "test", with_ci=False)
    s = outcome.summary
    assert outcome.status == "complete" and s["subset"]["name"] == "S500" and s["subset"]["n"] == 500
    assert s["lock"] == {k: entry[k] for k in ("locked_at", "reason", "config_hash", "subset_hash")}
    assert len(read_rows(outcome.predictions_path)) == 500


def test_a_configuration_change_after_locking_blocks_the_test_run(world):
    world.lock(GPT_OSS_20B)
    world.edit_systems("model: openai/gpt-oss-20b\n", "model: openai/gpt-oss-20b-2099\n")
    with pytest.raises(LockError, match="changed since"):
        world.run(GPT_OSS_20B, "test")
    assert world.bodies == []


def test_a_lock_for_one_subset_does_not_open_another(world):
    world.lock(GPT_OSS_20B, "S300")  # S300 is pre-registered, so it can be locked although no row runs on it
    with pytest.raises(LockError, match="not for S500"):
        world.run(GPT_OSS_20B, "test", subset="S500")


# --- free tiers: daily caps, stop cleanly, resume --------------------------------------------------------------


def test_a_daily_cap_stops_cleanly_and_resume_carries_on_without_repeating_a_request(world):
    world.set_limits(GEMINI, rpm=10, rpd=200)  # 500 items at 200 a day: three days, the last one short
    world.lock(GEMINI)
    sent_per_day = []
    for day in range(1, 8):
        before = len(world.bodies)
        outcome = world.run(GEMINI, "test", resume=day > 1)
        sent_per_day.append(len(world.bodies) - before)
        if outcome.status == "complete":
            break
        assert outcome.status == "quota_exhausted"
        assert outcome.summary["status"] == "partial" and outcome.summary_path.name == "summary.S500.partial.json"
        assert outcome.summary["stop"]["reason"] == "quota_exhausted" and outcome.summary["stop"]["reset_at"]
        assert outcome.summary["n_scored"] == sum(sent_per_day)
        world.ft.now += 86_400 + 600  # the next day, once everything sent today has aged out
    assert sent_per_day == [200, 200, 100]  # 500 items at 200 a day
    assert outcome.summary["status"] == "complete" and outcome.summary["n_scored"] == 500
    assert outcome.summary_path.name == "summary.S500.json"
    assert not (outcome.run_dir / "summary.S500.partial.json").exists()  # the stale partial is gone
    ids = [r["id"] for r in read_rows(outcome.predictions_path)]
    assert len(ids) == len(set(ids)) == 500  # nothing was ever sent twice


def test_a_server_side_daily_limit_is_reported_with_its_reset_time(world):
    world.lock(GPT_OSS_20B)

    def limited(request):
        world.bodies.append(json.loads(request.content))
        return httpx.Response(429, headers={"Retry-After": "7200"}, text="daily limit")

    outcome = world.run(GPT_OSS_20B, "test", transport=httpx.MockTransport(limited))
    assert outcome.status == "quota_exhausted" and outcome.summary["n_scored"] == 0
    assert outcome.summary["metrics"] is None
    assert outcome.batch.reset_at == pytest.approx(world.ft.now + 7200)
    assert len(world.bodies) == 1  # one probe, then a clean stop instead of a retry loop


# --- spend guard -------------------------------------------------------------------------------------------------


def test_max_usd_refuses_a_run_projected_to_cost_more_before_sending_anything(world):
    with pytest.raises(BudgetExceeded):
        world.run(GEMINI, "dev", subset="D100", max_usd=0.00001)
    assert world.bodies == []


def test_max_usd_needs_a_price(world):
    with pytest.raises(ValueError, match="needs a price"):
        world.run(LOCAL_BASE, "dev", subset="D50", max_usd=1.0)


# --- the script -----------------------------------------------------------------------------------------------------


def run_script(world, argv, **extra):
    script = load_script("run_eval")
    lines: list[str] = []
    code = script.main(argv, out=lines.append, **world.kwargs(**extra))
    return code, "\n".join(lines)


def test_script_exit_zero_and_the_headline(world):
    code, text = run_script(world, ["--system", GPT_OSS_20B, "--split", "dev", "--subset", "D50"], with_ci=True, n_resamples=200)
    assert code == 0 and text.count("[") >= 2
    assert "COMPLETE: 50 done" in text and "exact match    1.000 [1.000, 1.000]" in text
    assert "per 1,000 calls" in text and "free tier; priced at paid list price" in text


def test_script_exit_codes(world):
    code, text = run_script(world, ["--system", GPT_OSS_20B, "--split", "test"])
    assert code == 3 and "locked" in text
    code, text = run_script(world, ["--system", "nope", "--split", "dev"])
    assert code == 2 and "unknown system" in text
    code, text = run_script(world, ["--system", GPT_OSS_20B, "--split", "dev", "--subset", "D50", "--max-usd", "0.00001"])
    assert code == 2 and "--max-usd" in text
    code, text = run_script(world, ["--system", GPT_OSS_20B, "--split", "dev", "--subset", "D50", "--limit", "2", "--resume"], transport=httpx.MockTransport(lambda r: httpx.Response(401, text="bad token")))
    assert code == 77 and "AUTH_ERROR" in text


def test_script_reports_the_reset_time_and_exit_75_on_a_daily_cap(world):
    world.set_limits(GEMINI, rpm=10, rpd=200)
    world.lock(GEMINI)
    code, text = run_script(world, ["--system", GEMINI, "--split", "test"])
    assert code == 75
    assert "QUOTA_EXHAUSTED: 200 done" in text and "quota resets at:" in text and "--resume" in text
    assert "progress saved to" in text
    code, text = run_script(world, ["--system", GEMINI, "--split", "test", "--resume"])
    assert code == 75 and "QUOTA_EXHAUSTED: 0 done" in text  # still inside the same day: nothing new can be sent


def test_script_with_a_missing_key_is_a_clear_error(world):
    code, text = run_script(world, ["--system", GPT_OSS_20B, "--split", "dev", "--subset", "D50"], environ={"UNRELATED": "x"})
    assert code == 2 and "GROQ_API_KEY" in text
    code, text = run_script(world, ["--system", GEMINI, "--split", "dev", "--subset", "D50"], environ={"GROQ_API_KEY": "g"})
    assert code == 2 and "GEMINI_API_KEY" in text  # a key for another endpoint does not do
