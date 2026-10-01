"""The real configs/*.yaml: every system resolves, and the facts the plan gave are what is written."""

from __future__ import annotations

import json
import re

import pytest
import yaml

from conftest import ROOT
from finetune_vs_api import config, cost, prompts
from finetune_vs_api.cost import Price
from finetune_vs_api.subsets import SUBSET_NAMES

SOURCES = yaml.safe_load((ROOT / "configs" / "sources.yaml").read_text())
SYSTEMS = yaml.safe_load((ROOT / "configs" / "systems.yaml").read_text())


def spec(name):
    return config.resolve_system(name)


def test_the_five_planned_systems_exist_and_all_resolve():
    assert config.list_systems() == [
        "ft-qwen3-4b-lora", "base-qwen3-4b-k10", "gh-gpt-4.1-mini-k10", "gh-gpt-4.1-k10", "groq-gpt-oss-120b-k10",
    ]
    for name in config.list_systems():
        resolved = spec(name)
        assert resolved["prompt"] in prompts.PROMPTS
        assert set(resolved["dev_prompts"]) <= set(prompts.PROMPTS)
        assert resolved["test_subset"] in SUBSET_NAMES
        assert resolved["params"]["temperature"] == 0  # "temperature 0 where accepted"


def test_the_two_local_rows():
    ft, base = spec("ft-qwen3-4b-lora"), spec("base-qwen3-4b-k10")
    assert (ft["prompt"], base["prompt"]) == ("finetuned_v1", "fewshot_k10_v1")
    assert ft["price_id"] is None and base["price_id"] is None
    assert base["model"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert ft["api_key_env"] is None and ft["base_url"].startswith("http://127.0.0.1")
    assert ft["supports_json_schema"] is False  # the fine-tune's own format-following is what gets measured


def test_the_github_models_rows():
    mini, big = spec("gh-gpt-4.1-mini-k10"), spec("gh-gpt-4.1-k10")
    for row in (mini, big):
        assert row["base_url"] == "https://models.github.ai/inference"
        assert row["api_key_env"] == "GITHUB_MODELS_TOKEN"
        assert row["prompt"] == "fewshot_k10_v1" and row["dev_prompts"] == ["zeroshot_v1"]
        assert row["limits"]["max_concurrency"] == 2
        assert (row["limits"]["max_input_tokens"], row["limits"]["max_output_tokens"]) == (8000, 4000)
        assert row["params"]["temperature"] == 0 and row["drop_params"] == []
    assert (mini["model"], mini["tier"], mini["test_subset"]) == ("openai/gpt-4.1-mini", "low", "S500")
    assert (mini["limits"]["rpm"], mini["limits"]["rpd"]) == (15, 150)
    assert (big["model"], big["tier"], big["test_subset"]) == ("openai/gpt-4.1", "high", "S300")
    assert (big["limits"]["rpm"], big["limits"]["rpd"]) == (10, 50)


def test_the_groq_row_counts_reasoning_against_the_token_budgets():
    row = spec("groq-gpt-oss-120b-k10")
    assert row["base_url"] == "https://api.groq.com/openai/v1" and row["api_key_env"] == "GROQ_API_KEY"
    assert row["model"] == "openai/gpt-oss-120b" and row["prompt"] == "fewshot_k10_v1" and row["test_subset"] == "S500"
    assert row["params"]["reasoning_effort"] == "low"
    assert {k: row["limits"][k] for k in ("rpm", "rpd", "tpm", "tpd")} == {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000}
    assert row["reasoning_in_completion"] is True  # completion_tokens, which the budgets use, include reasoning


def test_every_api_row_has_a_price_entry():
    for name in config.list_systems():
        row = spec(name)
        if row["price_id"]:
            assert row["price_id"] in SOURCES["prices"], name
            Price.from_entry(SOURCES["prices"][row["price_id"]])
    assert [spec(n)["price_id"] for n in ("gh-gpt-4.1-mini-k10", "gh-gpt-4.1-k10", "groq-gpt-oss-120b-k10")] == [
        "openai-gpt-4.1-mini", "openai-gpt-4.1", "groq-gpt-oss-120b",
    ]


def test_row_limits_match_the_published_limits_in_sources():
    """systems.yaml and sources.yaml both carry the free-tier numbers; this keeps them from drifting."""
    github = SOURCES["free_tier_limits"]["github-models"]["tiers"]
    for name in ("gh-gpt-4.1-mini-k10", "gh-gpt-4.1-k10"):
        row, published = spec(name), github[spec(name)["tier"]]
        assert row["limits"]["rpm"] == published["requests_per_minute"]
        assert row["limits"]["rpd"] == published["requests_per_day"]
        assert row["limits"]["max_concurrency"] == published["concurrent_requests"]
        assert row["limits"]["max_input_tokens"] == published["max_input_tokens"]
        assert row["limits"]["max_output_tokens"] == published["max_output_tokens"]
    row = spec("groq-gpt-oss-120b-k10")
    published = SOURCES["free_tier_limits"]["groq"]["models"][row["model"]]
    assert (row["limits"]["rpm"], row["limits"]["rpd"]) == (published["requests_per_minute"], published["requests_per_day"])
    assert (row["limits"]["tpm"], row["limits"]["tpd"]) == (published["tokens_per_minute"], published["tokens_per_day"])


# --- sources.yaml ------------------------------------------------------------------------------------------


def test_the_planned_prices():
    prices = {k: v["usd_per_mtok"] for k, v in SOURCES["prices"].items()}
    assert prices == {
        "openai-gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
        "openai-gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
        "groq-gpt-oss-120b": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    }


def iter_entries():
    yield from SOURCES["prices"].items()
    for provider, entry in SOURCES["free_tier_limits"].items():
        yield f"free_tier_limits.{provider}", entry
    yield from SOURCES["gpu_rental"].items()
    yield "openai_deprecations", SOURCES["openai_deprecations"]


def test_every_entry_has_its_url_and_the_retrieval_date():
    entries = list(iter_entries())
    assert len(entries) == 3 + 2 + 1 + 1
    for key, entry in entries:
        assert re.fullmatch(r"https://\S+", entry["url"]), key
        assert entry["retrieved_on"] == "2026-10-01", key


def test_the_specific_sources_are_the_ones_named_in_the_plan():
    urls = {k: v["url"] for k, v in iter_entries()}
    assert urls["openai-gpt-4.1-mini"] == urls["openai-gpt-4.1"] == "https://developers.openai.com/api/docs/pricing"
    assert urls["groq-gpt-oss-120b"] == "https://console.groq.com/docs/model/openai/gpt-oss-120b"
    assert urls["free_tier_limits.groq"] == "https://console.groq.com/docs/rate-limits"
    assert urls["free_tier_limits.github-models"].startswith("https://docs.github.com/en/enterprise-cloud@latest/github-models/")
    assert urls["aws-g4dn.xlarge"] == "https://instances.vantage.sh/aws/ec2/g4dn.xlarge"
    assert urls["openai_deprecations"] == "https://developers.openai.com/api/docs/deprecations"


def test_observed_fields_are_empty_until_the_free_tiers_have_been_checked():
    for provider, entry in SOURCES["free_tier_limits"].items():
        assert entry["observed"] == {"checked_on": None, "models": None}, provider


def test_gpu_rental_and_deprecation_facts():
    g4 = SOURCES["gpu_rental"]["aws-g4dn.xlarge"]
    assert g4["usd_per_hour"] == {"on_demand": 0.526, "spot": 0.274} and "T4" in g4["gpu"]
    events = {(e["date"], e["relation"]): e["what"] for e in SOURCES["openai_deprecations"]["events"]}
    assert set(events) == {("2026-10-23", "on"), ("2026-10-31", "from"), ("2026-11-30", "on"), ("2027-01-06", "from")}
    for model in ("ft-gpt-3.5-turbo", "ft-gpt-4", "ft-gpt-4.1-nano-2025-04-14", "ft-babbage-002", "ft-davinci-002", "ft-o4-mini-2025-04-16"):
        assert model in events[("2026-10-23", "on")]
    assert "new fine-tuning jobs" in events[("2027-01-06", "from")]


def test_the_billing_basis_is_one_string_everywhere():
    assert SOURCES["billing_basis"] == cost.BILLING_BASIS == "free tier; priced at paid list price"


def test_dates_stay_strings_so_the_snapshot_serializes():
    json.dumps(SOURCES)


# --- the rest ------------------------------------------------------------------------------------------------------


def test_env_example_has_only_the_three_key_names_and_no_values():
    lines = [ln for ln in (ROOT / ".env.example").read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    assert lines == ["GITHUB_MODELS_TOKEN=", "GEMINI_API_KEY=", "GROQ_API_KEY="]


def test_every_key_a_system_reads_is_listed_in_env_example():
    listed = {ln.split("=")[0] for ln in (ROOT / ".env.example").read_text().splitlines() if "=" in ln and not ln.startswith("#")}
    needed = {e["api_key_env"] for e in SYSTEMS["endpoints"].values() if e["api_key_env"]}
    assert needed <= listed


def test_the_dataset_pin_is_set_and_matches_the_audit():
    data_cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    assert re.fullmatch(r"[0-9a-f]{64}", data_cfg["source"]["sha256"])
    assert data_cfg["source"]["url"].endswith("amazon-massive-dataset-1.1.tar.gz") and data_cfg["locale"] == "en-US"
    assert data_cfg["expected"] == {"train": 11514, "dev": 2033, "test": 2974, "intents": 60, "slot_types": 55}
    audit = ROOT / "results" / "data_audit.json"
    if audit.exists():
        assert json.loads(audit.read_text())["dataset"]["archive_sha256"] == data_cfg["source"]["sha256"]


def test_the_license_is_mit_with_the_authors_name():
    text = (ROOT / "LICENSE").read_text()
    assert text.startswith("MIT License") and "Copyright (c) 2026 Amey Pawar" in text


@pytest.mark.parametrize("path", ["configs/data.yaml", "configs/train.yaml", "configs/sources.yaml", "configs/systems.yaml", ".github/workflows/ci.yml"])
def test_yaml_files_parse(path):
    assert yaml.safe_load((ROOT / path).read_text())
