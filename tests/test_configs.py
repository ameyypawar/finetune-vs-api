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

LOCAL_ROWS = ["ft-qwen3-4b-lora", "base-qwen3-4b-k10"]
GROQ_ROWS = ["groq-gpt-oss-20b-k10", "groq-gpt-oss-120b-k10", "groq-qwen3.8-27b-k10"]
GEMINI_ROW = "gemini-3.8-flash-k10"
API_ROWS = [*GROQ_ROWS, GEMINI_ROW]


def spec(name):
    return config.resolve_system(name)


def test_the_six_planned_systems_exist_and_all_resolve():
    assert config.list_systems() == [*LOCAL_ROWS, *API_ROWS]
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


def test_the_groq_rows_count_reasoning_against_the_token_budgets():
    for name in GROQ_ROWS:
        row = spec(name)
        assert row["base_url"] == "https://api.groq.com/openai/v1" and row["api_key_env"] == "GROQ_API_KEY"
        assert row["prompt"] == "fewshot_k10_v1" and row["dev_prompts"] == ["zeroshot_v1"] and row["test_subset"] == "S500"
        assert row["supports_json_schema"] is True  # check_free_tiers, 2026-10-02: all three models accept the strict schema
        assert row["reasoning_in_completion"] is True  # completion_tokens, which the budgets use, include reasoning
        assert row["drop_params"] == []
        assert {k: row["limits"][k] for k in ("rpm", "rpd", "tpm", "tpd")} == {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000}
    for name, model in (("groq-gpt-oss-20b-k10", "openai/gpt-oss-20b"), ("groq-gpt-oss-120b-k10", "openai/gpt-oss-120b")):
        row = spec(name)  # the reasoning models: a low effort, and a budget for the reasoning as well as the JSON answer
        assert row["model"] == model
        assert row["params"] == {"temperature": 0, "reasoning_effort": "low", "max_completion_tokens": 1024}
    qwen = spec("groq-qwen3.8-27b-k10")  # instruct mode, no thinking: a short answer
    assert qwen["model"] == "qwen/qwen3.8-27b"
    assert qwen["params"] == {"temperature": 0, "reasoning_effort": "none", "max_completion_tokens": 256}


def test_the_gemini_row():
    row = spec(GEMINI_ROW)
    assert row["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert row["api_key_env"] == "GEMINI_API_KEY"
    assert row["model"] == "gemini-3.8-flash" and row["price_id"] == "google-gemini-3.8-flash"
    assert row["prompt"] == "fewshot_k10_v1" and row["dev_prompts"] == ["zeroshot_v1"] and row["test_subset"] == "S500"
    assert row["params"] == {"temperature": 0, "reasoning_effort": "low", "max_completion_tokens": 1024}
    assert row["supports_json_schema"] is False  # until check_free_tiers reports the strict schema accepted
    assert row["reasoning_in_completion"] is True and row["drop_params"] == []
    assert row["limits"] == {"rpm": 10, "rpd": 250}  # placeholders until the free-tier limits are confirmed


def test_every_api_row_runs_on_s500_and_s300_stays_pre_registered_but_unused():
    for name in API_ROWS:
        assert spec(name)["test_subset"] == "S500", name
    assert {spec(name)["test_subset"] for name in LOCAL_ROWS} == {"full"}
    assert not [name for name in config.list_systems() if spec(name)["test_subset"] == "S300"]
    s300 = json.loads((ROOT / "results" / "subsets.json").read_text())["subsets"]["S300"]  # committed with its hash
    assert (s300["split"], s300["n"], s300["parent"]) == ("test", 300, "S500") and len(s300["hash"]) == 64


def test_the_retired_github_models_route_is_gone():
    """GitHub Models was retired on 2026-07-30; on 2026-10-02 it answered a plain "OK" to every request."""
    assert set(SYSTEMS["endpoints"]) == {"local", "gemini", "groq"}
    assert not [name for name in SYSTEMS["systems"] if name.startswith("gh-")]
    assert not [price_id for price_id in SOURCES["prices"] if price_id.startswith("openai-")]
    assert set(SOURCES["free_tier_limits"]) == {"groq", "gemini"}
    assert "GITHUB_MODELS_TOKEN" not in (ROOT / ".env.example").read_text()


def test_every_api_row_has_a_price_entry():
    for name in config.list_systems():
        row = spec(name)
        if row["price_id"]:
            entry = SOURCES["prices"][row["price_id"]]
            Price.from_entry(entry)
            assert entry["model"] == row["model"], name  # the price is for the model the row sends
    assert [spec(name)["price_id"] for name in API_ROWS] == [
        "groq-gpt-oss-20b", "groq-gpt-oss-120b", "groq-qwen3.8-27b", "google-gemini-3.8-flash",
    ]
    assert [spec(name)["price_id"] for name in LOCAL_ROWS] == [None, None]
    # the Qwen page lists no cached-input price, so none is assumed and caching changes nothing for that row
    assert Price.from_entry(SOURCES["prices"]["groq-qwen3.8-27b"]).cached_input_per_mtok is None


PUBLISHED_LIMIT_KEYS = {"rpm": "requests_per_minute", "rpd": "requests_per_day", "tpm": "tokens_per_minute", "tpd": "tokens_per_day"}


def limit_mismatches(row, published):
    """(limit, the row's value, the published value) for each published limit that the row does not carry.

    A published value that is null is skipped: the Gemini limits are not transcribed yet, and its row holds
    placeholders until they are.
    """
    return [
        (short, row["limits"].get(short), published[long])
        for short, long in PUBLISHED_LIMIT_KEYS.items()
        if published.get(long) is not None and row["limits"].get(short) != published[long]
    ]


def test_row_limits_match_the_published_limits_in_sources():
    """systems.yaml and sources.yaml both carry the free-tier numbers; this keeps them from drifting."""
    for name in API_ROWS:
        row = spec(name)
        published = SOURCES["free_tier_limits"][row["endpoint"]]["models"][row["model"]]
        assert limit_mismatches(row, published) == [], name
    for name in GROQ_ROWS:  # every Groq number is published, so every one of them was compared
        published = SOURCES["free_tier_limits"]["groq"]["models"][spec(name)["model"]]
        assert all(published[long] is not None for long in PUBLISHED_LIMIT_KEYS.values()), name


def test_a_published_limit_that_is_null_is_skipped_and_one_that_is_set_is_compared():
    row = {"limits": {"rpm": 10, "rpd": 250}}
    assert limit_mismatches(row, {"requests_per_minute": None, "requests_per_day": None}) == []  # the Gemini placeholders
    assert limit_mismatches(row, {"requests_per_minute": 15, "requests_per_day": None}) == [("rpm", 10, 15)]
    assert limit_mismatches(row, {"requests_per_minute": 10, "requests_per_day": 250, "tokens_per_minute": 8000}) == [("tpm", None, 8000)]


# --- sources.yaml ------------------------------------------------------------------------------------------


def test_the_planned_prices():
    prices = {k: v["usd_per_mtok"] for k, v in SOURCES["prices"].items()}
    assert prices == {
        "groq-gpt-oss-20b": {"input": 0.075, "cached_input": 0.037, "output": 0.30},
        "groq-gpt-oss-120b": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
        "groq-qwen3.8-27b": {"input": 0.80, "output": 4.00},  # its page lists no cached-input price
        "google-gemini-3.8-flash": {"input": 0.75, "cached_input": 0.075, "output": 3.75},  # through 2026-12-31
    }


def iter_entries():
    yield from SOURCES["prices"].items()
    for provider, entry in SOURCES["free_tier_limits"].items():
        yield f"free_tier_limits.{provider}", entry
    yield from SOURCES["gpu_rental"].items()
    yield "openai_deprecations", SOURCES["openai_deprecations"]


#: The day each entry was read. The ones from 2026-10-02 were read from their pages that day; the gpt-oss-120b
#: price and the GPU price come from the project plan of 2026-10-01 (see the note at the top of sources.yaml).
RETRIEVED_ON = {
    "groq-gpt-oss-20b": "2026-10-02",
    "groq-gpt-oss-120b": "2026-10-01",
    "groq-qwen3.8-27b": "2026-10-02",
    "google-gemini-3.8-flash": "2026-10-02",
    "free_tier_limits.groq": "2026-10-02",
    "free_tier_limits.gemini": "2026-10-02",
    "aws-g4dn.xlarge": "2026-10-01",
    "openai_deprecations": "2026-10-01",
}


def test_every_entry_has_its_url_and_the_retrieval_date():
    entries = dict(iter_entries())
    assert set(entries) == set(RETRIEVED_ON)  # 4 prices, 2 free-tier pages, the GPU and the deprecations
    for key, entry in entries.items():
        assert re.fullmatch(r"https://\S+", entry["url"]), key
        assert entry["retrieved_on"] == RETRIEVED_ON[key], key


def test_the_specific_sources_are_the_ones_named_in_the_plan():
    urls = {k: v["url"] for k, v in iter_entries()}
    assert urls["groq-gpt-oss-20b"] == "https://console.groq.com/docs/model/openai/gpt-oss-20b"
    assert urls["groq-gpt-oss-120b"] == "https://console.groq.com/docs/model/openai/gpt-oss-120b"
    assert urls["groq-qwen3.8-27b"] == "https://console.groq.com/docs/model/qwen/qwen3.8-27b"
    assert urls["google-gemini-3.8-flash"] == "https://ai.google.dev/gemini-api/docs/pricing"
    assert urls["free_tier_limits.groq"] == "https://console.groq.com/docs/rate-limits"
    assert urls["free_tier_limits.gemini"] == "https://ai.google.dev/gemini-api/docs/rate-limits"
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


def test_env_example_has_only_the_two_key_names_and_no_values():
    lines = [ln for ln in (ROOT / ".env.example").read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    assert lines == ["GEMINI_API_KEY=", "GROQ_API_KEY="]


def test_every_key_a_system_reads_is_listed_in_env_example():
    listed = {ln.split("=")[0] for ln in (ROOT / ".env.example").read_text().splitlines() if "=" in ln and not ln.startswith("#")}
    needed = {e["api_key_env"] for e in SYSTEMS["endpoints"].values() if e["api_key_env"]}
    assert needed == {"GEMINI_API_KEY", "GROQ_API_KEY"} and needed <= listed


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


# --- the docs name the systems the configs define -------------------------------------------------------------------


def section(text, heading):
    """The body of the `## heading` section of a markdown page."""
    return text.split(f"## {heading}", 1)[1].split("\n## ", 1)[0]


def readme_table():
    """The "What will be compared" table of README.md: system name -> its cells."""
    body = section((ROOT / "README.md").read_text(), "What will be compared")
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in body.splitlines() if line.startswith("| `")]
    return {row[0].strip("`"): row for row in rows}


def test_the_readme_table_lists_the_systems_and_where_each_runs():
    table = readme_table()
    assert list(table) == config.list_systems()
    where = {"groq": "Groq, free tier", "gemini": "Google AI Studio, free tier"}
    for name in API_ROWS:
        row = spec(name)
        assert table[name][1:] == [row["model"], f"`{row['prompt']}`", where[row["endpoint"]]], name
    assert {table[name][3] for name in LOCAL_ROWS} == {"local server"}


def test_the_readme_example_command_runs_a_system_the_configs_define():
    systems = re.findall(r"scripts/run_eval\.py --system (\S+)", (ROOT / "README.md").read_text())
    assert systems and set(systems) <= set(config.list_systems())


def test_the_method_page_names_every_system_and_dates_the_change_of_plan():
    text = (ROOT / "docs" / "method.md").read_text()
    for name in config.list_systems():
        assert f"`{name}`" in section(text, "What is compared"), name
    changes = section(text, "Changes to the plan")
    assert "2026-10-02" in changes and "2026-07-30" in changes  # the day of the change, and the day GitHub Models was retired
    assert "https://github.blog/changelog/2026-06-16-github-models-is-no-longer-available-to-new-customers/" in changes
    for name in ("groq-gpt-oss-20b-k10", "groq-qwen3.8-27b-k10", "gemini-3.8-flash-k10"):
        assert f"`{name}`" in changes, name
    assert "Google may use content sent on its free tier to improve its products" in text  # and only MASSIVE text is sent


def test_nothing_but_the_change_of_plan_still_names_a_github_models_row():
    readme = (ROOT / "README.md").read_text()
    method = (ROOT / "docs" / "method.md").read_text()
    method_without_the_change = method.replace(section(method, "Changes to the plan"), "")
    for name, text in (("README.md", readme), ("docs/method.md", method_without_the_change)):
        assert "GitHub Models" not in text and "gh-gpt" not in text, name
