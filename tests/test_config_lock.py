"""The per-system test lock, config hashing, and git helpers."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import UTC, datetime

import pytest

from conftest import ROOT, build_processed, load_script
from finetune_vs_api import config, prompts
from finetune_vs_api.config import ConfigError, LockError

API = "groq-gpt-oss-20b-k10"  # test subset S500, price groq-gpt-oss-20b, no checkpoint block
OTHER_API = "groq-gpt-oss-120b-k10"  # the same endpoint, prompt and limits as API, with another model and price
LOCAL = "ft-qwen3-4b-lora"
API_ROWS = ["groq-gpt-oss-20b-k10", "groq-gpt-oss-120b-k10", "groq-qwen3.8-27b-k10", "gemini-3.5-flash-lite-k10"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A copy of configs/, a synthetic processed dir, and an empty lock file, all under tmp_path."""
    config_dir = tmp_path / "configs"
    shutil.copytree(ROOT / "configs", config_dir)
    monkeypatch.setattr(config, "git_commit", lambda cwd=None: None)
    monkeypatch.setattr(config, "git_dirty", lambda cwd=None: True)

    class Env:
        pass

    e = Env()
    e.config_dir = config_dir
    e.processed = build_processed(tmp_path)
    e.lock = tmp_path / "results" / "test_lock.jsonl"
    e.kw = {"config_dir": config_dir, "processed_dir": e.processed, "lock_path": e.lock}
    e.ckw = {"config_dir": config_dir, "processed_dir": e.processed}
    return e


def header(name):
    """The header line of an entry of systems.yaml (a row) or sources.yaml (a price), for `edit`."""
    return f"  {name}:"


def edit(env, filename, old, new, entry=None):
    """Replace the first `old` with `new` in this test's copy of a config file.

    The API rows are near-identical, so an edit that has to land in one of them names it with `entry`, the
    header line of its YAML block (see `header`), and is made inside that block only.
    """
    path = env.config_dir / filename
    text = path.read_text()
    start, end = 0, len(text)
    if entry is not None:
        block = re.search(rf"^{re.escape(entry)}\n(?:(?: {{3,}}.*)?\n)*", text, re.MULTILINE)  # the header, then what is indented under it
        assert block, f"{entry!r} not in {filename}"
        start, end = block.span()
    assert old in text[start:end], f"{old!r} not in {entry or filename}"
    path.write_text(text[:start] + text[start:end].replace(old, new, 1) + text[end:])


def lock(env, system=API, subset=None, reason="freeze before first test run", **extra):
    return config.write_test_lock(system, subset, reason, **env.kw, **extra)


def allowed(env, system=API, subset=None):
    return config.assert_test_allowed(system, subset, **env.kw)


# --- resolving a system --------------------------------------------------------------------------


def test_resolve_system_merges_the_endpoint_and_the_row(env):
    spec = config.resolve_system(API, env.config_dir)
    assert spec["base_url"] == "https://api.groq.com/openai/v1"
    assert spec["api_key_env"] == "GROQ_API_KEY"
    assert spec["model"] == "openai/gpt-oss-20b"
    assert spec["limits"]["rpm"] == 30 and spec["test_subset"] == "S500"
    assert spec["supports_json_schema"] is True and spec["reasoning_in_completion"] is True  # both from the endpoint
    gemini = config.resolve_system("gemini-3.5-flash-lite-k10", env.config_dir)  # another endpoint, merged the same way
    assert gemini["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert gemini["api_key_env"] == "GEMINI_API_KEY" and gemini["supports_json_schema"] is True


@pytest.mark.parametrize(
    ("system", "old", "new", "fragment"),
    [
        (API, "endpoint: groq", "endpoint: nowhere", "unknown endpoint"),
        (API, "prompt: fewshot_k10_v1", "prompt: nope_v9", "unknown prompt"),
        (API, "test_subset: S500", "test_subset: D100", "test_subset must be"),
        (API, "    price_id: groq-gpt-oss-20b\n", "", "price_id"),
        # a documented output cap is enforced against what the row asks for: max_completion_tokens here, max_tokens below
        (API, "    limits:\n", "    limits:\n      max_output_tokens: 512\n", "cap is 512"),
        (LOCAL, "    limits:\n", "    limits:\n      max_output_tokens: 128\n", "cap is 128"),
    ],
    ids=["unknown endpoint", "unknown prompt", "dev subset", "no price_id", "cap under max_completion_tokens", "cap under max_tokens"],
)
def test_bad_rows_are_rejected_with_a_clear_message(env, system, old, new, fragment):
    edit(env, "systems.yaml", old, new, entry=header(system))
    with pytest.raises(ConfigError, match=fragment):
        config.resolve_system(system, env.config_dir)


def test_unknown_system(env):
    with pytest.raises(ConfigError, match="unknown system"):
        config.resolve_system("nope", env.config_dir)


# --- what the hash covers --------------------------------------------------------------------------


def components(env, system=API):
    return config.config_components(system, **env.ckw)


def changed(before, after):
    return sorted(k for k in before if before[k] != after[k])


def test_hash_is_stable_and_covers_the_documented_parts(env):
    a, b = components(env), components(env)
    assert a == b
    assert set(a) == {"prompt", "schema", "system", "decoding", "checkpoint", "prices", "dataset"}
    assert config.config_hash(API, **env.ckw) == config.config_hash(API, **env.ckw)
    assert config.config_hash(API, **env.ckw) != config.config_hash(OTHER_API, **env.ckw)


def test_every_api_row_has_its_own_hash(env):
    """Two rows on one endpoint differ in model and price; the row on another endpoint differs in its endpoint too."""
    hashes = {name: config.config_hash(name, **env.ckw) for name in API_ROWS}
    assert len(set(hashes.values())) == len(API_ROWS)
    assert components(env, "gemini-3.5-flash-lite-k10")["system"] != components(env, API)["system"]


def test_changing_a_decoding_parameter_changes_only_decoding(env):
    before = components(env)
    edit(env, "systems.yaml", "      temperature: 0\n", "      temperature: 0.7\n", entry=header(API))
    assert changed(before, components(env)) == ["decoding"]


def test_changing_the_model_changes_the_system_component(env):
    before = components(env)
    edit(env, "systems.yaml", "model: openai/gpt-oss-20b", "model: openai/gpt-oss-20b-2099", entry=header(API))
    assert changed(before, components(env)) == ["system"]


def test_changing_a_price_changes_only_prices(env):
    before = components(env)
    edit(env, "sources.yaml", "      input: 0.075\n      cached_input: 0.037", "      input: 0.076\n      cached_input: 0.037", entry=header("groq-gpt-oss-20b"))
    assert changed(before, components(env)) == ["prices"]


def test_a_price_for_another_system_does_not_matter(env):
    before = components(env)
    edit(env, "sources.yaml", "      input: 0.15\n", "      input: 0.16\n", entry=header("groq-gpt-oss-120b"))  # the 120b price, not the 20b's
    assert components(env) == before


def test_changing_the_dataset_pin_changes_only_dataset(env):
    before = components(env)
    edit(env, "data.yaml", "sha256: 4cba5faa", "sha256: 4cba5fab")
    assert changed(before, components(env)) == ["dataset"]


def test_changing_the_prompt_text_changes_the_prompt_component(env, monkeypatch):
    before = components(env)
    monkeypatch.setattr(prompts, "_ZEROSHOT_TEMPLATE", prompts._ZEROSHOT_TEMPLATE + "\nBe careful.")
    assert changed(before, components(env)) == ["prompt"]


def test_changing_the_label_inventory_changes_the_schema_and_the_prompt(env):
    before = components(env)
    from conftest import ex
    from finetune_vs_api import data

    train = data.read_examples(env.processed / "train.jsonl")
    train.append(ex(77777, "train", "new_intent", "hello", [("new_slot", "hello")]))
    data.write_examples(env.processed / "train.jsonl", train)
    assert {"schema", "prompt"} <= set(changed(before, components(env)))


def test_the_checkpoint_is_part_of_the_hash(env):
    before = components(env, "base-qwen3-4b-k10")
    edit(env, "systems.yaml", "    checkpoint:\n      base_revision: cdbee75f17c01a7cc42f958dc650907174af0554", "    checkpoint:\n      base_revision: abc123")
    assert changed(before, components(env, "base-qwen3-4b-k10")) == ["checkpoint"]


def test_rate_limits_and_concurrency_are_not_part_of_the_hash(env):
    before = components(env)
    edit(env, "systems.yaml", "      rpm: 30\n      rpd: 1000\n      tpm: 8000\n      tpd: 200000", "      rpm: 12\n      rpd: 100\n      tpm: 4000\n      tpd: 100000\n      max_concurrency: 1", entry=header(API))
    assert components(env) == before


def test_max_input_tokens_is_part_of_the_hash_because_it_changes_what_is_sent(env):
    edit(env, "systems.yaml", "    limits:\n", "    limits:\n      max_input_tokens: 8000\n", entry=header(API))  # no row sets one now
    before = components(env)
    edit(env, "systems.yaml", "max_input_tokens: 8000", "max_input_tokens: 7000", entry=header(API))
    assert changed(before, components(env)) == ["system"]


# --- writing and checking a lock --------------------------------------------------------------------


def test_a_lock_records_hashes_reason_time_and_git_state(env):
    when = datetime(2026, 10, 2, 9, 30, tzinfo=UTC)
    entry = lock(env, now=when)
    assert entry["system"] == API and entry["subset"] == "S500"
    assert entry["config_hash"] == config.config_hash(API, **env.ckw)
    assert entry["components"] == components(env)
    assert entry["reason"] == "freeze before first test run"
    assert entry["locked_at"] == "2026-10-02T09:30:00Z"
    assert (entry["git_commit"], entry["git_dirty"]) == (None, True)  # a repo with no commits yet
    assert config.read_lock_history(env.lock) == [entry]


def test_the_lock_allows_exactly_what_was_locked(env):
    with pytest.raises(LockError, match="never been locked"):
        allowed(env)
    entry = lock(env)
    assert allowed(env) == entry
    assert allowed(env, subset="S500") == entry


def test_the_lock_is_per_system(env):
    lock(env, API)
    with pytest.raises(LockError, match="never been locked"):
        allowed(env, OTHER_API)
    lock(env, OTHER_API)
    edit(env, "systems.yaml", "      temperature: 0\n", "      temperature: 0.5\n", entry=header(OTHER_API))
    with pytest.raises(LockError, match="changed since"):
        allowed(env, OTHER_API)
    assert allowed(env, API)  # the other system's change does not touch this lock


def test_a_changed_configuration_is_refused_and_names_what_changed(env):
    lock(env)
    edit(env, "systems.yaml", "model: openai/gpt-oss-20b", "model: openai/gpt-oss-20b-2099", entry=header(API))
    with pytest.raises(LockError) as caught:
        allowed(env)
    assert "changed since it was locked" in str(caught.value)
    assert "system" in str(caught.value) and "lock_test.py --write" in str(caught.value)


def test_relocking_with_a_reason_allows_the_new_configuration(env):
    lock(env)
    edit(env, "systems.yaml", "model: openai/gpt-oss-20b", "model: openai/gpt-oss-20b-2099", entry=header(API))
    with pytest.raises(LockError):
        allowed(env)
    second = lock(env, reason="switched to the dated model id")
    assert allowed(env) == second
    assert len(config.read_lock_history(env.lock)) == 2


def test_reverting_a_change_makes_the_old_lock_valid_again(env):
    first = lock(env)
    edit(env, "systems.yaml", "model: openai/gpt-oss-20b", "model: openai/gpt-oss-20b-2099", entry=header(API))
    edit(env, "systems.yaml", "model: openai/gpt-oss-20b-2099", "model: openai/gpt-oss-20b", entry=header(API))
    assert allowed(env) == first


def test_a_lock_for_one_subset_does_not_cover_another(env):
    lock(env, API, "S500")
    with pytest.raises(LockError, match=r"locked for \['S500'\] but not for S300"):
        allowed(env, API, "S300")
    lock(env, API, "S300", reason="S300 is pre-registered, so a row can still be locked and run on it")
    assert allowed(env, API, "S300")["subset"] == "S300"


def test_a_change_in_the_subset_itself_is_refused(env):
    lock(env)
    path = env.processed / "subsets.json"
    doc = json.loads(path.read_text())
    doc["subsets"]["S500"]["hash"] = "0" * 64
    path.write_text(json.dumps(doc))
    with pytest.raises(LockError, match="subset S500 contents"):
        allowed(env)


def test_full_test_split_can_be_locked_and_is_hashed_from_subsets_json(env):
    entry = lock(env, API, "full")
    assert entry["subset_hash"] == json.loads((env.processed / "subsets.json").read_text())["full"]["test"]["hash"]


def test_dev_subsets_cannot_be_locked_for_the_test_split(env):
    with pytest.raises(LockError, match="dev subset"):
        lock(env, API, "D100")
    with pytest.raises(LockError, match="unknown subset"):
        lock(env, API, "S9000")


def test_a_reason_is_required(env):
    for reason in ("", "   ", None):
        with pytest.raises(LockError, match="reason"):
            config.write_test_lock(API, None, reason, **env.kw)
    assert not env.lock.exists()


def test_the_same_lock_cannot_be_written_twice(env):
    lock(env)
    with pytest.raises(LockError, match="already locked"):
        lock(env, reason="again")
    assert len(config.read_lock_history(env.lock)) == 1


def test_history_is_append_only(env):
    lock(env, API)
    before = env.lock.read_text()
    lock(env, OTHER_API)
    lock(env, API, "S300", reason="second subset")
    after = env.lock.read_text()
    assert after.startswith(before)
    assert len(after.splitlines()) == 3 and after.endswith("\n")


@pytest.mark.parametrize("system", API_ROWS)
def test_every_api_row_can_be_locked_as_shipped(env, system):
    assert config.lock_blockers(system, config_dir=env.config_dir) == []  # no checkpoint to pin, and a price entry exists
    entry = lock(env, system)
    assert (entry["system"], entry["subset"]) == (system, "S500")
    assert allowed(env, system) == entry


def test_an_incomplete_system_cannot_be_locked(env):
    blockers = {b.split(" ")[0] for b in config.lock_blockers(LOCAL, config_dir=env.config_dir)}
    assert {"checkpoint.adapter", "checkpoint.epoch"} <= blockers
    assert "checkpoint.base_revision" not in blockers  # pinned in configs/systems.yaml
    with pytest.raises(LockError, match="cannot lock"):
        lock(env, LOCAL, "full")
    assert not env.lock.exists()
    # unpin the base revision and it blocks too
    edit(env, "systems.yaml", "base_revision: cdbee75f17c01a7cc42f958dc650907174af0554 # Hugging Face commit", "base_revision: null # Hugging Face commit")
    assert "checkpoint.base_revision" in {b.split(" ")[0] for b in config.lock_blockers(LOCAL, config_dir=env.config_dir)}
    # fill the checkpoint in and it locks
    edit(env, "systems.yaml", "adapter: null # path or repo of the chosen adapter\n      epoch: null # which saved epoch won on dev\n      base_revision: null # Hugging Face commit of Qwen/Qwen3-4B-Instruct-2507", "adapter: kaggle/out/epoch-2\n      epoch: 2\n      base_revision: deadbeef")
    assert not config.lock_blockers(LOCAL, config_dir=env.config_dir)
    assert lock(env, LOCAL, "full")["system"] == LOCAL


def test_a_missing_price_entry_blocks_the_lock(env):
    edit(env, "systems.yaml", "price_id: groq-gpt-oss-20b", "price_id: no-such-price", entry=header(API))
    assert any("no-such-price" in b for b in config.lock_blockers(API, config_dir=env.config_dir))
    with pytest.raises(LockError, match="no-such-price"):
        lock(env)


def test_a_lock_needs_the_prepared_data(env, tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        config.write_test_lock(API, None, "x", config_dir=env.config_dir, processed_dir=tmp_path / "empty", lock_path=env.lock)


# --- the script -----------------------------------------------------------------------------------------


def test_lock_script_write_and_show(env):
    script = load_script("lock_test")
    lines: list[str] = []
    kw = {"config_dir": env.config_dir, "processed_dir": env.processed, "lock_path": env.lock, "out": lines.append}
    assert script.run(write=True, system=API, reason="go", **kw) == 0
    assert any("locked groq-gpt-oss-20b-k10 for S500" in line for line in lines)
    lines.clear()
    assert script.run(show=True, system=API, **kw) == 0
    text = "\n".join(lines)
    assert "LOCKED" in text and "NOT LOCKED" not in text and "history:" in text
    lines.clear()
    assert script.run(show=True, **kw) == 0
    assert "BLOCKED: checkpoint.adapter" in "\n".join(lines) and "NOT LOCKED" in "\n".join(lines)


def test_lock_script_errors(env):
    script = load_script("lock_test")
    lines: list[str] = []
    kw = {"config_dir": env.config_dir, "processed_dir": env.processed, "lock_path": env.lock, "out": lines.append}
    assert script.run(write=True, system=API, **kw) == 2  # no reason
    assert script.run(write=True, system="nope", reason="x", **kw) == 2
    assert script.run(write=True, system=LOCAL, reason="x", subset="full", **kw) == 2  # blocked
    assert script.run(**kw) == 2
    assert any("unknown system" in line for line in lines)


# --- git helpers -----------------------------------------------------------------------------------------


def git(path, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args], cwd=path, check=True, capture_output=True)


def test_git_commit_handles_a_repository_with_no_commits(tmp_path):
    git(tmp_path, "init", "-q")
    assert config.git_commit(tmp_path) is None
    (tmp_path / "f.txt").write_text("x")
    assert config.git_dirty(tmp_path) is True


def test_git_commit_returns_the_hash_once_there_is_one(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "f.txt").write_text("x")
    git(tmp_path, "add", "f.txt")
    git(tmp_path, "commit", "-q", "-m", "first")
    sha = config.git_commit(tmp_path)
    assert sha is not None and len(sha) == 40
    assert config.git_dirty(tmp_path) is False


def test_git_helpers_outside_a_repository(tmp_path):
    assert config.git_commit(tmp_path) is None
    assert config.git_dirty(tmp_path) is None
