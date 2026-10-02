"""kaggle/serve/serve_eval_on_kaggle.py: its checks, its two modes and its fallbacks, with the GPU, the
installs and the servers replaced by stubs.

Both modes run the real finetune_vs_api.evaluate.run_eval and the real test lock against the synthetic
data of conftest.build_processed. A stub chat endpoint answers from the gold labels, so every number
in a summary has a known right answer. What cannot be tested here (vLLM, llama.cpp, a T4) is listed in
the report; the commands that would start them are read as data in tests/test_kaggle_serve_files.py.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import socket
import sys
import tarfile
import zipfile
from pathlib import Path

import httpx
import pytest

from conftest import ROOT, ex
from finetune_vs_api import config, data
from serving_helpers import BASE, BASE_MODEL, FT, REVISION, Lab, write_adapter

T4 = [{"name": "Tesla T4", "memory": "15360 MiB", "compute_capability": "7.5", "driver": "550"}]


@pytest.fixture(scope="module")
def sv():
    spec = importlib.util.spec_from_file_location("serve_eval_on_kaggle", ROOT / "kaggle" / "serve" / "serve_eval_on_kaggle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def lab(tmp_path, monkeypatch):
    return Lab(tmp_path, monkeypatch)


# --- stand-ins for what touches the machine -----------------------------------------------------------------------------


class Launcher:
    """Replaces launch_session: records the sessions started, and serves each one from the lab's stub endpoint."""

    def __init__(self, sv, lab, *, fail=None, modes=None, normal_first=None):
        self.sv, self.lab = sv, lab
        self.fail = dict(fail or {})  # path -> why it cannot start
        self.modes = dict(modes or {})  # path -> stub mode (garbage, empty)
        self.normal_first = dict(normal_first or {})
        self.started: list[tuple[str, str, tuple[str, ...]]] = []

    @contextlib.contextmanager
    def __call__(self, ctx, plan, revision, base_model):
        assert revision == REVISION and base_model == BASE_MODEL
        if plan.path in self.fail:
            raise self.sv.PathFailed(self.fail[plan.path])
        self.started.append((plan.path, plan.name, tuple(v.key for v in plan.variants)))
        handler = self.lab.make_handler(self.modes.get(plan.path, "normal"), normal_first=self.normal_first.get(plan.path, 0))
        yield self.sv.Endpoint(plan.base_url, httpx.MockTransport(handler))


class Bench:
    """Replaces run_bench: records what it was asked and writes a stand-in results/serving/T4.json."""

    def __init__(self, status="ok"):
        self.status, self.calls = status, []

    def __call__(self, ctx, plan, variant, *, prompts, out, serving, extra):
        lines = prompts.read_text().splitlines()
        self.calls.append({"path": plan.path, "model": variant.model, "system": variant.system, "out": out, "serving": dict(serving),
                           "extra": dict(extra), "n_prompts": len(lines), "first_prompt": json.loads(lines[0])})
        if self.status != "ok":
            return {"status": "failed", "reason": "openai could not be installed"}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"system": variant.system, "operating_point": {"concurrency": 8}}))
        return {"status": "ok", "file": f"results/serving/{out.name}", "operating_point": {"concurrency": 8}, "cost": None}


def run(sv, lab, mode, *argv, launcher=None, bench=None, **overrides):
    launcher = launcher or Launcher(sv, lab)
    bench = bench or Bench()
    code = sv.main(
        ["--mode", mode, "--input-root", str(lab.input), "--working", str(lab.working), "--scratch", str(lab.scratch), *argv],
        gpu_query=lambda: T4, launcher=launcher, prepare=lambda *a: None, bench=bench, eval_kwargs=lab.eval_kwargs(), **overrides,
    )
    return code, launcher, bench


def read_json(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


@pytest.fixture
def reads(monkeypatch):
    """The names of the data files read through data.read_examples."""
    names: list[str] = []
    original = data.read_examples
    monkeypatch.setattr(data, "read_examples", lambda path: (names.append(Path(path).name), original(path))[1])
    return names


# --- finding the snapshot and the adapters ---------------------------------------------------------------------------


def make_snapshot(root: Path) -> Path:
    package = root / "src" / "finetune_vs_api"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    return root


def test_the_snapshot_is_found_at_its_mount_and_otherwise_by_structure(sv, tmp_path):
    mount = tmp_path / "in" / sv.SNAPSHOT_SLUG
    make_snapshot(mount)
    assert sv.find_repo_root(tmp_path / "in", tmp_path / "x") == mount
    elsewhere = tmp_path / "other" / "datasets" / "someone" / "else" / "repo"
    make_snapshot(elsewhere)
    assert sv.find_repo_root(tmp_path / "other", tmp_path / "x") == elsewhere  # a different mount name still works
    lookalike = tmp_path / "bad"
    (lookalike / "lib" / "finetune_vs_api").mkdir(parents=True)
    (lookalike / "lib" / "finetune_vs_api" / "__init__.py").write_text("")
    with pytest.raises(sv.Refused, match="no repository snapshot under"):
        sv.find_repo_root(lookalike, tmp_path / "x")  # the package must sit in a src/ directory
    with pytest.raises(sv.Refused, match=sv.SNAPSHOT_SLUG):
        sv.find_repo_root(tmp_path / "missing", tmp_path / "x")


@pytest.mark.parametrize("kind", ["tar.gz", "tar", "zip"])
def test_a_tarball_in_the_dataset_is_unpacked_when_kaggle_did_not_do_it(sv, tmp_path, kind):
    staged = make_snapshot(tmp_path / "stage" / "finetune-vs-api")
    mount = tmp_path / "in" / sv.SNAPSHOT_SLUG
    mount.mkdir(parents=True)
    archive = mount / f"finetune-vs-api-snapshot.{kind}"
    if kind == "zip":
        with zipfile.ZipFile(archive, "w") as zf:
            zf.write(staged / "src" / "finetune_vs_api" / "__init__.py", "finetune-vs-api/src/finetune_vs_api/__init__.py")
    else:
        with tarfile.open(archive, "w:gz" if kind == "tar.gz" else "w") as tf:
            tf.add(staged, arcname="finetune-vs-api")  # a top-level folder, as `tar` of a directory makes
    root = sv.find_repo_root(tmp_path / "in", tmp_path / "unpacked")
    assert root == tmp_path / "unpacked" / "finetune-vs-api-snapshot" / "finetune-vs-api"
    assert (root / "src" / "finetune_vs_api" / "__init__.py").exists()


@pytest.mark.parametrize("kind", ["tar.gz", "zip"])
def test_an_archive_that_writes_outside_its_folder_is_refused(sv, tmp_path, kind):
    archive = tmp_path / f"evil.{kind}"
    if kind == "zip":
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("../escaped.txt", "x")
    else:
        with tarfile.open(archive, "w:gz") as tf:
            info = tarfile.TarInfo("../escaped.txt")
            info.size = 1
            tf.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(sv.Refused, match="outside the archive"):
        sv.extract_archive(archive, tmp_path / "dest")
    assert not (tmp_path / "escaped.txt").exists()


def test_adapters_are_found_by_structure_and_listed_by_epoch(sv, tmp_path):
    train = tmp_path / sv.TRAIN_KERNEL_SLUG
    for epoch in (2, 1, 10):
        write_adapter(train / "adapters" / f"epoch-{epoch}")
    write_adapter(train / "adapters" / "final")  # not an epoch directory
    (train / "adapters" / "epoch-3").mkdir()
    (train / "adapters" / "epoch-3" / "adapter_config.json").write_text("{}")  # no weights: not an adapter
    found = sv.find_adapters(tmp_path)
    assert list(found) == [1, 2, 10] and found[2] == train / "adapters" / "epoch-2"
    other = tmp_path / "elsewhere" / "kernels" / "x"
    write_adapter(other / "epoch-7")
    assert list(sv.find_adapters(tmp_path / "elsewhere")) == [7]  # without the expected mount, found anywhere
    assert sv.find_adapters(tmp_path / "nothing") == {}


GOOD = {"peft_type": "LORA", "r": 16, "use_dora": False, "lora_bias": False, "modules_to_save": None,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({}, None),
        ({"r": 8}, None),
        ({"r": 32}, "above --max-lora-rank 16"),
        ({"r": "16"}, "rank r is"),
        ({"peft_type": "IA3"}, "not LORA"),
        ({"use_dora": True}, "DoRA"),
        ({"lora_bias": True}, "lora_bias"),
        ({"modules_to_save": ["lm_head"]}, "modules_to_save"),
        ({"rank_pattern": {"q_proj": 8}}, "rank_pattern or alpha_pattern"),
        ({"alpha_pattern": {"q_proj": 8}, "rank_pattern": {}}, "rank_pattern or alpha_pattern"),
        ({"rank_pattern": {}, "alpha_pattern": {}}, None),
        ({"target_modules": ["q_proj", "lm_head"]}, "lm_head"),
        ({"target_modules": "all-linear"}, "all-linear"),
        ({"target_modules": None}, "not among the seven"),
    ],
)
def test_adapter_configs_vllm_cannot_serve_are_named(sv, change, fragment):
    problems = sv.adapter_problems({**GOOD, **change})
    if fragment is None:
        assert problems == []
    else:
        assert any(fragment in p for p in problems), problems


def test_reading_an_adapter_records_its_hash_and_refuses_an_unservable_one_with_status_4(sv, tmp_path):
    write_adapter(tmp_path / "a", payload=b"abc")
    info = sv.read_adapter(tmp_path / "a")
    assert info["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" and info["rank"] == 16
    write_adapter(tmp_path / "big", rank=64)
    with pytest.raises(sv.Refused, match="cannot be served") as caught:
        sv.read_adapter(tmp_path / "big")
    assert caught.value.code == sv.EXIT_ADAPTER == 4
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "adapter_config.json").write_text("{not json")
    with pytest.raises(sv.Refused, match="cannot read"):
        sv.read_adapter(tmp_path / "broken")


# --- the pinned revision ------------------------------------------------------------------------------------------------------


def test_the_base_revision_must_be_pinned_a_full_hash_and_agreed_by_every_record(sv, lab):
    repo = sv.Repo(lab.snapshot)
    assert sv.pin_revision(repo, {"base_revision": REVISION}) == REVISION
    assert sv.pin_revision(repo, None, [REVISION, REVISION]) == REVISION  # no train_log is fine; the config still pins it
    with pytest.raises(sv.Refused, match="train_log.json says"):
        sv.pin_revision(repo, {"base_revision": "f" * 40})
    with pytest.raises(sv.Refused, match="checkpoint.base_revision"):
        sv.pin_revision(repo, None, [REVISION, "f" * 40])
    lab.edit("train.yaml", f"revision: {REVISION}", "revision: main")
    with pytest.raises(sv.Refused, match="not a full 40-character commit hash"):
        sv.pin_revision(repo, None)
    lab.edit("train.yaml", "revision: main", "revision: null")
    with pytest.raises(sv.Refused, match="not pinned"):
        sv.pin_revision(repo, None)


# --- the test lock -----------------------------------------------------------------------------------------------------------


def test_nothing_is_locked_until_all_six_pairs_are(sv, lab):
    repo = sv.Repo(lab.snapshot)
    problems = sv.lock_problems(repo)
    assert any("checkpoint.adapter is not set" in p for p in problems)  # the fine-tune has no checkpoint yet
    lab.pin_checkpoint()
    problems = sv.lock_problems(repo)
    assert len(problems) == 6 and all("never been locked" in p for p in problems)
    assert {p.split("]")[0].split("[")[1] for p in problems} == {"full", "S500", "S300"}
    lab.lock_all(subsets=("full",))  # the default test subset only
    problems = sv.lock_problems(repo)
    assert len(problems) == 4 and not any("[full]" in p for p in problems)
    assert all("lock_test.py --write" in p and ("S500" in p or "S300" in p) for p in problems)
    lab.lock_all(subsets=("S500", "S300"))
    assert sv.lock_problems(repo) == []
    entries = sv.lock_entries(repo)
    assert set(entries) == {FT, BASE} and set(entries[FT]) == {"full", "S500", "S300"}
    assert entries[FT]["full"]["reason"] == "frozen before the test run" and len(entries[FT]["S500"]["config_hash"]) == 64


def test_a_configuration_change_after_locking_is_caught_for_that_row_only(sv, lab):
    lab.pin_checkpoint()
    lab.lock_all()
    repo = sv.Repo(lab.snapshot)
    assert sv.lock_problems(repo) == []
    lab.edit("systems.yaml", "temperature: 0\n      max_tokens: 256\n    drop_params: []\n    limits:\n      max_concurrency: 8",
             "temperature: 0.3\n      max_tokens: 256\n    drop_params: []\n    limits:\n      max_concurrency: 8")
    problems = sv.lock_problems(repo)
    assert len(problems) == 3 and all(p.startswith(f"{FT} [") and "changed since" in p and "(changed: decoding)" in p for p in problems), problems
    # the base row was not touched, so its three locks still hold


def test_the_lock_checks_the_snapshots_own_data_not_the_repository_it_was_started_from(sv, lab):
    lab.pin_checkpoint()
    lab.lock_all()
    doc = read_json(lab.processed / "subsets.json")
    doc["subsets"]["S300"]["hash"] = "0" * 64
    (lab.processed / "subsets.json").write_text(json.dumps(doc))
    problems = sv.lock_problems(sv.Repo(lab.snapshot))
    assert problems and all("[S300]" in p or "subset" in p for p in problems)
    (lab.processed / "subsets.json").unlink()
    assert any("subsets.json" in p or "make_subsets" in p for p in sv.lock_problems(sv.Repo(lab.snapshot)))  # a missing file is a problem, not a crash


# --- variants, configs and the choice ---------------------------------------------------------------------------------------------


def test_the_epoch_with_the_most_correct_dev_items_wins_and_a_tie_goes_to_the_earlier_one(sv):
    def score(epoch, correct):
        return {"epoch": epoch, "correct": correct, "variant": f"epoch-{epoch}"}

    assert sv.pick_best_epoch([score(1, 1900), score(2, 1950)])["epoch"] == 2
    assert sv.pick_best_epoch([score(1, 1950), score(2, 1900)])["epoch"] == 1
    assert sv.pick_best_epoch([score(2, 1950), score(1, 1950)])["epoch"] == 1  # a tie: the earlier epoch, whatever the order
    assert sv.pick_best_epoch([score(3, 5), score(1, 5), score(2, 5)])["epoch"] == 1
    assert sv.pick_best_epoch([{"epoch": None, "correct": 9999, "variant": "base"}, score(2, 1)])["epoch"] == 2  # the base row is not a candidate
    with pytest.raises(ValueError):
        sv.pick_best_epoch([{"epoch": None, "correct": 9, "variant": "base"}])
    assert "highest exact match" in sv.SELECTION_RULE and "earlier epoch" in sv.SELECTION_RULE


def test_a_variants_config_names_its_model_and_checkpoint_and_leaves_the_repository_config_alone(sv, lab, tmp_path):
    repo = sv.Repo(lab.snapshot)
    before = {p.name: p.read_bytes() for p in lab.config_dir.iterdir()}
    epoch1 = sv.Variant("epoch-1", FT, f"{FT}-epoch-1", Path("/x/epoch-1"), 1)
    out = sv.write_variant_config(repo, tmp_path / "derived" / "epoch-1", epoch1, REVISION)
    spec = config.resolve_system(FT, out)
    assert spec["model"] == f"{FT}-epoch-1" and spec["checkpoint"] == {"adapter": "adapters/epoch-1", "epoch": 1, "base_revision": REVISION}
    assert spec["base_url"] == config.resolve_system(FT, lab.config_dir)["base_url"] and spec["params"] == config.resolve_system(FT, lab.config_dir)["params"]
    assert not config.lock_blockers(FT, config_dir=out)  # a checkpoint that is complete
    assert config.resolve_system(BASE, out)["model"] == BASE_MODEL  # the other row is untouched
    for name in ("data.yaml", "sources.yaml", "train.yaml"):
        assert (out / name).read_bytes() == before[name]
    base = sv.write_variant_config(repo, tmp_path / "derived" / "base", sv.Variant("base", BASE, BASE_MODEL), REVISION)
    assert config.resolve_system(BASE, base)["checkpoint"] == {"base_revision": REVISION}
    assert {p.name: p.read_bytes() for p in lab.config_dir.iterdir()} == before
    # two epochs differ in the checkpoint component of the config hash, so their summaries are told apart
    epoch2 = sv.Variant("epoch-2", FT, f"{FT}-epoch-2", Path("/x/epoch-2"), 2)
    out2 = sv.write_variant_config(repo, tmp_path / "derived" / "epoch-2", epoch2, REVISION)
    c1 = config.config_components(FT, config_dir=out, processed_dir=lab.processed)
    c2 = config.config_components(FT, config_dir=out2, processed_dir=lab.processed)
    assert sorted(k for k in c1 if c1[k] != c2[k]) == ["checkpoint", "system"]


# --- is the output plausible? ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("text", [None, "", "   \n", "!!!!!!!!!!!!", "\n\n\n\n\n\n\n\n\n", "ababababab", "\ufffd" * 10, "{\"a\": 1} " * 30 + "{\"a\": 1}"])
def test_empty_repeated_and_noise_outputs_look_degenerate(sv, text):
    assert sv.looks_degenerate(text)


@pytest.mark.parametrize("text", ['{"intent":"alarm_set","slots":[]}', '{"intent":"play_music","slots":[{"type":"artist_name","value":"bob dylan"}]}', "sure", "!!", "ok ok"])
def test_real_answers_and_short_ones_do_not(sv, text):
    assert not sv.looks_degenerate(text)


def valid_json(text):
    try:
        obj = json.loads(text)
    except (TypeError, ValueError):
        return False
    return isinstance(obj, dict) and set(obj) == {"intent", "slots"}


def reply(kind, text, error=None):
    return {"kind": kind, "text": text, "error": error}


GOOD_TEXT = '{"intent":"alarm_set","slots":[]}'


def test_the_probe_passes_sane_answers_and_fails_nan_style_garbage(sv):
    sane = [reply("short", GOOD_TEXT)] * 6 + [reply("long", GOOD_TEXT)] * 2
    assert sv.assess_probe(sane, expect_schema=True, is_valid=valid_json)[0]
    garbage = [reply("short", "!" * 64)] * 6 + [reply("long", "!" * 64)] * 2
    ok, why = sv.assess_probe(garbage, expect_schema=False, is_valid=valid_json)
    assert not ok and "fp16 overflow" in why and "8 of 8" in why
    some = [reply("short", GOOD_TEXT)] * 5 + [reply("short", "")] * 3  # a minority degenerate is tolerated: 3 of 8 > 25% is not
    assert not sv.assess_probe(some, expect_schema=False, is_valid=valid_json)[0]
    assert sv.assess_probe([reply("short", GOOD_TEXT)] * 7 + [reply("short", "")], expect_schema=False, is_valid=valid_json)[0]


def test_a_fine_tune_that_does_not_answer_in_the_schema_fails_the_probe_but_a_base_model_need_not(sv):
    prose = [reply("short", "Sure, I will set that alarm for you.")] * 6 + [reply("long", GOOD_TEXT)] * 2
    ok, why = sv.assess_probe(prose, expect_schema=True, is_valid=valid_json)
    assert not ok and "not being applied" in why and "0 of 6" in why
    assert sv.assess_probe(prose, expect_schema=False, is_valid=valid_json)[0]  # the base model is measured, not gated, on format
    half = [reply("short", GOOD_TEXT)] * 3 + [reply("short", "prose")] * 3
    assert sv.assess_probe(half, expect_schema=True, is_valid=valid_json)[0]  # at least half is enough: this is a sanity check


def test_failed_probe_requests_fail_the_probe(sv):
    ok, why = sv.assess_probe([reply("short", None, "ReadTimeout: x"), reply("short", GOOD_TEXT)], expect_schema=False, is_valid=valid_json)
    assert not ok and "1 of 2 probe requests failed: ReadTimeout" in why
    assert not sv.assess_probe([], expect_schema=False, is_valid=valid_json)[0]


def summary_with(valid, n=100):
    return {"metrics": {"schema_valid_rate": {"value": valid}}, "n_scored": n}


def prediction_rows(n, *, text=GOOD_TEXT, finish="stop", failed=0, empty=0, length=0):
    out = {}
    for i in range(n):
        if i < failed:
            out[str(i)] = {"error": "x", "text": None}
        elif i < failed + empty:
            out[str(i)] = {"error": None, "text": "", "finish_reason": "stop"}
        elif i < failed + empty + length:
            out[str(i)] = {"error": None, "text": "!" * 30, "finish_reason": "length"}
        else:
            out[str(i)] = {"error": None, "text": text, "finish_reason": finish}
    return out


def test_a_finished_run_is_implausible_for_garbage_and_failures_but_not_for_wrong_answers(sv):
    ok = sv.assess_predictions(prediction_rows(100))
    assert ok["failed"] == ok["degenerate"] == ok["truncated"] == 0 and sv.implausible(ok, summary_with(1.0)) is None
    assert sv.implausible(sv.assess_predictions(prediction_rows(100, text='{"intent":"x","slots":[]}')), summary_with(1.0)) is None  # wrong, not broken
    assert sv.implausible(sv.assess_predictions(prediction_rows(100, empty=5)), summary_with(0.95)) is None  # 5% is the line
    assert "empty or repeated" in sv.implausible(sv.assess_predictions(prediction_rows(100, empty=6)), summary_with(0.94))
    assert "ran to the token limit" in sv.implausible(sv.assess_predictions(prediction_rows(100, length=21, empty=0)), summary_with(0.9))
    assert "calls failed" in sv.implausible(sv.assess_predictions(prediction_rows(100, failed=6)), summary_with(0.94))
    assert "valid schema JSON" in sv.implausible(sv.assess_predictions(prediction_rows(100)), summary_with(0.3))
    why = sv.implausible(sv.assess_predictions(prediction_rows(100, empty=50)), summary_with(0.5))
    assert "fp16 overflow or a broken serving path, not a model result" in why
    assert sv.assess_predictions({}) == {"n": 0, "failed": 0, "degenerate": 0, "truncated": 0, "failed_fraction": 0.0, "degenerate_fraction": 0.0, "truncated_fraction": 0.0}


# --- the session plans ---------------------------------------------------------------------------------------------------------------


def variants(sv):
    return [sv.Variant("epoch-1", FT, f"{FT}-epoch-1", Path("/in/adapters/epoch-1"), 1), sv.Variant("epoch-2", FT, f"{FT}-epoch-2", Path("/in/adapters/epoch-2"), 2),
            sv.Variant("base", BASE, BASE_MODEL)]


def plan(sv, path, vs=None):
    return sv.plan_sessions(path, vs or variants(sv), base_model=BASE_MODEL, revision=REVISION, host="127.0.0.1", port=8000, scratch=Path("/tmp/s"))


def test_vllm_with_lora_is_one_server_that_serves_the_base_and_every_adapter_by_name(sv):
    (session,) = plan(sv, sv.PATH_VLLM_LORA)
    assert [v.key for v in session.variants] == ["epoch-1", "epoch-2", "base"] and session.base_url == "http://127.0.0.1:8000/v1"
    argv = session.argv
    assert argv[:3] == ["/tmp/s/vllm-env/bin/vllm", "serve", BASE_MODEL] and argv[argv.index("--revision") + 1] == REVISION
    modules = argv[argv.index("--lora-modules") + 1:]
    assert modules == [f"{FT}-epoch-1=/in/adapters/epoch-1", f"{FT}-epoch-2=/in/adapters/epoch-2"]  # the names clients send
    assert [s.kind for s in session.prep] == ["vllm_env"]


def test_the_other_paths_serve_one_model_per_server_with_the_base_first_and_the_fine_tune_last(sv):
    ft = [sv.Variant("ft", FT, FT, Path("/in/adapters/epoch-2"), 2), sv.Variant("base", BASE, BASE_MODEL)]
    merged = plan(sv, sv.PATH_VLLM_MERGED, ft)
    assert [s.variants[0].key for s in merged] == ["base", "ft"] and [s.name for s in merged] == ["vllm-merged-base", "vllm-merged-ft"]
    base_argv, ft_argv = (s.argv for s in merged)
    assert base_argv[2] == BASE_MODEL and "--lora-modules" not in base_argv and base_argv[base_argv.index("--served-model-name") + 1] == BASE_MODEL
    assert ft_argv[2] == "/tmp/s/merged/ft" and ft_argv[ft_argv.index("--served-model-name") + 1] == FT and "--enable-lora" not in ft_argv
    assert [s.kind for s in merged[1].prep] == ["vllm_env", "base_snapshot", "merge"] and [s.kind for s in merged[0].prep] == ["vllm_env"]
    gguf = plan(sv, sv.PATH_LLAMACPP, ft)
    assert [s.variants[0].key for s in gguf] == ["base", "ft"]
    assert gguf[1].argv[0].endswith("llama-server") and gguf[1].argv[gguf[1].argv.index("-m") + 1] == "/tmp/s/gguf/ft.gguf"
    assert gguf[1].argv[gguf[1].argv.index("--alias") + 1] == FT and gguf[0].argv[gguf[0].argv.index("--alias") + 1] == BASE_MODEL
    assert [s.kind for s in gguf[0].prep] == ["llama_cpp", "base_snapshot", "gguf"] and [s.kind for s in gguf[1].prep] == ["llama_cpp", "base_snapshot", "merge", "gguf"]
    hf = plan(sv, sv.PATH_HF, ft)
    assert [s.name for s in hf] == ["hf-transformers-base", "hf-transformers-ft"] and [s.variants[0].key for s in hf] == ["base", "ft"]
    base_argv, ft_argv = (s.argv for s in hf)
    assert base_argv[1:3] == ["-m", "finetune_vs_api.hf_server"] and base_argv[base_argv.index("--model-dir") + 1] == BASE_MODEL
    assert base_argv[base_argv.index("--revision") + 1] == REVISION and base_argv[base_argv.index("--served-name") + 1] == BASE_MODEL
    assert ft_argv[ft_argv.index("--model-dir") + 1] == "/tmp/s/merged/ft" and ft_argv[ft_argv.index("--served-name") + 1] == FT and "--revision" not in ft_argv
    assert [s.kind for s in hf[0].prep] == [] and [s.kind for s in hf[1].prep] == ["base_snapshot", "merge"]  # the base needs no merge
    with pytest.raises(ValueError, match="unknown serving path"):
        plan(sv, "vllm-turbo")


def test_the_serving_paths_are_in_the_order_the_brief_gives_and_only_the_last_has_no_throughput(sv):
    assert sv.SERVING_PATHS == ("vllm-lora", "vllm-merged", "llamacpp-gguf", "hf-transformers")
    assert [sv.PATH_INFO[p]["throughput"] for p in sv.SERVING_PATHS] == [True, True, True, False]
    assert sv.PATH_INFO["llamacpp-gguf"]["quantization"] == "q8_0" and sv.PATH_INFO["vllm-lora"]["engine"] == "vllm"
    assert sv.paths_from(None) == sv.SERVING_PATHS and sv.paths_from("llamacpp-gguf") == ("llamacpp-gguf", "hf-transformers")
    with pytest.raises(sv.Refused):
        sv.paths_from("vllm-turbo")


# --- the fallback ladder ---------------------------------------------------------------------------------------------------------------


def test_the_ladder_stops_at_the_first_path_that_works_and_records_every_attempt(sv):
    tried = []

    def attempt(path):
        tried.append(path)
        if path in ("vllm-lora", "vllm-merged"):
            raise sv.PathFailed(f"{path} broke")
        return f"result of {path}"

    failures = []
    path, result, attempts = sv.run_ladder(sv.SERVING_PATHS, attempt, on_failure=lambda p, why: failures.append((p, why)))
    assert (path, result) == ("llamacpp-gguf", "result of llamacpp-gguf") and tried == ["vllm-lora", "vllm-merged", "llamacpp-gguf"]  # the HF path never ran
    assert attempts == [
        {"path": "vllm-lora", "status": "failed", "reason": "vllm-lora broke"},
        {"path": "vllm-merged", "status": "failed", "reason": "vllm-merged broke"},
        {"path": "llamacpp-gguf", "status": "ok", "reason": None},
    ]
    assert failures == [("vllm-lora", "vllm-lora broke"), ("vllm-merged", "vllm-merged broke")]


def test_a_command_that_fails_counts_as_a_path_failure_but_a_bug_does_not(sv):
    def failing(path):
        raise sv.CommandFailed("pip exited 1")

    with pytest.raises(sv.NoServingPath) as caught:
        sv.run_ladder(("a", "b"), failing)
    assert [a["status"] for a in caught.value.attempts] == ["failed", "failed"] and "a: pip exited 1" in str(caught.value)

    def buggy(path):
        raise KeyError("a bug, not a broken server")

    with pytest.raises(KeyError):
        sv.run_ladder(("a", "b"), buggy)  # not swallowed into a silent fallback


def test_the_first_path_working_leaves_no_failures(sv):
    path, result, attempts = sv.run_ladder(sv.SERVING_PATHS, lambda p: 42)
    assert (path, result, attempts) == ("vllm-lora", 42, [{"path": "vllm-lora", "status": "ok", "reason": None}])


# --- the server process ----------------------------------------------------------------------------------------------------------------

HEALTH_SERVER = """
import http.server, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404); self.end_headers()
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_a_server_is_waited_for_until_its_health_endpoint_answers_and_killed_afterwards(sv, tmp_path):
    port = free_port()
    server = sv.ManagedServer("stub", [sys.executable, "-c", HEALTH_SERVER, str(port)], env=None, log_path=tmp_path / "logs" / "stub.log",
                              health_url=f"http://127.0.0.1:{port}/health", poll_s=0.05, ready_timeout_s=30)
    with server as running:
        pid = running.proc.pid
        assert running.proc.poll() is None and httpx.get(f"http://127.0.0.1:{port}/health").status_code == 200
    assert server.proc is None
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # gone, not a zombie waiting to be reaped
    assert (tmp_path / "logs" / "stub.log").exists()


def test_a_server_that_dies_while_starting_is_reported_with_the_end_of_its_log(sv, tmp_path):
    server = sv.ManagedServer("dies", [sys.executable, "-c", "import sys; print('CUDA out of memory'); sys.exit(3)"], env=None,
                              log_path=tmp_path / "dies.log", health_url="http://127.0.0.1:9/health", poll_s=0.05, ready_timeout_s=30)
    with pytest.raises(sv.PathFailed, match=r"(?s)exited with status 3.*CUDA out of memory"):
        with server:
            pytest.fail("a dead server must not be entered")


def test_a_server_that_never_gets_ready_is_killed_and_reported(sv, tmp_path):
    server = sv.ManagedServer("slow", [sys.executable, "-c", "import time; time.sleep(60)"], env=None, log_path=tmp_path / "slow.log",
                              health_url="http://127.0.0.1:9/health", poll_s=0.05, ready_timeout_s=0.4)
    with pytest.raises(sv.PathFailed, match="not ready after"):
        with server:
            pytest.fail("a server that is not ready must not be entered")
    assert server.proc is None  # stopped: the process group was killed


def test_a_failing_command_raises_with_the_tail_of_its_log(sv, tmp_path):
    log = tmp_path / "install.log"
    sv.run_cmd([sys.executable, "-c", "print('fine')"], log_path=log)
    with pytest.raises(sv.CommandFailed, match=r"(?s)exited 1.*no matching distribution"):
        sv.run_cmd([sys.executable, "-c", "import sys; print('ERROR: no matching distribution'); sys.exit(1)"], log_path=log)
    assert "fine" in log.read_text() and "no matching distribution" in log.read_text()  # appended, never overwritten


# --- the two modes, end to end -----------------------------------------------------------------------------------------------------------


def wrong_on(lab, model, count):
    lab.mark_wrong(model, [e.id for e in lab.splits["dev"]][:count])


def test_dev_select_scores_every_epoch_and_the_base_row_and_picks_the_best(sv, lab, reads):
    wrong_on(lab, f"{FT}-epoch-1", 20)
    wrong_on(lab, f"{FT}-epoch-2", 8)
    code, launcher, bench = run(sv, lab, "dev-select")
    assert code == 0 and bench.calls == []  # no throughput in dev-select
    assert launcher.started == [("vllm-lora", "vllm-lora", ("epoch-1", "epoch-2", "base"))]  # one server, every adapter on it
    record = read_json(lab.working / "results" / "serving" / "dev_select.json")
    assert record["mode"] == "dev-select" and record["rule"] == sv.SELECTION_RULE
    by_variant = {s["variant"]: s for s in record["scores"]}
    assert set(by_variant) == {"epoch-1", "epoch-2", "base"} and all(s["n_scored"] == 216 for s in by_variant.values())  # the whole dev split
    assert (by_variant["epoch-1"]["correct"], by_variant["epoch-2"]["correct"], by_variant["base"]["correct"]) == (196, 208, 216)
    assert by_variant["epoch-2"]["exact_match"] == pytest.approx(208 / 216) and by_variant["epoch-1"]["epoch"] == 1 and by_variant["base"]["epoch"] is None
    for s in by_variant.values():
        assert {"intent_accuracy", "slot_f1", "schema_valid_rate", "n_calls_failed", "exact_match_ci95", "summary", "model", "system"} <= set(s)
        assert (lab.working / s["summary"]).exists() and s["summary"].startswith("results/dev-select/")
    assert record["chosen"] == {"epoch": 2, "variant": "epoch-2", "exact_match": pytest.approx(208 / 216), "correct": 208,
                                "adapter": "adapters/epoch-2", "adapter_sha256": lab.adapter_sha(2)}  # the base row is not a candidate, though it scored highest
    assert record["adapters"] == {"1": {"adapter": "adapters/epoch-1", "sha256": lab.adapter_sha(1), "rank": 16},
                                  "2": {"adapter": "adapters/epoch-2", "sha256": lab.adapter_sha(2), "rank": 16}}
    assert (record["base_model"], record["base_revision"], record["dev_split_items"]) == (BASE_MODEL, REVISION, 216)
    assert record["serving"]["path"] == "vllm-lora" and record["serving"]["attempts"] == [{"path": "vllm-lora", "status": "ok", "reason": None}]
    assert record["serving"]["engine"] == "vllm" and record["serving"]["dtype"] == "float16" and record["serving"]["vllm_pin"] == sv.VLLM_PINS
    assert record["gpu"]["name"] == "Tesla T4" and {"configs/systems.yaml", "data/processed/dev.jsonl"} <= set(record["snapshot_files"])
    assert not (lab.working / "attempts").exists()  # nothing failed, so nothing is kept apart from the results
    assert [p["variant"] for p in record["probes"]] == ["epoch-1", "epoch-2", "base"] and all(p["ok"] for p in record["probes"])
    assert "test.jsonl" not in reads  # dev-select never reads the test split


def test_dev_select_writes_a_results_tree_per_variant_with_the_dev_subsets_and_names_the_epoch_in_each_summary(sv, lab):
    run(sv, lab, "dev-select")
    for variant, system in (("epoch-1", FT), ("epoch-2", FT), ("base", BASE)):
        run_dir = lab.working / "results" / "dev-select" / variant / "runs" / f"{system}__dev"
        assert sorted(p.name for p in run_dir.glob("summary.*.json")) == ["summary.D100.json", "summary.D50.json", "summary.full.json"]
        assert len(rows(run_dir / "predictions.jsonl")) == 216  # the nested subsets sent nothing twice
        full = read_json(run_dir / "summary.full.json")
        assert full["status"] == "complete" and full["split"] == "dev" and full["lock"] is None and full["subset"]["n"] == 216
        assert read_json(run_dir / "summary.D100.json")["subset"]["n"] == 100 and read_json(run_dir / "summary.D50.json")["subset"]["n"] == 50
    e1 = read_json(lab.working / "results/dev-select/epoch-1/runs" / f"{FT}__dev" / "summary.full.json")
    e2 = read_json(lab.working / "results/dev-select/epoch-2/runs" / f"{FT}__dev" / "summary.full.json")
    assert (e1["endpoint"]["model_requested"], e2["endpoint"]["model_requested"]) == (f"{FT}-epoch-1", f"{FT}-epoch-2")
    assert e1["config_components"]["checkpoint"] != e2["config_components"]["checkpoint"]  # told apart by what they stand for
    assert e1["endpoint"]["models_returned"] == {f"{FT}-epoch-1": 216}


def test_a_tie_between_epochs_goes_to_the_earlier_one(sv, lab):
    wrong_on(lab, f"{FT}-epoch-1", 10)
    wrong_on(lab, f"{FT}-epoch-2", 10)
    assert run(sv, lab, "dev-select")[0] == 0
    chosen = read_json(lab.working / "results/serving/dev_select.json")["chosen"]
    assert chosen["epoch"] == 1 and chosen["correct"] == 206


def test_dev_select_refuses_without_a_pinned_revision_before_starting_anything(sv, lab):
    lab.edit("train.yaml", f"revision: {REVISION}", "revision: null")
    code, launcher, _ = run(sv, lab, "dev-select")
    assert code == sv.EXIT_REFUSED and launcher.started == [] and not (lab.working / "results").exists()


def test_dev_select_refuses_when_there_are_no_adapters(sv, lab):
    import shutil

    shutil.rmtree(lab.train_output / "adapters")
    code, launcher, _ = run(sv, lab, "dev-select")
    assert code == sv.EXIT_REFUSED and launcher.started == []


def test_dev_select_refuses_an_adapter_vllm_cannot_serve_with_status_4(sv, lab):
    write_adapter(lab.train_output / "adapters" / "epoch-2", rank=64)
    code, launcher, _ = run(sv, lab, "dev-select")
    assert code == sv.EXIT_ADAPTER and launcher.started == []


# --- test mode: the lock ----------------------------------------------------------------------------------------------------------------------


def lock_refusal(sv, lab, reads, capsys):
    code, launcher, bench = run(sv, lab, "test")
    out = capsys.readouterr().out
    assert code == sv.EXIT_LOCKED == 3 and "REFUSED" in out
    assert launcher.started == [] and bench.calls == [] and "test.jsonl" not in reads  # nothing started, no test data read
    assert not (lab.working / "results").exists() and not (lab.working / "attempts").exists()
    return out


def test_the_test_split_is_refused_when_nothing_is_locked(sv, lab, reads, capsys):
    lab.pin_checkpoint()
    out = lock_refusal(sv, lab, reads, capsys)
    assert "never been locked" in out and "lock_test.py --write" in out


def test_the_test_split_is_refused_until_the_checkpoint_is_filled_in(sv, lab, reads, capsys):
    assert "checkpoint.adapter is not set" in lock_refusal(sv, lab, reads, capsys)


def test_a_lock_for_the_full_split_alone_does_not_open_the_s500_and_s300_summaries(sv, lab, reads, capsys):
    lab.pin_checkpoint()
    lab.lock_all(subsets=("full",))
    out = lock_refusal(sv, lab, reads, capsys)
    assert "[S500]" in out and "[S300]" in out and "[full]" not in out


def test_one_row_unlocked_is_enough_to_refuse(sv, lab, reads, capsys):
    lab.pin_checkpoint()
    lab.lock_all(systems=(FT,))
    out = lock_refusal(sv, lab, reads, capsys)
    assert out.count(BASE) >= 3 and f"{FT} [" not in out


def test_a_configuration_change_after_locking_is_refused_and_says_what_changed(sv, lab, reads, capsys):
    lab.pin_checkpoint()
    lab.lock_all()
    lab.edit("systems.yaml", "model: Qwen/Qwen3-4B-Instruct-2507\n    prompt: fewshot_k10_v1", "model: Qwen/Qwen3-4B-Instruct-2507-x\n    prompt: fewshot_k10_v1")
    out = lock_refusal(sv, lab, reads, capsys)
    assert "changed since it was locked" in out and "system" in out


def test_a_different_snapshot_of_the_data_is_refused(sv, lab, reads, capsys):
    lab.pin_checkpoint()
    lab.lock_all()
    train = data.read_examples(lab.processed / "train.jsonl")
    train.append(ex(77777, "train", "new_intent", "hello", [("new_slot", "hello")]))  # a new label: the schema and the prompt move
    data.write_examples(lab.processed / "train.jsonl", train)
    out = lock_refusal(sv, lab, reads, capsys)
    assert "changed since it was locked" in out and "schema" in out and "prompt" in out


def test_the_lock_is_the_first_thing_test_mode_checks(sv, lab, reads, capsys):
    import shutil

    shutil.rmtree(lab.train_output)  # no adapters either: the lock is still what refuses, because it is checked first
    out = lock_refusal(sv, lab, reads, capsys)
    assert "never been locked" in out and "no adapters" not in out


# --- test mode: the run -------------------------------------------------------------------------------------------------------------------------


def locked_lab(lab, epoch=2):
    lab.pin_checkpoint(epoch)
    lab.lock_all()
    return lab


def test_a_locked_test_run_runs_both_rows_on_the_full_split_with_all_three_summaries_and_then_the_sweep(sv, lab):
    locked_lab(lab)
    code, launcher, bench = run(sv, lab, "test")
    assert code == 0
    assert launcher.started == [("vllm-lora", "vllm-lora", ("base", "ft"))]  # one server, the fine-tuned row last
    asked = {}
    for body in lab.bodies:
        asked[body["model"]] = asked.get(body["model"], 0) + 1
    assert set(asked) == {FT, BASE_MODEL} and asked[FT] == 720 + 6 + 2 and asked[BASE_MODEL] == 720 + 6 + 2  # the full split once, plus the probe
    for system, model in ((FT, FT), (BASE, BASE_MODEL)):
        run_dir = lab.working / "results" / "runs" / f"{system}__test"
        assert sorted(p.name for p in run_dir.glob("summary.*.json")) == ["summary.S300.json", "summary.S500.json", "summary.full.json"]
        assert len(rows(run_dir / "predictions.jsonl")) == 720  # S500 and S300 sent nothing again
        for subset, n in (("full", 720), ("S500", 500), ("S300", 300)):
            summary = read_json(run_dir / f"summary.{subset}.json")
            assert (summary["status"], summary["split"], summary["subset"]["name"], summary["subset"]["n"], summary["n_scored"]) == ("complete", "test", subset, n, n)
            assert summary["lock"]["reason"] == "frozen before the test run" and summary["endpoint"]["model_requested"] == model
            assert summary["metrics"]["exact_match"]["value"] == 1.0 and summary["endpoint"]["models_returned"] == {model: n}
    # the summaries carry the lock: the same configuration hash the lock holds
    lock_entry = config.lock_status(FT, "S500", config_dir=lab.config_dir, processed_dir=lab.processed, lock_path=lab.lock_path)[0]
    assert read_json(lab.working / "results/runs" / f"{FT}__test/summary.S500.json")["config_hash"] == lock_entry["config_hash"]
    # the sweep: once, on the fine-tuned row, with the test requests as prompts, after both rows were run
    (call,) = bench.calls
    assert (call["path"], call["model"], call["system"]) == ("vllm-lora", FT, FT) and call["n_prompts"] == 720
    assert [m["role"] for m in call["first_prompt"]["messages"]] == ["system", "user"]  # the fine-tuned prompt: instruction and request
    assert call["serving"] == {"engine": "vllm", "engine_version": None, "dtype": "float16"}  # the version comes from the real venv
    assert call["extra"]["serving_path"] == "vllm-lora" and call["extra"]["adapter_epoch"] == 2 and call["extra"]["base_revision"] == REVISION
    assert call["extra"]["adapter_sha256"] == lab.adapter_sha(2) and call["extra"]["quantization"] is None
    assert call["out"].name == "T4.json" and (lab.working / "results" / "serving" / "T4.json").exists()


def test_the_test_run_record_names_the_locks_the_adapter_and_the_path(sv, lab):
    locked_lab(lab)
    lab.write_dev_select(chosen=2)
    run(sv, lab, "test")
    record = read_json(lab.working / "test_run.json")
    assert record["mode"] == "test" and set(record["locks"]) == {FT, BASE} and set(record["locks"][FT]) == {"full", "S500", "S300"}
    assert record["locks"][BASE]["S300"]["reason"] == "frozen before the test run" and len(record["locks"][BASE]["full"]["subset_hash"]) == 64
    assert record["adapter"] == {"path": "adapters/epoch-2", "epoch": 2, "sha256": lab.adapter_sha(2),
                                 "dev_select": {"record": "results/serving/dev_select.json", "chosen_epoch": 2, "locked_epoch": 2, "sha256_matches": True}}
    assert record["serving"]["path"] == "vllm-lora" and record["serving"]["throughput_measured"] is True
    assert record["throughput"]["status"] == "ok" and record["throughput"]["operating_point"] == {"concurrency": 8}
    assert record["runs"][FT]["S500"]["n_scored"] == 500 and record["runs"][BASE]["full"]["summary"] == f"results/runs/{BASE}__test/summary.full.json"
    assert {"configs/systems.yaml", "data/processed/test.jsonl", "results/test_lock.jsonl"} <= set(record["snapshot_files"])
    assert (record["base_model"], record["base_revision"]) == (BASE_MODEL, REVISION)


def test_the_locked_adapter_is_the_one_served_and_the_fine_tune_is_asked_for_by_its_locked_model_name(sv, lab):
    locked_lab(lab, epoch=1)
    seen = {}

    class Spy(Launcher):
        @contextlib.contextmanager
        def __call__(self, ctx, plan, revision, base_model):
            with super().__call__(ctx, plan, revision, base_model) as endpoint:
                seen["modules"] = plan.argv[plan.argv.index("--lora-modules") + 1:]
                yield endpoint

    run(sv, lab, "test", launcher=Spy(sv, lab))
    assert seen["modules"] == [f"{FT}={lab.train_output / 'adapters' / 'epoch-1'}"]  # only the locked adapter, under the locked name


def test_the_test_run_refuses_an_adapter_that_is_not_where_the_lock_says_with_status_4(sv, lab):
    locked_lab(lab, epoch=3)  # locked for an epoch that was never trained
    code, launcher, bench = run(sv, lab, "test")
    assert code == sv.EXIT_ADAPTER and launcher.started == [] and bench.calls == []


def test_the_adapter_must_be_the_file_dev_select_scored_and_a_different_epoch_only_warns(sv, lab, capsys):
    locked_lab(lab)
    lab.write_dev_select(chosen=2, adapters={"1": {"sha256": lab.adapter_sha(1)}, "2": {"sha256": "0" * 64}})
    code, launcher, _ = run(sv, lab, "test")
    out = capsys.readouterr().out
    assert code == sv.EXIT_ADAPTER and "is not the file dev-select scored" in out and "retrained" in out and launcher.started == []
    lab.write_dev_select(chosen=1)  # the rule chose epoch 1, the lock is for 2: allowed, loudly
    code, launcher, _ = run(sv, lab, "test")
    assert code == 0 and "warning: the lock is for epoch 2 but dev-select's rule chose epoch 1" in capsys.readouterr().out
    assert read_json(lab.working / "test_run.json")["adapter"]["dev_select"]["chosen_epoch"] == 1


def test_without_a_dev_select_record_in_the_snapshot_the_run_goes_on_and_says_so(sv, lab, capsys):
    locked_lab(lab)
    assert run(sv, lab, "test")[0] == 0
    assert "dev_select.json is not in the snapshot" in capsys.readouterr().out
    assert read_json(lab.working / "test_run.json")["adapter"]["dev_select"] is None


def test_every_record_must_agree_on_the_base_revision(sv, lab):
    lab.pin_checkpoint(revision="f" * 40)  # systems.yaml pins another commit than configs/train.yaml
    lab.lock_all()
    code, launcher, _ = run(sv, lab, "test")
    assert code == sv.EXIT_REFUSED and launcher.started == []


def test_a_failed_throughput_sweep_is_reported_and_does_not_undo_the_accuracy_results(sv, lab, capsys):
    locked_lab(lab)
    code, _, bench = run(sv, lab, "test", bench=Bench(status="failed"))
    assert code == 0 and len(bench.calls) == 1
    record = read_json(lab.working / "test_run.json")
    assert record["throughput"] == {"status": "failed", "reason": "openai could not be installed"}
    assert "WARNING: the throughput benchmark failed" in capsys.readouterr().out
    assert (lab.working / "results/runs" / f"{FT}__test/summary.full.json").exists()
    assert not (lab.working / "results" / "serving" / "T4.json").exists()  # no figure rather than a made-up one


# --- the fallbacks, end to end -------------------------------------------------------------------------------------------------------------------


def test_when_vllm_with_lora_cannot_start_the_adapter_is_merged_and_served_without_lora(sv, lab):
    code, launcher, _ = run(sv, lab, "dev-select", launcher=Launcher(sv, lab, fail={"vllm-lora": "triton LoRA kernel does not support sm75"}))
    assert code == 0
    assert launcher.started == [("vllm-merged", "vllm-merged-base", ("base",)), ("vllm-merged", "vllm-merged-epoch-1", ("epoch-1",)),
                                ("vllm-merged", "vllm-merged-epoch-2", ("epoch-2",))]
    record = read_json(lab.working / "results/serving/dev_select.json")
    assert record["serving"]["path"] == "vllm-merged" and record["serving"]["weights"].startswith("the adapter merged")
    assert record["serving"]["attempts"] == [
        {"path": "vllm-lora", "status": "failed", "reason": "triton LoRA kernel does not support sm75"},
        {"path": "vllm-merged", "status": "ok", "reason": None},
    ]
    assert record["chosen"]["epoch"] in (1, 2) and len(record["scores"]) == 3


def test_nan_style_garbage_at_the_probe_falls_back_instead_of_being_reported(sv, lab):
    launcher = Launcher(sv, lab, modes={"vllm-lora": "garbage"})
    code, launcher, _ = run(sv, lab, "dev-select", launcher=launcher)
    assert code == 0
    record = read_json(lab.working / "results/serving/dev_select.json")
    first = record["serving"]["attempts"][0]
    assert first["path"] == "vllm-lora" and first["status"] == "failed" and "probe of" in first["reason"] and "fp16 overflow" in first["reason"]
    assert record["serving"]["path"] == "vllm-merged"
    assert all(s["exact_match"] > 0.9 for s in record["scores"])  # the numbers are from the path that worked, not the garbage
    assert not (lab.working / "attempts" / "vllm-lora" / "results").exists()  # no run was even started on the broken path


def test_garbage_that_only_shows_up_after_the_probe_discards_the_run_and_falls_back(sv, lab):
    probes = 3 * 8  # three variants on the one server, eight probe requests each
    launcher = Launcher(sv, lab, modes={"vllm-lora": "garbage"}, normal_first={"vllm-lora": probes})
    code, _, _ = run(sv, lab, "dev-select", launcher=launcher)
    assert code == 0
    record = read_json(lab.working / "results/serving/dev_select.json")
    first = record["serving"]["attempts"][0]
    assert first["status"] == "failed" and "answers are empty or repeated characters" in first["reason"] and "not a model result" in first["reason"]
    assert record["serving"]["path"] == "vllm-merged"
    assert all(s["exact_match"] > 0.9 for s in record["scores"])
    # what the broken path produced is kept for inspection, but is not part of the results
    kept = list((lab.working / "attempts" / "vllm-lora" / "results").rglob("predictions.jsonl"))
    assert kept and any('"!!!' in line for line in kept[0].read_text().splitlines())
    final = list((lab.working / "results").rglob("predictions.jsonl"))
    assert final and not any('"!!!' in p.read_text() for p in final)


def test_the_test_run_falls_back_the_same_way_and_its_summaries_come_from_the_path_that_worked(sv, lab):
    locked_lab(lab)
    launcher = Launcher(sv, lab, fail={"vllm-lora": "no", "vllm-merged": "no either"})
    code, launcher, bench = run(sv, lab, "test", launcher=launcher)
    assert code == 0
    record = read_json(lab.working / "test_run.json")
    assert [a["path"] for a in record["serving"]["attempts"]] == ["vllm-lora", "vllm-merged", "llamacpp-gguf"]
    assert record["serving"]["path"] == "llamacpp-gguf" and record["serving"]["quantization"] == "q8_0"
    assert [s[1] for s in launcher.started] == ["llamacpp-gguf-base", "llamacpp-gguf-ft"]
    (call,) = bench.calls  # a serving path with throughput: the sweep runs on it
    assert call["path"] == "llamacpp-gguf" and call["extra"]["quantization"] == "q8_0" and call["serving"]["engine"] == "llama.cpp"
    assert call["serving"]["engine_version"] == sv.LLAMA_CPP_TAG and call["serving"]["dtype"] == "q8_0"
    summary = read_json(lab.working / "results/runs" / f"{FT}__test/summary.full.json")
    assert summary["lock"] is not None and summary["n_scored"] == 720


def test_the_last_resort_gives_accuracy_only_with_no_throughput_and_no_cost_figure(sv, lab, capsys):
    locked_lab(lab)
    fail = {"vllm-lora": "a", "vllm-merged": "b", "llamacpp-gguf": "c"}
    code, launcher, bench = run(sv, lab, "test", launcher=Launcher(sv, lab, fail=fail))
    assert code == 0 and bench.calls == []  # the sweep is never run on this path
    assert [s[1] for s in launcher.started if s[0] == "hf-transformers"] == ["hf-transformers-base", "hf-transformers-ft"]
    record = read_json(lab.working / "test_run.json")
    assert record["serving"]["path"] == "hf-transformers" and record["serving"]["throughput_measured"] is False
    assert record["throughput"]["status"] == "skipped" and "no throughput and no cost figure" in record["throughput"]["reason"]
    assert not (lab.working / "results" / "serving" / "T4.json").exists()
    assert [a["status"] for a in record["serving"]["attempts"]] == ["failed", "failed", "failed", "ok"]
    assert (lab.working / "results/runs" / f"{FT}__test/summary.S300.json").exists()  # the accuracy numbers are all there


def test_when_no_path_works_nothing_is_reported_and_the_attempts_are_written_down(sv, lab, capsys):
    fail = {p: f"{p} cannot start" for p in sv.SERVING_PATHS}
    code, launcher, _ = run(sv, lab, "dev-select", launcher=Launcher(sv, lab, fail=fail))
    assert code == sv.EXIT_NO_PATH == 5
    assert not (lab.working / "results").exists()
    failed = read_json(lab.working / "attempts" / "failed.json")
    assert [a["path"] for a in failed["attempts"]] == list(sv.SERVING_PATHS) and all(a["status"] == "failed" for a in failed["attempts"])
    assert "FAILED: no serving path worked" in capsys.readouterr().out


def test_a_bug_in_a_path_is_not_mistaken_for_a_broken_server(sv, lab):
    class Buggy(Launcher):
        @contextlib.contextmanager
        def __call__(self, ctx, plan, revision, base_model):
            raise RuntimeError("a bug in the launcher")
            yield  # pragma: no cover

    with pytest.raises(RuntimeError, match="a bug"):
        run(sv, lab, "dev-select", launcher=Buggy(sv, lab))


def test_start_at_skips_the_paths_before_it(sv, lab):
    code, launcher, _ = run(sv, lab, "dev-select", "--start-at", "llamacpp-gguf")
    assert code == 0 and {s[0] for s in launcher.started} == {"llamacpp-gguf"}
    assert [a["path"] for a in read_json(lab.working / "results/serving/dev_select.json")["serving"]["attempts"]] == ["llamacpp-gguf"]


def test_the_test_run_starts_with_the_path_dev_select_used(sv, lab):
    locked_lab(lab)
    lab.write_dev_select(chosen=2, path="vllm-merged")
    code, launcher, _ = run(sv, lab, "test")
    assert code == 0 and {s[0] for s in launcher.started} == {"vllm-merged"}  # the engine the checkpoint was chosen on
    assert [a["path"] for a in read_json(lab.working / "test_run.json")["serving"]["attempts"]] == ["vllm-merged"]


# --- the preparation steps and how a session is launched (the installs and servers themselves are stubbed) -----------------------


@pytest.fixture
def ctx(sv, lab, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)  # prepare_plan points it at the scratch cache; the test must not keep that
    context = sv.Context(mode="dev-select", input_root=lab.input, working=lab.working, scratch=lab.scratch)
    context.repo = sv.Repo(lab.snapshot)
    return context


def step_plan(sv, *steps):
    return sv.SessionPlan("vllm-merged", "x", (), "http://127.0.0.1:8000/v1", ["server"], "x.log", tuple(steps))


def test_the_base_model_is_downloaded_once_into_the_cache_the_servers_read(sv, ctx, lab, tmp_path, monkeypatch):
    import huggingface_hub

    calls = []

    def fake_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        (tmp_path / "snapshot").mkdir(exist_ok=True)
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    sv.prepare_plan(ctx, step_plan(sv, sv.PrepStep("base_snapshot"), sv.PrepStep("base_snapshot")), REVISION, BASE_MODEL)
    sv.prepare_plan(ctx, step_plan(sv, sv.PrepStep("base_snapshot")), REVISION, BASE_MODEL)
    assert calls == [(BASE_MODEL, {"revision": REVISION, "cache_dir": str(lab.scratch / "hf" / "hub")})]  # one download, at the pinned commit
    assert ctx.model_dirs["base"] == tmp_path / "snapshot" and os.environ["HF_HOME"] == str(lab.scratch / "hf")  # the servers' own cache


def test_an_adapter_is_merged_once_into_the_scratch_directory_and_a_merge_error_fails_the_path(sv, ctx, lab, tmp_path, monkeypatch):
    calls = []

    def fake_merge(base_dir, adapter_dir, out_dir):
        calls.append((base_dir, adapter_dir, out_dir))
        out_dir.mkdir(parents=True)
        (out_dir / "config.json").write_text("{}")
        return {"merged_modules": 252, "scale": 2.0, "shards": 3}

    monkeypatch.setattr(ctx.repo.lora_merge, "merge_adapter", fake_merge)
    ctx.model_dirs["base"] = tmp_path / "snapshot"
    variant = sv.Variant("epoch-2", FT, f"{FT}-epoch-2", lab.train_output / "adapters" / "epoch-2", 2)
    sv.prepare_plan(ctx, step_plan(sv, sv.PrepStep("merge", variant)), REVISION, BASE_MODEL)
    sv.prepare_plan(ctx, step_plan(sv, sv.PrepStep("merge", variant)), REVISION, BASE_MODEL)
    assert calls == [(tmp_path / "snapshot", variant.adapter, lab.scratch / "merged" / "epoch-2")]  # the second call found the result
    other = sv.Variant("epoch-1", FT, f"{FT}-epoch-1", lab.train_output / "adapters" / "epoch-1", 1)

    def refusing(*args):
        raise ctx.repo.lora_merge.MergeError("2 adapter modules are not in the base weights")

    monkeypatch.setattr(ctx.repo.lora_merge, "merge_adapter", refusing)
    with pytest.raises(sv.PathFailed, match="cannot merge .*epoch-1.*not in the base weights"):
        sv.prepare_plan(ctx, step_plan(sv, sv.PrepStep("merge", other)), REVISION, BASE_MODEL)


def test_gguf_conversion_reads_the_merged_weights_or_the_base_snapshot_and_runs_once(sv, ctx, lab, tmp_path, monkeypatch):
    commands = []

    def fake_run(argv, *, log_path, env=None, cwd=None):
        commands.append([str(a) for a in argv])
        Path(argv[argv.index("--outfile") + 1]).write_text("gguf")

    monkeypatch.setattr(sv, "run_cmd", fake_run)
    ctx.model_dirs["base"] = tmp_path / "snapshot"
    adapter = sv.Variant("ft", FT, FT, lab.train_output / "adapters" / "epoch-2", 2)
    base = sv.Variant("base", BASE, BASE_MODEL)
    sv.prepare_plan(ctx, step_plan(sv, sv.PrepStep("gguf", adapter), sv.PrepStep("gguf", base), sv.PrepStep("gguf", adapter)), REVISION, BASE_MODEL)
    src = lab.scratch / f"llama.cpp-{sv.LLAMA_CPP_TAG}"
    assert [c[1:3] for c in commands] == [[str(src / "convert_hf_to_gguf.py"), str(lab.scratch / "merged" / "ft")],
                                          [str(src / "convert_hf_to_gguf.py"), str(tmp_path / "snapshot")]]  # the third was already converted
    assert all(c[0] == sys.executable and c[c.index("--outtype") + 1] == "q8_0" for c in commands)
    venv_python = lab.scratch / "vllm-env" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    (lab.scratch / "gguf" / "ft.gguf").unlink()
    sv.prepare_plan(ctx, step_plan(sv, sv.PrepStep("gguf", adapter)), REVISION, BASE_MODEL)
    assert commands[-1][0] == str(venv_python)  # the converter prefers the vLLM venv's torch and transformers


def test_the_vllm_venv_is_built_without_a_cache_and_its_version_is_checked(sv, ctx, lab, monkeypatch):
    python = lab.scratch / "vllm-env" / "bin" / "python"
    python.parent.mkdir(parents=True)

    def write_python(version):
        python.write_text(f"#!/bin/sh\nprintf '{version}\\n2.9.0+cu128\\n'\n")
        python.chmod(0o755)

    commands = []
    monkeypatch.setattr(sv, "run_cmd", lambda argv, *, log_path, env=None, cwd=None: commands.append(([str(a) for a in argv], dict(env or {}))))
    write_python("0.11.2")
    assert sv.ensure_vllm_env(ctx) == lab.scratch / "vllm-env" / "bin"
    assert [c[0][2:4] for c in commands] == [["pip", "install"], ["uv", "venv"], ["uv", "pip"]]
    assert all(c[1]["UV_NO_CACHE"] == "1" for c in commands) and commands[2][0][-2:] == sv.VLLM_PINS
    assert ctx.engine_versions["vllm"] == "0.11.2" and ctx.engine_versions["torch"] == "2.9.0+cu128"
    assert (lab.scratch / "vllm-env" / ".ready").exists()
    sv.ensure_vllm_env(ctx)
    assert len(commands) == 3  # built once
    write_python("0.10.0")
    with pytest.raises(sv.PathFailed, match="expected vllm 0.11.2"):
        sv.ensure_vllm_env(ctx)


def test_package_versions_are_read_from_another_interpreter_without_importing(sv):
    assert sv.package_version(sys.executable, "pytest")[0].isdigit()
    assert sv.package_version(sys.executable, "no-such-package-anywhere") is None
    assert sv.package_version("/no/such/python", "pytest") is None


class FakeServer:
    """Replaces ManagedServer: records how it was asked to start and never starts anything."""

    started: list = []

    def __init__(self, name, argv, *, env, log_path, health_url, **kwargs):
        self.record = {"name": name, "argv": argv, "env": dict(env), "log": log_path, "health": health_url}

    def __enter__(self):
        FakeServer.started.append(self.record)
        return self

    def __exit__(self, *exc_info):
        pass


def test_a_session_is_started_on_one_gpu_with_the_models_cache_and_the_snapshots_src_for_the_hf_server(sv, lab, monkeypatch):
    FakeServer.started = []
    monkeypatch.setattr(sv, "ManagedServer", FakeServer)
    monkeypatch.setattr(sv, "wait_gpu_idle", lambda *a, **k: None)
    monkeypatch.setattr(sv, "package_version", lambda python, package: f"{package}-1.0")
    monkeypatch.setenv("PYTHONPATH", "/already/there")
    prepared = []
    context = sv.Context(mode="test", input_root=lab.input, working=lab.working, scratch=lab.scratch, prepare=lambda *a: prepared.append(a[1].name))
    context.repo = sv.Repo(lab.snapshot)
    ft = [sv.Variant("ft", FT, FT, Path("/in/adapters/epoch-2"), 2)]

    for path in (sv.PATH_VLLM_LORA, sv.PATH_HF):
        (session,) = plan(sv, path, ft)[-1:]
        with sv.launch_session(context, session, REVISION, BASE_MODEL) as endpoint:
            assert endpoint == sv.Endpoint("http://127.0.0.1:8000/v1", None)
    assert prepared == ["vllm-lora", "hf-transformers-ft"]  # preparation comes before the server
    lora, hf = FakeServer.started
    assert lora["env"]["CUDA_VISIBLE_DEVICES"] == "0" and lora["env"]["HF_HOME"] == str(lab.scratch / "hf") and lora["health"] == "http://127.0.0.1:8000/health"
    assert lora["env"]["PYTHONPATH"] == "/already/there" and lora["log"] == lab.working / "logs" / "vllm-lora.log"
    assert hf["env"]["PYTHONPATH"] == f"{lab.snapshot / 'src'}{os.pathsep}/already/there"  # finetune_vs_api.hf_server comes from the snapshot
    assert context.engine_versions["transformers"] == "transformers-1.0" and context.engine_versions["torch"] == "torch-1.0"


# --- --check: every refusal, no GPU -------------------------------------------------------------------------------------------------------------


def check(sv, lab, mode, *argv):
    return sv.main(["--check", "--mode", mode, "--repo-root", str(lab.snapshot), "--input-root", str(lab.input), "--working", str(lab.working), *argv],
                   gpu_query=lambda: pytest.fail("--check needs no GPU"), launcher=lambda *a: pytest.fail("--check starts nothing"),
                   prepare=lambda *a: pytest.fail("--check installs nothing"))


def test_check_passes_for_dev_select_without_a_gpu_and_writes_nothing(sv, lab, capsys):
    assert check(sv, lab, "dev-select") == 0
    out = capsys.readouterr().out
    assert "check passed for dev-select: 2 adapters" in out
    assert "would start:" in out and "--enable-lora" in out and f"{FT}-epoch-1=" in out  # what the first path would run
    assert not lab.working.exists() and not lab.scratch.exists()


def test_check_runs_the_lock_check_for_test_mode(sv, lab, capsys):
    assert check(sv, lab, "test") == sv.EXIT_LOCKED
    out = capsys.readouterr().out
    assert "REFUSED" in out and "checkpoint.adapter is not set" in out
    locked_lab(lab)
    assert check(sv, lab, "test") == 0
    assert "6 locks match" in capsys.readouterr().out and not lab.working.exists()


def test_check_covers_the_adapter_the_lock_names_and_the_dev_select_record(sv, lab):
    locked_lab(lab, epoch=3)  # locked for an epoch that was never trained
    assert check(sv, lab, "test") == sv.EXIT_ADAPTER
    lab.edit("systems.yaml", "adapter: adapters/epoch-3\n      epoch: 3", "adapter: adapters/epoch-2\n      epoch: 2")
    lab.lock_all(systems=(FT,))  # the fine-tuned row, locked again for the corrected checkpoint (the base row's locks still hold)
    assert check(sv, lab, "test") == 0
    lab.write_dev_select(chosen=2, adapters={"2": {"sha256": "0" * 64}})
    assert check(sv, lab, "test") == sv.EXIT_ADAPTER  # not the file dev-select scored


def test_a_locked_adapter_path_matches_by_directory_names_not_by_the_end_of_a_string(sv):
    adapters = {2: {"directory": Path("/kaggle/input/finetune-vs-api-train/adapters/epoch-2"), "sha256": "a" * 64}}
    locked = sv.resolve_locked_adapter({"adapter": "adapters/epoch-2", "epoch": 2}, adapters, None)
    assert (locked.epoch, locked.path, locked.dev_select) == (2, "adapters/epoch-2", None)
    assert sv.resolve_locked_adapter({"adapter": "/adapters/epoch-2/", "epoch": 2}, adapters, None).path == "adapters/epoch-2"
    assert sv.resolve_locked_adapter({"adapter": "epoch-2", "epoch": 2}, adapters, None).path == "epoch-2"
    for bad in ({"adapter": "old-adapters/epoch-2", "epoch": 2}, {"adapter": "adapters/epoch-2", "epoch": 1}, {"adapter": "adapters/epoch-2", "epoch": "2"},
                {"adapter": "kaggle/out/epoch-2", "epoch": 2}):
        with pytest.raises(sv.Refused, match="the lock names checkpoint.adapter") as caught:
            sv.resolve_locked_adapter(bad, adapters, None)
        assert caught.value.code == sv.EXIT_ADAPTER


def test_check_catches_a_changed_snapshot_that_would_waste_a_gpu_run(sv, lab):
    locked_lab(lab)
    assert check(sv, lab, "test") == 0
    lab.edit("data.yaml", "locale: en-US", "locale: en-GB")
    assert check(sv, lab, "test") == sv.EXIT_LOCKED  # the dataset pin is part of what was locked
