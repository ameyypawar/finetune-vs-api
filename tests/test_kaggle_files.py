"""The Kaggle files are written, not run: these tests check what can be checked without a GPU.

That is: the script agrees with configs/train.yaml, never pushes anywhere, gates on the GPU,
converts records exactly as the repository does, and the metadata files are well formed.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys

import pytest
import yaml

from conftest import ROOT, build_processed
from finetune_vs_api import config, data
from finetune_vs_api.prompts import finetune_record

KAGGLE = ROOT / "kaggle"
SCRIPT = KAGGLE / "train_on_kaggle.py"


@pytest.fixture(scope="module")
def tk():
    spec = importlib.util.spec_from_file_location("train_on_kaggle", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train_cfg():
    return yaml.safe_load((ROOT / "configs" / "train.yaml").read_text())


@pytest.fixture(scope="module")
def source():
    return SCRIPT.read_text()


# --- the script agrees with configs/train.yaml ------------------------------------------------------------


def test_the_scripts_hyperparameters_are_the_configs(tk, train_cfg):
    lora, training = train_cfg["lora"], train_cfg["training"]
    assert tk.BASE_MODEL == train_cfg["base_model"]["name"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert tk.BASE_REVISION == train_cfg["base_model"]["revision"]
    assert (tk.LORA_R, tk.LORA_ALPHA, tk.LORA_DROPOUT) == (lora["r"], lora["alpha"], lora["dropout"]) == (16, 32, 0.0)
    assert tk.TARGET_MODULES == lora["target_modules_expanded"] and lora["target_modules"] == "all-linear"
    assert tk.LEARNING_RATE == training["learning_rate"] == 2e-4
    assert tk.NUM_TRAIN_EPOCHS == training["num_train_epochs"] == 2 and training["save_each_epoch"] is True
    assert tk.PER_DEVICE_BATCH_SIZE == training["per_device_train_batch_size"]
    assert tk.GRADIENT_ACCUMULATION_STEPS == training["gradient_accumulation_steps"]
    assert tk.PER_DEVICE_BATCH_SIZE * tk.GRADIENT_ACCUMULATION_STEPS == training["effective_batch_size"] == 16
    assert tk.MAX_SEQ_LENGTH == training["max_seq_length"] == 512
    assert tk.SEED == training["seed"] == 3407
    assert (tk.LR_SCHEDULER_TYPE, tk.WARMUP_RATIO, tk.WEIGHT_DECAY, tk.OPTIM, tk.LOAD_IN_4BIT) == (
        training["lr_scheduler_type"], training["warmup_ratio"], training["weight_decay"], training["optim"], training["load_in_4bit"],
    )
    assert (tk.TRAIN_FILE, tk.EVAL_FILE) == (train_cfg["data"]["train_file"], train_cfg["data"]["eval_file"])


def test_the_installs_are_exact_pins_of_the_training_stack(tk):
    pins = dict(p.split("==") for p in tk.PINNED_PACKAGES)
    assert len(pins) == len(tk.PINNED_PACKAGES) and all(re.fullmatch(r"[\w.-]+==[\w.]+", p) for p in tk.PINNED_PACKAGES)
    assert set(pins) == {"unsloth", "unsloth_zoo", "trl", "transformers", "peft", "accelerate", "bitsandbytes", "datasets"}
    assert "torch" not in pins  # Kaggle ships its own


def test_the_base_revision_is_still_a_placeholder_and_the_script_refuses_to_run_on_it(tk, monkeypatch):
    monkeypatch.setattr(tk, "BASE_REVISION", None)
    monkeypatch.setattr(tk, "install_pinned_packages", lambda: pytest.fail("must not install anything"))
    with pytest.raises(SystemExit, match="BASE_REVISION is not pinned"):
        tk.main([])


# --- it never publishes -----------------------------------------------------------------------------------------


def test_nothing_is_pushed_to_the_hub(source):
    assert "push_to_hub=False" in source and 'report_to="none"' in source
    for forbidden in ("push_to_hub=True", ".push_to_hub(", "push_to_hub_merged", "HfApi", "upload_file", "upload_folder", "HF_TOKEN", "hf_token", "huggingface_hub.login", "notebook_login", "create_repo"):
        assert forbidden not in source, forbidden


def test_the_script_does_not_import_this_repository(source):
    tree = ast.parse(source)
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert "finetune_vs_api" not in imported  # self-contained: Kaggle runs this one file


def test_importing_the_script_pulls_in_no_training_libraries():
    # In a fresh interpreter, so libraries other tests may have imported cannot hide a problem.
    code = (
        "import sys, importlib.util as u\n"
        f"s = u.spec_from_file_location('t', {str(SCRIPT)!r}); m = u.module_from_spec(s); s.loader.exec_module(m)\n"
        "heavy = {'torch', 'unsloth', 'trl', 'peft', 'transformers', 'datasets', 'bitsandbytes'}\n"
        "print(sorted(heavy & set(sys.modules)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"


def test_the_script_uses_the_planned_training_stack(source):
    for needle in ("FastLanguageModel.from_pretrained", "FastLanguageModel.get_peft_model", "SFTTrainer", "SFTConfig",
                   "completion_only_loss=True", "use_exact_model_name=True", "revision=BASE_REVISION", "processing_class=tokenizer",
                   "is_bfloat16_supported", "fp16=not use_bf16", "save_strategy=\"no\"", "on_epoch_end", "train_log.json"):
        assert needle in source, needle
    assert "warmup_ratio=" not in source  # deprecated in the pinned transformers; warmup_steps takes the ratio


# --- the GPU gate ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["Tesla T4", "NVIDIA L4", "NVIDIA Tesla T4", "nvidia l4"])
def test_t4_and_l4_class_gpus_are_accepted(tk, name):
    assert tk.is_supported_gpu(name)


@pytest.mark.parametrize("name", ["Tesla P100-PCIE-16GB", "NVIDIA A100-SXM4-40GB", "NVIDIA L40S", "NVIDIA H100", "NVIDIA GeForce RTX 3090", "", "Tesla T40"])
def test_everything_else_is_refused(tk, name):
    assert not tk.is_supported_gpu(name)


def test_the_run_aborts_before_installing_anything_on_an_unsupported_gpu(tk, monkeypatch):
    monkeypatch.setattr(tk, "BASE_REVISION", "0" * 40)
    monkeypatch.setattr(tk, "query_gpu", lambda: [{"name": "Tesla P100-PCIE-16GB", "memory": "16384 MiB", "compute_capability": "6.0", "driver": "550"}])
    monkeypatch.setattr(tk, "install_pinned_packages", lambda: pytest.fail("must not install on an unsupported GPU"))
    with pytest.raises(SystemExit, match="T4- or L4-class.*P100 is below Unsloth"):
        tk.main([])


def test_the_run_aborts_with_no_gpu_at_all(tk, monkeypatch):
    monkeypatch.setattr(tk, "BASE_REVISION", "0" * 40)
    monkeypatch.setattr(tk, "query_gpu", lambda: [])
    with pytest.raises(SystemExit, match="no GPU found"):
        tk.main([])


def test_the_gpu_is_logged(tk, monkeypatch, capsys):
    monkeypatch.setattr(tk, "query_gpu", lambda: [{"name": "Tesla T4", "memory": "15360 MiB", "compute_capability": "7.5", "driver": "550"}])
    assert tk.require_supported_gpu()["name"] == "Tesla T4"
    assert "Tesla T4" in capsys.readouterr().out


# --- data handling ---------------------------------------------------------------------------------------------------


def test_the_scripts_conversion_is_the_repositorys(tk):
    records = [
        {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]},
        {"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]},
        {"messages": [{"role": "user", "content": "u", "name": "bob"}, {"role": "assistant", "content": "a"}]},
    ]
    for record in records:
        assert tk.chat_to_prompt_completion(record) == data.chat_to_prompt_completion(record)


@pytest.mark.parametrize(
    "record",
    [
        {"messages": []},
        {"messages": [{"role": "user", "content": "u"}]},
        {"messages": [{"role": "assistant", "content": "a"}, {"role": "user", "content": "u"}]},
        {"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]},
        {"messages": [{"role": "user", "content": " "}, {"role": "assistant", "content": "a"}]},
        {"messages": [{"role": "user", "content": "u"}, {"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}]},
        {"foo": 1},
    ],
)
def test_both_conversions_refuse_what_is_not_a_single_exchange(tk, record):
    with pytest.raises(ValueError):
        tk.chat_to_prompt_completion(record)
    with pytest.raises(ValueError):
        data.chat_to_prompt_completion(record)


def test_check_mode_validates_the_files_this_repository_writes(tk, tmp_path, capsys):
    processed = build_processed(tmp_path)
    for split in ("train", "dev"):
        rows = data.read_examples(processed / f"{split}.jsonl")
        data.write_chat_jsonl(tmp_path / f"sft_{split}.jsonl", (finetune_record(e) for e in rows))
    assert tk.main(["--check", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "sft_train.jsonl: 72 records" in out and "sft_dev.jsonl: 216 records" in out


def test_check_mode_rejects_a_bad_file_and_a_missing_one(tk, tmp_path):
    (tmp_path / "sft_train.jsonl").write_text('{"messages": []}\n')
    (tmp_path / "sft_dev.jsonl").write_text("")
    with pytest.raises(ValueError):
        tk.check_data(tmp_path)
    (tmp_path / "sft_train.jsonl").unlink()
    with pytest.raises(SystemExit, match="missing"):
        tk.check_data(tmp_path)


def test_find_data_dir_prefers_the_expected_mount_then_searches(tk, tmp_path):
    expected = tmp_path / tk.DATASET_SLUG
    expected.mkdir()
    (expected / tk.TRAIN_FILE).write_text("")
    assert tk.find_data_dir(tmp_path) == expected
    (expected / tk.TRAIN_FILE).unlink()
    nested = tmp_path / "datasets" / "someone" / "else"
    nested.mkdir(parents=True)
    (nested / tk.TRAIN_FILE).write_text("")
    assert tk.find_data_dir(tmp_path) == nested  # found by structure, not by an assumed path
    (nested / tk.TRAIN_FILE).unlink()
    with pytest.raises(SystemExit, match="not found under /kaggle/input"):
        tk.find_data_dir(tmp_path)
    with pytest.raises(SystemExit, match="not found"):
        tk.find_data_dir(tmp_path / "nonexistent")


def test_real_sft_files_pass_the_scripts_check(tk, capsys):
    processed = config.PROCESSED_DIR
    if not (processed / "sft_train.jsonl").exists():
        pytest.skip("data/processed not prepared")
    assert tk.main(["--check", str(processed)]) == 0
    assert "sft_train.jsonl: 11514 records" in capsys.readouterr().out


# --- the metadata files --------------------------------------------------------------------------------------------------


def load(name):
    return json.loads((KAGGLE / name).read_text())


def test_kernel_metadata():
    k = load("kernel-metadata.json")
    assert k["code_file"] == "train_on_kaggle.py" and (KAGGLE / k["code_file"]).exists()
    assert (k["language"], k["kernel_type"]) == ("python", "script")
    assert k["is_private"] is True and k["enable_gpu"] is True and k["enable_internet"] is True
    assert k["machine_shape"] == "NvidiaTeslaT4"  # a P100 is below Unsloth's minimum
    assert k["competition_sources"] == [] and k["kernel_sources"] == []
    assert re.fullmatch(r"[a-z0-9]+/[a-z0-9-]+", k["id"]) and k["title"] == k["id"].split("/")[1]


def test_the_kernel_reads_a_private_dataset_that_the_dataset_metadata_describes(tk):
    k, d = load("kernel-metadata.json"), load("dataset-metadata.json")
    assert k["dataset_sources"] == [d["id"]]
    assert d["id"].split("/")[1] == tk.DATASET_SLUG
    assert k["id"].split("/")[0] == d["id"].split("/")[0]  # one Kaggle account


def test_dataset_metadata_satisfies_the_kaggle_cli_and_carries_the_attribution():
    d = load("dataset-metadata.json")
    assert 6 <= len(d["id"].split("/")[1]) <= 50 and 6 <= len(d["title"]) <= 50
    assert 20 <= len(d["subtitle"]) <= 80
    assert len(d["licenses"]) == 1 and d["licenses"][0]["name"]
    for needle in ("MASSIVE", "SLURP", "CC BY 4.0", "NOTICE.md", "converted to JSON"):
        assert needle in d["description"], needle
    assert "isPrivate" not in d  # the CLI creates datasets private unless asked otherwise


def test_the_kaggle_directory_holds_only_what_belongs_there():
    assert sorted(p.name for p in KAGGLE.iterdir() if not p.name.startswith(("__", "."))) == [
        "dataset-metadata.json", "kernel-metadata.json", "serve", "train_on_kaggle.py",
    ]  # serve/ is the serving and evaluation stage: tests/test_kaggle_serve_files.py covers it
