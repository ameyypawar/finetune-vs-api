"""kaggle/serve is written, not run: these tests check what can be checked without a GPU or a network.

The metadata, the pinned stack, the commands the script would run (read as data, never run), the
GPU gate, and that the script cannot publish anything and does not import heavy libraries.
Its behaviour (preflight, the lock, the fallbacks, both modes) is in tests/test_kaggle_serve.py.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from conftest import ROOT
from finetune_vs_api import config

KAGGLE = ROOT / "kaggle"
SERVE = KAGGLE / "serve"
SCRIPT = SERVE / "serve_eval_on_kaggle.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sv():
    return load_module(SCRIPT, "serve_eval_on_kaggle")


@pytest.fixture(scope="module")
def tk():
    return load_module(KAGGLE / "train_on_kaggle.py", "train_on_kaggle")


@pytest.fixture(scope="module")
def source():
    return SCRIPT.read_text()


@pytest.fixture(scope="module")
def train_cfg():
    return yaml.safe_load((ROOT / "configs" / "train.yaml").read_text())


def load_json(path: Path):
    return json.loads(path.read_text())


# --- the metadata -----------------------------------------------------------------------------------------


def test_kernel_metadata():
    k = load_json(SERVE / "kernel-metadata.json")
    assert k["code_file"] == "serve_eval_on_kaggle.py" and (SERVE / k["code_file"]).exists()
    assert (k["language"], k["kernel_type"]) == ("python", "script")
    assert k["is_private"] is True and k["enable_gpu"] is True and k["enable_internet"] is True
    assert k["machine_shape"] == "NvidiaTeslaT4"
    assert k["competition_sources"] == []
    assert re.fullmatch(r"[a-z0-9]+/[a-z0-9-]+", k["id"]) and k["title"] == k["id"].split("/")[1]


def test_the_serving_kernel_is_set_up_like_the_training_kernel():
    k, train = load_json(SERVE / "kernel-metadata.json"), load_json(KAGGLE / "kernel-metadata.json")
    assert set(k) == set(train)
    for key in ("language", "kernel_type", "is_private", "enable_gpu", "enable_internet", "machine_shape", "competition_sources"):
        assert k[key] == train[key], key
    assert k["id"].split("/")[0] == train["id"].split("/")[0]  # one Kaggle account
    assert k["id"] != train["id"] and k["code_file"] != train["code_file"]


def test_the_kernel_reads_the_training_output_and_a_private_snapshot_dataset(sv):
    k, train = load_json(SERVE / "kernel-metadata.json"), load_json(KAGGLE / "kernel-metadata.json")
    snapshot = load_json(SERVE / "snapshot-dataset-metadata.json")
    assert k["kernel_sources"] == [train["id"]]  # the training kernel's output: the adapters
    assert k["dataset_sources"] == [snapshot["id"]]  # the repository plus the processed data
    assert snapshot["id"].split("/")[1] == sv.SNAPSHOT_SLUG and train["id"].split("/")[1] == sv.TRAIN_KERNEL_SLUG
    assert snapshot["id"].split("/")[0] == k["id"].split("/")[0]
    assert snapshot["id"] != load_json(KAGGLE / "dataset-metadata.json")["id"]  # not the SFT dataset, which leaves the test split out


def test_snapshot_dataset_metadata_satisfies_the_kaggle_cli_and_says_it_holds_the_test_split():
    d = load_json(SERVE / "snapshot-dataset-metadata.json")
    assert 6 <= len(d["id"].split("/")[1]) <= 50 and 6 <= len(d["title"]) <= 50 and d["title"] == d["id"].split("/")[1]
    assert 20 <= len(d["subtitle"]) <= 80
    assert len(d["licenses"]) == 1 and d["licenses"][0]["name"]
    for needle in ("MASSIVE", "SLURP", "CC BY 4.0", "NOTICE.md", "converted to JSON", "includes the test split", "must stay private"):
        assert needle in d["description"], needle
    assert "isPrivate" not in d  # the CLI creates datasets private unless asked otherwise


def test_the_serve_directory_holds_only_what_belongs_there():
    assert sorted(p.name for p in SERVE.iterdir() if not p.name.startswith(("__", "."))) == [
        "kernel-metadata.json", "serve_eval_on_kaggle.py", "snapshot-dataset-metadata.json",
    ]


# --- the pinned stack ------------------------------------------------------------------------------------------


def test_vllm_is_pinned_to_the_version_unsloths_kaggle_t4_notebooks_install(sv):
    # nb/Kaggle-*.ipynb: `_vllm = 'vllm==0.11.2' if is_t4 else 'vllm==0.15.1'`, then `transformers==4.56.2`
    assert sv.VLLM_VERSION == "0.11.2" and sv.VLLM_PINS == ["vllm==0.11.2", "transformers==4.56.2"]
    assert all(re.fullmatch(r"[\w.-]+==[\w.]+", pin) for pin in sv.VLLM_PINS)


def test_the_fallbacks_install_no_training_stack_next_to_the_running_process(source):
    code = source.split('"""', 2)[2]
    for needle in ("peft==", "pip install peft", "MERGE_PINS", "accelerate==", "transformers==5"):
        assert needle not in code, needle  # the merge is numpy (finetune_vs_api.lora_merge) and the accuracy-only server a process of its own
    assert "finetune_vs_api.hf_server" in code and "lora_merge" in code


def test_fastembed_is_the_version_the_repository_pins(sv):
    pinned = next(line for line in (ROOT / "requirements.txt").read_text().splitlines() if line.startswith("fastembed=="))
    assert sv.FASTEMBED_PIN == pinned.split(";")[0].strip()


def test_the_lora_settings_follow_the_training_config(sv, train_cfg):
    assert sv.MAX_LORA_RANK == train_cfg["lora"]["r"] == 16
    assert sv.SERVABLE_TARGETS == set(train_cfg["lora"]["target_modules_expanded"])
    assert sv.EXIT_LOCKED == 3  # the same status scripts/run_eval.py uses when the test split is locked


def test_the_local_rows_expect_the_server_where_the_script_starts_it(sv):
    ft, base = (config.resolve_system(name) for name in (sv.FT_SYSTEM, sv.BASE_SYSTEM))
    assert sv.server_address(ft["base_url"]) == sv.server_address(base["base_url"]) == ("127.0.0.1", 8000)
    assert (ft["endpoint"], base["endpoint"]) == (sv.LOCAL_ENDPOINT, sv.LOCAL_ENDPOINT)
    assert ft["supports_json_schema"] is False and base["supports_json_schema"] is False  # no constrained decoding
    assert ft["params"]["temperature"] == base["params"]["temperature"] == 0
    for bad in ("http://example.com:8000/v1", "http://10.0.0.5:8000/v1", "http://127.0.0.1/v1"):
        with pytest.raises(sv.Refused, match="127.0.0.1"):
            sv.server_address(bad)


def test_the_test_lock_needs_all_three_subsets_of_both_rows(sv):
    assert sv.LOCAL_SYSTEMS == ("ft-qwen3-4b-lora", "base-qwen3-4b-k10") and sv.TEST_SUBSETS == ("full", "S500", "S300")
    assert sv.DEV_SUBSETS == ("full", "D100", "D50")
    assert all(config.resolve_system(name)["test_subset"] == "full" for name in sv.LOCAL_SYSTEMS)


# --- what the script would run ----------------------------------------------------------------------------------


def test_the_vllm_command_for_the_t4(sv, train_cfg):
    adapters = {"ft-qwen3-4b-lora-epoch-1": Path("/in/adapters/epoch-1"), "ft-qwen3-4b-lora-epoch-2": Path("/in/adapters/epoch-2")}
    argv = sv.vllm_command("/venv/bin/vllm", model="Qwen/Qwen3-4B-Instruct-2507", host="127.0.0.1", port=8000, revision="a" * 40, lora_modules=adapters)
    flags = dict(zip(argv, argv[1:], strict=False))
    assert argv[:3] == ["/venv/bin/vllm", "serve", "Qwen/Qwen3-4B-Instruct-2507"]
    assert flags["--dtype"] == "half" and "bfloat16" not in argv  # a T4 has no bf16
    assert "--enable-lora" in argv and flags["--max-lora-rank"] == str(train_cfg["lora"]["r"])
    assert flags["--generation-config"] == "vllm" and json.loads(flags["--override-generation-config"]) == {"temperature": 0}  # greedy by default too
    assert flags["--revision"] == flags["--tokenizer-revision"] == "a" * 40
    assert (flags["--host"], flags["--port"]) == ("127.0.0.1", "8000")
    assert int(flags["--max-model-len"]) == 4096 and float(flags["--gpu-memory-utilization"]) <= 0.95
    assert argv[argv.index("--lora-modules") + 1:argv.index("--lora-modules") + 3] == [f"{n}={p}" for n, p in adapters.items()]
    assert int(flags["--max-num-seqs"]) >= 64  # the sweep goes up to concurrency 64
    assert "--served-model-name" not in argv and "--quantization" not in argv


def test_the_vllm_command_without_lora_serves_one_model_under_the_name_clients_use(sv):
    argv = sv.vllm_command("vllm", model="/tmp/merged/ft", host="127.0.0.1", port=8000, served_name="ft-qwen3-4b-lora")
    assert "--enable-lora" not in argv and "--lora-modules" not in argv and "--max-lora-rank" not in argv
    assert argv[argv.index("--served-model-name") + 1] == "ft-qwen3-4b-lora"
    assert "--revision" not in argv and "--tokenizer-revision" not in argv  # a local directory has no revision
    assert argv[argv.index("--generation-config") + 1] == "vllm" and argv[argv.index("--dtype") + 1] == "half"


def test_vllm_is_installed_pinned_into_a_venv_of_its_own_with_uv(sv):
    commands = sv.vllm_install_commands(Path("/tmp/serve/vllm-env"), "/usr/bin/python3")
    assert commands[0] == ["/usr/bin/python3", "-m", "pip", "install", "-q", "uv"]
    assert commands[1][:4] == ["/usr/bin/python3", "-m", "uv", "venv"] and "/tmp/serve/vllm-env" in commands[1]
    install = commands[2]
    assert install[:6] == ["/usr/bin/python3", "-m", "uv", "pip", "install", "-q"] and "/tmp/serve/vllm-env/bin/python" in install
    assert install[-2:] == sv.VLLM_PINS and "vllm==0.11.2" in install


def test_the_llama_cpp_fallback_builds_for_the_t4_and_serves_q8_gguf(sv):
    build, compile_ = sv.llama_build_commands(Path("/src"), Path("/src/build"), 4)
    assert "-DGGML_CUDA=ON" in build and "-DCMAKE_CUDA_ARCHITECTURES=75" in build  # compute capability 7.5
    assert "-DGGML_CUDA_NO_VMM=ON" in build  # no link to the driver library, which Kaggle's image does not expose
    assert compile_[-2:] == ["--target", "llama-server"] and compile_[compile_.index("-j") + 1] == "4"
    convert = sv.convert_gguf_command("python", Path("/src"), Path("/tmp/merged/ft"), Path("/tmp/gguf/ft.gguf"))
    assert convert[1].endswith("convert_hf_to_gguf.py") and convert[convert.index("--outtype") + 1] == "q8_0" == sv.GGUF_QUANT
    serve = sv.llama_server_command(Path("/src/build/bin/llama-server"), Path("/tmp/gguf/ft.gguf"), host="127.0.0.1", port=8000, alias="ft-qwen3-4b-lora")
    flags = dict(zip(serve, serve[1:], strict=False))
    assert (flags["-m"], flags["--host"], flags["--port"], flags["--alias"]) == ("/tmp/gguf/ft.gguf", "127.0.0.1", "8000", "ft-qwen3-4b-lora")
    assert flags["-ngl"] == "99" and int(flags["-c"]) // int(flags["--parallel"]) >= 4096  # a k=10 prompt fits a slot
    assert re.fullmatch(r"b\d+", sv.LLAMA_CPP_TAG) and sv.LLAMA_CPP_TAG in sv.LLAMA_CPP_URL and sv.LLAMA_CPP_URL.startswith("https://github.com/ggml-org/llama.cpp/")


def test_the_accuracy_only_server_is_a_module_of_this_repository_run_in_its_own_process(sv):
    model = "Qwen/Qwen3-4B-Instruct-2507"
    argv = sv.hf_server_command("python", model, served_name=model, host="127.0.0.1", port=8000, revision="a" * 40)
    assert argv[:3] == ["python", "-m", "finetune_vs_api.hf_server"]
    flags = dict(zip(argv, argv[1:], strict=False))
    assert (flags["--model-dir"], flags["--served-name"], flags["--host"], flags["--port"], flags["--revision"]) == (model, model, "127.0.0.1", "8000", "a" * 40)
    assert "--revision" not in sv.hf_server_command("python", "/tmp/merged/ft", served_name="ft-qwen3-4b-lora", host="127.0.0.1", port=8000)  # a local directory has none


def test_the_benchmark_is_started_with_the_flags_the_plan_gives(sv):
    argv = sv.bench_command(
        "python", Path("/snap"), base_url="http://127.0.0.1:8000/v1", model="ft-qwen3-4b-lora", prompts=Path("/tmp/p.jsonl"),
        out=Path("/kaggle/working/results/serving/T4.json"), system="ft-qwen3-4b-lora", engine="vllm", engine_version="0.11.2",
        dtype="float16", max_tokens=256, extra={"serving_path": "vllm-lora", "quantization": None, "adapter_epoch": 2},
    )
    flags = {}
    for flag, value in zip(argv, argv[1:], strict=False):
        flags.setdefault(flag, value)
    assert argv[:2] == ["python", "/snap/scripts/bench_throughput.py"]
    assert flags["--base-url"] == "http://127.0.0.1:8000/v1" and flags["--model"] == "ft-qwen3-4b-lora" and flags["--prompts"] == "/tmp/p.jsonl"
    assert flags["--concurrency"] == "1,8,32,64" and flags["--requests"] == "1000"
    assert flags["--gpu"] == "T4" and flags["--price-key"] == "aws_g4dn_xlarge_ondemand"
    assert flags["--out"] == "/kaggle/working/results/serving/T4.json" and flags["--dtype"] == "float16" and flags["--engine-version"] == "0.11.2"
    extras = [argv[i + 1] for i, a in enumerate(argv) if a == "--extra"]
    assert extras == ["serving_path=vllm-lora", "adapter_epoch=2"]  # a value of None is left out
    assert "--engine-version" not in sv.bench_command("python", Path("/s"), base_url="u", model="m", prompts=Path("p"), out=Path("o"), system="s",
                                                      engine="llama.cpp", engine_version=None, dtype="q8_0", max_tokens=1, extra={})


# --- the GPU gate ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["Tesla T4", "NVIDIA Tesla T4", "tesla t4"])
def test_a_t4_is_accepted(sv, name):
    assert sv.is_t4(name)


@pytest.mark.parametrize("name", ["NVIDIA L4", "Tesla P100-PCIE-16GB", "NVIDIA A100-SXM4-40GB", "NVIDIA L40S", "NVIDIA H100", "", "Tesla T40", "RTX 4090"])
def test_everything_else_is_refused(sv, name):
    assert not sv.is_t4(name)


def test_an_l4_is_fine_for_training_but_not_here_because_the_throughput_is_priced_as_a_t4(sv, tk):
    assert tk.is_supported_gpu("NVIDIA L4") and not sv.is_t4("NVIDIA L4")


def test_the_gpu_is_logged_and_anything_but_a_t4_aborts(sv, capsys):
    t4 = {"name": "Tesla T4", "memory": "15360 MiB", "compute_capability": "7.5", "driver": "550"}
    assert sv.require_t4(lambda: [t4, dict(t4)])["name"] == "Tesla T4"
    assert "Tesla T4" in capsys.readouterr().out
    with pytest.raises(sv.Refused, match="T4 only"):
        sv.require_t4(lambda: [{**t4, "name": "NVIDIA L4"}])
    with pytest.raises(sv.Refused, match="no GPU found"):
        sv.require_t4(lambda: [])
    assert "NVIDIA L4" in capsys.readouterr().out


def test_the_run_aborts_on_the_wrong_gpu_before_looking_at_anything_else(sv, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sv, "ensure_runtime_dependencies", lambda *a, **k: pytest.fail("must not install anything"))
    monkeypatch.setattr(sv, "find_repo_root", lambda *a, **k: pytest.fail("must not look for the snapshot"))
    code = sv.main(["--mode", "dev-select", "--working", str(tmp_path)], gpu_query=lambda: [{"name": "Tesla P100-PCIE-16GB", "memory": "16384 MiB", "compute_capability": "6.0", "driver": "550"}])
    assert code == sv.EXIT_REFUSED and "REFUSED" in capsys.readouterr().out
    assert not (tmp_path / "logs").exists()


# --- what it must never do ---------------------------------------------------------------------------------------------


def test_nothing_is_pushed_to_the_hub_and_no_token_is_read(source):
    code = source.split('"""', 2)[2]  # the module docstring tells the user to run the kaggle CLI; the code never does
    for forbidden in ("push_to_hub", "HfApi", "upload_file", "upload_folder", "HF_TOKEN", "hf_token", "huggingface_hub.login",
                      "notebook_login", "create_repo", "kaggle datasets", "kaggle kernels", "api_key", "KAGGLE_KEY"):
        assert forbidden not in code, forbidden


def test_importing_the_script_pulls_in_no_repository_code_and_no_heavy_libraries():
    code = (
        "import sys, importlib.util as u\n"
        f"s = u.spec_from_file_location('t', {str(SCRIPT)!r}); m = u.module_from_spec(s); s.loader.exec_module(m)\n"
        "heavy = {'finetune_vs_api', 'torch', 'vllm', 'transformers', 'peft', 'unsloth', 'trl', 'datasets', 'bitsandbytes', 'fastembed', 'openai', 'httpx', 'numpy', 'yaml'}\n"
        "print(sorted(heavy & set(sys.modules)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"


def test_the_mode_is_a_constant_because_kaggle_passes_no_arguments(sv, source, capsys):
    assert sv.MODE is None and "MODE: str | None = None" in source  # committed unset: pushing it unedited does nothing
    assert sv.main([]) == sv.EXIT_REFUSED  # and it did nothing else: it never got as far as asking for a GPU
    assert "no mode" in capsys.readouterr().out


def test_arguments_a_kaggle_kernel_adds_are_ignored_not_fatal(sv, capsys):
    assert sv.main(["-f", "/root/.local/share/jupyter/runtime/kernel-1.json"]) == sv.EXIT_REFUSED  # no mode: refused, not a usage error
    out = capsys.readouterr().out
    assert "ignoring arguments this script does not take: ['-f', '/root/.local/share/jupyter/runtime/kernel-1.json']" in out and "no mode" in out


def test_the_script_uses_the_planned_serving_stack(source):
    for needle in ("--enable-lora", "--lora-modules", "--max-lora-rank", "--generation-config", '"vllm"', "--dtype", "ManagedServer",
                   "start_new_session=True", "convert_hf_to_gguf.py", "merge_adapter", "assess_probe", "looks_degenerate",
                   "run_ladder", "lock_status", "run_eval", "bench_throughput.py", "CUDA_VISIBLE_DEVICES"):
        assert needle in source, needle
    assert "bfloat16" not in source  # fp16 on a T4
