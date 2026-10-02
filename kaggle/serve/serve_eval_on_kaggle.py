"""Serve the fine-tune on a free Kaggle T4, pick the checkpoint on dev, run the locked test, and
measure throughput. The stage after kaggle/train_on_kaggle.py. Written, not run: nothing here
has touched a GPU yet.

Two modes, so the test lock can be committed and pushed publicly between them as a timestamped
pre-registration. Kaggle starts a script kernel with no arguments, so the mode is the MODE
constant below: set it before pushing. --mode overrides it for local use.

    dev-select   Installs a pinned vLLM and serves the base model with every saved epoch's LoRA
                 adapter. Scores each adapter, and the base row, on the whole dev split through
                 finetune_vs_api.evaluate.run_eval, and picks the epoch with the highest dev
                 exact match (ties go to the earlier epoch). Writes every score and the rule to
                 results/serving/dev_select.json. It never reads the test split.
    test         Refuses to start unless both local rows are locked, for the full test split
                 and for S500 and S300, with the configuration in the snapshot (the existing
                 test lock, config.lock_status). Then runs the full test split for both rows,
                 writes their S500, S300 and full summaries, and finally runs
                 scripts/bench_throughput.py on the fine-tuned row.

Everything is written under /kaggle/working, laid out like this repository's results/ so it can
be copied across:

    results/serving/dev_select.json    dev-select: all scores, the rule, the choice, the path used
    results/dev-select/<variant>/...   dev-select: one run_eval results tree per variant
    results/runs/<system>__test/...    test: predictions.jsonl and summary.{full,S500,S300}.json
    results/serving/T4.json            test: the throughput benchmark (and the cost per 1,000 calls)
    test_run.json                      test: locks, adapter, serving path, what was run
    logs/ attempts/                    server and install logs; the output of any path that failed
                                       (it can take about 35 GB of scratch space in /tmp when every path is tried)

How to run it (one Kaggle account, private everything):

  1. Train: kaggle/train_on_kaggle.py. Its output holds adapters/epoch-N/ and train_log.json.
  2. Snapshot: a private dataset with one tarball of this repository's tracked files plus the
     processed data, named in kaggle/serve/snapshot-dataset-metadata.json (it holds the test
     split, so it must stay private):

         mkdir /tmp/snap && tar -czf /tmp/snap/finetune-vs-api-snapshot.tar.gz $(git ls-files) data/processed
         cp kaggle/serve/snapshot-dataset-metadata.json /tmp/snap/dataset-metadata.json
         kaggle datasets create -p /tmp/snap

  3. Set MODE = "dev-select" below and push: kaggle kernels push -p kaggle/serve
  4. Download the output. Copy results/serving/dev_select.json into the repository. In
     configs/systems.yaml fill checkpoint.adapter (adapters/epoch-N), checkpoint.epoch and
     checkpoint.base_revision for ft-qwen3-4b-lora, and checkpoint.base_revision for
     base-qwen3-4b-k10. Lock all six pairs, each with a reason:

         python scripts/lock_test.py --write --system NAME --subset full|S500|S300 --reason "..."

     Commit and push the lock publicly. That is the pre-registration.
  5. Make a new snapshot (step 2, same dataset: kaggle datasets version), set MODE = "test", push.
  6. Download the output; copy results/runs/* and results/serving/T4.json into results/.

    python kaggle/serve/serve_eval_on_kaggle.py --check --mode test --repo-root . --input-root DIR

runs every check that can refuse a run (the snapshot, the adapters, the pinned revision, the
locks) with no GPU and installs nothing; DIR is where the training output was downloaded.

What it does, in order, stopping at the first thing that is wrong:

    * logs the GPU and refuses anything that is not a T4 (throughput is priced as a g4dn T4);
    * finds the repository snapshot and the adapters by structure under /kaggle/input, checks that
      every adapter is a rank-16 LoRA vLLM can serve, and that the Hugging Face revision of the
      base model is pinned (a full commit hash) and agrees with train_log.json;
    * in test mode, checks the locks (exit status 3 if any is missing or no longer matches);
    * tries the serving paths in order, and records which one ran:
        1. vllm-lora      vLLM 0.11.2 (the version Unsloth's Kaggle T4 notebooks use), fp16,
                          --enable-lora --lora-modules, --max-lora-rank 16,
                          --generation-config vllm, temperature 0
        2. vllm-lora-triton  the same with Triton attention (VLLM_ATTENTION_BACKEND=TRITON_ATTN), which
                          builds no kernels when the server starts. On a T4 vLLM otherwise picks
                          FlashInfer, which compiles and links its kernels at startup
        3. vllm-merged    the adapter merged into fp16 weights (finetune_vs_api.lora_merge: numpy
                          and the safetensors format, the arithmetic PEFT does), served by vLLM
                          without LoRA
        4. llamacpp-gguf  llama.cpp's CUDA server on GGUF q8_0 weights of the merged model
        5. hf-transformers  accuracy only: transformers on the merged weights, one request at a
                          time (finetune_vs_api.hf_server): no throughput and no cost figure,
                          because that is not how anyone would serve it
      A path is abandoned, and its output kept under attempts/ and never reported, when it cannot
      be installed or started, when a probe request returns NaN-style garbage or too little valid
      JSON (fp16 overflow), or when a finished run is implausible (empty or repeated output,
      replies that never stop, many failed calls).
    * puts the CUDA toolkit's link-time stub of the driver library (stubs/libcuda.so) on LIBRARY_PATH
      for every server it starts. Kaggle's image has none on the linker's path, so on 2026-10-02 the
      first dev-select run could not link FlashInfer's kernels ("ld: cannot find -lcuda"), and every
      faster path fell through to hf-transformers;
    * serves the model on 127.0.0.1:8000, as configs/systems.yaml's local endpoint says: that
      address is part of the test lock.

It never pushes anything to the Hugging Face Hub and uses no token: the base model is public.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple
from urllib.parse import urlsplit

# Kaggle's "GPU T4 x2" shows two devices. The throughput is priced as one T4 (a g4dn.xlarge), so
# only one is used. Set before anything imports torch, and inherited by the servers started below.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

#: "dev-select" or "test". Kaggle passes a script kernel no arguments, so set this before pushing.
#: `--mode` on the command line overrides it.
MODE: str | None = None

# --- what is served --------------------------------------------------------------------------------------

FT_SYSTEM = "ft-qwen3-4b-lora"
BASE_SYSTEM = "base-qwen3-4b-k10"
LOCAL_SYSTEMS = (FT_SYSTEM, BASE_SYSTEM)
LOCAL_ENDPOINT = "local"
TEST_SUBSETS = ("full", "S500", "S300")  # the lock is per (system, subset): all six pairs are needed
DEV_SUBSETS = ("full", "D100", "D50")
DEV_SELECT_FILE = "results/serving/dev_select.json"
FT_PROMPT = "finetuned_v1"
MODES = ("dev-select", "test")

# Where Kaggle mounts things. The mount names come from kernel-metadata.json and the dataset
# metadata; the script looks for what it needs by structure, so a different mount still works.
SNAPSHOT_SLUG = "finetune-vs-api-snapshot"
TRAIN_KERNEL_SLUG = "finetune-vs-api-train"
INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
SCRATCH = Path("/tmp/serve-eval")  # venvs, merged weights, GGUF files: far too big to be kept as output

# --- the serving stack -------------------------------------------------------------------------------------

#: What Unsloth's free Kaggle notebooks install on a Tesla T4 (their install cell picks
#: vllm==0.11.2 when nvidia-smi says "Tesla T4", and vllm==0.15.1 otherwise), with the
#: transformers pin the same cell uses. Looked up in unslothai/notebooks, nb/Kaggle-*.ipynb.
VLLM_VERSION = "0.11.2"
VLLM_PINS = [f"vllm=={VLLM_VERSION}", "transformers==4.56.2"]
FASTEMBED_PIN = "fastembed==0.8.1"  # as requirements.txt: embeds the few-shot examples on the CPU
#: A llama.cpp release tag, checked to exist; its source tarball is built with CUDA for a T4.
LLAMA_CPP_TAG = "b7200"
LLAMA_CPP_URL = f"https://github.com/ggml-org/llama.cpp/archive/refs/tags/{LLAMA_CPP_TAG}.tar.gz"
GGUF_QUANT = "q8_0"

DTYPE = "half"  # a T4 has no bf16
MAX_MODEL_LEN = 4096  # a k=10 prompt is about 1.5k tokens; the model's own default (262144) cannot fit a T4
GPU_MEMORY_UTILIZATION = 0.90
MAX_NUM_SEQS = 64  # the highest concurrency the throughput sweep asks for
MAX_LORA_RANK = 16  # configs/train.yaml lora.r
LLAMA_CTX = 32768  # shared by the slots: 4096 each
LLAMA_SLOTS = 8
READY_TIMEOUT_S = 1800  # downloading 8 GB, loading, compiling and capturing graphs
HEALTH_POLL_S = 5.0

PATH_VLLM_LORA = "vllm-lora"
PATH_VLLM_LORA_TRITON = "vllm-lora-triton"
PATH_VLLM_MERGED = "vllm-merged"
PATH_LLAMACPP = "llamacpp-gguf"
PATH_HF = "hf-transformers"
SERVING_PATHS = (PATH_VLLM_LORA, PATH_VLLM_LORA_TRITON, PATH_VLLM_MERGED, PATH_LLAMACPP, PATH_HF)
#: The attention backend of vllm-lora-triton. On a T4 (compute capability 7.5) vLLM cannot use FlashAttention 2
#: and picks FlashInfer, which compiles its kernels at startup; Triton attention needs no build.
TRITON_ATTENTION = "TRITON_ATTN"
#: Where CUDA toolkits keep libcuda.so, the link-time stub of the driver library.
CUDA_STUB_DIRS = ("/usr/local/cuda/lib64/stubs", "/usr/local/cuda/targets/x86_64-linux/lib/stubs")
PATH_INFO: dict[str, dict[str, Any]] = {
    PATH_VLLM_LORA: {
        "engine": "vllm", "dtype": "float16", "quantization": None, "throughput": True,
        "weights": "fp16 base weights, the LoRA adapter applied by vLLM at request time",
    },
    PATH_VLLM_LORA_TRITON: {
        "engine": "vllm", "dtype": "float16", "quantization": None, "throughput": True,
        "weights": "fp16 base weights, the LoRA adapter applied by vLLM at request time, with Triton attention",
    },
    PATH_VLLM_MERGED: {
        "engine": "vllm", "dtype": "float16", "quantization": None, "throughput": True,
        "weights": "the adapter merged into fp16 weights, served without LoRA",
    },
    PATH_LLAMACPP: {
        "engine": "llama.cpp", "dtype": GGUF_QUANT, "quantization": GGUF_QUANT, "throughput": True,
        "weights": f"the adapter merged, converted to GGUF {GGUF_QUANT}",
    },
    PATH_HF: {
        "engine": "transformers", "dtype": "float16", "quantization": None, "throughput": False,
        "weights": "the adapter merged into fp16 weights, run with transformers one request at a time",
    },
}

# --- what makes an output implausible (fp16 overflow gives NaNs and garbage) ----------------------------------------

PROBE_SHORT = 6  # fine-tune-style requests sent right after a server is up
PROBE_LONG = 2  # k=10 requests: the longest prompts, where overflow shows first
PROBE_MAX_DEGENERATE = 0.25
PROBE_MIN_VALID = 0.5  # of the short probes, for a model that should answer in the schema
MAX_DEGENERATE_FRACTION = 0.05
MAX_TRUNCATED_FRACTION = 0.2  # replies that ran to max_tokens
MAX_FAILED_FRACTION = 0.05
MIN_VALID_RATE = 0.5

# --- adapters -----------------------------------------------------------------------------------------------------------

EPOCH_DIR = re.compile(r"epoch-(\d+)")
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
SERVABLE_TARGETS = frozenset({"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"})
COMMIT_HASH = re.compile(r"[0-9a-f]{40}")
SELECTION_RULE = (
    "highest exact match on the full dev split (point estimate); a tie goes to the earlier epoch"
)

# --- the throughput sweep -------------------------------------------------------------------------------------------------

SWEEP_CONCURRENCY = "1,8,32,64"
SWEEP_REQUESTS = 1000
PRICE_KEY = "aws_g4dn_xlarge_ondemand"
GPU_LABEL = "T4"
EXIT_REFUSED = 2
EXIT_LOCKED = 3
EXIT_ADAPTER = 4
EXIT_NO_PATH = 5


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class Refused(Exception):
    """The run must not start or go on. `code` is the exit status."""

    def __init__(self, message: str, code: int = EXIT_REFUSED):
        super().__init__(message)
        self.code = code


class PathFailed(Exception):
    """A serving path did not work (install, start, probe or a run): try the next one."""


class CommandFailed(PathFailed):
    """A subprocess exited non-zero."""


class NoServingPath(Exception):
    def __init__(self, attempts: list[dict[str, Any]]):
        self.attempts = attempts
        super().__init__("no serving path worked: " + "; ".join(f"{a['path']}: {a['reason']}" for a in attempts))


# --- small helpers --------------------------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def tail(path: Path, lines: int = 25) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no log)"


def run_cmd(argv: Sequence[Any], *, log_path: Path, env: Mapping[str, str] | None = None, cwd: Path | None = None) -> None:
    """Run a command with its output appended to `log_path`; CommandFailed, with the log's tail, if it fails."""
    argv = [str(a) for a in argv]
    log(f"$ {shlex.join(argv)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as handle:
        handle.write(f"\n$ {shlex.join(argv)}\n".encode())
        handle.flush()
        done = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, env=dict(env) if env else None, cwd=cwd)
    if done.returncode != 0:
        raise CommandFailed(f"`{shlex.join(argv)[:200]}` exited {done.returncode}. Last lines of {log_path.name}:\n{tail(log_path)}")


# --- hardware ---------------------------------------------------------------------------------------------------------------------


def query_gpu() -> list[dict[str, str]]:
    """GPU name, memory, compute capability and driver, from nvidia-smi (no torch import)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        name, memory, capability, driver = (part.strip() for part in line.split(","))
        gpus.append({"name": name, "memory": memory, "compute_capability": capability, "driver": driver})
    return gpus


def is_t4(name: str) -> bool:
    """Exactly a T4. Training also accepts an L4; the throughput here is priced as a g4dn T4."""
    return bool(re.search(r"\bT4\b", name or "", re.IGNORECASE))


def require_t4(query: Callable[[], list[dict[str, str]]] = query_gpu) -> dict[str, str]:
    gpus = query()
    log(f"GPU(s) visible: {gpus or 'none'}")
    if not gpus:
        raise Refused("no GPU found: the kernel needs a GPU accelerator (kernel-metadata.json machine_shape)")
    gpu = gpus[0]
    if not is_t4(gpu["name"]):
        raise Refused(
            f"unsupported GPU {gpu['name']!r}: this script runs on a T4 only, because the throughput it "
            "measures is priced as an AWS g4dn.xlarge (one T4)"
        )
    return gpu


def gpu_memory_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def wait_gpu_idle(
    timeout_s: float = 90.0, threshold_mib: int = 1500, *, used: Callable[[], int | None] = gpu_memory_used_mib,
    sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
) -> None:
    """Wait for the previous server's memory to be released (best effort: it never raises)."""
    deadline = clock() + timeout_s
    while clock() < deadline:
        memory = used()
        if memory is None or memory <= threshold_mib:
            return
        sleep(3.0)
    log(f"warning: GPU memory still in use after {timeout_s:.0f} s")


# --- the repository snapshot ---------------------------------------------------------------------------------------------------------


def _find_repo(base: Path) -> Path | None:
    if not base.is_dir():
        return None
    for marker in sorted(base.rglob("__init__.py")):
        if marker.parent.name == "finetune_vs_api" and marker.parent.parent.name == "src":
            return marker.parent.parent.parent
    return None


def extract_archive(archive: Path, dest: Path) -> None:
    """Unpack a .tar, .tar.gz, .tgz or .zip into `dest`, refusing members that would land outside it."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()

    def inside(name: str) -> bool:
        return (root / name).resolve().is_relative_to(root)

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            if not all(inside(n) for n in zf.namelist()):
                raise Refused(f"{archive.name} holds paths outside the archive")
            zf.extractall(dest)
        return
    with tarfile.open(archive) as tf:
        if not all(inside(m.name) for m in tf.getmembers()):
            raise Refused(f"{archive.name} holds paths outside the archive")
        tf.extractall(dest, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))


def find_repo_root(input_root: Path, extract_to: Path) -> Path:
    """The repository snapshot: the expected mount, else wherever src/finetune_vs_api turns up,
    else the first archive under the mount that unpacks to one (Kaggle may or may not unpack a
    tarball in a dataset)."""
    expected = input_root / SNAPSHOT_SLUG
    for base in (expected, input_root):
        found = _find_repo(base)
        if found is not None:
            return found
    for base in (expected, input_root):
        if not base.is_dir():
            continue
        for archive in sorted(p for p in base.rglob("*") if p.is_file() and p.name.endswith((".tar", ".tar.gz", ".tgz", ".zip"))):
            target = extract_to / archive.name.split(".")[0]
            if not target.exists():
                extract_archive(archive, target)
            found = _find_repo(target)
            if found is not None:
                return found
    raise Refused(
        f"no repository snapshot under {input_root}: is the private dataset '{SNAPSHOT_SLUG}' attached to this "
        "kernel (kernel-metadata.json dataset_sources)? It must hold src/finetune_vs_api/"
    )


class Repo:
    """The snapshot's own package, imported from its src/ (config, evaluate, ...), and its paths."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.config_dir = self.root / "configs"
        self.processed = self.root / "data" / "processed"
        self.lock_path = self.root / "results" / "test_lock.jsonl"
        src = self.root / "src"
        if src.is_dir() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from finetune_vs_api import (
            config,
            cost,
            data,
            evaluate,
            lora_merge,
            metrics,
            prompts,
            schema,
            subsets,
        )

        self.config, self.cost, self.data, self.evaluate = config, cost, data, evaluate
        self.lora_merge, self.metrics, self.prompts, self.schema, self.subsets = lora_merge, metrics, prompts, schema, subsets

    def manifest(self) -> dict[str, str]:
        """sha256 of the files the run depends on, so a result names exactly what it was made from."""
        files = [*sorted(self.config_dir.glob("*.yaml")), *sorted(self.processed.glob("*.jsonl")), self.processed / "subsets.json", self.lock_path]
        return {p.relative_to(self.root).as_posix(): sha256_file(p) for p in files if p.exists()}


def server_address(base_url: str) -> tuple[str, int]:
    """Host and port of the local endpoint. It must be this machine: the address is hashed into the test lock."""
    parts = urlsplit(base_url)
    if parts.hostname not in ("127.0.0.1", "localhost", "::1") or not parts.port:
        raise Refused(f"the local endpoint {base_url!r} must be http://127.0.0.1:<port>/v1; the model is served on this box")
    return parts.hostname, parts.port


# --- adapters ----------------------------------------------------------------------------------------------------------------------------


def find_adapters(input_root: Path) -> dict[int, Path]:
    """epoch -> adapter directory, found by structure: a directory called epoch-N holding the two PEFT files."""
    found: dict[int, Path] = {}
    expected = input_root / TRAIN_KERNEL_SLUG
    for base in (expected, input_root):
        if not base.is_dir():
            continue
        for config_file in sorted(base.rglob("adapter_config.json")):
            directory = config_file.parent
            match = EPOCH_DIR.fullmatch(directory.name)
            if match and all((directory / name).exists() for name in ADAPTER_FILES):
                found.setdefault(int(match.group(1)), directory)
        if found:
            break
    return dict(sorted(found.items()))


def adapter_problems(config: Mapping[str, Any]) -> list[str]:
    """Why vLLM (or the merge) could not use an adapter with this adapter_config.json, if it could not."""
    problems = []
    if str(config.get("peft_type", "")).upper() != "LORA":
        problems.append(f"peft_type is {config.get('peft_type')!r}, not LORA")
    rank = config.get("r")
    if not isinstance(rank, int) or rank < 1:
        problems.append(f"rank r is {rank!r}")
    elif rank > MAX_LORA_RANK:
        problems.append(f"rank {rank} is above --max-lora-rank {MAX_LORA_RANK}")
    if config.get("use_dora"):
        problems.append("use_dora is set: vLLM cannot serve DoRA")
    if config.get("lora_bias"):
        problems.append("lora_bias is set")
    if config.get("modules_to_save"):
        problems.append(f"modules_to_save is {config['modules_to_save']!r}")
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        problems.append("rank_pattern or alpha_pattern is set: layers with their own rank or alpha")
    targets = config.get("target_modules")
    targets = set(targets) if isinstance(targets, list | set | tuple) else {targets}
    if not targets <= SERVABLE_TARGETS:
        problems.append(f"target_modules {sorted(map(str, targets - SERVABLE_TARGETS))} are not among the seven projections")
    return problems


def read_adapter(directory: Path) -> dict[str, Any]:
    """What is known about one adapter directory; AdapterError-style Refused (status 4) if it cannot be served."""
    try:
        config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Refused(f"cannot read {directory / 'adapter_config.json'}: {exc}", EXIT_ADAPTER) from None
    problems = adapter_problems(config)
    if problems:
        raise Refused(f"adapter {directory} cannot be served: " + "; ".join(problems), EXIT_ADAPTER)
    return {
        "directory": directory,
        "sha256": sha256_file(directory / "adapter_model.safetensors"),
        "rank": config["r"],
        "lora_alpha": config.get("lora_alpha"),
        "base_model_name_or_path": config.get("base_model_name_or_path"),
    }


def find_train_log(input_root: Path) -> dict[str, Any] | None:
    for base in (input_root / TRAIN_KERNEL_SLUG, input_root):
        if base.is_dir():
            for path in sorted(base.rglob("train_log.json")):
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
    return None


def pin_revision(repo: Repo, train_log: Mapping[str, Any] | None, locked: Sequence[str] = ()) -> str:
    """The Hugging Face commit the base model is served from: a full hash that every record agrees on.

    configs/train.yaml names the revision the adapters were trained from, train_log.json records
    what the training run used, and (test mode) `locked` holds the checkpoint.base_revision of
    each local row. A branch name or a short hash would not say which weights were used.
    """
    train_cfg = repo.config.load_yaml("train", repo.config_dir)
    revision = train_cfg["base_model"].get("revision")
    if not revision:
        raise Refused("configs/train.yaml base_model.revision is not pinned: set it to the commit hash the adapters were trained from")
    if not COMMIT_HASH.fullmatch(str(revision)):
        raise Refused(f"base_model.revision {revision!r} is not a full 40-character commit hash")
    logged = (train_log or {}).get("base_revision")
    if logged and logged != revision:
        raise Refused(f"train_log.json says the adapters were trained from {logged}, but configs/train.yaml pins {revision}")
    for value in locked:
        if value != revision:
            raise Refused(f"a system's checkpoint.base_revision is {value!r}, but configs/train.yaml pins {revision}")
    return str(revision)


# --- the test lock -------------------------------------------------------------------------------------------------------------------------


def lock_problems(
    repo: Repo, *, systems: Sequence[str] = LOCAL_SYSTEMS, subsets: Sequence[str] = TEST_SUBSETS
) -> list[str]:
    """Everything wrong with the locks, using the existing lock machinery: for each local row and each of
    the full, S500 and S300 test subsets, a lock whose configuration and subset hash match what is in the
    snapshot now. Empty when the test split may be run."""
    problems: list[str] = []
    kw = {"config_dir": repo.config_dir, "processed_dir": repo.processed, "lock_path": repo.lock_path}
    for system in systems:
        try:
            problems += [f"{system}: {blocker}" for blocker in repo.config.lock_blockers(system, config_dir=repo.config_dir)]
        except (repo.config.ConfigError, OSError, KeyError) as exc:
            problems.append(f"{system}: {exc}")
            continue
        for subset in subsets:
            try:
                entry, why = repo.config.lock_status(system, subset, **kw)
            except (repo.config.ConfigError, repo.config.LockError, OSError, ValueError, KeyError) as exc:
                problems.append(f"{system} [{subset}]: {exc}")
                continue
            if entry is None:
                problems.append(f"{system} [{subset}]: {why}")
    return problems


def lock_entries(repo: Repo, systems: Sequence[str] = LOCAL_SYSTEMS, subsets: Sequence[str] = TEST_SUBSETS) -> dict[str, dict[str, Any]]:
    """The matching lock entry for each pair (call after lock_problems found nothing)."""
    kw = {"config_dir": repo.config_dir, "processed_dir": repo.processed, "lock_path": repo.lock_path}
    out: dict[str, dict[str, Any]] = {}
    for system in systems:
        out[system] = {}
        for subset in subsets:
            entry, _ = repo.config.lock_status(system, subset, **kw)
            out[system][subset] = {k: entry[k] for k in ("locked_at", "reason", "config_hash", "subset_hash", "git_commit")}
    return out


# --- variants and their configs -------------------------------------------------------------------------------------------------------------


class Variant(NamedTuple):
    """One model to evaluate. Clients always send `model`; every serving path answers to that name."""

    key: str  # "base", "epoch-1", "ft"
    system: str
    model: str
    adapter: Path | None = None
    epoch: int | None = None


def write_variant_config(repo: Repo, dest: Path, variant: Variant, revision: str) -> Path:
    """A copy of configs/ whose systems.yaml names this variant: the model it is served under and the
    checkpoint it stands for, so a dev summary says which epoch produced it. Dev only: a test run
    uses configs/ itself, because the test lock hashes it."""
    import yaml

    dest.mkdir(parents=True, exist_ok=True)
    for path in repo.config_dir.glob("*.y*ml"):
        shutil.copy2(path, dest / path.name)
    doc = yaml.safe_load((repo.config_dir / "systems.yaml").read_text(encoding="utf-8"))
    row = doc["systems"][variant.system]
    row["model"] = variant.model
    if variant.system == FT_SYSTEM:
        row["checkpoint"] = {"adapter": f"adapters/epoch-{variant.epoch}", "epoch": variant.epoch, "base_revision": revision}
    else:
        row["checkpoint"] = {"base_revision": revision}
    (dest / "systems.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return dest


def pick_best_epoch(scores: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The deterministic rule: most correct items on the full dev split, the earlier epoch on a tie."""
    candidates = [s for s in scores if s.get("epoch") is not None]
    if not candidates:
        raise ValueError("no adapter was scored")
    return max(candidates, key=lambda s: (s["correct"], -s["epoch"]))


def score_record(variant: Variant, summary: Mapping[str, Any], summary_path: Path, base: Path) -> dict[str, Any]:
    m = summary["metrics"]
    n = summary["n_scored"]
    exact = m["exact_match"]["value"]
    return {
        "variant": variant.key, "system": variant.system, "model": variant.model, "epoch": variant.epoch,
        "n_scored": n, "n_calls_failed": summary["n_calls_failed"], "correct": round(exact * n),
        "exact_match": exact, "exact_match_ci95": m["exact_match"].get("ci95"),
        "intent_accuracy": m["intent_accuracy"]["value"], "slot_f1": m["slot_f1"]["value"],
        "schema_valid_rate": m["schema_valid_rate"]["value"],
        "summary": summary_path.relative_to(base).as_posix() if summary_path.is_relative_to(base) else str(summary_path),
    }


# --- is the output plausible? (fp16 overflow gives NaNs and garbage) ----------------------------------------------------------------------------


def looks_degenerate(text: str | None) -> bool:
    """Empty, one or two characters repeated, replacement characters, or one short pattern repeated
    to the token limit: what a model returns when its logits have gone to NaN."""
    if text is None or not text.strip():
        return True
    stripped = text.strip()
    if len(stripped) >= 8 and len(set(stripped)) <= 2:
        return True
    if stripped.count("\ufffd") > len(stripped) // 4:
        return True
    raw = stripped.encode("utf-8")
    return len(raw) >= 64 and len(zlib.compress(raw)) / len(raw) < 0.08


def assess_probe(replies: Sequence[Mapping[str, Any]], *, expect_schema: bool, is_valid: Callable[[str | None], bool]) -> tuple[bool, str]:
    """(ok, why) for the probe requests sent to one model name. A reply is {"kind": "short"|"long", "text": ..., "error": ...}."""
    if not replies:
        return False, "no probe request was answered"
    errors = [r["error"] for r in replies if r.get("error")]
    if errors:
        return False, f"{len(errors)} of {len(replies)} probe requests failed: {errors[0]}"
    degenerate = sum(1 for r in replies if looks_degenerate(r.get("text")))
    if degenerate / len(replies) > PROBE_MAX_DEGENERATE:
        return False, f"{degenerate} of {len(replies)} probe answers are empty or repeated characters (NaN-style garbage: fp16 overflow?)"
    if expect_schema:
        short = [r for r in replies if r.get("kind") == "short"]
        valid = sum(1 for r in short if is_valid(r.get("text")))
        if short and valid / len(short) < PROBE_MIN_VALID:
            return False, f"only {valid} of {len(short)} short probe answers are valid schema JSON: the adapter is not being applied, or the output is garbage"
    return True, f"{len(replies)} probe answers look sane"


def assess_predictions(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Shares of empty or repeated, runaway and failed answers among the final rows of a run."""
    n = len(rows)
    answered = [r for r in rows.values() if r.get("error") is None]
    degenerate = sum(1 for r in answered if looks_degenerate(r.get("text")))
    truncated = sum(1 for r in answered if r.get("finish_reason") == "length")
    failed = n - len(answered)
    return {
        "n": n, "failed": failed, "degenerate": degenerate, "truncated": truncated,
        "failed_fraction": failed / n if n else 0.0,
        "degenerate_fraction": degenerate / n if n else 0.0,
        "truncated_fraction": truncated / n if n else 0.0,
    }


def implausible(assessment: Mapping[str, Any], summary: Mapping[str, Any]) -> str | None:
    """Why a finished run cannot be reported, or None. A genuine model failure (valid JSON, wrong labels)
    never trips this; empty and repeated output, replies that never stop and a flood of failed calls do."""
    reasons = []
    if assessment["degenerate_fraction"] > MAX_DEGENERATE_FRACTION:
        reasons.append(f"{assessment['degenerate']} of {assessment['n']} answers are empty or repeated characters")
    if assessment["truncated_fraction"] > MAX_TRUNCATED_FRACTION:
        reasons.append(f"{assessment['truncated']} of {assessment['n']} answers ran to the token limit")
    if assessment["failed_fraction"] > MAX_FAILED_FRACTION:
        reasons.append(f"{assessment['failed']} of {assessment['n']} calls failed")
    valid = summary["metrics"]["schema_valid_rate"]["value"] if summary.get("metrics") else 0.0
    if valid < MIN_VALID_RATE:
        reasons.append(f"only {valid:.0%} of the answers are valid schema JSON")
    return "; ".join(reasons) + " (fp16 overflow or a broken serving path, not a model result)" if reasons else None


# --- the commands (pure: tests read them) ---------------------------------------------------------------------------------------------------------


def vllm_command(
    vllm: Path | str, *, model: str, host: str, port: int, revision: str | None = None,
    served_name: str | None = None, lora_modules: Mapping[str, Path] | None = None,
) -> list[str]:
    """`vllm serve` for the T4: fp16, vLLM's own sampling defaults (the model's generation_config.json would
    otherwise set top_p and top_k) with temperature 0 as the default (every request also says so), a context that
    fits, and with `lora_modules` the adapters registered by name."""
    argv = [
        str(vllm), "serve", model, "--host", host, "--port", str(port),
        "--dtype", DTYPE, "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION), "--max-num-seqs", str(MAX_NUM_SEQS),
        "--generation-config", "vllm", "--override-generation-config", '{"temperature": 0}',
    ]
    if revision:
        argv += ["--revision", revision, "--tokenizer-revision", revision]
    if served_name:
        argv += ["--served-model-name", served_name]
    if lora_modules:
        argv += ["--enable-lora", "--max-lora-rank", str(MAX_LORA_RANK), "--lora-modules", *(f"{n}={p}" for n, p in lora_modules.items())]
    return argv


def vllm_install_commands(env_dir: Path, python: str | None = None) -> list[list[str]]:
    """Install vLLM, pinned, into a venv of its own (so its torch and transformers cannot disturb this
    interpreter's), the way Unsloth's notebooks install it: with uv."""
    python = python or sys.executable
    return [
        [python, "-m", "pip", "install", "-q", "uv"],
        [python, "-m", "uv", "venv", str(env_dir), "--python", python],
        [python, "-m", "uv", "pip", "install", "-q", "--python", str(env_dir / "bin" / "python"), *VLLM_PINS],
    ]


def llama_build_commands(src_dir: Path, build_dir: Path, jobs: int) -> list[list[str]]:
    return [
        # GGML_CUDA_NO_VMM: no virtual memory management, so ggml-cuda does not link the driver library, which
        # the first run's configure step could not find on Kaggle (target CUDA::cuda_driver "not found")
        ["cmake", "-S", str(src_dir), "-B", str(build_dir), "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=75",
         "-DGGML_CUDA_NO_VMM=ON", "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF"],
        ["cmake", "--build", str(build_dir), "--config", "Release", "-j", str(jobs), "--target", "llama-server"],
    ]


def convert_gguf_command(python: str, src_dir: Path, model_dir: Path, out_file: Path) -> list[str]:
    return [python, str(src_dir / "convert_hf_to_gguf.py"), str(model_dir), "--outfile", str(out_file), "--outtype", GGUF_QUANT]


def llama_server_command(binary: Path, gguf: Path, *, host: str, port: int, alias: str) -> list[str]:
    return [
        str(binary), "-m", str(gguf), "--host", host, "--port", str(port), "-ngl", "99",
        "-c", str(LLAMA_CTX), "--parallel", str(LLAMA_SLOTS), "--alias", alias, "--jinja",
    ]


def hf_server_command(python: str, model_dir: str, *, served_name: str, host: str, port: int, revision: str | None = None) -> list[str]:
    """The accuracy-only server: finetune_vs_api.hf_server in a process of its own (PYTHONPATH gives it the snapshot's src/)."""
    argv = [python, "-m", "finetune_vs_api.hf_server", "--model-dir", model_dir, "--served-name", served_name, "--host", host, "--port", str(port)]
    return argv + (["--revision", revision] if revision else [])


def bench_command(
    python: str, repo_root: Path, *, base_url: str, model: str, prompts: Path, out: Path, system: str,
    engine: str, engine_version: str | None, dtype: str, max_tokens: int, extra: Mapping[str, Any],
) -> list[str]:
    argv = [
        python, str(repo_root / "scripts" / "bench_throughput.py"),
        "--base-url", base_url, "--model", model, "--prompts", str(prompts),
        "--concurrency", SWEEP_CONCURRENCY, "--requests", str(SWEEP_REQUESTS),
        "--gpu", GPU_LABEL, "--price-key", PRICE_KEY, "--out", str(out), "--system", system,
        "--dtype", dtype, "--engine", engine, "--max-tokens", str(max_tokens),
        "--sources", str(repo_root / "configs" / "sources.yaml"),
    ]
    if engine_version:
        argv += ["--engine-version", engine_version]
    for key, value in extra.items():
        if value is not None:
            argv += ["--extra", f"{key}={value}"]
    return argv


# --- session plans: what to start, serving which variants ---------------------------------------------------------------------------------------------


class PrepStep(NamedTuple):
    kind: str  # vllm_env | base_snapshot | merge | llama_cpp | gguf
    variant: Variant | None = None


class SessionPlan(NamedTuple):
    """One server and the variants it serves."""

    path: str
    name: str
    variants: tuple[Variant, ...]
    base_url: str
    argv: list[str]
    log: str
    prep: tuple[PrepStep, ...]


class Endpoint(NamedTuple):
    """Where requests for a session go: a URL (a server started by this script), or, in tests, an httpx transport."""

    base_url: str
    transport: Any = None


def plan_sessions(
    path: str, variants: Sequence[Variant], *, base_model: str, revision: str, host: str, port: int, scratch: Path,
) -> list[SessionPlan]:
    """The sessions a path needs. vllm-lora serves every variant from one server; the others serve one
    model per server (the adapter merged into its weights), base first, so that the fine-tuned row comes last."""
    if path not in SERVING_PATHS:
        raise ValueError(f"unknown serving path {path!r}")
    base_url = f"http://{host}:{port}/v1"
    vllm = scratch / "vllm-env" / "bin" / "vllm"
    if path in (PATH_VLLM_LORA, PATH_VLLM_LORA_TRITON):  # the same server; session_env sets the attention backend
        adapters = {v.model: v.adapter for v in variants if v.adapter is not None}
        argv = vllm_command(vllm, model=base_model, host=host, port=port, revision=revision, lora_modules=adapters)
        return [SessionPlan(path, path, tuple(variants), base_url, argv, f"{path}.log", (PrepStep("vllm_env"),))]
    sessions = []
    for variant in sorted(variants, key=lambda v: v.adapter is not None):
        merged = scratch / "merged" / variant.key
        snapshot = (PrepStep("base_snapshot"),)
        merge = (*snapshot, PrepStep("merge", variant)) if variant.adapter is not None else ()
        if path == PATH_VLLM_MERGED:
            if variant.adapter is None:
                argv = vllm_command(vllm, model=base_model, host=host, port=port, revision=revision, served_name=variant.model)
            else:
                argv = vllm_command(vllm, model=str(merged), host=host, port=port, served_name=variant.model)
            prep = (PrepStep("vllm_env"), *merge)
        elif path == PATH_LLAMACPP:
            gguf = scratch / "gguf" / f"{variant.key}.gguf"
            argv = llama_server_command(scratch / f"llama.cpp-{LLAMA_CPP_TAG}" / "build" / "bin" / "llama-server", gguf, host=host, port=port, alias=variant.model)
            prep = (PrepStep("llama_cpp"), *(merge or snapshot), PrepStep("gguf", variant))
        else:
            if variant.adapter is None:
                argv = hf_server_command(sys.executable, base_model, served_name=variant.model, host=host, port=port, revision=revision)
            else:
                argv = hf_server_command(sys.executable, str(merged), served_name=variant.model, host=host, port=port)
            prep = merge
        sessions.append(SessionPlan(path, f"{path}-{variant.key}", (variant,), base_url, argv, f"{path}-{variant.key}.log", prep))
    return sessions


def paths_from(start_at: str | None) -> tuple[str, ...]:
    if start_at is None:
        return SERVING_PATHS
    if start_at not in SERVING_PATHS:
        raise Refused(f"--start-at must be one of {', '.join(SERVING_PATHS)}")
    return SERVING_PATHS[SERVING_PATHS.index(start_at):]


def run_ladder(
    paths: Sequence[str], attempt: Callable[[str], Any], *, on_failure: Callable[[str, str], None] = lambda path, reason: None,
) -> tuple[str, Any, list[dict[str, Any]]]:
    """Try each serving path in order until one completes. A path fails by raising PathFailed (or
    CommandFailed); any other exception is a bug and propagates. Returns (path, its result, the record
    of every attempt). Raises NoServingPath, carrying the record, when none worked."""
    attempts: list[dict[str, Any]] = []
    for path in paths:
        try:
            result = attempt(path)
        except PathFailed as exc:
            reason = str(exc)[:3000]
            attempts.append({"path": path, "status": "failed", "reason": reason})
            on_failure(path, reason)
            continue
        attempts.append({"path": path, "status": "ok", "reason": None})
        return path, result, attempts
    raise NoServingPath(attempts)


# --- starting and stopping a server --------------------------------------------------------------------------------------------------------------------------


class ManagedServer:
    """A server subprocess in its own process group, with its output in a log file."""

    def __init__(
        self, name: str, argv: Sequence[str], *, env: Mapping[str, str] | None, log_path: Path, health_url: str,
        ready_timeout_s: float = READY_TIMEOUT_S, poll_s: float = HEALTH_POLL_S,
        sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
    ):
        self.name, self.argv, self.env, self.log_path, self.health_url = name, [str(a) for a in argv], env, log_path, health_url
        self.ready_timeout_s, self.poll_s, self._sleep, self._clock = ready_timeout_s, poll_s, sleep, clock
        self.proc: subprocess.Popen | None = None
        self._log = None

    def start(self) -> None:
        log(f"starting {self.name}: {shlex.join(self.argv)}")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(
            self.argv, stdout=self._log, stderr=subprocess.STDOUT, env=dict(self.env) if self.env else None, start_new_session=True,
        )

    def wait_ready(self) -> None:
        import httpx

        assert self.proc is not None
        deadline = self._clock() + self.ready_timeout_s
        while True:
            code = self.proc.poll()
            if code is not None:
                raise PathFailed(f"{self.name} exited with status {code} before it was ready. Last lines of {self.log_path.name}:\n{tail(self.log_path)}")
            try:
                if httpx.get(self.health_url, timeout=5.0).status_code == 200:
                    log(f"{self.name} is ready")
                    return
            except httpx.HTTPError:
                pass
            if self._clock() > deadline:
                raise PathFailed(f"{self.name} was not ready after {self.ready_timeout_s:.0f} s. Last lines of {self.log_path.name}:\n{tail(self.log_path)}")
            self._sleep(self.poll_s)

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        if self._log is not None:
            self._log.close()
            self._log = None

    def __enter__(self) -> ManagedServer:
        self.start()
        try:
            self.wait_ready()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


# --- the run context and the seams tests replace ----------------------------------------------------------------------------------------------------------------


class Context:
    """Directories, the repository, and the pieces that touch the machine. Tests replace `launcher`
    (which starts a server and yields an Endpoint), `prepare` (installs, merges, conversions),
    `bench` and `gpu_query`; `eval_kwargs` are passed to run_eval last (a stub transport, a fake embedder)."""

    def __init__(
        self, *, mode: str, input_root: Path = INPUT_ROOT, working: Path = WORKING, scratch: Path = SCRATCH,
        repo_root: Path | None = None, eval_kwargs: Mapping[str, Any] | None = None, launcher: Callable[..., Any] | None = None,
        prepare: Callable[..., None] | None = None, bench: Callable[..., dict[str, Any]] | None = None,
        gpu_query: Callable[[], list[dict[str, str]]] = query_gpu, start_at: str | None = None, check_only: bool = False,
    ):
        self.mode, self.input_root, self.working, self.scratch = mode, Path(input_root), Path(working), Path(scratch)
        self.repo_root = Path(repo_root) if repo_root else None
        self.eval_kwargs = dict(eval_kwargs or {})
        self.launcher = launcher or launch_session
        self.prepare = prepare or prepare_plan
        self.bench = bench or run_bench
        self.gpu_query, self.start_at, self.check_only = gpu_query, start_at, check_only
        self.logs = self.working / "logs"
        self.repo: Repo | None = None
        self.gpu: dict[str, str] | None = None
        self.engine_versions: dict[str, str | None] = {}
        self.model_dirs: dict[str, Path] = {}  # where the base model's files are, once downloaded


# --- preparing a session: installs, merges, conversions --------------------------------------------------------------------------------------------------------------


def cuda_stub_dirs(candidates: Sequence[str] = CUDA_STUB_DIRS) -> list[str]:
    """The candidate directories that hold libcuda.so, the link-time stub of the CUDA driver library."""
    return [d for d in candidates if (Path(d) / "libcuda.so").is_file()]


def server_env(ctx: Context) -> dict[str, str]:
    env = {
        **os.environ, "CUDA_VISIBLE_DEVICES": "0", "HF_HOME": str(ctx.scratch / "hf"), "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    stubs = cuda_stub_dirs()
    if stubs:  # for linking kernels built at startup only: at run time the real driver library is loaded
        env["LIBRARY_PATH"] = os.pathsep.join([*stubs, *filter(None, [os.environ.get("LIBRARY_PATH")])])
    return env


def session_env(ctx: Context, plan: SessionPlan) -> dict[str, str]:
    """server_env plus what one path needs: the snapshot's src/ for hf_server, Triton attention for vllm-lora-triton."""
    env = server_env(ctx)
    if plan.path == PATH_HF:  # finetune_vs_api.hf_server is imported from the snapshot's src/, in a process of its own
        assert ctx.repo is not None
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ctx.repo.root / "src"), env.get("PYTHONPATH")]))
    if plan.path == PATH_VLLM_LORA_TRITON:
        env["VLLM_ATTENTION_BACKEND"] = TRITON_ATTENTION
    return env


def vllm_python(ctx: Context) -> Path:
    return ctx.scratch / "vllm-env" / "bin" / "python"


def install_into(ctx: Context, python: str | Path, packages: Sequence[str], log_name: str) -> None:
    """pip install into the environment of `python`. The vLLM venv has no pip (uv made it), so uv installs into that one."""
    python = str(python)
    if Path(python) == vllm_python(ctx):
        argv = [sys.executable, "-m", "uv", "pip", "install", "-q", "--python", python, *packages]
    else:
        argv = [python, "-m", "pip", "install", "-q", *packages]
    run_cmd(argv, log_path=ctx.logs / log_name, env=server_env(ctx))


def ensure_runtime_dependencies(ctx: Context, *, find_spec: Callable[[str], Any] = importlib.util.find_spec) -> list[str]:
    """What the repository's package needs to import: normally already on Kaggle. Returns what was installed."""
    missing = [pkg for module, pkg in (("numpy", "numpy"), ("yaml", "pyyaml"), ("httpx", "httpx")) if find_spec(module) is None]
    if missing:
        install_into(ctx, sys.executable, missing, "install-runtime.log")
    return missing


def ensure_fastembed(ctx: Context, *, find_spec: Callable[[str], Any] = importlib.util.find_spec) -> None:
    if find_spec("fastembed") is None:
        install_into(ctx, sys.executable, [FASTEMBED_PIN], "install-fastembed.log")


def ensure_vllm_env(ctx: Context) -> Path:
    """The venv holding the pinned vLLM (built once), and the vLLM version it really has."""
    env_dir = ctx.scratch / "vllm-env"
    marker = env_dir / ".ready"
    if not marker.exists():
        for argv in vllm_install_commands(env_dir):
            run_cmd(argv, log_path=ctx.logs / "install-vllm.log", env={**server_env(ctx), "UV_NO_CACHE": "1"})
        marker.write_text(VLLM_VERSION)
    done = subprocess.run(
        [str(env_dir / "bin" / "python"), "-c", "import importlib.metadata as m; print(m.version('vllm')); print(m.version('torch'))"],
        capture_output=True, text=True,
    )
    found = done.stdout.split()
    if done.returncode != 0 or not found or found[0] != VLLM_VERSION:
        raise PathFailed(f"expected vllm {VLLM_VERSION} in {env_dir}, found {found[:1] or done.stderr[-300:]}")
    ctx.engine_versions["vllm"] = found[0]
    ctx.engine_versions["torch"] = found[1] if len(found) > 1 else None
    return env_dir / "bin"


def convert_python(ctx: Context) -> Path:
    """The interpreter that runs llama.cpp's converter: the vLLM venv (torch and transformers 4.56.2) if there is one."""
    return vllm_python(ctx) if vllm_python(ctx).exists() else Path(sys.executable)


def ensure_llama_cpp(ctx: Context) -> Path:
    """Download the pinned llama.cpp source and build llama-server with CUDA for a T4 (compute capability 7.5)."""
    src = ctx.scratch / f"llama.cpp-{LLAMA_CPP_TAG}"
    binary = src / "build" / "bin" / "llama-server"
    if binary.exists():
        return binary
    archive = ctx.scratch / f"llama.cpp-{LLAMA_CPP_TAG}.tar.gz"
    if not src.exists():
        ctx.scratch.mkdir(parents=True, exist_ok=True)
        log(f"downloading {LLAMA_CPP_URL}")
        urllib.request.urlretrieve(LLAMA_CPP_URL, archive)
        unpacked = ctx.scratch / "llama.cpp-unpack"
        try:
            extract_archive(archive, unpacked)
        except Refused as exc:
            raise PathFailed(str(exc)) from None
        (next(unpacked.iterdir())).rename(src)
    for argv in llama_build_commands(src, src / "build", os.cpu_count() or 4):
        run_cmd(argv, log_path=ctx.logs / "build-llama-cpp.log", env=server_env(ctx))
    install_into(ctx, convert_python(ctx), [str(src / "gguf-py"), "sentencepiece"], "build-llama-cpp.log")
    if not binary.exists():
        raise PathFailed(f"llama.cpp built, but {binary} does not exist")
    ctx.engine_versions["llama.cpp"] = LLAMA_CPP_TAG
    return binary


def package_version(python: str | Path, package: str) -> str | None:
    """The installed version of a package in another interpreter, without importing it; None if there is none."""
    try:
        done = subprocess.run([str(python), "-c", f"import importlib.metadata as m; print(m.version({package!r}))"], capture_output=True, text=True)
    except OSError:
        return None
    return (done.stdout.strip() or None) if done.returncode == 0 else None


def prepare_plan(ctx: Context, plan: SessionPlan, revision: str, base_model: str) -> None:
    """Carry out a plan's prep steps (each is skipped when its result already exists)."""
    assert ctx.repo is not None
    os.environ.setdefault("HF_HOME", str(ctx.scratch / "hf"))
    for step in plan.prep:
        variant = step.variant
        if step.kind == "vllm_env":
            ensure_vllm_env(ctx)
        elif step.kind == "base_snapshot":
            if "base" not in ctx.model_dirs:
                from huggingface_hub import snapshot_download

                # in the cache the servers read (HF_HOME/hub), so the 8 GB are downloaded once
                ctx.model_dirs["base"] = Path(snapshot_download(base_model, revision=revision, cache_dir=str(ctx.scratch / "hf" / "hub")))
        elif step.kind == "merge":
            out = ctx.scratch / "merged" / variant.key
            if not (out / "config.json").exists():
                log(f"merging {variant.adapter} into fp16 weights -> {out}")
                try:
                    report = ctx.repo.lora_merge.merge_adapter(ctx.model_dirs["base"], variant.adapter, out)
                except ctx.repo.lora_merge.MergeError as exc:
                    raise PathFailed(f"cannot merge {variant.adapter}: {exc}") from None
                log(f"  merged {report['merged_modules']} modules (scale {report['scale']:g}) into {report['shards']} shards")
        elif step.kind == "llama_cpp":
            ensure_llama_cpp(ctx)
        elif step.kind == "gguf":
            src = ctx.scratch / f"llama.cpp-{LLAMA_CPP_TAG}"
            out = ctx.scratch / "gguf" / f"{variant.key}.gguf"
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                model_dir = ctx.scratch / "merged" / variant.key if variant.adapter is not None else ctx.model_dirs["base"]
                run_cmd(convert_gguf_command(str(convert_python(ctx)), src, model_dir, out), log_path=ctx.logs / f"convert-{variant.key}.log")
        else:
            raise ValueError(f"unknown prep step {step.kind!r}")


@contextlib.contextmanager
def launch_session(ctx: Context, plan: SessionPlan, revision: str, base_model: str) -> Iterator[Endpoint]:
    """Prepare the session, start its server, and yield the Endpoint. Whatever happens, the server is stopped and
    the GPU memory is waited for before returning."""
    ctx.prepare(ctx, plan, revision, base_model)
    assert ctx.repo is not None
    env = session_env(ctx, plan)
    if plan.path == PATH_HF:
        for package in ("transformers", "torch"):
            ctx.engine_versions[package] = package_version(sys.executable, package)
    health = plan.base_url.rsplit("/v1", 1)[0] + "/health"
    server = ManagedServer(plan.name, plan.argv, env=env, log_path=ctx.logs / plan.log, health_url=health)
    try:
        with server:
            yield Endpoint(plan.base_url)
    finally:
        wait_gpu_idle()


# --- probing and running -----------------------------------------------------------------------------------------------------------------------------------------------


def probe_requests(repo: Repo) -> list[tuple[str, list[dict[str, str]]]]:
    """Requests built from dev and train examples only: a few short fine-tune-style prompts and a few of the
    long k=10 prompts (the first ten train examples as shots, no retrieval needed)."""
    dev = repo.data.read_examples(repo.processed / "dev.jsonl")
    train = repo.data.read_examples(repo.processed / "train.jsonl")
    inventory = repo.schema.load_inventory(repo.processed)
    short = [("short", repo.prompts.render_messages(FT_PROMPT, e.text)) for e in dev[:PROBE_SHORT]]
    shots = train[:10]
    long = [
        ("long", repo.prompts.render_messages("fewshot_k10_v1", e.text, inventory=inventory, shots=shots))
        for e in dev[PROBE_SHORT:PROBE_SHORT + PROBE_LONG]
    ]
    return short + long


def probe_endpoint(repo: Repo, endpoint: Endpoint, variants: Sequence[Variant], *, max_tokens: int = 256) -> list[dict[str, Any]]:
    """Send the probe requests to each variant's model name and judge the answers. PathFailed on the
    first variant whose answers look like garbage."""
    import httpx

    requests = probe_requests(repo)
    inventory = repo.schema.load_inventory(repo.processed)

    def is_valid(text: str | None) -> bool:
        parsed = repo.schema.parse_output(text)
        return parsed is not None and repo.schema.is_schema_valid(parsed, inventory)

    report = []
    with httpx.Client(base_url=endpoint.base_url, transport=endpoint.transport, timeout=600.0) as http:
        for variant in variants:
            replies = []
            for kind, messages in requests:
                try:
                    response = http.post("/chat/completions", json={"model": variant.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0})
                    response.raise_for_status()
                    text = response.json()["choices"][0]["message"]["content"]
                    replies.append({"kind": kind, "text": text, "error": None})
                except (httpx.HTTPError, KeyError, ValueError) as exc:
                    replies.append({"kind": kind, "text": None, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            ok, why = assess_probe(replies, expect_schema=variant.adapter is not None, is_valid=is_valid)
            report.append({"variant": variant.key, "model": variant.model, "ok": ok, "why": why})
            log(f"probe {variant.model}: {why}")
            if not ok:
                raise PathFailed(f"probe of {variant.model} failed: {why}")
    return report


def eval_options(ctx: Context, endpoint: Endpoint, *, config_dir: Path, results_dir: Path, lock_path: Path | None) -> dict[str, Any]:
    assert ctx.repo is not None

    def progress(done: int, total: int, row: Mapping[str, Any]) -> None:
        if done % 250 == 0 or done == total or row.get("error"):
            log(f"  [{done}/{total}]" + (f" FAILED {row.get('error_kind')}" if row.get("error") else ""))

    options: dict[str, Any] = {
        "config_dir": config_dir, "processed_dir": ctx.repo.processed, "results_dir": results_dir,
        "cache_dir": ctx.scratch / "cache" / "retrieval", "state_dir": ctx.scratch / "cache" / "ratelimit",
        "announce": log, "progress": progress,
    }
    if lock_path is not None:
        options["lock_path"] = lock_path
    if endpoint.transport is not None:
        options["transport"] = endpoint.transport
    return {**options, **ctx.eval_kwargs}


def run_subsets(
    ctx: Context, endpoint: Endpoint, variant: Variant, split: str, subsets: Sequence[str], *, config_dir: Path,
    results_dir: Path, lock_path: Path | None,
) -> dict[str, dict[str, Any]]:
    """run_eval on the first subset (the whole split), then on each nested subset: the predictions are shared,
    so nothing is sent twice and each later call only writes its summary. PathFailed if a run stops or its
    output is implausible."""
    assert ctx.repo is not None
    summaries: dict[str, dict[str, Any]] = {}
    for index, subset in enumerate(subsets):
        outcome = ctx.repo.evaluate.run_eval(
            variant.system, split, subset=subset, resume=index > 0,
            **eval_options(ctx, endpoint, config_dir=config_dir, results_dir=results_dir, lock_path=lock_path),
        )
        if outcome.status != "complete" or outcome.summary["status"] != "complete":
            raise PathFailed(f"{variant.system} on {split}/{subset} stopped ({outcome.status}): {outcome.batch.message}")
        summaries[subset] = {"summary": outcome.summary, "path": outcome.summary_path, "predictions": outcome.predictions_path}
        if index == 0:
            rows = ctx.repo.evaluate.latest_rows(ctx.repo.evaluate.read_rows(outcome.predictions_path))
            reason = implausible(assess_predictions(rows), outcome.summary)
            if reason:
                raise PathFailed(f"{variant.model} on {split}: {reason}")
    return summaries


# --- the two modes ---------------------------------------------------------------------------------------------------------------------------------------------------------


class LockedAdapter(NamedTuple):
    """The adapter test mode serves: the one the locked checkpoint names."""

    epoch: int
    path: str  # checkpoint.adapter as locked, e.g. adapters/epoch-2
    info: dict[str, Any]  # read_adapter()
    dev_select: dict[str, Any] | None  # what check_dev_select found


class Preflight(NamedTuple):
    repo: Repo
    adapters: dict[int, dict[str, Any]]  # epoch -> read_adapter()
    revision: str
    base_model: str
    host: str
    port: int
    train_log: Mapping[str, Any] | None
    locks: dict[str, dict[str, Any]] | None = None
    dev_select: Mapping[str, Any] | None = None
    locked_adapter: LockedAdapter | None = None


def require_locks(repo: Repo) -> dict[str, dict[str, Any]]:
    """The test lock entries for all six (system, subset) pairs, or Refused with status 3: the one gate between this
    script and the test split, checked before anything else is looked at."""
    problems = lock_problems(repo)
    if problems:
        raise Refused(
            "the test split is locked. What is missing, or no longer matches:\n" + "\n".join(f"  - {problem}" for problem in problems)
            + "\nLock all six pairs (python scripts/lock_test.py --write --system NAME --subset full|S500|S300 --reason '...'), "
            "commit and push, and make a new snapshot.",
            EXIT_LOCKED,
        )
    return lock_entries(repo)


def preflight(ctx: Context) -> Preflight:
    """Everything that can refuse a run, before anything is installed or started. In test mode the locks come first."""
    root = ctx.repo_root or find_repo_root(ctx.input_root, ctx.scratch / "snapshot")
    if not ctx.check_only:
        ensure_runtime_dependencies(ctx)
    repo = Repo(root)
    ctx.repo = repo
    log(f"repository snapshot: {root} ({len(repo.manifest())} files that the run depends on)")

    ft, base = (repo.config.resolve_system(name, repo.config_dir) for name in (FT_SYSTEM, BASE_SYSTEM))
    host, port = server_address(ft["base_url"])
    locks = require_locks(repo) if ctx.mode == "test" else None
    train_cfg = repo.config.load_yaml("train", repo.config_dir)
    if base["model"] != train_cfg["base_model"]["name"]:
        raise Refused(f"{BASE_SYSTEM} serves {base['model']!r}, but the adapters were trained on {train_cfg['base_model']['name']!r}")

    adapter_dirs = find_adapters(ctx.input_root)
    if not adapter_dirs:
        raise Refused(
            f"no adapters (epoch-N/adapter_config.json) under {ctx.input_root}: is the training kernel "
            f"'{TRAIN_KERNEL_SLUG}' attached to this kernel (kernel-metadata.json kernel_sources)?"
        )
    adapters = {epoch: read_adapter(directory) for epoch, directory in adapter_dirs.items()}
    for epoch, info in adapters.items():
        log(f"adapter epoch {epoch}: {info['directory']} (rank {info['rank']}, sha256 {info['sha256'][:12]})")
        if info["base_model_name_or_path"] and info["base_model_name_or_path"] != base["model"]:
            log(f"  note: it names {info['base_model_name_or_path']!r} as its base, which is served here as {base['model']!r}")
    expected = train_cfg["training"]["num_train_epochs"]
    if len(adapters) != expected:
        log(f"warning: configs/train.yaml trains {expected} epochs but {len(adapters)} adapters were found")
    train_log = find_train_log(ctx.input_root)

    locked_revisions = [(row["checkpoint"] or {}).get("base_revision") for row in (ft, base)] if locks else []
    revision = pin_revision(repo, train_log, locked_revisions)
    dev_path = root / DEV_SELECT_FILE
    dev_record = json.loads(dev_path.read_text(encoding="utf-8")) if ctx.mode == "test" and dev_path.exists() else None
    locked = resolve_locked_adapter(ft["checkpoint"], adapters, dev_record) if ctx.mode == "test" else None
    return Preflight(repo, adapters, revision, base["model"], host, port, train_log, locks, dev_record, locked)


def resolve_locked_adapter(checkpoint: Mapping[str, Any], adapters: Mapping[int, Mapping[str, Any]], dev_record: Mapping[str, Any] | None) -> LockedAdapter:
    """The adapter directory the locked checkpoint (adapter: adapters/epoch-N, epoch: N) names, found among the training
    output, and whether it is the file dev-select scored. Refused with status 4 if it is not there."""
    epoch, wanted = checkpoint["epoch"], str(checkpoint["adapter"]).strip("/")
    wanted_parts = PurePosixPath(wanted).parts
    found = adapters.get(epoch) if isinstance(epoch, int) else None
    if found is None or PurePosixPath(found["directory"].as_posix()).parts[-len(wanted_parts):] != wanted_parts:
        raise Refused(
            f"the lock names checkpoint.adapter {wanted!r}, epoch {epoch}: a path inside the training kernel's output. The adapters found are "
            f"{ {e: i['directory'].as_posix() for e, i in adapters.items()} }", EXIT_ADAPTER,
        )
    return LockedAdapter(epoch, wanted, dict(found), check_dev_select(dev_record, epoch, found["sha256"]))


def _bench_python(ctx: Context, path: str) -> str:
    """The vLLM venv has the openai package (vLLM needs it); otherwise this interpreter, which gets it installed."""
    return str(vllm_python(ctx)) if path in (PATH_VLLM_LORA, PATH_VLLM_LORA_TRITON, PATH_VLLM_MERGED) and vllm_python(ctx).exists() else sys.executable


def run_bench(
    ctx: Context, plan: SessionPlan, variant: Variant, *, prompts: Path, out: Path, serving: Mapping[str, Any], extra: Mapping[str, Any],
) -> dict[str, Any]:
    """scripts/bench_throughput.py on the fine-tuned row. A failure here is reported, never fatal: the accuracy
    results do not depend on it."""
    assert ctx.repo is not None
    python = _bench_python(ctx, plan.path)
    try:
        if subprocess.run([python, "-c", "import openai"], capture_output=True).returncode != 0:
            install_into(ctx, python, ["openai"], "install-openai.log")
        spec = ctx.repo.config.resolve_system(variant.system, ctx.repo.config_dir)
        argv = bench_command(
            python, ctx.repo.root, base_url=plan.base_url, model=variant.model, prompts=prompts, out=out, system=variant.system,
            engine=serving["engine"], engine_version=serving.get("engine_version"), dtype=serving["dtype"],
            max_tokens=int(spec["params"].get("max_tokens", 256)), extra=extra,
        )
        run_cmd(argv, log_path=ctx.logs / "bench-throughput.log")
        log("throughput benchmark:\n" + tail(ctx.logs / "bench-throughput.log", 12))
        doc = json.loads(out.read_text(encoding="utf-8"))
    except (CommandFailed, OSError, ValueError) as exc:
        return {"status": "failed", "reason": str(exc)[:2000]}
    return {
        "status": "ok", "file": f"results/serving/{out.name}", "operating_point": doc.get("operating_point"),
        "cost": (doc.get("cost") or {}).get("at_operating_point"),
    }


def write_throughput_prompts(repo: Repo, out: Path) -> int:
    """The fine-tuned row's test requests, as the prompts the benchmark cycles through (labels are not used)."""
    examples = repo.data.read_examples(repo.processed / "test.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        for e in examples:
            handle.write(json.dumps({"messages": repo.prompts.render_messages(FT_PROMPT, e.text)}, ensure_ascii=False) + "\n")
    return len(examples)


def serving_record(path: str, ctx: Context, attempts: list[dict[str, Any]]) -> dict[str, Any]:
    info = PATH_INFO[path]
    versions = ctx.engine_versions
    return {
        "path": path, "engine": info["engine"], "engine_version": versions.get(info["engine"]), "dtype": info["dtype"],
        "quantization": info["quantization"], "weights": info["weights"], "throughput_measured": info["throughput"],
        "vllm_pin": VLLM_PINS if info["engine"] == "vllm" else None,
        "packages": {k: v for k, v in versions.items() if v}, "attempts": attempts,
    }


def promote(attempt_dir: Path, working: Path) -> None:
    """Move the winning attempt's results tree to /kaggle/working/results. What stays under attempts/ is
    the output of the paths that failed, kept for inspection and never reported."""
    source = attempt_dir / "results"
    if source.exists():
        shutil.copytree(source, working / "results", dirs_exist_ok=True)
        shutil.rmtree(source)
    for empty in (attempt_dir, attempt_dir.parent):  # attempts/<path>, then attempts/ unless a failed path left something
        with contextlib.suppress(OSError):
            empty.rmdir()


def fresh_attempt_dir(ctx: Context, path: str) -> Path:
    directory = ctx.working / "attempts" / path
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    return directory


def list_variants(pre: Preflight) -> list[Variant]:
    """What the mode evaluates. Dev: every epoch's adapter, each under its own model name, and the base row. Test: the
    base row and the locked adapter under the name the lock hashes (the fine-tuned row last: the sweep follows both)."""
    repo = pre.repo
    base = Variant("base", BASE_SYSTEM, repo.config.resolve_system(BASE_SYSTEM, repo.config_dir)["model"])
    if pre.locked_adapter is not None:
        ft_model = repo.config.resolve_system(FT_SYSTEM, repo.config_dir)["model"]
        return [base, Variant("ft", FT_SYSTEM, ft_model, pre.locked_adapter.info["directory"], pre.locked_adapter.epoch)]
    epochs = [Variant(f"epoch-{e}", FT_SYSTEM, f"{FT_SYSTEM}-epoch-{e}", info["directory"], e) for e, info in pre.adapters.items()]
    return [*epochs, base]


def run_dev_select(ctx: Context, pre: Preflight) -> int:
    repo = pre.repo
    variants = list_variants(pre)
    if not ctx.eval_kwargs.get("embedder"):
        ensure_fastembed(ctx)

    def attempt(path: str) -> dict[str, Any]:
        directory = fresh_attempt_dir(ctx, path)
        results = directory / "results"
        scores: list[dict[str, Any]] = []
        probes: list[dict[str, Any]] = []
        for plan in plan_sessions(path, variants, base_model=pre.base_model, revision=pre.revision, host=pre.host, port=pre.port, scratch=ctx.scratch):
            with ctx.launcher(ctx, plan, pre.revision, pre.base_model) as endpoint:
                probes += probe_endpoint(repo, endpoint, plan.variants)
                for variant in plan.variants:
                    log(f"dev: {variant.key} ({variant.model}) on the full dev split")
                    config_dir = write_variant_config(repo, ctx.scratch / "config" / variant.key, variant, pre.revision)
                    results_dir = results / "dev-select" / variant.key
                    summaries = run_subsets(ctx, endpoint, variant, "dev", DEV_SUBSETS, config_dir=config_dir, results_dir=results_dir, lock_path=None)
                    full = summaries["full"]
                    scores.append(score_record(variant, full["summary"], full["path"], directory))
                    log(f"  exact match {scores[-1]['exact_match']:.4f} on {scores[-1]['n_scored']} items")
        return {"scores": scores, "probes": probes, "directory": directory}

    path, outcome, attempts = run_ladder(paths_from(ctx.start_at), attempt, on_failure=lambda p, why: log(f"serving path {p} failed: {why[:500]}"))
    chosen = pick_best_epoch(outcome["scores"])
    info = pre.adapters[chosen["epoch"]]
    promote(outcome["directory"], ctx.working)
    record = {
        "schema_version": 1, "mode": "dev-select", "created_at": now_iso(), "rule": SELECTION_RULE,
        "chosen": {
            "epoch": chosen["epoch"], "variant": chosen["variant"], "exact_match": chosen["exact_match"],
            "correct": chosen["correct"], "adapter": f"adapters/epoch-{chosen['epoch']}", "adapter_sha256": info["sha256"],
        },
        "scores": outcome["scores"],
        "adapters": {str(e): {"adapter": f"adapters/epoch-{e}", "sha256": i["sha256"], "rank": i["rank"]} for e, i in pre.adapters.items()},
        "base_model": pre.base_model, "base_revision": pre.revision, "dev_split_items": outcome["scores"][0]["n_scored"],
        "serving": serving_record(path, ctx, attempts), "probes": outcome["probes"],
        "training": {"train_log_found": pre.train_log is not None, "gpu": (pre.train_log or {}).get("gpu"), "adapters": (pre.train_log or {}).get("adapters")},
        "gpu": ctx.gpu, "snapshot_files": repo.manifest(),
    }
    write_json(ctx.working / DEV_SELECT_FILE, record)
    log("dev scores:")
    for s in outcome["scores"]:
        log(f"  {s['variant']:>8}  exact match {s['exact_match']:.4f}  intent {s['intent_accuracy']:.4f}  slot F1 {s['slot_f1']:.4f}  valid {s['schema_valid_rate']:.4f}")
    log(f"chosen: epoch {chosen['epoch']} ({SELECTION_RULE}); served by {path}")
    log("next: copy results/serving/dev_select.json into the repository, fill checkpoint in configs/systems.yaml "
        f"(adapter: adapters/epoch-{chosen['epoch']}, epoch: {chosen['epoch']}, base_revision: {pre.revision}), lock all six pairs, commit and push, then run test mode")
    return 0


def check_dev_select(record: Mapping[str, Any] | None, epoch: int, adapter_sha256: str) -> dict[str, Any] | None:
    """Does the locked adapter match what dev-select scored? A different file for the same epoch is refused:
    the dev scores would not be about the adapter being tested. A different epoch only warns."""
    if record is None:
        log(f"note: {DEV_SELECT_FILE} is not in the snapshot, so the adapter cannot be checked against dev-select")
        return None
    scored = (record.get("adapters") or {}).get(str(epoch))
    chosen = (record.get("chosen") or {}).get("epoch")
    if scored and scored.get("sha256") != adapter_sha256:
        raise Refused(
            f"adapters/epoch-{epoch} is not the file dev-select scored (sha256 {adapter_sha256[:12]}, dev-select saw "
            f"{str(scored.get('sha256'))[:12]}): was the adapter retrained since? Run dev-select again.", EXIT_ADAPTER,
        )
    if chosen is not None and chosen != epoch:
        log(f"warning: the lock is for epoch {epoch} but dev-select's rule chose epoch {chosen}")
    return {"record": DEV_SELECT_FILE, "chosen_epoch": chosen, "locked_epoch": epoch, "sha256_matches": bool(scored)}


def run_test_mode(ctx: Context, pre: Preflight) -> int:
    repo = pre.repo
    assert pre.locked_adapter is not None
    epoch, wanted, match, dev_check = pre.locked_adapter
    variants = list_variants(pre)
    ft = variants[-1]
    if not ctx.eval_kwargs.get("embedder"):
        ensure_fastembed(ctx)
    prompts = ctx.scratch / "throughput_prompts.jsonl"
    write_throughput_prompts(repo, prompts)

    def attempt(path: str) -> dict[str, Any]:
        directory = fresh_attempt_dir(ctx, path)
        results = directory / "results"
        runs: dict[str, dict[str, Any]] = {}
        throughput: dict[str, Any] = {"status": "skipped", "reason": f"{path} is the accuracy-only path: no throughput and no cost figure"}
        probes: list[dict[str, Any]] = []
        for plan in plan_sessions(path, variants, base_model=pre.base_model, revision=pre.revision, host=pre.host, port=pre.port, scratch=ctx.scratch):
            with ctx.launcher(ctx, plan, pre.revision, pre.base_model) as endpoint:
                probes += probe_endpoint(repo, endpoint, plan.variants)
                for variant in plan.variants:
                    log(f"test: {variant.system} on the full test split, then S500 and S300")
                    summaries = run_subsets(ctx, endpoint, variant, "test", TEST_SUBSETS, config_dir=repo.config_dir, results_dir=results, lock_path=repo.lock_path)
                    runs[variant.system] = {
                        subset: {"summary": s["path"].relative_to(directory).as_posix(), "exact_match": s["summary"]["metrics"]["exact_match"],
                                 "n_scored": s["summary"]["n_scored"], "lock": s["summary"]["lock"]}
                        for subset, s in summaries.items()
                    }
                if PATH_INFO[path]["throughput"] and any(v.system == FT_SYSTEM for v in plan.variants):
                    info = PATH_INFO[path]
                    throughput = ctx.bench(
                        ctx, plan, ft, prompts=prompts, out=results / "serving" / f"{GPU_LABEL}.json",
                        serving={"engine": info["engine"], "engine_version": ctx.engine_versions.get(info["engine"]) or (LLAMA_CPP_TAG if info["engine"] == "llama.cpp" else None), "dtype": info["dtype"]},
                        extra={"serving_path": path, "quantization": info["quantization"], "adapter_epoch": epoch,
                               "adapter_sha256": match["sha256"], "base_revision": pre.revision, "weights": info["weights"]},
                    )
                    if throughput["status"] != "ok":
                        log(f"WARNING: the throughput benchmark failed ({throughput.get('reason')}); the accuracy results stand, the cost figure does not exist")
        return {"runs": runs, "throughput": throughput, "probes": probes, "directory": directory}

    path, outcome, attempts = run_ladder(paths_from(ctx.start_at or (pre.dev_select or {}).get("serving", {}).get("path")), attempt, on_failure=lambda p, why: log(f"serving path {p} failed: {why[:500]}"))
    promote(outcome["directory"], ctx.working)
    record = {
        "schema_version": 1, "mode": "test", "created_at": now_iso(), "serving": serving_record(path, ctx, attempts),
        "locks": pre.locks,
        "adapter": {"path": wanted, "epoch": epoch, "sha256": match["sha256"], "dev_select": dev_check},
        "base_model": pre.base_model, "base_revision": pre.revision, "runs": outcome["runs"], "throughput": outcome["throughput"],
        "probes": outcome["probes"], "gpu": ctx.gpu, "snapshot_files": repo.manifest(),
    }
    write_json(ctx.working / "test_run.json", record)
    for system, subsets in outcome["runs"].items():
        for subset, r in subsets.items():
            log(f"  {system:>20} {subset:>5}  exact match {r['exact_match']['value']:.4f} on {r['n_scored']} items")
    log(f"served by {path}; throughput: {outcome['throughput']['status']}")
    return 0


# --- the command line ------------------------------------------------------------------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--mode", choices=MODES, help="overrides the MODE constant")
    parser.add_argument("--check", action="store_true", help="run every check that can refuse a run, with no GPU, and exit")
    parser.add_argument("--repo-root", type=Path, help="use this repository snapshot instead of looking under --input-root")
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT, help="where Kaggle mounts datasets and kernel outputs")
    parser.add_argument("--working", type=Path, default=WORKING, help="where everything is written")
    parser.add_argument("--scratch", type=Path, default=SCRATCH, help="large temporary files")
    parser.add_argument("--start-at", choices=SERVING_PATHS, help="skip the serving paths before this one")
    return parser


def main(argv: list[str] | None = None, **context_overrides: Any) -> int:
    """`context_overrides` go to Context (tests pass a launcher, eval_kwargs, ...)."""
    # Whatever starts a Kaggle kernel may add arguments of its own (a Jupyter kernel passes -f <file>): unknown ones are ignored, and said so.
    args, unknown = build_parser().parse_known_args(argv)
    if unknown:
        log(f"ignoring arguments this script does not take: {unknown}")
    mode = args.mode or MODE
    if mode not in MODES:
        log(f"error: no mode: set MODE at the top of this file to one of {MODES} before pushing, or pass --mode")
        return EXIT_REFUSED
    ctx = Context(
        mode=mode, input_root=args.input_root, working=args.working, scratch=args.scratch, repo_root=args.repo_root,
        start_at=args.start_at, check_only=args.check, **context_overrides,
    )
    try:
        if not ctx.check_only:
            ctx.gpu = require_t4(ctx.gpu_query)
        pre = preflight(ctx)
        if ctx.check_only:
            for session in plan_sessions(paths_from(args.start_at)[0], list_variants(pre), base_model=pre.base_model, revision=pre.revision, host=pre.host, port=pre.port, scratch=ctx.scratch):
                log(f"would start: {shlex.join(session.argv or [])}")
            log(f"check passed for {mode}: {len(pre.adapters)} adapters, base revision {pre.revision}"
                + ("" if pre.locks is None else f", {sum(len(v) for v in pre.locks.values())} locks match"))
            return 0
        ctx.working.mkdir(parents=True, exist_ok=True)
        ctx.scratch.mkdir(parents=True, exist_ok=True)
        return run_dev_select(ctx, pre) if mode == "dev-select" else run_test_mode(ctx, pre)
    except Refused as exc:
        log(f"REFUSED: {exc}")
        return exc.code
    except NoServingPath as exc:
        write_json(ctx.working / "attempts" / "failed.json", {"mode": mode, "created_at": now_iso(), "attempts": exc.attempts})
        log(f"FAILED: {exc}")
        return EXIT_NO_PATH


if __name__ == "__main__":
    raise SystemExit(main())
