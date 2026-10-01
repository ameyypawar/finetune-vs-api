"""LoRA fine-tune Qwen3-4B-Instruct-2507 on a Kaggle GPU, then save one adapter per epoch.

Self-contained on purpose: Kaggle runs this one file, so it imports nothing from this
repository. Its hyperparameters are copied from configs/train.yaml, and
tests/test_kaggle_files.py fails if the two ever differ.

Run it (this has not been run yet):

    1. Put sft_train.jsonl and sft_dev.jsonl (from scripts/prepare_data.py) next to
       kaggle/dataset-metadata.json in a folder, and create the private dataset:

           kaggle datasets create -p <that folder>

    2. Push this script as a private batch kernel with a T4 GPU and internet on:

           kaggle kernels push -p kaggle/

    3. When it finishes, download /kaggle/working: one adapter per epoch under adapters/,
       and train_log.json (config, package versions, GPU, losses). Pick the epoch on the dev
       split, then pin it in configs/systems.yaml before locking.

What it does, in order, stopping at the first thing that is wrong:

    * refuses to run until BASE_REVISION is pinned to a Hugging Face commit hash;
    * logs the GPU and refuses anything that is not T4- or L4-class (a P100 is below
      Unsloth's minimum);
    * installs exact pinned versions of the training stack;
    * trains with Unsloth's FastLanguageModel and TRL's SFTTrainer on prompt/completion records,
      so the loss is on the completion only, in fp16 on a T4 (bf16 where the GPU has it);
    * saves an adapter after every epoch and writes train_log.json.

It never pushes anything to the Hugging Face Hub, and uses no token: the base model is public.

    python train_on_kaggle.py --check DIR     validate the data files in DIR and exit; needs no GPU
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Kaggle's "GPU T4 x2" shows two devices; Unsloth trains on one, and the Trainer would
# otherwise wrap the model in DataParallel. Set before anything imports torch.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("WANDB_DISABLED", "true")

# --- what to train (copied from configs/train.yaml) ------------------------------------------------

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
#: Hugging Face commit hash of BASE_MODEL. Placeholder: pin it (here and in configs/train.yaml)
#: before the first run, so the adapter can be traced to the exact weights it was trained on.
BASE_REVISION: str | None = None

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
#: "All linear layers" for Qwen3: the four attention and three MLP projections.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 2
PER_DEVICE_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2  # 8 x 2 = an effective batch of 16
MAX_SEQ_LENGTH = 512
SEED = 3407
LR_SCHEDULER_TYPE = "linear"
WARMUP_RATIO = 0.03  # train.yaml calls it warmup_ratio; passed to TrainingArguments as warmup_steps
WEIGHT_DECAY = 0.01
OPTIM = "adamw_8bit"
LOAD_IN_4BIT = True  # the base is quantized to fit a 16 GB T4 (QLoRA)

# --- environment -------------------------------------------------------------------------------------

DATASET_SLUG = "finetune-vs-api-sft"  # see dataset-metadata.json
TRAIN_FILE = "sft_train.jsonl"
EVAL_FILE = "sft_dev.jsonl"
KAGGLE_INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
TRAINER_DIR = Path("/tmp/trainer")  # checkpoints we do not keep stay out of the saved output

#: Exact versions of the training stack. unsloth caps trl at 0.24.0 and transformers at 5.5.0,
#: and these eight resolve together (checked with `uv pip compile` for Linux, Python 3.11 and
#: 3.12). torch is deliberately not pinned: Kaggle ships one that satisfies them.
PINNED_PACKAGES = [
    "unsloth==2026.9.12",
    "unsloth_zoo==2026.9.8",
    "trl==0.24.0",
    "transformers==5.5.0",
    "peft==0.21.1",
    "accelerate==1.15.0",
    "bitsandbytes==0.50.2",
    "datasets==4.3.0",
]

SUPPORTED_GPU = re.compile(r"\b(T4|L4)\b", re.IGNORECASE)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --- data ----------------------------------------------------------------------------------------------


def chat_to_prompt_completion(record: dict) -> dict:
    """`{"messages": [system?, user, assistant]}` -> `{"prompt": [...], "completion": [...]}`.

    TRL's conversational prompt/completion format: trained this way, the loss falls on the
    completion only. Identical to `finetune_vs_api.data.chat_to_prompt_completion` (a test
    keeps them in step), and as strict: anything but a single user/assistant exchange raises.
    """
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("record needs a messages list ending with an assistant turn")
    roles = [m.get("role") for m in messages]
    if roles[-1] != "assistant" or roles.count("assistant") != 1 or roles.count("user") != 1:
        raise ValueError(f"expected [system?, user, assistant], got roles {roles}")
    if "system" in roles[1:]:
        raise ValueError("a system message may only come first")
    if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
        raise ValueError("every message needs non-empty string content")
    clean = [{"role": m["role"], "content": m["content"]} for m in messages]
    return {"prompt": clean[:-1], "completion": clean[-1:]}


def find_data_dir(root: Path = KAGGLE_INPUT) -> Path:
    """The directory holding the SFT files: the expected mount, else wherever they turn up."""
    expected = root / DATASET_SLUG
    if (expected / TRAIN_FILE).exists():
        return expected
    for hit in sorted(root.rglob(TRAIN_FILE)) if root.exists() else []:
        return hit.parent
    raise SystemExit(
        f"{TRAIN_FILE} not found under /kaggle/input: is the private dataset "
        f"'{DATASET_SLUG}' attached to this kernel (kernel-metadata.json dataset_sources)?"
    )


def read_records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def check_data(directory: Path) -> dict:
    """Validate both SFT files and say what they hold. Needs nothing but the standard library."""
    report = {}
    for name in (TRAIN_FILE, EVAL_FILE):
        path = directory / name
        if not path.exists():
            raise SystemExit(f"{path} is missing")
        records = read_records(path)
        converted = [chat_to_prompt_completion(r) for r in records]  # raises on the first bad record
        report[name] = {"records": len(converted), "sha256": sha256_of(path)}
    return report


# --- hardware --------------------------------------------------------------------------------------------


def is_supported_gpu(name: str) -> bool:
    """T4- or L4-class. A P100 (compute capability 6.0) is below Unsloth's minimum."""
    return bool(SUPPORTED_GPU.search(name or ""))


def query_gpu() -> list[dict]:
    """GPU name, memory, compute capability and driver, from nvidia-smi (no torch import yet)."""
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


def require_supported_gpu() -> dict:
    gpus = query_gpu()
    log(f"GPU(s) visible: {gpus or 'none'}")
    if not gpus:
        raise SystemExit("no GPU found: the kernel needs a GPU accelerator (kernel-metadata.json machine_shape)")
    gpu = gpus[0]
    if not is_supported_gpu(gpu["name"]):
        raise SystemExit(
            f"unsupported GPU {gpu['name']!r} (compute capability {gpu['compute_capability']}): "
            "this script runs on T4- or L4-class GPUs only. A P100 is below Unsloth's minimum."
        )
    return gpu


# --- packages ----------------------------------------------------------------------------------------------


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def install_pinned_packages() -> dict:
    """pip-install the exact pins that are not already satisfied, and return what is installed."""
    missing = [p for p in PINNED_PACKAGES if installed_version(p.split("==")[0]) != p.split("==")[1]]
    if missing:
        log(f"installing: {' '.join(missing)}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINNED_PACKAGES], check=True)
    versions = {p.split("==")[0]: installed_version(p.split("==")[0]) for p in PINNED_PACKAGES}
    for name in ("torch", "xformers", "tokenizers", "huggingface-hub"):
        versions[name] = installed_version(name)
    log(f"versions: {versions}")
    return versions


# --- training ------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--check", type=Path, metavar="DIR", help="validate the SFT files in DIR and exit")
    args = parser.parse_args(argv)
    if args.check:
        for name, info in check_data(args.check).items():
            log(f"{name}: {info['records']} records, sha256 {info['sha256']}")
        return 0

    started = time.time()
    if not BASE_REVISION:
        raise SystemExit(
            "BASE_REVISION is not pinned. Set it to the Hugging Face commit hash of "
            f"{BASE_MODEL} here and in configs/train.yaml, so the adapter is traceable."
        )
    gpu = require_supported_gpu()
    data_dir = find_data_dir()
    data_report = check_data(data_dir)
    log(f"data: {data_dir} {data_report}")
    versions = install_pinned_packages()

    # unsloth first: it patches transformers and trl as it imports.
    from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: I001

    import torch
    from datasets import load_dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit("torch cannot see the GPU even though nvidia-smi can")
    log(f"torch {torch.__version__}, CUDA {torch.version.cuda}, device {torch.cuda.get_device_name(0)}")
    use_bf16 = bool(is_bfloat16_supported())  # False on a T4, so it trains in fp16
    log(f"precision: {'bf16' if use_bf16 else 'fp16'}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        revision=BASE_REVISION,
        use_exact_model_name=True,  # train the pinned Qwen repo, not a mirror Unsloth might substitute
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
    )

    files = {"train": str(data_dir / TRAIN_FILE), "validation": str(data_dir / EVAL_FILE)}
    dataset = load_dataset("json", data_files=files)
    dataset = dataset.map(chat_to_prompt_completion, remove_columns=["messages"])
    log(f"dataset: {dataset}")

    class SaveAdapterEachEpoch(TrainerCallback):
        """Write the LoRA adapter at the end of every epoch, so the epoch can be picked on dev."""

        def __init__(self) -> None:
            self.saved: list[dict] = []

        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            epoch = int(round(state.epoch))
            target = WORKING / "adapters" / f"epoch-{epoch}"
            model.save_pretrained(str(target))
            self.saved.append({"epoch": epoch, "path": str(target), "global_step": state.global_step})
            log(f"saved adapter {target}")
            return control

    saver = SaveAdapterEachEpoch()
    config = SFTConfig(
        output_dir=str(TRAINER_DIR),
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_steps=WARMUP_RATIO,  # transformers 5: a float in [0, 1) is a ratio; `warmup_ratio` is deprecated
        weight_decay=WEIGHT_DECAY,
        optim=OPTIM,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="no",  # the callback saves the adapter; full checkpoints are not kept
        seed=SEED,
        data_seed=SEED,
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        completion_only_loss=True,  # prompt/completion records: the loss is on the completion
        dataset_num_proc=2,
        report_to="none",
        push_to_hub=False,  # nothing leaves this kernel
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=config,
        callbacks=[saver],
    )

    masking: dict = {}
    try:  # evidence for train_log.json that the prompt is masked out of the loss
        sample = trainer.train_dataset[0]
        mask = sample["completion_mask"]
        masking = {"tokens": len(mask), "completion_tokens": int(sum(mask)), "prompt_tokens": len(mask) - int(sum(mask))}
        log(f"first training example: {masking}")
    except Exception as error:  # a logging aid must never stop the run
        log(f"could not inspect the completion mask: {error}")

    result = trainer.train()
    elapsed = time.time() - started

    WORKING.mkdir(parents=True, exist_ok=True)
    train_log = {
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "config": {
            "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "target_modules": TARGET_MODULES},
            "learning_rate": LEARNING_RATE, "epochs": NUM_TRAIN_EPOCHS,
            "per_device_batch_size": PER_DEVICE_BATCH_SIZE, "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "max_seq_length": MAX_SEQ_LENGTH, "seed": SEED, "lr_scheduler_type": LR_SCHEDULER_TYPE,
            "warmup_ratio": WARMUP_RATIO, "weight_decay": WEIGHT_DECAY, "optim": OPTIM, "load_in_4bit": LOAD_IN_4BIT,
            "precision": "bf16" if use_bf16 else "fp16", "completion_only_loss": True,
        },
        "gpu": gpu,
        "packages": versions,
        "data": data_report,
        "completion_mask_first_example": masking,
        "adapters": saver.saved,
        "train_metrics": result.metrics,
        "log_history": trainer.state.log_history,
        "elapsed_seconds": round(elapsed, 1),
    }
    (WORKING / "train_log.json").write_text(json.dumps(train_log, indent=2, default=str) + "\n", encoding="utf-8")
    log(f"done in {elapsed / 60:.1f} min; wrote {WORKING / 'train_log.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
