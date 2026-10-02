"""Merge a LoRA adapter into a model's weights with numpy and the safetensors file format alone.

Used by the serving fallbacks (kaggle/serve/serve_eval_on_kaggle.py): a server that cannot apply an
adapter at request time is given weights with the adapter already in them. No torch, transformers
or peft is needed, so nothing has to be installed next to a running process and nothing depends on
the version of the library that wrote the adapter.

The arithmetic is PEFT's. For every adapted linear layer, with A of shape (r, in) and B of shape
(out, r),

    W' = W + scale * (B @ A)        scale = lora_alpha / r      (lora_alpha / sqrt(r) with rsLoRA)

computed in float32 and stored as float16. Every other tensor is stored as float16 as it is. A value
that does not fit in float16 (or is not finite to begin with) raises MergeError instead of being
stored as an infinity: that is the fp16 overflow a T4 cannot avoid. The base model's config,
tokenizer and index files are copied unchanged, so the result is read back by exactly the library
that reads the base model.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import struct
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

#: How a saved PEFT adapter names its tensors: `base_model.model.<module>.lora_A.weight` (some versions
#: keep the adapter name: `.lora_A.default.weight`).
ADAPTER_KEY = re.compile(r"(?:base_model\.model\.)?(?P<module>.+)\.lora_(?P<ab>[AB])(?:\.[^.]+)?\.weight")
FP16_MAX = float(np.finfo(np.float16).max)
_MAX_HEADER = 100 * 1024 * 1024
_ITEM_SIZE = {"F16": 2, "BF16": 2, "F32": 4}


class MergeError(ValueError):
    """The adapter cannot be merged into these weights, or the result would not be usable."""


# --- reading and writing safetensors ----------------------------------------------------------------------


class SafetensorsFile:
    """Read-only, memory-mapped access to the tensors of one .safetensors file as float32 arrays."""

    def __init__(self, path: Path):
        self.path = Path(path)
        size = self.path.stat().st_size
        with open(self.path, "rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise MergeError(f"{self.path.name} is too short to be a safetensors file")
            (length,) = struct.unpack("<Q", raw)
            if length > _MAX_HEADER or 8 + length > size:
                raise MergeError(f"{self.path.name} has an implausible header length {length}")
            try:
                header = json.loads(handle.read(length))
            except ValueError:
                raise MergeError(f"{self.path.name} has an unreadable header") from None
        self.metadata: dict[str, str] | None = header.pop("__metadata__", None)
        self.header: dict[str, dict[str, Any]] = header
        self._start = 8 + length
        self._data = np.memmap(self.path, dtype=np.uint8, mode="r", offset=self._start) if size > self._start else np.zeros(0, np.uint8)
        for name, entry in header.items():
            begin, end = entry["data_offsets"]
            if entry["dtype"] not in _ITEM_SIZE:
                raise MergeError(f"{self.path.name}: tensor {name} has dtype {entry['dtype']}; only F16, BF16 and F32 are handled")
            if not 0 <= begin <= end <= self._data.size or end - begin != _ITEM_SIZE[entry["dtype"]] * math.prod(entry["shape"]):
                raise MergeError(f"{self.path.name}: tensor {name} does not fit its shape and offsets")

    def names(self) -> list[str]:
        """Tensor names in the order their data lies in the file."""
        return sorted(self.header, key=lambda n: self.header[n]["data_offsets"][0])

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.header[name]["shape"])

    def read(self, name: str) -> np.ndarray:
        entry = self.header[name]
        begin, end = entry["data_offsets"]
        raw = np.array(self._data[begin:end])  # a copy: the file may be closed before the array is used
        if entry["dtype"] == "F16":
            values = raw.view("<f2").astype(np.float32)
        elif entry["dtype"] == "F32":
            values = raw.view("<f4").astype(np.float32)
        else:  # BF16 is the top half of a float32
            values = (raw.view("<u2").astype(np.uint32) << 16).view(np.float32)
        return values.reshape(entry["shape"])


def write_fp16_safetensors(
    path: Path, shapes: Mapping[str, Sequence[int]], produce: Callable[[str], np.ndarray], metadata: Mapping[str, str] | None = None
) -> None:
    """Write a float16 safetensors file with the tensors of `shapes`, in that order. `produce(name)` makes each tensor
    (float32 or float16) just before it is written, so only one is in memory at a time. MergeError if any value
    would not be finite in float16."""
    header: dict[str, Any] = {}
    offset = 0
    for name, shape in shapes.items():
        size = 2 * math.prod(shape)
        header[name] = {"dtype": "F16", "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    if metadata:
        header["__metadata__"] = dict(metadata)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)  # the data starts on an 8-byte boundary
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for name, shape in shapes.items():
            values = produce(name)
            if tuple(values.shape) != tuple(shape):
                raise MergeError(f"{name}: produced shape {tuple(values.shape)}, expected {tuple(shape)}")
            if not np.isfinite(values).all():
                raise MergeError(f"{name}: not finite before conversion to float16")
            with np.errstate(over="ignore"):
                half = values.astype("<f2")
            if not np.isfinite(half).all():
                raise MergeError(f"{name}: a value exceeds the float16 range (|x| > {FP16_MAX:.0f}): fp16 overflow")
            handle.write(half.tobytes())


# --- the adapter -----------------------------------------------------------------------------------------------


def lora_scale(config: Mapping[str, Any]) -> float:
    """alpha / r (alpha / sqrt(r) for rsLoRA), refusing the adapter variants this merge does not implement."""
    rank, alpha = config.get("r"), config.get("lora_alpha")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1 or not isinstance(alpha, int | float) or isinstance(alpha, bool):
        raise MergeError(f"adapter_config.json needs an integer r and a numeric lora_alpha, got r={rank!r}, lora_alpha={alpha!r}")
    for key in ("use_dora", "lora_bias", "fan_in_fan_out"):
        if config.get(key):
            raise MergeError(f"adapter_config.json sets {key}: that adapter variant cannot be merged here")
    if config.get("modules_to_save"):
        raise MergeError(f"adapter_config.json has modules_to_save {config['modules_to_save']!r}: those layers are not LoRA")
    for key in ("rank_pattern", "alpha_pattern"):
        if config.get(key):
            raise MergeError(f"adapter_config.json has a per-layer {key}: this merge uses one r and one lora_alpha for every layer")
    return float(alpha) / math.sqrt(rank) if config.get("use_rslora") else float(alpha) / rank


def read_adapter(adapter_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    """module path -> {"A": lora_A, "B": lora_B} (float32), for example `model.layers.0.self_attn.q_proj`."""
    weights = SafetensorsFile(Path(adapter_dir) / "adapter_model.safetensors")
    pairs: dict[str, dict[str, np.ndarray]] = {}
    for key in weights.names():
        match = ADAPTER_KEY.fullmatch(key)
        if match is None:
            raise MergeError(f"adapter tensor {key!r} is not a LoRA A or B weight")
        pairs.setdefault(match["module"], {})[match["ab"]] = weights.read(key)
    for module, parts in pairs.items():
        if set(parts) != {"A", "B"}:
            raise MergeError(f"adapter module {module} has only lora_{next(iter(parts))}")
    if not pairs:
        raise MergeError("the adapter holds no LoRA weights")
    return pairs


# --- the merge --------------------------------------------------------------------------------------------------------


def merge_adapter(base_dir: Path, adapter_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Write `base_dir` with the adapter merged into its weights, as float16, to `out_dir`.

    Every *.safetensors shard of the base is rewritten (same file names, same tensor names, same order);
    every other file in `base_dir` is copied as it is. MergeError if the adapter cannot be merged, names a
    layer the base does not have, or a value overflows float16. Returns what was done.
    """
    base_dir, adapter_dir, out_dir = Path(base_dir), Path(adapter_dir), Path(out_dir)
    try:
        config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MergeError(f"cannot read {adapter_dir / 'adapter_config.json'}: {exc}") from None
    scale = lora_scale(config)
    pairs = read_adapter(adapter_dir)
    shards = sorted(base_dir.glob("*.safetensors"))
    if not shards:
        raise MergeError(f"no .safetensors files in {base_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    merged: set[str] = set()
    tensors = 0
    for shard in shards:
        source = SafetensorsFile(shard)
        names = source.names()
        tensors += len(names)

        def produce(name: str, source: SafetensorsFile = source) -> np.ndarray:
            weight = source.read(name)
            module = name[: -len(".weight")] if name.endswith(".weight") else None
            if module in pairs:
                a, b = pairs[module]["A"], pairs[module]["B"]
                if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1] or weight.shape != (b.shape[0], a.shape[1]):
                    raise MergeError(f"{module}: lora_A {a.shape} and lora_B {b.shape} do not fit a weight of shape {weight.shape}")
                weight = weight + scale * (b @ a)
                merged.add(module)
            return weight

        write_fp16_safetensors(out_dir / shard.name, {n: source.shape(n) for n in names}, produce, source.metadata)

    missing = sorted(set(pairs) - merged)
    if missing:
        raise MergeError(f"{len(missing)} adapter modules are not in the base weights, e.g. {missing[:3]}")
    for entry in sorted(base_dir.iterdir()):
        if entry.is_file() and entry.suffix != ".safetensors":
            shutil.copy2(entry, out_dir / entry.name)
    return {"shards": len(shards), "tensors": tensors, "merged_modules": len(merged), "scale": scale, "rank": config["r"]}
