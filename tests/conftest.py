"""Shared test helpers: synthetic data, a fake embedder, a synthetic MASSIVE archive, and a
loader for the scripts in scripts/. Nothing here touches the network or a real model."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import tarfile
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import httpx
import numpy as np
import pytest
import yaml

from finetune_vs_api.data import Example, Slot

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_script(name: str) -> ModuleType:
    """Import scripts/<name>.py as a module so its run()/main() can be called directly."""
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ex(
    id: object,
    split: str,
    intent: str,
    text: str,
    slots: Sequence[tuple[str, str]] = (),
    scenario: str = "alarm",
) -> Example:
    return Example(str(id), split, scenario, intent, text, tuple(Slot(t, v) for t, v in slots))


def fake_embed(texts: Sequence[str]) -> np.ndarray:
    """Deterministic bag-of-words embedding: texts that share words get a high cosine."""
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z0-9']+", text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % 64
            out[row, bucket] += 1.0
    return out


def massive_row(
    id: object,
    partition: str,
    intent: str,
    annot: str,
    scenario: str = "alarm",
    locale: str = "en-US",
) -> dict:
    return {
        "id": str(id),
        "locale": locale,
        "partition": partition,
        "scenario": scenario,
        "intent": intent,
        "utt": re.sub(r"\[[^\[\]]*? : ([^\[\]]*)\]", r"\1", annot),
        "annot_utt": annot,
        "worker_id": "1",
    }


def make_archive(path: Path, rows: Sequence[dict], locale: str = "en-US") -> str:
    """Write a tiny tarball laid out like the real one; return its sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(r) for r in rows).encode()  # no trailing newline, like the real file
    notice = b"# NOTICE\n"
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in ((f"1.1/data/{locale}.jsonl", body), ("1.1/NOTICE.md", notice)):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- a small synthetic dataset on disk ------------------------------------------------------------

SCENARIOS_18 = [f"scenario_{i:02d}" for i in range(18)]
INTENT_POOL = ["alarm_set", "play_music", "weather_query"]
DAYS = ["monday", "tuesday", "friday", "sunday"]


def _synthetic(split: str, per_scenario: int, start: int) -> list[Example]:
    items: list[Example] = []
    n = start
    for scenario in SCENARIOS_18:
        for j in range(per_scenario):
            day, hour = DAYS[(n + j) % len(DAYS)], str(1 + (n % 11))
            slots = [("time", hour)] if j % 3 else [("time", hour), ("date", day)]
            text = f"wake me at {hour} on {day} please {n}" if len(slots) == 2 else f"alarm for {hour} number {n}"
            items.append(ex(n, split, INTENT_POOL[n % 3], text, slots, scenario=scenario))
            n += 1
    return items


def build_processed(root: Path) -> Path:
    """Write a small train/dev/test split plus subsets.json under root/processed."""
    from finetune_vs_api import data
    from finetune_vs_api.subsets import make_subsets

    processed = root / "processed"
    train, dev, test = _synthetic("train", 4, 1000), _synthetic("dev", 12, 5000), _synthetic("test", 40, 9000)
    for name, rows in (("train", train), ("dev", dev), ("test", test)):
        data.write_examples(processed / f"{name}.jsonl", rows)
    (processed / "subsets.json").write_text(json.dumps(make_subsets(dev, test), indent=2))
    return processed


# --- the prepare_data pipeline on a synthetic archive -----------------------------------------------


ROWS = [
    massive_row(0, "test", "alarm_set", "wake me up at [time : five am] [date : this week]", "alarm"),
    massive_row(1, "train", "alarm_set", "wake me at [time : nine am] on [date : friday]", "alarm"),
    massive_row(2, "train", "alarm_set", "set an alarm for [time : 5:30]", "alarm"),
    massive_row(3, "train", "general_greet", "hello there", "general"),
    massive_row(4, "dev", "general_greet", "hi", "general"),
    massive_row(5, "dev", "alarm_set", "wake me at [time : six]", "alarm"),
    massive_row(6, "test", "general_greet", "hello there", "general"),  # shares text with train
]


@pytest.fixture
def prep(tmp_path):
    archive = tmp_path / "stub.tar.gz"
    sha = make_archive(archive, ROWS)
    payload = archive.read_bytes()
    archive.unlink()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    def write_config(sha256=sha, expected=None):
        (config_dir / "data.yaml").write_text(
            yaml.safe_dump(
                {
                    "source": {"name": "MASSIVE", "version": "1.1", "url": "https://stub.example/massive.tar.gz", "sha256": sha256, "license": "CC-BY-4.0"},
                    "locale": "en-US",
                    "expected": expected or {"train": 3, "dev": 2, "test": 2, "intents": 2, "slot_types": 2},
                }
            )
        )

    write_config()
    script = load_script("prepare_data")
    lines: list[str] = []
    paths = dict(
        config_dir=config_dir, raw_dir=tmp_path / "raw", processed_dir=tmp_path / "processed",
        results_dir=tmp_path / "results", transport=httpx.MockTransport(lambda r: httpx.Response(200, content=payload)),
        out=lines.append,
    )

    class P:
        pass

    p = P()
    p.script, p.lines, p.paths, p.sha, p.write_config, p.root = script, lines, paths, sha, write_config, tmp_path
    p.run = lambda **kw: script.run(**{**paths, **kw})
    return p


