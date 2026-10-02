"""Synthetic results for the comparison, figure and render tests.

A small processed dataset (conftest.build_processed), test runs written the way
scripts/run_eval.py writes them (the summaries come from the real evaluate.build_summary, so a
change in their shape shows up here), a data audit made by the real data.audit, a throughput
benchmark, a training log and an error-analysis file. Prices and the GPU rental price are round
numbers, so the expected costs can be worked out by hand. Nothing here touches a network or a model.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml

from conftest import ROOT, build_processed
from finetune_vs_api import config, cost, data, evaluate, schema, subsets
from finetune_vs_api.client import BatchResult, latest_rows
from finetune_vs_api.schema import target_json

FT = "ft-qwen3-4b-lora"
BASE = "base-qwen3-4b-k10"
GPT_OSS_20B = "groq-gpt-oss-20b-k10"
GPT_OSS_120B = "groq-gpt-oss-120b-k10"
QWEN_27B = "groq-qwen3.8-27b-k10"
GEMINI = "gemini-3.5-flash-lite-k10"
API_SYSTEMS = (GPT_OSS_20B, GPT_OSS_120B, QWEN_27B, GEMINI)
SYSTEMS = (FT, BASE, *API_SYSTEMS)  # the order of configs/systems.yaml

#: USD per million tokens. Round numbers: a call of 1,000 prompt tokens (800 of them the static
#: prefix) and 100 completion tokens costs exactly 0.0012 (no caching) and 0.0006 (prefix cached)
#: on groq-gpt-oss-20b. groq-qwen3.8-27b has no cached-input price, as in configs/sources.yaml, so
#: its two bounds are equal (0.0024).
ROUND_PRICES = {
    "groq-gpt-oss-20b": {"input": 1.0, "cached_input": 0.25, "output": 2.0},
    "groq-gpt-oss-120b": {"input": 0.5, "cached_input": 0.25, "output": 1.0},
    "groq-qwen3.8-27b": {"input": 2.0, "output": 4.0},
    "google-gemini-3.5-flash-lite": {"input": 1.0, "cached_input": 0.5, "output": 4.0},
}
GPU_ON_DEMAND = 0.5  # USD per hour
USAGE = (1000, 100)  # prompt tokens, completion tokens per call
PREFIX_TOKENS = 800
STUB_MODEL = "stub-model-2026-09-01"
LOCK = {"locked_at": "2026-10-02T09:00:00Z", "reason": "frozen for the test run"}


def wrong_answer(example: data.Example) -> str:
    """A well-formed call that is never an exact match, so it is valid for the schema.

    Items with slots and an even id get the right intent and no slots; every other item gets the wrong
    intent. Intent accuracy and exact match therefore differ, as they do on real data.
    """
    if example.slots and int(example.id) % 2 == 0:
        return json.dumps({"intent": example.intent, "slots": []})
    other = "weather_query" if example.intent != "weather_query" else "alarm_set"
    return json.dumps({"intent": other, "slots": []})


class Lab:
    """A temporary repository layout: processed data, configs, and a results directory."""

    def __init__(self, root: Path, *, planted_overlap: Sequence[int] = (3, 4)):
        self.root = Path(root)
        self.processed = build_processed(self.root)
        self.results = self.root / "results"
        self.config_dir = self.root / "configs"
        shutil.copytree(ROOT / "configs", self.config_dir)
        sources = yaml.safe_load((self.config_dir / "sources.yaml").read_text(encoding="utf-8"))
        for key, usd in ROUND_PRICES.items():
            sources["prices"][key]["usd_per_mtok"] = dict(usd)
        sources["gpu_rental"]["aws-g4dn.xlarge"]["usd_per_hour"]["on_demand"] = GPU_ON_DEMAND
        (self.config_dir / "sources.yaml").write_text(
            yaml.safe_dump(sources, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        self.sources = sources
        self.test = data.read_examples(self.processed / "test.jsonl")
        self.subsets_doc = subsets.load_subsets(self.processed / "subsets.json")
        self.inventory = schema.load_inventory(self.processed)
        self.planted: list[str] = self._plant_overlap(planted_overlap)
        self.by_id = {e.id: e for e in self.test}
        self.all_ids = [e.id for e in sorted(self.test, key=lambda e: subsets.id_sort_key(e.id))]
        self.wrong: dict[str, set[str]] = {}

    # --- the configs -----------------------------------------------------------------------------

    def move_to_subset(self, system: str, subset: str) -> None:
        """Point `system` at another test subset in this lab's copy of configs/systems.yaml.

        Every API row in the real configs runs on S500; S300 is pre-registered in results/subsets.json
        and no row uses it. This is how a test gets a row that is compared on a smaller subset than the
        headline one, the case the comparison, the figure labels and the write-up notes still handle.
        """
        path = self.config_dir / "systems.yaml"
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        doc["systems"][system]["test_subset"] = subset
        path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # --- the data ------------------------------------------------------------------------------

    def _plant_overlap(self, positions: Sequence[int]) -> list[str]:
        """Give some test items the text and label of a train item, as MASSIVE's duplicates do.

        `positions` index the S300 ids (so they are in S300 and S500) with the first one counting
        from 0; one more is planted outside S500, so the full split has more of them than S500.
        """
        train = data.read_examples(self.processed / "train.jsonl")
        s500 = set(self.subsets_doc["subsets"]["S500"]["ids"])
        s300 = self.subsets_doc["subsets"]["S300"]["ids"]
        outside = [e.id for e in self.test if e.id not in s500][:1]
        chosen = [s300[p] for p in positions] + outside
        by_id = {e.id: e for e in self.test}
        for n, item_id in enumerate(chosen):
            source, target = train[n], by_id[item_id]
            by_id[item_id] = data.Example(
                target.id, "test", target.scenario, source.intent, source.text, source.slots
            )
        self.test = [by_id[e.id] for e in self.test]
        data.write_examples(self.processed / "test.jsonl", self.test)
        return sorted(chosen, key=subsets.id_sort_key)

    def ids(self, subset: str) -> tuple[str, ...]:
        return subsets.resolve_subset(self.subsets_doc, subset, "test", [e.id for e in self.test]).ids

    def write_audit(self) -> dict[str, Any]:
        """results/data_audit.json from the real audit of the synthetic splits."""
        examples = [data.read_examples(self.processed / f"{s}.jsonl") for s in ("train", "dev", "test")]
        audit = data.audit([e for rows in examples for e in rows])
        self.results.mkdir(parents=True, exist_ok=True)
        (self.results / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        return audit

    def write_subsets(self) -> Path:
        """results/subsets.json, the tracked copy that scripts/make_subsets.py writes next to the processed one."""
        path = self.results / "subsets.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((self.processed / "subsets.json").read_text(encoding="utf-8"), encoding="utf-8")
        return path

    # --- a test run, written the way run_eval writes it ---------------------------------------------

    def row(
        self, item_id: str, *, wrong: bool = False, model: str = STUB_MODEL, latency: float = 0.5,
        usage: tuple[int, int] = USAGE, cached: int | None = None,
    ) -> dict[str, Any]:
        example = self.by_id[item_id]
        usage_obj = cost.Usage(prompt_tokens=usage[0], completion_tokens=usage[1], cached_tokens=cached)
        return {
            "id": item_id, "config_hash": "0" * 64,
            "text": wrong_answer(example) if wrong else target_json(example),
            "usage": usage_obj.to_dict(), "latency_s": latency, "wall_s": latency, "finish_reason": "stop",
            "retries": 0, "model_returned": model, "status_code": 200, "ratelimit_headers": {},
            "error": None, "retriable": None, "finished_at": "2026-10-02T10:00:00Z",
        }

    @staticmethod
    def error_row(item_id: str, kind: str = "http_400") -> dict[str, Any]:
        return {
            "id": item_id, "config_hash": "0" * 64, "text": None, "usage": None, "error": "stub error",
            "error_kind": kind, "status_code": 400, "retriable": False, "retries": 0,
            "ratelimit_headers": {}, "finished_at": "2026-10-02T10:00:00Z",
        }

    def write_run(
        self,
        system: str,
        *,
        wrong: Iterable[str] = (),
        ids: Sequence[str] | None = None,
        errors: Iterable[str] = (),
        models: Callable[[int], str] | None = None,
        latency: Callable[[int], float] | None = None,
        usage: tuple[int, int] = USAGE,
        subset: str | None = None,
        locked: bool = True,
        summary: bool = True,
    ) -> Path:
        """Write predictions.jsonl (and the summary) for `system` on its test subset.

        `ids` limits the rows written, which makes the run partial. `wrong` ids get a well-formed
        wrong answer, `errors` ids a failed call (which counts as wrong). Rows are written in the
        order of `ids`; `models(position)` and `latency(position)` take a 1-based position.
        """
        spec = config.resolve_system(system, self.config_dir)
        subset = subset or spec["test_subset"]
        resolved = subsets.resolve_subset(self.subsets_doc, subset, "test", [e.id for e in self.test])
        ids = list(resolved.ids if ids is None else ids)
        wrong_ids, error_ids = set(wrong), set(errors)
        self.wrong[system] = wrong_ids | error_ids
        rows = []
        for position, item_id in enumerate(ids, start=1):
            if item_id in error_ids:
                rows.append(self.error_row(item_id))
                continue
            rows.append(
                self.row(
                    item_id, wrong=item_id in wrong_ids, usage=usage,
                    model=models(position) if models else STUB_MODEL,
                    latency=latency(position) if latency else 0.1 + 0.01 * (position % 50),
                )
            )
        run_dir = self.results / "runs" / f"{system}__test"
        run_dir.mkdir(parents=True, exist_ok=True)
        predictions = run_dir / "predictions.jsonl"
        predictions.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        if summary:
            self._write_summary(spec, resolved, rows, run_dir, predictions, locked)
        return run_dir

    def _write_summary(self, spec, resolved, rows, run_dir: Path, predictions: Path, locked: bool) -> None:
        components = config.config_components(
            spec["name"], inventory=self.inventory, config_dir=self.config_dir, processed_dir=self.processed
        )
        price = self.sources["prices"].get(spec["price_id"]) if spec["price_id"] else None
        lock = (
            {**LOCK, "config_hash": config.hash_of(components), "subset_hash": resolved.hash} if locked else None
        )
        summary = evaluate.build_summary(
            spec=spec, split="test", subset=resolved, prompt_name=spec["prompt"], inventory=self.inventory,
            examples=[self.by_id[i] for i in resolved.ids], rows=latest_rows(rows), components=components,
            price_entry=price, prefix_tokens=PREFIX_TOKENS if price else 0, batch=BatchResult(status="complete"),
            lock=lock, limit=None, resumed=False, predictions=predictions, with_ci=False,
        )
        suffix = "" if summary["status"] == "complete" else ".partial"
        path = run_dir / f"summary.{resolved.name}{suffix}.json"
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # --- the standard set of results -------------------------------------------------------------

    def populate(self) -> Lab:
        """Test runs for all six systems, the audit, and a benchmark for the fine-tune.

        Which items each system gets wrong follows its position k in the sorted test ids:
        ft k % 10 == 0, base k % 5 in (0, 1), gpt-oss-20b k % 10 == 0 or k % 7 == 0, gpt-oss-120b
        k % 4 == 0, qwen3.8-27b k % 10 == 5 (as many errors as the fine-tune, on other items, so the
        two are level) and gemini k % 20 == 0 (a subset of the fine-tune's errors, so it is ahead).
        Exposed as `lab.wrong[system]`.
        """
        position = {item_id: k for k, item_id in enumerate(self.all_ids)}
        rules: dict[str, Callable[[int], bool]] = {
            FT: lambda k: k % 10 == 0,
            BASE: lambda k: k % 5 in (0, 1),
            GPT_OSS_20B: lambda k: k % 10 == 0 or k % 7 == 0,
            GPT_OSS_120B: lambda k: k % 4 == 0,
            QWEN_27B: lambda k: k % 10 == 5,
            GEMINI: lambda k: k % 20 == 0,
        }
        for system, rule in rules.items():
            spec = config.resolve_system(system, self.config_dir)
            ids = self.ids(spec["test_subset"])
            self.write_run(system, ids=ids, wrong=[i for i in ids if rule(position[i])])
        self.write_audit()
        self.write_subsets()
        self.write_serving()
        return self

    # --- the other inputs -------------------------------------------------------------------------------

    def write_serving(self, name: str = "t4", doc: dict[str, Any] | None = None) -> Path:
        """results/serving/<name>.json: concurrency 1 and 8 for the fine-tune, 8 the operating point.

        At the operating point 10 requests/s on a $0.50/h GPU is 0.5 / 3600 / 10 * 1000 USD per 1,000
        calls, and one GPU serves 10 * 3600 * 730 calls a month.
        """
        doc = doc or {
            "gpu": "Tesla T4",
            "system": FT,
            "levels": [
                {"concurrency": 1, "requests_per_s": 2.0, "latency_s": {"p50": 0.4, "p95": 0.6}, "n": 100},
                {"concurrency": 8, "requests_per_s": 10.0, "latency_s": {"p50": 0.7, "p95": 1.2}, "n": 400},
            ],
            "operating_point": {"concurrency": 8},
        }
        path = self.results / "serving" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        return path

    def write_train_log(self) -> Path:
        """results/train_log.json in the shape kaggle/train_on_kaggle.py writes."""
        log = {
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "base_revision": "0123456789abcdef0123456789abcdef01234567",
            "config": {
                "lora": {"r": 16, "alpha": 32, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
                "learning_rate": 0.0002, "epochs": 2, "per_device_batch_size": 8,
                "gradient_accumulation_steps": 2, "max_seq_length": 512, "seed": 3407,
                "lr_scheduler_type": "linear", "warmup_ratio": 0.03, "weight_decay": 0.01,
                "optim": "adamw_8bit", "load_in_4bit": True, "precision": "fp16", "completion_only_loss": True,
            },
            "gpu": {"name": "Tesla T4", "memory": "15360 MiB", "compute_capability": "7.5", "driver": "550.54"},
            "packages": {"unsloth": "2026.9.1", "trl": "0.24.0", "transformers": "5.5.0", "torch": "2.8.0"},
            "data": {"sft_train.jsonl": {"records": 72, "sha256": "a" * 64}, "sft_dev.jsonl": {"records": 216, "sha256": "b" * 64}},
            "adapters": [{"epoch": 1, "path": "adapters/epoch-1", "global_step": 5}, {"epoch": 2, "path": "adapters/epoch-2", "global_step": 10}],
            "train_metrics": {"train_runtime": 321.5, "train_loss": 0.4321, "epoch": 2.0},
            "log_history": [
                {"loss": 1.2, "epoch": 0.5, "step": 25},
                {"eval_loss": 0.3333, "epoch": 1.0, "step": 5},
                {"eval_loss": 0.2222, "epoch": 2.0, "step": 10},
                {"train_loss": 0.4321, "epoch": 2.0, "step": 10},
            ],
            "elapsed_seconds": 612.3,
        }
        path = self.results / "train_log.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
        return path

    def write_error_analysis(self, categories: Sequence[str]) -> Path:
        """results/error_analysis.csv: one row per hand-labelled error, with a category column."""
        path = self.results / "error_analysis.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "system", "category", "note"])
            for n, category in enumerate(categories):
                writer.writerow([self.all_ids[n], FT, category, "checked by hand"])
        return path
