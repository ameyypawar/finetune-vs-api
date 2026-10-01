"""The evaluation runner: build each request's prompt, send it, score the answers, write a summary.

`scripts/run_eval.py` is a thin command line over `run_eval` here, so the whole path can be
tested against a stub server.

Layout of a run, for system S, split P and prompt N (the system's own prompt unless a dev run
asks for another):

    results/runs/S__P/predictions.jsonl            raw answers, appended as they arrive
    results/runs/S__P/summary.<subset>.json        metrics, once every item of the subset is done
    results/runs/S__P/summary.<subset>.partial.json  the same, while items are still missing
    (for a prompt other than the system's own: results/runs/S__P__N/...)

Predictions are shared by all subsets of a split, so the nested subsets (D50 in D100, S300 in
S500) never spend a request twice. A summary is per subset. Rows carry the configuration
hash they were produced under, and a resumed run refuses rows from a different one.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from . import config, cost, data, metrics, prompts, schema, subsets
from .client import (
    BatchItem,
    BatchResult,
    ChatClient,
    Endpoint,
    is_final,
    latest_rows,
    read_rows,
    run_batch,
)
from .retrieval import Embedder, RetrievalIndex

EVAL_SPLITS = ("dev", "test")


class EvalError(ValueError):
    """A bad combination of options (not a failure of the run itself)."""


@dataclass
class EvalOutcome:
    status: str  # complete | quota_exhausted | budget | auth_error | too_many_errors
    run_dir: Path
    predictions_path: Path
    summary_path: Path | None
    summary: dict[str, Any] | None
    batch: BatchResult


def run_dir_name(system: str, split: str, prompt: str, system_prompt: str) -> str:
    base = f"{system}__{split}"
    return base if prompt == system_prompt else f"{base}__{prompt}"


def score_rows(
    examples: Sequence[data.Example], rows: Mapping[str, Mapping[str, Any]], inventory: schema.LabelInventory
) -> list[metrics.Item]:
    """Score the examples that have a final row. Failed calls and invalid output count as wrong."""
    items = []
    for example in examples:
        row = rows[example.id]
        parsed = schema.parse_output(row.get("text")) if row.get("error") is None else None
        valid = parsed is not None and schema.is_schema_valid(parsed, inventory)
        items.append(metrics.Item(example.text, schema.target_obj(example), parsed if valid else None, valid))
    return items


def _percentiles(values: Sequence[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "p50": metrics.percentile(values, 50),
        "p95": metrics.percentile(values, 95),
        "n": len(values),
        "method": "nearest-rank",
        "note": "seconds for the successful attempt; excludes rate-limit waits and backoff",
    }


def _stop_info(batch: BatchResult) -> dict[str, Any] | None:
    if batch.status == "complete":
        return None
    reset = datetime.fromtimestamp(batch.reset_at, UTC).isoformat(timespec="seconds") if batch.reset_at else None
    return {"reason": batch.status, "message": batch.message, "reset_at": reset, "remaining": batch.remaining}


def build_summary(
    *,
    spec: Mapping[str, Any],
    split: str,
    subset: subsets.ResolvedSubset,
    prompt_name: str,
    inventory: schema.LabelInventory,
    examples: Sequence[data.Example],
    rows: Mapping[str, Mapping[str, Any]],
    components: Mapping[str, str],
    price_entry: Mapping[str, Any] | None,
    prefix_tokens: int,
    batch: BatchResult,
    lock: Mapping[str, Any] | None,
    limit: int | None,
    resumed: bool,
    predictions: Path,
    with_ci: bool = True,
    n_resamples: int = metrics.BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    done = [e for e in examples if e.id in rows and is_final(rows[e.id])]
    done_rows = {e.id: rows[e.id] for e in done}
    ok = [r for r in done_rows.values() if r.get("error") is None]
    latencies = [r["latency_s"] for r in ok if r.get("latency_s") is not None]
    usages = [cost.Usage.from_dict(r["usage"]) for r in ok]
    usages = [u for u in usages if u.is_complete]

    result_cost: dict[str, Any] = {"billing_basis": cost.BILLING_BASIS}
    if price_entry is not None and usages:
        result_cost.update(cost.cost_summary(usages, cost.Price.from_entry(price_entry), prefix_tokens))
        result_cost["price"] = {"id": spec["price_id"], **price_entry}
    elif price_entry is None:
        result_cost["note"] = "no API price for this system; self-hosted cost comes from the throughput benchmark"

    tokens = {
        "calls_with_reported_usage": sum(1 for u in usages if u.reported and not u.estimated),
        "calls_with_estimated_usage": sum(1 for u in usages if u.estimated),
        "calls_without_usage": len(ok) - len(usages),
    }
    complete = len(done) == len(subset.ids)
    return {
        "system": spec["name"],
        "split": split,
        "subset": {"name": subset.name, "n": len(subset.ids), "hash": subset.hash},
        "status": "complete" if complete else "partial",
        "stop": _stop_info(batch),
        "n_scored": len(done),
        "n_calls_failed": len(done) - len(ok),
        "limit": limit,
        "resumed": resumed,
        "billing_basis": cost.BILLING_BASIS,
        "prompt": {"name": prompt_name, "hash": components["prompt"]},
        "config_hash": config.hash_of(dict(components)),
        "config_components": dict(components),
        "git_commit": config.git_commit(),
        "git_dirty": config.git_dirty(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "lock": None if lock is None else {k: lock[k] for k in ("locked_at", "reason", "config_hash", "subset_hash")},
        "endpoint": {
            "name": spec["endpoint"],
            "base_url": spec["base_url"],
            "model_requested": spec["model"],
            "models_returned": dict(Counter(r.get("model_returned") for r in ok)),
            "tier": spec["tier"],
        },
        "decoding": {
            "params": spec["params"],
            "drop_params": spec["drop_params"],
            "strict_json_schema_sent": spec["supports_json_schema"],
        },
        "metrics": metrics.summarize(score_rows(done, done_rows, inventory), with_ci=with_ci, n_resamples=n_resamples)
        if done
        else None,
        "latency_s": _percentiles(latencies),
        "tokens": tokens,
        "cost": result_cost,
        "calls": {
            "finish_reasons": dict(Counter(r.get("finish_reason") for r in ok)),
            "errors_by_kind": dict(Counter(r.get("error_kind") for r in done_rows.values() if r.get("error"))),
            "retries_total": sum(r.get("retries") or 0 for r in done_rows.values()),
            "last_ratelimit_headers": next((r["ratelimit_headers"] for r in reversed(ok) if r.get("ratelimit_headers")), {}),
        },
        "predictions": str(predictions),
    }


def run_eval(
    system: str,
    split: str,
    *,
    subset: str | None = None,
    prompt: str | None = None,
    limit: int | None = None,
    concurrency: int | None = None,
    max_usd: float | None = None,
    resume: bool = False,
    transport: httpx.AsyncBaseTransport | None = None,
    embedder: Embedder | None = None,
    counter: cost.Counter | None = None,
    environ: Mapping[str, str] | None = None,
    state_dir: Path | None = None,
    cache_dir: Path | None = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: Callable[[], float] = random.random,
    config_dir: Path | None = None,
    processed_dir: Path | None = None,
    results_dir: Path | None = None,
    lock_path: Path | None = None,
    progress: Callable[[int, int, Mapping[str, Any]], None] | None = None,
    announce: Callable[[str], None] = lambda message: None,
    with_ci: bool = True,
    n_resamples: int = metrics.BOOTSTRAP_RESAMPLES,
) -> EvalOutcome:
    """Run one system on a split (or a fixed subset of it) and write predictions and a summary.

    Raises `config.LockError` before reading any test data if the system is not locked for the
    test subset, and `EvalError` for a bad option combination.
    """
    if split not in EVAL_SPLITS:
        raise EvalError(f"--split must be one of {EVAL_SPLITS}, got {split!r}")
    spec = config.resolve_system(system, config_dir)
    prompt_name = prompt or spec["prompt"]
    allowed = [spec["prompt"]] if split == "test" else [spec["prompt"], *spec["dev_prompts"]]
    if prompt_name not in allowed:
        raise EvalError(f"prompt {prompt_name!r} is not allowed for {system} on {split}; allowed: {allowed}")
    if limit is not None and limit < 1:
        raise EvalError("--limit must be at least 1")
    subset_name = subset or (spec["test_subset"] if split == "test" else subsets.FULL)

    processed = processed_dir or config.PROCESSED_DIR
    results = results_dir or config.RESULTS_DIR
    lock_path = lock_path or results / "test_lock.jsonl"
    inventory = schema.load_inventory(processed)
    subsets_doc = subsets.load_subsets(processed / "subsets.json")

    lock = None
    if split == "test":
        # Before any test data is read or any request is sent.
        lock = config.assert_test_allowed(
            system, subset_name, config_dir=config_dir, processed_dir=processed, lock_path=lock_path, inventory=inventory
        )

    split_examples = data.read_examples(processed / f"{split}.jsonl")
    resolved = subsets.resolve_subset(subsets_doc, subset_name, split, [e.id for e in split_examples])
    by_id = {e.id: e for e in split_examples}
    subset_examples = [by_id[i] for i in resolved.ids]
    selected = subset_examples if limit is None else subset_examples[:limit]

    components = config.config_components(system, prompt=prompt_name, inventory=inventory, config_dir=config_dir, processed_dir=processed)
    chash = config.hash_of(components)
    run_dir = results / "runs" / run_dir_name(system, split, prompt_name, spec["prompt"])
    predictions = run_dir / "predictions.jsonl"

    pspec = prompts.get_prompt(prompt_name)
    shots: Sequence[Sequence[data.Example]] = [()] * len(selected)
    if pspec.k:
        train_path = processed / "train.jsonl"
        index = RetrievalIndex.build(
            data.read_examples(train_path),
            train_sha=data.sha256_file(train_path),
            cache_dir=cache_dir or config.CACHE_DIR / "retrieval",
            embedder=embedder,
        )
        shots = index.topk_batch([e.text for e in selected], pspec.k)
    fmt = schema.response_format(inventory.intents, inventory.slot_types)
    batch_items = [
        BatchItem(e.id, prompts.render_messages(prompt_name, e.text, inventory=inventory, shots=s), fmt)
        for e, s in zip(selected, shots, strict=True)
    ]

    sources = config.load_yaml("sources", config_dir)
    price_entry = sources["prices"].get(spec["price_id"]) if spec["price_id"] else None
    if spec["price_id"] and price_entry is None:
        raise EvalError(f"price_id {spec['price_id']!r} has no entry in configs/sources.yaml")
    price = cost.Price.from_entry(price_entry) if price_entry else None
    client = ChatClient(
        Endpoint.from_system(spec), transport=transport, state_dir=state_dir, clock=clock, sleep=sleep, rng=rng,
        counter=counter, environ=environ,
    )
    prefix_tokens = 0
    if price is not None:
        prefix = [{"role": "system", "content": prompts.static_prefix(prompt_name, inventory)}]
        prefix_tokens = cost.count_message_tokens(prefix, counter)

    announce(
        f"{system} on {split}/{subset_name} with {prompt_name}: {len(selected)} of {len(subset_examples)} items "
        f"-> {spec['base_url']} model {spec['model']}"
    )

    async def go() -> BatchResult:
        async with client:
            return await run_batch(
                client, batch_items, predictions,
                concurrency=concurrency or spec["limits"].get("max_concurrency") or 4,
                resume=resume, config_hash=chash, price=price, max_usd=max_usd,
                prefix_tokens=prefix_tokens, progress=progress,
            )

    batch = asyncio.run(go())

    rows = latest_rows(read_rows(predictions))
    summary = build_summary(
        spec=spec, split=split, subset=resolved, prompt_name=prompt_name, inventory=inventory,
        examples=subset_examples, rows=rows, components=components, price_entry=price_entry,
        prefix_tokens=prefix_tokens, batch=batch, lock=lock, limit=limit, resumed=resume,
        predictions=predictions.relative_to(results.parent) if predictions.is_relative_to(results.parent) else predictions,
        with_ci=with_ci, n_resamples=n_resamples,
    )
    suffix = "" if summary["status"] == "complete" else ".partial"
    summary_path = run_dir / f"summary.{subset_name}{suffix}.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if suffix == "":
        (run_dir / f"summary.{subset_name}.partial.json").unlink(missing_ok=True)
    return EvalOutcome(batch.status, run_dir, predictions, summary_path, summary, batch)
