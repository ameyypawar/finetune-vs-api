"""Compare the systems on the fixed test subsets and write results/comparison.json.

    python scripts/compare.py [--out PATH] [--reference SYSTEM] [--n-resamples N]

Inputs. A missing one is reported under "warnings" and never guessed. None of them stops the run
except the test labels, which nothing can be scored without:

    results/runs/<system>__test/predictions.jsonl       per-item answers (scripts/run_eval.py)
    results/runs/<system>__test/summary.<subset>.json   cost, latency and provenance of the run
    data/processed/{train,test}.jsonl, subsets.json     gold labels, label inventory, subsets
    results/data_audit.json                             test items whose text also occurs in train
    results/serving/<gpu>.json                          throughput benchmark of the self-hosted rows
    configs/{systems,sources}.yaml                      the systems, list prices, GPU rental price

What it computes (definitions in docs/method.md):

    pairing    every system is scored on its own test subset: S500 for every API row (S300 is
               pre-registered, but no row uses it now). A row that ran on the full test split is
               compared on S500, and also scored on the full split as a secondary column.
    accuracy   exact match with 95% bootstrap intervals; the difference against the reference
               (system minus reference) with a paired-bootstrap interval and an exact McNemar test;
               exact match on the test items whose text never occurs in train; exact match per
               scenario.
    cost       API rows: cost per 1,000 calls at paid list price, as a no-caching bound and a
               cached-prefix bound. Self-hosted rows: cost per 1,000 calls at full utilization.
               The monthly volume at which a GPU rented 24/7 (730 h, on-demand price) becomes
               cheaper than each API.
    latency    headline: the self-hosted rows measured on the box (the throughput benchmark): p50
               and p95 at concurrency 1, and at the operating point. Appendix: the API rows, as
               observed on free tiers.
    warnings   partial runs, a model name that changed during a run, runs outside the test lock,
               estimated token counts, an interval and a McNemar test that disagree, ...

Wording rule: a system "beats" another only when the 95% interval of the paired difference
excludes 0. Otherwise the comparison is "no significant difference".

The throughput benchmark is a separate script. This is the shape read from results/serving/<gpu>.json
(times in seconds). The file whose "gpu" label or file name contains the rented GPU's name (T4, for
the g4dn entry in configs/sources.yaml) is used:

    {"gpu": "Tesla T4", "system": "ft-qwen3-4b-lora",            # or "systems": {name: {...}}
     "levels": [{"concurrency": 1, "requests_per_s": 2.4, "latency_s": {"p50": 0.4, "p95": 0.6}},
                {"concurrency": 8, ...}],                          # a list, or {"1": {...}, "8": {...}}
     "operating_point": {"concurrency": 8}}                        # or just the number

Exit status: 0 written (also when results are missing); 2 a precondition failed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from finetune_vs_api import config, cost, data, evaluate, metrics, schema, subsets
from finetune_vs_api.client import is_final, latest_rows, read_rows

SCHEMA_VERSION = 1
REFERENCE = "ft-qwen3-4b-lora"
HEADLINE_SUBSET = "S500"  # where a row that ran on the full split is compared with the API rows
GPU_RENTAL_ID = "aws-g4dn.xlarge"  # the entry in configs/sources.yaml used for self-hosted cost
PRICE_BASIS = "on_demand"
CONFIDENCE = 0.95
EXIT_ERROR = 2

#: Printed with every API latency number: in the JSON, on the figures and in the generated docs.
API_LATENCY_LABEL = "observed on free tiers from India; not representative of paid tiers"
LATENCY_POLICY = (
    "Headline latency is the self-hosted rows measured on the box (p50 and p95 at concurrency 1, "
    "p95 at the operating point). API latency is an appendix."
)
EXACT_MATCH_DIFFERENCE = "system minus reference"
SECONDS_PER_MONTH = 3600 * cost.HOURS_PER_MONTH


class CompareError(ValueError):
    """A precondition failed: no test labels, subsets that no longer match, an unknown reference."""


# --- reading the runs ---------------------------------------------------------------------------


class Run:
    """What one system's test run left on disk.

    A plain class, not a dataclass: tests load the scripts without putting them in sys.modules,
    which dataclasses combined with `from __future__ import annotations` cannot cope with.
    """

    def __init__(self, name: str, spec: Mapping[str, Any], run_dir: Path):
        self.name = name
        self.spec = spec
        self.run_dir = run_dir
        self.predictions = run_dir / "predictions.jsonl"
        self.rows: dict[str, Mapping[str, Any]] = {}  # the last row per id
        self.answered: list[Mapping[str, Any]] = []  # successful rows, in file order
        self.summary: dict[str, Any] | None = None
        self.summary_path: Path | None = None
        self.summary_is_partial = False
        self.problem: str | None = None
        self.items: dict[str, metrics.Item] = {}  # scored, by example id


def rel(path: Path | None, base: Path) -> str | None:
    """`path` relative to `base` (the repository root) when it lies under it."""
    if path is None:
        return None
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return str(path)


def load_run(name: str, spec: Mapping[str, Any], results_dir: Path, warnings: list[str]) -> Run:
    """Read a system's test predictions and its summary. An unreadable file becomes a `problem`."""
    run_dir = results_dir / "runs" / evaluate.run_dir_name(name, "test", spec["prompt"], spec["prompt"])
    run = Run(name, spec, run_dir)
    if run.predictions.exists():
        try:
            rows = read_rows(run.predictions)
            run.rows = dict(latest_rows(rows))
            run.answered = [row for row in rows if row.get("error") is None]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            run.problem = f"{name}: could not read {run.predictions.name}: {exc}"
            warnings.append(run.problem)
    for suffix, partial in ((".json", False), (".partial.json", True)):
        path = run_dir / f"summary.{spec['test_subset']}{suffix}"
        if not path.exists():
            continue
        try:
            run.summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            warnings.append(f"{name}: could not read {path.name}: {exc}")
        else:
            run.summary_path, run.summary_is_partial = path, partial
        break
    return run


class Gold:
    """The test labels and the subsets every system is scored against."""

    def __init__(
        self,
        by_id: dict[str, data.Example],
        inventory: schema.LabelInventory,
        subset_ids: dict[str, tuple[str, ...]],
        subset_info: dict[str, dict[str, Any]],
    ):
        self.by_id = by_id
        self.inventory = inventory
        self.subset_ids = subset_ids
        self.subset_info = subset_info


def load_test_data(processed_dir: Path, results_dir: Path, subset_names: Sequence[str]) -> Gold:
    test_path = processed_dir / "test.jsonl"
    if not test_path.exists():
        raise CompareError(f"{test_path} not found; run scripts/prepare_data.py first (it holds the test labels)")
    examples = data.read_examples(test_path)
    inventory = schema.load_inventory(processed_dir)
    doc_path = processed_dir / "subsets.json"
    if not doc_path.exists():
        doc_path = results_dir / "subsets.json"
    try:
        doc = subsets.load_subsets(doc_path)
        resolved = {
            name: subsets.resolve_subset(doc, name, "test", [e.id for e in examples])
            for name in dict.fromkeys(subset_names)
        }
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise CompareError(f"the test subsets are unusable ({exc}); run scripts/make_subsets.py") from exc
    return Gold(
        by_id={e.id: e for e in examples},
        inventory=inventory,
        subset_ids={name: r.ids for name, r in resolved.items()},
        subset_info={name: {"n": len(r.ids), "hash": r.hash} for name, r in resolved.items()},
    )


def score_run(run: Run, test: Gold) -> None:
    """Score every item the run has a final answer for. Failed and invalid answers count as wrong."""
    ids = [i for i in test.by_id if i in run.rows and is_final(run.rows[i])]
    items = evaluate.score_rows([test.by_id[i] for i in ids], run.rows, test.inventory)
    run.items = dict(zip(ids, items, strict=True))


def load_audit(results_dir: Path, warnings: list[str]) -> tuple[set[str] | None, dict[str, Any] | None]:
    """Ids of the test items whose text also occurs in train, from results/data_audit.json."""
    path = results_dir / "data_audit.json"
    if not path.exists():
        warnings.append("results/data_audit.json not found; no exact-match figure for unseen text")
        return None, None
    try:
        overlap = json.loads(path.read_text(encoding="utf-8"))["overlap"]
        ids = {str(i) for i in overlap["test_item_ids_in_train"]}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        warnings.append(f"results/data_audit.json could not be used for the unseen-text figure: {exc}")
        return None, None
    info = {
        "path": "results/data_audit.json",
        "overlap_key": "overlap.test_item_ids_in_train",
        "normalization": overlap.get("normalization"),
        "test_items_with_text_in_train": len(ids),
    }
    return ids, info


# --- accuracy -----------------------------------------------------------------------------------


def comparison_subset(spec: Mapping[str, Any]) -> str:
    """The subset a system is compared on: its own test subset, or S500 for a full-split row."""
    return HEADLINE_SUBSET if spec["test_subset"] == subsets.FULL else spec["test_subset"]


def pick(run: Run, ids: Sequence[str]) -> tuple[list[str], list[metrics.Item]]:
    """The items of `ids` the run has an answer for, with their ids (subset order is kept)."""
    kept = [i for i in ids if i in run.items]
    return kept, [run.items[i] for i in kept]


def em_block(items: Sequence[metrics.Item], n_resamples: int) -> dict[str, Any]:
    em = metrics.item_scores(items)["em"]
    low, high = metrics.bootstrap_ci(em, n_resamples=n_resamples)
    return {"value": float(em.mean()), "ci95": [low, high], "n": len(items)}


def unseen_block(
    kept: Sequence[str], items: Sequence[metrics.Item], overlap: set[str] | None, n_resamples: int
) -> dict[str, Any] | None:
    """Exact match without the items whose text also occurs in train."""
    if overlap is None:
        return None
    unseen = [item for i, item in zip(kept, items, strict=True) if i not in overlap]
    if not unseen:
        return None
    return {**em_block(unseen, n_resamples), "excluded": len(items) - len(unseen)}


def per_scenario_block(
    kept: Sequence[str], items: Sequence[metrics.Item], by_id: Mapping[str, data.Example]
) -> dict[str, Any]:
    em = metrics.item_scores(items)["em"]
    groups: dict[str, list[float]] = {}
    for item_id, value in zip(kept, em, strict=True):
        groups.setdefault(by_id[item_id].scenario, []).append(float(value))
    return {s: {"n": len(v), "exact_match": float(np.mean(v))} for s, v in sorted(groups.items())}


def verdict(system: str, reference: str, low: float, high: float) -> dict[str, str]:
    """The wording rule. "beats" is allowed only when the interval of the difference excludes 0."""
    if low > 0:
        return {"relation": "system_beats_reference", "text": f"{system} beats {reference}"}
    if high < 0:
        return {"relation": "reference_beats_system", "text": f"{reference} beats {system}"}
    return {
        "relation": "no_significant_difference",
        "text": f"no significant difference between {system} and {reference}",
    }


def pair_block(
    run: Run, ref: Run, ids: Sequence[str], subset: str, reference: str, n_resamples: int, warnings: list[str]
) -> dict[str, Any] | None:
    """The system against the reference on the items both have an answer for."""
    paired = [i for i in ids if i in run.items and i in ref.items]
    if not paired:
        return None
    mine = metrics.item_scores([run.items[i] for i in paired])["em"]
    theirs = metrics.item_scores([ref.items[i] for i in paired])["em"]
    diff = metrics.paired_bootstrap(mine, theirs, n_resamples=n_resamples)
    mc = metrics.mcnemar_exact([bool(v) for v in mine], [bool(v) for v in theirs])
    verdict_ = verdict(run.name, reference, diff.ci_low, diff.ci_high)
    alpha = 1 - CONFIDENCE
    if (verdict_["relation"] != "no_significant_difference") != (mc.p_value < alpha):
        warnings.append(
            f"{run.name}: the paired-bootstrap interval ({diff.ci_low:+.4f} to {diff.ci_high:+.4f}) and the "
            f"exact McNemar test (p = {mc.p_value:.4f}) disagree about significance at {alpha:.2f}; "
            "the wording follows the interval"
        )
    return {
        "reference": reference,
        "subset": subset,
        "n": diff.n,
        "complete": len(paired) == len(ids),
        "system_exact_match": {"value": float(mine.mean()), "ci95": list(metrics.bootstrap_ci(mine, n_resamples=n_resamples))},
        "reference_exact_match": {"value": float(theirs.mean()), "ci95": list(metrics.bootstrap_ci(theirs, n_resamples=n_resamples))},
        "difference": {
            "value": diff.diff,
            "ci95": [diff.ci_low, diff.ci_high],
            "direction": EXACT_MATCH_DIFFERENCE,
            "bootstrap_p": diff.p_value,
        },
        "mcnemar": {"system_only": mc.a_only, "reference_only": mc.b_only, "p_value": mc.p_value},
        "verdict": verdict_,
    }


def model_names(run: Run) -> dict[str, Any]:
    """The model names the endpoint returned, in the order the answers were written.

    Segments are runs of the same name. More than one distinct name means the model behind the
    endpoint changed (or was routed differently) during the run.
    """
    segments: list[dict[str, Any]] = []
    unreported = 0
    for position, row in enumerate(run.answered, start=1):
        name = row.get("model_returned")
        if not name:
            unreported += 1
        elif segments and segments[-1]["model"] == name:
            segments[-1]["rows"] += 1
            segments[-1]["last_row"] = position
        else:
            segments.append({"model": name, "rows": 1, "first_row": position, "last_row": position})
    returned = list(dict.fromkeys(s["model"] for s in segments))
    return {
        "requested": run.spec["model"],
        "returned": returned,
        "changed": len(returned) > 1,
        "segments": segments,
        "answers_without_a_name": unreported,
    }


# --- what the summaries add: cost, latency, provenance -----------------------------------------------


def provenance(run: Run, root: Path) -> dict[str, Any] | None:
    summary = run.summary
    if summary is None:
        return None
    return {
        "summary": rel(run.summary_path, root),
        "predictions": rel(run.predictions, root),
        "summary_is_partial": run.summary_is_partial,
        "config_hash": summary.get("config_hash"),
        "git_commit": summary.get("git_commit"),
        "git_dirty": summary.get("git_dirty"),
        "created_at": summary.get("created_at"),
        "lock": summary.get("lock"),
        "subset": summary.get("subset"),
    }


def check_summary(
    run: Run, subset_hash: str | None, price_entry: Mapping[str, Any] | None, warnings: list[str]
) -> None:
    """Flag what makes a run less trustworthy than its numbers look."""
    summary = run.summary
    if summary is None:
        what = "cost, latency or provenance" if run.spec["price_id"] else "provenance (config hash, test lock)"
        warnings.append(f"{run.name}: no summary file for {run.spec['test_subset']}, so no {what}")
        return
    if run.summary_is_partial:
        warnings.append(
            f"{run.name}: only a partial summary exists ({run.summary_path.name}); its cost and latency cover "
            f"{summary.get('n_scored')} of {(summary.get('subset') or {}).get('n')} items"
        )
    if summary.get("lock") is None:
        warnings.append(f"{run.name}: the run was not made under the test lock (its summary records no lock)")
    recorded = (summary.get("subset") or {}).get("hash")
    if subset_hash and recorded and recorded != subset_hash:
        warnings.append(f"{run.name}: the run used a different {run.spec['test_subset']} than subsets.json holds now")
    tokens = summary.get("tokens") or {}
    estimated = (tokens.get("calls_with_estimated_usage") or 0) + (tokens.get("calls_without_usage") or 0)
    if estimated:
        warnings.append(f"{run.name}: {estimated} calls had no token usage from the provider; their cost is estimated or missing")
    used = (summary.get("cost") or {}).get("price")
    if used and price_entry and {k: v for k, v in used.items() if k != "id"} != dict(price_entry):
        warnings.append(f"{run.name}: its price entry in configs/sources.yaml changed after the run; the cost uses the entry the run recorded")


def api_cost(run: Run) -> dict[str, Any] | None:
    """Cost per 1,000 calls at list price, as the summary recorded it (upper: no caching)."""
    block = (run.summary or {}).get("cost") or {}
    if not block.get("per_1k_calls_usd"):
        return None
    keys = ("billing_basis", "calls", "prompt_tokens", "completion_tokens", "reasoning_tokens",
            "cacheable_prefix_tokens", "per_1k_calls_usd", "price")
    return {k: block[k] for k in keys if k in block}


def latency_observed(run: Run) -> dict[str, Any] | None:
    block = (run.summary or {}).get("latency_s")
    return dict(block) if block else None


# --- the throughput benchmark ---------------------------------------------------------------------


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def _count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    return value if isinstance(value, int) and value > 0 else None


def _first(level: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> float | None:
    for path in paths:
        node: Any = level
        for key in path:
            node = node.get(key) if isinstance(node, Mapping) else None
        number = _number(node)
        if number is not None:
            return number
    return None


# Accepted spellings. Seconds only: a level in milliseconds is reported as unusable, not rescaled.
_REQUESTS_PER_S = (("requests_per_s",), ("req_per_s",), ("rps",))
_P50_S = (("latency_s", "p50"), ("p50_s",))
_P95_S = (("latency_s", "p95"), ("p95_s",))


def _levels(raw: Any) -> dict[int, dict[str, Any]]:
    if isinstance(raw, Mapping):
        entries = [
            value if "concurrency" in value else {"concurrency": key, **value}
            for key, value in raw.items()
            if isinstance(value, Mapping)
        ]
    elif isinstance(raw, list):
        entries = [value for value in raw if isinstance(value, Mapping)]
    else:
        return {}
    levels: dict[int, dict[str, Any]] = {}
    for entry in entries:
        concurrency = _count(entry.get("concurrency"))
        if concurrency is not None:
            levels[concurrency] = {
                "concurrency": concurrency,
                "requests_per_s": _first(entry, _REQUESTS_PER_S),
                "p50_s": _first(entry, _P50_S),
                "p95_s": _first(entry, _P95_S),
                "n": _count(entry.get("n", entry.get("requests"))),
            }
    return dict(sorted(levels.items()))


def _operating_point(block: Mapping[str, Any], levels: Mapping[int, dict[str, Any]], problems: list[str]) -> dict[str, Any] | None:
    raw = block.get("operating_point", block.get("operating_concurrency"))
    if raw is None:
        note = block.get("operating_point_note")  # the benchmark says why when no level qualified
        problems.append(f"no operating point ({note})" if note else "no operating point declared")
        return None
    concurrency = _count(raw.get("concurrency")) if isinstance(raw, Mapping) else _count(raw)
    if concurrency not in levels:
        problems.append(f"the operating point ({raw!r}) is not one of the measured concurrency levels {list(levels)}")
        return None
    return levels[concurrency]


def normalize_benchmark(doc: Mapping[str, Any], default_system: str) -> dict[str, dict[str, Any]]:
    """One entry per benchmarked system: its levels, the concurrency-1 level, the operating point."""
    blocks = doc["systems"] if isinstance(doc.get("systems"), Mapping) else {str(doc.get("system") or default_system): doc}
    out: dict[str, dict[str, Any]] = {}
    for name, block in blocks.items():
        if not isinstance(block, Mapping):
            continue
        problems: list[str] = []
        levels = _levels(block.get("levels"))
        operating = _operating_point(block, levels, problems) if levels else None
        if not levels:
            problems.append("no usable levels (each needs a concurrency)")
        elif 1 not in levels:
            problems.append("no level at concurrency 1")
        for level in levels.values():
            if level["p50_s"] is None or level["p95_s"] is None or level["requests_per_s"] is None:
                problems.append(
                    f"concurrency {level['concurrency']} lacks latency_s.p50, latency_s.p95 or requests_per_s (seconds)"
                )
        out[str(name)] = {
            "levels": list(levels.values()),
            "single_stream": levels.get(1),
            "operating_point": operating,
            "problems": problems,
        }
    return out


def load_benchmark(
    results_dir: Path, gpu_name: str | None, default_system: str, root: Path, warnings: list[str]
) -> dict[str, Any] | None:
    """The benchmark file for the rented GPU, or None. Files for other GPUs are listed, not used."""
    folder = results_dir / "serving"
    files = sorted(folder.glob("*.json")) if folder.is_dir() else []
    if not files:
        return None
    token = (gpu_name or "").split()[-1].lower() if gpu_name else ""
    matcher = re.compile(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])") if token else None
    chosen: dict[str, Any] | None = None
    others: list[str] = []
    for path in files:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(doc, Mapping):
                raise ValueError("not a JSON object")
        except (OSError, ValueError) as exc:
            warnings.append(f"could not read {rel(path, root)}: {exc}")
            continue
        label = str(doc.get("gpu") or path.stem)
        if chosen is None and matcher and matcher.search(f"{label} {path.stem}".lower()):
            chosen = {"path": rel(path, root), "gpu": label, "systems": normalize_benchmark(doc, default_system)}
        else:
            others.append(rel(path, root) or str(path))
    if chosen is None:
        if others:
            warnings.append(
                f"results/serving has {', '.join(others)} but none for the rented GPU ({gpu_name}); "
                "no self-hosted cost or latency"
            )
        return None
    chosen["other_files"] = others
    for name, entry in chosen["systems"].items():
        warnings.extend(f"{chosen['path']}, {name}: {problem}" for problem in entry["problems"])
    return chosen


# --- cost model -----------------------------------------------------------------------------------


def rental_entry(sources: Mapping[str, Any], warnings: list[str]) -> dict[str, Any] | None:
    entry = (sources.get("gpu_rental") or {}).get(GPU_RENTAL_ID)
    price = ((entry or {}).get("usd_per_hour") or {}).get(PRICE_BASIS)
    if price is None:
        warnings.append(f"configs/sources.yaml has no {PRICE_BASIS} price for {GPU_RENTAL_ID}; no self-hosted cost")
        return None
    return {
        "id": GPU_RENTAL_ID,
        "gpu": entry["gpu"],
        "url": entry["url"],
        "retrieved_on": entry["retrieved_on"],
        "price_basis": PRICE_BASIS,
        "usd_per_hour": price,
        "hours_per_month": cost.HOURS_PER_MONTH,
        "monthly_usd": price * cost.HOURS_PER_MONTH,
    }


def self_hosted_cost(rental: Mapping[str, Any] | None, bench: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Cost per 1,000 calls with the GPU kept busy at the operating point, and what it can serve a month."""
    operating = (bench or {}).get("operating_point")
    rate = (operating or {}).get("requests_per_s")
    if rental is None or not rate:
        return None
    return {
        "basis": f"{rental['price_basis'].replace('_', '-')} price of {rental['id']}, kept busy at the operating point",
        "concurrency": operating["concurrency"],
        "requests_per_s": rate,
        "per_1k_calls_usd": cost.selfhost_per_1k(rental["usd_per_hour"], rate),
        "capacity_calls_per_month": rate * SECONDS_PER_MONTH,
    }


def api_break_even(per_1k: Mapping[str, float], rental: Mapping[str, Any], capacity: float | None) -> dict[str, Any]:
    """Monthly calls at which the API bill equals the rental, at each cost bound.

    Above that volume the GPU is cheaper, provided one GPU can serve it (`within_capacity`).
    `no_caching` uses the upper cost bound, so it is the smaller volume.
    """
    volumes: dict[str, float | None] = {}
    for label, bound in (("no_caching", "upper"), ("cached_prefix", "lower")):
        volume = cost.breakeven_calls_per_month(rental["usd_per_hour"], per_1k[bound] / 1000)
        volumes[label] = None if math.isinf(volume) else volume
    return {
        "calls_per_month": volumes,
        "requests_per_s": {k: None if v is None else v / SECONDS_PER_MONTH for k, v in volumes.items()},
        "within_capacity": {k: None if v is None or capacity is None else v <= capacity for k, v in volumes.items()},
    }


# --- assembling the document ------------------------------------------------------------------------


def system_record(
    run: Run,
    *,
    test: Gold | None,
    ref: Run,
    reference: str,
    overlap: set[str] | None,
    price_entry: Mapping[str, Any] | None,
    n_resamples: int,
    root: Path,
    warnings: list[str],
) -> dict[str, Any]:
    spec = run.spec
    subset = comparison_subset(spec)
    record: dict[str, Any] = {
        "name": run.name,
        "kind": "api" if spec["price_id"] else "self-hosted",
        "endpoint": spec["endpoint"],
        "model": spec["model"],
        "prompt": spec["prompt"],
        "tier": spec["tier"],
        "comparison_subset": subset,
        "status": "unreadable" if run.problem else "missing",
        "n_subset": None,
        "n_scored": 0,
        "n_calls_failed": 0,
        "metrics": None,
        "unseen_text": None,
        "per_scenario": None,
        "vs_reference": None,
        "full_test": None,
        "model_names": None,
        "cost": None,
        "latency_observed": None,
        "run": None,
    }
    if test is None or not run.items:
        return record
    ids = test.subset_ids[subset]
    kept, items = pick(run, ids)
    record["n_subset"] = len(ids)
    if not kept:
        return record
    record["status"] = "complete" if len(kept) == len(ids) else "partial"
    if record["status"] == "partial":
        warnings.append(f"{run.name}: only {len(kept)} of the {len(ids)} {subset} items have an answer")
    record["n_scored"] = len(kept)
    record["n_calls_failed"] = sum(1 for i in kept if run.rows[i].get("error"))
    record["metrics"] = metrics.summarize(items, n_resamples=n_resamples)
    record["unseen_text"] = unseen_block(kept, items, overlap, n_resamples)
    record["per_scenario"] = per_scenario_block(kept, items, test.by_id)
    if run.name != reference and ref.items:
        record["vs_reference"] = pair_block(run, ref, ids, subset, reference, n_resamples, warnings)
    if record["kind"] == "self-hosted":
        full_ids = test.subset_ids[subsets.FULL]
        full_kept, full_items = pick(run, full_ids)
        record["full_test"] = {
            "n_expected": len(full_ids),
            "n_scored": len(full_kept),
            "complete": len(full_kept) == len(full_ids),
            "metrics": metrics.summarize(full_items, n_resamples=n_resamples),
            "unseen_text": unseen_block(full_kept, full_items, overlap, n_resamples),
        }
        if len(full_kept) < len(full_ids):
            warnings.append(f"{run.name}: only {len(full_kept)} of the {len(full_ids)} full-split items have an answer")
    names = model_names(run)
    record["model_names"] = names
    if names["changed"]:
        first = names["segments"][0]
        warnings.append(
            f"{run.name}: the model name returned by the endpoint changed during the run "
            f"({' then '.join(names['returned'])}; the first name ends at answer {first['last_row']} of "
            f"{len(run.answered)}). Treat the run as a mix of models."
        )
    check_summary(run, test.subset_info[spec["test_subset"]]["hash"], price_entry, warnings)
    record["run"] = provenance(run, root)
    if record["kind"] == "api":
        record["cost"] = api_cost(run)
        record["latency_observed"] = latency_observed(run)
    return record


def build_comparison(
    *,
    results_dir: Path,
    processed_dir: Path,
    config_dir: Path | None = None,
    reference: str = REFERENCE,
    n_resamples: int = metrics.BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Everything results/comparison.json holds, as plain JSON types."""
    results_dir, processed_dir = Path(results_dir), Path(processed_dir)
    root = results_dir.parent
    warnings: list[str] = []
    specs = {name: config.resolve_system(name, config_dir) for name in config.list_systems(config_dir)}
    if reference not in specs:
        raise CompareError(f"reference system {reference!r} is not in configs/systems.yaml (known: {sorted(specs)})")
    sources = config.load_yaml("sources", config_dir)
    runs = {name: load_run(name, spec, results_dir, warnings) for name, spec in specs.items()}

    test: Gold | None = None
    overlap: set[str] | None = None
    audit: dict[str, Any] | None = None
    if any(run.rows for run in runs.values()):
        wanted = [subsets.FULL, HEADLINE_SUBSET, *(spec["test_subset"] for spec in specs.values())]
        test = load_test_data(processed_dir, results_dir, wanted)
        for run in runs.values():
            score_run(run, test)
            unknown = set(run.rows) - set(test.by_id)
            if unknown:
                warnings.append(f"{run.name}: {len(unknown)} predictions are for ids that are not in the test split and were ignored")
        overlap, audit = load_audit(results_dir, warnings)
        if not runs[reference].items:
            warnings.append(f"the reference system {reference} has no results, so nothing is paired against it")

    records = []
    for name, run in runs.items():
        price_entry = (sources.get("prices") or {}).get(run.spec["price_id"]) if run.spec["price_id"] else None
        records.append(
            system_record(
                run, test=test, ref=runs[reference], reference=reference, overlap=overlap,
                price_entry=price_entry, n_resamples=n_resamples, root=root, warnings=warnings,
            )
        )
        if run.rows and not run.items and not run.problem:
            warnings.append(f"{name}: predictions exist but none is a final answer")
    by_name = {r["name"]: r for r in records}

    rental = rental_entry(sources, warnings)
    bench = load_benchmark(results_dir, (rental or {}).get("gpu"), reference, root, warnings) if rental else None
    if bench is None and rental is not None and any(r["metrics"] for r in records) and not (results_dir / "serving").is_dir():
        warnings.append("no throughput benchmark under results/serving; no self-hosted cost, latency or capacity")
    bench_systems = (bench or {}).get("systems", {})
    self_hosted_systems: dict[str, Any] = {}
    for name, record in by_name.items():
        if record["kind"] != "self-hosted":
            continue
        entry = bench_systems.get(name)
        priced = self_hosted_cost(rental, entry)
        record["cost"] = priced
        self_hosted_systems[name] = {
            "levels": (entry or {}).get("levels"),
            "single_stream": (entry or {}).get("single_stream"),
            "operating_point": (entry or {}).get("operating_point"),
            "cost": priced,
        }
    capacity = ((by_name[reference]["cost"] or {}).get("capacity_calls_per_month")) if reference in self_hosted_systems else None
    break_even = None
    if rental is not None:
        apis = {
            name: {"per_1k_calls_usd": r["cost"]["per_1k_calls_usd"], **api_break_even(r["cost"]["per_1k_calls_usd"], rental, capacity)}
            for name, r in by_name.items()
            if r["kind"] == "api" and r["cost"]
        }
        break_even = {
            "gpu": rental,
            "self_hosted_system": reference,
            "requests_per_s": (by_name[reference]["cost"] or {}).get("requests_per_s"),
            "capacity_calls_per_month": capacity,
            "apis": apis,
        }
    latency = {
        "policy": LATENCY_POLICY,
        "self_hosted": {
            "basis": "on the box (no network path)",
            "gpu": (bench or {}).get("gpu"),
            "systems": {
                name: {"concurrency_1": s["single_stream"], "operating_point": s["operating_point"]}
                for name, s in self_hosted_systems.items()
                if s["single_stream"] or s["operating_point"]
            },
        },
        "api_appendix": {
            "label": API_LATENCY_LABEL,
            "systems": {n: r["latency_observed"] for n, r in by_name.items() if r["latency_observed"]},
        },
    }
    prices = {
        r["cost"]["price"]["id"]: {k: v for k, v in r["cost"]["price"].items() if k != "id"}
        for r in records
        if r["cost"] and r["cost"].get("price")
    }
    return _plain(
        {
            "schema_version": SCHEMA_VERSION,
            "has_results": any(r["metrics"] for r in records),
            "reference": reference,
            "billing_basis": sources.get("billing_basis", cost.BILLING_BASIS),
            "bootstrap": {
                "resamples": n_resamples,
                "seed": metrics.BOOTSTRAP_SEED,
                "confidence": CONFIDENCE,
                "interval": "percentile; the paired bootstrap resamples the same items for both systems",
            },
            "subsets": {"headline": HEADLINE_SUBSET, "info": test.subset_info if test else {}},
            "data": {"audit": audit},
            "systems": records,
            "self_hosted": {"benchmark": None if bench is None else {k: v for k, v in bench.items() if k != "systems"},
                            "systems": self_hosted_systems},
            "break_even": break_even,
            "latency": latency,
            "sources": {"prices": prices, "gpu_rental": rental},
            "warnings": warnings,
        }
    )


def _plain(obj: Any) -> Any:
    """Plain JSON types only: numpy scalars become Python numbers, non-finite floats become None."""
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


# --- command line -----------------------------------------------------------------------------------


def _percent(entry: Mapping[str, Any] | None) -> str:
    if not entry:
        return "-"
    low, high = entry["ci95"]
    return f"{entry['value'] * 100:.1f}% [{low * 100:.1f}, {high * 100:.1f}]"


def report(doc: Mapping[str, Any], path: Path, out: Callable[[str], None]) -> None:
    out(f"wrote {path}")
    if not doc["has_results"]:
        out("  no test results found for any system yet")
    for system in doc["systems"]:
        line = f"  {system['name']:24s} {system['status']:10s}"
        if system["metrics"]:
            line += f" {system['comparison_subset']} n={system['n_scored']:<4d} exact match {_percent(system['metrics']['exact_match'])}"
            pair = system["vs_reference"]
            if pair:
                low, high = pair["difference"]["ci95"]
                line += f"  vs {doc['reference']} {pair['difference']['value'] * 100:+.1f} pp [{low * 100:+.1f}, {high * 100:+.1f}]: {pair['verdict']['text']}"
        out(line)
    for message in doc["warnings"]:
        out(f"  warning: {message}")


def run(
    *,
    results_dir: Path | None = None,
    processed_dir: Path | None = None,
    config_dir: Path | None = None,
    out_path: Path | None = None,
    reference: str = REFERENCE,
    n_resamples: int = metrics.BOOTSTRAP_RESAMPLES,
    out: Callable[[str], None] = print,
) -> int:
    results_dir = Path(results_dir or config.RESULTS_DIR)
    processed_dir = Path(processed_dir or config.PROCESSED_DIR)
    try:
        doc = build_comparison(
            results_dir=results_dir, processed_dir=processed_dir, config_dir=config_dir,
            reference=reference, n_resamples=n_resamples,
        )
    except (CompareError, config.ConfigError, FileNotFoundError) as exc:
        out(f"error: {exc}")
        return EXIT_ERROR
    path = Path(out_path or results_dir / "comparison.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    report(doc, path, out)
    return 0


def main(argv: list[str] | None = None, **overrides: Any) -> int:
    """`overrides` go straight to `run` (other directories, a quiet `out`)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--out", type=Path, help="where to write the JSON (default: results/comparison.json)")
    parser.add_argument("--reference", default=REFERENCE, help=f"the system the others are compared with (default: {REFERENCE})")
    parser.add_argument("--n-resamples", type=int, default=metrics.BOOTSTRAP_RESAMPLES, help="bootstrap resamples (default: %(default)s)")
    args = parser.parse_args(argv)
    return run(out_path=args.out, reference=args.reference, n_resamples=args.n_resamples, **overrides)


if __name__ == "__main__":
    raise SystemExit(main())
