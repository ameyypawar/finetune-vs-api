"""Render the generated parts of the documentation from the results and the configs.

    python scripts/render.py --target readme|card|writeup|all [--check]

    readme    the region of README.md between <!-- results:start --> and <!-- results:end -->
              (templates/readme_results.md.j2). Nothing else in README.md is touched.
    card      hf/README.md, the model card for the adapter (templates/model_card.md.j2)
    writeup   docs/writeup.md, a draft of the write-up (templates/writeup.md.j2)

Every number in the output comes from a file: results/comparison.json (scripts/compare.py),
results/data_audit.json, results/subsets.json, results/train_log.json, configs/*.yaml, NOTICE.md,
or the code. The templates hold prose and no figures; tests/test_render.py fails if a template
types a number. The output has no date or time in it, so rendering twice gives the same bytes.

Every table is followed by the same notice, because the tables are only ever built by `md_table` here,
which adds it. The URLs the prices and the GPU rental price were read from, with the dates, are listed
once per document (`sources_note`), after its last table of costs.

The fine-tune's cost and latency at every measured load level (`cost_by_load` in the comparison) are one
table, `by_load_table`: in full in the README and the card, and with fewer columns in the write-up, which
has a length limit. No level is the headline: an operating point is marked only when a level met the rule,
and without one the table replaces the README's self-hosted latency table and says why. The README shows
the latency figure under that table, since the figure draws every level too. The note that says why there
is no operating point is shown once: the benchmark's own warning to the same effect is not listed again
(`shown_warnings`), though it stays in the comparison. The write-up's other compact tables drop a column
only when every row reads the same (Items, Priced as).

With no results (no results/comparison.json, or one in which no system has been scored) the README
region is empty, so README.md stays exactly as it is committed. The card and the write-up are then
drafts that say the results are pending.

Optional inputs: results/train_log.json (training details for the card), results/error_analysis.csv
(a `category` column is counted), and docs/error_analysis.md (prose included verbatim in the
write-up, so a hand-written error analysis survives re-rendering).

--check writes nothing. It exits 1 and names each file whose generated content differs from what
is on disk (CI runs it), and 0 when they are all current.

Exit status: 0 written or current; 1 --check found stale files; 2 an input is missing or unusable.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import jinja2
import yaml

from finetune_vs_api import config, cost, metrics, prompts

RESULTS_START = "<!-- results:start -->"
RESULTS_END = "<!-- results:end -->"
TARGETS = ("readme", "card", "writeup")
TEMPLATES = {"readme": "readme_results.md.j2", "card": "model_card.md.j2", "writeup": "writeup.md.j2"}
OUTPUTS = {"readme": "README.md", "card": "hf/README.md", "writeup": "docs/writeup.md"}
FIGURES = {"accuracy": "results/figures/accuracy_vs_cost.png", "latency": "results/figures/latency.png"}
COMPARISON_SCHEMA_VERSION = 1
EXIT_STALE = 1
EXIT_ERROR = 2

#: Printed under every table. The wording is the project's, not a result.
TABLE_NOTICE = "All API rows ran on free tiers; no money was spent; costs are at paid list prices."
#: The label on API latency; the same words as scripts/compare.py writes into comparison.json (a test keeps them equal).
API_LATENCY_LABEL = "observed on free tiers from India; not representative of paid tiers"
#: The reference system when there is no comparison to name one; the same as scripts/compare.py (a test keeps them equal).
DEFAULT_REFERENCE = "ft-qwen3-4b-lora"
HUB_DATASET_ID = "AmazonScience/massive"  # the Hugging Face Hub id of MASSIVE, for the card's metadata
ENDPOINT_NAMES = {"local": "a local server", "groq": "Groq", "gemini": "Google AI Studio"}
READING = {
    "system_beats_reference": "beats the fine-tune",
    "reference_beats_system": "the fine-tune beats it",
    "no_significant_difference": "no significant difference",
}
STATUS_TEXT = {"missing": "no results yet", "unreadable": "results unreadable"}
ADAPTER_PLACEHOLDER = "<adapter repo id or local path>"
#: Where the adapter is published, once it is: `adapter_repo` in configs/release.yaml, a file that does not exist
#: until the upload. It is kept out of configs/systems.yaml on purpose: the fine-tuned row is locked, and publishing
#: must not change a locked row. Its checkpoint.adapter is a path inside the training output, not a location for
#: readers.
RELEASE_FILE = "release.yaml"
MODEL_INDEX_METRICS = (
    ("exact_match", "exact_match", "Exact match"),
    ("intent_accuracy", "accuracy", "Intent accuracy"),
    ("slot_f1", "f1", "Slot F1"),
)


class RenderError(ValueError):
    """An input is missing or unusable, or a template needs something it was not given."""


# --- formatting ---------------------------------------------------------------------------------



def published_adapter(config_dir: Path) -> str | None:
    """`adapter_repo` from configs/release.yaml, or None before the adapter is published."""
    path = Path(config_dir) / RELEASE_FILE
    if not path.is_file():
        return None
    return (yaml.safe_load(path.read_text()) or {}).get("adapter_repo") or None

def _signed(value: float, digits: int = 1) -> str:
    rounded = round(value, digits) or 0.0  # never "-0.0"
    return f"{rounded:+.{digits}f}"


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def pct_interval(entry: Mapping[str, Any], digits: int = 1) -> str:
    """`91.2% [88.6, 93.4]` from a {"value", "ci95"} entry."""
    low, high = entry["ci95"]
    return f"{pct(entry['value'], digits)} [{low * 100:.{digits}f}, {high * 100:.{digits}f}]"


def pp(value: float, digits: int = 1) -> str:
    return f"{_signed(value * 100, digits)} pp"


def pp_interval(entry: Mapping[str, Any]) -> str:
    """`-3.2 pp [-6.0, -0.4]` from a {"value", "ci95"} difference."""
    low, high = entry["ci95"]
    return f"{pp(entry['value'])} [{_signed(low * 100)}, {_signed(high * 100)}]"


def half_width(entry: Mapping[str, Any]) -> str:
    """Half the width of an interval, in percentage points: `2.6 pp`."""
    low, high = entry["ci95"]
    return f"{(high - low) / 2 * 100:.1f} pp"


def usd(value: float) -> str:
    """Dollars with the digits a cost of that size needs: $383.98, $1.20, $0.600, $0.0139."""
    if value >= 1:
        return f"${value:,.2f}"
    return f"${value:.3f}" if value >= 0.1 else f"${value:.4f}"


def count(value: float) -> str:
    return f"{round(value):,}"


def bounds_cell(low: str, high: str, *, same: str = "") -> str:
    """A pair of bounds as "low to high", or the one value (followed by `same`) when both read alike."""
    return f"{low}{same}" if low == high else f"{low} to {high}"


def seconds(value: float) -> str:
    return f"{value:.2f} s"


def per_second(value: float) -> str:
    return f"{value:.2f}"


def p_value(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def yaml_str(value: Any) -> str:
    """A scalar for a YAML header. JSON strings are valid YAML, which keeps every quoting rule out of the template."""
    return json.dumps(str(value), ensure_ascii=False)


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


# --- the notice and the tables ---------------------------------------------------------------------------


def sources_note(doc: Mapping[str, Any]) -> str | None:
    """Where the prices and the GPU rental price were read, with the dates. Each document shows it once, after
    its last table of costs; the notice under every table says what the costs are."""
    sources = doc.get("sources") or {}
    parts = []
    prices = {entry["url"]: entry["retrieved_on"] for entry in (sources.get("prices") or {}).values()}
    if prices:
        parts.append("Prices: " + ", ".join(f"<{url}> (retrieved {day})" for url, day in prices.items()) + ".")
    gpu = sources.get("gpu_rental")
    if gpu:
        parts.append(f"GPU rental: <{gpu['url']}> (retrieved {gpu['retrieved_on']}).")
    return "*" + " ".join(parts) + "*" if parts else None


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], align: str) -> str:
    """A markdown table followed by the notice. `align` has one letter per column: l (left) or r (right)."""
    if len(align) != len(headers):
        raise ValueError(f"{len(headers)} columns but align={align!r}")
    rules = ["---" if a == "l" else "---:" for a in align]
    lines = ["| " + " | ".join(_cell(h) for h in headers) + " |", "|" + "|".join(rules) + "|"]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines) + "\n\n*" + TABLE_NOTICE + "*"


def _code(name: str) -> str:
    return f"`{name}`"


def accuracy_table(doc: Mapping[str, Any], *, compact: bool = False) -> str:
    """Exact match per system, paired against the reference. `compact` drops the McNemar and full-split columns,
    and the Items column when it reads the same on every row (the write-up's setup paragraph names the subsets)."""
    rows = []
    reference = doc["reference"]
    for s in doc["systems"]:
        name = _code(s["name"])
        if not s["metrics"]:
            rows.append([name, "-", STATUS_TEXT.get(s["status"], s["status"]), "-", "-", "-", "-"])
            continue
        pair = s["vs_reference"]
        items = f"{s['comparison_subset']} ({s['n_scored']}"
        if s["status"] == "partial":
            items += f" of {s['n_subset']}"
        if pair and pair["n"] != s["n_scored"]:
            items += f", {pair['n']} paired"
        items += ")"
        full = s["full_test"]
        rows.append(
            [
                name,
                items,
                pct_interval(s["metrics"]["exact_match"]),
                pp_interval(pair["difference"]) if pair else ("reference" if s["name"] == reference else "-"),
                p_value(pair["mcnemar"]["p_value"]) if pair else "-",
                READING[pair["verdict"]["relation"]] if pair else ("reference" if s["name"] == reference else "-"),
                f"{pct_interval(full['metrics']['exact_match'])} (n={full['n_scored']})" if full else "-",
            ]
        )
    headers = ["System", "Items", "Exact match", "Difference from the fine-tune", "McNemar p", "Reading", "Full test split"]
    align = "llrrrlr"
    if compact:
        keep = [0, 1, 2, 3, 5]
        if len({row[1] for row in rows}) == 1:
            keep.remove(1)
        headers, align = [headers[i] for i in keep], "".join(align[i] for i in keep)
        rows = [[row[i] for i in keep] for row in rows]
    return md_table(headers, rows, align)


def cost_table(doc: Mapping[str, Any], *, compact: bool = False) -> str | None:
    """Cost and break-even per system. `compact` (the write-up) drops "Priced as" when every row is an API row:
    the column would say "paid list price" on each line, which the notice under the table already says."""
    rows = []
    be = (doc.get("break_even") or {}).get("apis", {})
    for s in doc["systems"]:
        c = s["cost"]
        if not c:
            continue
        if s["kind"] == "api":
            bounds = c["per_1k_calls_usd"]
            volumes = be.get(s["name"], {}).get("calls_per_month", {})
            volume_cell = "-"
            if volumes.get("cached_prefix") is not None and volumes.get("no_caching") is not None:
                volume_cell = bounds_cell(count(volumes["cached_prefix"]), count(volumes["no_caching"]))
            cost_cell = bounds_cell(usd(bounds["lower"]), usd(bounds["upper"]), same=" with or without caching")
            rows.append([_code(s["name"]), "paid list price", cost_cell, volume_cell])
        else:
            rows.append([_code(s["name"]), "GPU rental at the on-demand price, kept busy", usd(c["per_1k_calls_usd"]), "-"])
    if not rows:
        return None
    headers = ["System", "Priced as", "Cost per 1,000 calls (cached prefix to no caching)", "Break-even calls per month (same order)"]
    align = "llrr"
    if compact and all(s["kind"] == "api" for s in doc["systems"] if s["cost"]):
        keep = [0, 2, 3]
        headers, align = [headers[i] for i in keep], "".join(align[i] for i in keep)
        rows = [[row[i] for i in keep] for row in rows]
    return md_table(headers, rows, align)


def latency_table(doc: Mapping[str, Any]) -> str | None:
    rows = []
    for name, entry in doc["latency"]["self_hosted"]["systems"].items():
        single, op = entry.get("concurrency_1") or {}, entry.get("operating_point") or {}
        rows.append(
            [
                _code(name),
                seconds(single["p50_s"]) if single.get("p50_s") is not None else "-",
                seconds(single["p95_s"]) if single.get("p95_s") is not None else "-",
                seconds(op["p95_s"]) if op.get("p95_s") is not None else "-",
                f"concurrency {op['concurrency']}, {op['requests_per_s']:.1f} requests/s" if op else "-",
            ]
        )
    if not rows:
        return None
    headers = ["System", "p50, concurrency 1", "p95, concurrency 1", "p95 at the operating point", "Operating point"]
    return md_table(headers, rows, "lrrrl")


def by_load_block(doc: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The reference system's `cost_by_load` (scripts/compare.py), or None when it has no benchmark levels."""
    block = (by_name(doc).get(doc["reference"]) or {}).get("cost_by_load")
    return block if block and block["levels"] else None


def _or_dash(value: float | None, fmt: Callable[[float], str]) -> str:
    return "-" if value is None else fmt(value)


def by_load_table(doc: Mapping[str, Any], *, compact: bool = False) -> str | None:
    """The reference system's cost and latency at every measured load level, none picked as the headline.

    The operating point, when a level met the rule, is marked. The full table also lists, for each level, the APIs
    whose whole break-even range one GPU can serve there (left out when no API has a break-even volume). `compact`
    (the write-up) keeps the load, p95, the cost at the on-demand price with the spot price beside it, and what one
    GPU serves a month.
    """
    block = by_load_block(doc)
    if block is None:
        return None
    apis = (doc.get("break_even") or {}).get("apis") or {}
    rows = []
    for level in block["levels"]:
        load = level["concurrency"]
        per_1k = level["per_1k_calls_usd"]
        on_demand, spot = _or_dash(per_1k.get("on_demand"), usd), _or_dash(per_1k.get("spot"), usd)
        label = f"{load} (operating point)" if load == block["operating_point"] else load
        p95, capacity = _or_dash(level["p95_s"], seconds), _or_dash(level["capacity_calls_per_month"], count)
        if compact:
            seconds_95 = _or_dash(level["p95_s"], lambda value: f"{value:.2f}")  # the unit is in the header
            rows.append([label, seconds_95, on_demand if spot == "-" else f"{on_demand} ({spot})", capacity])
            continue
        flags = {name: (entry.get("within_capacity_by_concurrency") or {}).get(str(load)) or {} for name, entry in apis.items()}
        served = [_code(name) for name, flag in flags.items() if flag and all(v is True for v in flag.values())]
        rows.append(
            [label, _or_dash(level["requests_per_s"], per_second), _or_dash(level["p50_s"], seconds), p95, on_demand, spot, capacity]
            + ([", ".join(served) or "none"] if apis else [])
        )
    if compact:
        headers = ["Concurrency", "p95 (s)", "On-demand (spot) per 1,000 calls", "Calls one GPU serves a month"]
        return md_table(headers, rows, "rrrr")
    headers = [
        "Concurrency", "Requests/s", "p50", "p95", "Cost per 1,000 calls, on-demand", "Cost per 1,000 calls, spot",
        "Calls one GPU serves a month",
    ]
    if apis:
        headers.append("APIs whose break-even range one GPU can serve")
    return md_table(headers, rows, "r" * 7 + ("l" if apis else ""))


def shown_warnings(doc: Mapping[str, Any]) -> list[str]:
    """The comparison's warnings as the README and the card list them.

    When the fine-tune's `cost_by_load` carries a note, the by-load section already says why there is no operating
    point, so the benchmark's own warning to that effect ("no operating point (...)", or "no operating point
    declared" when the file gave no reason) is left out: the note is shown once, and not the warning as well. The
    warning stays in results/comparison.json. Every other warning, including one about an operating point the
    benchmark named but that is not a measured level, is shown as before. Only the fine-tune's warning is dropped,
    because its note is the one the documents show.
    """
    warnings = list(doc["warnings"])
    block = by_load_block(doc)
    if not block or not block["note"]:
        return warnings
    path = ((doc.get("self_hosted") or {}).get("benchmark") or {}).get("path")
    prefix = f"{path}, {doc['reference']}: no operating point"
    return [w for w in warnings if not (w == f"{prefix} declared" or w.startswith(f"{prefix} ("))]


def api_latency_table(doc: Mapping[str, Any]) -> str | None:
    rows = [
        [_code(name), seconds(entry["p50"]), seconds(entry["p95"]), entry["n"]]
        for name, entry in doc["latency"]["api_appendix"]["systems"].items()
    ]
    if not rows:
        return None
    return md_table(["System", "p50", "p95", "Calls"], rows, "lrrr")


def other_metrics_table(doc: Mapping[str, Any]) -> str:
    rows = []
    for s in doc["systems"]:
        m = s["metrics"]
        if not m:
            continue
        unseen = s["unseen_text"]
        rows.append(
            [
                _code(s["name"]),
                pct(m["intent_accuracy"]["value"]),
                pct(m["slot_f1"]["value"]),
                pct(m["schema_valid_rate"]["value"]),
                pct(m["unfound_value_rate"]["value"]),
                f"{pct(unseen['value'])} (n={unseen['n']})" if unseen else "-",
            ]
        )
    headers = ["System", "Intent accuracy", "Slot F1", "Schema-valid", "Slot values not in the request", "Exact match without items whose text is in train"]
    return md_table(headers, rows, "lrrrrr")


def scenario_table(doc: Mapping[str, Any]) -> str | None:
    scored = [s for s in doc["systems"] if s["per_scenario"]]
    if not scored:
        return None
    scenarios = sorted({scenario for s in scored for scenario in s["per_scenario"]})
    rows = []
    for scenario in scenarios:
        cells = []
        for s in scored:
            cell = s["per_scenario"].get(scenario)
            cells.append(f"{pct(cell['exact_match'], 0)} ({cell['n']})" if cell else "-")
        rows.append([scenario, *cells])
    return md_table(["Scenario", *[_code(s["name"]) for s in scored]], rows, "l" + "r" * len(scored))


# --- what the prose says --------------------------------------------------------------------------------------


def by_name(doc: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {s["name"]: s for s in doc["systems"]}


def _names(names: Sequence[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def finding_summary(doc: Mapping[str, Any]) -> str:
    """One sentence on exact match: who the fine-tune beats, who beats it, who is level. Uses the wording rule."""
    groups: dict[str, list[str]] = {"reference_beats_system": [], "system_beats_reference": [], "no_significant_difference": []}
    for s in doc["systems"]:
        if s["vs_reference"]:
            groups[s["vs_reference"]["verdict"]["relation"]].append(_code(s["name"]))
    parts = []
    if groups["reference_beats_system"]:
        parts.append(f"the fine-tune beats {_names(groups['reference_beats_system'])}")
    if groups["system_beats_reference"]:
        names = groups["system_beats_reference"]
        parts.append(f"{_names(names)} {'beats' if len(names) == 1 else 'beat'} the fine-tune")
    if groups["no_significant_difference"]:
        parts.append(f"there is no significant difference with {_names(groups['no_significant_difference'])}")
    return f"On exact match, {'; '.join(parts)}." if parts else ""


def subset_notes(doc: Mapping[str, Any]) -> list[str]:
    """Where a system is compared on a smaller subset than the headline one, what the fine-tune scores there."""
    return [
        f"On the {s['vs_reference']['subset']} items, where {_code(s['name'])} is compared, the fine-tune scores "
        f"{pct_interval(s['vs_reference']['reference_exact_match'])}."
        for s in doc["systems"]
        if s["vs_reference"] and s["vs_reference"]["subset"] != doc["subsets"]["headline"]
    ]


def gap_lines(doc: Mapping[str, Any]) -> list[str]:
    lines = []
    for s in doc["systems"]:
        m = s["metrics"]
        if not m:
            continue
        lines.append(
            f"{_code(s['name'])}: intent {pct(m['intent_accuracy']['value'])}, slot F1 {pct(m['slot_f1']['value'])}, "
            f"schema-valid {pct(m['schema_valid_rate']['value'])}, values not in the request {pct(m['unfound_value_rate']['value'])}."
        )
    return lines


def weakest_scenario(system: Mapping[str, Any]) -> dict[str, Any] | None:
    cells = system.get("per_scenario") or {}
    if not cells:
        return None
    name, cell = min(cells.items(), key=lambda kv: (kv[1]["exact_match"], kv[0]))
    return {"name": name, "em": pct(cell["exact_match"]), "n": cell["n"]}


# --- loading -----------------------------------------------------------------------------------------------------


def read_json(path: Path) -> Any | None:
    """Parsed JSON, or None when the file does not exist."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RenderError(f"{path} is not valid JSON: {exc}") from exc


def load_comparison(results_dir: Path) -> dict[str, Any] | None:
    """results/comparison.json, or None when there are no results to show."""
    doc = read_json(results_dir / "comparison.json")
    if doc is None:
        return None
    if not isinstance(doc, dict) or doc.get("schema_version") != COMPARISON_SCHEMA_VERSION:
        raise RenderError(f"{results_dir / 'comparison.json'} has the wrong schema_version; run scripts/compare.py again")
    return doc if doc.get("has_results") else None


def read_citations(notice_path: Path) -> dict[str, str]:
    """The MASSIVE and SLURP BibTeX entries, from NOTICE.md, which is where they are kept."""
    try:
        text = notice_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RenderError(f"cannot read {notice_path}: {exc}") from exc
    blocks = re.findall(r"```\n(.*?)\n```", text, flags=re.DOTALL)
    found = {
        key: next((b.strip() for b in blocks if b.lstrip().startswith(start)), None)
        for key, start in (("massive", "@misc{fitzgerald2022massive"), ("slurp", "@inproceedings{slurp"))
    }
    if not all(found.values()):
        raise RenderError("NOTICE.md no longer holds the MASSIVE and SLURP BibTeX entries that the model card quotes")
    year = re.search(r"year=\{(\d{4})\}", found["massive"])
    return {**found, "massive_year": year.group(1) if year else ""}


def read_error_analysis(results_dir: Path, docs_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    summary = None
    path = results_dir / "error_analysis.csv"
    if path.exists():
        with open(path, encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        categories = None
        if rows and "category" in rows[0]:
            counts: dict[str, int] = {}
            for row in rows:
                category = row["category"] or "(blank)"
                counts[category] = counts.get(category, 0) + 1
            categories = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if categories != [("(blank)", len(rows))]:  # a sample with no label yet (scripts/sample_errors.py) says nothing
            summary = {"path": "results/error_analysis.csv", "n": len(rows), "categories": categories}
    prose = docs_dir / "error_analysis.md"
    return summary, prose.read_text(encoding="utf-8").strip() if prose.exists() else None


def training_view(train_cfg: Mapping[str, Any], log: Mapping[str, Any] | None) -> dict[str, Any]:
    """Training details: the planned configuration, replaced by what the training log recorded once it exists."""
    lora, tr = train_cfg["lora"], train_cfg["training"]
    view: dict[str, Any] = {
        "source": "config",
        "base_model": train_cfg["base_model"]["name"],
        "base_revision": train_cfg["base_model"].get("revision"),
        "lora": {"r": lora["r"], "alpha": lora["alpha"], "dropout": lora["dropout"], "target_modules": list(lora["target_modules_expanded"])},
        "learning_rate": tr["learning_rate"], "epochs": tr["num_train_epochs"],
        "batch_size": tr["per_device_train_batch_size"], "grad_accum": tr["gradient_accumulation_steps"],
        "max_seq_length": tr["max_seq_length"], "seed": tr["seed"], "scheduler": tr["lr_scheduler_type"],
        "warmup_ratio": tr["warmup_ratio"], "weight_decay": tr["weight_decay"], "optim": tr["optim"],
        "load_in_4bit": tr["load_in_4bit"], "precision": None,
        "gpu": None, "packages": None, "elapsed_minutes": None, "train_loss": None, "eval_losses": [], "data": None, "adapters": [],
    }
    if not log:
        return view
    c = log.get("config") or {}
    lg = c.get("lora") or {}
    view.update(
        source="log",
        base_model=log.get("base_model", view["base_model"]),
        base_revision=log.get("base_revision", view["base_revision"]),
        lora={"r": lg.get("r", lora["r"]), "alpha": lg.get("alpha", lora["alpha"]), "dropout": lg.get("dropout", lora["dropout"]),
              "target_modules": list(lg.get("target_modules") or view["lora"]["target_modules"])},
        learning_rate=c.get("learning_rate", view["learning_rate"]), epochs=c.get("epochs", view["epochs"]),
        batch_size=c.get("per_device_batch_size", view["batch_size"]), grad_accum=c.get("gradient_accumulation_steps", view["grad_accum"]),
        max_seq_length=c.get("max_seq_length", view["max_seq_length"]), seed=c.get("seed", view["seed"]),
        scheduler=c.get("lr_scheduler_type", view["scheduler"]), warmup_ratio=c.get("warmup_ratio", view["warmup_ratio"]),
        weight_decay=c.get("weight_decay", view["weight_decay"]), optim=c.get("optim", view["optim"]),
        load_in_4bit=c.get("load_in_4bit", view["load_in_4bit"]), precision=c.get("precision"),
        gpu=(log.get("gpu") or {}).get("name"), packages=log.get("packages"),
        elapsed_minutes=round(log["elapsed_seconds"] / 60, 1) if log.get("elapsed_seconds") else None,
        train_loss=(log.get("train_metrics") or {}).get("train_loss"),
        eval_losses=[(h["epoch"], h["eval_loss"]) for h in log.get("log_history") or [] if "eval_loss" in h],
        data={name: {"records": info.get("records"), "sha256": info.get("sha256")} for name, info in (log.get("data") or {}).items()},
        adapters=[a.get("epoch") for a in log.get("adapters") or []],
    )
    return view


def model_index(ft: Mapping[str, Any] | None, dataset: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The card's evaluation results for the adapter: the full split first, then the subset every system shares."""
    if not ft or not ft["metrics"]:
        return []
    entries = []
    scopes = []
    if ft["full_test"]:
        scopes.append((f"{dataset['name']} {dataset['version']} {dataset['locale']}, test split (n={ft['full_test']['n_scored']})", ft["full_test"]["metrics"]))
    scopes.append((f"{dataset['name']} {dataset['version']} {dataset['locale']}, test subset {ft['comparison_subset']} (n={ft['n_scored']})", ft["metrics"]))
    for label, m in scopes:
        entries.append(
            {"dataset_name": label, "metrics": [{"type": t, "name": n, "value": round(m[key]["value"], 4)} for key, t, n in MODEL_INDEX_METRICS]}
        )
    return entries


# --- the context -----------------------------------------------------------------------------------------------------


def _where(spec: Mapping[str, Any]) -> str:
    if spec["endpoint"] == "local":
        return "self-hosted"
    return f"{ENDPOINT_NAMES.get(spec['endpoint'], spec['endpoint'])}, free tier"


def _prompt_note(name: str) -> str:
    spec = prompts.get_prompt(name)
    if spec.k:
        return f"{spec.k} retrieved examples"
    return "label lists, no examples" if spec.uses_labels else "one-line instruction"


def build_context(root: Path, results_dir: Path, config_dir: Path, *, need_dataset: bool = True) -> dict[str, Any]:
    """Everything the templates may use. Nothing in it is a date or a time."""
    try:
        specs = {name: config.resolve_system(name, config_dir) for name in config.list_systems(config_dir)}
        sources = config.load_yaml("sources", config_dir)
        data_cfg = config.load_yaml("data", config_dir)
        train_cfg = config.load_yaml("train", config_dir)
    except (config.ConfigError, OSError, KeyError) as exc:
        raise RenderError(f"cannot read the configs in {config_dir}: {exc}") from exc
    comparison = load_comparison(results_dir)
    reference = comparison["reference"] if comparison else DEFAULT_REFERENCE
    if reference not in specs:
        raise RenderError(f"the reference system {reference!r} is not in {config_dir / 'systems.yaml'}")
    ft_spec = specs[reference]
    audit = read_json(results_dir / "data_audit.json")
    subsets_doc = read_json(results_dir / "subsets.json")
    if need_dataset and (audit is None or subsets_doc is None):
        missing = "results/data_audit.json" if audit is None else "results/subsets.json"
        raise RenderError(f"{missing} not found; run scripts/prepare_data.py and scripts/make_subsets.py first")
    log = read_json(results_dir / "train_log.json")
    analysis, analysis_text = read_error_analysis(results_dir, root / "docs")
    citations = read_citations(root / "NOTICE.md") if need_dataset else {}

    dataset = None
    if audit is not None and subsets_doc is not None:
        src = data_cfg["source"]
        dataset = {
            "name": src["name"], "version": src["version"], "locale": data_cfg["locale"], "license": src["license"],
            "n_train": audit["splits"]["train"], "n_dev": audit["splits"]["dev"], "n_test": audit["splits"]["test"],
            "n_intents": audit["labels"]["intents"]["n_total"], "n_slot_types": audit["labels"]["slot_types"]["n_total"],
            "n_test_text_in_train": len(audit["overlap"]["test_item_ids_in_train"]),
            "subsets": {name: {"n": entry["n"], "split": entry["split"], "parent": entry["parent"]} for name, entry in subsets_doc["subsets"].items()},
        }
    systems = [
        {
            "name": name, "model": spec["model"], "prompt": spec["prompt"], "prompt_note": _prompt_note(spec["prompt"]),
            "where": _where(spec),
        }
        for name, spec in specs.items()
    ]
    endpoint = endpoint_of(config_dir, ft_spec)
    checkpoint = ft_spec.get("checkpoint") or {}
    training = training_view(train_cfg, log)
    ctx: dict[str, Any] = {
        "has_results": comparison is not None,
        "reference": reference,
        "systems": systems,
        "dataset": dataset,
        "citations": citations,
        "hub_dataset_id": HUB_DATASET_ID,
        "base_model": {"name": train_cfg["base_model"]["name"], "license": train_cfg["base_model"]["license"]},
        "training": training,
        "ft": {
            "name": reference, "model": ft_spec["model"], "params": ft_spec["params"],
            "adapter": published_adapter(config_dir) or ADAPTER_PLACEHOLDER, "adapter_pinned": bool(published_adapter(config_dir)),
            "epoch": checkpoint.get("epoch"), "base_url": endpoint["base_url"],
            "instruction": prompts.FINETUNED_INSTRUCTION,
        },
        "deprecations": sources["openai_deprecations"],
        "table_notice": TABLE_NOTICE,
        "sources_note": None,
        "bootstrap": {"resamples": metrics.BOOTSTRAP_RESAMPLES, "confidence": 0.95},
        "hours_per_month": cost.HOURS_PER_MONTH,
        "error_analysis": analysis,
        "error_analysis_text": analysis_text,
        "figures": {key: (path if (root / path).exists() else None) for key, path in FIGURES.items()},
        "tables": {},
        "finding_summary": "",
        "subset_notes": [],
        "gap_lines": [],
        "warnings": [],
        "ft_facts": None,
        "gpu": None,
        "model_index": [],
        "pair_half_width_max": None,
        "headline_subset": None,
        "subset_sizes": {},
        "pending_names": "",
        "by_load": None,
        "api_latency": {},
        "api_latency_label": API_LATENCY_LABEL,
    }
    if comparison:
        ctx.update(comparison_context(comparison, dataset))
    return ctx


def endpoint_of(config_dir: Path, spec: Mapping[str, Any]) -> Mapping[str, Any]:
    """The endpoint a system sends its requests to, from configs/systems.yaml."""
    endpoints = config.load_yaml("systems", config_dir)["endpoints"]
    return endpoints[spec["endpoint"]]


def comparison_context(doc: Mapping[str, Any], dataset: Mapping[str, Any] | None) -> dict[str, Any]:
    """The parts of the context that exist only when there are results."""
    named = by_name(doc)
    ref = named.get(doc["reference"])
    tables: dict[str, str] = {"accuracy": accuracy_table(doc), "accuracy_compact": accuracy_table(doc, compact=True), "other": other_metrics_table(doc)}
    for key, table in (
        ("cost", cost_table(doc)), ("cost_compact", cost_table(doc, compact=True)),
        ("by_load", by_load_table(doc)), ("by_load_compact", by_load_table(doc, compact=True)),
        ("latency", latency_table(doc)), ("api_latency", api_latency_table(doc)), ("scenario", scenario_table(doc)),
    ):
        if table:
            tables[key] = table
    block = by_load_block(doc)
    by_load = block and {"note": block["note"], "operating_point": block["operating_point"]}
    # With no operating point the by-load table already holds every level's latency; the latency table would show
    # only the concurrency-1 numbers beside empty operating-point columns.
    if by_load and not any(entry.get("operating_point") for entry in doc["latency"]["self_hosted"]["systems"].values()):
        tables.pop("latency", None)
    ft_facts = None
    if ref and ref["metrics"]:
        full = ref["full_test"]
        ft_facts = {
            "subset": ref["comparison_subset"],
            "em_half_width": half_width(ref["metrics"]["exact_match"]),
            "full_n": full["n_scored"] if full else None,
            "full_em": pct_interval(full["metrics"]["exact_match"]) if full else None,
            "full_half_width": half_width(full["metrics"]["exact_match"]) if full else None,
            "unseen": ref["unseen_text"] and {"n": ref["unseen_text"]["n"], "excluded": ref["unseen_text"]["excluded"], "em": pct_interval(ref["unseen_text"])},
            "weakest": weakest_scenario(ref),
        }
    pair_widths = [(s["vs_reference"]["difference"]["ci95"][1] - s["vs_reference"]["difference"]["ci95"][0]) / 2 for s in doc["systems"] if s["vs_reference"]]
    be = doc.get("break_even") or {}
    gpu = be.get("gpu")
    return {
        "tables": tables,
        "finding_summary": finding_summary(doc),
        "subset_notes": subset_notes(doc),
        "gap_lines": gap_lines(doc),
        "warnings": shown_warnings(doc),
        "ft_facts": ft_facts,
        "pair_half_width_max": f"{max(pair_widths) * 100:.1f} pp" if pair_widths else None,
        "gpu": gpu and {
            "label": gpu["gpu"], "usd_per_hour": f"${gpu['usd_per_hour']:g}", "monthly": usd(gpu["monthly_usd"]),
            "price_basis": gpu["price_basis"].replace("_", "-"),
            "benchmark_gpu": ((doc.get("self_hosted") or {}).get("benchmark") or {}).get("gpu"),
        },
        "by_load": by_load,
        "api_latency_label": doc["latency"]["api_appendix"]["label"],
        "api_latency": doc["latency"]["api_appendix"]["systems"],
        "model_index": model_index(ref, dataset) if dataset else [],
        "headline_subset": doc["subsets"]["headline"],
        "subset_sizes": {name: info["n"] for name, info in doc["subsets"]["info"].items()},
        "bootstrap": {"resamples": doc["bootstrap"]["resamples"], "confidence": doc["bootstrap"]["confidence"]},
        "pending_names": ", ".join(_code(s["name"]) for s in doc["systems"] if not s["metrics"]),
        "sources_note": sources_note(doc),
    }


# --- templates -------------------------------------------------------------------------------------------------------


def make_env(templates_dir: Path) -> jinja2.Environment:
    # Plain markdown, not HTML, so there is nothing to escape. StrictUndefined: a template that
    # asks for something the context does not have is an error, never an empty string.
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(templates_dir)),
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        autoescape=False,
    )
    env.filters.update({"yaml": yaml_str, "pct": pct, "count": count, "usd": usd})
    return env


def render_template(env: jinja2.Environment, target: str, ctx: Mapping[str, Any]) -> str:
    try:
        return env.get_template(TEMPLATES[target]).render(**ctx)
    except jinja2.TemplateNotFound as exc:
        raise RenderError(f"template {exc.name} not found in {getattr(env.loader, 'searchpath', '?')}") from exc
    except jinja2.UndefinedError as exc:
        raise RenderError(f"{TEMPLATES[target]} needs something the context does not have: {exc.message}") from exc


def splice_readme(text: str, body: str) -> str:
    """README with the generated region replaced by `body`, set off by blank lines. An empty body leaves an empty region."""
    if text.count(RESULTS_START) != 1 or text.count(RESULTS_END) != 1:
        raise RenderError(f"README.md must contain exactly one {RESULTS_START} and one {RESULTS_END}")
    head, rest = text.split(RESULTS_START, 1)
    if RESULTS_END not in rest:
        raise RenderError(f"{RESULTS_END} comes before {RESULTS_START} in README.md")
    _, tail = rest.split(RESULTS_END, 1)
    newline = "\r\n" if "\r\n" in text else "\n"
    body = body.strip("\n").replace("\n", newline)
    region = newline + (newline + body + newline + newline if body else "")
    return head + RESULTS_START + region + RESULTS_END + tail


def generate(target: str, root: Path, ctx: Mapping[str, Any], env: jinja2.Environment) -> str:
    """The full content `target` should have."""
    if target == "readme":
        path = root / OUTPUTS["readme"]
        try:
            text = path.read_bytes().decode("utf-8")
        except OSError as exc:
            raise RenderError(f"cannot read {path}: {exc}") from exc
        return splice_readme(text, render_template(env, "readme", ctx))
    return render_template(env, target, ctx)


# --- command line ------------------------------------------------------------------------------------------------------------


def run(
    targets: Sequence[str] = TARGETS,
    *,
    check: bool = False,
    root: Path | None = None,
    results_dir: Path | None = None,
    config_dir: Path | None = None,
    templates_dir: Path | None = None,
    out: Callable[[str], None] = print,
) -> int:
    root = Path(root or config.REPO_ROOT)
    results_dir = Path(results_dir or root / "results")
    config_dir = Path(config_dir or root / "configs")
    templates_dir = Path(templates_dir or root / "templates")
    targets = list(dict.fromkeys(targets))
    try:
        ctx = build_context(root, results_dir, config_dir, need_dataset=any(t != "readme" for t in targets))
        env = make_env(templates_dir)
        planned = {t: generate(t, root, ctx, env) for t in targets}
    except (RenderError, FileNotFoundError, OSError) as exc:
        out(f"error: {exc}")
        return EXIT_ERROR
    stale, written = [], []
    for target, content in planned.items():
        path = root / OUTPUTS[target]
        current = path.read_bytes() if path.exists() else None
        encoded = content.encode("utf-8")
        if current == encoded:
            out(f"current: {OUTPUTS[target]}")
        elif check:
            stale.append(OUTPUTS[target])
            out(f"STALE: {OUTPUTS[target]} ({'missing' if current is None else 'differs from what the results and templates produce'})")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(encoded)
            written.append(OUTPUTS[target])
            out(f"wrote {OUTPUTS[target]}")
    if stale:
        out("run: python scripts/render.py --target all   (then commit the result)")
        return EXIT_STALE
    return 0


def main(argv: list[str] | None = None, **overrides: Any) -> int:
    """`overrides` go straight to `run` (another root, a quiet `out`)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--target", required=True, choices=(*TARGETS, "all"), help="what to render")
    parser.add_argument("--check", action="store_true", help="write nothing; exit 1 if a generated file is out of date")
    args = parser.parse_args(argv)
    targets = TARGETS if args.target == "all" else (args.target,)
    return run(targets, check=args.check, **overrides)


if __name__ == "__main__":
    raise SystemExit(main())
