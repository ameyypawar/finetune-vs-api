"""Draw the two figures from results/comparison.json (run scripts/compare.py first).

    python scripts/make_figures.py [--comparison PATH] [--out-dir DIR]

    results/figures/accuracy_vs_cost.png   exact match against cost per 1,000 calls, log cost axis
    results/figures/latency.png            self-hosted latency at every measured load, on the box

Works headless: it draws straight to files with matplotlib's Agg backend and never opens a window,
so it runs the same on a server, in CI or over SSH.

Accuracy against cost. One row per system with a result and a cost. Colour says where the system
runs (blue: self-hosted, orange: API). A self-hosted row is one exact-match result drawn at every
load level the throughput benchmark measured (`cost_by_load` in the comparison): a marker at that
level's cost, the GPU rental price at the on-demand rate with the GPU kept busy, joined by a thin
line and labelled with its concurrency ("x8" is eight concurrent requests). No level is the headline.
When a level met the operating-point rule it is labelled as the operating point and carries the
interval bar, as the single point did before; otherwise the bar is drawn once, at the cheapest level,
which is clear of the API rows. An API's cost is a range at paid list price: the filled end is the
no-caching bound, the hollow end the bound with the static prompt prefix cached. The vertical bars
are 95% bootstrap intervals. Every row carries its name, so identity never rests on colour alone. A name
sits beside its own row: a place where another row's mark is at least as near the name as its own marks
are is passed over for one beside its own marks, even one that overlaps a little (`AMBIGUOUS_COST`).

Latency. Headline only: the fine-tune on the box, p50 and p95 at every measured concurrency level,
with the operating point marked when a level met the rule and nothing marked when none did. API
latency is not drawn; it is an appendix table, labelled as observed on free tiers.

Exit status: 0 done (a figure with nothing to draw is skipped and said so); 2 no usable
comparison.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")  # files only, no display: must be chosen before pyplot is imported

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter, PercentFormatter
from matplotlib.transforms import Bbox

from finetune_vs_api import config

SCHEMA_VERSION = 1
ACCURACY_FIGURE = "accuracy_vs_cost.png"
LATENCY_FIGURE = "latency.png"
EXIT_ERROR = 2
DPI = 200

# The reference palette's light-mode values (validated with its checks: the two categorical slots
# pass all-pairs, the three-step blue ramp passes the ordinal checks).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SELF_HOSTED = "#2a78d6"  # categorical slot 1
API = "#eb6834"  # categorical slot 2
RAMP = ("#86b6ef", "#2a78d6", "#184f95")  # blue steps 250, 450, 600: p50 and p95 are the first two

LINE_WIDTH = 1.6
CHAIN_WIDTH = 1.0  # the thin line that joins the load levels of one self-hosted row
QUANTILES = ("p50", "p95")  # the latency bars at each load, in order; their colours are the first RAMP steps
ON_DEMAND = "on_demand"  # the GPU price a load level is drawn at (scripts/compare.py: PRICE_BASIS)
LABEL_BESIDE_PT = 11  # how far, in points, a label beside an end of a range sits from its marker
MARKER_SIZE = 8.5  # points; at least 8
RING = 1.6
FONT = 8.5
FOOTNOTE_WIDTH = 118  # characters per line of the note under the legend
# Where a label may go, in the order tried: (the part of the mark it hangs from, points right, points
# up, horizontal and vertical alignment). The earliest slot with the fewest overlaps wins. Above and
# below hang from the middle of the cost range. The slots beside an end of the range clear its marker,
# and the ones that end or start 8 points from the filled end sit above or below the range without
# crossing its interval bar. The last two are fallbacks for a crowded plot, used only when every slot
# before them overlaps something: above the top end of the row's own interval bar and below its bottom
# end, each ending 6 points left of the bar, which takes the label past the bars and markers of the
# rows around it and never across its own bar.
LABEL_SLOTS = (
    ("middle", 0, 9, "center", "bottom"),
    ("middle", 0, -9, "center", "top"),
    ("high", -8, 9, "right", "bottom"),
    ("high", 8, 9, "left", "bottom"),
    ("high", -8, -9, "right", "top"),
    ("high", 8, -9, "left", "top"),
    ("high", LABEL_BESIDE_PT, 0, "left", "center"),
    ("low", -LABEL_BESIDE_PT, 0, "right", "center"),
    ("top", -6, 3, "right", "bottom"),
    ("bottom", -6, -3, "right", "top"),
)
# The slots of a load-level label ("x8"), which hangs from its own marker: below it first, so that the row's
# name keeps the room above the line, then beside it on the right (clear only past the last marker, where the
# line ends, and better there than above it, where the label would stand straight after the row's name and
# read as part of it), then above, beside it on the left, and the four corners. Every anchor part of such a
# mark is the marker itself, so "middle" is used throughout.
LEVEL_SLOTS = (
    ("middle", 0, -9, "center", "top"),
    ("middle", LABEL_BESIDE_PT, 0, "left", "center"),
    ("middle", 0, 9, "center", "bottom"),
    ("middle", -LABEL_BESIDE_PT, 0, "right", "center"),
    ("middle", 8, -9, "left", "top"),
    ("middle", -8, -9, "right", "top"),
    ("middle", 8, 9, "left", "bottom"),
    ("middle", -8, 9, "right", "bottom"),
)


class FigureError(ValueError):
    """The comparison file cannot be used."""


# --- reading the comparison -----------------------------------------------------------------------


def load_comparison(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FigureError(f"{path} not found; run scripts/compare.py first")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FigureError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema_version") != SCHEMA_VERSION:
        raise FigureError(f"{path} has schema_version {doc.get('schema_version') if isinstance(doc, dict) else None!r}; this script reads {SCHEMA_VERSION}")
    return doc


def priced_levels(system: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[int]]:
    """The measured load levels of a self-hosted system that can be drawn, and the concurrencies that cannot.

    Read from the system's `cost_by_load` (scripts/compare.py): every level with an on-demand cost, in order of
    concurrency, with its label ("x8", or "x8 (operating point)" for the level the rule chose, when one did), and
    the concurrencies with no cost to draw: no throughput was measured there, or the cost is 0, which a log axis
    cannot show.
    """
    block = system.get("cost_by_load") or {}
    operating = block.get("operating_point")
    placed: list[dict[str, Any]] = []
    unpriced: list[int] = []
    for level in block.get("levels") or []:
        concurrency = level["concurrency"]
        usd = (level.get("per_1k_calls_usd") or {}).get(ON_DEMAND)
        if usd is None or usd <= 0:
            unpriced.append(concurrency)
            continue
        is_operating = operating is not None and concurrency == operating
        text = f"x{concurrency}" + (" (operating point)" if is_operating else "")
        placed.append({"concurrency": concurrency, "x": usd, "operating": is_operating, "text": text})
    return placed, unpriced


def accuracy_points(doc: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[str]], list[str]]:
    """The systems that can be placed, those left out (by why), and the load levels left out of a system that is placed.

    A system can be placed with an exact-match result and a positive cost. An API's cost is a range. A
    self-hosted system is one point for each of its measured load levels (`levels`, empty for an API), with
    `x_low` and `x_high` the cheapest and the dearest of them. `bar_x` is where its interval bar stands: at
    the operating point when a level met the rule, else at the cheapest level, which is clear of the API rows
    (an API's own bar stands at its filled end). A comparison from before the load levels existed has only the
    single operating point, which is drawn as one point.
    """
    points: list[dict[str, Any]] = []
    left_out: dict[str, list[str]] = {}
    unpriced: list[str] = []
    headline = doc["subsets"]["headline"]
    for system in doc["systems"]:
        em = (system.get("metrics") or {}).get("exact_match")
        per_1k = (system.get("cost") or {}).get("per_1k_calls_usd")
        levels: list[dict[str, Any]] = []
        missing: list[int] = []
        if system["kind"] == "api" and per_1k:
            low, high = per_1k["lower"], per_1k["upper"]
        elif system["kind"] == "self-hosted":
            levels, missing = priced_levels(system)
            if levels:
                low, high = min(level["x"] for level in levels), max(level["x"] for level in levels)
            else:
                low = high = per_1k
        else:
            low = high = None
        if em is None or low is None or low <= 0:  # a log axis cannot show a cost of 0
            why = "no result yet" if em is None else "no cost measured" if low is None else "a cost of 0 on a log axis"
            left_out.setdefault(why, []).append(system["name"])
            continue
        if missing:
            unpriced.append(f"{system['name']} at concurrency {', '.join(str(c) for c in missing)}")
        subset = system["comparison_subset"]
        points.append(
            {
                "name": system["name"] if subset == headline else f"{system['name']} ({subset})",
                "kind": system["kind"],
                "y": em["value"], "y_low": em["ci95"][0], "y_high": em["ci95"][1],
                "x_low": low, "x_high": high,
                "bar_x": next((level["x"] for level in levels if level["operating"]), low if levels else high),
                "levels": levels,
            }
        )
    return points, left_out, unpriced


def latency_levels(doc: Mapping[str, Any]) -> dict[str, Any] | None:
    """The fine-tune's p50 and p95 at every measured load level, or None when no level has a latency.

    Read from the reference system's `cost_by_load` (scripts/compare.py), which holds every level the throughput
    benchmark measured, in order of concurrency. `operating` is the concurrency of the level the rule chose, or
    None when no level met it. A level that lacks one of the two keeps its place and the missing bar is not drawn.
    """
    system = next((s for s in doc["systems"] if s["name"] == doc["reference"]), None)
    block = (system or {}).get("cost_by_load") or {}
    levels = [
        {"concurrency": level["concurrency"], "p50": level["p50_s"], "p95": level["p95_s"]}
        for level in block.get("levels") or []
        if level["p50_s"] is not None or level["p95_s"] is not None
    ]
    if not levels:
        return None
    return {"name": doc["reference"], "levels": levels, "operating": block.get("operating_point")}


# --- drawing ------------------------------------------------------------------------------------------


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")  # solid hairlines, never dashed
    ax.set_axisbelow(True)
    ax.tick_params(colors=BASELINE, labelcolor=INK_SECONDARY, labelsize=FONT, length=3)


def _new_figure(size: tuple[float, float], bottom: float) -> tuple[Figure, Any]:
    """A figure with a fixed layout: the plot above `bottom`, the legend and the notes in the space under it."""
    fig = plt.figure(figsize=size, dpi=DPI, facecolor=SURFACE)
    fig.subplots_adjust(left=0.115, right=0.975, top=0.92, bottom=bottom)
    return fig, fig.add_subplot()


def _footer(fig: Figure, handles: Sequence[Line2D], note: str, *, legend_top: float, ncol: int = 1) -> None:
    """The legend, then a muted note, in the reserved space at the bottom of the figure."""
    legend = fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.012, legend_top), ncol=ncol, frameon=False,
        fontsize=FONT, labelcolor=INK_SECONDARY, handlelength=2.6, handletextpad=0.5, columnspacing=1.6,
    )
    fig.canvas.draw()
    bottom_of_legend = legend.get_window_extent(fig.canvas.get_renderer()).y0 / fig.bbox.height
    fig.text(0.012, bottom_of_legend - 0.025, textwrap.fill(note, width=FOOTNOTE_WIDTH, break_on_hyphens=False), ha="left", va="top",
             fontsize=FONT - 0.5, color=MUTED, linespacing=1.35)


def _usd(value: float, _position: int | None = None) -> str:
    return f"${value:g}"


#: A reader attaches a label to the nearest mark, so a slot is ambiguous when a mark of another row is at least as near the
#: label as the label's own marks are. What it costs, counted in overlaps: more than one, so that a slot beside the label's own
#: mark with a small overlap beats a clean slot that reads as another row's label, and less than two.
AMBIGUOUS_COST = 1.5


def _gap(a: Bbox, b: Bbox) -> float:
    """The distance in pixels between two boxes: 0 when they touch or overlap."""
    return math.hypot(max(a.x0 - b.x1, b.x0 - a.x1, 0.0), max(a.y0 - b.y1, b.y0 - a.y1, 0.0))


def _place_labels(fig: Figure, ax, marks: Sequence[dict[str, Any]], obstacles: Sequence[Bbox]) -> None:
    """Direct labels. Each takes the first slot that overlaps no mark, no earlier label, and stays in the plot.

    A slot where another row's mark is at least as near the label as the label's own marks costs AMBIGUOUS_COST on top
    of its overlaps. When no slot is free the one that costs least is used. `marks` hold the label text and the anchor
    points its slots hang from (data coordinates): `low`, `middle` and `high` of the row's cost range, and the `top` and
    `bottom` ends of its interval bar. A mark may name its own `slots` in place of LABEL_SLOTS, and then needs only the
    parts they use. A mark may also list its row's own mark boxes as `own`: they are among the `obstacles`, which are
    boxes in pixels, and every other obstacle belongs to another row. A mark is placed after the ones before it, so the
    order says which labels choose first.
    """
    renderer = fig.canvas.get_renderer()
    taken = list(obstacles)
    inside = ax.bbox

    def measure(mark: dict[str, Any], slot: tuple) -> Bbox:
        part, dx, dy, ha, va = slot
        note = ax.annotate(mark["text"], mark[part], xytext=(dx, dy), textcoords="offset points",
                           ha=ha, va=va, fontsize=FONT, annotation_clip=False)
        box = note.get_window_extent(renderer)
        note.remove()
        return box

    def ambiguous(box: Bbox, mark: dict[str, Any]) -> bool:
        """Whether a mark of another row is at least as near the label as the label's own marks are."""
        own = mark.get("own") or ()
        if not own:
            return False
        mine = min(_gap(box, b) for b in own)
        theirs = min((_gap(box, b) for b in obstacles if not any(b is o for o in own)), default=math.inf)
        return theirs <= mine

    def cost(box: Bbox, mark: dict[str, Any]) -> float:
        hits = sum(1 for other in taken if box.overlaps(other))
        outside = not (inside.x0 <= box.x0 and box.x1 <= inside.x1 and inside.y0 <= box.y0 and box.y1 <= inside.y1)
        return hits + (1 if outside else 0) + (AMBIGUOUS_COST if ambiguous(box, mark) else 0)

    for mark in marks:
        scored = [(cost(measure(mark, slot), mark), n, slot) for n, slot in enumerate(mark.get("slots", LABEL_SLOTS))]
        _, _, (part, dx, dy, ha, va) = min(scored)  # the cheapest; the earliest slot wins a tie
        note = ax.annotate(mark["text"], mark[part], xytext=(dx, dy), textcoords="offset points",
                           ha=ha, va=va, fontsize=FONT, color=INK, annotation_clip=False)
        taken.append(note.get_window_extent(renderer))


def _text_width(fig: Figure, text: str) -> float:
    """The width in pixels of `text` at the label size."""
    note = fig.text(0, 0, text, fontsize=FONT)
    width = note.get_window_extent(fig.canvas.get_renderer()).width
    note.remove()
    return width


def _upper_cost_limit(fig: Figure, ax, lows: Sequence[float], highs: Sequence[float], widest_label: float) -> float:
    """The upper limit of the cost axis, leaving room right of the highest filled end for the widest label.

    The axis is logarithmic, so with the lower limit fixed the room is exact arithmetic: the highest end
    sits at the fraction log(high / low_limit) / log(high_limit / low_limit) of the axes width, and
    the rest of the width must hold the label and its offset from the marker.
    """
    width = ax.get_position().width * fig.bbox.width
    needed = widest_label + LABEL_BESIDE_PT * DPI / 72 + 8
    lower = min(lows) / 2.2
    if needed >= width / 2:  # a name too long to place beside a mark at all: just leave generous room
        return max(highs) * 6
    return lower * 10 ** (math.log10(max(highs) / lower) / (1 - needed / width))


def plot_accuracy_vs_cost(doc: Mapping[str, Any]) -> Figure | None:
    """Exact match (y) against cost per 1,000 calls (x, log scale), or None when nothing can be placed."""
    points, left_out, unpriced = accuracy_points(doc)
    if not points:
        return None
    # 0.25 in taller than the plot needs, so the note under the legend can run to five lines (every optional sentence
    # and several rows left out) without leaving the figure; the plot is the size it was
    fig, ax = _new_figure((7.6, 5.5), bottom=0.348)
    _style(ax)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_usd))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    lows = [p["x_low"] for p in points]
    highs = [p["x_high"] for p in points]
    # room on the right for the widest label beside the filled end of its range
    widest = max(_text_width(fig, text) for p in points for text in (p["name"], *(level["text"] for level in p["levels"])))
    ax.set_xlim(min(lows) / 2.2, _upper_cost_limit(fig, ax, lows, highs, widest))
    y_min = min(p["y_low"] for p in points)
    y_max = max(p["y_high"] for p in points)
    pad = max(0.02, (y_max - y_min) * 0.12)
    ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))

    for p in points:
        colour = SELF_HOSTED if p["kind"] == "self-hosted" else API
        ax.vlines(p["bar_x"], p["y_low"], p["y_high"], color=colour, linewidth=LINE_WIDTH, zorder=3)
        if p["levels"]:  # a self-hosted row: a thin line through one marker per measured load, all of them filled
            if p["x_low"] < p["x_high"]:
                ax.hlines(p["y"], p["x_low"], p["x_high"], color=colour, linewidth=CHAIN_WIDTH, zorder=3)
            for level in p["levels"]:
                ax.plot(level["x"], p["y"], "o", markersize=MARKER_SIZE, markerfacecolor=colour,
                        markeredgecolor=SURFACE, markeredgewidth=RING, zorder=5)
            continue
        if p["x_low"] < p["x_high"]:
            ax.hlines(p["y"], p["x_low"], p["x_high"], color=colour, linewidth=LINE_WIDTH, zorder=3)
            ax.plot(p["x_low"], p["y"], "o", markersize=MARKER_SIZE, markerfacecolor=SURFACE,
                    markeredgecolor=colour, markeredgewidth=LINE_WIDTH, zorder=4)
        ax.plot(p["x_high"], p["y"], "o", markersize=MARKER_SIZE, markerfacecolor=colour,
                markeredgecolor=SURFACE, markeredgewidth=RING, zorder=5)

    ax.set_xlabel("Cost per 1,000 calls (USD, log scale)", color=INK_SECONDARY, fontsize=FONT + 0.5)
    ax.set_ylabel("Exact match", color=INK_SECONDARY, fontsize=FONT + 0.5)
    ax.set_title("Exact match against cost per 1,000 calls", loc="left", color=INK, fontsize=11.5, fontweight="bold")

    fig.canvas.draw()  # fixes the layout, so data coordinates map to final pixels
    obstacles, names, loads = [], [], []
    reach = MARKER_SIZE * DPI / 72 / 2 + 3  # a marker's radius in pixels, plus its ring
    for p in points:
        x_low, y_mid = ax.transData.transform((p["x_low"], p["y"]))
        x_high, _ = ax.transData.transform((p["x_high"], p["y"]))
        x_bar, y_low = ax.transData.transform((p["bar_x"], p["y_low"]))
        _, y_high = ax.transData.transform((p["bar_x"], p["y_high"]))
        range_box = Bbox([[x_low - reach, y_mid - reach], [x_high + reach, y_mid + reach]])  # range and markers
        bar_box = Bbox([[x_bar - 3, y_low], [x_bar + 3, y_high]])  # the interval bar
        obstacles += [range_box, bar_box]
        own = (range_box, bar_box)  # what a label of this row must stay nearer to than to any other row's marks
        names.append(
            {
                "text": p["name"],
                "low": (p["x_low"], p["y"]),
                "high": (p["x_high"], p["y"]),
                "middle": ((p["x_low"] * p["x_high"]) ** 0.5, p["y"]),
                "top": (p["bar_x"], p["y_high"]),
                "bottom": (p["bar_x"], p["y_low"]),
                "own": own,
            }
        )
        for level in p["levels"]:  # the load a marker stands for, hung from the marker itself
            spot = (level["x"], p["y"])
            loads.append({"text": level["text"], "low": spot, "high": spot, "middle": spot, "slots": LEVEL_SLOTS, "own": own})
    _place_labels(fig, ax, names + loads, obstacles)  # every name chooses its place before any load label does

    chain = bool(loads)
    handles = [
        Line2D([], [], color=SELF_HOSTED, linestyle="-" if chain else "", linewidth=CHAIN_WIDTH, marker="o",
               markersize=MARKER_SIZE, markerfacecolor=SELF_HOSTED, markeredgecolor=SURFACE, markeredgewidth=RING,
               label="self-hosted: GPU kept busy, on-demand price" + ("; xN is N concurrent requests" if chain else "")),
        Line2D([], [], color=API, linewidth=LINE_WIDTH, marker="o", markersize=MARKER_SIZE, markerfacecolor=API,
               markeredgecolor=SURFACE, markeredgewidth=RING,
               label="API: paid list price, from cached prefix (hollow end) to no caching (filled end)"),
    ]
    note = f"Error bars: 95% bootstrap interval. API cost basis: {doc['billing_basis']} (no money was spent)."
    if chain:
        note += " The self-hosted row is one exact-match result shown at each measured load, so it has one bar."
    for why, names in left_out.items():
        note += f" Not shown, {why}: {', '.join(names)}."
    if unpriced:
        note += f" Load levels not shown, no cost: {'; '.join(unpriced)}."
    _footer(fig, handles, note, legend_top=0.25)
    return fig


def plot_latency(doc: Mapping[str, Any]) -> Figure | None:
    """The fine-tune's p50 and p95 at every measured load level, or None. The operating point is marked when a
    level met the rule; when none did, nothing is marked."""
    data = latency_levels(doc)
    if data is None:
        return None
    levels = data["levels"]
    fig, ax = _new_figure((7.0, 5.2), bottom=0.34)
    _style(ax)
    ax.grid(False, axis="x")
    width = 0.2
    top = 0.0
    for i, level in enumerate(levels):
        for b, quantile in enumerate(QUANTILES):
            value = level[quantile]
            if value is None:
                continue
            x = i + (b - (len(QUANTILES) - 1) / 2) * (width + 0.03)
            ax.bar(x, value, width=width, color=RAMP[b], zorder=3, linewidth=0)
            ax.text(x, value, f"{value:.2f}", ha="center", va="bottom", fontsize=FONT - 0.5, color=INK_SECONDARY)
            top = max(top, value)
    limit = top * 1.18 if top else 1.0
    ax.set_ylim(0, limit)
    ax.set_xlim(-0.6, len(levels) - 0.4)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([str(level["concurrency"]) for level in levels])
    operating = next((i for i, level in enumerate(levels) if level["concurrency"] == data["operating"]), None)
    if operating is not None:  # the level the rule chose, when one did: a light column behind its bars, and its name
        ax.axvspan(operating - 0.5, operating + 0.5, color=SELF_HOSTED, alpha=0.1, linewidth=0, zorder=1)
        ax.annotate("operating point", (operating, limit), xytext=(0, -4), textcoords="offset points",
                    ha="center", va="top", fontsize=FONT - 0.5, color=INK_SECONDARY, annotation_clip=False)
    ax.set_xlabel("Concurrent requests", color=INK_SECONDARY, fontsize=FONT + 0.5)
    ax.set_ylabel("Seconds per request", color=INK_SECONDARY, fontsize=FONT + 0.5)
    ax.set_title("Self-hosted latency by load, on the box", loc="left", color=INK, fontsize=11.5, fontweight="bold")
    handles = [
        Line2D([], [], marker="s", linestyle="", markersize=MARKER_SIZE, markerfacecolor=RAMP[i],
               markeredgecolor=RAMP[i], label=quantile)
        for i, quantile in enumerate(QUANTILES)
    ]
    gpu = doc["latency"]["self_hosted"].get("gpu")
    note = f"System: {data['name']}. " + (f"GPU: {gpu}. " if gpu else "") + (
        f"API latency is not drawn here: it is an appendix table, {doc['latency']['api_appendix']['label']}."
    )
    _footer(fig, handles, note, legend_top=0.205, ncol=len(QUANTILES))
    return fig


def save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # No "Software" text chunk, so the PNG does not change with the matplotlib version string.
    fig.savefig(path, dpi=DPI, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)


# --- command line ------------------------------------------------------------------------------------------


def run(
    *,
    comparison_path: Path | None = None,
    out_dir: Path | None = None,
    out: Callable[[str], None] = print,
) -> int:
    comparison_path = Path(comparison_path or config.RESULTS_DIR / "comparison.json")
    out_dir = Path(out_dir or comparison_path.parent / "figures")
    try:
        doc = load_comparison(comparison_path)
    except FigureError as exc:
        out(f"error: {exc}")
        return EXIT_ERROR
    for filename, draw, why in (
        (ACCURACY_FIGURE, plot_accuracy_vs_cost, "no system has both an exact-match result and a cost yet"),
        (LATENCY_FIGURE, plot_latency, "no self-hosted latency yet (the throughput benchmark has not been read)"),
    ):
        fig = draw(doc)
        if fig is None:
            out(f"skipped {filename}: {why}")
            continue
        save(fig, out_dir / filename)
        out(f"wrote {out_dir / filename}")
    return 0


def main(argv: list[str] | None = None, **overrides: Any) -> int:
    """`overrides` go straight to `run` (another directory, a quiet `out`)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--comparison", type=Path, help="the comparison file (default: results/comparison.json)")
    parser.add_argument("--out-dir", type=Path, help="where to write the PNGs (default: results/figures)")
    args = parser.parse_args(argv)
    return run(comparison_path=args.comparison, out_dir=args.out_dir, **overrides)


if __name__ == "__main__":
    raise SystemExit(main())
