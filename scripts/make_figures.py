"""Draw the two figures from results/comparison.json (run scripts/compare.py first).

    python scripts/make_figures.py [--comparison PATH] [--out-dir DIR]

    results/figures/accuracy_vs_cost.png   exact match against cost per 1,000 calls, log cost axis
    results/figures/latency.png            self-hosted latency, measured on the box

Works headless: it draws straight to files with matplotlib's Agg backend and never opens a window,
so it runs the same on a server, in CI or over SSH.

Accuracy against cost. One point per system with a result and a cost. Colour says where the system
runs (blue: self-hosted, orange: API). Self-hosted cost is the GPU rental price with the GPU kept
busy at the operating point. An API's cost is a range at paid list price: the filled end is the
no-caching bound, the hollow end the bound with the static prompt prefix cached. The vertical bars
are 95% bootstrap intervals. Every point carries its name, so identity never rests on colour alone.

Latency. Headline only: the self-hosted rows on the box, p50 and p95 at concurrency 1 and p95 at the
operating point. API latency is not drawn; it is an appendix table, labelled as observed on free
tiers.

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
RAMP = ("#86b6ef", "#2a78d6", "#184f95")  # blue steps 250, 450, 600: p50 and p95 at concurrency 1, p95 at load

LINE_WIDTH = 1.6
LABEL_BESIDE_PT = 11  # how far, in points, a label beside an end of a range sits from its marker
MARKER_SIZE = 8.5  # points; at least 8
RING = 1.6
FONT = 8.5
FOOTNOTE_WIDTH = 118  # characters per line of the note under the legend
# Where a label may go, in the order tried: (the part of the mark it hangs from, points right, points
# up, horizontal and vertical alignment). The earliest slot with the fewest overlaps wins. Above and
# below hang from the middle of the cost range. The slots beside an end of the range clear its marker,
# and the ones that end or start 8 points from the filled end sit above or below the range without
# crossing its interval bar.
LABEL_SLOTS = (
    ("middle", 0, 9, "center", "bottom"),
    ("middle", 0, -9, "center", "top"),
    ("high", -8, 9, "right", "bottom"),
    ("high", 8, 9, "left", "bottom"),
    ("high", -8, -9, "right", "top"),
    ("high", 8, -9, "left", "top"),
    ("high", LABEL_BESIDE_PT, 0, "left", "center"),
    ("low", -LABEL_BESIDE_PT, 0, "right", "center"),
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


def accuracy_points(doc: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """The systems that can be placed (an exact-match result and a positive cost), and those left out."""
    points: list[dict[str, Any]] = []
    left_out: list[str] = []
    headline = doc["subsets"]["headline"]
    for system in doc["systems"]:
        em = (system.get("metrics") or {}).get("exact_match")
        per_1k = (system.get("cost") or {}).get("per_1k_calls_usd")
        if system["kind"] == "api" and per_1k:
            low, high = per_1k["lower"], per_1k["upper"]
        elif system["kind"] == "self-hosted" and per_1k is not None:
            low = high = per_1k
        else:
            low = high = None
        if em is None or low is None or low <= 0:  # a log axis cannot show a cost of 0
            left_out.append(system["name"])
            continue
        subset = system["comparison_subset"]
        points.append(
            {
                "name": system["name"] if subset == headline else f"{system['name']} ({subset})",
                "kind": system["kind"],
                "y": em["value"], "y_low": em["ci95"][0], "y_high": em["ci95"][1],
                "x_low": low, "x_high": high,
            }
        )
    return points, left_out


def latency_groups(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One group per self-hosted system with benchmark latency: (label, seconds) bars, in order."""
    groups = []
    for name, entry in doc["latency"]["self_hosted"]["systems"].items():
        single, operating = entry.get("concurrency_1") or {}, entry.get("operating_point") or {}
        bars = [
            ("p50, concurrency 1", single.get("p50_s")),
            ("p95, concurrency 1", single.get("p95_s")),
            ("p95 at the operating point", operating.get("p95_s")),
        ]
        if any(value is not None for _, value in bars):
            groups.append({"name": name, "bars": bars, "operating_concurrency": operating.get("concurrency")})
    return groups


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


def _place_labels(fig: Figure, ax, marks: Sequence[dict[str, Any]], obstacles: Sequence[Bbox]) -> None:
    """Direct labels. Each takes the first slot that overlaps no mark, no earlier label, and stays in the plot.

    When no slot is free the one that overlaps least is used. `marks` hold the label text and the three
    anchor points (data coordinates); `obstacles` are boxes in pixels.
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

    def cost(box: Bbox) -> float:
        hits = sum(1 for other in taken if box.overlaps(other))
        outside = not (inside.x0 <= box.x0 and box.x1 <= inside.x1 and inside.y0 <= box.y0 and box.y1 <= inside.y1)
        return hits + (1 if outside else 0)

    for mark in marks:
        scored = [(cost(measure(mark, slot)), n, slot) for n, slot in enumerate(LABEL_SLOTS)]
        _, _, (part, dx, dy, ha, va) = min(scored)  # fewest overlaps; the earliest slot wins a tie
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
    points, left_out = accuracy_points(doc)
    if not points:
        return None
    fig, ax = _new_figure((7.6, 5.25), bottom=0.317)
    _style(ax)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_usd))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    lows = [p["x_low"] for p in points]
    highs = [p["x_high"] for p in points]
    # room on the right for the widest name beside the filled end of its range
    widest = max(_text_width(fig, p["name"]) for p in points)
    ax.set_xlim(min(lows) / 2.2, _upper_cost_limit(fig, ax, lows, highs, widest))
    y_min = min(p["y_low"] for p in points)
    y_max = max(p["y_high"] for p in points)
    pad = max(0.02, (y_max - y_min) * 0.12)
    ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))

    for p in points:
        colour = SELF_HOSTED if p["kind"] == "self-hosted" else API
        ax.vlines(p["x_high"], p["y_low"], p["y_high"], color=colour, linewidth=LINE_WIDTH, zorder=3)
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
    obstacles, marks = [], []
    reach = MARKER_SIZE * DPI / 72 / 2 + 3  # a marker's radius in pixels, plus its ring
    for p in points:
        x_low, y_mid = ax.transData.transform((p["x_low"], p["y"]))
        x_high, _ = ax.transData.transform((p["x_high"], p["y"]))
        _, y_low = ax.transData.transform((p["x_high"], p["y_low"]))
        _, y_high = ax.transData.transform((p["x_high"], p["y_high"]))
        obstacles.append(Bbox([[x_low - reach, y_mid - reach], [x_high + reach, y_mid + reach]]))  # range and markers
        obstacles.append(Bbox([[x_high - 3, y_low], [x_high + 3, y_high]]))  # the interval bar
        marks.append(
            {
                "text": p["name"],
                "low": (p["x_low"], p["y"]),
                "high": (p["x_high"], p["y"]),
                "middle": ((p["x_low"] * p["x_high"]) ** 0.5, p["y"]),
            }
        )
    _place_labels(fig, ax, marks, obstacles)

    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=MARKER_SIZE, markerfacecolor=SELF_HOSTED,
               markeredgecolor=SURFACE, markeredgewidth=RING, label="self-hosted: GPU kept busy, on-demand price"),
        Line2D([], [], color=API, linewidth=LINE_WIDTH, marker="o", markersize=MARKER_SIZE, markerfacecolor=API,
               markeredgecolor=SURFACE, markeredgewidth=RING,
               label="API: paid list price, from cached prefix (hollow end) to no caching (filled end)"),
    ]
    note = f"Error bars: 95% bootstrap interval. API cost basis: {doc['billing_basis']} (no money was spent)."
    if left_out:
        note += f" Not shown, no result or no cost yet: {', '.join(left_out)}."
    _footer(fig, handles, note, legend_top=0.214)
    return fig


def plot_latency(doc: Mapping[str, Any]) -> Figure | None:
    """p50 and p95 at concurrency 1 and p95 at the operating point, per self-hosted system, or None."""
    groups = latency_groups(doc)
    if not groups:
        return None
    fig, ax = _new_figure((7.0, 5.2), bottom=0.34)
    _style(ax)
    ax.grid(False, axis="x")
    n_bars = 3
    width = 0.2
    top = 0.0
    for g, group in enumerate(groups):
        for b, (_, value) in enumerate(group["bars"]):
            if value is None:
                continue
            x = g + (b - (n_bars - 1) / 2) * (width + 0.03)
            ax.bar(x, value, width=width, color=RAMP[b], zorder=3, linewidth=0)
            ax.text(x, value, f"{value:.2f}", ha="center", va="bottom", fontsize=FONT - 0.5, color=INK_SECONDARY)
            top = max(top, value)
    ax.set_ylim(0, top * 1.18 if top else 1.0)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(
        [
            g["name"] + (f"\noperating point: concurrency {g['operating_concurrency']}" if g["operating_concurrency"] else "")
            for g in groups
        ]
    )
    ax.set_ylabel("Seconds per request", color=INK_SECONDARY, fontsize=FONT + 0.5)
    ax.set_title("Self-hosted latency, on the box", loc="left", color=INK, fontsize=11.5, fontweight="bold")
    labels = [label for label, _ in groups[0]["bars"]]
    handles = [
        Line2D([], [], marker="s", linestyle="", markersize=MARKER_SIZE, markerfacecolor=RAMP[i],
               markeredgecolor=RAMP[i], label=label)
        for i, label in enumerate(labels)
    ]
    gpu = doc["latency"]["self_hosted"].get("gpu")
    note = (f"GPU: {gpu}. " if gpu else "") + (
        f"API latency is not drawn here: it is an appendix table, {doc['latency']['api_appendix']['label']}."
    )
    _footer(fig, handles, note, legend_top=0.205, ncol=3)
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
