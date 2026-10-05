"""scripts/make_figures.py: the files are written, the numbers are where the comparison says they are,
the cost axis is logarithmic, the self-hosted row is drawn at every measured load, and it all works without a
display. Synthetic results, and the real throughput benchmark (results/serving/T4.json) in a scratch directory."""

from __future__ import annotations

import copy
import json
import math
import os
import re
import subprocess
import sys

import pytest
import yaml
from matplotlib.colors import to_rgba
from PIL import Image

from conftest import ROOT, SCRIPTS, load_script
from finetune_vs_api import config
from results_fixtures import (
    API_SYSTEMS,
    BASE,
    FT,
    GEMINI,
    GPT_OSS_20B,
    GPT_OSS_120B,
    NO_OPERATING_POINT_SERVING,
    QWEN_27B,
    Lab,
)

compare = load_script("compare")
figures = load_script("make_figures")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
REAL_T4 = ROOT / "results" / "serving" / "T4.json"


@pytest.fixture(autouse=True)
def close_figures():
    """The tests draw many figures without saving them; matplotlib keeps each open until it is told otherwise."""
    yield
    figures.plt.close("all")


def build_world(tmp_path_factory, name, serving=None):
    """The standard results compared once, with `serving` as the throughput benchmark when one is given; the
    comparison is written where make_figures looks for it."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "git_commit", lambda cwd=None: None)
        mp.setattr(config, "git_dirty", lambda cwd=None: False)
        lab = Lab(tmp_path_factory.mktemp(name)).populate()
        if serving is not None:
            lab.write_serving(doc=serving)
        doc = compare.build_comparison(
            results_dir=lab.results, processed_dir=lab.processed, config_dir=lab.config_dir, n_resamples=100
        )
    path = lab.results / "comparison.json"
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return lab, doc, path


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The standard results: the fine-tune measured at concurrency 1 and 8, and 8 is its operating point."""
    return build_world(tmp_path_factory, "figures")


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    """The standard results with a benchmark shaped like the real T4 run: four loads (1, 8, 32, 64), no operating point."""
    return build_world(tmp_path_factory, "figures-sweep", NO_OPERATING_POINT_SERVING)


@pytest.fixture(params=["with an operating point", "without one"])
def either(request, world, sweep):
    """The two states of the self-hosted row: a load met the operating-point rule, or none did (the real T4)."""
    return world if request.param == "with an operating point" else sweep


def texts(fig) -> list[str]:
    """Every piece of text in a figure: its own, each axes' labels and annotations, and the legend."""
    found = [t.get_text() for t in fig.texts]
    for ax in fig.axes:
        found += [ax.get_title(loc="left"), ax.get_xlabel(), ax.get_ylabel()]
        found += [t.get_text() for t in ax.texts]
        found += [t.get_text() for t in ax.get_xticklabels()]
    for legend in fig.legends:
        found += [t.get_text() for t in legend.get_texts()]
    return [t for t in found if t]


def ft_of(doc):
    return next(s for s in doc["systems"] if s["name"] == FT)


def drawn_markers(fig):
    """(x, y, face colour) of every marker on the cost axes, rounded so that floats compare."""
    return {(round(line.get_xdata()[0], 12), round(line.get_ydata()[0], 12), line.get_markerfacecolor()) for line in fig.axes[0].lines}


def segments(fig):
    """Every line drawn with hlines or vlines on the first axes: ((x0, y0), (x1, y1), line width), rounded."""
    found = []
    for collection in fig.axes[0].collections:
        width = float(collection.get_linewidths()[0])
        for (x0, y0), (x1, y1) in collection.get_segments():
            found.append(((round(x0, 12), round(y0, 12)), (round(x1, 12), round(y1, 12)), width))
    return found


def direct_labels(fig):
    """The direct labels on the first axes (names and loads), in the order they were placed."""
    return [t.get_text() for t in fig.axes[0].texts]


# --- files --------------------------------------------------------------------------------------------------


def test_both_figures_are_written_as_png_files(either, tmp_path):
    _, _, path = either
    lines: list[str] = []
    assert figures.run(comparison_path=path, out_dir=tmp_path / "figures", out=lines.append) == 0
    for name in ("accuracy_vs_cost.png", "latency.png"):
        data = (tmp_path / "figures" / name).read_bytes()
        assert data.startswith(PNG_SIGNATURE) and len(data) > 10_000
        with Image.open(tmp_path / "figures" / name) as image:
            width, height = image.size
        assert width > 1000 and height > 800  # drawn at a size that reads well in a README
        assert any(name in line for line in lines)


def test_the_default_location_is_results_figures_next_to_the_comparison(world):
    lab, _, path = world
    assert figures.run(comparison_path=path, out=lambda line: None) == 0
    assert (lab.results / "figures" / "accuracy_vs_cost.png").exists() and (lab.results / "figures" / "latency.png").exists()


def test_the_figures_are_byte_for_byte_reproducible(either, tmp_path):
    _, _, path = either
    figures.run(comparison_path=path, out_dir=tmp_path / "a", out=lambda line: None)
    figures.run(comparison_path=path, out_dir=tmp_path / "b", out=lambda line: None)
    for name in ("accuracy_vs_cost.png", "latency.png"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


# --- accuracy against cost ---------------------------------------------------------------------------------------


def test_the_cost_axis_is_logarithmic(world):
    _, doc, _ = world
    fig = figures.plot_accuracy_vs_cost(doc)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log" and ax.get_yscale() == "linear"
    assert ax.get_xlabel() == "Cost per 1,000 calls (USD, log scale)"
    low, high = ax.get_xlim()
    visible = [label.get_text() for tick, label in zip(ax.get_xticks(), ax.get_xticklabels(), strict=True) if low <= tick <= high]
    assert visible == ["$0.01", "$0.1", "$1", "$10"]  # one tick per decade of the cost range, in dollars


def test_every_point_is_where_the_comparison_puts_it(world):
    _, doc, _ = world
    fig = figures.plot_accuracy_vs_cost(doc)
    drawn = {(round(line.get_xdata()[0], 12), round(line.get_ydata()[0], 12), line.get_markerfacecolor()) for line in fig.axes[0].lines}
    by_name = {s["name"]: s for s in doc["systems"]}
    ft = by_name[FT]
    em = round(ft["metrics"]["exact_match"]["value"], 12)
    assert (round(ft["cost"]["per_1k_calls_usd"], 12), em, figures.SELF_HOSTED) in drawn  # the operating point, as before
    levels = ft["cost_by_load"]["levels"]
    assert len(levels) == 2  # the standard benchmark has concurrency 1 and 8
    for level in levels:  # and a marker at each measured load, at its on-demand cost
        assert (round(level["per_1k_calls_usd"]["on_demand"], 12), em, figures.SELF_HOSTED) in drawn
    for name in (GPT_OSS_20B, GPT_OSS_120B, GEMINI):
        s = by_name[name]
        y = round(s["metrics"]["exact_match"]["value"], 12)
        bounds = s["cost"]["per_1k_calls_usd"]
        assert (round(bounds["upper"], 12), y, figures.API) in drawn  # filled: no caching
        assert (round(bounds["lower"], 12), y, figures.SURFACE) in drawn  # hollow: cached prefix
    qwen = by_name[QWEN_27B]  # no cached-input price: one point, drawn filled
    assert (round(qwen["cost"]["per_1k_calls_usd"]["upper"], 12), round(qwen["metrics"]["exact_match"]["value"], 12), figures.API) in drawn
    assert len(drawn) == len(levels) + 3 * 2 + 1  # the operating point is one of the loads, so it adds no marker of its own


def test_a_price_without_a_cached_rate_is_one_filled_point_and_no_hollow_end(world):
    """groq-qwen3.8-27b has no cached-input price, so its two cost bounds are equal."""
    _, doc, _ = world
    fig = figures.plot_accuracy_vs_cost(doc)
    qwen = next(s for s in doc["systems"] if s["name"] == QWEN_27B)
    bounds = qwen["cost"]["per_1k_calls_usd"]
    assert bounds["lower"] == bounds["upper"]
    at_cost = [line for line in fig.axes[0].lines if round(line.get_xdata()[0], 12) == round(bounds["upper"], 12)]
    assert [line.get_markerfacecolor() for line in at_cost] == [figures.API]  # filled only: nothing hollow underneath
    assert QWEN_27B in texts(fig)


def test_the_interval_bars_are_the_bootstrap_intervals(world):
    _, doc, _ = world
    fig = figures.plot_accuracy_vs_cost(doc)
    spans = set()
    for collection in fig.axes[0].collections:
        for segment in collection.get_segments():
            (x0, y0), (x1, y1) = segment
            if x0 == x1:
                spans.add((round(y0, 12), round(y1, 12)))
    for s in doc["systems"]:
        if s["name"] != BASE:  # the base model has no cost yet, so it is not drawn
            low, high = s["metrics"]["exact_match"]["ci95"]
            assert (round(low, 12), round(high, 12)) in spans


def test_a_system_without_a_cost_is_left_out_and_named(world):
    _, doc, _ = world
    fig = figures.plot_accuracy_vs_cost(doc)
    labels = texts(fig)
    assert BASE not in labels  # no point and no label of its own
    note = next(t for t in labels if t.startswith("Error bars"))
    assert "Not shown, no result or no cost yet: base-qwen3-4b-k10." in note.replace("\n", " ")
    assert "free tier; priced at paid list price (no money was spent)" in note.replace("\n", " ")


def test_every_drawn_system_is_named_on_the_chart_not_only_by_colour(world):
    _, doc, _ = world
    labels = texts(figures.plot_accuracy_vs_cost(doc))
    for name in (FT, *API_SYSTEMS):
        assert name in labels


def test_a_system_on_a_smaller_subset_carries_it_in_its_label(world):
    """No real row uses S300 now; the figure still names the subset of a row compared on a smaller one."""
    _, doc, _ = world
    smaller = copy.deepcopy(doc)
    next(s for s in smaller["systems"] if s["name"] == GEMINI)["comparison_subset"] = "S300"
    labels = texts(figures.plot_accuracy_vs_cost(smaller))
    assert f"{GEMINI} (S300)" in labels and GEMINI not in labels
    assert {FT, GPT_OSS_20B, GPT_OSS_120B, QWEN_27B} <= set(labels)


def test_the_legend_explains_what_the_colours_the_ends_and_the_load_labels_mean(either):
    _, doc, _ = either
    fig = figures.plot_accuracy_vs_cost(doc)
    entries = [t.get_text() for t in fig.legends[0].get_texts()]
    assert entries == [
        "self-hosted: GPU kept busy, on-demand price; xN is N concurrent requests",
        "API: paid list price, from cached prefix (hollow end) to no caching (filled end)",
    ]


# --- the self-hosted row at every measured load -------------------------------------------------------------------------


def operating_label(load: int, operating) -> str:
    return f"x{load}" + (" (operating point)" if load == operating else "")


def test_every_measured_load_is_a_filled_marker_at_its_on_demand_cost_joined_by_a_thin_line(either):
    _, doc, _ = either
    ft = ft_of(doc)
    levels = ft["cost_by_load"]["levels"]
    assert len(levels) >= 2  # not vacuous: a row of one marker would have nothing to join
    fig = figures.plot_accuracy_vs_cost(doc)
    ax = fig.axes[0]
    em = round(ft["metrics"]["exact_match"]["value"], 12)
    costs = sorted(round(level["per_1k_calls_usd"]["on_demand"], 12) for level in levels)
    markers = [line for line in ax.lines if line.get_markerfacecolor() == figures.SELF_HOSTED]
    assert sorted(round(line.get_xdata()[0], 12) for line in markers) == costs  # exactly one marker for each load
    assert {round(line.get_ydata()[0], 12) for line in markers} == {em}  # all at the one exact-match result
    assert all(line.get_markersize() >= 8 for line in markers)
    chain = [(a, b, width) for a, b, width in segments(fig) if a[1] == b[1] == em and a[0] != b[0] and a[0] == costs[0]]
    assert chain == [((costs[0], em), (costs[-1], em), figures.CHAIN_WIDTH)]  # one horizontal line, cheapest load to dearest
    assert figures.CHAIN_WIDTH < figures.LINE_WIDTH  # thinner than an API's range


def test_each_load_is_labelled_with_its_concurrency_beside_its_own_marker(either):
    _, doc, _ = either
    ft = ft_of(doc)
    operating = ft["cost_by_load"]["operating_point"]
    fig = figures.plot_accuracy_vs_cost(doc)
    renderer = _render(fig)
    ax = fig.axes[0]
    levels = ft["cost_by_load"]["levels"]
    wanted = {operating_label(level["concurrency"], operating): level for level in levels}
    boxes = {t.get_text(): t.get_window_extent(renderer) for t in ax.texts}
    assert set(wanted) <= set(boxes)
    y = ax.transData.transform((1.0, ft["metrics"]["exact_match"]["value"]))[1]
    centres = {
        label: ax.transData.transform((level["per_1k_calls_usd"]["on_demand"], ft["metrics"]["exact_match"]["value"]))[0]
        for label, level in wanted.items()
    }

    def distance(box, x):  # from the marker to the nearest point of the label, in pixels
        return ((max(box.x0 - x, 0, x - box.x1)) ** 2 + (max(box.y0 - y, 0, y - box.y1)) ** 2) ** 0.5

    for label in wanted:  # no other load's marker is nearer to this label than its own
        assert distance(boxes[label], centres[label]) == min(distance(boxes[label], x) for x in centres.values()), label


def test_the_row_name_sits_above_its_line_and_the_loads_below_it(sweep):
    """The name takes the room above the chain and the load labels hang under their markers, so a load is never mistaken
    for part of a name. (x64, whose marker carries the interval bar, goes beside it; x1 and the others go below.)"""
    _, doc, _ = sweep
    ft = ft_of(doc)
    fig = figures.plot_accuracy_vs_cost(doc)
    renderer = _render(fig)
    ax = fig.axes[0]
    boxes = {t.get_text(): t.get_window_extent(renderer) for t in ax.texts}
    y = ax.transData.transform((1.0, ft["metrics"]["exact_match"]["value"]))[1]
    assert boxes[FT].y0 > y  # the name is above the line
    for load in ("x1", "x8", "x32"):
        assert boxes[load].y1 < y, load  # and these labels are below it


def test_without_an_operating_point_nothing_is_called_one_and_the_bar_stands_at_the_cheapest_load(sweep):
    _, doc, _ = sweep
    ft = ft_of(doc)
    assert ft["cost_by_load"]["operating_point"] is None and ft["cost"] is None  # the real T4's state
    fig = figures.plot_accuracy_vs_cost(doc)
    labels = direct_labels(fig)
    assert [t for t in labels if "operating point" in t] == []
    assert {"x1", "x8", "x32", "x64"} <= set(labels)
    low, high = (round(v, 12) for v in ft["metrics"]["exact_match"]["ci95"])
    cheapest = round(min(level["per_1k_calls_usd"]["on_demand"] for level in ft["cost_by_load"]["levels"]), 12)
    bars = [(a, b) for a, b, _ in segments(fig) if a[0] == b[0] and (a[1], b[1]) == (low, high)]
    assert bars == [((cheapest, low), (cheapest, high))]  # one bar for the row, at its cheapest load: clear of the API rows
    assert "so it has one bar" in next(t for t in texts(fig) if t.startswith("Error bars")).replace("\n", " ")


def test_with_an_operating_point_its_single_point_and_its_interval_bar_are_kept(world):
    _, doc, _ = world
    ft = ft_of(doc)
    assert ft["cost_by_load"]["operating_point"] == 8 == ft["cost"]["concurrency"]
    em = round(ft["metrics"]["exact_match"]["value"], 12)
    low, high = (round(v, 12) for v in ft["metrics"]["exact_match"]["ci95"])
    at_the_point = round(ft["cost"]["per_1k_calls_usd"], 12)
    fig = figures.plot_accuracy_vs_cost(doc)
    assert (at_the_point, em, figures.SELF_HOSTED) in drawn_markers(fig)  # the point the figure always drew
    bars = [(a, b) for a, b, _ in segments(fig) if a[0] == b[0] and (a[1], b[1]) == (low, high)]
    assert bars == [((at_the_point, low), (at_the_point, high))]  # and its bar stays with it, not at the dearest load
    labels = direct_labels(fig)
    assert "x8 (operating point)" in labels and "x1" in labels  # only the level the rule chose says so
    assert len([t for t in labels if "operating point" in t]) == 1


def test_a_comparison_from_before_the_load_levels_still_draws_its_single_operating_point(world):
    """Older comparison.json files have the operating point's cost and no `cost_by_load`: one point, as it always was."""
    _, doc, _ = world
    old = copy.deepcopy(doc)
    for system in old["systems"]:
        system.pop("cost_by_load")
    ft = ft_of(old)
    fig = figures.plot_accuracy_vs_cost(old)
    markers = [line for line in fig.axes[0].lines if line.get_markerfacecolor() == figures.SELF_HOSTED]
    assert [round(line.get_xdata()[0], 12) for line in markers] == [round(ft["cost"]["per_1k_calls_usd"], 12)]
    assert not [t for t in direct_labels(fig) if t.startswith("x")]  # no load labels, and no line to join one marker
    assert [t.get_text() for t in fig.legends[0].get_texts()][0] == "self-hosted: GPU kept busy, on-demand price"
    assert figures.plot_latency(old) is None  # the latency figure needs the levels


def test_a_single_measured_load_is_one_labelled_marker_with_no_line_to_join_it(sweep):
    _, doc, _ = sweep
    one = copy.deepcopy(doc)
    ft = ft_of(one)
    ft["cost_by_load"]["levels"] = [level for level in ft["cost_by_load"]["levels"] if level["concurrency"] == 8]
    fig = figures.plot_accuracy_vs_cost(one)
    markers = [line for line in fig.axes[0].lines if line.get_markerfacecolor() == figures.SELF_HOSTED]
    assert len(markers) == 1 and "x8" in direct_labels(fig) and "x1" not in direct_labels(fig)
    em = round(ft["metrics"]["exact_match"]["value"], 12)
    assert [s for s in segments(fig) if s[0][1] == s[1][1] == em and s[0][0] != s[1][0]] == []  # one marker: nothing to join
    assert len(figures.plot_latency(one).axes[0].containers) == 2  # and one pair of bars


def test_a_load_level_without_a_cost_is_left_out_and_named(sweep):
    _, doc, _ = sweep
    gap = copy.deepcopy(doc)
    level = next(level for level in ft_of(gap)["cost_by_load"]["levels"] if level["concurrency"] == 8)
    level.update(requests_per_s=None, per_1k_calls_usd={"on_demand": None, "spot": None}, capacity_calls_per_month=None)
    fig = figures.plot_accuracy_vs_cost(gap)
    labels = direct_labels(fig)
    assert "x8" not in labels and {"x1", "x32", "x64"} <= set(labels)
    assert len([line for line in fig.axes[0].lines if line.get_markerfacecolor() == figures.SELF_HOSTED]) == 3
    note = next(t for t in texts(fig) if t.startswith("Error bars")).replace("\n", " ")
    assert f"Load levels not shown, no cost: {FT} at concurrency 8." in note
    # a cost of 0 cannot be shown on a log axis either, so it is left out the same way
    level.update(per_1k_calls_usd={"on_demand": 0.0, "spot": 0.0})
    fig = figures.plot_accuracy_vs_cost(gap)
    assert "x8" not in direct_labels(fig) and f"{FT} at concurrency 8" in " ".join(texts(fig)).replace("\n", " ")
    # with no cost at any load the whole row has nothing to draw and is named like any other system left out
    for level in ft_of(gap)["cost_by_load"]["levels"]:
        level["per_1k_calls_usd"] = {"on_demand": None, "spot": None}
    note = next(t for t in texts(figures.plot_accuracy_vs_cost(gap)) if t.startswith("Error bars")).replace("\n", " ")
    assert f"Not shown, no result or no cost yet: {FT}, {BASE}." in note


def test_grid_lines_are_solid_hairlines_never_dashed(world):
    _, doc, _ = world
    for fig in (figures.plot_accuracy_vs_cost(doc), figures.plot_latency(doc)):
        for line in fig.axes[0].get_ygridlines():
            assert line.get_linestyle() == "-" and line.get_linewidth() <= 1.0


def _render(fig):
    fig.canvas.draw()
    return fig.canvas.get_renderer()


def test_points_that_nearly_coincide_still_get_labels_that_do_not_overlap(world):
    _, doc, _ = world
    crowded = copy.deepcopy(doc)
    by_name = {s["name"]: s for s in crowded["systems"]}
    small, large = by_name[GPT_OSS_20B], by_name[GPT_OSS_120B]
    large["metrics"]["exact_match"] = {"value": small["metrics"]["exact_match"]["value"] + 0.002, "ci95": list(small["metrics"]["exact_match"]["ci95"])}
    large["cost"]["per_1k_calls_usd"] = {k: v * 1.03 for k, v in small["cost"]["per_1k_calls_usd"].items()}
    fig = figures.plot_accuracy_vs_cost(crowded)
    renderer = _render(fig)
    boxes = {t.get_text(): t.get_window_extent(renderer) for t in fig.axes[0].texts}
    assert {GPT_OSS_20B, GPT_OSS_120B} <= set(boxes)
    names = list(boxes)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            assert not boxes[first].overlaps(boxes[second]), (first, second)


#: The fine-tune's cost on the real T4 sweep (results/serving/T4.json, at the on-demand rate): USD per 1,000 calls at
#: each concurrency, from the cheapest load (64) to the dearest (1).
REAL_SWEEP = {1: 0.18488, 8: 0.027131, 32: 0.010148, 64: 0.0067601}


def priced_at_the_real_sweep(doc, operating=None):
    """A copy of `doc` in which the fine-tune is measured at the four loads of the real T4 sweep and priced as it was
    there. `operating` makes one of the loads the operating point (the real T4 has none)."""
    doc = copy.deepcopy(doc)
    ft = ft_of(doc)
    ft["cost_by_load"] = {
        **ft["cost_by_load"],
        "operating_point": operating,
        "note": None if operating else compare.NO_OPERATING_POINT_NOTE,
        "levels": [
            {"concurrency": load, "requests_per_s": 1.0, "p50_s": 1.0, "p95_s": 2.0, "capacity_calls_per_month": 1.0,
             "per_1k_calls_usd": {"on_demand": usd, "spot": usd / 2}}
            for load, usd in REAL_SWEEP.items()
        ],
    }
    ft["cost"] = None if operating is None else {**(ft["cost"] or {}), "concurrency": operating, "per_1k_calls_usd": REAL_SWEEP[operating]}
    return doc


def set_results(doc, values):
    """Give each named system the exact match with its interval and, for an API, the cost range in `values`."""
    by_name = {s["name"]: s for s in doc["systems"]}
    for name, (em, interval, price) in values.items():
        by_name[name]["metrics"]["exact_match"] = {"value": em, "ci95": list(interval), "n": 500}
        if price is not None:
            by_name[name]["cost"]["per_1k_calls_usd"] = price


def mark_boxes(fig, pad=2):
    """Every drawn mark as a box in pixels, found from the artists and not from the placement code."""
    ax = fig.axes[0]
    marks = []
    for line in ax.lines:
        (x, y), r = ax.transData.transform((line.get_xdata()[0], line.get_ydata()[0])), line.get_markersize() * fig.dpi / 72 / 2
        marks.append((x - r, y - r, x + r, y + r))
    for collection in ax.collections:
        for (x0, y0), (x1, y1) in (ax.transData.transform(segment) for segment in collection.get_segments()):
            marks.append((min(x0, x1) - pad, min(y0, y1) - pad, max(x0, x1) + pad, max(y0, y1) + pad))
    return marks


def assert_labels_clear(fig, wanted):
    """Each label in `wanted` is inside the plot, on no mark and on no other label of the figure."""
    renderer = _render(fig)
    ax = fig.axes[0]
    labels = {t.get_text(): t.get_window_extent(renderer) for t in ax.texts}
    marks = mark_boxes(fig)
    inside = ax.bbox
    for name in wanted:
        box = labels[name]
        assert inside.x0 <= box.x0 and box.x1 <= inside.x1 and inside.y0 <= box.y0 and box.y1 <= inside.y1, f"{name} leaves the plot"
        for x0, y0, x1, y1 in marks:
            assert box.x1 < x0 or box.x0 > x1 or box.y1 < y0 or box.y0 > y1, f"{name} overlaps a mark"
        for other, other_box in labels.items():
            assert other == name or not box.overlaps(other_box), (name, other)


#: Exact match with its interval, and the cost (a range for an API; the fine-tune's costs are its loads), as the real runs
#: were expected to come out: a 10-shot prompt of 1,000 to 1,500 tokens and 30 to 200 answer tokens a call, at the list
#: prices in configs/sources.yaml.
ESTIMATED_RESULTS = {
    FT: (0.908, (0.882, 0.932), None),
    GPT_OSS_20B: (0.848, (0.816, 0.878), {"lower": 0.094, "upper": 0.128}),
    GPT_OSS_120B: (0.844, (0.812, 0.876), {"lower": 0.188, "upper": 0.255}),
    QWEN_27B: (0.878, (0.846, 0.907), {"lower": 0.891, "upper": 0.891}),  # no cached-input price: one point
    GEMINI: (0.882, (0.850, 0.910), {"lower": 1.27, "upper": 1.88}),
}


@pytest.mark.parametrize("operating", [None, 8], ids=["no operating point", "operating point at 8"])
@pytest.mark.parametrize("gemini_subset", ["S500", "S300"], ids=["every name plain", "one name carries its subset"])
def test_a_realistic_crowd_of_api_points_gets_labels_clear_of_every_mark(world, gemini_subset, operating):
    """Costs and accuracies like the real runs will have: a 10-shot prompt of 1,000 to 1,500 tokens and 30 to
    200 answer tokens a call, at the list prices in configs/sources.yaml. That puts four API marks in the middle
    of the plot (a range where the price has a cached rate, one point where it has none), and a long name on
    the right (the longest when it carries its subset). The fine-tune is the real T4 sweep: four loads from
    about $0.007 to $0.185, so its row runs left into the API marks."""
    _, doc, _ = world
    crowd = priced_at_the_real_sweep(doc, operating)
    set_results(crowd, ESTIMATED_RESULTS)
    next(s for s in crowd["systems"] if s["name"] == GEMINI)["comparison_subset"] = gemini_subset
    gemini_label = GEMINI if gemini_subset == "S500" else f"{GEMINI} (S300)"
    fig = figures.plot_accuracy_vs_cost(crowd)
    loads = {operating_label(load, operating) for load in REAL_SWEEP}
    assert set(direct_labels(fig)) == {FT, GPT_OSS_20B, GPT_OSS_120B, QWEN_27B, gemini_label} | loads
    assert_labels_clear(fig, direct_labels(fig))


#: The fine-tune's exact match and the four API rows at the scale of the test runs in progress on 2026-10-04: the
#: three Groq rows and Gemini cost from about $0.10 to $0.90 per 1,000 calls, with one range overlapping the next.
IN_PROGRESS_RESULTS = {
    FT: (0.750, (0.712, 0.786), None),
    GPT_OSS_20B: (0.621, (0.555, 0.684), {"lower": 0.1058, "upper": 0.1277}),
    GPT_OSS_120B: (0.640, (0.575, 0.702), {"lower": 0.2144, "upper": 0.2577}),
    QWEN_27B: (0.705, (0.655, 0.752), {"lower": 0.896, "upper": 0.896}),
    GEMINI: (0.678, (0.636, 0.718), {"lower": 0.2004, "upper": 0.3562}),
}

#: The same rows at the scale of the dev-split runs (the exact match and costs in results/runs/*__dev/summary.D50.json), with
#: intervals as wide as S500's: here two API rows stand above the fine-tune's row as well as two below it.
DEV_SCALE_RESULTS = {
    FT: (0.750, (0.712, 0.786), None),
    GPT_OSS_20B: (0.680, (0.645, 0.715), {"lower": 0.1065, "upper": 0.1284}),
    GPT_OSS_120B: (0.760, (0.725, 0.795), {"lower": 0.2122, "upper": 0.2554}),
    QWEN_27B: (0.820, (0.785, 0.855), {"lower": 0.8907, "upper": 0.8907}),
    GEMINI: (0.820, (0.785, 0.855), {"lower": 0.1980, "upper": 0.3538}),
}

#: Four API rows crowded round the fine-tune's dearest load ($0.185) and its exact match: the gpt-oss-120b row stands right
#: beside the fine-tune's last marker, with the others above, below and to the right of it.
CROWDED_RESULTS = {
    FT: (0.8009, (0.7709, 0.8309), None),
    GPT_OSS_20B: (0.8785, (0.8342, 0.9227), {"lower": 0.0973, "upper": 0.1221}),
    GPT_OSS_120B: (0.8059, (0.7633, 0.8485), {"lower": 0.1271, "upper": 0.1669}),
    QWEN_27B: (0.7299, (0.7023, 0.7575), {"lower": 0.0965, "upper": 0.1743}),
    GEMINI: (0.7980, (0.7538, 0.8423), {"lower": 0.4937, "upper": 0.6984}),
}


def every_label(operating, names=(FT, *API_SYSTEMS)):
    """The direct labels of the cost figure with the real sweep: each row's name and each load's label."""
    return {*names, *(operating_label(load, operating) for load in REAL_SWEEP)}


@pytest.mark.parametrize("operating", [None, 8], ids=["no operating point", "operating point at 8"])
@pytest.mark.parametrize("results", [IN_PROGRESS_RESULTS, DEV_SCALE_RESULTS], ids=["test runs in progress", "dev runs"])
def test_the_real_sweep_next_to_the_real_scale_api_rows_keeps_every_label_clear_of_every_mark(world, results, operating):
    """The row from $0.007 to $0.185 is readable beside API ranges from $0.10 to $0.90: every label, the API rows'
    included, sits on no mark (a marker, a line or an interval bar), on no other label, and inside the plot."""
    _, doc, _ = world
    crowd = priced_at_the_real_sweep(doc, operating)
    set_results(crowd, results)
    fig = figures.plot_accuracy_vs_cost(crowd)
    assert set(direct_labels(fig)) == every_label(operating)
    assert_labels_clear(fig, direct_labels(fig))


def plain_slots():
    """The slots before the two fallbacks, which come last in LABEL_SLOTS."""
    assert [slot[0] for slot in figures.LABEL_SLOTS[8:]] == ["top", "bottom"]
    return figures.LABEL_SLOTS[:8]


def label_box(fig, text):
    renderer = _render(fig)
    return next(t for t in fig.axes[0].texts if t.get_text() == text).get_window_extent(renderer)


def bar_pixels(fig, doc, name):
    """(x, bottom, top) in pixels of an API row's interval bar, which stands at its upper cost bound. Draw the figure first."""
    system = next(s for s in doc["systems"] if s["name"] == name)
    low, high = system["metrics"]["exact_match"]["ci95"]
    x = system["cost"]["per_1k_calls_usd"]["upper"]
    transform = fig.axes[0].transData.transform
    (px, bottom), (_, top) = transform((x, low)), transform((x, high))
    return px, bottom, top


def test_a_name_with_no_clear_slot_beside_its_row_moves_above_its_own_interval_bar(world, monkeypatch):
    """At the scale of the runs in progress the plain slots leave the Gemini name on the interval bar of gpt-oss-120b. The
    first fallback takes it above the top end of Gemini's own bar, ending just left of it."""
    _, doc, _ = world
    crowd = priced_at_the_real_sweep(doc)
    set_results(crowd, IN_PROGRESS_RESULTS)
    with monkeypatch.context() as patched:  # not vacuous: without the fallbacks this layout has a label on a mark
        patched.setattr(figures, "LABEL_SLOTS", plain_slots())
        fig = figures.plot_accuracy_vs_cost(crowd)
        with pytest.raises(AssertionError, match=f"{GEMINI} overlaps a mark"):
            assert_labels_clear(fig, direct_labels(fig))
    fig = figures.plot_accuracy_vs_cost(crowd)
    assert_labels_clear(fig, direct_labels(fig))
    box = label_box(fig, GEMINI)
    x, _, top = bar_pixels(fig, crowd, GEMINI)
    assert box.y0 >= top and box.x1 <= x


def test_a_name_with_no_clear_slot_beside_its_row_moves_below_its_own_interval_bar(world, monkeypatch):
    """With the API rows crowded round the fine-tune's dearest load, the gpt-oss-120b name has no clear slot above, beside or
    below its row. The second fallback takes it below the bottom end of its own bar, ending just left of it."""
    _, doc, _ = world
    crowd = priced_at_the_real_sweep(doc)
    set_results(crowd, CROWDED_RESULTS)
    with monkeypatch.context() as patched:  # not vacuous: without the fallbacks this layout has a label on a mark
        patched.setattr(figures, "LABEL_SLOTS", plain_slots())
        fig = figures.plot_accuracy_vs_cost(crowd)
        with pytest.raises(AssertionError):
            assert_labels_clear(fig, direct_labels(fig))
    fig = figures.plot_accuracy_vs_cost(crowd)
    assert set(direct_labels(fig)) == every_label(None)
    assert_labels_clear(fig, direct_labels(fig))
    box = label_box(fig, GPT_OSS_120B)
    x, bottom, _ = bar_pixels(fig, crowd, GPT_OSS_120B)
    assert box.y1 <= bottom and box.x1 <= x


@pytest.mark.parametrize("operating", [None, 8], ids=["no operating point", "operating point at 8"])
def test_no_load_label_runs_on_from_the_row_name(world, operating):
    """With the Gemini name above its own bar, the fine-tune's last load has no room below its marker. It goes beside the
    marker, past the end of the line, and not above it, where it would stand straight after the row's name and read as
    part of it ("ft-qwen3-4b-lora x1"): no load label is on the name's line within a couple of letters of it."""
    _, doc, _ = world
    crowd = priced_at_the_real_sweep(doc, operating)
    set_results(crowd, IN_PROGRESS_RESULTS)
    fig = figures.plot_accuracy_vs_cost(crowd)
    name = label_box(fig, FT)
    for load in REAL_SWEEP:
        box = label_box(fig, operating_label(load, operating))
        on_the_names_line = name.y0 < box.y1 and box.y0 < name.y1
        assert not on_the_names_line or box.x0 - name.x1 > 24 or name.x0 - box.x1 > 24, load
    marker_x = fig.axes[0].transData.transform((REAL_SWEEP[1], IN_PROGRESS_RESULTS[FT][0]))[0]
    assert label_box(fig, "x1").x0 > marker_x  # the last load stands past the end of the line


def label_boxes(fig):
    renderer = _render(fig)
    return {t.get_text(): tuple(round(float(v), 6) for v in t.get_window_extent(renderer).extents) for t in fig.axes[0].texts}


def test_the_fallback_slots_change_nothing_where_the_first_slots_were_already_clear(world, sweep, monkeypatch):
    """They come last, and a slot wins only by overlapping less than every slot before it: a layout in which every label
    already had a clear place is laid out exactly as it was."""
    layouts = [world[1], sweep[1]]
    for operating in (None, 8):
        estimated = priced_at_the_real_sweep(world[1], operating)
        set_results(estimated, ESTIMATED_RESULTS)
        layouts.append(estimated)
    for doc in layouts:
        with_fallbacks = label_boxes(figures.plot_accuracy_vs_cost(doc))
        with monkeypatch.context() as patched:
            patched.setattr(figures, "LABEL_SLOTS", plain_slots())
            assert label_boxes(figures.plot_accuracy_vs_cost(doc)) == with_fallbacks


# --- every name beside its own row ---------------------------------------------------------------------------------------


#: The dry run of 2026-10-05: the API rows at their S500 exact match and cost per 1,000 calls, gpt-oss-20b still partial (413
#: of 500 items, so its final values will move a little). On this layout the Qwen name used to land far left of its own point
#: ($0.88), over the top of Gemini's no-caching bar ($0.36), where a reader took it for Gemini's.
DRY_RUN_RESULTS = {
    FT: (0.75, (0.712, 0.788), None),
    GPT_OSS_20B: (0.6198547, (0.5738499, 0.6682809), {"lower": 0.1072727, "upper": 0.1291987}),
    GPT_OSS_120B: (0.63, (0.588, 0.674), {"lower": 0.2166591, "upper": 0.2599341}),
    QWEN_27B: (0.708, (0.668, 0.748), {"lower": 0.8960672, "upper": 0.8960672}),
    GEMINI: (0.678, (0.636, 0.718), {"lower": 0.2003744, "upper": 0.3561644}),
}


def gap(a, b):
    """The distance in pixels between two boxes given as (x0, y0, x1, y1): 0 when they touch or overlap."""
    return math.hypot(max(a[0] - b[2], b[0] - a[2], 0), max(a[1] - b[3], b[1] - a[3], 0))


def row_marks(fig, doc, pad=2):
    """Everything drawn for each row as boxes in pixels (its markers, its range or chain line, its interval bar), found from
    the artists and attributed to a row by the exact match it is drawn at (a marker or a line) or by its interval (a bar),
    not from the placement code."""
    _render(fig)
    ax = fig.axes[0]
    results = {
        s["name"]: (round(s["metrics"]["exact_match"]["value"], 12), tuple(round(v, 12) for v in s["metrics"]["exact_match"]["ci95"]))
        for s in doc["systems"]
        if (s.get("metrics") or {}).get("exact_match") is not None
    }
    by_value = {value: name for name, (value, _) in results.items()}
    by_interval = {interval: name for name, (_, interval) in results.items()}
    assert len(by_value) == len(by_interval) == len(results)  # attribution by data needs distinct results
    marks: dict[str, list] = {name: [] for name in results}
    transform = ax.transData.transform
    for line in ax.lines:
        owner = by_value.get(round(line.get_ydata()[0], 12))
        if owner:
            (x, y), r = transform((line.get_xdata()[0], line.get_ydata()[0])), line.get_markersize() * fig.dpi / 72 / 2
            marks[owner].append((x - r, y - r, x + r, y + r))
    for collection in ax.collections:
        for (x0, y0), (x1, y1) in collection.get_segments():
            owner = by_interval.get((round(y0, 12), round(y1, 12))) if x0 == x1 else by_value.get(round(y0, 12))
            if owner:
                (px0, py0), (px1, py1) = transform((x0, y0)), transform((x1, y1))
                marks[owner].append((min(px0, px1) - pad, min(py0, py1) - pad, max(px0, px1) + pad, max(py0, py1) + pad))
    return {name: boxes for name, boxes in marks.items() if boxes}


LOAD_LABEL = re.compile(r"^x\d+( \(operating point\))?$")


def assert_labels_beside_their_own_rows(fig, doc):
    """Every row's name, and every load label of the fine-tune's row, is nearer to a mark of its own row than to any mark of
    any other row."""
    renderer = _render(fig)
    rows = row_marks(fig, doc)
    owners = {}  # label -> the row it belongs to
    for text in fig.axes[0].texts:
        label = text.get_text()
        if label in rows:
            owners[label] = label
        elif LOAD_LABEL.match(label):
            owners[label] = doc["reference"]
    assert set(owners.values()) == set(rows) and len([o for label, o in owners.items() if label != o]) >= 1  # not vacuous
    for text in fig.axes[0].texts:
        label = text.get_text()
        extents = text.get_window_extent(renderer)
        box = (extents.x0, extents.y0, extents.x1, extents.y1)
        own = min(gap(box, mark) for mark in rows[owners[label]])
        theirs, other = min((gap(box, mark), name) for name, marks in rows.items() if name != owners[label] for mark in marks)
        assert own < theirs, f"{label} is {own:.0f} px from its own marks and {theirs:.0f} px from those of {other}"


@pytest.mark.parametrize("operating", [None, 8], ids=["no operating point", "operating point at 8"])
def test_on_the_dry_run_every_name_sits_beside_its_own_row(world, monkeypatch, operating):
    """Every name, and every load label, is nearer its own row than any other. The first slot free of marks for the Qwen name
    was above-left of its point, 274 px wide and over the top of Gemini's bar, nearer Gemini's mark than Qwen's. A slot nearer
    another row's mark than its own now costs more than a small overlap, so the name goes beside its own point, where there
    is plenty of room."""
    _, doc, _ = world
    crowd = priced_at_the_real_sweep(doc, operating)
    set_results(crowd, DRY_RUN_RESULTS)
    with monkeypatch.context() as patched:  # not vacuous: with no cost for it, the old placement is back, and reads as Gemini's
        patched.setattr(figures, "AMBIGUOUS_COST", 0)
        fig = figures.plot_accuracy_vs_cost(crowd)
        with pytest.raises(AssertionError, match=f"{QWEN_27B} is"):
            assert_labels_beside_their_own_rows(fig, crowd)
    fig = figures.plot_accuracy_vs_cost(crowd)
    assert set(direct_labels(fig)) == every_label(operating)
    assert_labels_clear(fig, direct_labels(fig))
    assert_labels_beside_their_own_rows(fig, crowd)
    x, _, _ = bar_pixels(fig, crowd, QWEN_27B)
    assert label_box(fig, QWEN_27B).x0 > x  # Qwen's name stands to the right of its own point, in the free space there


#: The dev-split rows tie on exact match (Qwen and Gemini both 0.82), and the measure attributes a mark to a row by the exact
#: match it is drawn at, so Qwen is nudged by a tenth of a point here.
DEV_SCALE_UNTIED = {**DEV_SCALE_RESULTS, QWEN_27B: (0.821, (0.786, 0.856), DEV_SCALE_RESULTS[QWEN_27B][2])}
#: The real-scale and crowded layouts on which every label has a comfortable margin: another row's mark is at least a third
#: farther from it than its own. The layout of the runs in progress is left out: the gpt-oss-120b name stands within 3 px of
#: equally near to its own marker and to the foot of Gemini's bar, which the measure cannot call either way.
NAME_LAYOUTS = {
    "dry run": DRY_RUN_RESULTS, "dev runs": DEV_SCALE_UNTIED, "estimates": ESTIMATED_RESULTS, "crowded": CROWDED_RESULTS,
}


@pytest.mark.parametrize("operating", [None, 8], ids=["no operating point", "operating point at 8"])
@pytest.mark.parametrize("results", list(NAME_LAYOUTS.values()), ids=list(NAME_LAYOUTS))
def test_every_label_sits_nearer_its_own_row_than_any_other_on_the_real_scale_layouts(world, results, operating):
    """Each row's name, and each load label of the fine-tune, is nearer its own row's marks than any other row's."""
    _, doc, _ = world
    crowd = priced_at_the_real_sweep(doc, operating)
    set_results(crowd, results)
    fig = figures.plot_accuracy_vs_cost(crowd)
    assert_labels_beside_their_own_rows(fig, crowd)


def test_the_gap_between_boxes_is_zero_when_they_touch_and_the_straight_distance_otherwise():
    box = figures.Bbox([[0, 0], [10, 10]])
    assert figures._gap(box, figures.Bbox([[10, 0], [20, 10]])) == 0  # touching
    assert figures._gap(box, figures.Bbox([[5, 5], [20, 20]])) == 0  # overlapping
    assert figures._gap(box, figures.Bbox([[13, 0], [20, 10]])) == 3  # beside it
    assert figures._gap(box, figures.Bbox([[0, 14], [10, 20]])) == 4  # above it
    other = figures.Bbox([[13, 14], [20, 20]])  # across a corner: the 3-4-5 triangle
    assert figures._gap(box, other) == figures._gap(other, box) == 5


ABOVE, BELOW = ("middle", 0, 30, "center", "bottom"), ("middle", 0, -30, "center", "top")


def bare_axes():
    """A 100 x 100 plot with nothing in it, drawn, so that data and pixel coordinates are fixed."""
    fig = figures.plt.figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    fig.canvas.draw()
    return fig, ax


def slot_box(fig, ax, slot, anchor):
    """The box in pixels that the label "row A" would take in `slot`, which leaves nothing on the axes."""
    note = ax.annotate("row A", anchor, xytext=slot[1:3], textcoords="offset points", ha=slot[3], va=slot[4], fontsize=figures.FONT)
    box = note.get_window_extent(fig.canvas.get_renderer())
    note.remove()
    return box


def crossing(box, n=0):
    """A thin box that stands across `box`: the foot of an interval bar, in the way of a label there."""
    return figures.Bbox([[box.x0 + 8 + 12 * n, box.y0 - 4], [box.x0 + 12 + 12 * n, box.y1 + 4]])


def row_a_goes_above(fig, ax, anchor, own, others, cost, monkeypatch, before=()):
    """Whether the label of a row anchored at `anchor` takes ABOVE rather than BELOW, with AMBIGUOUS_COST set to `cost`.
    `before` are marks placed first, whose labels are then among the labels already taken."""
    mark = {"text": "row A", "middle": anchor, "own": own, "slots": (ABOVE, BELOW)}
    with monkeypatch.context() as patched:
        patched.setattr(figures, "AMBIGUOUS_COST", cost)
        figures._place_labels(fig, ax, [*before, mark], [*own, *others])
    return ax.texts[-1].get_window_extent(fig.canvas.get_renderer()).y0 > ax.transData.transform(anchor)[1]


def test_a_slot_nearer_another_rows_mark_costs_more_than_one_overlap_and_less_than_two(monkeypatch):
    """The cost of putting a name where it reads as another row's, in overlaps. A name with a clean slot above it that is 3 px
    from a neighbour's mark and a slot below it that its own bar crosses takes the one with the small overlap; with two of
    its own marks across the slot below it takes the clean one; and with no cost for ambiguity it takes the clean one."""
    assert 1 < figures.AMBIGUOUS_COST < 2
    fig, ax = bare_axes()
    anchor = (50, 50)
    x, y = ax.transData.transform(anchor)
    top, bottom = slot_box(fig, ax, ABOVE, anchor), slot_box(fig, ax, BELOW, anchor)
    marker = figures.Bbox([[x - 5, y - 5], [x + 5, y + 5]])
    neighbour = figures.Bbox([[top.x0, top.y1 + 3], [top.x1, top.y1 + 10]])  # another row's mark, 3 px over the slot above
    one, two = (marker, crossing(bottom)), (marker, crossing(bottom), crossing(bottom, 1))
    assert not row_a_goes_above(fig, ax, anchor, one, [neighbour], figures.AMBIGUOUS_COST, monkeypatch)
    assert row_a_goes_above(fig, ax, anchor, two, [neighbour], figures.AMBIGUOUS_COST, monkeypatch)
    assert row_a_goes_above(fig, ax, anchor, one, [neighbour], 0, monkeypatch)


def test_a_slot_that_touches_another_rows_mark_as_well_as_its_own_is_ambiguous(monkeypatch):
    """At a distance of 0 from both, a slot is as near the neighbour as its own row: ambiguous. With the two overlaps it has,
    that costs more than a slot below that three of the row's own marks cross, which is near no other row."""
    fig, ax = bare_axes()
    anchor = (50, 50)
    x, y = ax.transData.transform(anchor)
    top, bottom = slot_box(fig, ax, ABOVE, anchor), slot_box(fig, ax, BELOW, anchor)
    marker = figures.Bbox([[x - 5, y - 5], [x + 5, y + 5]])
    own_above = figures.Bbox([[top.x0 + 5, top.y0 - 3], [top.x0 + 9, top.y1 + 3]])  # the row's own mark across the slot above
    neighbour = figures.Bbox([[top.x1 - 9, top.y0 - 3], [top.x1 - 5, top.y1 + 3]])  # and another row's mark across it too
    own = (marker, own_above, crossing(bottom), crossing(bottom, 1), crossing(bottom, 2))
    assert not row_a_goes_above(fig, ax, anchor, own, [neighbour], figures.AMBIGUOUS_COST, monkeypatch)
    assert row_a_goes_above(fig, ax, anchor, own, [neighbour], 0, monkeypatch)  # it is the cost of ambiguity that decides


def test_a_neighbours_label_beside_a_slot_does_not_make_it_ambiguous(monkeypatch):
    """A reader attaches a label to the nearest mark, and a label is not a mark. The label of another row, placed first, that
    stands 3 px from the slot above costs that slot nothing: the name takes it and not the slot below that its own bar crosses."""
    fig, ax = bare_axes()
    anchor = (50, 50)
    x, y = ax.transData.transform(anchor)
    top, bottom = slot_box(fig, ax, ABOVE, anchor), slot_box(fig, ax, BELOW, anchor)
    marker = figures.Bbox([[x - 5, y - 5], [x + 5, y + 5]])
    far = figures.Bbox([[10, 10], [20, 20]])  # the mark of the other row, far from both slots
    beside = tuple(ax.transData.inverted().transform((top.x0 + 4, top.y1 + 3)))  # where its label's corner stands
    neighbour = {"text": "row B", "middle": beside, "own": (far,), "slots": (("middle", 0, 0, "left", "bottom"),)}
    assert row_a_goes_above(fig, ax, anchor, (marker, crossing(bottom)), [far], figures.AMBIGUOUS_COST, monkeypatch, before=[neighbour])


@pytest.mark.parametrize("draw", ["plot_accuracy_vs_cost", "plot_latency"])
def test_nothing_is_clipped_by_the_edge_of_the_figure_and_nothing_collides(either, draw):
    _, doc, _ = either
    fig = getattr(figures, draw)(doc)
    renderer = _render(fig)
    frame = fig.bbox
    items = list(fig.texts) + list(fig.axes[0].texts) + list(fig.legends[0].get_texts())
    boxes = []
    for item in items:
        if not item.get_text():
            continue
        box = item.get_window_extent(renderer)
        assert frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0 and box.y1 <= frame.y1, item.get_text()
        boxes.append((item.get_text(), box))
    legend_box = fig.legends[0].get_window_extent(renderer)
    note = next(t for t in fig.texts if t.get_text()).get_window_extent(renderer)
    assert not legend_box.overlaps(note)  # the note sits under the legend
    if draw == "plot_accuracy_vs_cost":  # the direct labels do not sit on each other
        names = set(direct_labels(fig))
        labels = [(t, b) for t, b in boxes if t in names]
        assert {FT, *API_SYSTEMS} <= {t for t, _ in labels}
        for i, (_, a) in enumerate(labels):
            for _, b in labels[i + 1 :]:
                assert not a.overlaps(b)


@pytest.mark.parametrize("draw", ["plot_accuracy_vs_cost", "plot_latency"])
def test_the_axis_text_stays_inside_the_figure_and_clear_of_the_legend(either, draw):
    """The tick labels and the axis labels (the latency figure's are two lines tall) are not cut off by the figure's
    edge and do not run into the legend under the plot."""
    _, doc, _ = either
    fig = getattr(figures, draw)(doc)
    renderer = _render(fig)
    ax = fig.axes[0]
    low, high = ax.get_xlim()
    ticks = [t for t in ax.get_xticklabels() if t.get_text() and low <= t.get_position()[0] <= high]
    assert ticks  # not vacuous
    legend_box = fig.legends[0].get_window_extent(renderer)
    for item in (*ticks, ax.xaxis.label, ax.yaxis.label):
        box = item.get_window_extent(renderer)
        assert fig.bbox.x0 <= box.x0 and box.x1 <= fig.bbox.x1 and fig.bbox.y0 <= box.y0 and box.y1 <= fig.bbox.y1, item.get_text()
        assert not box.overlaps(legend_box), item.get_text()


def test_the_note_under_the_cost_figure_fits_when_every_optional_sentence_is_there(sweep):
    """The note under the legend gets a sentence for the self-hosted row, one for rows with no result or cost yet and
    one for loads with no cost; with all three it is still inside the figure and clear of the legend."""
    _, doc, _ = sweep
    long = copy.deepcopy(doc)
    next(level for level in ft_of(long)["cost_by_load"]["levels"] if level["concurrency"] == 8).update(
        per_1k_calls_usd={"on_demand": None, "spot": None}
    )
    for system in long["systems"]:
        if system["name"] in (GPT_OSS_120B, QWEN_27B):
            system["metrics"] = None  # no result yet: left out and named
    fig = figures.plot_accuracy_vs_cost(long)
    renderer = _render(fig)
    note = next(t for t in fig.texts if t.get_text())
    for sentence in ("so it has one bar", "no result or no cost yet", "Load levels not shown, no cost"):
        assert sentence in note.get_text().replace("\n", " ")
    box = note.get_window_extent(renderer)
    lines = note.get_text().count("\n") + 1
    assert lines >= 4  # not vacuous: this is a longer note than the plain one
    assert fig.bbox.x0 <= box.x0 and box.x1 <= fig.bbox.x1 and fig.bbox.y0 <= box.y0 and box.y1 <= fig.bbox.y1
    assert box.y0 >= box.height / lines  # with room under it for one more line, as the footer is sized to allow
    assert not fig.legends[0].get_window_extent(renderer).overlaps(box)


# --- latency -------------------------------------------------------------------------------------------------------


def test_the_latency_figure_shows_p50_and_p95_at_every_measured_load(either):
    _, doc, _ = either
    levels = ft_of(doc)["cost_by_load"]["levels"]
    assert len(levels) >= 2
    fig = figures.plot_latency(doc)
    ax = fig.axes[0]
    bars = [container.patches[0] for container in ax.containers]  # one bar a call, so one container each
    expected = [level[f"{quantile}_s"] for level in levels for quantile in ("p50", "p95")]
    assert [round(bar.get_height(), 9) for bar in bars] == [round(value, 9) for value in expected]
    # p50 then p95 side by side at each load, loads left to right, in the first two steps of the blue ramp
    assert [bar.get_facecolor() for bar in bars] == [to_rgba(figures.RAMP[i % 2]) for i in range(len(bars))]
    centres = [bar.get_x() + bar.get_width() / 2 for bar in bars]
    for i in range(len(levels)):
        assert (centres[2 * i] + centres[2 * i + 1]) / 2 == pytest.approx(i)  # the pair straddles its tick
        assert centres[2 * i] < centres[2 * i + 1]
    assert [t.get_text() for t in ax.get_xticklabels()] == [str(level["concurrency"]) for level in levels]
    assert ax.get_xlabel() == "Concurrent requests" and ax.get_ylabel() == "Seconds per request"
    assert ax.get_title(loc="left") == "Self-hosted latency by load, on the box"
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["p50", "p95"]
    assert sorted(t.get_text() for t in ax.texts if t.get_text() != "operating point") == sorted(f"{value:.2f}" for value in expected)


def test_the_latency_figure_shows_nothing_from_the_apis(either):
    _, doc, _ = either
    joined = " ".join(texts(figures.plot_latency(doc)))
    for name in (BASE, *API_SYSTEMS):
        assert name not in joined
    assert "free tiers" in joined.replace("\n", " ")  # only the note that API latency is an appendix table, with its label


def test_the_operating_point_is_marked_only_when_a_level_met_the_rule(world, sweep):
    _, with_point, _ = world
    fig = figures.plot_latency(with_point)
    ax = fig.axes[0]
    loads = [level["concurrency"] for level in ft_of(with_point)["cost_by_load"]["levels"]]
    bars = {patch for container in ax.containers for patch in container}
    columns = [patch for patch in ax.patches if patch not in bars]
    assert len(columns) == 1  # one light column, behind the bars of the level the rule chose
    _render(fig)
    extent = columns[0].get_extents()  # in pixels; back to the axis units, where a level is its position among the loads
    inverse = ax.transData.inverted()
    assert (inverse.transform((extent.x0, 0))[0], inverse.transform((extent.x1, 0))[0]) == pytest.approx(
        (loads.index(8) - 0.5, loads.index(8) + 0.5)
    )
    assert [t.get_text() for t in ax.texts].count("operating point") == 1 and len(ax.lines) == 0  # a column and its name
    _, without, _ = sweep
    assert ft_of(without)["cost_by_load"]["operating_point"] is None
    fig = figures.plot_latency(without)
    ax = fig.axes[0]
    bars = {patch for container in ax.containers for patch in container}
    assert [patch for patch in ax.patches if patch not in bars] == [] and len(ax.lines) == 0  # nothing marked
    assert [t for t in texts(fig) if "operating point" in t] == []  # and nothing said about one


def test_a_level_missing_one_of_its_latencies_keeps_its_place_and_draws_the_other(sweep):
    _, doc, _ = sweep
    gap = copy.deepcopy(doc)
    next(level for level in ft_of(gap)["cost_by_load"]["levels"] if level["concurrency"] == 8)["p50_s"] = None
    ax = figures.plot_latency(gap).axes[0]
    assert len(ax.containers) == 7 and [t.get_text() for t in ax.get_xticklabels()] == ["1", "8", "32", "64"]


def test_the_latency_figure_is_the_fine_tunes_whatever_else_was_benchmarked(sweep):
    """It reads the reference system's loads; another self-hosted row with a benchmark is not added to it."""
    _, doc, _ = sweep
    two = copy.deepcopy(doc)
    next(s for s in two["systems"] if s["name"] == BASE)["cost_by_load"] = copy.deepcopy(ft_of(two)["cost_by_load"])
    fig = figures.plot_latency(two)
    assert len(fig.axes[0].containers) == 8
    assert figures.latency_levels(two)["name"] == FT
    assert FT in next(t for t in texts(fig) if t.startswith("System:")) and BASE not in " ".join(texts(fig))


def test_the_latency_figure_names_its_system_and_points_to_the_api_appendix_with_its_label(either):
    _, doc, _ = either
    note = next(t for t in texts(figures.plot_latency(doc)) if t.startswith("System:")).replace("\n", " ")
    assert note.startswith(f"System: {FT}. GPU: Tesla T4. ")
    assert "observed on free tiers from India; not representative of paid tiers" in note


# --- the real throughput benchmark ------------------------------------------------------------------------------------------


def set_gpu_prices(lab, usd_per_hour):
    path = lab.config_dir / "sources.yaml"
    sources = yaml.safe_load(path.read_text(encoding="utf-8"))
    sources["gpu_rental"]["aws-g4dn.xlarge"]["usd_per_hour"] = dict(usd_per_hour)
    path.write_text(yaml.safe_dump(sources, sort_keys=False, allow_unicode=True), encoding="utf-8")


@pytest.mark.skipif(not REAL_T4.exists(), reason="results/serving/T4.json is not in this checkout")
def test_the_real_t4_benchmark_goes_through_compare_into_both_figures(tmp_path, monkeypatch):
    """results/serving/T4.json, copied into a scratch layout, compared and drawn: four loads (1, 8, 32 and 64), none of
    which met the p95 rule, so a marker and a pair of bars at each and nothing marked as the operating point."""
    raw = json.loads(REAL_T4.read_text(encoding="utf-8"))
    monkeypatch.setattr(config, "git_commit", lambda cwd=None: None)
    monkeypatch.setattr(config, "git_dirty", lambda cwd=None: False)
    lab = Lab(tmp_path / "lab").populate()  # the six systems' runs, and a stand-in benchmark
    set_gpu_prices(lab, raw["cost"]["rental"]["usd_per_hour"])  # the prices the benchmark was costed at, so its own costs apply
    (lab.results / "serving" / "t4.json").unlink()
    lab.write_serving(name="T4", doc=raw)
    doc = compare.build_comparison(results_dir=lab.results, processed_dir=lab.processed, config_dir=lab.config_dir, n_resamples=100)
    comparison = lab.results / "comparison.json"
    comparison.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    assert doc["self_hosted"]["benchmark"]["path"] == "results/serving/T4.json"
    lines: list[str] = []
    assert figures.run(comparison_path=comparison, out_dir=tmp_path / "figures", out=lines.append) == 0
    assert not [line for line in lines if line.startswith("skipped")]
    for name in ("accuracy_vs_cost.png", "latency.png"):
        assert (tmp_path / "figures" / name).read_bytes().startswith(PNG_SIGNATURE)

    ft = ft_of(doc)
    assert ft["cost"] is None and ft["cost_by_load"]["operating_point"] is None  # the rule found no level on the T4
    recorded = raw["cost"]["by_concurrency"]  # what the benchmark itself says each load costs, at the on-demand rate
    fig = figures.plot_accuracy_vs_cost(doc)
    markers = sorted(
        (line.get_xdata()[0], line.get_ydata()[0]) for line in fig.axes[0].lines if line.get_markerfacecolor() == figures.SELF_HOSTED
    )
    assert [x for x, _ in markers] == pytest.approx(sorted(recorded[str(load)]["on_demand"] for load in (1, 8, 32, 64)), rel=1e-9)
    assert {y for _, y in markers} == {ft["metrics"]["exact_match"]["value"]}
    assert {"x1", "x8", "x32", "x64"} <= set(direct_labels(fig))
    assert [t for t in texts(fig) if "operating point" in t] == []
    fig = figures.plot_latency(doc)
    bars = [container.patches[0].get_height() for container in fig.axes[0].containers]
    assert bars == pytest.approx([level["latency_s"][q] for level in raw["levels"] for q in ("p50", "p95")])
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["1", "8", "32", "64"]
    assert [t for t in texts(fig) if "operating point" in t] == []


# --- missing results --------------------------------------------------------------------------------------------------------


def test_a_comparison_with_no_results_writes_no_figures_and_says_why(tmp_path):
    results = tmp_path / "results"
    compare.run(results_dir=results, processed_dir=tmp_path / "no-data", out=lambda line: None)
    lines: list[str] = []
    assert figures.run(comparison_path=results / "comparison.json", out=lines.append) == 0
    assert not (results / "figures").exists()
    assert any(line.startswith("skipped accuracy_vs_cost.png") for line in lines)
    assert any(line.startswith("skipped latency.png") for line in lines)


def test_results_without_a_benchmark_draw_no_figure_at_all_for_lack_of_costs_and_latency(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "git_commit", lambda cwd=None: None)
    monkeypatch.setattr(config, "git_dirty", lambda cwd=None: False)
    lab = Lab(tmp_path)
    lab.write_run(FT)  # a self-hosted row and no throughput benchmark: no cost, no latency
    compare.run(results_dir=lab.results, processed_dir=lab.processed, config_dir=lab.config_dir, n_resamples=50, out=lambda line: None)
    lines: list[str] = []
    assert figures.run(comparison_path=lab.results / "comparison.json", out=lines.append) == 0
    assert len(lines) == 2 and all(line.startswith("skipped") for line in lines)


def test_an_api_only_set_of_results_still_draws_the_cost_figure(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "git_commit", lambda cwd=None: None)
    monkeypatch.setattr(config, "git_dirty", lambda cwd=None: False)
    lab = Lab(tmp_path)
    lab.write_run(GPT_OSS_20B)
    lab.write_audit()
    compare.run(results_dir=lab.results, processed_dir=lab.processed, config_dir=lab.config_dir, n_resamples=50, out=lambda line: None)
    lines: list[str] = []
    assert figures.run(comparison_path=lab.results / "comparison.json", out=lines.append) == 0
    assert (lab.results / "figures" / "accuracy_vs_cost.png").exists() and not (lab.results / "figures" / "latency.png").exists()


def test_a_missing_or_unusable_comparison_is_an_error(tmp_path):
    lines: list[str] = []
    assert figures.run(comparison_path=tmp_path / "nope.json", out=lines.append) == 2
    assert "run scripts/compare.py first" in lines[-1]
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert figures.run(comparison_path=bad, out=lines.append) == 2 and "not valid JSON" in lines[-1]
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"schema_version": 99}))
    assert figures.run(comparison_path=old, out=lines.append) == 2 and "schema_version 99" in lines[-1]
    assert not (tmp_path / "figures").exists()


def test_main_takes_the_paths_on_the_command_line(world, tmp_path):
    _, _, path = world
    assert figures.main(["--comparison", str(path), "--out-dir", str(tmp_path / "out")], out=lambda line: None) == 0
    assert (tmp_path / "out" / "accuracy_vs_cost.png").exists()


# --- headless -----------------------------------------------------------------------------------------------------------------


def test_it_runs_as_a_script_with_no_display_and_a_gui_backend_requested(either, tmp_path):
    _, _, path = either
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    env["MPLBACKEND"] = "TkAgg"  # a GUI backend, which would fail or open a window; the script must override it
    done = subprocess.run(
        [sys.executable, str(SCRIPTS / "make_figures.py"), "--comparison", str(path), "--out-dir", str(tmp_path / "headless")],
        env=env, capture_output=True, text=True, timeout=180,
    )
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "headless" / "accuracy_vs_cost.png").read_bytes().startswith(PNG_SIGNATURE)
    assert (tmp_path / "headless" / "latency.png").exists()


def test_importing_the_script_selects_the_non_interactive_backend():
    probe = (
        "import importlib.util, matplotlib;"
        f"spec = importlib.util.spec_from_file_location('mf', r'{SCRIPTS / 'make_figures.py'}');"
        "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module);"
        "print(matplotlib.get_backend().lower())"
    )
    env = {**os.environ, "MPLBACKEND": "pdf"}
    done = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "agg"
