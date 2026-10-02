"""scripts/make_figures.py: the files are written, the numbers are where the comparison says they are,
the cost axis is logarithmic, and it all works without a display. Synthetic results only."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import pytest
from PIL import Image

from conftest import SCRIPTS, load_script
from finetune_vs_api import config
from results_fixtures import BASE, FT, FULLER, GROQ, MINI, Lab

compare = load_script("compare")
figures = load_script("make_figures")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The standard results compared once; the comparison written where make_figures looks for it."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "git_commit", lambda cwd=None: None)
        mp.setattr(config, "git_dirty", lambda cwd=None: False)
        lab = Lab(tmp_path_factory.mktemp("figures")).populate()
        doc = compare.build_comparison(
            results_dir=lab.results, processed_dir=lab.processed, config_dir=lab.config_dir, n_resamples=100
        )
    path = lab.results / "comparison.json"
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return lab, doc, path


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


# --- files --------------------------------------------------------------------------------------------------


def test_both_figures_are_written_as_png_files(world, tmp_path):
    _, _, path = world
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


def test_the_figures_are_byte_for_byte_reproducible(world, tmp_path):
    _, _, path = world
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
    assert (round(ft["cost"]["per_1k_calls_usd"], 12), round(ft["metrics"]["exact_match"]["value"], 12), figures.SELF_HOSTED) in drawn
    for name in (MINI, FULLER, GROQ):
        s = by_name[name]
        y = round(s["metrics"]["exact_match"]["value"], 12)
        bounds = s["cost"]["per_1k_calls_usd"]
        assert (round(bounds["upper"], 12), y, figures.API) in drawn  # filled: no caching
        assert (round(bounds["lower"], 12), y, figures.SURFACE) in drawn  # hollow: cached prefix
    assert len(drawn) == 1 + 3 * 2


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
    for name in (FT, MINI, GROQ, f"{FULLER} (S300)"):  # gpt-4.1's subset is part of its label
        assert name in labels


def test_the_legend_explains_what_the_colours_and_the_ends_mean(world):
    _, doc, _ = world
    fig = figures.plot_accuracy_vs_cost(doc)
    entries = [t.get_text() for t in fig.legends[0].get_texts()]
    assert entries == [
        "self-hosted: GPU kept busy, on-demand price",
        "API: paid list price, from cached prefix (hollow end) to no caching (filled end)",
    ]


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
    mini, groq = by_name[MINI], by_name[GROQ]
    groq["metrics"]["exact_match"] = {"value": mini["metrics"]["exact_match"]["value"] + 0.002, "ci95": list(mini["metrics"]["exact_match"]["ci95"])}
    groq["cost"]["per_1k_calls_usd"] = {k: v * 1.03 for k, v in mini["cost"]["per_1k_calls_usd"].items()}
    fig = figures.plot_accuracy_vs_cost(crowded)
    renderer = _render(fig)
    boxes = {t.get_text(): t.get_window_extent(renderer) for t in fig.axes[0].texts}
    assert {MINI, GROQ} <= set(boxes)
    names = list(boxes)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            assert not boxes[first].overlaps(boxes[second]), (first, second)


def test_a_realistic_crowd_of_api_points_gets_labels_clear_of_every_mark(world):
    """Costs and accuracies like the real runs will have: three API ranges close together, one long name on the right."""
    _, doc, _ = world
    crowd = copy.deepcopy(doc)
    by_name = {s["name"]: s for s in crowd["systems"]}
    values = {  # exact match with its interval, and the cost (a range for an API)
        FT: (0.908, (0.882, 0.932), 0.0099),
        MINI: (0.848, (0.816, 0.878), {"lower": 0.301, "upper": 0.751}),
        GROQ: (0.844, (0.812, 0.876), {"lower": 0.333, "upper": 0.445}),
        FULLER: (0.870, (0.830, 0.907), {"lower": 1.51, "upper": 3.76}),
    }
    for name, (em, interval, price) in values.items():
        by_name[name]["metrics"]["exact_match"] = {"value": em, "ci95": list(interval), "n": 500}
        by_name[name]["cost"]["per_1k_calls_usd"] = price
    fig = figures.plot_accuracy_vs_cost(crowd)
    renderer = _render(fig)
    ax = fig.axes[0]
    pad = 2  # pixels
    marks = []  # every drawn mark, as a box in pixels, found from the artists and not from the placement code
    for line in ax.lines:
        (x, y), r = ax.transData.transform((line.get_xdata()[0], line.get_ydata()[0])), line.get_markersize() * fig.dpi / 72 / 2
        marks.append((x - r, y - r, x + r, y + r))
    for collection in ax.collections:
        for (x0, y0), (x1, y1) in (ax.transData.transform(segment) for segment in collection.get_segments()):
            marks.append((min(x0, x1) - pad, min(y0, y1) - pad, max(x0, x1) + pad, max(y0, y1) + pad))
    labels = {t.get_text(): t.get_window_extent(renderer) for t in ax.texts}
    assert set(labels) == {FT, MINI, GROQ, f"{FULLER} (S300)"}
    inside = ax.bbox
    for name, box in labels.items():
        assert inside.x0 <= box.x0 and box.x1 <= inside.x1 and inside.y0 <= box.y0 and box.y1 <= inside.y1, f"{name} leaves the plot"
        for x0, y0, x1, y1 in marks:
            assert box.x1 < x0 or box.x0 > x1 or box.y1 < y0 or box.y0 > y1, f"{name} overlaps a mark"
    boxes = list(labels.items())
    for i, (first, a) in enumerate(boxes):
        for second, b in boxes[i + 1 :]:
            assert not a.overlaps(b), (first, second)


@pytest.mark.parametrize("draw", ["plot_accuracy_vs_cost", "plot_latency"])
def test_nothing_is_clipped_by_the_edge_of_the_figure_and_nothing_collides(world, draw):
    _, doc, _ = world
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
        labels = [(t, b) for t, b in boxes if t in {FT, MINI, GROQ, f"{FULLER} (S300)"}]
        for i, (_, a) in enumerate(labels):
            for _, b in labels[i + 1 :]:
                assert not a.overlaps(b)


# --- latency -------------------------------------------------------------------------------------------------------


def test_the_latency_figure_shows_the_self_hosted_numbers_and_nothing_from_the_apis(world):
    _, doc, _ = world
    fig = figures.plot_latency(doc)
    ax = fig.axes[0]
    heights = [round(patch.get_height(), 9) for patch in ax.patches]
    assert heights == [0.4, 0.6, 1.2]  # p50 and p95 at concurrency 1, p95 at the operating point
    labels = texts(fig)
    assert ax.get_title(loc="left") == "Self-hosted latency, on the box"
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["p50, concurrency 1", "p95, concurrency 1", "p95 at the operating point"]
    assert "ft-qwen3-4b-lora\noperating point: concurrency 8" in labels
    joined = " ".join(labels)
    for api in (MINI, FULLER, GROQ):
        assert api not in joined


def test_the_latency_figure_points_to_the_api_appendix_with_its_label(world):
    _, doc, _ = world
    note = next(t for t in texts(figures.plot_latency(doc)) if t.startswith("GPU:")).replace("\n", " ")
    assert "GPU: Tesla T4." in note
    assert "observed on free tiers from India; not representative of paid tiers" in note


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
    lab.write_run(MINI)
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


def test_it_runs_as_a_script_with_no_display_and_a_gui_backend_requested(world, tmp_path):
    _, _, path = world
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
