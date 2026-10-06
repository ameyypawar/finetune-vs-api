"""scripts/sample_errors.py: the errors drawn for hand labelling, on synthetic runs (no network, no model)."""

from __future__ import annotations

import csv

import pytest

from conftest import load_script
from finetune_vs_api import config
from results_fixtures import BASE, FT, GEMINI, GPT_OSS_20B, GPT_OSS_120B, QWEN_27B, Lab

sample_errors = load_script("sample_errors")
render = load_script("render")


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """Runs in which qwen3.8-27b is the strongest API row (3 errors to the others' 6 or 8).

    On the first ids of S500: the fine-tune gets 0, 1 and 2 wrong; qwen3.8-27b gets 0, 3 and 4 wrong, 4 by a failed
    call. So the gap stratum is 3 and 4, and the ft stratum 0, 1 and 2.
    """
    monkeypatch.setattr(config, "git_commit", lambda cwd=None: None)
    monkeypatch.setattr(config, "git_dirty", lambda cwd=None: False)
    lab = Lab(tmp_path / "lab")
    ids = lab.ids("S500")
    assert len(ids) >= 10  # not vacuous
    lab.write_run(FT, wrong=ids[:3])
    lab.write_run(BASE, wrong=ids[:8])
    lab.write_run(GPT_OSS_20B, wrong=ids[:8])
    lab.write_run(GPT_OSS_120B, wrong=ids[:8])
    lab.write_run(QWEN_27B, wrong=[ids[0], ids[3]], errors=[ids[4]])
    lab.write_run(GEMINI, wrong=ids[:6])
    return lab


def write(lab: Lab, name: str = "error_analysis.csv", **kwargs) -> tuple[int, list[dict[str, str]], list[str]]:
    lines: list[str] = []
    path = lab.results / name
    code = sample_errors.run(
        results_dir=lab.results, processed_dir=lab.processed, config_dir=lab.config_dir, out_path=path,
        out=lines.append, **kwargs,
    )
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline=""))) if path.exists() else []
    return code, rows, lines


def test_the_two_strata_against_the_strongest_api_row(lab):
    ids = lab.ids("S500")
    code, rows, lines = write(lab)
    assert code == 0
    assert list(rows[0]) == sample_errors.COLUMNS
    gap = [row for row in rows if row["stratum"] == f"ft right, {QWEN_27B} wrong"]
    ft = [row for row in rows if row["stratum"] == "ft wrong"]
    assert len(gap) + len(ft) == len(rows) == 5
    assert {row["id"] for row in gap} == {ids[3], ids[4]} and {row["id"] for row in ft} == set(ids[:3])
    assert {row["api_system"] for row in rows} == {QWEN_27B}
    assert all(row["ft_differs"] == "none" for row in gap) and all(row["ft_differs"] != "none" for row in ft)
    failed = next(row for row in rows if row["id"] == ids[4])
    assert failed["api_answer"] == "(no answer: the call failed)" and failed["api_differs"] == "no valid answer"
    assert all(row["category"] == "" and row["note"] == "" for row in rows)  # left to the person labelling
    by_id = {e.id: e for e in lab.test}
    for row in rows:
        assert row["request"] == by_id[row["id"]].text and row["scenario"] == by_id[row["id"]].scenario
    assert f"strongest API row {QWEN_27B}" in lines[0] and "ft wrong: 3 of 3" in lines[-1]


def test_a_stratum_over_the_cap_is_a_seeded_sample(lab):
    ids = lab.ids("S500")
    _, first, lines = write(lab, "a.csv", per_stratum=2)
    _, again, _ = write(lab, "b.csv", per_stratum=2)
    assert first == again  # the same seed draws the same items
    ft = {row["id"] for row in first if row["stratum"] == "ft wrong"}
    assert len(ft) == 2 and ft <= set(ids[:3]) and "ft wrong: 2 of 3" in lines[-1]
    assert {row["id"] for row in first if row["stratum"] != "ft wrong"} == {ids[3], ids[4]}  # 2 of 2: all of them
    draws = {frozenset(sample_errors.draw(ids[:3], 2, seed, "ft")) for seed in range(20)}
    assert len(draws) > 1  # the seed matters


def test_the_strata_draw_independently():
    """A larger pool in one stratum never changes what is drawn from the other."""
    pool = [str(n) for n in range(30)]
    assert sample_errors.draw(pool, 5, 7, "ft") == sample_errors.draw(pool, 5, 7, "ft")
    assert sample_errors.draw(pool, 5, 7, "ft") != sample_errors.draw(pool, 5, 7, "gap")


def test_rows_are_ordered_by_scenario_within_each_stratum(lab):
    _, rows, _ = write(lab)
    for stratum in {row["stratum"] for row in rows}:
        scenarios = [row["scenario"] for row in rows if row["stratum"] == stratum]
        assert scenarios == sorted(scenarios)


def test_it_never_overwrites_hand_labels(lab):
    path = lab.results / "error_analysis.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("id,category\n1,label noise\n", encoding="utf-8")
    code, _, lines = write(lab)
    assert code == sample_errors.EXIT_EXISTS and "nothing written" in lines[0]
    assert path.read_text(encoding="utf-8") == "id,category\n1,label noise\n"
    code, rows, _ = write(lab, force=True)
    assert code == 0 and len(rows) == 5


def test_no_complete_reference_run_is_an_error(lab):
    (lab.results / "runs" / f"{FT}__test" / "predictions.jsonl").write_text("", encoding="utf-8")
    code, rows, lines = write(lab)
    assert code == sample_errors.EXIT_ERROR and rows == [] and FT in lines[0]


def test_render_counts_the_categories_once_labelled(lab, tmp_path):
    write(lab)
    assert render.read_error_analysis(lab.results, tmp_path / "docs") == (None, None)  # nothing labelled: no claim yet
    path = lab.results / "error_analysis.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    for row, category in zip(rows, ["convention", "convention", "label noise", "misread", ""], strict=True):
        row["category"] = category
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_errors.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary, _ = render.read_error_analysis(lab.results, tmp_path / "docs")
    assert summary["n"] == 5
    assert summary["categories"] == [("convention", 2), ("(blank)", 1), ("label noise", 1), ("misread", 1)]


@pytest.mark.parametrize(
    ("pred", "expected"),
    [
        (None, "no valid answer"),
        ({"intent": "alarm_set", "slots": [{"type": "time", "value": "Nine  AM"}]}, "none"),  # compared as the metrics compare
        ({"intent": "alarm_query", "slots": [{"type": "time", "value": "nine am"}]}, "intent alarm_query (gold alarm_set)"),
        ({"intent": "alarm_set", "slots": [{"type": "time", "value": "nine"}]}, "time 'nine' (gold 'nine am')"),
        ({"intent": "alarm_set", "slots": [{"type": "timeofday", "value": "nine am"}]}, "'nine am' as timeofday (gold time)"),
        ({"intent": "alarm_set", "slots": []}, "missing time=nine am"),
        (
            {"intent": "alarm_set", "slots": [{"type": "time", "value": "nine am"}, {"type": "date", "value": "friday"}]},
            "extra date=friday",
        ),
    ],
)
def test_differences_say_what_an_answer_gets_wrong(pred, expected):
    gold = {"intent": "alarm_set", "slots": [{"type": "time", "value": "nine am"}]}
    assert sample_errors.differences(pred, gold) == expected


def test_a_call_reads_on_one_line():
    call = {"intent": "alarm_set", "slots": [{"type": "time", "value": "nine am"}, {"type": "date", "value": "friday"}]}
    assert sample_errors.call_text(call) == "alarm_set | time=nine am; date=friday"
    assert sample_errors.call_text({"intent": "qa_factoid", "slots": []}) == "qa_factoid"
