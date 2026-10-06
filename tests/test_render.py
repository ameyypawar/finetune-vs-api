"""scripts/render.py and its templates: the README region, the model card, the write-up and --check.

Everything runs on scratch copies of the real README, NOTICE.md and templates with synthetic
results (tests/results_fixtures.py), and one test checks the committed files against the real repository.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path

import pytest
import yaml

from conftest import ROOT, load_script
from finetune_vs_api import config
from results_fixtures import (
    API_SYSTEMS,
    BASE,
    BREAK_EVEN_SERVING,
    FT,
    GEMINI,
    GPT_OSS_20B,
    GPT_OSS_120B,
    NO_OPERATING_POINT_SERVING,
    QWEN_27B,
    SYSTEMS,
    Lab,
)

compare = load_script("compare")
figures = load_script("make_figures")
render = load_script("render")

START, END = "<!-- results:start -->", "<!-- results:end -->"
NOTICE = "*All API rows ran on free tiers; no money was spent; costs are at paid list prices."
FIGURE_FILES = ("accuracy_vs_cost.png", "latency.png")
REAL_T4 = ROOT / "results" / "serving" / "T4.json"


# --- scratch repositories ---------------------------------------------------------------------------------


def comparison_for(lab: Lab, n_resamples: int = 100) -> dict:
    doc = compare.build_comparison(
        results_dir=lab.results, processed_dir=lab.processed, config_dir=lab.config_dir, n_resamples=n_resamples
    )
    (lab.results / "comparison.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


@pytest.fixture(scope="module")
def standard(tmp_path_factory):
    """The standard synthetic results, compared once, with a training log, an error analysis and figures."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "git_commit", lambda cwd=None: None)
        mp.setattr(config, "git_dirty", lambda cwd=None: False)
        lab = Lab(tmp_path_factory.mktemp("render")).populate()
        lab.write_train_log()
        lab.write_error_analysis(["wrong intent", "wrong intent", "span boundary", "label noise"])
        doc = comparison_for(lab)
    figures.run(comparison_path=lab.results / "comparison.json", out=lambda line: None)
    return lab, doc


@pytest.fixture(scope="module")
def no_operating_point(tmp_path_factory):
    """The standard results with a benchmark in which no level meets the p95 limit, as on the real T4: four load levels."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "git_commit", lambda cwd=None: None)
        mp.setattr(config, "git_dirty", lambda cwd=None: False)
        lab = Lab(tmp_path_factory.mktemp("render-no-operating-point")).populate()
        lab.write_serving(doc=NO_OPERATING_POINT_SERVING)
        lab.write_train_log()
        lab.write_error_analysis(["wrong intent", "wrong intent", "span boundary", "label noise"])
        doc = comparison_for(lab)
    figures.run(comparison_path=lab.results / "comparison.json", out=lambda line: None)
    return lab, doc


@pytest.fixture(scope="module")
def straddling(tmp_path_factory):
    """Levels that serve some API break-even volumes at one cost bound only (see BREAK_EVEN_SERVING)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "git_commit", lambda cwd=None: None)
        mp.setattr(config, "git_dirty", lambda cwd=None: False)
        lab = Lab(tmp_path_factory.mktemp("render-straddling")).populate()
        lab.write_serving(doc=BREAK_EVEN_SERVING)
        doc = comparison_for(lab)
    return lab, doc


class Repo:
    """A scratch repository: the real README, NOTICE.md and templates, and whatever else a test adds."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True)
        for name in ("README.md", "NOTICE.md"):
            shutil.copy(ROOT / name, self.root / name)
        shutil.copytree(ROOT / "templates", self.root / "templates")

    @classmethod
    def committed(cls, root: Path) -> Repo:
        """A checkout with no results: the real configs, the committed audit and subsets, and README.md with
        nothing between the markers."""
        repo = cls(root)
        repo.path("README.md").write_text(without_results(repo.read("README.md")), encoding="utf-8")
        shutil.copytree(ROOT / "configs", repo.root / "configs")
        (repo.root / "results").mkdir()
        for name in ("data_audit.json", "subsets.json"):
            shutil.copy(ROOT / "results" / name, repo.root / "results" / name)
        return repo

    @classmethod
    def with_results(cls, root: Path, lab: Lab, *, figures_: bool = True, train_log: bool = True, analysis: bool = True) -> Repo:
        repo = cls(root)
        shutil.copytree(lab.config_dir, repo.root / "configs")
        shutil.copytree(lab.results, repo.root / "results", ignore=shutil.ignore_patterns("runs", "figures"))
        if figures_:
            shutil.copytree(lab.results / "figures", repo.root / "results" / "figures")
        if not train_log:
            (repo.root / "results" / "train_log.json").unlink(missing_ok=True)
        if not analysis:
            (repo.root / "results" / "error_analysis.csv").unlink(missing_ok=True)
        return repo

    def path(self, name: str) -> Path:
        return self.root / name

    def read(self, name: str) -> str:
        return (self.root / name).read_text(encoding="utf-8")

    def render(self, *targets: str, check: bool = False) -> tuple[int, list[str]]:
        lines: list[str] = []
        code = render.run(targets or render.TARGETS, check=check, root=self.root, out=lines.append)
        return code, lines

    def region(self) -> str:
        text = self.read("README.md")
        return text[text.index(START) + len(START) : text.index(END)]


def without_results(readme: str) -> str:
    """README.md as a checkout with no results has it: nothing between the markers."""
    head, _, rest = readme.partition(START)
    _, _, tail = rest.partition(END)
    return head + START + "\n" + END + tail


def tables_in(text: str) -> list[tuple[list[str], str]]:
    """Every markdown table in `text` (its lines) with the first non-blank line after it."""
    lines = text.splitlines()
    found, i = [], 0
    while i < len(lines):
        if not lines[i].startswith("|"):
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].startswith("|"):
            j += 1
        k = j
        while k < len(lines) and not lines[k].strip():
            k += 1
        found.append((lines[i:j], lines[k] if k < len(lines) else ""))
        i = j
    return found


def table_rows(table: list[str]) -> list[list[str]]:
    return [[c.strip() for c in line.strip().strip("|").split(" | ")] for line in table[2:]]


def by_load_table(text: str) -> tuple[list[str], dict[str, list[str]], str]:
    """The cost-and-latency-by-load table of a document: its header cells, its rows by the Concurrency cell, and the
    first line after it (which must be the notice)."""
    table, after = next(t for t in tables_in(text) if t[0][0].startswith("| Concurrency"))
    header = [c.strip() for c in table[0].strip().strip("|").split(" | ")]
    return header, {row[0]: row for row in table_rows(table)}, after


def front_matter(card: str) -> dict:
    assert card.startswith("---\n")
    return yaml.safe_load(card[4:].split("\n---\n", 1)[0])


# --- the README region ---------------------------------------------------------------------------------------------


def test_only_the_region_between_the_markers_changes(standard, tmp_path):
    lab, _ = standard
    repo = Repo.with_results(tmp_path / "repo", lab)
    original = repo.read("README.md")
    code, _ = repo.render("readme")
    assert code == 0
    updated = repo.read("README.md")
    head, _, rest = original.partition(START)
    _, _, tail = rest.partition(END)
    assert updated.startswith(head + START + "\n") and updated.endswith(END + tail)  # outside the markers: untouched
    assert updated.count(START) == updated.count(END) == 1
    assert len(repo.region()) > 1500 and "### Exact match, paired against the fine-tune" in repo.region()  # not vacuous


def test_rendering_again_gives_the_same_file_and_replaces_stale_content(standard, tmp_path):
    lab, _ = standard
    repo = Repo.with_results(tmp_path / "repo", lab)
    text = repo.read("README.md")
    repo.path("README.md").write_text(text.replace(START + "\n" + END, START + "\nSTALE TABLE\n" + END), encoding="utf-8")
    repo.render("readme")
    first = repo.read("README.md")
    assert "STALE TABLE" not in first
    code, lines = repo.render("readme")
    assert code == 0 and lines == ["current: README.md"] and repo.read("README.md") == first


def test_with_no_results_a_readme_with_an_empty_region_is_byte_identical(tmp_path):
    repo = Repo.committed(tmp_path / "repo")
    before = without_results((ROOT / "README.md").read_text()).encode()
    assert repo.render("readme") == (0, ["current: README.md"])
    assert repo.path("README.md").read_bytes() == before
    assert repo.render("readme", check=True) == (0, ["current: README.md"])


def test_a_stub_comparison_with_no_scored_system_also_leaves_the_readme_alone(tmp_path):
    repo = Repo.committed(tmp_path / "repo")
    compare.run(results_dir=repo.path("results"), processed_dir=tmp_path / "none", config_dir=repo.path("configs"), out=lambda line: None)
    assert json.loads(repo.read("results/comparison.json"))["has_results"] is False
    before = repo.path("README.md").read_bytes()
    assert repo.render("readme")[0] == 0 and repo.path("README.md").read_bytes() == before


def test_results_that_go_away_clear_the_region(tmp_path):
    repo = Repo.committed(tmp_path / "repo")
    text = repo.read("README.md").replace(START + "\n" + END, START + "\nold generated tables\n" + END)
    repo.path("README.md").write_text(text, encoding="utf-8")
    repo.render("readme")
    assert repo.region() == "\n" and repo.read("README.md") == without_results((ROOT / "README.md").read_text())


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("no markers here\n", "exactly one"),
        (f"{START}\n{START}\n{END}\n", "exactly one"),
        (f"{START}\n{END}\n{END}\n", "exactly one"),
        (f"{END}\nx\n{START}\n", "comes before"),
    ],
)
def test_the_markers_must_be_there_once_and_in_order(text, message):
    with pytest.raises(render.RenderError, match=message):
        render.splice_readme(text, "body")


def test_splice_readme_keeps_the_line_endings_of_the_file():
    crlf = f"intro\r\n{START}\r\n{END}\r\nend\r\n"
    out = render.splice_readme(crlf, "a\nb\n")
    assert out == f"intro\r\n{START}\r\n\r\na\r\nb\r\n\r\n{END}\r\nend\r\n"  # blank lines around the generated block
    assert render.splice_readme(f"x\n{START}\nold\n{END}\ny\n", "") == f"x\n{START}\n{END}\ny\n"


# --- --check -------------------------------------------------------------------------------------------------------------


def test_check_reports_stale_files_and_writes_nothing(standard, tmp_path):
    lab, _ = standard
    repo = Repo.with_results(tmp_path / "repo", lab)
    before = repo.path("README.md").read_bytes()
    code, lines = repo.render("readme", "card", "writeup", check=True)
    assert code == 1
    assert "STALE: README.md (differs from what the results and templates produce)" in lines
    assert "STALE: hf/README.md (missing)" in lines and "STALE: docs/writeup.md (missing)" in lines
    assert lines[-1].startswith("run: python scripts/render.py --target all")
    assert repo.path("README.md").read_bytes() == before
    assert not repo.path("hf").exists() and not repo.path("docs").exists()


def test_check_passes_once_everything_is_rendered_and_fails_again_after_an_edit(standard, tmp_path):
    lab, _ = standard
    repo = Repo.with_results(tmp_path / "repo", lab)
    assert repo.render()[0] == 0
    assert repo.render(check=True) == (0, ["current: README.md", "current: hf/README.md", "current: docs/writeup.md"])
    repo.path("docs/writeup.md").write_text(repo.read("docs/writeup.md") + "a hand edit\n", encoding="utf-8")
    code, lines = repo.render(check=True)
    assert code == 1 and "STALE: docs/writeup.md (differs from what the results and templates produce)" in lines
    assert repo.render()[0] == 0 and repo.render(check=True)[0] == 0  # rendering puts it right


def test_check_notices_a_changed_result(standard, tmp_path):
    lab, _ = standard
    repo = Repo.with_results(tmp_path / "repo", lab)
    repo.render()
    doc = json.loads(repo.read("results/comparison.json"))
    doc["systems"][0]["metrics"]["exact_match"]["value"] = 0.5
    repo.path("results/comparison.json").write_text(json.dumps(doc))
    code, lines = repo.render(check=True)
    assert code == 1 and "STALE: README.md (differs from what the results and templates produce)" in lines


def test_main_parses_the_targets_and_the_check_flag(standard, tmp_path):
    lab, _ = standard
    repo = Repo.with_results(tmp_path / "repo", lab)
    quiet = {"root": repo.root, "out": lambda line: None}
    assert render.main(["--target", "card"], **quiet) == 0
    assert repo.path("hf/README.md").exists() and not repo.path("docs").exists()
    assert render.main(["--target", "card", "--check"], **quiet) == 0
    assert render.main(["--target", "writeup", "--check"], **quiet) == 1
    assert render.main(["--target", "all"], **quiet) == 0 and repo.path("docs/writeup.md").exists()
    with pytest.raises(SystemExit):
        render.main(["--target", "everything"], **quiet)


def test_the_committed_generated_files_are_up_to_date():
    """The same check CI runs, on this repository. If it fails: python scripts/render.py --target all"""
    lines: list[str] = []
    code = render.run(render.TARGETS, check=True, root=ROOT, out=lines.append)
    assert code == 0, "\n".join(lines)


# --- tables and the notice -----------------------------------------------------------------------------------------------


def all_outputs(repo: Repo) -> dict[str, str]:
    repo.render()
    return {"README.md": repo.region(), "hf/README.md": repo.read("hf/README.md"), "docs/writeup.md": repo.read("docs/writeup.md")}


def test_every_table_carries_the_notice_and_each_document_lists_the_sources_once(standard, tmp_path):
    lab, doc = standard
    outputs = all_outputs(Repo.with_results(tmp_path / "repo", lab))
    counts = {name: len(tables_in(text)) for name, text in outputs.items()}
    assert counts["README.md"] >= 5 and counts["hf/README.md"] >= 1 and counts["docs/writeup.md"] >= 2  # not vacuous
    urls = {e["url"]: e["retrieved_on"] for e in doc["sources"]["prices"].values()}
    urls[doc["sources"]["gpu_rental"]["url"]] = doc["sources"]["gpu_rental"]["retrieved_on"]
    assert len(urls) >= 2  # not vacuous: price pages and the GPU rental page
    for name, text in outputs.items():
        for table, after in tables_in(text):
            assert after == NOTICE + "*", (name, table[0])  # the notice alone: the sources are not repeated per table
        for url, day in urls.items():
            assert text.count(f"<{url}> (retrieved {day})") == 1, (name, url)
        assert text.count("Prices: <https://") == 1 and text.count("GPU rental: <https://") == 1, name


def test_the_sources_are_listed_where_each_document_puts_its_costs(standard, tmp_path):
    lab, _ = standard
    outputs = all_outputs(Repo.with_results(tmp_path / "repo", lab))
    readme, card, writeup = outputs["README.md"], outputs["hf/README.md"], outputs["docs/writeup.md"]
    # outside the collapsed block in the README, so a reader who never opens it still sees them
    assert readme.index("</details>") < readme.index("*Prices: <") < readme.index("Definitions and caveats")
    assert writeup.index("## Cost and break-even") < writeup.index("*Prices: <") < writeup.index("## Why now")
    assert card.index("## Evaluation") < card.index("*Prices: <") < card.index("## Limitations")


def test_with_no_cost_table_the_writeup_lists_the_sources_under_its_results(tmp_path, monkeypatch):
    """Only the self-hosted rows have results and no benchmark was read: no table of costs, so the GPU rental
    source follows the results table instead."""
    monkeypatch.setattr(config, "git_commit", lambda cwd=None: None)
    monkeypatch.setattr(config, "git_dirty", lambda cwd=None: False)
    lab = Lab(tmp_path / "lab")
    for system in (FT, BASE):
        lab.write_run(system)
    lab.write_audit()
    lab.write_subsets()
    doc = comparison_for(lab)
    assert not doc["sources"]["prices"] and doc["sources"]["gpu_rental"]  # not vacuous
    repo = Repo.with_results(tmp_path / "repo", lab, figures_=False)
    repo.render("writeup")
    writeup = repo.read("docs/writeup.md")
    assert writeup.count("GPU rental: <https://") == 1 and "Prices: <" not in writeup
    assert writeup.index("## Results") < writeup.index("*GPU rental: <") < writeup.index("## Where the gap comes from")


def test_the_templates_never_write_a_table_by_hand():
    for path in (ROOT / "templates").glob("*.j2"):
        assert not [line for line in path.read_text().splitlines() if line.lstrip().startswith("|")], path.name


def test_the_notice_reads_as_the_project_says_it():
    assert render.TABLE_NOTICE == "All API rows ran on free tiers; no money was spent; costs are at paid list prices."


def test_the_api_latency_label_is_the_one_compare_writes():
    assert render.API_LATENCY_LABEL == compare.API_LATENCY_LABEL


def test_the_default_reference_is_the_one_compare_uses():
    assert render.DEFAULT_REFERENCE == compare.REFERENCE


def test_md_table_escapes_pipes_and_checks_its_alignment():
    table = render.md_table(["a|b", "c"], [["x|y", "z"]], "lr")
    assert table == "| a\\|b | c |\n|---|---:|\n| x\\|y | z |\n\n" + NOTICE + "*"
    with pytest.raises(ValueError, match="2 columns"):
        render.md_table(["a", "b"], [], "l")


# --- generated numbers only --------------------------------------------------------------------------------------------------

JINJA = re.compile(r"\{#.*?#\}|\{\{.*?\}\}|\{%.*?%\}", re.DOTALL)
CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
STANDALONE_NUMBER = re.compile(r"(?<![\w.\-/:$])\d[\d,.]*%?(?![\w\-/])")
NUMBER_WORDS = re.compile(r"\b(two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million)\b", re.IGNORECASE)
UNIT = "1,000 calls"  # the unit of the cost figures ("per 1,000 calls"), not a result


def prose_of(template: str) -> str:
    """A template without its Jinja, its code and its allowed unit: what is left is typed prose."""
    return CODE.sub("", JINJA.sub("", template)).replace(UNIT, "")


@pytest.mark.parametrize("name", ["readme_results.md.j2", "model_card.md.j2", "writeup.md.j2"])
def test_templates_type_no_numbers(name):
    prose = prose_of((ROOT / "templates" / name).read_text())
    assert STANDALONE_NUMBER.findall(prose) == [], name
    assert NUMBER_WORDS.findall(prose) == [], name


def test_the_number_check_would_catch_a_typed_number():
    assert STANDALONE_NUMBER.findall(prose_of("score of 91% on 500 items {{ ok }}")) == ["91%", "500"]
    assert NUMBER_WORDS.findall(prose_of("the ten most similar")) == ["ten"]
    assert STANDALONE_NUMBER.findall(prose_of("S500, p95, qwen3.8-27b, gemini-3.5-flash-lite and `0.5` are names; per 1,000 calls is the unit")) == []


def test_the_rendered_numbers_are_the_ones_in_the_comparison(standard, tmp_path):
    lab, doc = standard
    repo = Repo.with_results(tmp_path / "repo", lab)
    repo.render("readme")
    ft = doc["systems"][0]
    low, high = ft["metrics"]["exact_match"]["ci95"]
    assert f"{ft['metrics']['exact_match']['value'] * 100:.1f}% [{low * 100:.1f}, {high * 100:.1f}]" in repo.region()
    # change a number in the comparison and the README follows; a typed number could not
    changed = json.loads(repo.read("results/comparison.json"))
    changed["systems"][0]["metrics"]["exact_match"] = {"value": 0.5, "ci95": [0.4, 0.6], "n": 500}
    repo.path("results/comparison.json").write_text(json.dumps(changed))
    repo.render("readme")
    assert accuracy_rows(repo.region())[FT][2] == "50.0% [40.0, 60.0]"


# --- the wording rule in the rendered output ---------------------------------------------------------------------------------


def accuracy_rows(text: str) -> dict[str, list[str]]:
    table, _ = next(t for t in tables_in(text) if t[0][0].startswith("| System | Items"))
    return {row[0].strip("`"): row for row in table_rows(table)}


def test_beats_appears_only_where_the_interval_excludes_zero(standard, tmp_path):
    lab, doc = standard
    repo = Repo.with_results(tmp_path / "repo", lab)
    repo.render()
    for text in (repo.region(), repo.read("hf/README.md")):
        rows = accuracy_rows(text)
        assert rows[FT][5] == "reference"
        for name in (BASE, *API_SYSTEMS):
            interval = re.search(r"\[([+-]?[\d.]+), ([+-]?[\d.]+)\]", rows[name][3])
            low, high = float(interval.group(1)), float(interval.group(2))
            excludes_zero = low > 0 or high < 0
            assert ("beats" in rows[name][5]) == excludes_zero, rows[name]
            if low > 0:  # the system is ahead of the fine-tune
                assert rows[name][5] == "beats the fine-tune"
            elif high < 0:  # the fine-tune is ahead
                assert rows[name][5] == "the fine-tune beats it"
            else:
                assert rows[name][5] == "no significant difference"


@pytest.fixture(scope="module")
def tie(tmp_path_factory):
    """The fine-tune and groq-gpt-oss-20b separated by 12 items against 10: no significant difference."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "git_commit", lambda cwd=None: None)
        mp.setattr(config, "git_dirty", lambda cwd=None: False)
        lab = Lab(tmp_path_factory.mktemp("tie"))
        ids = list(lab.ids("S500"))
        lab.write_run(FT, wrong=ids[:12] + ids[22:42])
        lab.write_run(GPT_OSS_20B, wrong=ids[12:22] + ids[22:42])
        lab.write_audit()
        lab.write_subsets()
        lab.write_serving()
        comparison_for(lab, 500)
    return lab


def test_a_tie_is_called_no_significant_difference_everywhere(tie, tmp_path):
    repo = Repo.with_results(tmp_path / "repo", tie, figures_=False, train_log=False, analysis=False)
    repo.render()
    for text in (repo.region(), repo.read("hf/README.md")):
        rows = accuracy_rows(text)
        assert rows[GPT_OSS_20B][5] == "no significant difference" and not any("beats" in row[5] for row in rows.values())
    writeup = repo.read("docs/writeup.md")
    assert f"there is no significant difference with `{GPT_OSS_20B}`" in writeup
    sentence = next(line for line in writeup.splitlines() if line.startswith("A system beats another only"))
    assert sentence.count("beats") == 1  # the rule itself; no finding says "beats"
    assert "beats the fine-tune" not in sentence and "the fine-tune beats" not in sentence


def test_the_writeup_summary_follows_the_relations(standard, tmp_path):
    lab, _ = standard
    repo = Repo.with_results(tmp_path / "repo", lab)
    repo.render("writeup")
    text = repo.read("docs/writeup.md")
    # all three readings in one sentence: the fine-tune ahead, a system ahead of it, and one level with it
    assert (
        f"On exact match, the fine-tune beats `{BASE}`, `{GPT_OSS_20B}` and `{GPT_OSS_120B}`; "
        f"`{GEMINI}` beats the fine-tune; there is no significant difference with `{QWEN_27B}`."
    ) in text
    assert "On the S300 items" not in text  # every row is compared on S500, so there is no smaller-subset note


@pytest.fixture(scope="module")
def smaller(tmp_path_factory):
    """The fine-tune and one API row compared on S300, the pre-registered subset no real row uses."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "git_commit", lambda cwd=None: None)
        mp.setattr(config, "git_dirty", lambda cwd=None: False)
        lab = Lab(tmp_path_factory.mktemp("smaller"))
        lab.move_to_subset(GEMINI, "S300")
        lab.write_run(FT)
        lab.write_run(GEMINI)
        lab.write_audit()
        lab.write_subsets()
        doc = comparison_for(lab)
    return lab, doc


def test_a_row_compared_on_a_smaller_subset_says_so_and_what_the_fine_tune_scores_there(smaller, tmp_path):
    lab, doc = smaller
    repo = Repo.with_results(tmp_path / "repo", lab, figures_=False, train_log=False, analysis=False)
    repo.render()
    reference_on_300 = next(s for s in doc["systems"] if s["name"] == GEMINI)["vs_reference"]["reference_exact_match"]
    note = f"On the S300 items, where `{GEMINI}` is compared, the fine-tune scores {render.pct_interval(reference_on_300)}."
    assert note in repo.read("docs/writeup.md")
    assert accuracy_rows(repo.region())[GEMINI][1] == "S300 (300)"  # the Items column names the subset
    assert accuracy_rows(repo.region())[FT][1] == "S500 (500)"


# --- the model card ---------------------------------------------------------------------------------------------------------------


def test_the_model_card_header_parses_and_holds_the_results(standard, tmp_path):
    lab, doc = standard
    repo = Repo.with_results(tmp_path / "repo", lab)
    repo.render("card")
    meta = front_matter(repo.read("hf/README.md"))
    assert meta["license"] == "apache-2.0" and meta["base_model"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert meta["library_name"] == "peft" and meta["language"] == ["en"] and meta["datasets"] == ["AmazonScience/massive"]
    index = meta["model-index"]
    assert [m["name"] for m in index] == [FT]
    full, subset = index[0]["results"]
    ft = doc["systems"][0]
    for entry, source, label in ((full, ft["full_test"]["metrics"], "test split (n=720)"), (subset, ft["metrics"], "test subset S500 (n=500)")):
        assert entry["task"]["type"] == "text-generation"
        assert entry["dataset"] == {"name": f"MASSIVE 1.1 en-US, {label}", "type": "AmazonScience/massive", "config": "en-US", "split": "test"}
        values = {m["type"]: m["value"] for m in entry["metrics"]}
        assert values == {
            "exact_match": round(source["exact_match"]["value"], 4),
            "accuracy": round(source["intent_accuracy"]["value"], 4),
            "f1": round(source["slot_f1"]["value"], 4),
        }
        assert all(isinstance(v, float) and 0 <= v <= 1 for v in values.values())


def test_without_results_the_card_header_still_parses_and_has_no_model_index(tmp_path):
    repo = Repo.committed(tmp_path / "repo")
    repo.render("card")
    meta = front_matter(repo.read("hf/README.md"))
    assert "model-index" not in meta and meta["license"] == "apache-2.0" and meta["library_name"] == "peft"
    assert "Evaluation results are not in yet." in repo.read("hf/README.md")


def test_yaml_scalars_survive_awkward_text():
    for text in ("plain", "has: a colon", 'say "hi"', "it's", "line\nbreak", "# not a comment", "- dash", "yes", "1.0", "ünï"):
        assert yaml.safe_load(f"k: {render.yaml_str(text)}")["k"] == text


def test_the_card_has_the_usage_snippets_the_attribution_and_the_limits(standard, tmp_path):
    lab, _ = standard
    repo = Repo.with_results(tmp_path / "repo", lab)
    repo.render("card")
    card = repo.read("hf/README.md")
    notice = (ROOT / "NOTICE.md").read_text()
    assert "vllm serve Qwen/Qwen3-4B-Instruct-2507" in card and "--enable-lora" in card
    assert "--lora-modules ft-qwen3-4b-lora=<adapter repo id or local path>" in card and "--max-lora-rank 16" in card
    assert "PeftModel.from_pretrained(model," in card and "AutoModelForCausalLM.from_pretrained(base" in card
    for citation in re.findall(r"```\n(@.*?)\n```", notice, flags=re.DOTALL):
        assert citation in card  # the BibTeX is NOTICE.md's, not retyped
    for heading in ("## Use", "### With vLLM", "### With transformers and peft", "## Training details", "## Evaluation", "## Limitations", "## License and attribution"):
        assert heading in card
    assert "SLURP" in card and "Apache-2.0" in card and "CC-BY-4.0" in card
    assert "MTEB" in card and "public since 2022" in card


def test_the_python_snippets_in_the_card_are_valid_python(standard, tmp_path):
    lab, _ = standard
    repo = Repo.with_results(tmp_path / "repo", lab)
    repo.render("card")
    blocks = re.findall(r"```python\n(.*?)```", repo.read("hf/README.md"), flags=re.DOTALL)
    assert len(blocks) == 2
    for block in blocks:
        ast.parse(block)
    assert 'base_url="http://127.0.0.1:8000/v1"' in blocks[0] and "temperature=0," in blocks[0] and "max_tokens=256," in blocks[0]
    assert "Convert the request into a JSON function call with an intent and slots." in blocks[0]


def test_training_details_come_from_the_log_when_there_is_one(standard, tmp_path):
    lab, _ = standard
    repo = Repo.with_results(tmp_path / "repo", lab)
    repo.render("card")
    card = repo.read("hf/README.md")
    assert "Read from the training log (`results/train_log.json`" in card
    assert "revision `0123456789abcdef0123456789abcdef01234567`" in card
    assert "Hardware: Tesla T4, 10.2 minutes." in card
    assert "Final training loss 0.4321; validation loss 0.3333 after epoch 1, 0.2222 after epoch 2." in card
    assert "Precision: fp16." in card and "unsloth 2026.9.1" in card and "`sft_train.jsonl` (72 records" in card
    assert "No training run has finished yet" not in card


def test_without_a_log_the_card_says_these_are_planned_settings(standard, tmp_path):
    lab, _ = standard
    repo = Repo.with_results(tmp_path / "repo", lab, train_log=False)
    repo.render("card")
    card = repo.read("hf/README.md")
    assert "No training run has finished yet. These are the planned settings from `configs/train.yaml`" in card
    assert "revision `cdbee75f17c01a7cc42f958dc650907174af0554`" in card and "rank 16, alpha 32" in card
    assert "Hardware:" not in card and "Final training loss" not in card
    # with the revision unpinned, the card says so rather than showing a value
    train = repo.path("configs/train.yaml")
    train.write_text(re.sub(r"(?m)^(\s*revision: )\S+", r"\g<1>null", train.read_text()))
    repo.render("card")
    assert "revision not pinned yet" in repo.read("hf/README.md")


def test_a_published_adapter_replaces_the_placeholder(standard, tmp_path):
    lab, _ = standard
    repo = Repo.with_results(tmp_path / "repo", lab)
    path = repo.path("configs/systems.yaml")
    systems = yaml.safe_load(path.read_text())
    systems["systems"][FT]["checkpoint"].update({"adapter": "adapters/epoch-2", "epoch": 2})
    path.write_text(yaml.safe_dump(systems, sort_keys=False))
    repo.render("card")
    unpublished = repo.read("hf/README.md")
    # the locked row's checkpoint is a path inside the training output: never shown to readers as a location
    assert "adapters/epoch-2" not in unpublished and "placeholder until the adapter is published" in unpublished
    repo.path("configs/release.yaml").write_text("adapter_repo: me/qwen3-massive-lora\n")
    repo.render("card")
    card = repo.read("hf/README.md")
    assert "--lora-modules ft-qwen3-4b-lora=me/qwen3-massive-lora" in card and 'PeftModel.from_pretrained(model, "me/qwen3-massive-lora")' in card
    assert "placeholder" not in card and "epoch 2 was chosen on the dev split" in card


# --- the write-up ---------------------------------------------------------------------------------------------------------------------


def words(text: str) -> int:
    return len(text.split())  # what wc -w counts: tables, URLs and code included


def test_the_writeup_is_a_draft_of_800_to_1200_words_with_results_and_without(standard, tmp_path):
    lab, _ = standard
    with_results = Repo.with_results(tmp_path / "with", lab)
    with_results.render("writeup")
    without = Repo.committed(tmp_path / "without")
    without.render("writeup")
    assert 800 <= words(with_results.read("docs/writeup.md")) <= 1200
    assert 800 <= words(without.read("docs/writeup.md")) <= 1200


def test_the_writeup_stays_within_1200_words_at_the_scale_of_the_real_benchmark(no_operating_point, tmp_path):
    """Four load levels and no operating point (the real T4), with the error analysis written and while it is still the
    slot's own text (the longer draft). The by-load table and its notice are most of what the write-up gained."""
    lab, _ = no_operating_point
    written = Repo.with_results(tmp_path / "written", lab)
    slot = Repo.with_results(tmp_path / "slot", lab, analysis=False)
    for repo in (written, slot):
        repo.render("writeup")
        assert 800 <= words(repo.read("docs/writeup.md")) <= 1200
    assert "This is the slot for the hand-labelled error analysis" in slot.read("docs/writeup.md")


def test_the_writeup_has_its_sections_in_order(standard, tmp_path):
    lab, _ = standard
    repo = Repo.with_results(tmp_path / "repo", lab)
    repo.render("writeup")
    text = repo.read("docs/writeup.md")
    headings = re.findall(r"^#{1,3} .*$", text, flags=re.MULTILINE)
    assert headings == [
        "# A small fine-tune against API models, on one narrow task",
        "## The question", "## The setup", "## Results", "## Where the gap comes from", "### Error analysis",
        "## Cost and break-even", "## Why now", "## Limits", "## Reproduce", "## Appendix: API latency",
    ]
    assert "![Exact match against cost per 1,000 calls](../results/figures/accuracy_vs_cost.png)" in text
    assert "python scripts/render.py --target all" in text


def test_the_setup_lists_every_system_and_where_it_runs(tmp_path):
    repo = Repo.committed(tmp_path / "repo")  # the real configs, no results
    repo.render("writeup")
    lines = repo.read("docs/writeup.md").splitlines()
    fewshot = "prompt `fewshot_k10_v1` (10 retrieved examples)"
    assert f"The comparison has {len(SYSTEMS)} systems:" in lines
    assert [line for line in lines if line.startswith("- `") and line.endswith((", self-hosted.", ", free tier."))] == [
        "- `ft-qwen3-4b-lora`: Qwen/Qwen3-4B-Instruct-2507 plus the LoRA adapter, prompt `finetuned_v1` (one-line instruction), self-hosted.",
        f"- `base-qwen3-4b-k10`: Qwen/Qwen3-4B-Instruct-2507, {fewshot}, self-hosted.",
        f"- `groq-gpt-oss-20b-k10`: openai/gpt-oss-20b, {fewshot}, Groq, free tier.",
        f"- `groq-gpt-oss-120b-k10`: openai/gpt-oss-120b, {fewshot}, Groq, free tier.",
        f"- `groq-qwen3.8-27b-k10`: qwen/qwen3.8-27b, {fewshot}, Groq, free tier.",
        f"- `gemini-3.5-flash-lite-k10`: gemini-3.5-flash-lite, {fewshot}, Google AI Studio, free tier.",
    ]
    assert "GitHub" not in "\n".join(lines)


def test_every_endpoint_in_the_configs_has_a_display_name_and_no_other_does():
    endpoints = yaml.safe_load((ROOT / "configs" / "systems.yaml").read_text())["endpoints"]
    assert render.ENDPOINT_NAMES == {"local": "a local server", "groq": "Groq", "gemini": "Google AI Studio"}
    assert set(render.ENDPOINT_NAMES) == set(endpoints)  # none missing (it would show the key), none left over


def unfenced_lines(text: str) -> list[tuple[int, str]]:
    """(line number, line) for every line outside a fenced code block."""
    keep, fenced = [], False
    for number, line in enumerate(text.splitlines()):
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced:
            keep.append((number, line))
    return keep


@pytest.mark.parametrize("state", ["results", "no results", "no operating point"])
def test_every_heading_has_a_blank_line_before_and_after_it(standard, no_operating_point, tmp_path, state):
    lab, _ = no_operating_point if state == "no operating point" else standard
    repo = Repo.committed(tmp_path / "repo") if state == "no results" else Repo.with_results(tmp_path / "repo", lab)
    repo.render()
    documents = {"hf/README.md": repo.read("hf/README.md"), "docs/writeup.md": repo.read("docs/writeup.md")}
    if state != "no results":
        documents["README.md"] = repo.read("README.md")
    for name, text in documents.items():
        lines = text.splitlines()
        body_start = lines.index("---", 1) + 1 if name == "hf/README.md" else 0  # past the YAML header
        headings = [n for n, line in unfenced_lines(text) if re.match(r"#{1,6} ", line) and n >= body_start]
        assert len(headings) >= 4, name
        for n in headings:
            assert n == body_start or lines[n - 1] == "", (name, lines[n])
            assert n + 1 == len(lines) or lines[n + 1] == "", (name, lines[n])


def test_the_writeup_gives_the_fine_tunes_slot_f1_and_intent_against_the_range_of_the_others(standard, tmp_path):
    lab, doc = standard
    repo = Repo.with_results(tmp_path / "repo", lab)
    repo.render("writeup")
    section = repo.read("docs/writeup.md").split("## Where the gap comes from\n", 1)[1].split("\n### ", 1)[0]
    by_name = {s["name"]: s["metrics"] for s in doc["systems"]}
    others = [m for name, m in by_name.items() if name != FT]
    assert len(others) == len(SYSTEMS) - 1  # not vacuous: every other system has a result

    def spread(key):
        low, high = min(m[key]["value"] for m in others), max(m[key]["value"] for m in others)
        assert low < high  # the standard set has a real range, so both ends are shown
        return f"{render.pct(low)} to {render.pct(high)}"

    assert (
        f"The fine-tune's slot F1 is {render.pct(by_name[FT]['slot_f1']['value'])} against {spread('slot_f1')} for the "
        f"other systems; its intent accuracy is {render.pct(by_name[FT]['intent_accuracy']['value'])} against "
        f"{spread('intent_accuracy')}."
    ) in section
    assert "\n- " not in section  # one sentence, not a line per system: the README's table has those


def test_the_gap_sentence_names_a_lone_other_system_and_shows_one_figure_when_the_ends_meet():
    def system(name, slot_f1, intent):
        return {"name": name, "metrics": {"slot_f1": {"value": slot_f1}, "intent_accuracy": {"value": intent}}}

    doc = {"reference": FT, "systems": [system(FT, 0.84, 0.91), system(QWEN_27B, 0.794, 0.896)]}
    assert render.gap_sentence(doc).endswith(f"is 84.0% against 79.4% for `{QWEN_27B}`; its intent accuracy is 91.0% against 89.6%.")
    doc["systems"].append(system(GEMINI, 0.794, 0.896))
    assert "against 79.4% for the other systems" in render.gap_sentence(doc)
    doc["systems"][0]["metrics"] = None  # no result for the fine-tune: nothing to say
    assert render.gap_sentence(doc) == ""


def test_the_writeup_gives_the_cost_break_even_and_latency_from_the_results(standard, tmp_path):
    lab, doc = standard
    repo = Repo.with_results(tmp_path / "repo", lab)
    repo.render("writeup")
    text = repo.read("docs/writeup.md")
    assert "608,333 to 304,167" in text and "$0.0139" in text and "$0.600 to $1.20" in text
    # groq-qwen3.8-27b has no cached-input price, so its cost and its break-even are one figure each, not a range
    assert re.search(rf"\| `{QWEN_27B}` \| paid list price \| \$[\d.,]+ with or without caching \| [\d,]+ \|", text)
    assert "rented around the clock for 730 hours at the on-demand price ($0.5 an hour) costs $365.00 a month" in text
    # what one GPU serves and the latency, at the operating point and at a single stream, are rows of the by-load table
    # (they were a sentence before: 26,280,000 calls a month; p95 1.20 s under load and 0.60 s for a single stream)
    _, rows, _ = by_load_table(text)
    assert rows["8 (operating point)"] == ["8 (operating point)", "1.20", "$0.0139 ($0.0069)", "26,280,000"]
    assert rows["1"] == ["1", "0.60", "$0.0694 ($0.0347)", "5,256,000"]
    assert "Latency of the API systems, observed on free tiers from India; not representative of paid tiers:" in text


def test_the_writeup_uses_only_the_dates_in_sources_yaml(standard, tmp_path):
    lab, _ = standard
    sources_text = (ROOT / "configs" / "sources.yaml").read_text()
    sources = yaml.safe_load(sources_text)
    allowed = set(re.findall(r"\d{4}-\d{2}-\d{2}", sources_text))  # every date written anywhere in sources.yaml
    assert {e["date"] for e in sources["openai_deprecations"]["events"]} <= allowed
    for repo in (Repo.with_results(tmp_path / "with", lab), Repo.committed(tmp_path / "without")):
        repo.render("writeup")
        text = repo.read("docs/writeup.md")
        assert set(re.findall(r"\d{4}-\d{2}-\d{2}", text)) <= allowed
        for event in sources["openai_deprecations"]["events"]:
            assert f"{'On' if event['relation'] == 'on' else 'From'} {event['date']}: " in text
            assert event["what"] in text


def test_no_output_carries_a_date_or_time_of_its_own(standard, tmp_path):
    lab, _ = standard
    outputs = all_outputs(Repo.with_results(tmp_path / "repo", lab))
    for name, text in outputs.items():
        assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}", text), name  # no timestamps: rendering twice is byte-identical


def test_the_error_analysis_slot(standard, tmp_path):
    lab, _ = standard
    repo = Repo.with_results(tmp_path / "repo", lab)
    repo.render("writeup")
    assert "4 errors are labelled by hand in `results/error_analysis.csv`, by category: wrong intent (2), label noise (1), span boundary (1)." in repo.read("docs/writeup.md")
    # a hand-written reading is included verbatim, and survives a re-render
    (repo.root / "docs" / "error_analysis.md").write_text("Most errors are a wrong intent.\n\nSecond paragraph.\n", encoding="utf-8")
    repo.render("writeup")
    text = repo.read("docs/writeup.md")
    assert "### Error analysis\n\nMost errors are a wrong intent.\n\nSecond paragraph.\n\n## Cost" in text
    assert "errors are labelled by hand" not in text
    assert repo.render("writeup", check=True)[0] == 0
    # with neither file there is the slot
    bare = Repo.with_results(tmp_path / "bare", lab, analysis=False)
    bare.render("writeup")
    assert "This is the slot for the hand-labelled error analysis" in bare.read("docs/writeup.md")


def test_the_error_analysis_counts_categories_and_tolerates_blanks(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "error_analysis.csv").write_text("id,category\n1,b\n2,a\n3,\n4,b\n5,\n")
    summary, prose = render.read_error_analysis(tmp_path / "results", tmp_path / "docs")
    assert prose is None and summary["n"] == 5
    assert summary["categories"] == [("(blank)", 2), ("b", 2), ("a", 1)]  # by count, then by name
    (tmp_path / "results" / "error_analysis.csv").write_text("id,note\n1,x\n2,y\n")
    summary, _ = render.read_error_analysis(tmp_path / "results", tmp_path / "docs")
    assert summary == {"path": "results/error_analysis.csv", "n": 2, "categories": None}  # no category column: only the count


@pytest.mark.parametrize("state", ["operating point", "no operating point"])
def test_figures_are_linked_only_when_they_exist(standard, no_operating_point, tmp_path, state):
    lab, _ = standard if state == "operating point" else no_operating_point
    with_figures = Repo.with_results(tmp_path / "with", lab)
    with_figures.render()
    assert "](results/figures/accuracy_vs_cost.png)" in with_figures.region() and "](results/figures/latency.png)" in with_figures.region()
    assert "](../results/figures/accuracy_vs_cost.png)" in with_figures.read("docs/writeup.md")
    without = Repo.with_results(tmp_path / "without", lab, figures_=False)
    without.render()
    assert "![" not in without.region() and "![" not in without.read("docs/writeup.md")


# --- the table of cost and latency by load ---------------------------------------------------------------------------------------

NO_OPERATING_POINT_NOTE = "No level met the rule for an operating point (p95 latency at or under 1 s), so cost is reported at every measured level instead."
FULL_HEADERS = [
    "Concurrency", "Requests/s", "p50", "p95", "Cost per 1,000 calls, on-demand", "Cost per 1,000 calls, spot",
    "Calls one GPU serves a month", "APIs whose break-even range one GPU can serve",
]
COMPACT_HEADERS = ["Concurrency", "p95 (s)", "On-demand (spot) per 1,000 calls", "Calls one GPU serves a month"]
ALL_APIS = ", ".join(f"`{name}`" for name in API_SYSTEMS)  # in the order of configs/systems.yaml
#: The levels of NO_OPERATING_POINT_SERVING worked out by hand at $0.50 and $0.25 an hour: requests/s, p50, p95, cost per
#: 1,000 calls at each price (0.5 / 3600 / rate * 1000), and the calls one GPU serves a month (rate * 3,600 * 730).
BY_LOAD_ROWS = {
    "1": ["1", "1.00", "1.26 s", "2.27 s", "$0.139", "$0.0694", "2,628,000"],
    "8": ["8", "5.00", "1.47 s", "2.57 s", "$0.0278", "$0.0139", "13,140,000"],
    "32": ["32", "15.00", "2.16 s", "3.69 s", "$0.0093", "$0.0046", "39,420,000"],
    "64": ["64", "20.00", "2.86 s", "4.67 s", "$0.0069", "$0.0035", "52,560,000"],
}


def test_the_by_load_table_is_in_the_readme_the_card_and_the_writeup_with_every_level(no_operating_point, tmp_path):
    lab, doc = no_operating_point
    repo = Repo.with_results(tmp_path / "repo", lab)
    repo.render()
    readme, card, writeup = repo.region(), repo.read("hf/README.md"), repo.read("docs/writeup.md")
    for text in (readme, card):  # the full table: throughput, p50 and p95, both prices, what one GPU serves, which APIs it beats
        header, rows, after = by_load_table(text)
        assert header == FULL_HEADERS and list(rows) == list(BY_LOAD_ROWS)
        for load, expected in BY_LOAD_ROWS.items():
            assert rows[load] == [*expected, ALL_APIS], load
        assert after.startswith(NOTICE)  # the table is built by md_table, so the notice follows it
    header, rows, after = by_load_table(writeup)  # the compact one, for a write-up of at most 1,200 words
    assert header == COMPACT_HEADERS and list(rows) == list(BY_LOAD_ROWS) and after.startswith(NOTICE)
    for load, row in BY_LOAD_ROWS.items():
        _, _, _, p95, on_demand, spot, calls = row
        assert rows[load] == [load, p95.removesuffix(" s"), f"{on_demand} ({spot})", calls], load
    block = next(s for s in doc["systems"] if s["name"] == FT)["cost_by_load"]  # the cells come from the comparison
    assert [level["concurrency"] for level in block["levels"]] == [1, 8, 32, 64]


def test_the_note_says_plainly_that_no_level_met_the_rule_and_nothing_is_marked_as_the_headline(no_operating_point, standard, tmp_path):
    lab, doc = no_operating_point
    assert doc["systems"][0]["cost_by_load"]["note"] == NO_OPERATING_POINT_NOTE  # the sentence is the comparison's, not the template's
    repo = Repo.with_results(tmp_path / "repo", lab)
    repo.render()
    for name, text in (("README.md", repo.region()), ("hf/README.md", repo.read("hf/README.md")), ("docs/writeup.md", repo.read("docs/writeup.md"))):
        assert NO_OPERATING_POINT_NOTE in text, name
        assert "(operating point)" not in text and "The operating point" not in text, name  # no level is picked
    # the cost table has no self-hosted row either: nothing is priced at a level the rule did not choose
    assert f"| `{FT}` | GPU rental" not in repo.region()
    # with an operating point, the same documents mark it and make no claim that the rule failed
    other = Repo.with_results(tmp_path / "other", standard[0])
    other.render()
    for name, text in (("README.md", other.region()), ("hf/README.md", other.read("hf/README.md")), ("docs/writeup.md", other.read("docs/writeup.md"))):
        assert "No level met" not in text and "The operating point" in text, name
        assert list(by_load_table(text)[1]) == ["1", "8 (operating point)"], name  # only that row carries the mark


def test_the_latency_table_gives_way_to_the_by_load_table_only_when_there_is_no_operating_point(standard, no_operating_point, tmp_path):
    with_point = Repo.with_results(tmp_path / "with", standard[0])
    with_point.render("readme")
    assert "### Self-hosted latency" in with_point.region() and "### Self-hosted cost and latency by load" in with_point.region()
    assert "p95 at the operating point" in with_point.region()  # unchanged when a level met the rule
    without = Repo.with_results(tmp_path / "without", no_operating_point[0])
    without.render("readme")
    assert "### Self-hosted latency" not in without.region() and "p95 at the operating point" not in without.region()
    assert "### Self-hosted cost and latency by load" in without.region()
    # the latency figure draws every load, not a single stream and the operating point, so it stays: under the by-load table
    assert without.region().count("](results/figures/latency.png)") == 1
    assert "it has its own table below, at every measured load" in without.region()
    assert "kept busy at the operating point" in with_point.region() and "own table below" not in with_point.region()


@pytest.mark.parametrize("state", ["operating point", "no operating point"])
def test_the_figure_captions_say_what_the_figures_now_show(standard, no_operating_point, tmp_path, state):
    """The alt text of the two figures describes the loads they draw, not a single operating point as the headline."""
    lab, _ = standard if state == "operating point" else no_operating_point
    repo = Repo.with_results(tmp_path / "repo", lab)
    repo.render("readme")
    region = repo.region()
    captions = {path: alt for alt, path in re.findall(r"!\[(.*?)\]\((.*?)\)", region)}
    assert set(captions) == {"results/figures/accuracy_vs_cost.png", "results/figures/latency.png"}
    accuracy, latency = captions["results/figures/accuracy_vs_cost.png"], captions["results/figures/latency.png"]
    assert "logarithmic cost axis" in accuracy and "each API as a range from its cached-prefix cost to its no-caching cost" in accuracy
    assert "the fine-tune as one marker for each measured load" in accuracy
    assert "single stream" not in latency and "operating point" not in accuracy
    assert "every measured load" in latency and "p50 and p95 for each number of concurrent requests" in latency
    assert "the operating point marked only when a level met the rule" in latency
    # the latency figure sits right under the by-load table and its notice, before the next section
    heading = region.index("### Self-hosted cost and latency by load")
    figure = region.index("](results/figures/latency.png)")
    between = region[heading:figure]
    assert "\n### " not in between and "| Concurrency " in between and NOTICE in between
    assert region.count("latency.png") == 1


def test_the_note_is_shown_once_and_the_older_no_operating_point_warning_is_not_listed_beside_it(no_operating_point, tmp_path):
    lab, doc = no_operating_point
    ft = next(s for s in doc["systems"] if s["name"] == FT)
    assert ft["cost_by_load"]["note"] == NO_OPERATING_POINT_NOTE
    older = f"results/serving/t4.json, {FT}: no operating point (no concurrency level had p95 latency <= 1 s with no failed request)"
    assert doc["warnings"] == [older]  # not vacuous: the comparison carries it, and it is the only warning
    repo = Repo.with_results(tmp_path / "repo", lab)
    repo.render()
    for name, text in (("README.md", repo.region()), ("hf/README.md", repo.read("hf/README.md"))):
        assert text.count(NO_OPERATING_POINT_NOTE) == 1, name
        assert "no operating point" not in text and "no concurrency level had" not in text, name
        assert "Warnings from the comparison" not in text, name  # it was the only warning, so the section goes with it
    assert json.loads(repo.read("results/comparison.json"))["warnings"] == [older]  # only the rendering leaves it out


def test_the_declared_form_of_the_warning_is_left_out_too(tmp_path, monkeypatch):
    """A benchmark file with no operating point and no reason for it gets "no operating point declared"."""
    monkeypatch.setattr(config, "git_commit", lambda cwd=None: None)
    monkeypatch.setattr(config, "git_dirty", lambda cwd=None: False)
    lab = Lab(tmp_path / "lab")
    lab.write_run(FT)
    lab.write_serving(doc={k: v for k, v in NO_OPERATING_POINT_SERVING.items() if k not in ("operating_point", "operating_point_note")})
    lab.write_audit()
    lab.write_subsets()
    doc = comparison_for(lab)
    assert doc["warnings"] == [f"results/serving/t4.json, {FT}: no operating point declared"]
    repo = Repo.with_results(tmp_path / "repo", lab, figures_=False, train_log=False, analysis=False)
    repo.render()
    for text in (repo.region(), repo.read("hf/README.md")):
        assert "no operating point" not in text and NO_OPERATING_POINT_NOTE in text  # the levels all miss the limit, so the note says so


def test_other_warnings_are_listed_as_before_beside_the_note(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "git_commit", lambda cwd=None: None)
    monkeypatch.setattr(config, "git_dirty", lambda cwd=None: False)
    lab = Lab(tmp_path / "lab")
    lab.write_run(FT)
    lab.write_run(GPT_OSS_20B, models=lambda position: "model-a" if position <= 100 else "model-b")
    lab.write_serving(doc=NO_OPERATING_POINT_SERVING)
    lab.write_audit()
    lab.write_subsets()
    doc = comparison_for(lab)
    other = next(w for w in doc["warnings"] if "model name" in w)
    older = next(w for w in doc["warnings"] if "no operating point" in w)
    repo = Repo.with_results(tmp_path / "repo", lab, figures_=False, train_log=False, analysis=False)
    repo.render()
    for text in (repo.region(), repo.read("hf/README.md")):
        assert "Warnings from the comparison" in text and f"- {other}" in text
        assert older not in text and text.count(NO_OPERATING_POINT_NOTE) == 1


def test_a_warning_about_an_operating_point_that_is_not_a_measured_load_is_still_listed(tmp_path, monkeypatch):
    """The note says no operating point was chosen; this warning says the file named one that cannot be used."""
    monkeypatch.setattr(config, "git_commit", lambda cwd=None: None)
    monkeypatch.setattr(config, "git_dirty", lambda cwd=None: False)
    lab = Lab(tmp_path / "lab")
    lab.write_run(FT)
    lab.write_serving(doc={**NO_OPERATING_POINT_SERVING, "operating_point": {"concurrency": 99}})
    lab.write_audit()
    lab.write_subsets()
    doc = comparison_for(lab)
    warning = next(w for w in doc["warnings"] if "is not one of the measured concurrency levels" in w)
    repo = Repo.with_results(tmp_path / "repo", lab, figures_=False, train_log=False, analysis=False)
    repo.render()
    for text in (repo.region(), repo.read("hf/README.md")):
        assert f"- {warning}" in text and NO_OPERATING_POINT_NOTE in text


def test_shown_warnings_drops_only_the_fine_tunes_own_no_operating_point_warning():
    path = "results/serving/T4.json"
    ours = [f"{path}, {FT}: no operating point (no level qualified)", f"{path}, {FT}: no operating point declared"]
    others = [
        f"{path}, {FT}: the operating point (99) is not one of the measured concurrency levels [1, 8]",
        f"{path}, {FT}: concurrency 8 lacks latency_s.p50, latency_s.p95 or requests_per_s (seconds)",
        f"{path}, {BASE}: no operating point declared",  # another row's: its note is not shown, so neither is it dropped
        f"results/serving/other.json, {FT}: no operating point declared",  # another file's
        f"{FT}: only 100 of the 500 S500 items have an answer",
    ]
    block = {"operating_point": None, "note": "a note", "levels": [{"concurrency": 1}]}
    doc = {
        "reference": FT, "warnings": [*ours, *others], "self_hosted": {"benchmark": {"path": path}},
        "systems": [{"name": FT, "cost_by_load": block}, {"name": BASE, "cost_by_load": None}],
    }
    assert render.shown_warnings(doc) == others
    assert doc["warnings"] == [*ours, *others]  # the comparison's own list is not changed
    doc["systems"][0]["cost_by_load"] = {**block, "operating_point": 1, "note": None}  # a level met the rule: no note
    assert render.shown_warnings(doc) == [*ours, *others]
    doc["systems"][0]["cost_by_load"] = None  # no benchmark levels at all
    assert render.shown_warnings(doc) == [*ours, *others]


@pytest.mark.skipif(not REAL_T4.exists(), reason="results/serving/T4.json is not in this checkout")
def test_the_real_t4_benchmark_shows_its_note_once_and_without_its_older_warning(tmp_path, monkeypatch):
    raw = json.loads(REAL_T4.read_text(encoding="utf-8"))
    monkeypatch.setattr(config, "git_commit", lambda cwd=None: None)
    monkeypatch.setattr(config, "git_dirty", lambda cwd=None: False)
    lab = Lab(tmp_path / "lab").populate()
    (lab.results / "serving" / "t4.json").unlink()  # the stand-in benchmark; the real file takes its place
    lab.write_serving(name="T4", doc=raw)
    doc = comparison_for(lab)
    older = f"results/serving/T4.json, {FT}: no operating point ({raw['operating_point_note']})"
    assert older in doc["warnings"]
    repo = Repo.with_results(tmp_path / "repo", lab, figures_=False, train_log=False, analysis=False)
    repo.render()
    for text in (repo.region(), repo.read("hf/README.md")):
        assert text.count(NO_OPERATING_POINT_NOTE) == 1
        assert older not in text and raw["operating_point_note"] not in text


def test_the_cost_table_keeps_its_self_hosted_row_when_a_level_met_the_rule(standard, tmp_path):
    repo = Repo.with_results(tmp_path / "repo", standard[0])
    repo.render()
    for text in (repo.region(), repo.read("docs/writeup.md")):
        table = next(t for t in tables_in(text) if t[0][0].startswith("| System | Priced as"))[0]
        assert table_rows(table)[0][:2] == [f"`{FT}`", "GPU rental at the on-demand price, kept busy"]


def test_the_per_level_column_lists_the_apis_whose_whole_break_even_range_one_gpu_can_serve(straddling, tmp_path):
    lab, _ = straddling
    repo = Repo.with_results(tmp_path / "repo", lab, figures_=False, train_log=False, analysis=False)
    repo.render()
    served = {load: row[7] for load, row in by_load_table(repo.region())[1].items()}
    # the levels serve 131,400 / 262,800 / 525,600 / 788,400 / 1,051,200 calls a month (tests/results_fixtures.py): an API
    # counts only when both ends of its break-even range fit, so gemini at 262,800 (260,714 to 365,000) does not
    assert served == {
        "1": "none",
        "2": "`groq-qwen3.8-27b-k10`",
        "4": "`groq-qwen3.8-27b-k10`, `gemini-3.5-flash-lite-k10`",
        "8": "`groq-gpt-oss-20b-k10`, `groq-qwen3.8-27b-k10`, `gemini-3.5-flash-lite-k10`",
        "16": ALL_APIS,
    }


def test_without_a_benchmark_there_is_no_by_load_table_or_section(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "git_commit", lambda cwd=None: None)
    monkeypatch.setattr(config, "git_dirty", lambda cwd=None: False)
    lab = Lab(tmp_path / "lab")
    lab.write_run(FT)
    lab.write_run(GPT_OSS_20B)
    lab.write_audit()
    lab.write_subsets()
    comparison_for(lab)
    repo = Repo.with_results(tmp_path / "repo", lab, figures_=False, train_log=False, analysis=False)
    repo.render()
    for text in (repo.region(), repo.read("hf/README.md"), repo.read("docs/writeup.md")):
        assert not [t for t in tables_in(text) if t[0][0].startswith("| Concurrency")]
    assert "### Self-hosted cost and latency by load" not in repo.region() and "### Serving cost and latency" not in repo.read("hf/README.md")
    assert "It does not depend on the load" not in repo.region() and "own table below" not in repo.region()


def test_with_no_priced_api_the_by_load_table_has_no_feasibility_column_but_still_appears(tmp_path, monkeypatch):
    """A benchmark and the fine-tune's run, and no API row to compare the break-even with."""
    monkeypatch.setattr(config, "git_commit", lambda cwd=None: None)
    monkeypatch.setattr(config, "git_dirty", lambda cwd=None: False)
    lab = Lab(tmp_path / "lab")
    lab.write_run(FT)
    lab.write_serving(doc=NO_OPERATING_POINT_SERVING)
    lab.write_audit()
    lab.write_subsets()
    comparison_for(lab)
    repo = Repo.with_results(tmp_path / "repo", lab, figures_=False, train_log=False, analysis=False)
    repo.render()
    for text in (repo.region(), repo.read("hf/README.md")):
        header, rows, after = by_load_table(text)
        assert header == FULL_HEADERS[:-1] and after.startswith(NOTICE)
        assert all(len(row) == len(header) for row in rows.values())  # every row has as many cells as the header
        assert rows["8"] == BY_LOAD_ROWS["8"]
    writeup = repo.read("docs/writeup.md")
    assert by_load_table(writeup)[0] == COMPACT_HEADERS and NO_OPERATING_POINT_NOTE in writeup
    assert "API costs are computed from tokens" not in writeup  # the section is not the no-results draft
    assert "### Cost and break-even" not in repo.region() and "### Self-hosted cost and latency by load" in repo.region()  # no API cost table, so no such section


def test_the_model_card_has_a_serving_section_only_when_results_exist(no_operating_point, tmp_path):
    repo = Repo.with_results(tmp_path / "with", no_operating_point[0])
    repo.render("card")
    card = repo.read("hf/README.md")
    assert "\n### Serving cost and latency\n" in card and card.index("### Serving cost and latency") < card.index("## Limitations")
    assert card.index("### Serving cost and latency") > card.index("## Evaluation")
    without = Repo.committed(tmp_path / "without")
    without.render("card")
    assert "Serving cost and latency" not in without.read("hf/README.md")


def test_with_no_results_the_generated_files_have_no_results_in_them(tmp_path):
    """The by-load section exists only in the results branches of the templates. (That the committed files match the
    committed results is CI's `render.py --check`.)"""
    repo = Repo.committed(tmp_path / "repo")
    for optional in ("results/train_log.json", "results/error_analysis.csv", "docs/error_analysis.md"):  # inputs the real files used
        if (ROOT / optional).exists():
            repo.path(optional).parent.mkdir(exist_ok=True)
            shutil.copy(ROOT / optional, repo.path(optional))
    code, _ = repo.render()
    assert code == 0
    assert repo.read("README.md") == without_results((ROOT / "README.md").read_text())
    assert "Concurrency" not in repo.read("docs/writeup.md") and "by load" not in repo.read("hf/README.md")


@pytest.mark.parametrize("state", ["results", "no operating point", "no results"])
def test_no_generated_line_ends_in_whitespace(standard, no_operating_point, tmp_path, state):
    """Jinja's block tags swallow or leave newlines and spaces at the end of a line; a stray one shows up here."""
    lab, _ = no_operating_point if state == "no operating point" else standard
    repo = Repo.committed(tmp_path / "repo") if state == "no results" else Repo.with_results(tmp_path / "repo", lab)
    repo.render()
    for name in ("README.md", "hf/README.md", "docs/writeup.md"):
        lines = repo.read(name).split("\n")
        assert [n for n, line in enumerate(lines, start=1) if line != line.rstrip()] == [], name


def test_the_compact_tables_drop_a_column_only_where_every_row_says_the_same(standard, no_operating_point, smaller, tmp_path):
    # exact match: "Items" is the same on every row of the standard results, so the write-up leaves it to its setup paragraph
    plain = Repo.with_results(tmp_path / "plain", standard[0])
    plain.render()
    header = next(t for t in tables_in(plain.read("docs/writeup.md")) if t[0][0].startswith("| System"))[0][0]
    assert "Items" not in header and "McNemar" not in header and "Full test split" not in header
    assert "| System | Items | Exact match |" in plain.region()  # the README keeps it
    # ... but it stays when a row differs (one is on the S300 subset)
    lab, _ = smaller
    mixed = Repo.with_results(tmp_path / "mixed", lab, figures_=False, train_log=False, analysis=False)
    mixed.render()
    table = next(t for t in tables_in(mixed.read("docs/writeup.md")) if t[0][0].startswith("| System"))[0]
    assert "Items" in table[0] and any("S300 (300)" in row for row in table[2:])
    # cost: "Priced as" is dropped when every priced row is an API row (the notice says list prices), and kept beside a rental row
    no_point = Repo.with_results(tmp_path / "no-point", no_operating_point[0])
    no_point.render()
    cost_header = next(t for t in tables_in(no_point.read("docs/writeup.md")) if "Break-even" in t[0][0])[0][0]
    assert "Priced as" not in cost_header and "| System | Cost per 1,000 calls" in cost_header
    assert "| System | Priced as |" in no_point.region()  # the README keeps it


# --- partial and warned results ---------------------------------------------------------------------------------------------------


def test_systems_without_results_are_listed_and_the_rest_still_render(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "git_commit", lambda cwd=None: None)
    monkeypatch.setattr(config, "git_dirty", lambda cwd=None: False)
    lab = Lab(tmp_path / "lab")
    lab.write_run(FT)
    lab.write_run(GPT_OSS_20B)
    lab.write_audit()
    lab.write_subsets()
    comparison_for(lab)
    repo = Repo.with_results(tmp_path / "repo", lab, figures_=False, train_log=False, analysis=False)
    repo.render("readme", "writeup")
    region = repo.region()
    rows = accuracy_rows(region)
    assert all(rows[name][2] == "no results yet" for name in (BASE, GPT_OSS_120B, QWEN_27B, GEMINI))
    assert rows[GPT_OSS_20B][2].endswith("]")
    assert f"Not compared yet, because they have no results: `{BASE}`, `{GPT_OSS_120B}`, `{QWEN_27B}`, `{GEMINI}`." in region
    assert "### Cost and break-even" in region and f"| `{GPT_OSS_20B}` | paid list price |" in region  # what can be priced still is
    assert f"| `{FT}` |" not in region.split("### Cost and break-even")[1].split("###")[0]  # no benchmark: no self-hosted cost row


def test_warnings_from_the_comparison_are_shown(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "git_commit", lambda cwd=None: None)
    monkeypatch.setattr(config, "git_dirty", lambda cwd=None: False)
    lab = Lab(tmp_path / "lab")
    lab.write_run(FT)
    lab.write_run(GPT_OSS_20B, models=lambda position: "model-a" if position <= 100 else "model-b")
    lab.write_audit()
    lab.write_subsets()
    doc = comparison_for(lab)
    warning = next(w for w in doc["warnings"] if "model name" in w)
    repo = Repo.with_results(tmp_path / "repo", lab, figures_=False, train_log=False, analysis=False)
    repo.render()
    assert "#### Warnings from the comparison" in repo.region() and f"- {warning}" in repo.region()
    assert f"- {warning}" in repo.read("hf/README.md")


def test_a_partial_system_says_how_many_items_it_has(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "git_commit", lambda cwd=None: None)
    monkeypatch.setattr(config, "git_dirty", lambda cwd=None: False)
    lab = Lab(tmp_path / "lab")
    ids = lab.ids("S500")
    lab.write_run(FT)
    lab.write_run(GPT_OSS_20B, ids=ids[:200])
    lab.write_audit()
    lab.write_subsets()
    comparison_for(lab)
    repo = Repo.with_results(tmp_path / "repo", lab, figures_=False, train_log=False, analysis=False)
    repo.render("readme")
    assert accuracy_rows(repo.region())[GPT_OSS_20B][1] == "S500 (200 of 500)"


# --- errors -------------------------------------------------------------------------------------------------------------------------


def test_a_missing_template_is_an_error(tmp_path):
    repo = Repo.committed(tmp_path / "repo")
    repo.path("templates/writeup.md.j2").unlink()
    code, lines = repo.render("writeup")
    assert code == 2 and "writeup.md.j2" in lines[-1] and "not found" in lines[-1]


def test_a_template_that_asks_for_something_missing_fails_loudly(tmp_path):
    repo = Repo.committed(tmp_path / "repo")
    repo.path("templates/writeup.md.j2").write_text("# {{ no_such_value }}\n")
    code, lines = repo.render("writeup")
    assert code == 2 and "needs something the context does not have" in lines[-1] and "no_such_value" in lines[-1]
    assert not repo.path("docs").exists()


def test_a_comparison_from_another_version_is_refused(tmp_path):
    repo = Repo.committed(tmp_path / "repo")
    repo.path("results/comparison.json").write_text(json.dumps({"schema_version": 99, "has_results": True}))
    code, lines = repo.render("readme")
    assert code == 2 and "wrong schema_version" in lines[-1] and "scripts/compare.py" in lines[-1]
    repo.path("results/comparison.json").write_text("{broken")
    assert repo.render("readme")[0] == 2


def test_the_card_and_the_writeup_need_the_data_audit_but_the_readme_does_not(tmp_path):
    repo = Repo.committed(tmp_path / "repo")
    repo.path("results/data_audit.json").unlink()
    code, lines = repo.render("card")
    assert code == 2 and "results/data_audit.json not found" in lines[-1]
    assert repo.render("readme")[0] == 0  # no results, nothing to say, and no audit needed to say it


def test_the_card_needs_the_citations_in_notice_md(tmp_path):
    repo = Repo.committed(tmp_path / "repo")
    repo.path("NOTICE.md").write_text("# Notice\n\nno citations here\n")
    code, lines = repo.render("card")
    assert code == 2 and "MASSIVE and SLURP BibTeX" in lines[-1]


def test_a_readme_without_markers_is_an_error_and_is_not_touched(tmp_path):
    repo = Repo.committed(tmp_path / "repo")
    repo.path("README.md").write_text("# no markers\n")
    code, lines = repo.render("readme")
    assert code == 2 and "exactly one" in lines[-1] and repo.read("README.md") == "# no markers\n"


def test_a_reference_that_is_not_a_system_is_an_error(standard, tmp_path):
    lab, _ = standard
    repo = Repo.with_results(tmp_path / "repo", lab)
    doc = json.loads(repo.read("results/comparison.json"))
    doc["reference"] = "nope"
    repo.path("results/comparison.json").write_text(json.dumps(doc))
    code, lines = repo.render("readme")
    assert code == 2 and "'nope'" in lines[-1]
