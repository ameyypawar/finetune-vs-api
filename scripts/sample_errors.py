"""Draw the errors to be labelled by hand and write them to results/error_analysis.csv.

    python scripts/sample_errors.py [--per-stratum N] [--seed N] [--out PATH] [--force]

The sample is drawn once, with a fixed seed, before anything is labelled. It has two strata on the
headline test subset (S500), each capped at N items (default 50):

    gap   the items the fine-tune gets right and the strongest API row gets wrong, the API row
          with the highest exact match on the subset. All of them, or a seeded N when there are
          more: they say what the closest API loses that the fine-tune keeps.
    ft    the fine-tune's errors, all of them or a seeded N: what it still gets wrong.

Each row has the request, the gold call, the fine-tune's answer and the API row's answer, and
for each answer what differs from the gold, worked out mechanically: the intent, a slot whose
value or type differs, a missing or an extra slot. `category` and `note` are left blank for the
person labelling: the category says why the wrong answer in the row is wrong (the API's in the
gap stratum, the fine-tune's in the ft stratum). docs/method.md lists the categories, and the
write-up counts them.

The file holds hand labels, so it is never overwritten: --force replaces it.

Exit status: 0 written; 1 the file exists, nothing written; 2 a precondition failed.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare
from finetune_vs_api import config, metrics, subsets

EXIT_EXISTS = 1
EXIT_ERROR = 2
PER_STRATUM = 50
SEED = subsets.SUBSET_SEED
COLUMNS = [
    "id", "stratum", "scenario", "request", "gold",
    "ft_answer", "ft_differs", "api_system", "api_answer", "api_differs",
    "category", "note",
]  # fmt: skip


class SampleError(ValueError):
    """A precondition failed: no complete fine-tune run, or no complete API run to compare it with."""


def call_text(call: Mapping[str, Any]) -> str:
    """A call on one line: `alarm_set | time=nine am; date=friday`."""
    slots = "; ".join(f"{slot['type']}={slot['value']}" for slot in call.get("slots") or [])
    return f"{call.get('intent')} | {slots}" if slots else str(call.get("intent"))


def answer_text(run: compare.Run, item_id: str) -> str:
    item = run.items[item_id]
    if item.pred is not None:
        return call_text(item.pred)
    return "(no answer: the call failed)" if run.rows[item_id].get("error") else "(invalid output)"


def differences(pred: Mapping[str, Any] | None, gold: Mapping[str, Any]) -> str:
    """What an answer gets wrong, slot values compared as the metrics compare them.

    A missing and an extra slot of the same type are one slot whose value differs, often its span;
    of the same value, one slot whose type differs. "none" when the answer is an exact match.
    """
    if pred is None:
        return "no valid answer"
    parts = []
    if pred.get("intent") != gold["intent"]:
        parts.append(f"intent {pred.get('intent')} (gold {gold['intent']})")
    predicted, wanted = metrics.slot_multiset(pred.get("slots")), metrics.slot_multiset(gold["slots"])
    missing, extra = sorted((wanted - predicted).elements()), sorted((predicted - wanted).elements())
    for same, label in ((0, "value"), (1, "type")):
        for slot in list(missing):
            other = next((x for x in extra if x[same] == slot[same]), None)
            if other is None:
                continue
            missing.remove(slot)
            extra.remove(other)
            if label == "value":
                parts.append(f"{slot[0]} '{other[1]}' (gold '{slot[1]}')")
            else:
                parts.append(f"'{slot[1]}' as {other[0]} (gold {slot[0]})")
    parts += [f"missing {kind}={value}" for kind, value in missing]
    parts += [f"extra {kind}={value}" for kind, value in extra]
    return "; ".join(parts) or "none"


def draw(ids: Sequence[str], n: int, seed: int, stratum: str) -> list[str]:
    """All of `ids` when there are at most `n`, else `n` of them. Each stratum has its own stream,
    so the size of one never changes what is drawn from the other."""
    if len(ids) <= n:
        return list(ids)
    return random.Random(f"{seed}:{stratum}").sample(list(ids), n)


def load(results_dir: Path, processed_dir: Path, config_dir: Path | None, reference: str) -> tuple[compare.Gold, dict[str, compare.Run]]:
    specs = {name: config.resolve_system(name, config_dir) for name in config.list_systems(config_dir)}
    if reference not in specs:
        raise SampleError(f"{reference!r} is not in configs/systems.yaml")
    test = compare.load_test_data(processed_dir, results_dir, [compare.HEADLINE_SUBSET])
    ids = test.subset_ids[compare.HEADLINE_SUBSET]
    runs = {}
    for name, spec in specs.items():
        run = compare.load_run(name, spec, results_dir, [])
        compare.score_run(run, test)
        if all(i in run.items for i in ids):  # only a run that answered the whole subset can be paired
            runs[name] = run
    if reference not in runs:
        raise SampleError(f"{reference} has no complete run on {compare.HEADLINE_SUBSET}")
    return test, runs


def strongest_api(runs: Mapping[str, compare.Run], ids: Sequence[str]) -> str:
    """The API row with the highest exact match on `ids`; on a tie, the first in configs/systems.yaml."""
    scores = {
        name: sum(metrics.exact_match(run.items[i].pred, run.items[i].gold) for i in ids)
        for name, run in runs.items()
        if run.spec["price_id"]
    }
    if not scores:
        raise SampleError(f"no API row has a complete run on {compare.HEADLINE_SUBSET}")
    return max(scores, key=lambda name: scores[name])  # max keeps the first of equal scores


def sample(
    test: compare.Gold, runs: Mapping[str, compare.Run], reference: str, per_stratum: int, seed: int
) -> tuple[str, list[dict[str, str]], dict[str, tuple[int, int]]]:
    """The strongest API row, the rows of the file, and per stratum how many were drawn of how many there are."""
    ids = test.subset_ids[compare.HEADLINE_SUBSET]
    api = strongest_api(runs, ids)
    ft, other = runs[reference], runs[api]

    def right(run: compare.Run, item_id: str) -> bool:
        return metrics.exact_match(run.items[item_id].pred, run.items[item_id].gold)

    strata = {
        "gap": [i for i in ids if right(ft, i) and not right(other, i)],
        "ft": [i for i in ids if not right(ft, i)],
    }
    names = {"gap": f"ft right, {api} wrong", "ft": "ft wrong"}
    rows, counts = [], {}
    for stratum, pool in strata.items():
        chosen = draw(pool, per_stratum, seed, stratum)
        counts[stratum] = (len(chosen), len(pool))
        for item_id in sorted(chosen, key=lambda i: (test.by_id[i].scenario, subsets.id_sort_key(i))):
            example, gold = test.by_id[item_id], ft.items[item_id].gold
            rows.append(
                {
                    "id": item_id, "stratum": names[stratum], "scenario": example.scenario,
                    "request": example.text, "gold": call_text(gold),
                    "ft_answer": answer_text(ft, item_id), "ft_differs": differences(ft.items[item_id].pred, gold),
                    "api_system": api, "api_answer": answer_text(other, item_id),
                    "api_differs": differences(other.items[item_id].pred, gold),
                    "category": "", "note": "",
                }
            )
    return api, rows, counts


def run(
    *,
    results_dir: Path | None = None,
    processed_dir: Path | None = None,
    config_dir: Path | None = None,
    out_path: Path | None = None,
    reference: str = compare.REFERENCE,
    per_stratum: int = PER_STRATUM,
    seed: int = SEED,
    force: bool = False,
    out: Callable[[str], None] = print,
) -> int:
    results_dir = Path(results_dir or config.RESULTS_DIR)
    processed_dir = Path(processed_dir or config.PROCESSED_DIR)
    path = Path(out_path or results_dir / "error_analysis.csv")
    if path.exists() and not force:
        out(f"{path} exists and may hold hand labels; nothing written (--force replaces it)")
        return EXIT_EXISTS
    try:
        test, runs = load(results_dir, processed_dir, config_dir, reference)
        api, rows, counts = sample(test, runs, reference, per_stratum, seed)
    except (SampleError, compare.CompareError, config.ConfigError, FileNotFoundError) as exc:
        out(f"error: {exc}")
        return EXIT_ERROR
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    out(f"wrote {path}: {len(rows)} rows, strongest API row {api}")
    out(f"  ft right, {api} wrong: {counts['gap'][0]} of {counts['gap'][1]}")
    out(f"  ft wrong: {counts['ft'][0]} of {counts['ft'][1]}")
    return 0


def main(argv: list[str] | None = None, **overrides: Any) -> int:
    """`overrides` go straight to `run` (other directories, a quiet `out`)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--out", type=Path, help="where to write the CSV (default: results/error_analysis.csv)")
    parser.add_argument("--per-stratum", type=int, default=PER_STRATUM, help="items per stratum at most (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=SEED, help="the sampling seed (default: %(default)s)")
    parser.add_argument("--force", action="store_true", help="replace an existing file, and any labels in it")
    args = parser.parse_args(argv)
    return run(out_path=args.out, per_stratum=args.per_stratum, seed=args.seed, force=args.force, **overrides)


if __name__ == "__main__":
    raise SystemExit(main())
