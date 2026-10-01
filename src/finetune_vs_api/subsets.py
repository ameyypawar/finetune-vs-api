"""Fixed evaluation subsets, stratified by MASSIVE scenario.

API systems run on free tiers with daily request caps, so they cannot be evaluated on the
whole test split. Instead every system is evaluated on one of a few fixed, nested subsets:

    dev:   D50 within D100        test:  S300 within S500

Each subset is stratified by scenario (MASSIVE has 18) with proportional allocation and a
floor of `min(5, size // 18)` items per scenario. The floor is 5 wherever 5 per scenario
fits (D100, S300, S500); D50 cannot hold 5 for each of 18 scenarios (18 x 5 = 90 > 50) so
it gets 2. Sampling is seeded, so the same data always gives the same subsets, and each
subset carries a hash of its ids that the test lock records.

`full` is not stored: it means every item of the split.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import Example

SUBSET_SEED = 20261001
MIN_PER_SCENARIO = 5
FULL = "full"
#: name -> (split, size, parent). A subset with a parent is drawn from inside the parent.
SUBSET_PLAN: dict[str, tuple[str, int, str | None]] = {
    "D100": ("dev", 100, None),
    "D50": ("dev", 50, "D100"),
    "S500": ("test", 500, None),
    "S300": ("test", 300, "S500"),
}
SUBSET_NAMES = (FULL, *SUBSET_PLAN)
_SPLIT_STREAM = {"dev": 1, "test": 2}  # independent random streams per split


def id_sort_key(item_id: str) -> tuple[int, str]:
    """Order numeric-looking ids numerically ("9" before "10"), others lexicographically."""
    return (len(item_id), item_id)


def subset_hash(ids: Iterable[str]) -> str:
    """sha256 of the ids, sorted and newline-joined, so the order they are listed in is moot."""
    return hashlib.sha256("\n".join(sorted(str(i) for i in ids)).encode("utf-8")).hexdigest()


def allocate(counts: Mapping[str, int], total: int, min_per: int) -> dict[str, int]:
    """Split `total` picks across strata: proportional to size, never below `min_per`.

    Largest-remainder rounding with ties broken by stratum name, so the result is
    deterministic. A stratum smaller than `min_per` contributes all it has.
    """
    names = sorted(counts)
    population = sum(counts.values())
    if total > population:
        raise ValueError(f"cannot pick {total} from {population}")
    floor = {s: min(min_per, counts[s]) for s in names}
    if sum(floor.values()) > total:
        raise ValueError(f"a minimum of {min_per} per stratum needs {sum(floor.values())} > {total}")
    ideal = {s: total * counts[s] / population for s in names}
    picks = {s: min(counts[s], max(floor[s], math.floor(ideal[s]))) for s in names}
    while sum(picks.values()) < total:
        pool = [s for s in names if picks[s] < counts[s]]
        s = max(pool, key=lambda name: (ideal[name] - picks[name], -names.index(name)))
        picks[s] += 1
    while sum(picks.values()) > total:
        pool = [s for s in names if picks[s] > floor[s]]
        s = max(pool, key=lambda name: (picks[name] - ideal[name], names.index(name)))
        picks[s] -= 1
    return picks


def _effective_min(size: int, n_strata: int, min_per_scenario: int) -> int:
    return min(min_per_scenario, size // n_strata)


def _draw_split(
    examples: Sequence[Example], plan: Sequence[tuple[str, int, str | None]], seed: int, stream: int, min_per: int
) -> dict[str, Any]:
    by_scenario: dict[str, list[str]] = defaultdict(list)
    for example in sorted(examples, key=lambda e: id_sort_key(e.id)):
        by_scenario[example.scenario].append(example.id)
    scenarios = sorted(by_scenario)
    rng = np.random.default_rng([seed, stream])
    order = {s: [by_scenario[s][i] for i in rng.permutation(len(by_scenario[s]))] for s in scenarios}

    picks: dict[str, dict[str, int]] = {}
    out: dict[str, Any] = {}
    for name, size, parent in plan:
        floor = _effective_min(size, len(scenarios), min_per)
        pool = {s: len(by_scenario[s]) for s in scenarios} if parent is None else picks[parent]
        picks[name] = allocate(pool, size, floor)
        # Every subset is a prefix of the same per-scenario shuffle, so a subset with a
        # parent is automatically contained in it (its allocation never exceeds the parent's).
        ids = sorted((i for s in scenarios for i in order[s][: picks[name][s]]), key=id_sort_key)
        out[name] = {
            "n": len(ids),
            "hash": subset_hash(ids),
            "parent": parent,
            "min_per_scenario_applied": floor,
            "per_scenario": dict(picks[name]),
            "ids": ids,
        }
    return out


def make_subsets(
    dev: Sequence[Example],
    test: Sequence[Example],
    *,
    seed: int = SUBSET_SEED,
    min_per_scenario: int = MIN_PER_SCENARIO,
) -> dict[str, Any]:
    """Build every subset in SUBSET_PLAN from the dev and test examples."""
    for label, rows in (("dev", dev), ("test", test)):
        wrong = {e.split for e in rows} - {label}
        if wrong:
            raise ValueError(f"{label} examples include {sorted(wrong)}")
    subsets: dict[str, Any] = {}
    for split, rows in (("dev", dev), ("test", test)):
        plan = [(name, size, parent) for name, (sp, size, parent) in SUBSET_PLAN.items() if sp == split]
        # Parents must be drawn first; SUBSET_PLAN lists them before their children.
        for name, entry in _draw_split(rows, plan, seed, _SPLIT_STREAM[split], min_per_scenario).items():
            subsets[name] = {"split": split, **entry}
    scenarios = {e.scenario for e in [*dev, *test]}
    return {
        "schema_version": 1,
        "seed": seed,
        "stratify_by": "scenario",
        "n_scenarios": len(scenarios),
        "min_per_scenario": min_per_scenario,
        "full": {
            "dev": {"n": len(dev), "hash": subset_hash(e.id for e in dev)},
            "test": {"n": len(test), "hash": subset_hash(e.id for e in test)},
        },
        "subsets": subsets,
    }


def load_subsets(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/make_subsets.py first")
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ResolvedSubset:
    name: str
    split: str
    ids: tuple[str, ...]
    hash: str


def resolve_subset(
    subsets: Mapping[str, Any], name: str, split: str, split_ids: Sequence[str]
) -> ResolvedSubset:
    """The ids and hash of subset `name` for `split`, checked against the actual split.

    Raises ValueError if the name is unknown, belongs to the other split, no longer
    matches its recorded hash, or names an id that is not in the split.
    """
    if name not in SUBSET_NAMES:
        raise ValueError(f"unknown subset {name!r}; choose from {list(SUBSET_NAMES)}")
    if name == FULL:
        recorded = subsets.get("full", {}).get(split, {}).get("hash")
        actual = subset_hash(split_ids)
        if recorded and recorded != actual:
            raise ValueError(f"the {split} split no longer matches subsets.json; re-run make_subsets.py")
        return ResolvedSubset(name, split, tuple(split_ids), actual)
    entry = subsets["subsets"][name]
    if entry["split"] != split:
        raise ValueError(f"subset {name} is a {entry['split']} subset and cannot be used with --split {split}")
    if subset_hash(entry["ids"]) != entry["hash"]:
        raise ValueError(f"subset {name} no longer matches its recorded hash; subsets.json was edited")
    missing = sorted(set(entry["ids"]) - set(split_ids))
    if missing:
        raise ValueError(f"subset {name} names ids not in the {split} split, e.g. {missing[:3]}")
    return ResolvedSubset(name, split, tuple(entry["ids"]), entry["hash"])
