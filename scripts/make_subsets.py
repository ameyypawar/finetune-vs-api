"""Build the fixed, nested, scenario-stratified evaluation subsets.

    python scripts/make_subsets.py [--check]

Reads data/processed/{dev,test}.jsonl (run scripts/prepare_data.py first) and writes the same
JSON to data/processed/subsets.json (what the evaluation runner reads) and
results/subsets.json (tracked, so the subsets and their hashes are public).

    D50 within D100 (from dev)       S300 within S500 (from test)
    seed 20261001, stratified by the 18 MASSIVE scenarios, at least min(5, size // 18) per
    scenario (D50 gets 2: 18 x 5 = 90 does not fit in 50).

With --check nothing is written; the exit status is 1 if the stored files differ from what
would be generated now.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finetune_vs_api import config, data, subsets


def run(
    check: bool = False,
    *,
    processed_dir: Path | None = None,
    results_dir: Path | None = None,
    out: Callable[[str], None] = print,
) -> int:
    processed_dir = processed_dir or config.PROCESSED_DIR
    results_dir = results_dir or config.RESULTS_DIR
    for split in ("dev", "test"):
        if not (processed_dir / f"{split}.jsonl").exists():
            out(f"{processed_dir / (split + '.jsonl')} not found; run scripts/prepare_data.py first")
            return 2
    dev = data.read_examples(processed_dir / "dev.jsonl")
    test = data.read_examples(processed_dir / "test.jsonl")
    document = subsets.make_subsets(dev, test)
    text = json.dumps(document, indent=2) + "\n"
    targets = [processed_dir / "subsets.json", results_dir / "subsets.json"]

    out(f"seed {document['seed']}, {document['n_scenarios']} scenarios")
    for name, entry in document["subsets"].items():
        counts = entry["per_scenario"].values()
        out(
            f"  {name:5s} {entry['split']:4s} n={entry['n']:<4d} "
            f"per scenario {min(counts)}..{max(counts)}  sha256 {entry['hash'][:16]}"
        )
    for split in ("dev", "test"):
        full = document["full"][split]
        out(f"  full  {split:4s} n={full['n']:<4d} sha256 {full['hash'][:16]}")

    if check:
        stale = [str(p) for p in targets if not p.exists() or p.read_text(encoding="utf-8") != text]
        if stale:
            out("STALE or missing: " + ", ".join(stale))
            return 1
        out("stored subsets match what would be generated now")
        return 0
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        out(f"wrote {target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify the stored files instead of writing")
    return run(parser.parse_args(argv).check)


if __name__ == "__main__":
    raise SystemExit(main())
