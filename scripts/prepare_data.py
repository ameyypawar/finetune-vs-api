"""Download MASSIVE 1.1, convert one locale to JSON function calls, and write the audit.

    python scripts/prepare_data.py [--locale en-US] [--refresh]

Writes (all under data/ except the audit, which is small enough to keep in git):

    data/raw/<archive>.tar.gz                  the pinned download
    data/processed/{train,dev,test}.jsonl      labelled examples
    data/processed/sft_{train,dev}.jsonl       OpenAI chat format, one record per example
    data/processed/sft_{train,dev}.ids.txt     which example each chat record came from
    results/data_audit.json                    split sizes, label coverage, duplicates, overlap

Prints the sha256 of the archive so it can be pinned in configs/data.yaml. Training labels
come from MASSIVE's human-annotated train split only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx

from finetune_vs_api import config, data
from finetune_vs_api.prompts import finetune_record
from finetune_vs_api.schema import label_inventory

EXIT_CHECKSUM = 2
EXIT_UNEXPECTED_DATA = 3


def run(
    locale: str | None = None,
    refresh: bool = False,
    *,
    config_dir: Path | None = None,
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    results_dir: Path | None = None,
    transport: httpx.BaseTransport | None = None,
    out: Callable[[str], None] = print,
) -> int:
    cfg = config.load_yaml("data", config_dir)
    source = cfg["source"]
    locale = locale or cfg["locale"]
    raw_dir = raw_dir or config.RAW_DIR
    processed_dir = processed_dir or config.PROCESSED_DIR
    results_dir = results_dir or config.RESULTS_DIR
    archive = raw_dir / Path(source["url"]).name

    out(f"archive: {archive}")
    try:
        actual = data.download_massive(
            source["url"], archive, source.get("sha256"), refresh=refresh, transport=transport
        )
    except data.ChecksumError as exc:
        out(f"CHECKSUM ERROR: {exc}")
        return EXIT_CHECKSUM
    pinned = source.get("sha256")
    out(f"sha256: {actual}")
    out(
        "  pinned in configs/data.yaml: "
        + ("matches" if pinned else "NOT PINNED yet. Set source.sha256 to the value above.")
    )

    rows = data.read_locale(archive, locale)
    examples = [data.to_example(row) for row in rows]
    by_split = {s: [e for e in examples if e.split == s] for s in data.SPLITS}
    for split, items in by_split.items():
        data.write_examples(processed_dir / f"{split}.jsonl", items)

    for split in ("train", "dev"):
        items = by_split[split]
        data.write_chat_jsonl(
            processed_dir / f"sft_{split}.jsonl",
            (finetune_record(e) for e in items),
            ids=[e.id for e in items],
        )

    expected = cfg.get("expected") if locale == cfg["locale"] else None
    report = data.audit(examples, expected=expected)
    inventory = label_inventory(by_split["train"])
    mismatches = sum(1 for row in rows if data.parse_annot_utt(row["annot_utt"]).text != row["utt"])
    dropped = sum(data.parse_annot_utt(row["annot_utt"]).empty_dropped for row in rows)
    document = {
        "dataset": {
            "name": source["name"],
            "version": source["version"],
            "locale": locale,
            "license": source["license"],
            "url": source["url"],
            "archive_sha256": actual,
        },
        **report,
        "integrity": {
            "markup_text_differs_from_utt": mismatches,
            "empty_value_slots_dropped": dropped,
            "slot_values_not_found_in_text": report["slots"]["values_not_found_in_text"],
        },
        "inventory": inventory.to_dict(),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "data_audit.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    s, lab, ov = report["splits"], report["labels"], report["overlap"]
    dup = report["duplicates_within_train"]
    out(f"splits: train {s['train']}, dev {s['dev']}, test {s['test']} (total {s['total']})")
    out(
        f"labels: {lab['intents']['per_split']['train']} intents and "
        f"{lab['slot_types']['per_split']['train']} slot types in train; "
        f"{report['scenarios']['n_total']} scenarios"
    )
    out(
        f"duplicates in train: {dup['duplicate_text_groups']} texts "
        f"({dup['groups_with_conflicting_intent']} with conflicting intents)"
    )
    out(
        f"text shared with train: dev {ov['train_dev']['dev_items']} items, "
        f"test {ov['train_test']['test_items']} items"
    )
    out(f"wrote {processed_dir} and {results_dir / 'data_audit.json'}")
    if expected is not None:
        verdict = report["expected"]
        if verdict["matches"]:
            out("audit MATCHES the expected counts")
        else:
            out("audit DIFFERS from the expected counts:")
            for line in verdict["differences"]:
                out(f"  {line}")
            return EXIT_UNEXPECTED_DATA
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--locale", default=None, help="locale to prepare (default: the one in configs/data.yaml)")
    parser.add_argument("--refresh", action="store_true", help="download the archive again even if it is present")
    args = parser.parse_args(argv)
    return run(args.locale, args.refresh)


if __name__ == "__main__":
    raise SystemExit(main())
