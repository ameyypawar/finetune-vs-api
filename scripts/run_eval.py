"""Run one system on the dev or test split, or on a fixed subset of it.

    python scripts/run_eval.py --system NAME --split dev|test
        [--subset full|S500|S300|D100|D50] [--prompt NAME] [--limit N]
        [--concurrency N] [--max-usd X] [--resume]

Writes, under results/runs/<system>__<split>/:

    predictions.jsonl             raw answers, appended as they arrive (never overwritten)
    summary.<subset>.json         metrics with 95% intervals, cost per 1,000 calls, p50/p95
                                  latency, the price entry used (URL and date), config hash
                                  and git commit; written once every item of the subset is done
    summary.<subset>.partial.json the same while items are still missing

--subset reads data/processed/subsets.json (make it with scripts/make_subsets.py). Its default
is the system's test_subset on test and the whole split on dev. --prompt selects another prompt
the system allows on dev; test always uses the system's own.

--split test refuses to start unless the system is locked for that subset with an unchanged
configuration (scripts/lock_test.py).

API systems run on free tiers with daily caps. When one runs out the run saves what it has and
stops, printing when the cap resets; run the same command with --resume after that and it
continues from where it stopped. Items already answered are never sent again.

Exit status: 0 finished; 75 stopped on a daily quota (resumable); 76 stopped by --max-usd;
77 authentication failed; 78 too many consecutive failures; 3 the test split is locked;
2 any other error; 130 interrupted (progress is saved).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finetune_vs_api import client, config, evaluate, subsets

EXIT_BY_STATUS = {
    "complete": 0,
    "quota_exhausted": 75,
    "budget": 76,
    "auth_error": 77,
    "too_many_errors": 78,
}
EXIT_LOCKED = 3
EXIT_ERROR = 2
EXIT_INTERRUPTED = 130
PROGRESS_EVERY = 10


def headline(summary: dict[str, Any]) -> list[str]:
    m = summary["metrics"]
    if m is None:
        return ["no scored items"]

    def fmt(key: str) -> str:
        entry = m[key]
        low, high = entry.get("ci95", (None, None))
        interval = f" [{low:.3f}, {high:.3f}]" if low is not None else ""
        return f"{entry['value']:.3f}{interval}"

    lines = [
        f"  items scored   {summary['n_scored']} of {summary['subset']['n']} in {summary['subset']['name']}"
        f" ({summary['n_calls_failed']} calls failed)",
        f"  exact match    {fmt('exact_match')}",
        f"  intent acc     {fmt('intent_accuracy')}",
        f"  slot F1        {fmt('slot_f1')}",
        f"  schema valid   {fmt('schema_valid_rate')}",
    ]
    if summary["latency_s"]:
        lines.append(f"  latency        p50 {summary['latency_s']['p50']:.2f}s  p95 {summary['latency_s']['p95']:.2f}s")
    per_1k = summary["cost"].get("per_1k_calls_usd")
    if per_1k:
        lines.append(f"  per 1,000 calls ${per_1k['lower']:.4f} to ${per_1k['upper']:.4f}  ({summary['billing_basis']})")
    return lines


def main(
    argv: list[str] | None = None,
    *,
    out: Callable[[str], None] = print,
    **overrides: Any,
) -> int:
    """`overrides` go straight to `evaluate.run_eval` (a stub transport, other directories, ...)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--system", required=True, help="system name from configs/systems.yaml")
    parser.add_argument("--split", required=True, choices=evaluate.EVAL_SPLITS)
    parser.add_argument("--subset", choices=subsets.SUBSET_NAMES, help="fixed subset (default: see above)")
    parser.add_argument("--prompt", help="another prompt the system allows on dev")
    parser.add_argument("--limit", type=int, help="run at most N items of the subset")
    parser.add_argument("--concurrency", type=int, help="parallel requests (default: the system's cap, else 4)")
    parser.add_argument("--max-usd", type=float, help="stop if projected spend at list price would pass this")
    parser.add_argument("--resume", action="store_true", help="continue an earlier run of the same configuration")
    args = parser.parse_args(argv)
    if not overrides.get("environ"):
        config.load_env()

    def progress(done: int, total: int, row: dict[str, Any]) -> None:
        if done % PROGRESS_EVERY == 0 or done == total or row.get("error"):
            note = f"  FAILED {row['error_kind']}" if row.get("error") else ""
            out(f"  [{done}/{total}] id {row['id']}{note}")

    try:
        outcome = evaluate.run_eval(
            args.system, args.split, subset=args.subset, prompt=args.prompt, limit=args.limit,
            concurrency=args.concurrency, max_usd=args.max_usd, resume=args.resume,
            progress=progress, announce=out, **overrides,
        )
    except config.LockError as exc:
        out(f"error: {exc}")
        return EXIT_LOCKED
    except KeyboardInterrupt:
        out("interrupted; progress is saved. Run the same command with --resume to continue.")
        return EXIT_INTERRUPTED
    except (
        evaluate.EvalError, config.ConfigError, client.ClientError, FileNotFoundError, FileExistsError, ValueError,
    ) as exc:
        out(f"error: {exc}")
        return EXIT_ERROR

    batch = outcome.batch
    out(f"{outcome.status.upper()}: {batch.completed} done, {batch.errors} failed, {batch.skipped} already done, {batch.remaining} left")
    if outcome.status == "quota_exhausted":
        when = (
            datetime.fromtimestamp(batch.reset_at, UTC).strftime("%Y-%m-%d %H:%M UTC") if batch.reset_at else "unknown"
        )
        out(f"  {batch.message}")
        out(f"  progress saved to {outcome.predictions_path}")
        out(f"  quota resets at: {when}. Then run the same command again with --resume.")
    elif outcome.status != "complete":
        out(f"  {batch.message}")
    for line in headline(outcome.summary):
        out(line)
    out(f"  summary: {outcome.summary_path}")
    return EXIT_BY_STATUS.get(outcome.status, EXIT_ERROR)


if __name__ == "__main__":
    raise SystemExit(main())
