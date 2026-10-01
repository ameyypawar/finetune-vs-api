"""Lock a system's configuration before it touches the test split.

    python scripts/lock_test.py --write --system NAME [--subset S500] --reason "why now"
    python scripts/lock_test.py --show [--system NAME]

`run_eval.py --split test` refuses to start unless the system has a lock whose configuration
hash and test-subset hash both match what is on disk now. A lock is per system, so an API row
can be locked and run before the fine-tuned model exists. History is append-only
(results/test_lock.jsonl, tracked in git): every lock, and why, stays on the record.

--show prints, for each system, what still blocks locking it, whether its current
configuration is locked for its test subset, and the lock history. It always exits 0.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finetune_vs_api import config


def run(
    *,
    write: bool = False,
    show: bool = False,
    system: str | None = None,
    subset: str | None = None,
    reason: str | None = None,
    config_dir: Path | None = None,
    processed_dir: Path | None = None,
    lock_path: Path | None = None,
    out: Callable[[str], None] = print,
) -> int:
    paths = {"config_dir": config_dir, "processed_dir": processed_dir, "lock_path": lock_path}
    try:
        if write:
            if not system or not reason:
                out("--write needs --system and --reason")
                return 2
            entry = config.write_test_lock(system, subset, reason, **paths)
            out(f"locked {entry['system']} for {entry['subset']} at {entry['locked_at']}")
            out(f"  config hash : {entry['config_hash']}")
            for name, value in entry["components"].items():
                out(f"    {name:11s} {value[:16]}")
            out(f"  subset hash : {entry['subset_hash']}")
            out(f"  git commit  : {entry['git_commit'] or '(no commits yet)'}"
                + (" (uncommitted changes)" if entry["git_dirty"] else ""))
            out("Any change to the above now needs a new lock, with a reason, before a test run.")
            return 0
        if show:
            names = [system] if system else config.list_systems(config_dir)
            for name in names:
                spec = config.resolve_system(name, config_dir)
                target = subset or spec["test_subset"]
                out(f"{name}  (test subset {target})")
                for blocker in config.lock_blockers(name, config_dir=config_dir):
                    out(f"  BLOCKED: {blocker}")
                entry, why = config.lock_status(name, target, **paths)
                out(f"  {'LOCKED' if entry else 'NOT LOCKED'}: {why}")
                for past in (e for e in config.read_lock_history(lock_path) if e["system"] == name):
                    out(f"  history: {past['locked_at']} {past['subset']} {past['config_hash'][:12]} - {past['reason']}")
            return 0
    except (config.LockError, config.ConfigError, FileNotFoundError) as exc:
        out(f"error: {exc}")
        return 2
    out("choose --write or --show")
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="append a lock entry for --system")
    mode.add_argument("--show", action="store_true", help="show lock status and history")
    parser.add_argument("--system", help="system name from configs/systems.yaml")
    parser.add_argument("--subset", help="test subset to lock (default: the system's test_subset)")
    parser.add_argument("--reason", help="why the configuration is being locked now (required with --write)")
    args = parser.parse_args(argv)
    return run(write=args.write, show=args.show, system=args.system, subset=args.subset, reason=args.reason)


if __name__ == "__main__":
    raise SystemExit(main())
