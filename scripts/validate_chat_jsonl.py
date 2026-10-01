"""Check a chat-format JSONL file before using it as training data.

    python scripts/validate_chat_jsonl.py path/to/file.jsonl

This is the entry point for bringing your own OpenAI fine-tuning file. Each line must be
`{"messages": [...]}` with roles system / user / assistant, string content, and a final
assistant turn. Valid OpenAI features that v1 does not handle (tools and tool_calls,
multimodal content parts, per-message weights, multi-turn records) are reported as
unsupported rather than silently dropped.

Exit status: 0 if the file is usable, 1 if it has errors or unsupported features, 2 if the
file cannot be read.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finetune_vs_api.data import validate_chat_jsonl

MAX_SHOWN = 10


def run(path: Path, out: Callable[[str], None] = print) -> int:
    try:
        report = validate_chat_jsonl(path)
    except (OSError, UnicodeDecodeError) as exc:
        out(f"cannot read {path}: {exc}")
        return 2
    out(f"{report.path}: {report.n_records} records")
    for kind, label in (("error", "errors"), ("unsupported", "unsupported in v1"), ("warning", "warnings")):
        issues = report.of_kind(kind)
        if not issues:
            continue
        out(f"{len(issues)} {label}:")
        for issue in issues[:MAX_SHOWN]:
            where = f"line {issue.line}" if issue.line else "file"
            out(f"  {where}: {issue.message}")
        if len(issues) > MAX_SHOWN:
            out(f"  ... and {len(issues) - MAX_SHOWN} more")
    out("OK" if report.ok else "NOT USABLE as-is")
    return 0 if report.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("path", type=Path, help="chat-format JSONL file")
    return run(parser.parse_args(argv).path)


if __name__ == "__main__":
    raise SystemExit(main())
