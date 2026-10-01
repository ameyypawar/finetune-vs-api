"""MASSIVE 1.1 (en-US): download, parse, audit, and the chat-format export.

MASSIVE ships each locale as one JSONL file inside a tarball. A row looks like

    {"id": "1", "partition": "train", "scenario": "alarm", "intent": "alarm_set",
     "utt": "wake me up at nine am on friday",
     "annot_utt": "wake me up at [time : nine am] on [date : friday]", ...}

The labelled call is recovered from `annot_utt`: the intent is given, and the slots are
the `[label : value]` spans, in order of appearance. Values are exact substrings of the
request, which is what lets the task say "copy the span verbatim".
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .metrics import normalize_value, slot_multiset

SPLITS = ("train", "dev", "test")

# --- types -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Slot:
    type: str
    value: str


@dataclass(frozen=True)
class Example:
    """One labelled request. `slots` are in order of appearance in `text`."""

    id: str
    split: str
    scenario: str
    intent: str
    text: str
    slots: tuple[Slot, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "split": self.split,
            "scenario": self.scenario,
            "intent": self.intent,
            "text": self.text,
            "slots": [{"type": s.type, "value": s.value} for s in self.slots],
        }

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> Example:
        return cls(
            id=str(obj["id"]),
            split=obj["split"],
            scenario=obj["scenario"],
            intent=obj["intent"],
            text=obj["text"],
            slots=tuple(Slot(s["type"], s["value"]) for s in obj["slots"]),
        )


@dataclass(frozen=True)
class ParsedAnnotation:
    text: str  # the request with the markup removed
    slots: tuple[Slot, ...]  # in order of appearance
    empty_dropped: int = 0  # slots whose value was empty, which have no span to copy


# --- download ----------------------------------------------------------------------------


class ChecksumError(RuntimeError):
    """The downloaded or existing archive does not match the pinned sha256."""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def download_massive(
    url: str,
    dest: Path,
    sha256: str | None = None,
    *,
    refresh: bool = False,
    transport: httpx.BaseTransport | None = None,
) -> str:
    """Fetch the MASSIVE archive to `dest`, verify it, and return its sha256.

    The Hugging Face loader for this dataset no longer works, so the canonical tarball is
    the source. With `sha256` given, a mismatch raises `ChecksumError` and leaves nothing
    behind that could be mistaken for good data. With `sha256=None` nothing is verified
    and the computed hash is returned, which is how the hash is first obtained for
    pinning. An existing file is reused unless `refresh` is set.
    """
    dest = Path(dest)
    expected = sha256.lower() if sha256 else None
    if dest.exists() and not refresh:
        actual = sha256_file(dest)
        if expected and actual != expected:
            raise ChecksumError(
                f"{dest} has sha256 {actual}, expected {expected}. "
                "Delete it or re-run with --refresh to download it again."
            )
        return actual

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    timeout = httpx.Timeout(30.0, read=120.0)
    try:
        with httpx.Client(transport=transport, follow_redirects=True, timeout=timeout) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with open(part, "wb") as handle:
                    for block in response.iter_bytes(1 << 20):
                        digest.update(block)
                        handle.write(block)
        actual = digest.hexdigest()
        if expected and actual != expected:
            raise ChecksumError(
                f"downloaded {url} has sha256 {actual}, expected {expected}; discarded."
            )
        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)
    return actual


def read_locale(tar_path: Path, locale: str = "en-US") -> list[dict[str, Any]]:
    """Raw rows for one locale, read straight out of the tarball (nothing is extracted)."""
    suffix = f"/data/{locale}.jsonl"
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if member.isfile() and ("/" + member.name).endswith(suffix):
                handle = tar.extractfile(member)
                if handle is None:
                    raise FileNotFoundError(f"cannot read {member.name} in {tar_path}")
                with handle:
                    return [
                        json.loads(line)
                        for line in handle.read().decode("utf-8").splitlines()
                        if line.strip()
                    ]
    raise FileNotFoundError(f"no member ending in {suffix!r} in {tar_path}")


# --- parsing -----------------------------------------------------------------------------

# `[label : value]`. The label is everything up to the FIRST " : " (so a value such as
# "5:30", or even "5 : 30", survives intact); the value runs to the closing bracket.
# Neither part may contain a bracket, so spans cannot nest or run together.
_SPAN = re.compile(r"\[([^\[\]]+?) : ([^\[\]]*)\]")


def parse_annot_utt(annot_utt: str) -> ParsedAnnotation:
    """Parse MASSIVE's `[label : value]` markup.

    Returns the plain request and the slots in order of appearance. A request with no
    markup has no slots. A span with an empty value is dropped and counted (there is
    nothing in the request to copy). Any bracket left over after the spans are
    consumed means the markup is malformed, and raises ValueError rather than guessing.
    """
    slots: list[Slot] = []
    dropped = 0
    pieces: list[str] = []
    cursor = 0
    for match in _SPAN.finditer(annot_utt):
        pieces.append(annot_utt[cursor : match.start()])
        label, value = match.group(1).strip(), match.group(2)
        pieces.append(value)
        if value.strip():
            slots.append(Slot(label, value))
        else:
            dropped += 1
        cursor = match.end()
    pieces.append(annot_utt[cursor:])
    text = "".join(pieces)
    if "[" in text or "]" in text:
        raise ValueError(f"malformed slot markup: {annot_utt!r}")
    return ParsedAnnotation(text=text, slots=tuple(slots), empty_dropped=dropped)


def to_example(row: Mapping[str, Any]) -> Example:
    """Convert one raw MASSIVE row into an `Example`."""
    split = row["partition"]
    if split not in SPLITS:
        raise ValueError(f"unknown partition {split!r} in row {row.get('id')!r}")
    parsed = parse_annot_utt(row["annot_utt"])
    return Example(
        id=str(row["id"]),
        split=split,
        scenario=row["scenario"],
        intent=row["intent"],
        text=row["utt"],
        slots=parsed.slots,
    )


def write_examples(path: Path, examples: Iterable[Example]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_json(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_examples(path: Path) -> list[Example]:
    with open(path, encoding="utf-8") as handle:
        return [Example.from_json(json.loads(line)) for line in handle if line.strip()]


# --- audit -------------------------------------------------------------------------------

_APOSTROPHES = re.compile(r"['\u2019`]")
_LOOSE = re.compile(r"[^a-z0-9]+")


def _loose(text: str) -> str:
    """Normalization that also ignores punctuation, to catch near-duplicates.

    Apostrophes are dropped ("what's" becomes "whats") and any other punctuation becomes a
    space, so contractions and hyphenation do not hide a repeated request.
    """
    return _LOOSE.sub(" ", _APOSTROPHES.sub("", text.lower())).strip()


def _label_key(example: Example) -> tuple[str, tuple[tuple[str, str, int], ...]]:
    counts = slot_multiset(example.slots)
    return (example.intent, tuple(sorted((t, v, n) for (t, v), n in counts.items())))


def _coverage(by_split: Mapping[str, Sequence[Example]], pick) -> dict[str, Any]:
    """Which labels each split uses, and which are missing or absent from train."""
    seen = {s: {label for e in by_split[s] for label in pick(e)} for s in SPLITS}
    universe = set().union(*seen.values())
    return {
        "n_total": len(universe),
        "per_split": {s: len(seen[s]) for s in SPLITS},
        "missing_from_split": {s: sorted(universe - seen[s]) for s in SPLITS},
        "not_in_train": {s: sorted(seen[s] - seen["train"]) for s in ("dev", "test")},
    }


def _duplicates_within(train: Sequence[Example]) -> dict[str, Any]:
    groups: dict[str, list[Example]] = defaultdict(list)
    for example in train:
        groups[normalize_value(example.text)].append(example)
    dup = {text: rows for text, rows in groups.items() if len(rows) > 1}
    identical = intent_conflict = slot_conflict = 0
    for rows in dup.values():
        keys = {_label_key(r) for r in rows}
        if len(keys) == 1:
            identical += 1
        elif len({r.intent for r in rows}) > 1:
            intent_conflict += 1
        else:
            slot_conflict += 1
    shown = sorted(dup.items(), key=lambda kv: (kv[1][0].id.zfill(8), kv[0]))[:10]
    return {
        "normalization": "lowercase, trim, collapse whitespace",
        "duplicate_text_groups": len(dup),
        "rows_in_groups": sum(len(rows) for rows in dup.values()),
        "groups_with_identical_label": identical,
        "groups_with_conflicting_intent": intent_conflict,
        "groups_with_same_intent_different_slots": slot_conflict,
        "first_groups": [
            {"text": text, "ids": [r.id for r in rows], "intents": sorted({r.intent for r in rows})}
            for text, rows in shown
        ],
    }


def _overlap(by_split: Mapping[str, Sequence[Example]], norm) -> dict[str, Any]:
    texts = {s: {norm(e.text) for e in by_split[s]} for s in SPLITS}
    train_labels: dict[str, set] = defaultdict(set)
    for e in by_split["train"]:
        train_labels[norm(e.text)].add(_label_key(e))

    def pair(a: str, b: str) -> dict[str, int]:
        return {
            "shared_texts": len(texts[a] & texts[b]),
            f"{a}_items": sum(1 for e in by_split[a] if norm(e.text) in texts[b]),
            f"{b}_items": sum(1 for e in by_split[b] if norm(e.text) in texts[a]),
        }

    out: dict[str, Any] = {
        "train_dev": pair("train", "dev"),
        "train_test": pair("train", "test"),
        "dev_test": pair("dev", "test"),
    }
    for split in ("dev", "test"):
        hits = [e for e in by_split[split] if norm(e.text) in texts["train"]]
        out[f"{split}_item_ids_in_train"] = sorted((e.id for e in hits), key=lambda i: (len(i), i))
        out[f"{split}_items_with_identical_label_in_train"] = sum(
            1 for e in hits if _label_key(e) in train_labels[norm(e.text)]
        )
    return out


def audit(examples: Sequence[Example], expected: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Facts about the data a reader needs before trusting a number computed on it.

    Split sizes, label coverage, duplicate requests within train, and request text shared
    between train, dev and test, under two normalizations (the metric one, and one that
    also ignores punctuation). With `expected` (keys train, dev, test, intents,
    slot_types) the report also says whether the data matches.
    """
    bad = {e.split for e in examples} - set(SPLITS)
    if bad:
        raise ValueError(f"unknown split(s) in examples: {sorted(bad)}")
    by_split = {s: [e for e in examples if e.split == s] for s in SPLITS}
    slot_values_missing = sum(1 for e in examples for s in e.slots if s.value not in e.text)
    report: dict[str, Any] = {
        "splits": {**{s: len(by_split[s]) for s in SPLITS}, "total": len(examples)},
        "labels": {
            "intents": _coverage(by_split, lambda e: {e.intent}),
            "slot_types": _coverage(by_split, lambda e: {s.type for s in e.slots}),
        },
        "scenarios": {
            "n_total": len({e.scenario for e in examples}),
            "per_split": {s: dict(sorted(Counter(e.scenario for e in by_split[s]).items())) for s in SPLITS},
        },
        "slots": {
            "total": sum(len(e.slots) for e in examples),
            "examples_without_slots": {s: sum(1 for e in by_split[s] if not e.slots) for s in SPLITS},
            "max_per_example": max((len(e.slots) for e in examples), default=0),
            "values_not_found_in_text": slot_values_missing,
        },
        "duplicates_within_train": _duplicates_within(by_split["train"]),
        "overlap": {
            "normalization": "lowercase, trim, collapse whitespace",
            **_overlap(by_split, normalize_value),
            "ignoring_punctuation": _overlap(by_split, _loose),
        },
    }
    if expected is not None:
        got = {
            **report["splits"],
            "intents": report["labels"]["intents"]["per_split"]["train"],
            "slot_types": report["labels"]["slot_types"]["per_split"]["train"],
        }
        diffs = [f"{k}: expected {v}, got {got.get(k)}" for k, v in expected.items() if got.get(k) != v]
        report["expected"] = {"values": dict(expected), "matches": not diffs, "differences": diffs}
    return report


# --- chat-format JSONL ---------------------------------------------------------------------

CHAT_ROLES = ("system", "user", "assistant")
_TOOL_RECORD_KEYS = ("tools", "tool_choice", "functions", "function_call", "parallel_tool_calls")
_TOOL_MESSAGE_KEYS = ("tool_calls", "function_call", "tool_call_id")


def write_chat_jsonl(
    path: Path, records: Iterable[Mapping[str, Any]], ids: Sequence[str] | None = None
) -> int:
    """Write OpenAI chat-format records, one `{"messages": [...]}` object per line.

    The file holds nothing but `messages`, so it is also valid as an OpenAI fine-tuning
    file. When `ids` is given they go to a sidecar next to it (`sft_train.jsonl` ->
    `sft_train.ids.txt`, one id per line, same order); the leakage tests use it to prove
    which split every training record came from.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    if ids is not None:
        if len(ids) != count:
            raise ValueError(f"{len(ids)} ids for {count} records")
        sidecar_path(path).write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    return count


def sidecar_path(path: Path) -> Path:
    return Path(path).with_suffix(".ids.txt")


def read_ids(path: Path) -> list[str]:
    return [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


@dataclass(frozen=True)
class Issue:
    line: int  # 1-based; 0 for problems with the file as a whole
    kind: str  # "error" | "unsupported" | "warning"
    message: str


@dataclass
class ChatReport:
    path: str
    n_records: int = 0
    issues: list[Issue] = field(default_factory=list)

    def of_kind(self, kind: str) -> list[Issue]:
        return [i for i in self.issues if i.kind == kind]

    @property
    def errors(self) -> list[Issue]:
        return self.of_kind("error")

    @property
    def unsupported(self) -> list[Issue]:
        return self.of_kind("unsupported")

    @property
    def warnings(self) -> list[Issue]:
        return self.of_kind("warning")

    @property
    def ok(self) -> bool:
        """True when nothing is invalid or unsupported. Warnings do not fail a file."""
        return not self.errors and not self.unsupported


def _check_record(obj: Any, line: int, report: ChatReport) -> None:
    def add(kind: str, message: str) -> None:
        report.issues.append(Issue(line, kind, message))

    if not isinstance(obj, dict):
        return add("error", "record is not a JSON object")
    for key in _TOOL_RECORD_KEYS:
        if key in obj:
            add("unsupported", f"'{key}' (tool / function calling) is not supported in v1")
    extra = sorted(set(obj) - {"messages", *_TOOL_RECORD_KEYS})
    if extra:
        add("warning", f"extra top-level keys {extra} are ignored here; remove them if the file is also uploaded elsewhere")
    messages = obj.get("messages")
    if not isinstance(messages, list) or not messages:
        return add("error", "'messages' must be a non-empty list")

    roles: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            add("error", f"messages[{index}] is not an object")
            continue
        role = message.get("role")
        if role in ("tool", "function"):
            add("unsupported", f"messages[{index}] has role '{role}' (tool / function calling)")
        elif role not in CHAT_ROLES:
            add("error", f"messages[{index}] has invalid role {role!r}; expected one of {list(CHAT_ROLES)}")
        for key in _TOOL_MESSAGE_KEYS:
            if key in message:
                add("unsupported", f"messages[{index}] has '{key}' (tool / function calling)")
        if "weight" in message:
            add("unsupported", f"messages[{index}] has 'weight' (per-message training weight)")
        content = message.get("content")
        if isinstance(content, list):
            add("unsupported", f"messages[{index}] has multimodal content parts; only text is supported")
        elif not isinstance(content, str):
            if not any(k in message for k in _TOOL_MESSAGE_KEYS):
                add("error", f"messages[{index}] content must be a string, got {type(content).__name__}")
        elif not content.strip():
            add("error", f"messages[{index}] has empty content")
        if isinstance(role, str):
            roles.append(role)

    if roles and roles[-1] != "assistant":
        add("error", "the last message must be from the assistant")
    if "system" in roles[1:]:
        add("error", "a system message may only come first")
    if roles.count("user") > 1 or roles.count("assistant") > 1:
        add("unsupported", "multi-turn records (more than one user or assistant message) are not supported in v1")
    elif roles.count("user") == 0 and roles:
        add("error", "no user message")


def validate_chat_jsonl(path: Path) -> ChatReport:
    """Check a chat-format JSONL file and say what v1 of this pipeline can and cannot use.

    Errors are things that are simply invalid: bad JSON, a missing or misshapen
    `messages`, a role outside system / user / assistant, non-string content, empty
    content, or a final turn that is not the assistant's. Unsupported covers things
    that are valid OpenAI fine-tuning data but that v1 does not handle: tools and
    tool_calls, multimodal content parts, per-message weights, and multi-turn records.
    """
    path = Path(path)
    report = ChatReport(path=str(path))
    with open(path, encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            report.n_records += 1
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                report.issues.append(Issue(number, "error", f"invalid JSON: {exc.msg}"))
                continue
            _check_record(obj, number, report)
    if report.n_records == 0:
        report.issues.append(Issue(0, "error", "the file has no records"))
    return report


def chat_to_prompt_completion(record: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    """TRL's conversational prompt/completion format.

    `{"messages": [system?, user, assistant]}` becomes
    `{"prompt": [system?, user], "completion": [assistant]}`. Trained in that form, the
    loss falls on the completion only. Anything the validator would flag raises ValueError.
    """
    probe = ChatReport(path="<record>")
    _check_record(record, 1, probe)
    problems = [i.message for i in probe.issues if i.kind in ("error", "unsupported")]
    if problems:
        raise ValueError("cannot convert record: " + "; ".join(problems))
    messages = [{"role": m["role"], "content": m["content"]} for m in record["messages"]]
    return {"prompt": messages[:-1], "completion": messages[-1:]}


def iter_chat_records(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
