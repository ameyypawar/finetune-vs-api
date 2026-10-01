"""Scoring and statistics.

Everything here works on *already parsed* predictions. Whether a model output parsed and
satisfied the schema is decided by `schema.py` and passed in as a flag, so this module
has no opinion about JSON, prompts or models.

Task metrics, per example:

* exact match: the intent is equal AND the multiset of (slot type, normalized value)
  pairs is equal. Order does not matter; duplicates do.
* intent accuracy: the intent is equal.
* slot precision / recall / F1: micro-averaged over the whole set, from multiset
  true-positive / false-positive / false-negative counts.

An output that is not schema-valid scores zero on every task metric, and its gold slots
count as false negatives. The schema-valid rate is reported next to the task metrics so
a reader can see how much of a system's score is lost to format rather than to the task.

Statistics: percentile bootstrap confidence intervals, a paired bootstrap for the
difference between two systems on the same items, and an exact McNemar test.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

#: Seed for every bootstrap in this repository. Fixed so a published interval can be
#: reproduced exactly from the predictions file.
BOOTSTRAP_SEED = 20261001
BOOTSTRAP_RESAMPLES = 10_000
#: Resamples are drawn in blocks to bound memory. A module constant, not an argument:
#: the block size changes how the random stream is consumed, so it is part of what
#: makes a seeded interval reproducible.
_BLOCK = 250

SlotPair = tuple[str, str]  # (slot type, normalized value)


# --- normalization and per-example scores ----------------------------------------------


def normalize_value(value: object) -> str:
    """Lowercase, trim and collapse internal whitespace.

    This is the only normalization applied before comparing slot values. Punctuation and
    word order are left alone on purpose: a value that differs in either is a different
    prediction.
    """
    return " ".join(str(value).lower().split())


def _pair(slot: Any) -> SlotPair:
    """Accept {'type': .., 'value': ..}, (type, value), or an object with .type/.value."""
    if isinstance(slot, Mapping):
        slot_type, value = slot["type"], slot["value"]
    elif hasattr(slot, "type") and hasattr(slot, "value"):
        slot_type, value = slot.type, slot.value
    else:
        slot_type, value = slot
    return (str(slot_type), normalize_value(value))


def slot_multiset(slots: Iterable[Any] | None) -> Counter[SlotPair]:
    """The multiset of (type, normalized value) pairs for a slot list."""
    return Counter(_pair(s) for s in (slots or ()))


class SlotCounts(NamedTuple):
    tp: int
    fp: int
    fn: int


def slot_counts(pred_slots: Iterable[Any] | None, gold_slots: Iterable[Any] | None) -> SlotCounts:
    """Multiset true positives, false positives and false negatives for one example.

    Duplicates are counted: predicting the same slot twice when the gold has it once is
    one true positive and one false positive.
    """
    pred, gold = slot_multiset(pred_slots), slot_multiset(gold_slots)
    tp = sum((pred & gold).values())
    return SlotCounts(tp=tp, fp=sum(pred.values()) - tp, fn=sum(gold.values()) - tp)


def exact_match(pred: Mapping[str, Any] | None, gold: Mapping[str, Any]) -> bool:
    """Intent equal and slot multisets equal. `pred=None` (unparseable) never matches."""
    if pred is None:
        return False
    return pred.get("intent") == gold["intent"] and slot_multiset(pred.get("slots")) == slot_multiset(
        gold["slots"]
    )


def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    """Precision, recall and F1 from counts. A zero denominator gives 0.0, not NaN."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


@dataclass(frozen=True)
class Item:
    """One scored example.

    `text` is the request, `gold` the labelled call, `pred` the parsed model output (or
    None), and `valid` whether that output satisfied the schema.
    """

    text: str
    gold: Mapping[str, Any]
    pred: Mapping[str, Any] | None
    valid: bool


def item_scores(items: Sequence[Item]) -> dict[str, np.ndarray]:
    """Per-item score arrays, aligned with `items`, for the statistics below.

    Keys: em, intent, valid (0/1); tp, fp, fn (counts); n_pred and n_unfound (predicted
    slot values, and how many of them are not found in the request). Slot-level numbers
    for an invalid output are zero apart from fn.
    """
    n = len(items)
    cols = {k: np.zeros(n) for k in ("em", "intent", "valid", "tp", "fp", "fn", "n_pred", "n_unfound")}
    for i, item in enumerate(items):
        gold_slots = item.gold["slots"]
        cols["valid"][i] = float(item.valid)
        if not item.valid or item.pred is None:
            cols["fn"][i] = len(gold_slots)
            continue
        counts = slot_counts(item.pred["slots"], gold_slots)
        cols["em"][i] = float(exact_match(item.pred, item.gold))
        cols["intent"][i] = float(item.pred["intent"] == item.gold["intent"])
        cols["tp"][i], cols["fp"][i], cols["fn"][i] = counts
        text = normalize_value(item.text)
        values = [normalize_value(s["value"]) for s in item.pred["slots"]]
        cols["n_pred"][i] = len(values)
        cols["n_unfound"][i] = sum(1 for v in values if v not in text)
    return cols


# --- bootstrap statistics --------------------------------------------------------------

Statistic = Callable[[np.ndarray], np.ndarray]


def mean_stat(sample: np.ndarray) -> np.ndarray:
    """Mean of the first column. `sample` has shape (resamples, items, columns)."""
    return sample[..., 0].mean(axis=1)


def _sums(sample: np.ndarray) -> np.ndarray:
    return sample.sum(axis=1)


def precision_stat(sample: np.ndarray) -> np.ndarray:
    """Micro precision from columns (tp, fp, fn)."""
    s = _sums(sample)
    den = s[:, 0] + s[:, 1]
    return np.divide(s[:, 0], den, out=np.zeros_like(den), where=den > 0)


def recall_stat(sample: np.ndarray) -> np.ndarray:
    """Micro recall from columns (tp, fp, fn)."""
    s = _sums(sample)
    den = s[:, 0] + s[:, 2]
    return np.divide(s[:, 0], den, out=np.zeros_like(den), where=den > 0)


def f1_stat(sample: np.ndarray) -> np.ndarray:
    """Micro F1 from columns (tp, fp, fn)."""
    p, r = precision_stat(sample), recall_stat(sample)
    den = p + r
    return np.divide(2 * p * r, den, out=np.zeros_like(den), where=den > 0)


def ratio_stat(sample: np.ndarray) -> np.ndarray:
    """Ratio of sums from columns (numerator, denominator); 0 when the denominator is 0."""
    s = _sums(sample)
    return np.divide(s[:, 0], s[:, 1], out=np.zeros(len(s)), where=s[:, 1] > 0)


def _as_matrix(values: Any) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("expected a non-empty 1-D or 2-D array of per-item values")
    return x


def _resample_stats(
    x: np.ndarray, statistic: Statistic, n_resamples: int, rng: np.random.Generator
) -> np.ndarray:
    """`statistic` evaluated on each of `n_resamples` bootstrap resamples of the rows of x."""
    n = x.shape[0]
    out = np.empty(n_resamples)
    done = 0
    while done < n_resamples:
        m = min(_BLOCK, n_resamples - done)
        out[done : done + m] = statistic(x[rng.integers(0, n, size=(m, n))])
        done += m
    return out


def bootstrap_ci(
    values: Any,
    statistic: Statistic = mean_stat,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval for `statistic` over items.

    Items are resampled with replacement `n_resamples` times (10,000 by default) from a
    generator seeded with `seed`, so the same inputs always give the same interval.
    `values` is one value per item (shape (n,)) or several columns per item (shape
    (n, k)); `statistic` reduces a (resamples, n, k) array to one number per resample.
    """
    x = _as_matrix(values)
    rng = np.random.default_rng(seed)
    stats = _resample_stats(x, statistic, n_resamples, rng)
    tail = (1 - confidence) / 2
    low, high = np.quantile(stats, [tail, 1 - tail])
    return float(low), float(high)


class PairedResult(NamedTuple):
    diff: float  # statistic(a) - statistic(b) on the full data
    ci_low: float
    ci_high: float
    p_value: float
    n: int


def paired_bootstrap(
    a: Any,
    b: Any,
    statistic: Statistic = mean_stat,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> PairedResult:
    """Paired bootstrap for `statistic(a) - statistic(b)` on the same items.

    The same resampled indices are applied to both systems, which is what makes the
    comparison paired. The interval is the percentile interval of the difference. The
    p-value is two-sided, 2 * min(share of resamples with diff <= 0, share with diff
    >= 0), with a +1 correction in numerator and denominator so it is never exactly 0.
    """
    xa, xb = _as_matrix(a), _as_matrix(b)
    if xa.shape != xb.shape:
        raise ValueError(f"paired inputs must have the same shape, got {xa.shape} and {xb.shape}")
    n = xa.shape[0]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    done = 0
    while done < n_resamples:
        m = min(_BLOCK, n_resamples - done)
        idx = rng.integers(0, n, size=(m, n))
        diffs[done : done + m] = statistic(xa[idx]) - statistic(xb[idx])
        done += m
    observed = float(statistic(xa[None])[0] - statistic(xb[None])[0])
    tail = (1 - confidence) / 2
    low, high = np.quantile(diffs, [tail, 1 - tail])
    le, ge = int((diffs <= 0).sum()), int((diffs >= 0).sum())
    p = min(1.0, 2 * (min(le, ge) + 1) / (n_resamples + 1))
    return PairedResult(observed, float(low), float(high), p, n)


class McNemarResult(NamedTuple):
    a_only: int  # a correct, b wrong
    b_only: int  # b correct, a wrong
    p_value: float


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> McNemarResult:
    """Exact two-sided McNemar test on paired correctness.

    Only the discordant pairs carry information. Under the null each discordant pair is
    equally likely to favour either system, so the smaller count follows
    Binomial(discordant, 0.5); the two-sided p-value is twice its lower tail, capped at 1.
    Zero discordant pairs gives p = 1.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("paired inputs must have the same length")
    a_only = sum(1 for x, y in zip(a_correct, b_correct, strict=True) if x and not y)
    b_only = sum(1 for x, y in zip(a_correct, b_correct, strict=True) if y and not x)
    n = a_only + b_only
    if n == 0:
        return McNemarResult(0, 0, 1.0)
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return McNemarResult(a_only, b_only, min(1.0, 2 * tail / 2**n))


def percentile(values: Iterable[float], q: float) -> float:
    """Nearest-rank percentile (no interpolation).

    With N values sorted ascending, P(q) is the value at rank ceil(q/100 * N), counting
    from 1, and P(0) is the minimum. The result is always an observed value, which is
    what you want for latency: "95% of calls took no longer than this".
    """
    xs = sorted(values)
    if not xs:
        raise ValueError("percentile of an empty sequence")
    if not 0 <= q <= 100:
        raise ValueError(f"q must be in [0, 100], got {q}")
    rank = max(1, math.ceil(round(q * len(xs) / 100, 9)))
    return xs[rank - 1]


# --- the summary -----------------------------------------------------------------------


def _metric(value: float, ci: tuple[float, float] | None) -> dict[str, Any]:
    out: dict[str, Any] = {"value": value}
    if ci is not None:
        out["ci95"] = [ci[0], ci[1]]
    return out


def summarize(
    items: Sequence[Item],
    *,
    with_ci: bool = True,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Headline metrics for a set of scored items, each with a 95% bootstrap interval.

    Metrics: exact_match, intent_accuracy, slot_precision, slot_recall, slot_f1,
    schema_valid_rate, and unfound_value_rate (the share of predicted slot values that do
    not appear in the request, among slots from schema-valid outputs). All task metrics
    are over every item, so an invalid output counts against the system.
    """
    if not items:
        raise ValueError("cannot summarize zero items")
    s = item_scores(items)
    tp, fp, fn = s["tp"].sum(), s["fp"].sum(), s["fn"].sum()
    precision, recall, f1 = prf(tp, fp, fn)
    n_pred = s["n_pred"].sum()
    unfound = float(s["n_unfound"].sum() / n_pred) if n_pred else 0.0

    def ci(values: np.ndarray, stat: Statistic = mean_stat) -> tuple[float, float] | None:
        if not with_ci:
            return None
        return bootstrap_ci(values, stat, n_resamples=n_resamples, seed=seed)

    counts = np.stack([s["tp"], s["fp"], s["fn"]], axis=1)
    ratio = np.stack([s["n_unfound"], s["n_pred"]], axis=1)
    return {
        "n": len(items),
        "exact_match": _metric(float(s["em"].mean()), ci(s["em"])),
        "intent_accuracy": _metric(float(s["intent"].mean()), ci(s["intent"])),
        "slot_precision": _metric(precision, ci(counts, precision_stat)),
        "slot_recall": _metric(recall, ci(counts, recall_stat)),
        "slot_f1": _metric(f1, ci(counts, f1_stat)),
        "schema_valid_rate": _metric(float(s["valid"].mean()), ci(s["valid"])),
        "unfound_value_rate": _metric(unfound, ci(ratio, ratio_stat)),
        "slot_counts": {"tp": int(tp), "fp": int(fp), "fn": int(fn)},
        "predicted_slots": int(n_pred),
    }
