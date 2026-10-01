"""Cost arithmetic.

The API systems run on free tiers, so nothing is actually billed. Every cost in this
repository is therefore what the same calls WOULD cost at the provider's paid list price,
and says so (`BILLING_BASIS`). Because a provider's prompt caching may or may not apply,
cost is given as a pair of bounds: an upper bound with no caching, and a lower bound with
the static prompt prefix priced at the cached rate.

Token conventions (the OpenAI Chat Completions ones):

* `prompt_tokens` is all input tokens, including any that were served from cache.
* `cached_tokens` is the cached subset of `prompt_tokens`.
* `completion_tokens` is all billable output tokens INCLUDING reasoning tokens.
  `reasoning_tokens` is informational: the reasoning subset of `completion_tokens`.
  The client normalizes providers that report reasoning separately into this convention.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple

BILLING_BASIS = "free tier; priced at paid list price"
HOURS_PER_MONTH = 730
TOKENS_PER_MTOK = 1_000_000

Counter = Callable[[str], int]


@dataclass(frozen=True)
class Price:
    """USD per million tokens. `cached_input_per_mtok=None` means no caching discount."""

    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float | None = None

    @classmethod
    def from_entry(cls, entry: Mapping[str, Any]) -> Price:
        """Build from a `prices:` entry of configs/sources.yaml."""
        usd = entry["usd_per_mtok"]
        return cls(usd["input"], usd["output"], usd.get("cached_input"))

    @property
    def cached(self) -> float:
        return self.input_per_mtok if self.cached_input_per_mtok is None else self.cached_input_per_mtok


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    reported: bool = True  # False when the endpoint sent no usage object at all
    estimated: bool = False  # True when the counts were made locally
    estimate_method: str | None = None

    @property
    def is_complete(self) -> bool:
        return self.prompt_tokens is not None and self.completion_tokens is not None

    @property
    def total_tokens(self) -> int | None:
        if not self.is_complete:
            return None
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any] | None) -> Usage:
        """Rebuild from `to_dict()` output. No object at all means the endpoint reported none."""
        if not obj:
            return cls(reported=False)
        names = ("prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens",
                 "reported", "estimated", "estimate_method")
        return cls(**{name: obj[name] for name in names if name in obj})


class CostBounds(NamedTuple):
    upper: float  # no caching: every prompt token at the full input price
    lower: float  # the static prefix priced at the cached rate


def _require_complete(usage: Usage) -> tuple[int, int]:
    if not usage.is_complete:
        raise ValueError("usage has no token counts; estimate them first with estimate_usage()")
    return usage.prompt_tokens, usage.completion_tokens  # type: ignore[return-value]


def api_call_cost(usage: Usage, price: Price) -> float:
    """Cost in USD of one call: uncached input, cached input, and output (reasoning included)."""
    prompt, completion = _require_complete(usage)
    cached = min(max(usage.cached_tokens or 0, 0), prompt)
    uncached = prompt - cached
    return (
        uncached * price.input_per_mtok + cached * price.cached + completion * price.output_per_mtok
    ) / TOKENS_PER_MTOK


def api_cost_bounds(usage: Usage, price: Price, cacheable_prefix_tokens: int) -> CostBounds:
    """Bounds on what a call would cost at list price, whether or not caching applies.

    `upper` assumes no caching. `lower` assumes the first `cacheable_prefix_tokens` prompt
    tokens (the static prefix; capped at the prompt length) are read from cache. They are
    equal when the provider has no cached-input discount.
    """
    prompt, completion = _require_complete(usage)
    prefix = min(max(cacheable_prefix_tokens, 0), prompt)
    out = completion * price.output_per_mtok
    upper = (prompt * price.input_per_mtok + out) / TOKENS_PER_MTOK
    lower = ((prompt - prefix) * price.input_per_mtok + prefix * price.cached + out) / TOKENS_PER_MTOK
    return CostBounds(upper=upper, lower=lower)


def per_1k(total_cost: float, n_calls: int) -> float:
    """Cost per 1,000 calls given a total over `n_calls`."""
    if n_calls <= 0:
        raise ValueError("n_calls must be positive")
    return total_cost / n_calls * 1000


def selfhost_per_1k(usd_hr: float, req_s: float) -> float:
    """Cost per 1,000 requests for a GPU rented at `usd_hr` and kept busy at `req_s` requests/s.

    This is the price at full utilization: an idle or half-used GPU costs proportionally more.
    """
    if req_s <= 0:
        raise ValueError("req_s must be positive")
    return usd_hr / (3600 * req_s) * 1000


def breakeven_calls_per_month(usd_hr: float, api_cost_per_call: float, hours: float = HOURS_PER_MONTH) -> float:
    """Monthly API calls at which API spend equals the cost of keeping the GPU rented.

    Assumes the GPU is paid for every hour of the month (`hours`, 730 by default) and that
    self-hosting has no per-call cost beyond that. Returns infinity when the API call is free.
    """
    if api_cost_per_call < 0 or usd_hr < 0:
        raise ValueError("costs must not be negative")
    if api_cost_per_call == 0:
        return math.inf
    return usd_hr * hours / api_cost_per_call


# --- local token counting (used only when a provider sends no usage) --------------------------


@functools.lru_cache(maxsize=1)
def default_counter() -> tuple[Counter, str]:
    """A token counter and the name of its method.

    tiktoken's `o200k_base` when it loads (it fetches its vocabulary on first use), else a
    4-characters-per-token rule of thumb. The method name travels with every estimate so a
    summary can say which one produced it.
    """
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("o200k_base")
        return (lambda text: len(encoding.encode(text, disallowed_special=()))), "tiktoken:o200k_base"
    except Exception:  # offline or unavailable: fall back to a rule of thumb, and say so
        return (lambda text: max(1, math.ceil(len(text) / 4)) if text else 0), "chars/4 (tiktoken unavailable)"


PER_MESSAGE_OVERHEAD = 3  # tokens of chat framing per message, and 3 to prime the reply


def count_message_tokens(messages: Sequence[Mapping[str, str]], counter: Counter | None = None) -> int:
    """Approximate prompt tokens for a chat request. Provider counts differ slightly."""
    count = counter or default_counter()[0]
    return sum(count(m["content"]) + PER_MESSAGE_OVERHEAD for m in messages) + PER_MESSAGE_OVERHEAD


def estimate_usage(
    messages: Sequence[Mapping[str, str]],
    output_text: str | None,
    counter: Counter | None = None,
    method: str | None = None,
) -> Usage:
    """Usage counted locally, flagged `estimated=True`. For calls where the provider sent none."""
    if counter is None:
        counter, found = default_counter()
        method = method or found
    return Usage(
        prompt_tokens=count_message_tokens(messages, counter),
        completion_tokens=counter(output_text) if output_text else 0,
        reported=False,
        estimated=True,
        estimate_method=method or "custom counter",
    )


def cost_summary(usages: Sequence[Usage], price: Price, cacheable_prefix_tokens: int) -> dict[str, Any]:
    """Totals and per-1,000-call figures over many calls, as upper/lower bounds."""
    if not usages:
        raise ValueError("no calls to summarize")
    bounds = [api_cost_bounds(u, price, cacheable_prefix_tokens) for u in usages]
    upper, lower = sum(b.upper for b in bounds), sum(b.lower for b in bounds)
    return {
        "billing_basis": BILLING_BASIS,
        "calls": len(usages),
        "calls_with_reported_usage": sum(1 for u in usages if u.reported and not u.estimated),
        "calls_with_estimated_usage": sum(1 for u in usages if u.estimated),
        "prompt_tokens": sum(u.prompt_tokens for u in usages),
        "completion_tokens": sum(u.completion_tokens for u in usages),
        "reasoning_tokens": sum(u.reasoning_tokens or 0 for u in usages),
        "cacheable_prefix_tokens": cacheable_prefix_tokens,
        "total_usd": {"upper": upper, "lower": lower},
        "per_1k_calls_usd": {"upper": per_1k(upper, len(usages)), "lower": per_1k(lower, len(usages))},
    }
