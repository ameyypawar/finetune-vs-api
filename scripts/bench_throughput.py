"""Measure how fast one OpenAI-compatible server answers a fixed set of prompts, at several
concurrencies, and price the result as self-hosted cost per 1,000 calls.

    python scripts/bench_throughput.py --base-url http://127.0.0.1:8000/v1 --model ft-qwen3-4b-lora \\
        --prompts throughput_prompts.jsonl [--concurrency 1,8,32,64] [--requests 1000] [--gpu T4] \\
        [--price-key aws_g4dn_xlarge_ondemand] [--out PATH] [--system NAME] [--dtype float16] \\
        [--engine vllm] [--engine-version X] [--max-tokens 256] [--extra KEY=VALUE ...]

Built to run on the Kaggle T4 next to the server (kaggle/serve/serve_eval_on_kaggle.py starts
it). It talks to the server with the `openai` package, which vLLM installs, and otherwise uses
the standard library; it also reads this repository's `cost` module (standard library only) and
`configs/sources.yaml` (PyYAML, which vLLM installs too). It sends nothing but chat requests to
--base-url, so point it at a server of your own.

--prompts is a JSONL file with one request per line, either {"messages": [{"role": ..., "content":
...}, ...]} or {"text": "..."} (a single user message); a .txt file holds one user message per
line. The prompts are shuffled once with --seed and then cycled to make each level's --requests
requests. Every request is temperature 0 with --max-tokens.

For each concurrency level it keeps exactly that many requests in flight until --requests have
finished, and records requests per second, output tokens per second and the latency of each
successful request (p50 and p95 by the nearest-rank rule the run summaries use). Eight warm-up
requests are sent first and not counted. A level with a failed request is reported but cannot be
the operating point.

Operating point: the highest concurrency whose p95 latency is at most --p95-limit seconds (1.0).
If none qualifies it is null, and nothing is invented in its place.

Cost: `cost.selfhost_per_1k(usd_per_hour, requests_per_s)` at the operating point, for the
on-demand and the spot price of the rental that --price-key names in configs/sources.yaml
(`aws_g4dn_xlarge_ondemand` is `gpu_rental["aws-g4dn.xlarge"]` at its on-demand price; it is the
headline). It is the price with the GPU kept fully busy: an idle or half-used GPU costs more.

Writes results/serving/<gpu>.json (or --out). The shape scripts/compare.py reads is at the top
level: "gpu", "system", "levels" (each with "concurrency", "requests_per_s", "latency_s": {"p50",
"p95"}) and "operating_point" ({"concurrency": ...}); the rest records what was measured and how.

Exit status: 0 written; 2 bad options, an unreadable prompt or price file, or a server that
answers none of the warm-up requests (nothing is written then).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import http.client
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finetune_vs_api import cost

SCHEMA_VERSION = 1
DEFAULT_LEVELS = (1, 8, 32, 64)
DEFAULT_REQUESTS = 1000
DEFAULT_GPU = "T4"
DEFAULT_PRICE_KEY = "aws_g4dn_xlarge_ondemand"
DEFAULT_MAX_TOKENS = 256
DEFAULT_P95_LIMIT_S = 1.0
DEFAULT_TIMEOUT_S = 120.0
WARMUP_REQUESTS = 8
ERROR_SAMPLES = 3
EXIT_ERROR = 2
LATENCY_METHOD = "nearest-rank"
COST_BASIS = "measured on a free Kaggle GPU; priced as if rented at the list price below, kept fully busy"
COST_FORMULA = "usd_per_hour / (3600 * requests_per_s) * 1000  (finetune_vs_api.cost.selfhost_per_1k)"


class BenchError(ValueError):
    """A bad option, an unreadable input, or a server that does not answer."""


# --- small helpers -------------------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile: the value at rank ceil(q/100 * N). Same rule as metrics.percentile
    (checked by a test); repeated here so this script needs no numpy."""
    xs = sorted(values)
    if not xs:
        raise ValueError("percentile of an empty sequence")
    if not 0 <= q <= 100:
        raise ValueError(f"q must be in [0, 100], got {q}")
    rank = max(1, math.ceil(round(q * len(xs) / 100, 9)))
    return xs[rank - 1]


def parse_levels(text: str) -> tuple[int, ...]:
    """'1,8,32,64' -> (1, 8, 32, 64): positive, distinct, ascending."""
    try:
        levels = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise BenchError(f"--concurrency must be comma-separated integers, got {text!r}") from None
    if not levels or any(n < 1 for n in levels):
        raise BenchError(f"--concurrency needs positive integers, got {text!r}")
    return tuple(sorted(set(levels)))


def load_prompts(path: Path) -> list[list[dict[str, str]]]:
    """The requests in a prompt file, each as a chat message list."""
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BenchError(f"cannot read --prompts {path}: {exc}") from None
    prompts: list[list[dict[str, str]]] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        if path.suffix == ".txt":
            prompts.append([{"role": "user", "content": line.strip()}])
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            raise BenchError(f"{path}:{number} is not valid JSON") from None
        if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
            messages = obj["messages"]
            if not messages or not all(
                isinstance(m, dict) and isinstance(m.get("role"), str) and isinstance(m.get("content"), str)
                for m in messages
            ):
                raise BenchError(f"{path}:{number}: 'messages' must be a list of {{role, content}} strings")
            prompts.append([{"role": m["role"], "content": m["content"]} for m in messages])
        elif isinstance(obj, dict) and isinstance(obj.get("text"), str) and obj["text"].strip():
            prompts.append([{"role": "user", "content": obj["text"]}])
        else:
            raise BenchError(f"{path}:{number} needs a 'messages' list or a 'text' string")
    if not prompts:
        raise BenchError(f"{path} holds no prompts")
    return prompts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def safe_label(label: str) -> str:
    """A GPU label as a file name: letters, digits, dot, dash and underscore only."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", label.strip()).strip("-.")
    if not cleaned:
        raise BenchError("--gpu needs a label such as T4")
    return cleaned


# --- the requests --------------------------------------------------------------------------------------


class Reply(NamedTuple):
    """What one successful chat completion tells the benchmark."""

    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    finish_reason: str | None = None


Sender = Callable[[list[dict[str, str]], int], Awaitable[Reply]]


class OpenAISender:
    """Sends one chat completion per call with the `openai` package: temperature 0, no retries
    (a retry would hide a slow or failing server inside the latency)."""

    def __init__(self, base_url: str, model: str, *, api_key: str = "EMPTY", timeout: float = DEFAULT_TIMEOUT_S):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise BenchError("the openai package is required (python -m pip install openai)") from None
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
        self._model = model

    async def __call__(self, messages: list[dict[str, str]], max_tokens: int) -> Reply:
        response = await self._client.chat.completions.create(
            model=self._model, messages=messages, max_tokens=max_tokens, temperature=0
        )
        usage = response.usage
        return Reply(
            completion_tokens=getattr(usage, "completion_tokens", None),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            finish_reason=response.choices[0].finish_reason if response.choices else None,
        )

    async def aclose(self) -> None:
        await self._client.close()


# --- one concurrency level --------------------------------------------------------------------------------


class LevelRaw:
    """What one concurrency level measured, before it is summarized. A plain class, not a
    dataclass: tests load scripts without registering them in sys.modules, which dataclasses
    under `from __future__ import annotations` cannot cope with."""

    def __init__(self, concurrency: int, requests: int):
        self.concurrency = concurrency
        self.requests = requests
        self.wall_s = 0.0
        self.latencies: list[float] = []
        self.completion_tokens: list[int] = []
        self.prompt_tokens: list[int] = []
        self.finish_reasons: dict[str, int] = {}
        self.errors = 0
        self.error_samples: list[str] = []


async def run_level(
    sender: Sender,
    prompts: Sequence[list[dict[str, str]]],
    concurrency: int,
    requests: int,
    *,
    max_tokens: int,
    clock: Callable[[], float] = time.perf_counter,
) -> LevelRaw:
    """Send `requests` requests with exactly `concurrency` in flight, and time each one."""
    raw = LevelRaw(concurrency, requests)
    next_index = iter(range(requests))  # asyncio is single-threaded: workers never race on this

    async def worker() -> None:
        for index in next_index:
            messages = prompts[index % len(prompts)]
            started = clock()
            try:
                reply = await sender(messages, max_tokens)
            except Exception as exc:  # a failed request is data, not a reason to stop the level
                raw.errors += 1
                if len(raw.error_samples) < ERROR_SAMPLES:
                    raw.error_samples.append(f"{type(exc).__name__}: {str(exc)[:200]}")
                continue
            raw.latencies.append(clock() - started)
            if reply.completion_tokens is not None:
                raw.completion_tokens.append(int(reply.completion_tokens))
            if reply.prompt_tokens is not None:
                raw.prompt_tokens.append(int(reply.prompt_tokens))
            key = reply.finish_reason or "unknown"
            raw.finish_reasons[key] = raw.finish_reasons.get(key, 0) + 1

    began = clock()
    await asyncio.gather(*(worker() for _ in range(min(concurrency, requests))))
    raw.wall_s = clock() - began
    return raw


def summarize_level(raw: LevelRaw, p95_limit_s: float) -> dict[str, Any]:
    """The numbers for one level. Rates count successful requests only."""
    ok = len(raw.latencies)
    wall = raw.wall_s
    latency: dict[str, Any] | None = None
    if ok:
        latency = {
            "p50": percentile(raw.latencies, 50),
            "p95": percentile(raw.latencies, 95),
            "p99": percentile(raw.latencies, 99),
            "mean": sum(raw.latencies) / ok,
            "max": max(raw.latencies),
            "n": ok,
            "method": LATENCY_METHOD,
            "note": "seconds from sending a request to its complete answer; failed requests excluded",
        }
    reported = len(raw.completion_tokens)
    return {
        "concurrency": raw.concurrency,
        "requests": raw.requests,
        "ok": ok,
        "errors": raw.errors,
        "error_rate": raw.errors / raw.requests if raw.requests else 0.0,
        "error_samples": raw.error_samples,
        "wall_s": wall,
        "requests_per_s": ok / wall if wall > 0 else 0.0,
        # None when the server reported no usage for any request: nothing is estimated here.
        "output_tokens_per_s": sum(raw.completion_tokens) / wall if reported and wall > 0 else None,
        "mean_output_tokens": sum(raw.completion_tokens) / reported if reported else None,
        "mean_prompt_tokens": sum(raw.prompt_tokens) / len(raw.prompt_tokens) if raw.prompt_tokens else None,
        "finish_reasons": dict(sorted(raw.finish_reasons.items())),
        "latency_s": latency,
        "meets_p95_limit": bool(latency and raw.errors == 0 and latency["p95"] <= p95_limit_s),
    }


def operating_point(levels: Sequence[Mapping[str, Any]], p95_limit_s: float) -> dict[str, Any] | None:
    """The highest concurrency whose p95 latency is at most `p95_limit_s`, among levels with no failed request."""
    qualifying = [lv for lv in levels if lv["meets_p95_limit"]]
    if not qualifying:
        return None
    best = max(qualifying, key=lambda lv: lv["concurrency"])
    return {
        "concurrency": best["concurrency"],
        "requests_per_s": best["requests_per_s"],
        "output_tokens_per_s": best["output_tokens_per_s"],
        "latency_s": {"p50": best["latency_s"]["p50"], "p95": best["latency_s"]["p95"]},
        "rule": f"the highest concurrency whose p95 latency is at most {p95_limit_s:g} s, with no failed request",
    }


# --- cost ------------------------------------------------------------------------------------------------------


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def known_price_keys(rentals: Mapping[str, Any]) -> list[str]:
    return sorted(
        f"{_normalize(rental_id)}_{billing.replace('_', '')}"
        for rental_id, entry in rentals.items()
        for billing in entry["usd_per_hour"]
    )


def resolve_price_key(rentals: Mapping[str, Any], key: str) -> tuple[str, str]:
    """('aws-g4dn.xlarge', 'on_demand') from 'aws_g4dn_xlarge_ondemand': the entry of
    sources.yaml `gpu_rental` and the billing option within its `usd_per_hour`."""
    wanted = _normalize(key)
    for rental_id, entry in rentals.items():
        prefix = _normalize(rental_id) + "_"
        if wanted.startswith(prefix):
            rest = wanted[len(prefix):].replace("_", "")
            for billing in entry["usd_per_hour"]:
                if billing.replace("_", "") == rest:
                    return rental_id, billing
    raise BenchError(f"unknown --price-key {key!r}; known keys: {', '.join(known_price_keys(rentals))}")


def load_rentals(sources_path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError:
        raise BenchError("PyYAML is needed to read configs/sources.yaml (python -m pip install pyyaml)") from None
    try:
        doc = yaml.safe_load(Path(sources_path).read_text(encoding="utf-8"))
        return doc["gpu_rental"]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise BenchError(f"cannot read gpu_rental from {sources_path}: {exc}") from None


def selfhost_costs(rentals: Mapping[str, Any], price_key: str, levels: Sequence[Mapping[str, Any]], operating: Mapping[str, Any] | None) -> dict[str, Any]:
    """Cost per 1,000 calls at every level, and at the operating point, for each billing option."""
    rental_id, headline = resolve_price_key(rentals, price_key)
    rental = rentals[rental_id]
    prices = {billing: float(usd) for billing, usd in rental["usd_per_hour"].items()}

    def per_1k(rate: float) -> dict[str, float]:
        return {billing: cost.selfhost_per_1k(usd, rate) for billing, usd in prices.items()}

    by_concurrency = {
        str(lv["concurrency"]): per_1k(lv["requests_per_s"]) for lv in levels if lv["requests_per_s"] > 0
    }
    at_operating = None
    if operating is not None and operating["requests_per_s"] > 0:
        at_operating = {
            "concurrency": operating["concurrency"],
            "requests_per_s": operating["requests_per_s"],
            "per_1k_calls_usd": per_1k(operating["requests_per_s"]),
            "headline_billing": headline,
            "headline_usd": cost.selfhost_per_1k(prices[headline], operating["requests_per_s"]),
        }
    return {
        "billing_basis": COST_BASIS,
        "formula": COST_FORMULA,
        "assumes": "the GPU is rented for every hour and kept fully busy at the operating point; an idle or half-used GPU costs proportionally more",
        "price_key": price_key,
        "rental": {
            "id": rental_id,
            "gpu": rental.get("gpu"),
            "url": rental.get("url"),
            "retrieved_on": rental.get("retrieved_on"),
            "usd_per_hour": prices,
        },
        "headline_billing": headline,
        "at_operating_point": at_operating,
        "by_concurrency": by_concurrency,
    }


# --- what the box and the server say about themselves ----------------------------------------------------------------


def query_gpu(runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """Name, memory and driver of the first GPU from nvidia-smi; empty when it is not available."""
    try:
        done = runner(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=30,
        )
        name, memory, driver = (part.strip() for part in done.stdout.strip().splitlines()[0].split(","))
        return {"name": name, "memory_mib": int(memory), "driver": driver}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}


def _get_json(url: str, timeout: float = 5.0) -> Any:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, http.client.HTTPException):  # URLError is an OSError
        return None


def detect_engine(base_url: str) -> dict[str, str | None]:
    """Best-effort name and version of the server: vLLM answers GET /version, llama.cpp GET /props."""
    root = re.sub(r"/v1/?$", "", base_url.rstrip("/"))
    doc = _get_json(f"{root}/version")
    if isinstance(doc, dict) and isinstance(doc.get("version"), str):
        return {"engine": "vllm", "version": doc["version"]}
    doc = _get_json(f"{root}/props")
    if isinstance(doc, dict) and "build_info" in doc:
        return {"engine": "llama.cpp", "version": str(doc["build_info"])}
    return {"engine": None, "version": None}


# --- the whole run ---------------------------------------------------------------------------------------------------


async def run_sweep(
    sender: Sender,
    prompts: Sequence[list[dict[str, str]]],
    levels: Sequence[int],
    requests: int,
    *,
    max_tokens: int,
    p95_limit_s: float,
    warmup: int = WARMUP_REQUESTS,
    clock: Callable[[], float] = time.perf_counter,
    out: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Warm up, then run every level in ascending order. Raises BenchError if the server answers no warm-up request."""
    if warmup > 0:
        first = await run_level(sender, prompts, min(4, warmup), warmup, max_tokens=max_tokens, clock=clock)
        if not first.latencies:
            sample = first.error_samples[0] if first.error_samples else "no reply"
            raise BenchError(f"the server answered none of {warmup} warm-up requests ({sample})")
    results = []
    for concurrency in levels:
        raw = await run_level(sender, prompts, concurrency, requests, max_tokens=max_tokens, clock=clock)
        level = summarize_level(raw, p95_limit_s)
        results.append(level)
        p = level["latency_s"]
        out(
            f"  concurrency {concurrency:>3}: {level['requests_per_s']:.2f} req/s"
            + (f", p50 {p['p50']:.3f}s p95 {p['p95']:.3f}s" if p else ", no successful request")
            + (f", {level['errors']} failed" if level["errors"] else "")
        )
    return results


def parse_extra(pairs: Sequence[str]) -> dict[str, str]:
    extra: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise BenchError(f"--extra needs KEY=VALUE, got {pair!r}")
        extra[key.strip()] = value
    return extra


def default_sources_path() -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / "sources.yaml"


def default_out_path(gpu: str) -> Path:
    return Path(__file__).resolve().parent.parent / "results" / "serving" / f"{safe_label(gpu)}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--base-url", required=True, help="the server's OpenAI-compatible base, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True, help="the model name to send in each request")
    parser.add_argument("--prompts", required=True, type=Path, help="JSONL (or .txt) file of requests")
    parser.add_argument("--concurrency", default=",".join(map(str, DEFAULT_LEVELS)), help="comma-separated levels (default 1,8,32,64)")
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS, help="requests per level (default 1000)")
    parser.add_argument("--gpu", default=DEFAULT_GPU, help="label of the GPU, and the file name (default T4)")
    parser.add_argument("--price-key", default=DEFAULT_PRICE_KEY, help="rental price to headline (default aws_g4dn_xlarge_ondemand)")
    parser.add_argument("--out", type=Path, help="output file (default results/serving/<gpu>.json)")
    parser.add_argument("--sources", type=Path, help="prices file (default configs/sources.yaml)")
    parser.add_argument("--system", help="the system this benchmarks (default: --model)")
    parser.add_argument("--dtype", help="the dtype the server runs the model in; the server does not report it")
    parser.add_argument("--engine", help="name of the serving engine (default: detected)")
    parser.add_argument("--engine-version", help="version of the serving engine (default: detected)")
    parser.add_argument("--gpu-name", help="the GPU's product name (default: nvidia-smi)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--p95-limit", type=float, default=DEFAULT_P95_LIMIT_S, help="seconds (default 1.0)")
    parser.add_argument("--warmup", type=int, default=WARMUP_REQUESTS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="per-request timeout in seconds")
    parser.add_argument("--seed", type=int, default=0, help="shuffles the prompts once")
    parser.add_argument("--extra", action="append", default=[], metavar="KEY=VALUE", help="recorded under 'extra' (repeatable)")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    sender_factory: Callable[..., Sender] | None = None,
    out: Callable[[str], None] = print,
    clock: Callable[[], float] = time.perf_counter,
    engine_probe: Callable[[str], Mapping[str, str | None]] = detect_engine,
    gpu_probe: Callable[[], Mapping[str, Any]] = query_gpu,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> int:
    """`sender_factory(base_url, model, timeout=...)` makes the object that sends one request (the
    openai package by default); tests pass a stub."""
    args = build_parser().parse_args(argv)
    try:
        if args.requests < 1 or args.max_tokens < 1 or args.p95_limit <= 0 or args.warmup < 0:
            raise BenchError("--requests, --max-tokens and --p95-limit must be positive and --warmup not negative")
        levels = parse_levels(args.concurrency)
        extra = parse_extra(args.extra)
        label = safe_label(args.gpu)  # the label goes into the report and, by default, the file name
        out_path = args.out or default_out_path(label)
        prompts = load_prompts(args.prompts)
        rentals = load_rentals(args.sources or default_sources_path())
        resolve_price_key(rentals, args.price_key)  # fail before the sweep, not after it
        order = list(range(len(prompts)))
        random.Random(args.seed).shuffle(order)
        shuffled = [prompts[i] for i in order]

        factory = sender_factory or OpenAISender
        sender = factory(args.base_url, args.model, timeout=args.timeout)
        out(f"benchmarking {args.model} at {args.base_url}: levels {list(levels)}, {args.requests} requests each")

        async def sweep_then_close() -> list[dict[str, Any]]:
            # The client is closed in the loop that used it.
            try:
                return await run_sweep(
                    sender, shuffled, levels, args.requests, max_tokens=args.max_tokens,
                    p95_limit_s=args.p95_limit, warmup=args.warmup, clock=clock, out=out,
                )
            finally:
                close = getattr(sender, "aclose", None)
                if close is not None:
                    await close()

        sweep = asyncio.run(sweep_then_close())
    except BenchError as exc:
        out(f"error: {exc}")
        return EXIT_ERROR

    operating = operating_point(sweep, args.p95_limit)
    detected = engine_probe(args.base_url)
    gpu = dict(gpu_probe())
    engine = args.engine or detected.get("engine")
    version = args.engine_version or detected.get("version")
    token_means = [lv["mean_output_tokens"] for lv in sweep if lv["mean_output_tokens"] is not None]
    prompt_means = [lv["mean_prompt_tokens"] for lv in sweep if lv["mean_prompt_tokens"] is not None]
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now().astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "gpu": args.gpu_name or gpu.get("name") or args.gpu,
        "gpu_label": args.gpu,
        "gpu_memory_mib": gpu.get("memory_mib"),
        "gpu_driver": gpu.get("driver"),
        "system": args.system or args.model,
        "model": args.model,
        "engine": engine,
        "engine_version": version,
        "vllm_version": version if engine == "vllm" else None,
        "dtype": args.dtype,
        "base_url": args.base_url,
        "workload": {
            "prompts_file": str(args.prompts),
            "prompts_sha256": sha256_file(args.prompts),
            "n_prompts": len(prompts),
            "requests_per_level": args.requests,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "seed": args.seed,
            "warmup_requests": args.warmup,
            "mean_prompt_tokens": sum(prompt_means) / len(prompt_means) if prompt_means else None,
            "mean_output_tokens": sum(token_means) / len(token_means) if token_means else None,
        },
        "p95_limit_s": args.p95_limit,
        "levels": sweep,
        "operating_point": operating,
        "operating_point_note": None if operating else (
            f"no concurrency level had p95 latency <= {args.p95_limit:g} s with no failed request"
        ),
        "cost": selfhost_costs(rentals, args.price_key, sweep, operating),
        "extra": extra,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, out_path)
    out(f"operating point: {operating['concurrency'] if operating else 'none'}; wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
