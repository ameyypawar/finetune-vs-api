"""Probe each configured API endpoint once, and record what it actually does.

    python scripts/check_free_tiers.py [--endpoint NAME ...] [--model ID] [--dry-run]

For every model that a system in configs/systems.yaml uses on an endpoint with an API key,
this sends ONE small request carrying this repository's real strict JSON schema and records:

    the model id the server says it ran       whether the strict schema was accepted
    whether usage was reported (and which fields)   the x-ratelimit-* headers
    whether the answer was valid for the schema      latency

If the strict schema is rejected (HTTP 400 or 422) one plain request follows, so the other
facts are still recorded. Nothing is retried: a probe must not burn quota. For GitHub Models
it also lists the model catalog with each model's tier.

Results go to results/free_tiers/<endpoint>.json, merged by model, so a later run keeps the
entries it did not re-check. Use them to confirm configs/systems.yaml and to fill the
`observed` fields in configs/sources.yaml. An endpoint whose key is not set in the
environment is skipped. --dry-run prints what would be sent and sends nothing.

Exit status: 0 every probe got an answer; 1 at least one did not; 2 nothing could be probed.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx

from finetune_vs_api import client, config, prompts, schema

PROBE_REQUEST = "wake me up at five am this week"
PROBE_PROMPT = "zeroshot_v1"
CATALOG_KEYS = ("id", "name", "publisher", "summary", "capabilities", "limits", "html_url")
SCHEMA_REJECTED = (400, 422)


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def usage_summary(usage: Any) -> dict[str, Any]:
    return {
        "usage_reported": bool(usage.reported),
        "usage_fields_present": [
            name
            for name in ("prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens")
            if getattr(usage, name) is not None
        ],
        "usage": usage.to_dict(),
    }


async def probe_once(
    endpoint: client.Endpoint,
    messages: list[dict[str, str]],
    response_format: dict[str, Any] | None,
    inventory: schema.LabelInventory,
    transport: httpx.AsyncBaseTransport | None,
    environ: Mapping[str, str] | None,
) -> dict[str, Any]:
    """One request, no retries. Returns what happened, whether or not it succeeded."""
    attempt: dict[str, Any] = {
        "kind": "strict_json_schema" if response_format is not None else "plain",
        "sent_at": now_iso(),
    }
    probe = client.ChatClient(endpoint, transport=transport, environ=environ, state_dir=None)
    try:
        async with probe:
            done = await probe.complete(messages, response_format=response_format)
    except client.ApiStatusError as exc:
        attempt.update(status=exc.status, error=exc.body[:300], ratelimit_headers=client.ratelimit_headers(exc.headers))
    except client.RetriesExhausted as exc:  # max_retries=0: the first 429 / 5xx / transport failure
        attempt.update(status=exc.last_status, error=exc.last_error[:300], ratelimit_headers=client.ratelimit_headers(exc.headers))
    except client.QuotaExhausted as exc:
        reset = datetime.fromtimestamp(exc.reset_at, UTC).isoformat(timespec="seconds") if exc.reset_at else None
        attempt.update(status=429, error=str(exc), quota_reset_at=reset)
    except client.MalformedResponse as exc:
        attempt.update(status=200, error=f"malformed response: {exc}")
    else:
        parsed = schema.parse_output(done.text)
        attempt.update(
            status=200,
            error=None,
            model_returned=done.model_returned,
            finish_reason=done.finish_reason,
            latency_s=round(done.latency_s, 4),
            ratelimit_headers=done.ratelimit_headers,
            answer_preview=(done.text or "")[:200],
            answer_valid_for_schema=parsed is not None and schema.is_schema_valid(parsed, inventory),
            **usage_summary(done.usage),
        )
    return attempt


async def check_model(
    spec: Mapping[str, Any],
    inventory: schema.LabelInventory,
    transport: httpx.AsyncBaseTransport | None,
    environ: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Probe one (endpoint, model): strict schema first, then a plain request if it was rejected."""
    base = client.Endpoint.from_system(spec)
    probe = client.Endpoint(
        name=base.name, base_url=base.base_url, model=base.model, api_key_env=base.api_key_env,
        params=base.params, drop_params=base.drop_params, reasoning_in_completion=base.reasoning_in_completion,
        max_retries=0, supports_json_schema=True,
    )
    messages = prompts.render_messages(PROBE_PROMPT, PROBE_REQUEST, inventory=inventory)
    fmt = schema.response_format(inventory.intents, inventory.slot_types)
    entry: dict[str, Any] = {
        "model_requested": spec["model"],
        "tier": spec["tier"],
        "checked_at": now_iso(),
        "probe": {"prompt": PROBE_PROMPT, "request": PROBE_REQUEST},
    }
    attempts = [await probe_once(probe, messages, fmt, inventory, transport, environ)]
    first = attempts[0]
    if first["status"] == 200:
        entry["strict_json_schema_accepted"] = True
    elif first["status"] in SCHEMA_REJECTED:
        entry["strict_json_schema_accepted"] = False
        plain = dataclasses.replace(probe, supports_json_schema=False)
        attempts.append(await probe_once(plain, messages, None, inventory, transport, environ))
    else:
        entry["strict_json_schema_accepted"] = None  # no verdict: auth, quota, outage or missing model
    answered = next((a for a in attempts if a["status"] == 200), None)
    entry["attempts"] = attempts
    entry["answered"] = answered is not None
    if answered:
        entry.update(
            model_returned=answered["model_returned"],
            usage_reported=answered["usage_reported"],
            usage_fields_present=answered["usage_fields_present"],
            ratelimit_headers=answered["ratelimit_headers"],
        )
    else:
        entry["ratelimit_headers"] = attempts[-1].get("ratelimit_headers", {})
    return entry


def parse_catalog(payload: Any) -> list[dict[str, Any]] | None:
    """Catalog entries with their tier, from a list or a {models: [...]} object; None if unreadable."""
    items = payload.get("models") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return None
    models = []
    for item in items:
        if not isinstance(item, dict):
            continue
        entry = {key: item[key] for key in CATALOG_KEYS if key in item}
        entry["tier"] = item.get("rate_limit_tier", item.get("tier"))
        models.append(entry)
    return models


async def fetch_catalog(
    url: str, token: str | None, transport: httpx.AsyncBaseTransport | None
) -> dict[str, Any]:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    out: dict[str, Any] = {"url": url, "checked_at": now_iso()}
    try:
        async with httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(30.0), transport=transport) as http:
            response = await http.get(url)
    except httpx.HTTPError as exc:
        return {**out, "status": None, "error": f"{type(exc).__name__}: {exc}"}
    out["status"] = response.status_code
    if response.status_code != 200:
        return {**out, "error": response.text[:300]}
    try:
        models = parse_catalog(response.json())
    except ValueError:
        models = None
    if models is None:
        return {**out, "error": "unrecognized catalog format", "sample": response.text[:300]}
    tiers: dict[str, int] = {}
    for model in models:
        tiers[str(model["tier"])] = tiers.get(str(model["tier"]), 0) + 1
    return {**out, "error": None, "n_models": len(models), "models_per_tier": tiers, "models": models}


def plan(config_dir: Path | None, only: list[str] | None, extra_model: str | None) -> dict[str, list[dict[str, Any]]]:
    """endpoint name -> one resolved system spec per distinct model on that endpoint."""
    by_endpoint: dict[str, dict[str, dict[str, Any]]] = {}
    for name in config.list_systems(config_dir):
        spec = config.resolve_system(name, config_dir)
        if not spec["api_key_env"] or (only and spec["endpoint"] not in only):
            continue
        by_endpoint.setdefault(spec["endpoint"], {}).setdefault(spec["model"], spec)
    if extra_model:
        if not only or len(only) != 1 or only[0] not in by_endpoint:
            raise ValueError("--model needs exactly one --endpoint that has a system using it")
        template = next(iter(by_endpoint[only[0]].values()))
        by_endpoint[only[0]].setdefault(extra_model, {**template, "model": extra_model, "tier": None})
    return {endpoint: list(models.values()) for endpoint, models in by_endpoint.items()}


def merge_into(path: Path, update: dict[str, Any]) -> dict[str, Any]:
    """Write `update` over the file at `path`, keeping model entries it does not mention."""
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            existing = {}
    merged = {**existing, **{k: v for k, v in update.items() if k != "models"}}
    merged["models"] = {**existing.get("models", {}), **update["models"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return merged


def run(
    *,
    endpoints: list[str] | None = None,
    model: str | None = None,
    dry_run: bool = False,
    config_dir: Path | None = None,
    processed_dir: Path | None = None,
    results_dir: Path | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    environ: Mapping[str, str] | None = None,
    out: Callable[[str], None] = print,
) -> int:
    env = os.environ if environ is None else environ
    results = (results_dir or config.RESULTS_DIR) / "free_tiers"
    try:
        wanted = plan(config_dir, endpoints, model)
        inventory = schema.load_inventory(processed_dir or config.PROCESSED_DIR)
    except (ValueError, FileNotFoundError, config.ConfigError) as exc:
        out(f"error: {exc}")
        return 2
    if not wanted:
        out("no API endpoints to probe (none configured with an API key variable)")
        return 2

    async def go() -> int:
        probed = failed = 0
        for endpoint, specs in wanted.items():
            key_name = specs[0]["api_key_env"]
            token = env.get(key_name, "").strip()
            if not token:
                out(f"{endpoint}: {key_name} is not set; skipping")
                continue
            if dry_run:
                for spec in specs:
                    out(f"{endpoint}: would send 1 request to {spec['base_url']}/chat/completions, model {spec['model']}")
                if specs[0]["catalog_url"]:
                    out(f"{endpoint}: would list the catalog at {specs[0]['catalog_url']}")
                continue
            document: dict[str, Any] = {
                "endpoint": endpoint, "base_url": specs[0]["base_url"], "checked_at": now_iso(), "models": {},
            }
            for spec in specs:
                entry = await check_model(spec, inventory, transport, env)
                document["models"][spec["model"]] = entry
                probed += 1
                failed += 0 if entry["answered"] else 1
                out(
                    f"{endpoint} / {spec['model']}: "
                    + ("answered" if entry["answered"] else f"NO ANSWER (HTTP {entry['attempts'][-1]['status']})")
                    + f"; strict schema accepted: {entry['strict_json_schema_accepted']}"
                    + (f"; model returned: {entry['model_returned']}; usage reported: {entry['usage_reported']}" if entry["answered"] else "")
                )
                if entry["attempts"][-1].get("status") in (401, 403):
                    out(f"{endpoint}: the key was refused; not probing the remaining models")
                    break
            if specs[0]["catalog_url"]:
                catalog = await fetch_catalog(specs[0]["catalog_url"], token, transport)
                document["catalog"] = catalog
                out(
                    f"{endpoint}: catalog "
                    + (f"{catalog['n_models']} models, tiers {catalog['models_per_tier']}" if not catalog["error"] else f"unavailable ({catalog['error'][:80]})")
                )
            merge_into(results / f"{endpoint}.json", document)
            out(f"{endpoint}: wrote {results / (endpoint + '.json')}")
        if dry_run:
            out("dry run: nothing was sent")
            return 0
        if probed == 0:
            return 2
        return 1 if failed else 0

    return asyncio.run(go())


def main(argv: list[str] | None = None, **overrides: Any) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--endpoint", action="append", help="only this endpoint (repeatable)")
    parser.add_argument("--model", help="also probe this model id (needs exactly one --endpoint)")
    parser.add_argument("--dry-run", action="store_true", help="print what would be sent and send nothing")
    args = parser.parse_args(argv)
    if "environ" not in overrides:
        config.load_env()
    return run(endpoints=args.endpoint, model=args.model, dry_run=args.dry_run, **overrides)


if __name__ == "__main__":
    raise SystemExit(main())
