"""Repository paths, config loading, hashing, git helpers, and the test lock.

The test lock exists so the test split cannot be tuned against. Before a system may be run
with `--split test`, its configuration is hashed and locked together with the hash of the
test subset it will be run on, with a reason, in an append-only history. If anything that
affects results (prompt, schema, model, decoding, checkpoint, price entry, dataset) changes
afterwards, or a different subset is requested, `assert_test_allowed` refuses until the system
is locked again. The lock is per system, so API rows can be locked and run before the
fine-tuned model exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from . import prompts, schema, subsets

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
LOCK_PATH = RESULTS_DIR / "test_lock.jsonl"


def load_yaml(name: str, config_dir: Path | None = None) -> Any:
    """Parse `configs/<name>.yaml` (the `.yaml` suffix is optional)."""
    filename = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    with open((config_dir or CONFIG_DIR) / filename, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_env(path: Path | None = None) -> list[str]:
    """Load KEY=VALUE lines from `.env` into the environment without overriding anything.

    Returns the names that were set, never the values. A missing file is not an error.
    """
    env_path = path or (REPO_ROOT / ".env")
    if not env_path.exists():
        return []
    loaded = []
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def canonical_json(obj: Any) -> str:
    """Stable JSON for hashing: sorted keys, no whitespace, non-ASCII kept."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_of(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def _git(args: list[str], cwd: Path | None) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd or REPO_ROOT, capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return None


def git_commit(cwd: Path | None = None) -> str | None:
    """The current commit hash, or None.

    None covers every case where there is no answer: git is missing, this is not a
    repository, or the repository has no commits yet (`git rev-parse HEAD` fails there).
    """
    result = _git(["rev-parse", "--verify", "-q", "HEAD"], cwd)
    if result is None or result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_dirty(cwd: Path | None = None) -> bool | None:
    """True if the working tree has uncommitted changes, None if git has no answer."""
    result = _git(["status", "--porcelain"], cwd)
    if result is None or result.returncode != 0:
        return None
    return bool(result.stdout.strip())


# --- systems ---------------------------------------------------------------------------------


class ConfigError(ValueError):
    """A problem in configs/*.yaml."""


def list_systems(config_dir: Path | None = None) -> list[str]:
    return list(load_yaml("systems", config_dir)["systems"])


def resolve_system(name: str, config_dir: Path | None = None) -> dict[str, Any]:
    """One system's row merged with its endpoint and validated, as a flat dict."""
    doc = load_yaml("systems", config_dir)
    if name not in doc["systems"]:
        raise ConfigError(f"unknown system {name!r}; known systems: {sorted(doc['systems'])}")
    row = doc["systems"][name]
    endpoint_name = row.get("endpoint")
    if endpoint_name not in doc["endpoints"]:
        raise ConfigError(f"system {name!r} uses unknown endpoint {endpoint_name!r}")
    endpoint = doc["endpoints"][endpoint_name]
    for key in ("model", "prompt", "price_id", "params"):
        if key not in row:
            raise ConfigError(f"system {name!r} is missing {key!r} (use null for price_id on a local row)")

    system = {
        "name": name,
        "endpoint": endpoint_name,
        "base_url": endpoint["base_url"],
        "catalog_url": endpoint.get("catalog_url"),
        "api_key_env": endpoint.get("api_key_env"),
        "supports_json_schema": row.get("supports_json_schema", endpoint.get("supports_json_schema", False)),
        "reasoning_in_completion": row.get(
            "reasoning_in_completion", endpoint.get("reasoning_in_completion", True)
        ),
        "model": row["model"],
        "tier": row.get("tier"),
        "prompt": row["prompt"],
        "dev_prompts": list(row.get("dev_prompts") or []),
        "test_subset": row.get("test_subset", subsets.FULL),
        "price_id": row["price_id"],
        "checkpoint": row.get("checkpoint"),
        "params": dict(row["params"] or {}),
        "drop_params": list(row.get("drop_params") or []),
        "limits": dict(row.get("limits") or {}),
    }
    try:
        for prompt_name in [system["prompt"], *system["dev_prompts"]]:
            prompts.get_prompt(prompt_name)
    except KeyError as exc:
        raise ConfigError(f"system {name!r}: {exc.args[0]}") from None
    subset = system["test_subset"]
    if subset not in subsets.SUBSET_NAMES or (subset != subsets.FULL and subsets.SUBSET_PLAN[subset][0] != "test"):
        raise ConfigError(f"system {name!r}: test_subset must be full, S500 or S300, got {subset!r}")
    cap = system["limits"].get("max_output_tokens")
    requested = system["params"].get("max_tokens") or system["params"].get("max_completion_tokens")
    if cap and requested and requested > cap:
        raise ConfigError(f"system {name!r}: requests {requested} output tokens but the cap is {cap}")
    return system


# --- hashing what a result depends on ----------------------------------------------------------

#: Limits that change how fast a run goes, not what it measures. Left out of the hash.
OPERATIONAL_LIMITS = ("rpm", "rpd", "tpm", "tpd", "max_concurrency")


def config_components(
    system: str,
    *,
    prompt: str | None = None,
    inventory: schema.LabelInventory | None = None,
    config_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> dict[str, str]:
    """Hash of each part of a system's configuration that a result depends on.

    `prompt` is the prompt actually used. It defaults to the system's own, which is the only
    one allowed on the test split; dev runs may use another prompt the row allows.
    """
    spec = resolve_system(system, config_dir)
    prompt = prompt or spec["prompt"]
    inventory = inventory or schema.load_inventory(processed_dir or PROCESSED_DIR)
    sources = load_yaml("sources", config_dir)
    data_cfg = load_yaml("data", config_dir)
    price = sources["prices"].get(spec["price_id"]) if spec["price_id"] else None
    behavioural_limits = {k: v for k, v in spec["limits"].items() if k not in OPERATIONAL_LIMITS}
    return {
        "prompt": prompts.prompt_hash(prompt, inventory),
        "schema": hash_of(schema.json_schema(inventory.intents, inventory.slot_types)),
        "system": hash_of(
            {
                "endpoint": spec["endpoint"],
                "base_url": spec["base_url"],
                "model": spec["model"],
                "prompt": prompt,
                "limits": behavioural_limits,
            }
        ),
        "decoding": hash_of(
            {
                "params": spec["params"],
                "drop_params": spec["drop_params"],
                "supports_json_schema": spec["supports_json_schema"],
                "reasoning_in_completion": spec["reasoning_in_completion"],
            }
        ),
        "checkpoint": hash_of(spec["checkpoint"]),
        "prices": hash_of(price),
        "dataset": hash_of(
            {
                "url": data_cfg["source"]["url"],
                "sha256": data_cfg["source"]["sha256"],
                "locale": data_cfg["locale"],
            }
        ),
    }


def config_hash(system: str, **kwargs: Any) -> str:
    """One hash over `config_components`: prompt, schema, system, decoding, checkpoint, prices, dataset."""
    return hash_of(config_components(system, **kwargs))


# --- the test lock ------------------------------------------------------------------------------


class LockError(RuntimeError):
    """The test split may not be used: no matching lock, or the configuration moved since."""


def read_lock_history(lock_path: Path | None = None) -> list[dict[str, Any]]:
    path = lock_path or LOCK_PATH
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def lock_blockers(system: str, *, config_dir: Path | None = None) -> list[str]:
    """Reasons a system cannot be locked yet (an unset checkpoint, a missing price entry)."""
    spec = resolve_system(system, config_dir)
    blockers: list[str] = []
    if spec["price_id"] and spec["price_id"] not in load_yaml("sources", config_dir)["prices"]:
        blockers.append(f"price_id {spec['price_id']!r} has no entry in configs/sources.yaml")
    for key, value in (spec["checkpoint"] or {}).items():
        if value in (None, ""):
            blockers.append(f"checkpoint.{key} is not set in configs/systems.yaml")
    return blockers


def _test_subset_hash(subset: str, processed_dir: Path | None) -> str:
    doc = subsets.load_subsets((processed_dir or PROCESSED_DIR) / "subsets.json")
    if subset == subsets.FULL:
        return doc["full"]["test"]["hash"]
    if subset not in doc["subsets"]:
        raise LockError(f"unknown subset {subset!r}; choose from {list(subsets.SUBSET_NAMES)}")
    entry = doc["subsets"][subset]
    if entry["split"] != "test":
        raise LockError(f"subset {subset} is a {entry['split']} subset; the test lock covers test subsets")
    return entry["hash"]


def _now_iso(now: datetime | None) -> str:
    return (now or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_test_lock(
    system: str,
    subset: str | None,
    reason: str,
    *,
    config_dir: Path | None = None,
    processed_dir: Path | None = None,
    lock_path: Path | None = None,
    inventory: schema.LabelInventory | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append a lock entry for (system, test subset) and return it. History is never rewritten."""
    if not reason or not reason.strip():
        raise LockError("a reason is required")
    spec = resolve_system(system, config_dir)
    subset = subset or spec["test_subset"]
    blockers = lock_blockers(system, config_dir=config_dir)
    if blockers:
        raise LockError(f"cannot lock {system}: " + "; ".join(blockers))
    components = config_components(system, inventory=inventory, config_dir=config_dir, processed_dir=processed_dir)
    entry = {
        "lock_version": 1,
        "system": system,
        "subset": subset,
        "subset_hash": _test_subset_hash(subset, processed_dir),
        "config_hash": hash_of(components),
        "components": components,
        "reason": reason.strip(),
        "locked_at": _now_iso(now),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
    }
    path = lock_path or LOCK_PATH
    for existing in read_lock_history(path):
        same = all(existing.get(k) == entry[k] for k in ("system", "subset", "subset_hash", "config_hash"))
        if same:
            raise LockError(
                f"{system} is already locked for {subset} with this configuration "
                f"(locked at {existing['locked_at']}: {existing['reason']!r})"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def lock_status(
    system: str,
    subset: str | None = None,
    *,
    config_dir: Path | None = None,
    processed_dir: Path | None = None,
    lock_path: Path | None = None,
    inventory: schema.LabelInventory | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """(matching entry or None, explanation). A match needs the same system, subset, subset
    hash and configuration hash; any lock in the history that matches will do."""
    spec = resolve_system(system, config_dir)
    subset = subset or spec["test_subset"]
    components = config_components(system, inventory=inventory, config_dir=config_dir, processed_dir=processed_dir)
    current_hash = hash_of(components)
    current_subset = _test_subset_hash(subset, processed_dir)
    history = [e for e in read_lock_history(lock_path) if e["system"] == system]
    for entry in history:
        if (entry["subset"], entry["subset_hash"], entry["config_hash"]) == (subset, current_subset, current_hash):
            return entry, f"locked at {entry['locked_at']}: {entry['reason']}"

    fix = f"python scripts/lock_test.py --write --system {system} --subset {subset} --reason '...'"
    same_subset = [e for e in history if e["subset"] == subset]
    if not history:
        return None, f"{system} has never been locked. Lock it first: {fix}"
    if not same_subset:
        locked = sorted({e["subset"] for e in history})
        return None, f"{system} is locked for {locked} but not for {subset}. Lock it: {fix}"
    latest = same_subset[-1]
    changed = [k for k in components if components[k] != latest["components"].get(k)]
    if latest["subset_hash"] != current_subset:
        changed.append(f"subset {subset} contents")
    return None, (
        f"{system} changed since it was locked for {subset} (changed: {', '.join(changed) or 'unknown'}). "
        f"If the change is intended, lock it again with a reason: {fix}"
    )


def assert_test_allowed(system: str, subset: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Return the matching lock entry, or raise LockError. Called before any `--split test` run."""
    entry, why = lock_status(system, subset, **kwargs)
    if entry is None:
        raise LockError(f"the test split is locked for {system}: {why}")
    return entry
