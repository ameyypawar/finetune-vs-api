"""Helpers for the serving tests: a stub OpenAI-compatible server on a real local socket, an
httpx sender and a stand-in `openai` module for scripts/bench_throughput.py, and a small
snapshot of the repository (configs, processed data, adapters, a test lock) for the Kaggle
serving script. Nothing here touches the network beyond 127.0.0.1, or a GPU, or a model."""

from __future__ import annotations

import json
import re
import shutil
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from conftest import ROOT, build_processed
from finetune_vs_api import config, data
from finetune_vs_api.schema import target_json

STUB_JSON = '{"intent":"alarm_set","slots":[]}'


class StubChatServer:
    """A stdlib HTTP server that answers /v1/chat/completions like a minimal vLLM.

    `delay_s` is how long one request takes to serve; `capacity` is how many are served at
    once (the rest wait their turn, like requests queued on a saturated GPU); `fail_every=k`
    answers every k-th request with HTTP 500. `bodies` holds every request body, `max_inflight`
    the most requests being served at the same time. GET /health, /version (when `version` is
    given) and /v1/models answer as vLLM does.
    """

    def __init__(
        self,
        *,
        delay_s: float = 0.0,
        capacity: int | None = None,
        completion_tokens: int = 20,
        text: str = STUB_JSON,
        version: str | None = None,
        models: tuple[str, ...] = ("stub-model",),
        fail_every: int = 0,
        finish_reason: str = "stop",
    ):
        self.delay_s, self.completion_tokens, self.text = delay_s, completion_tokens, text
        self.version, self.models, self.fail_every, self.finish_reason = version, models, fail_every, finish_reason
        self.bodies: list[dict] = []
        self.max_inflight = 0
        self._inflight = 0
        self._lock = threading.Lock()
        self._slots = threading.Semaphore(capacity) if capacity else None
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # life cycle -----------------------------------------------------------------------------
    def __enter__(self) -> StubChatServer:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # silence the per-request log
                pass

            def _send(self, status: int, payload: dict | None) -> None:
                body = b"" if payload is None else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/health":
                    return self._send(200, None)
                if self.path == "/version" and owner.version:
                    return self._send(200, {"version": owner.version})
                if self.path == "/v1/models":
                    return self._send(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in owner.models]})
                self._send(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path != "/v1/chat/completions":
                    return self._send(404, {"error": "not found"})
                status, payload = owner._answer(body)
                self._send(status, payload)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        assert self._server is not None
        self._server.shutdown()
        self._server.server_close()

    @property
    def root(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    @property
    def url(self) -> str:
        return self.root + "/v1"

    # the answer -----------------------------------------------------------------------------
    def _answer(self, body: dict) -> tuple[int, dict]:
        with self._lock:
            self.bodies.append(body)
            number = len(self.bodies)
        if self.fail_every and number % self.fail_every == 0:
            return 500, {"error": {"message": "stub failure"}}
        if self._slots:
            self._slots.acquire()
        try:
            with self._lock:
                self._inflight += 1
                self.max_inflight = max(self.max_inflight, self._inflight)
            if self.delay_s:
                time.sleep(self.delay_s)
            with self._lock:
                self._inflight -= 1
        finally:
            if self._slots:
                self._slots.release()
        prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in body.get("messages", []))
        return 200, {
            "id": f"chatcmpl-{number}",
            "object": "chat.completion",
            "model": body.get("model", "stub-model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": self.text}, "finish_reason": self.finish_reason}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": self.completion_tokens,
                      "total_tokens": prompt_tokens + self.completion_tokens},
        }


class HttpxSender:
    """The benchmark's sender over httpx, for tests that run without the `openai` package."""

    def __init__(self, base_url: str, model: str, *, timeout: float = 30.0):
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)
        self._model = model

    async def __call__(self, messages, max_tokens):
        response = await self._client.post(
            "/chat/completions",
            json={"model": self._model, "messages": messages, "max_tokens": max_tokens, "temperature": 0},
        )
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage") or {}
        return types.SimpleNamespace(
            completion_tokens=usage.get("completion_tokens"),
            prompt_tokens=usage.get("prompt_tokens"),
            finish_reason=body["choices"][0]["finish_reason"],
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def install_fake_openai(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Put a stand-in `openai` module in sys.modules: `AsyncOpenAI(...).chat.completions.create(...)`
    and `.close()`, backed by httpx, returning objects with the attributes the real client's
    responses have (`choices[0].message.content`, `choices[0].finish_reason`, `usage.*`).
    Returns the list that records every constructor call's keyword arguments."""
    constructed: list[dict] = []

    class Completions:
        def __init__(self, client: httpx.AsyncClient):
            self._client = client

        async def create(self, *, model, messages, max_tokens=None, temperature=None, **extra):
            body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, **extra}
            response = await self._client.post("/chat/completions", json=body)
            response.raise_for_status()
            payload = response.json()
            choice = payload["choices"][0]
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content=choice["message"]["content"]),
                    finish_reason=choice["finish_reason"],
                )],
                usage=types.SimpleNamespace(**payload["usage"]),
            )

    class AsyncOpenAI:
        def __init__(self, *, base_url, api_key, timeout=None, max_retries=2, **kwargs):
            constructed.append({"base_url": base_url, "api_key": api_key, "timeout": timeout, "max_retries": max_retries, **kwargs})
            self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)
            self.chat = types.SimpleNamespace(completions=Completions(self._client))

        async def close(self) -> None:
            await self._client.aclose()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=AsyncOpenAI))
    return constructed


# --- a snapshot of the repository, as the serving script finds it on Kaggle ------------------------------

FT = "ft-qwen3-4b-lora"
BASE = "base-qwen3-4b-k10"
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "0123456789abcdef0123456789abcdef01234567"
TRAIN_KERNEL = "finetune-vs-api-train"
SNAPSHOT = "finetune-vs-api-snapshot"
LOCKED_SUBSETS = ("full", "S500", "S300")


def write_adapter(directory, *, rank: int = 16, payload: bytes = b"weights") -> None:
    """A stand-in for one saved LoRA adapter: the two files a PEFT adapter directory holds."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "r": rank, "lora_alpha": 2 * rank, "use_dora": False, "use_rslora": False,
        "lora_bias": False, "bias": "none", "modules_to_save": None, "base_model_name_or_path": BASE_MODEL,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }))
    (directory / "adapter_model.safetensors").write_bytes(payload)


class Lab:
    """A tmp-dir stand-in for what Kaggle mounts: the repository snapshot (code, configs, processed data,
    a lock file), the training kernel's output (adapters, train_log.json), and /kaggle/working.

    The data is the synthetic split of conftest.build_processed. `make_handler()` is a stub
    chat-completions endpoint that answers each request from the gold label of the example it is
    about; `mark_wrong(model, ids)` makes it answer those items wrongly when asked as that model
    name, so different epochs score differently by a known amount.
    """

    def __init__(self, tmp_path, monkeypatch, *, epochs: int = 2):
        self.tmp = tmp_path
        self.input = tmp_path / "input"
        self.working = tmp_path / "working"
        self.scratch = tmp_path / "scratch"
        self.snapshot = self.input / SNAPSHOT
        self.train_output = self.input / TRAIN_KERNEL
        self.snapshot.mkdir(parents=True)
        monkeypatch.setattr(sys, "path", list(sys.path))  # the script puts the snapshot's src/ on sys.path
        ignore = shutil.ignore_patterns("__pycache__", "*.egg-info")
        shutil.copytree(ROOT / "src", self.snapshot / "src", ignore=ignore)
        shutil.copytree(ROOT / "scripts", self.snapshot / "scripts", ignore=ignore)
        shutil.copytree(ROOT / "configs", self.snapshot / "configs")
        self.processed = build_processed(self.snapshot / "data")
        (self.snapshot / "results").mkdir()
        self.config_dir = self.snapshot / "configs"
        # the state before dev-select: the shipped config pins the chosen checkpoint; these tests pin it themselves
        self.edit_re("systems.yaml", r"(?m)^(      adapter: )\S+", r"\g<1>null")
        self.edit_re("systems.yaml", r"(?m)^(      epoch: )\S+", r"\g<1>null")
        self.lock_path = self.snapshot / "results" / "test_lock.jsonl"
        self.epochs = list(range(1, epochs + 1))
        for epoch in self.epochs:
            write_adapter(self.train_output / "adapters" / f"epoch-{epoch}", payload=f"epoch-{epoch}".encode())
        (self.train_output / "train_log.json").write_text(json.dumps({
            "base_model": BASE_MODEL, "base_revision": REVISION, "adapters": [{"epoch": e} for e in self.epochs],
        }))
        self.pin_revision(REVISION)
        self.splits = {s: data.read_examples(self.processed / f"{s}.jsonl") for s in ("train", "dev", "test")}
        self.by_text = {e.text: e for rows in self.splits.values() for e in rows}
        self.wrong: dict[str, set[str]] = {}
        self.bodies: list[dict] = []
        monkeypatch.setattr(config, "git_commit", lambda cwd=None: None)
        monkeypatch.setattr(config, "git_dirty", lambda cwd=None: True)

    # configuration the user edits between the two modes ------------------------------------------
    def edit(self, filename: str, old: str, new: str) -> None:
        path = self.config_dir / filename
        text = path.read_text()
        assert old in text, f"{old!r} not in {filename}"
        path.write_text(text.replace(old, new, 1))

    def edit_re(self, filename: str, pattern: str, repl: str) -> None:
        """Replace every match of `pattern`; at least one must exist."""
        path = self.config_dir / filename
        text, n = re.subn(pattern, repl, path.read_text())
        assert n, f"{pattern!r} not in {filename}"
        path.write_text(text)

    def pin_revision(self, revision: str) -> None:
        """Set the base-model revision in train.yaml, whatever it is now (pinned or null)."""
        self.edit_re("train.yaml", r"(?m)^(\s*revision: )\S+", rf"\g<1>{revision}")

    def pin_checkpoint(self, epoch: int = 2, revision: str = REVISION) -> None:
        """What the user does after dev-select: fill the two checkpoint blocks in systems.yaml."""
        self.edit_re("systems.yaml", r"(?m)^(      adapter: ).*$", rf"\g<1>adapters/epoch-{epoch}")
        self.edit_re("systems.yaml", r"(?m)^(      epoch: ).*$", rf"\g<1>{epoch}")
        self.edit_re("systems.yaml", r"(?m)^(      base_revision: ).*$", rf"\g<1>{revision}")  # both local rows

    def lock_all(self, subsets=LOCKED_SUBSETS, systems=(FT, BASE)) -> None:
        for system in systems:
            for subset in subsets:
                config.write_test_lock(system, subset, "frozen before the test run", config_dir=self.config_dir,
                                       processed_dir=self.processed, lock_path=self.lock_path)

    def write_dev_select(self, *, chosen: int, adapters: dict | None = None, path: str = "vllm-lora") -> None:
        record = {"chosen": {"epoch": chosen}, "serving": {"path": path},
                  "adapters": adapters if adapters is not None else {str(e): {"sha256": self.adapter_sha(e)} for e in self.epochs}}
        target = self.snapshot / "results" / "serving" / "dev_select.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record))

    def adapter_sha(self, epoch: int) -> str:
        import hashlib

        return hashlib.sha256((self.train_output / "adapters" / f"epoch-{epoch}" / "adapter_model.safetensors").read_bytes()).hexdigest()

    # the stub endpoint -----------------------------------------------------------------------------
    def mark_wrong(self, model: str, ids) -> None:
        self.wrong.setdefault(model, set()).update(ids)

    def make_handler(self, mode: str = "normal", *, normal_first: int = 0):
        """A chat-completions stub. mode "garbage" answers "!!!!..." and runs to the token limit, like a
        model whose logits went to NaN; "empty" answers nothing. `normal_first` requests are answered
        properly before that starts, so a probe can pass and the run itself still go wrong."""
        seen = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.bodies.append(body)
            seen["n"] += 1
            example = self.by_text[body["messages"][-1]["content"]]
            finish = "stop"
            if mode != "normal" and seen["n"] > normal_first:
                text, finish = ("!" * 64, "length") if mode == "garbage" else ("", "stop")
            elif example.id in self.wrong.get(body["model"], ()):
                text = json.dumps({"intent": "weather_query" if example.intent != "weather_query" else "alarm_set", "slots": []})
            else:
                text = target_json(example)
            return httpx.Response(200, json={
                "id": "chatcmpl-1", "object": "chat.completion", "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
            })

        return handler

    def eval_kwargs(self) -> dict:
        """What run_eval needs besides the endpoint: a fake embedder and token counter, no bootstrap."""
        from conftest import fake_embed
        from stubs import words

        return {"embedder": fake_embed, "counter": words, "environ": {"UNRELATED": "x"}, "with_ci": False}

    def dev_ids(self, subset: str = "D100") -> list[str]:
        doc = json.loads((self.processed / "subsets.json").read_text())
        return list(doc["subsets"][subset]["ids"])
