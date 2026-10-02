"""A minimal OpenAI-compatible chat server around transformers: the accuracy-only serving path.

    python -m finetune_vs_api.hf_server --model-dir DIR_OR_REPO --served-name NAME [--revision R]
                                        [--host 127.0.0.1] [--port 8000]

It serves one model, decodes greedily, and answers one request at a time (requests wait for each
other). It exists so that an evaluation can still be run, through the same client, prompts and
scoring, when no real serving engine works on the machine (kaggle/serve/serve_eval_on_kaggle.py,
the last of its fallbacks). How fast it is says nothing about how the model would be served, so
nothing is ever benchmarked through it.

Only the standard library is imported up front; torch and transformers are imported when the model
is loaded, in a process of its own, so nothing installed or imported elsewhere can disturb them.
"""

from __future__ import annotations

import argparse
import json
import threading
from collections.abc import Callable, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

#: (messages, max_tokens) -> (text, prompt_tokens, completion_tokens, finish_reason)
Generate = Callable[[list[dict[str, str]], int], tuple[str, int, int, str]]
DEFAULT_MAX_TOKENS = 256


def make_server(generate: Generate, served_name: str, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """A server (not yet started) answering GET /health, GET /v1/models and POST /v1/chat/completions for the
    model name `served_name`. `generate` is called under a lock: one request at a time."""
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:  # one line per request would bury the log
            pass

        def _send(self, status: int, payload: dict[str, Any] | None = None) -> None:
            body = b"" if payload is None else json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._send(status, {"error": {"message": message}})

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path == "/health":
                return self._send(200)
            if path == "/v1/models":
                return self._send(200, {"object": "list", "data": [{"id": served_name, "object": "model"}]})
            self._error(404, f"no such path {path!r}")

        def do_POST(self) -> None:
            if self.path.split("?")[0] != "/v1/chat/completions":
                return self._error(404, f"no such path {self.path!r}")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                messages = body["messages"]
                if not messages or not all(isinstance(m, dict) and isinstance(m.get("role"), str) and isinstance(m.get("content"), str) for m in messages):
                    raise ValueError("messages must be a non-empty list of {role, content} strings")
                limit = int(body.get("max_tokens") or body.get("max_completion_tokens") or DEFAULT_MAX_TOKENS)
                if limit < 1:
                    raise ValueError("max_tokens must be positive")
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                return self._error(400, f"bad request: {exc}")
            if body.get("model") != served_name:
                return self._error(404, f"unknown model {body.get('model')!r}; this server serves {served_name!r}")
            try:
                with lock:
                    text, prompt_tokens, completion_tokens, finish = generate(messages, limit)
            except Exception as exc:  # a model error is the client's to see, and the server stays up
                return self._error(500, f"{type(exc).__name__}: {exc}")
            self._send(200, {
                "id": "chatcmpl-hf", "object": "chat.completion", "model": served_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens},
            })

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def load_generate(model_dir: str, revision: str | None = None) -> Generate:
    """Load the model in fp16 on the GPU with transformers and return a greedy `generate`."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    options = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(model_dir, **options)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16, **options)
    except TypeError:  # transformers before the dtype= spelling
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, **options)
    model = model.to("cuda").eval()

    def generate(messages: list[dict[str, str]], max_tokens: int) -> tuple[str, int, int, str]:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        # temperature, top_p and top_k are cleared so the model's generation_config.json does not turn sampling back on
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, temperature=None, top_p=None, top_k=None)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        new = output[0, prompt_tokens:]
        completion_tokens = int(new.shape[0])
        return tokenizer.decode(new, skip_special_tokens=True), prompt_tokens, completion_tokens, "length" if completion_tokens >= max_tokens else "stop"

    return generate


def main(
    argv: Sequence[str] | None = None, *, loader: Callable[[str, str | None], Generate] = load_generate, block: bool = True
) -> ThreadingHTTPServer | None:
    """Load the model, then serve until killed. With block=False the started server is returned instead (tests)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model-dir", required=True, help="a directory of weights, or a Hugging Face repo id")
    parser.add_argument("--revision", help="the commit to load when --model-dir is a repo id")
    parser.add_argument("--served-name", required=True, help="the only model name this server answers to")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    generate = loader(args.model_dir, args.revision)
    server = make_server(generate, args.served_name, args.host, args.port)
    print(f"serving {args.served_name} on http://{args.host}:{server.server_address[1]}/v1", flush=True)
    if not block:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server
    server.serve_forever()
    return None


if __name__ == "__main__":
    main()
