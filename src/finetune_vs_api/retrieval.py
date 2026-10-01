"""Nearest-neighbour retrieval of training examples, for the few-shot prompt.

The index is built from the TRAIN split and nothing else. `RetrievalIndex.build` refuses any
example whose split is not "train", so a dev or test request can be matched against the
training set but can never retrieve another dev or test example (or itself).

Embeddings come from fastembed's `BAAI/bge-small-en-v1.5` (ONNX, runs on CPU; the build fastembed fetches is about 65 MB).
Requests are compared with requests, so both sides use the plain `embed` method rather than
the query-instruction variant that is meant for question-to-passage search.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from .data import Example

MODEL_NAME = "BAAI/bge-small-en-v1.5"
Embedder = Callable[[Sequence[str]], np.ndarray]


class LeakageError(ValueError):
    """Something other than a train example was offered to the retrieval index."""


def fastembed_embedder(model_name: str = MODEL_NAME, cache_dir: Path | None = None) -> Embedder:
    """An embedder backed by fastembed. The model is downloaded on first use."""
    from fastembed import TextEmbedding  # imported here so tests need not install the model

    model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None)

    def embed(texts: Sequence[str]) -> np.ndarray:
        return np.asarray(list(model.embed(list(texts))), dtype=np.float32)

    return embed


def _unit(vectors: np.ndarray) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(norms == 0, 1.0, norms)


def content_sha(train: Sequence[Example]) -> str:
    """Hash of the training examples' ids and texts, for when no file hash is available."""
    digest = hashlib.sha256()
    for example in train:
        digest.update(f"{example.id}\t{example.text}\n".encode())
    return digest.hexdigest()


class RetrievalIndex:
    def __init__(self, examples: Sequence[Example], vectors: np.ndarray, embedder: Embedder):
        self.examples = list(examples)
        self.vectors = _unit(vectors)
        self._embedder = embedder

    @classmethod
    def build(
        cls,
        train: Sequence[Example],
        *,
        train_sha: str | None = None,
        cache_dir: Path | None = None,
        embedder: Embedder | None = None,
        model_name: str = MODEL_NAME,
        batch_size: int = 512,
    ) -> RetrievalIndex:
        """Embed the training examples (or load them from cache) and index them.

        `train_sha` is the sha256 of the train file the examples came from; with a
        `cache_dir`, vectors are cached under that hash and the model name, so the cache
        goes stale exactly when the training data or the embedding model changes.
        """
        offenders = [e for e in train if e.split != "train"]
        if offenders:
            sample = ", ".join(f"{e.id} ({e.split})" for e in offenders[:5])
            raise LeakageError(
                f"{len(offenders)} non-train example(s) offered to the retrieval index: {sample}"
            )
        ids = [e.id for e in train]
        if len(set(ids)) != len(ids):
            raise LeakageError("duplicate ids in the training examples")
        if embedder is None:
            embedder = fastembed_embedder(model_name, cache_dir=(cache_dir / "fastembed") if cache_dir else None)

        cache_file = None
        if cache_dir is not None:
            sha = train_sha or content_sha(train)
            slug = re.sub(r"[^A-Za-z0-9]+", "-", model_name).strip("-")
            cache_file = Path(cache_dir) / f"retrieval-{slug}-{sha[:16]}.npz"
            if cache_file.exists():
                with np.load(cache_file, allow_pickle=False) as cached:
                    if json.loads(str(cached["ids"])) == ids:
                        return cls(train, cached["vectors"], embedder)

        texts = [e.text for e in train]
        vectors = np.concatenate(
            [_unit(embedder(texts[i : i + batch_size])) for i in range(0, len(texts), batch_size)]
        )
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_file, vectors=vectors, ids=np.array(json.dumps(ids)))
        return cls(train, vectors, embedder)

    def topk_batch(self, texts: Sequence[str], k: int) -> list[list[Example]]:
        """For each text, the k most similar training examples, most similar first.

        Ties break towards the earlier training example, so results are deterministic.
        """
        if not 0 < k <= len(self.examples):
            raise ValueError(f"k must be between 1 and {len(self.examples)}, got {k}")
        queries = _unit(self._embedder(list(texts)))
        results: list[list[Example]] = []
        for scores in queries @ self.vectors.T:
            order = np.lexsort((np.arange(len(scores)), -scores))[:k]
            results.append([self.examples[i] for i in order])
        return results

    def topk(self, text: str, k: int) -> list[Example]:
        return self.topk_batch([text], k)[0]
