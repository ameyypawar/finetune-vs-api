"""The retrieval index: train-only, deterministic, cached by the train file's hash."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import ex, fake_embed
from finetune_vs_api import config
from finetune_vs_api.retrieval import MODEL_NAME, LeakageError, RetrievalIndex, content_sha

TRAIN = [
    ex(1, "train", "alarm_set", "wake me up at six", [("time", "six")]),
    ex(2, "train", "alarm_set", "set an alarm for seven", [("time", "seven")]),
    ex(3, "train", "play_music", "play some jazz", [("music_genre", "jazz")]),
    ex(4, "train", "play_music", "play the beatles", [("artist_name", "the beatles")]),
    ex(5, "train", "weather_query", "what is the weather today", [("date", "today")]),
    ex(6, "train", "weather_query", "will it rain tomorrow", [("date", "tomorrow")]),
]


def build(train=TRAIN, **kw):
    kw.setdefault("embedder", fake_embed)
    return RetrievalIndex.build(train, **kw)


def test_the_most_similar_example_comes_first():
    index = build()
    top = index.topk("wake me up at seven", 3)
    assert [e.id for e in top][0] == "1"  # shares "wake me up at"
    assert index.topk("play jazz music", 1)[0].id == "3"
    assert [e.id for e in index.topk("what is the weather", 2)][0] == "5"


def test_it_returns_exactly_k_distinct_examples():
    top = build().topk("anything at all", 4)
    assert len(top) == 4 and len({e.id for e in top}) == 4


def test_k_must_be_between_one_and_the_size_of_the_index():
    index = build()
    for k in (0, -1, len(TRAIN) + 1):
        with pytest.raises(ValueError, match="k must be"):
            index.topk("x", k)
    assert len(index.topk("x", len(TRAIN))) == len(TRAIN)


def test_ties_break_towards_the_earlier_training_example():
    twins = [ex(10, "train", "a", "same words"), ex(11, "train", "a", "same words"), ex(12, "train", "a", "other")]
    assert [e.id for e in build(twins).topk("same words", 2)] == ["10", "11"]


def test_batch_and_single_queries_agree():
    index = build()
    queries = ["wake me up", "play the beatles", "will it rain"]
    assert index.topk_batch(queries, 3) == [index.topk(q, 3) for q in queries]


def test_unit_vectors_make_scores_cosine_similarities():
    index = build()
    assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1.0, atol=1e-5)


# --- train only ------------------------------------------------------------------------------------


def test_anything_but_a_train_example_is_refused():
    for bad in ("dev", "test"):
        with pytest.raises(LeakageError, match=f"1 non-train example.*9 \\({bad}\\)"):
            build([*TRAIN, ex(9, bad, "alarm_set", "wake me")])


def test_the_error_names_the_offenders_and_how_many_there_are():
    many = [ex(100 + i, "dev", "a", f"t{i}") for i in range(8)]
    with pytest.raises(LeakageError) as caught:
        build([*TRAIN, *many])
    assert "8 non-train" in str(caught.value) and "100 (dev)" in str(caught.value) and "105" not in str(caught.value)


def test_duplicate_ids_are_refused():
    with pytest.raises(LeakageError, match="duplicate ids"):
        build([*TRAIN, ex(1, "train", "a", "again")])


def test_dev_and_test_queries_only_ever_retrieve_train_examples():
    index = build()
    for query in ("wake me up at six", "play some jazz", "what is the weather today"):  # even exact train texts
        assert {e.split for e in index.topk(query, 6)} == {"train"}
    assert all(e in TRAIN for e in index.examples)


# --- caching --------------------------------------------------------------------------------------------


class Counting:
    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(len(texts))
        return fake_embed(texts)


def test_vectors_are_cached_under_the_train_files_hash(tmp_path):
    first = Counting()
    a = build(cache_dir=tmp_path, embedder=first, train_sha="a" * 64)
    assert first.calls == [6]
    second = Counting()
    b = build(cache_dir=tmp_path, embedder=second, train_sha="a" * 64)
    assert second.calls == []  # served from cache
    assert np.array_equal(a.vectors, b.vectors) and a.topk("wake me", 3) == b.topk("wake me", 3)
    assert len(list(tmp_path.glob("retrieval-*.npz"))) == 1


def test_a_new_train_hash_or_embedding_model_means_a_new_cache_entry(tmp_path):
    build(cache_dir=tmp_path, train_sha="a" * 64)
    fresh = Counting()
    build(cache_dir=tmp_path, embedder=fresh, train_sha="b" * 64)
    assert fresh.calls == [6]
    other_model = Counting()
    build(cache_dir=tmp_path, embedder=other_model, train_sha="a" * 64, model_name="some/other-model")
    assert other_model.calls == [6]
    assert len(list(tmp_path.glob("retrieval-*.npz"))) == 3


def test_a_cache_that_does_not_match_the_examples_is_not_used(tmp_path):
    build(cache_dir=tmp_path, train_sha="a" * 64)
    reordered = Counting()
    index = build(list(reversed(TRAIN)), cache_dir=tmp_path, embedder=reordered, train_sha="a" * 64)
    assert reordered.calls == [6]  # same hash but different ids/order: re-embedded, never mixed up
    assert [e.id for e in index.examples] == ["6", "5", "4", "3", "2", "1"]


def test_without_a_train_hash_the_content_hash_is_used(tmp_path):
    build(cache_dir=tmp_path)
    again = Counting()
    build(cache_dir=tmp_path, embedder=again)
    assert again.calls == []
    assert content_sha(TRAIN) == content_sha(list(TRAIN)) != content_sha(TRAIN[:-1])
    assert len(content_sha(TRAIN)) == 64


def test_embedding_is_done_in_batches():
    counting = Counting()
    build(embedder=counting, batch_size=4)
    assert counting.calls == [4, 2]


# --- the real embedder, when its model is already on disk ------------------------------------------------------------

model_cache = config.CACHE_DIR / "retrieval" / "fastembed"


@pytest.mark.skipif(not model_cache.exists(), reason="the bge-small model is not downloaded (a one-off ~65 MB)")
def test_the_real_embedder_gives_384_dimensional_vectors_for_similar_requests():
    from finetune_vs_api.retrieval import fastembed_embedder

    embed = fastembed_embedder(MODEL_NAME, cache_dir=model_cache)
    vectors = embed(["turn off the lights", "switch the lights off", "what is the capital of france"])
    assert vectors.shape == (3, 384)
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    assert unit[0] @ unit[1] > unit[0] @ unit[2]
