
import numpy as np
import pytest

from ingestion.embeddings.embedding_generator import EmbeddingGenerator


def test_embed_chunks_returns_one_embedding_per_chunk(
    embedder,
    make_chunks,
):
    chunks = make_chunks(["First document", "Second document"])

    result = embedder.embed_chunks(chunks)

    assert len(result) == 2
    assert all(len(vector) == 384 for vector in result)


def test_embed_chunks_returns_python_lists(embedder, make_chunks):
    chunks = make_chunks(["First document"])

    result = embedder.embed_chunks(chunks)

    assert isinstance(result, list)
    assert isinstance(result[0], list)
    assert all(isinstance(value, float) for value in result[0])


def test_embed_chunks_adds_passage_prefix(
    embedder,
    fake_model,
    make_chunks,
):
    chunks = make_chunks(["Machine learning"])

    embedder.embed_chunks(chunks)

    assert fake_model.received_texts == [
        "passage: Machine learning"
    ]


def test_embed_chunks_preserves_input_order(
    embedder,
    fake_model,
    make_chunks,
):
    chunks = make_chunks(["First", "Second", "Third"])

    embedder.embed_chunks(chunks)

    assert fake_model.received_texts == [
        "passage: First",
        "passage: Second",
        "passage: Third",
    ]


def test_embed_chunks_uses_configured_batch_size(
    embedder,
    fake_model,
    make_chunks,
):
    chunks = make_chunks(["First", "Second"])

    embedder.embed_chunks(chunks)

    assert fake_model.received_batch_size == 2


def test_embed_chunks_uses_configured_normalization(
    embedder,
    fake_model,
    make_chunks,
):
    chunks = make_chunks(["First"])

    embedder.embed_chunks(chunks)

    assert fake_model.received_normalization is True


def test_embed_chunks_handles_empty_input(embedder):
    assert embedder.embed_chunks([]) == []


def test_embed_chunks_rejects_wrong_dimension(
    embedding_config,
    make_chunks,
):
    from tests.test_embeddings.conftest import FakeEmbeddingModel

    model = FakeEmbeddingModel(dimension=128)
    embedder = EmbeddingGenerator(
        config=embedding_config,
        model=model,
    )

    with pytest.raises(ValueError, match="Expected embeddings"):
        embedder.embed_chunks(make_chunks(["Test document"]))


def test_embed_chunks_rejects_non_finite_values(
    embedding_config,
    make_chunks,
):
    class InvalidModel:
        def encode(self, sentences, **kwargs):
            result = np.ones((len(sentences), 384), dtype=np.float32)
            result[0, 0] = np.nan
            return result

    embedder = EmbeddingGenerator(
        config=embedding_config,
        model=InvalidModel(),
    )

    with pytest.raises(ValueError, match="invalid values"):
        embedder.embed_chunks(make_chunks(["Test document"]))


def test_model_is_not_loaded_during_initialization(embedding_config):
    embedder = EmbeddingGenerator(config=embedding_config)

    assert embedder._model is None
def test_embed_query_returns_384_dimensional_embedding(embedder):
    result = embedder.embed_query("What is machine learning?")

    assert isinstance(result, list)
    assert len(result) == 384
    assert all(isinstance(value, float) for value in result)


def test_embed_query_uses_query_prefix(embedder, fake_model):
    embedder.embed_query("What is machine learning?")

    assert fake_model.received_texts == [
        "query: What is machine learning?"
    ]


def test_embed_query_rejects_empty_query(embedder):
    with pytest.raises(ValueError, match="Query cannot be empty"):
        embedder.embed_query("")


def test_embed_query_rejects_whitespace_query(embedder):
    with pytest.raises(ValueError, match="Query cannot be empty"):
        embedder.embed_query("   ")
