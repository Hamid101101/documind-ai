
import numpy as np
import pytest

from ingestion.chunking.base import DocumentChunk
from ingestion.embeddings.config import EmbeddingConfig
from ingestion.embeddings.embedding_generator import EmbeddingGenerator


class FakeEmbeddingModel:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.received_texts = []
        self.received_batch_size = None
        self.received_normalization = None

    def encode(
        self,
        sentences,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        self.received_texts = (
            [sentences] if isinstance(sentences, str) else list(sentences)
        )
        self.received_batch_size = batch_size
        self.received_normalization = normalize_embeddings

        if isinstance(sentences, str):
            return np.ones(self.dimension, dtype=np.float32)

        return np.ones(
            (len(sentences), self.dimension),
            dtype=np.float32,
        )


@pytest.fixture
def embedding_config():
    return EmbeddingConfig(
        model_name="test-model",
        batch_size=2,
        device="cpu",
        expected_dimension=384,
    )


@pytest.fixture
def fake_model():
    return FakeEmbeddingModel()


@pytest.fixture
def embedder(embedding_config, fake_model):
    return EmbeddingGenerator(
        config=embedding_config,
        model=fake_model,
    )


@pytest.fixture
def make_chunks():
    def _make_chunks(contents: list[str]) -> list[DocumentChunk]:
        return [
            DocumentChunk(
                chunk_index=index,
                content=content,
                page_start=None,
                page_end=None,
                token_count=len(content.split()),
                metadata={},
            )
            for index, content in enumerate(contents)
        ]

    return _make_chunks


def test_embed_chunks_rejects_wrong_dimension(
    embedding_config,
    make_chunks,
):
    class WrongDimensionModel:
        def encode(self, sentences, **kwargs):
            return np.ones((len(sentences), 128), dtype=np.float32)

    embedder = EmbeddingGenerator(
        config=embedding_config,
        model=WrongDimensionModel(),
    )

    with pytest.raises(ValueError, match="Expected embeddings"):
        embedder.embed_chunks(make_chunks(["Test document"]))
