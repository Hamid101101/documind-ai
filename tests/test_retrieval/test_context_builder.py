
from types import SimpleNamespace

import pytest

from retrieval.config import RetrievalConfig
from retrieval.context_builder import ContextBuilder


class FakeTokenizer:
    """Simple tokenizer for isolated ContextBuilder tests."""

    def encode(self, text, add_special_tokens=False):
        return text.split()


class FakeEmbeddingGenerator:
    """Avoid loading the real embedding model during tests."""

    def __init__(self):
        self.model = SimpleNamespace(tokenizer=FakeTokenizer())


@pytest.fixture
def context_builder():
    return ContextBuilder(
        embedding_generator=FakeEmbeddingGenerator(),
        config=RetrievalConfig(max_context_tokens=4000),
    )


@pytest.fixture
def retrieval_results():
    return [
        {
            "id": 1,
            "document_id": "doc-1",
            "document_filename": "handbook.pdf",
            "chunk_index": 0,
            "content": "Machine learning extracts patterns from data.",
            "page_start": 4,
            "page_end": 4,
            "similarity": 0.92,
        },
        {
            "id": 2,
            "document_id": "doc-1",
            "document_filename": "handbook.pdf",
            "chunk_index": 1,
            "content": "Deep learning uses neural networks.",
            "page_start": 5,
            "page_end": 6,
            "similarity": 0.87,
        },
    ]


def test_build_context(context_builder, retrieval_results):
    result = context_builder.build(retrieval_results)

    assert result["included_chunks"] == 2
    assert len(result["sources"]) == 2
    assert "Machine learning extracts patterns" in result["context"]
    assert "Deep learning uses neural networks" in result["context"]


def test_preserve_source_references(context_builder, retrieval_results):
    result = context_builder.build(retrieval_results)

    first_source = result["sources"][0]
    second_source = result["sources"][1]

    assert first_source["document_filename"] == "handbook.pdf"
    assert first_source["page_start"] == 4
    assert first_source["page_end"] == 4

    assert second_source["page_start"] == 5
    assert second_source["page_end"] == 6


def test_use_custom_token_budget(context_builder, retrieval_results):
    result = context_builder.build(
        retrieval_results,
        max_tokens=10,
    )

    assert result["total_tokens"] <= 10
    assert result["included_chunks"] < 2


def test_skip_chunks_exceeding_budget(context_builder, retrieval_results):
    first_chunk = retrieval_results[0]

    first_chunk_tokens = context_builder._count_tokens(
        "[Source 1: handbook.pdf, Page 4]\n"
        "Machine learning extracts patterns from data."
    )

    result = context_builder.build(
        retrieval_results,
        max_tokens=first_chunk_tokens,
    )

    assert result["included_chunks"] == 1
    assert result["sources"][0]["chunk_id"] == first_chunk["id"]


def test_empty_results(context_builder):
    result = context_builder.build([])

    assert result["context"] == ""
    assert result["sources"] == []
    assert result["total_tokens"] == 0
    assert result["included_chunks"] == 0


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_reject_invalid_token_budget(context_builder, max_tokens):
    with pytest.raises(ValueError, match="greater than zero"):
        context_builder.build([], max_tokens=max_tokens)


def test_use_default_token_budget(retrieval_results):
    config = RetrievalConfig(max_context_tokens=100)

    builder = ContextBuilder(
        embedding_generator=FakeEmbeddingGenerator(),
        config=config,
    )

    result = builder.build(retrieval_results)

    assert result["total_tokens"] <= 100
