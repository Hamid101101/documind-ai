
import pytest

from retrieval.config import RetrievalConfig
from retrieval.filters import RetrievalFilters
from retrieval.vector_search import VectorSearch


class FakeEmbeddingGenerator:
    def __init__(self):
        self.received_query = None

    def embed_query(self, query):
        self.received_query = query
        return [0.1] * 384


class FakeChunkRepository:
    def __init__(self):
        self.received_embedding = None
        self.received_top_k = None
        self.received_filters = None
        self.received_query_text = None
        self.received_keyword_top_k = None
        self.received_keyword_filters = None

        self.results = []
        self.keyword_results = []

    def search_similar(self, query_embedding, top_k, filters=None):
        self.received_embedding = query_embedding
        self.received_top_k = top_k
        self.received_filters = filters
        return self.results

    def search_keyword(self, query_text, top_k, filters=None):
        self.received_query_text = query_text
        self.received_keyword_top_k = top_k
        self.received_keyword_filters = filters
        return self.keyword_results

class FakeContextBuilder:
    def __init__(self):
        self.received_results = None
        self.received_max_tokens = None

    def build(self, results, max_tokens=None):
        self.received_results = results
        self.received_max_tokens = max_tokens

        return {
            "context": "Formatted context",
            "sources": [],
            "total_tokens": 100,
            "included_chunks": len(results),
        }


@pytest.fixture
def dependencies():
    embedding_generator = FakeEmbeddingGenerator()
    chunk_repository = FakeChunkRepository()
    context_builder = FakeContextBuilder()

    return embedding_generator, chunk_repository, context_builder


@pytest.fixture
def vector_search(dependencies):
    embedding_generator, chunk_repository, context_builder = dependencies

    return VectorSearch(
        embedding_generator=embedding_generator,
        chunk_repository=chunk_repository,
        context_builder=context_builder,
    )


def test_search_generates_query_embedding(vector_search, dependencies):
    embedding_generator, chunk_repository, _ = dependencies

    vector_search.search("What is machine learning?")

    assert embedding_generator.received_query == "What is machine learning?"
    assert chunk_repository.received_embedding == [0.1] * 384


def test_search_uses_default_top_k(vector_search, dependencies):
    _, chunk_repository, _ = dependencies

    vector_search.search("Test query")

    assert chunk_repository.received_top_k == 5


def test_search_accepts_custom_top_k(vector_search, dependencies):
    _, chunk_repository, _ = dependencies

    vector_search.search("Test query", top_k=10)

    assert chunk_repository.received_top_k == 10


def test_search_forwards_filters(vector_search, dependencies):
    _, chunk_repository, _ = dependencies

    filters = RetrievalFilters(
        document_metadata={"category": "research"}
    )

    vector_search.search("Test query", filters=filters)

    assert chunk_repository.received_filters == filters


@pytest.mark.parametrize("query", ["", " ", "\n"])
def test_search_rejects_empty_query(vector_search, dependencies, query):
    embedding_generator, chunk_repository, _ = dependencies

    with pytest.raises(ValueError, match="Query cannot be empty"):
        vector_search.search(query)

    assert embedding_generator.received_query is None
    assert chunk_repository.received_embedding is None


@pytest.mark.parametrize("top_k", [0, -1])
def test_search_rejects_invalid_top_k(vector_search, top_k):
    with pytest.raises(ValueError, match="top_k"):
        vector_search.search("Test query", top_k=top_k)


def test_search_rejects_top_k_above_maximum(vector_search):
    with pytest.raises(ValueError, match="cannot exceed"):
        vector_search.search("Test query", top_k=21)


def test_search_applies_similarity_threshold(dependencies):
    embedding_generator, chunk_repository, _ = dependencies

    chunk_repository.results = [
        {"content": "Relevant chunk", "similarity": 0.95},
        {"content": "Weakly relevant chunk", "similarity": 0.65},
        {"content": "Irrelevant chunk", "similarity": 0.30},
    ]

    search = VectorSearch(
        embedding_generator=embedding_generator,
        chunk_repository=chunk_repository,
        config=RetrievalConfig(similarity_threshold=0.7),
    )

    results = search.search("Test query")

    assert len(results) == 1
    assert results[0]["content"] == "Relevant chunk"


def test_search_keeps_all_results_without_threshold(dependencies):
    embedding_generator, chunk_repository, _ = dependencies

    chunk_repository.results = [
        {"content": "First chunk", "similarity": 0.95},
        {"content": "Second chunk", "similarity": 0.30},
    ]

    search = VectorSearch(
        embedding_generator=embedding_generator,
        chunk_repository=chunk_repository,
    )

    results = search.search("Test query")

    assert len(results) == 2

def test_search_with_context_assembles_retrieved_results(
    vector_search, dependencies
):
    embedding_generator, chunk_repository, context_builder = dependencies

    chunk_repository.results = [
        {"content": "First chunk", "similarity": 0.95},
        {"content": "Second chunk", "similarity": 0.85},
    ]

    result = vector_search.search_with_context("Test query")

    assert result["context"] == "Formatted context"
    assert result["total_tokens"] == 100
    assert result["included_chunks"] == 2

    assert context_builder.received_results == chunk_repository.results
    assert context_builder.received_max_tokens is None

def test_search_with_context_forwards_custom_max_tokens(
    vector_search, dependencies
):
    _, _, context_builder = dependencies

    vector_search.search_with_context(
        query="Test query",
        max_tokens=2000,
    )

    assert context_builder.received_max_tokens == 2000

def test_search_with_context_forwards_filters(
    vector_search, dependencies
):
    _, chunk_repository, _ = dependencies

    filters = RetrievalFilters(
        document_metadata={"category": "research"}
    )

    vector_search.search_with_context(
        query="Test query",
        filters=filters,
    )

    assert chunk_repository.received_filters == filters

def test_fuse_results_combines_rankings(vector_search):
    semantic_results = [
        {"id": 1, "content": "Chunk A", "similarity": 0.95},
        {"id": 2, "content": "Chunk B", "similarity": 0.85},
    ]

    keyword_results = [
        {"id": 2, "content": "Chunk B", "keyword_score": 0.9},
        {"id": 3, "content": "Chunk C", "keyword_score": 0.8},
    ]

    results = vector_search._fuse_results(
        semantic_results=semantic_results,
        keyword_results=keyword_results,
        top_k=3,
    )

    assert [result["id"] for result in results] == [2, 1, 3]
    assert results[0]["rrf_score"] == pytest.approx(
        (1 / 62) + (1 / 61)
    )


def test_fuse_results_respects_top_k(vector_search):
    semantic_results = [
        {"id": 1, "content": "Chunk A"},
        {"id": 2, "content": "Chunk B"},
        {"id": 3, "content": "Chunk C"},
    ]

    results = vector_search._fuse_results(
        semantic_results=semantic_results,
        keyword_results=[],
        top_k=2,
    )

    assert len(results) == 2
    assert [result["id"] for result in results] == [1, 2]


def test_fuse_results_handles_empty_results(vector_search):
    results = vector_search._fuse_results(
        semantic_results=[],
        keyword_results=[],
        top_k=5,
    )

    assert results == []


def test_hybrid_search_combines_semantic_and_keyword_results(
    vector_search, dependencies
):
    embedding_generator, chunk_repository, _ = dependencies

    chunk_repository.results = [
        {"id": 1, "content": "Semantic result A", "similarity": 0.95},
        {"id": 2, "content": "Semantic result B", "similarity": 0.85},
    ]

    chunk_repository.keyword_results = [
        {"id": 2, "content": "Keyword result B", "keyword_score": 0.9},
        {"id": 3, "content": "Keyword result C", "keyword_score": 0.8},
    ]

    results = vector_search.hybrid_search("PostgreSQL search")

    assert embedding_generator.received_query == "PostgreSQL search"
    assert chunk_repository.received_embedding == [0.1] * 384
    assert chunk_repository.received_query_text == "PostgreSQL search"

    assert [result["id"] for result in results] == [2, 1, 3]
    assert all("rrf_score" in result for result in results)


def test_hybrid_search_forwards_filters(vector_search, dependencies):
    _, chunk_repository, _ = dependencies

    filters = RetrievalFilters(
        document_metadata={"category": "research"}
    )

    vector_search.hybrid_search("Test query", filters=filters)

    assert chunk_repository.received_filters == filters
    assert chunk_repository.received_keyword_filters == filters

def test_hybrid_search_applies_similarity_threshold(dependencies):
    embedding_generator, chunk_repository, _ = dependencies

    chunk_repository.results = [
        {"id": 1, "content": "Relevant", "similarity": 0.95},
        {"id": 2, "content": "Weak", "similarity": 0.40},
    ]

    chunk_repository.keyword_results = []

    search = VectorSearch(
        embedding_generator=embedding_generator,
        chunk_repository=chunk_repository,
        config=RetrievalConfig(similarity_threshold=0.7),
    )

    results = search.hybrid_search("Test query")

    assert [result["id"] for result in results] == [1]

@pytest.mark.parametrize("query", ["", " ", "\n"])
def test_hybrid_search_rejects_empty_query(vector_search, query):
    with pytest.raises(ValueError, match="Query cannot be empty"):
        vector_search.hybrid_search(query)

def test_hybrid_search_with_context_assembles_results(
    vector_search, dependencies
):
    _, chunk_repository, context_builder = dependencies

    chunk_repository.results = [
        {"id": 1, "content": "Semantic chunk", "similarity": 0.95},
    ]

    chunk_repository.keyword_results = [
        {"id": 2, "content": "Keyword chunk", "keyword_score": 0.8},
    ]

    result = vector_search.hybrid_search_with_context("Test query")

    assert result["context"] == "Formatted context"
    assert result["included_chunks"] == 2

    assert context_builder.received_results is not None
    assert [item["id"] for item in context_builder.received_results] == [1, 2]

def test_hybrid_search_with_context_forwards_options(
    vector_search, dependencies
):
    _, chunk_repository, context_builder = dependencies

    filters = RetrievalFilters(
        document_metadata={"category": "research"}
    )

    vector_search.hybrid_search_with_context(
        query="Test query",
        filters=filters,
        top_k=3,
        max_tokens=2000,
    )

    assert chunk_repository.received_filters == filters
    assert chunk_repository.received_keyword_filters == filters
    assert chunk_repository.received_top_k == 3
    assert chunk_repository.received_keyword_top_k == 3
    assert context_builder.received_max_tokens == 2000