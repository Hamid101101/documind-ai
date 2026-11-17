
from uuid import uuid4

import pytest

from database.repositories.chunk_repository import ChunkRepository
from database.repositories.document_repository import DocumentRepository
from ingestion.chunking.base import DocumentChunk

from retrieval.filters import RetrievalFilters

from ingestion.embeddings.embedding_generator import EmbeddingGenerator
from retrieval.config import RetrievalConfig
from retrieval.vector_search import VectorSearch


@pytest.fixture
def repositories():
    return {
        "documents": DocumentRepository(),
        "chunks": ChunkRepository(),
    }


@pytest.mark.integration
def test_vector_search_returns_most_similar_chunks(repositories):
    documents = repositories["documents"]
    chunks = repositories["chunks"]

    document = documents.create(
        filename="vector_search_test.txt",
        mime_type="text/plain",
        file_extension=".txt",
        file_size_bytes=100,
        file_checksum=None,
        storage_bucket="documind-documents",
        storage_object_key=f"integration-tests/{uuid4()}.txt",
        metadata={"test": "vector_search"},
    )

    document_id = document["id"]

    # Three normalized embeddings with known cosine similarities.
    embeddings = [
        [1.0] + [0.0] * 383,
        [0.8, 0.6] + [0.0] * 382,
        [0.0, 1.0] + [0.0] * 382,
    ]

    test_chunks = [
        DocumentChunk(
            chunk_index=0,
            content="Most relevant chunk",
            page_start=1,
            page_end=1,
            token_count=3,
            metadata={"test": "vector_search"},
        ),
        DocumentChunk(
            chunk_index=1,
            content="Second relevant chunk",
            page_start=2,
            page_end=2,
            token_count=3,
            metadata={"test": "vector_search"},
        ),
        DocumentChunk(
            chunk_index=2,
            content="Least relevant chunk",
            page_start=3,
            page_end=3,
            token_count=3,
            metadata={"test": "vector_search"},
        ),
    ]

    try:
        inserted = chunks.create_many(
            document_id=document_id,
            chunks=test_chunks,
            embeddings=embeddings,
        )

        assert inserted == 3

        results = chunks.search_similar(
            query_embedding=embeddings[0],
            top_k=3,
            filters=RetrievalFilters(document_ids=[document_id]),
        )

        assert len(results) == 3

        assert results[0]["content"] == "Most relevant chunk"
        assert results[1]["content"] == "Second relevant chunk"
        assert results[2]["content"] == "Least relevant chunk"

        assert results[0]["similarity"] == pytest.approx(1.0, abs=1e-5)
        assert results[1]["similarity"] == pytest.approx(0.8, abs=1e-5)
        assert results[2]["similarity"] == pytest.approx(0.0, abs=1e-5)

        for result in results:
            assert result["document_id"] == document_id
            assert result["page_start"] is not None
            assert result["page_end"] is not None

    finally:
        documents.delete_record(document_id)


@pytest.mark.integration
def test_vector_search_applies_metadata_filters(repositories):
    documents = repositories["documents"]
    chunks = repositories["chunks"]

    research_document = documents.create(
        filename="research_test.txt",
        mime_type="text/plain",
        file_extension=".txt",
        file_size_bytes=100,
        file_checksum=None,
        storage_bucket="documind-documents",
        storage_object_key=f"integration-tests/{uuid4()}.txt",
        metadata={"category": "research"},
    )

    finance_document = documents.create(
        filename="finance_test.txt",
        mime_type="text/plain",
        file_extension=".txt",
        file_size_bytes=100,
        file_checksum=None,
        storage_bucket="documind-documents",
        storage_object_key=f"integration-tests/{uuid4()}.txt",
        metadata={"category": "finance"},
    )

    research_id = research_document["id"]
    finance_id = finance_document["id"]

    embedding = [1.0] + [0.0] * 383

    try:
        chunks.create_many(
            document_id=research_id,
            chunks=[
                DocumentChunk(
                    chunk_index=0,
                    content="Research introduction",
                    page_start=1,
                    page_end=1,
                    token_count=3,
                    metadata={"section": "introduction"},
                )
            ],
            embeddings=[embedding],
        )

        chunks.create_many(
            document_id=finance_id,
            chunks=[
                DocumentChunk(
                    chunk_index=0,
                    content="Finance conclusion",
                    page_start=1,
                    page_end=1,
                    token_count=3,
                    metadata={"section": "conclusion"},
                )
            ],
            embeddings=[embedding],
        )

        # Document ID filtering
        results = chunks.search_similar(
            query_embedding=embedding,
            filters=RetrievalFilters(
                document_ids=[research_id],
            ),
        )

        assert len(results) == 1
        assert results[0]["document_id"] == research_id

        # Document metadata filtering
        results = chunks.search_similar(
            query_embedding=embedding,
            filters=RetrievalFilters(
                document_metadata={"category": "finance"},
            ),
        )

        assert len(results) == 1
        assert results[0]["document_id"] == finance_id

        # Chunk metadata filtering
        results = chunks.search_similar(
            query_embedding=embedding,
            filters=RetrievalFilters(
                chunk_metadata={"section": "introduction"},
            ),
        )

        assert len(results) == 1
        assert results[0]["content"] == "Research introduction"

        # Combined filtering
        results = chunks.search_similar(
            query_embedding=embedding,
            filters=RetrievalFilters(
                document_ids=[research_id, finance_id],
                document_metadata={"category": "research"},
                chunk_metadata={"section": "introduction"},
            ),
        )

        assert len(results) == 1
        assert results[0]["document_id"] == research_id
        assert results[0]["content"] == "Research introduction"

    finally:
        documents.delete_record(research_id)
        documents.delete_record(finance_id)



@pytest.mark.integration
def test_vector_search_end_to_end(repositories):
    documents = repositories["documents"]
    chunks = repositories["chunks"]

    embedding_generator = EmbeddingGenerator()

    vector_search = VectorSearch(
        embedding_generator=embedding_generator,
        chunk_repository=chunks,
        config=RetrievalConfig(top_k=3, max_top_k=10),
    )

    document = documents.create(
        filename="semantic_search_test.txt",
        mime_type="text/plain",
        file_extension=".txt",
        file_size_bytes=500,
        file_checksum=None,
        storage_bucket="documind-documents",
        storage_object_key=f"integration-tests/{uuid4()}.txt",
        metadata={"test": "semantic_search"},
    )

    document_id = document["id"]

    test_chunks = [
        DocumentChunk(
            chunk_index=0,
            content=(
                "Machine learning is a branch of artificial intelligence "
                "that enables computers to learn from data."
            ),
            page_start=1,
            page_end=1,
            token_count=20,
            metadata={"topic": "machine_learning"},
        ),
        DocumentChunk(
            chunk_index=1,
            content=(
                "Deep learning uses artificial neural networks "
                "with multiple layers to learn complex patterns."
            ),
            page_start=2,
            page_end=2,
            token_count=18,
            metadata={"topic": "deep_learning"},
        ),
        DocumentChunk(
            chunk_index=2,
            content=(
                "PostgreSQL is a relational database management "
                "system that supports SQL queries."
            ),
            page_start=3,
            page_end=3,
            token_count=16,
            metadata={"topic": "databases"},
        ),
    ]

    try:
        embeddings = embedding_generator.embed_chunks(test_chunks)

        inserted = chunks.create_many(
            document_id=document_id,
            chunks=test_chunks,
            embeddings=embeddings,
        )

        assert inserted == 3

        results = vector_search.search(
            query="What is machine learning?",
            filters=RetrievalFilters(
                document_ids=[document_id],
            ),
        )

        assert len(results) > 0
        assert len(results) <= 3

        assert results[0]["content"] == test_chunks[0].content

        for result in results:
            assert result["document_id"] == document_id
            assert -1.0 <= result["similarity"] <= 1.0

    finally:
        documents.delete_record(document_id)


@pytest.mark.integration
def test_keyword_search_returns_matching_chunks(repositories):
    documents = repositories["documents"]
    chunks = repositories["chunks"]

    document = documents.create(
        filename="keyword_search_test.txt",
        mime_type="text/plain",
        file_extension=".txt",
        file_size_bytes=500,
        file_checksum=None,
        storage_bucket="documind-documents",
        storage_object_key=f"integration-tests/{uuid4()}.txt",
        metadata={"test": "keyword_search"},
    )

    document_id = document["id"]

    test_chunks = [
        DocumentChunk(
            chunk_index=0,
            content=(
                "PostgreSQL provides powerful full text search "
                "capabilities for database applications."
            ),
            page_start=1,
            page_end=1,
            token_count=12,
            metadata={"topic": "postgresql"},
        ),
        DocumentChunk(
            chunk_index=1,
            content=(
                "PostgreSQL supports indexing and querying "
                "structured relational data."
            ),
            page_start=2,
            page_end=2,
            token_count=10,
            metadata={"topic": "postgresql"},
        ),
        DocumentChunk(
            chunk_index=2,
            content=(
                "Machine learning models identify patterns "
                "from large datasets."
            ),
            page_start=3,
            page_end=3,
            token_count=10,
            metadata={"topic": "machine_learning"},
        ),
    ]

    # Semantic embeddings are not relevant to this keyword-search test.
    embeddings = [
        [1.0] + [0.0] * 383,
        [0.8, 0.6] + [0.0] * 382,
        [0.0, 1.0] + [0.0] * 382,
    ]

    try:
        inserted = chunks.create_many(
            document_id=document_id,
            chunks=test_chunks,
            embeddings=embeddings,
        )

        assert inserted == 3

        results = chunks.search_keyword(
            query_text="PostgreSQL",
            top_k=5,
            filters=RetrievalFilters(
                document_ids=[document_id],
            ),
        )

        # Only chunks containing PostgreSQL should be returned.
        assert len(results) == 2

        returned_contents = [result["content"] for result in results]

        assert test_chunks[0].content in returned_contents
        assert test_chunks[1].content in returned_contents
        assert test_chunks[2].content not in returned_contents

        # Matching chunks should have a positive keyword score.
        for result in results:
            assert result["keyword_score"] > 0
            assert result["document_id"] == document_id

        # Verify document source information is preserved.
        assert all(
            result["document_filename"] == "keyword_search_test.txt"
            for result in results
        )

    finally:
        documents.delete_record(document_id)

@pytest.mark.integration
def test_hybrid_search_end_to_end(repositories):
    documents = repositories["documents"]
    chunks = repositories["chunks"]

    embedding_generator = EmbeddingGenerator()

    vector_search = VectorSearch(
        embedding_generator=embedding_generator,
        chunk_repository=chunks,
        config=RetrievalConfig(top_k=3, max_top_k=10),
    )

    document = documents.create(
        filename="hybrid_search_test.txt",
        mime_type="text/plain",
        file_extension=".txt",
        file_size_bytes=500,
        file_checksum=None,
        storage_bucket="documind-documents",
        storage_object_key=f"integration-tests/{uuid4()}.txt",
        metadata={"test": "hybrid_search"},
    )

    document_id = document["id"]

    test_chunks = [
        DocumentChunk(
            chunk_index=0,
            content=(
                "PostgreSQL provides powerful full text search "
                "capabilities for database applications."
            ),
            page_start=1,
            page_end=1,
            token_count=12,
            metadata={"topic": "postgresql"},
        ),
        DocumentChunk(
            chunk_index=1,
            content=(
                "Relational database systems organize information "
                "using structured tables and relationships."
            ),
            page_start=2,
            page_end=2,
            token_count=12,
            metadata={"topic": "databases"},
        ),
        DocumentChunk(
            chunk_index=2,
            content=(
                "Machine learning models identify patterns "
                "from large datasets."
            ),
            page_start=3,
            page_end=3,
            token_count=10,
            metadata={"topic": "machine_learning"},
        ),
    ]

    try:
        embeddings = embedding_generator.embed_chunks(test_chunks)

        inserted = chunks.create_many(
            document_id=document_id,
            chunks=test_chunks,
            embeddings=embeddings,
        )

        assert inserted == 3

        results = vector_search.hybrid_search(
            query="PostgreSQL full text search",
            filters=RetrievalFilters(
                document_ids=[document_id],
            ),
            top_k=3,
        )

        assert len(results) > 0
        assert len(results) <= 3

        # Every returned chunk must belong to the test document.
        assert all(
            result["document_id"] == document_id
            for result in results
        )

        # RRF scores must be present and positive.
        assert all(
            result["rrf_score"] > 0
            for result in results
        )

        # The exact keyword match should be retrieved.
        returned_contents = [
            result["content"] for result in results
        ]

        assert test_chunks[0].content in returned_contents

        # Source information must be preserved.
        for result in results:
            assert result["document_filename"] == "hybrid_search_test.txt"
            assert result["page_start"] is not None
            assert result["page_end"] is not None
            assert result["chunk_index"] >= 0

    finally:
        documents.delete_record(document_id)
