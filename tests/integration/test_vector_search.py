
from uuid import uuid4

import pytest

from database.repositories.chunk_repository import ChunkRepository
from database.repositories.document_repository import DocumentRepository
from ingestion.chunking.base import DocumentChunk


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
            top_k=10,
        )

        results = [
            result
            for result in results
            if result["document_id"] == document_id
        ]

        results = results[:3]

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

