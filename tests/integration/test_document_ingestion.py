
import os
from pathlib import Path
from uuid import UUID

import pytest

from database.repositories.chunk_repository import ChunkRepository
from database.repositories.document_repository import DocumentRepository
from infra.storage.minio_client import MinIOClient
from ingestion.chunking.text_chunker import TextChunker
from ingestion.cleaning.text_cleaner import TextCleaner
from ingestion.embeddings.embedding_generator import EmbeddingGenerator
from ingestion.loaders.docx_loader import DOCXLoader
from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.loaders.txt_loader import TXTLoader
from ingestion.pipeline import DocumentIngestionPipeline


DOCUMENT_ID = UUID("172affbe-fdec-4fcc-9cad-cdf79bea1421")


@pytest.fixture
def integration_dependencies():
    """Initialize real infrastructure clients and ingestion components."""

    document_repository = DocumentRepository()
    chunk_repository = ChunkRepository()
    minio_client = MinIOClient()

    pipeline = DocumentIngestionPipeline(
        loaders={
            ".pdf": PDFLoader(),
            ".docx": DOCXLoader(),
            ".txt": TXTLoader(),
        },
        cleaner=TextCleaner(),
        chunker=TextChunker(),
        embedder=EmbeddingGenerator(),
        chunk_repository=chunk_repository,
        document_repository=document_repository,
    )

    return {
        "documents": document_repository,
        "chunks": chunk_repository,
        "minio": minio_client,
        "pipeline": pipeline,
    }


@pytest.mark.integration
def test_ingest_existing_cv(integration_dependencies, tmp_path):
    documents = integration_dependencies["documents"]
    chunks = integration_dependencies["chunks"]
    minio = integration_dependencies["minio"]
    pipeline = integration_dependencies["pipeline"]

    document = documents.get_by_id(DOCUMENT_ID)

    assert document is not None
    assert document["filename"] == "Abdelhamid_Boukhris_CV.pdf"

    object_key = document["storage_object_key"]

    assert minio.object_exists(object_key)

    local_file = tmp_path / document["filename"]

    minio.download_file(object_key, local_file)

    assert local_file.exists()
    assert local_file.stat().st_size > 0

    result = pipeline.process(
        document_id=DOCUMENT_ID,
        file_path=local_file,
    )

    assert result.status == "completed"
    assert result.chunk_count > 0
    assert result.embedding_model

    updated_document = documents.get_by_id(DOCUMENT_ID)
    persisted_chunks = chunks.get_by_document_id(DOCUMENT_ID)

    assert updated_document["status"] == "completed"
    assert updated_document["chunk_count"] == result.chunk_count
    assert len(persisted_chunks) == result.chunk_count

    for chunk in persisted_chunks:
        assert chunk["content"].strip()
        assert chunk["embedding"] is not None
        assert chunk["token_count"] >= 0
