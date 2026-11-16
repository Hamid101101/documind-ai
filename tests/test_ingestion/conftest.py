
from unittest.mock import Mock
from uuid import uuid4

import pytest

from ingestion.chunking.base import DocumentChunk
from ingestion.loaders.base import DocumentPage, LoadedDocument
from ingestion.pipeline import DocumentIngestionPipeline


@pytest.fixture
def document_id():
    return uuid4()


@pytest.fixture
def loaded_document():
    return LoadedDocument(
        source_format="txt",
        page_count=None,
        pages=[
            DocumentPage(
                page_number=None,
                text="This is a test document for DocuMind AI.",
            )
        ],
    )


@pytest.fixture
def chunks():
    return [
        DocumentChunk(
            chunk_index=0,
            content="This is a test document for DocuMind AI.",
            page_start=None,
            page_end=None,
            token_count=9,
            metadata={},
        )
    ]


@pytest.fixture
def pipeline_dependencies(loaded_document, chunks):
    loader = Mock()
    loader.load.return_value = loaded_document

    cleaner = Mock()
    cleaner.clean_document.return_value = loaded_document

    chunker = Mock()
    chunker.chunk_document.return_value = chunks

    embedder = Mock()
    embedder.config.model_name = "intfloat/multilingual-e5-small"
    embedder.embed_chunks.return_value = [[0.1] * 384]

    chunk_repository = Mock()
    chunk_repository.create_many.return_value = len(chunks)

    document_repository = Mock()
    document_repository.get_by_id.return_value = {"id": uuid4()}
    document_repository.mark_processing.return_value = {"status": "processing"}
    document_repository.mark_completed.return_value = {"status": "completed"}

    dependencies = {
        "loader": loader,
        "cleaner": cleaner,
        "chunker": chunker,
        "embedder": embedder,
        "chunk_repository": chunk_repository,
        "document_repository": document_repository,
    }

    return dependencies


@pytest.fixture
def pipeline(pipeline_dependencies):
    deps = pipeline_dependencies

    return DocumentIngestionPipeline(
        loaders={".txt": deps["loader"]},
        cleaner=deps["cleaner"],
        chunker=deps["chunker"],
        embedder=deps["embedder"],
        chunk_repository=deps["chunk_repository"],
        document_repository=deps["document_repository"],
    )
