
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from database.repositories.chunk_repository import ChunkRepository
from database.repositories.document_repository import DocumentRepository
from ingestion.chunking.text_chunker import TextChunker
from ingestion.cleaning.text_cleaner import TextCleaner
from ingestion.embeddings.embedding_generator import EmbeddingGenerator
from ingestion.loaders.base import BaseLoader


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    document_id: UUID
    status: str
    page_count: int | None
    chunk_count: int
    embedding_model: str


class DocumentIngestionPipeline:
    """Orchestrate document loading, cleaning, chunking and embedding."""

    def __init__(
        self,
        *,
        loaders: dict[str, BaseLoader],
        cleaner: TextCleaner,
        chunker: TextChunker,
        embedder: EmbeddingGenerator,
        chunk_repository: ChunkRepository,
        document_repository: DocumentRepository,
    ) -> None:
        self.loaders = {
            extension.lower(): loader
            for extension, loader in loaders.items()
        }
        self.cleaner = cleaner
        self.chunker = chunker
        self.embedder = embedder
        self.chunk_repository = chunk_repository
        self.document_repository = document_repository

    def process(
        self,
        document_id: UUID,
        file_path: str | Path,
    ) -> IngestionResult:
        """Process a document and persist its chunks and embeddings."""

        path = Path(file_path)
        extension = path.suffix.lower()

        if extension not in self.loaders:
            raise ValueError(
                f"Unsupported document extension: {extension}"
            )

        document = self.document_repository.get_by_id(document_id)

        if document is None:
            raise ValueError(f"Document not found: {document_id}")

        chunks_created = False

        try:
            processing_document = (
                self.document_repository.mark_processing(document_id)
            )

            if processing_document is None:
                raise ValueError(
                    f"Document not found: {document_id}"
                )

            logger.info(
                "Starting ingestion for document %s",
                document_id,
            )

            # 1. Load
            loader = self.loaders[extension]
            loaded_document = loader.load(str(path))

            # 2. Clean
            cleaned_document = self.cleaner.clean_document(
                loaded_document
            )

            # 3. Chunk
            chunks = self.chunker.chunk_document(cleaned_document)

            if not chunks:
                raise ValueError(
                    "Document contains no extractable text after cleaning."
                )

            # 4. Generate embeddings
            embeddings = self.embedder.embed_chunks(chunks)

            if len(embeddings) != len(chunks):
                raise ValueError(
                    "Embedding count does not match chunk count."
                )

            # 5. Replace previous chunks and persist the new batch.
            # This also makes repeated task execution idempotent
            # for the initial single-worker processing workflow.
            self.chunk_repository.delete_by_document_id(document_id)

            inserted_count = self.chunk_repository.create_many(
                document_id,
                chunks,
                embeddings,
            )

            if inserted_count != len(chunks):
                raise RuntimeError(
                    f"Expected {len(chunks)} inserted chunks, "
                    f"received {inserted_count}."
                )

            chunks_created = True

            # 6. Update document processing metadata
            model_name = self.embedder.config.model_name

            completed_document = self.document_repository.mark_completed(
                document_id,
                chunk_count=inserted_count,
                page_count=cleaned_document.page_count,
                embedding_model=model_name,
            )

            if completed_document is None:
                raise RuntimeError(
                    f"Could not finalize document {document_id}."
                )

            logger.info(
                "Document %s processed successfully: %d chunks",
                document_id,
                inserted_count,
            )

            return IngestionResult(
                document_id=document_id,
                status="completed",
                page_count=cleaned_document.page_count,
                chunk_count=inserted_count,
                embedding_model=model_name,
            )

        except Exception as exc:
            logger.exception(
                "Document ingestion failed: %s",
                document_id,
            )

            # Remove newly inserted chunks if finalization failed.
            if chunks_created:
                try:
                    self.chunk_repository.delete_by_document_id(
                        document_id
                    )
                except Exception:
                    logger.exception(
                        "Could not clean up chunks for document %s",
                        document_id,
                    )

            try:
                self.document_repository.update_status(
                    document_id,
                    status="failed",
                    error_message=str(exc)[:5000],
                )
            except Exception:
                logger.exception(
                    "Could not mark document %s as failed",
                    document_id,
                )

            raise
