
import logging
import tempfile
from pathlib import Path
from uuid import UUID

from worker.celery_app import celery_app

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


logger = logging.getLogger(__name__)


@celery_app.task(name="worker.tasks.health_check")
def health_check() -> dict:
    return {
        "status": "success",
        "message": "DocuMind Celery worker is operational.",
    }


def build_ingestion_pipeline(
    document_repository: DocumentRepository,
    chunk_repository: ChunkRepository,
) -> DocumentIngestionPipeline:
    """Build the document ingestion pipeline."""

    return DocumentIngestionPipeline(
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


@celery_app.task(
    name="worker.tasks.process_document",
    bind=True,
)
def process_document(self, document_id: str) -> dict:
    """Download and process a document asynchronously."""

    document_uuid = UUID(document_id)

    logger.info(
        "Starting document processing task: %s (task_id=%s)",
        document_uuid,
        self.request.id,
    )

    document_repository = DocumentRepository()
    chunk_repository = ChunkRepository()
    minio_client = MinIOClient()

    document = document_repository.get_by_id(document_uuid)

    if document is None:
        raise ValueError(f"Document not found: {document_uuid}")

    try:
        # Ensure the document references the configured MinIO bucket.
        if document["storage_bucket"] != minio_client.bucket:
            raise ValueError(
                f"Document bucket '{document['storage_bucket']}' "
                f"does not match configured bucket '{minio_client.bucket}'."
            )

        # Use only the filename, preventing directory traversal.
        filename = Path(document["filename"]).name

        if not filename:
            raise ValueError("Document filename is invalid.")

        # Download the original document to a temporary directory.
        with tempfile.TemporaryDirectory(
            prefix="documind-ingestion-"
        ) as temp_dir:
            local_file = Path(temp_dir) / filename

            minio_client.download_file(
                document["storage_object_key"],
                local_file,
            )

            if not local_file.is_file() or local_file.stat().st_size == 0:
                raise RuntimeError(
                    "Downloaded document is missing or empty."
                )

            pipeline = build_ingestion_pipeline(
                document_repository=document_repository,
                chunk_repository=chunk_repository,
            )

            result = pipeline.process(
                document_id=document_uuid,
                file_path=local_file,
            )

        logger.info(
            "Document processing completed: %s (%s chunks)",
            document_uuid,
            result.chunk_count,
        )

        return {
            "document_id": str(result.document_id),
            "status": result.status,
            "page_count": result.page_count,
            "chunk_count": result.chunk_count,
            "embedding_model": result.embedding_model,
        }

    except Exception as exc:
        logger.exception(
            "Document processing task failed: %s",
            document_uuid,
        )

        # The pipeline handles its own failures. This covers failures
        # occurring before the pipeline starts, such as MinIO downloads.
        current_document = document_repository.get_by_id(document_uuid)

        if current_document and current_document["status"] != "failed":
            document_repository.update_status(
                document_uuid,
                status="failed",
                error_message=str(exc)[:5000],
            )

        raise
