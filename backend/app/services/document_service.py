
import logging
import os
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

from database.repositories.document_repository import DocumentRepository
from infra.storage.minio_client import MinIOClient


logger = logging.getLogger(__name__)


class DocumentValidationError(ValueError):
    """Raised when an uploaded document fails validation."""


class DocumentService:
    """Coordinates document storage and metadata persistence."""

    ALLOWED_MIME_TYPES = {
        ".pdf": "application/pdf",
        ".docx": (
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
        ".txt": "text/plain",
    }

    def __init__(
        self,
        minio_client: MinIOClient | None = None,
        document_repository: DocumentRepository | None = None,
        max_upload_size_bytes: int | None = None,
    ) -> None:
        self.minio = minio_client or MinIOClient()
        self.repository = document_repository or DocumentRepository()

        self.max_upload_size_bytes = (
            max_upload_size_bytes
            if max_upload_size_bytes is not None
            else int(os.getenv("MAX_UPLOAD_SIZE_BYTES", 50 * 1024 * 1024))
        )

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Extract a safe filename without directory components."""

        filename = filename.replace("\\", "/")
        filename = filename.split("/")[-1].strip()

        if not filename or filename in {".", ".."}:
            raise DocumentValidationError("Invalid filename.")

        if len(filename) > 255:
            raise DocumentValidationError(
                "Filename cannot exceed 255 characters."
            )

        return filename

    def _validate_file(
        self,
        filename: str,
        file_size_bytes: int,
    ) -> tuple[str, str]:
        """Validate file extension and declared size."""

        if file_size_bytes <= 0:
            raise DocumentValidationError("The uploaded file is empty.")

        if file_size_bytes > self.max_upload_size_bytes:
            max_size_mb = self.max_upload_size_bytes / (1024 * 1024)

            raise DocumentValidationError(
                f"File exceeds the maximum allowed size of "
                f"{max_size_mb:.1f} MiB."
            )

        file_extension = Path(filename).suffix.lower()

        mime_type = self.ALLOWED_MIME_TYPES.get(file_extension)

        if mime_type is None:
            allowed = ", ".join(self.ALLOWED_MIME_TYPES.keys())

            raise DocumentValidationError(
                f"Unsupported file extension. Allowed extensions: {allowed}."
            )

        return file_extension, mime_type

    def _cleanup_storage_object(self, object_key: str) -> None:
        """Attempt to remove an object after a failed operation."""

        try:
            self.minio.delete_file(object_key)

        except Exception:
            logger.exception(
                "Failed to clean up MinIO object '%s'.",
                object_key,
            )

    def upload_document(
        self,
        file_obj: BinaryIO,
        filename: str,
        file_size_bytes: int,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Upload a document to MinIO and persist its metadata in PostgreSQL.

        The file object must be seekable and positioned at its beginning.
        """

        filename = self._sanitize_filename(filename)

        file_extension, mime_type = self._validate_file(
            filename,
            file_size_bytes,
        )

        document_id = uuid4()

        object_key = (
            f"documents/{document_id}{file_extension}"
        )

        try:
            file_obj.seek(0)

            # 1. Upload original file to MinIO.
            upload_result = self.minio.upload_multipart_file(
                file_obj=file_obj,
                object_key=object_key,
                content_type=mime_type,
                metadata={
                    "document_id": str(document_id),
                    "original_filename": filename,
                },
            )

            actual_size = upload_result["size"]
            checksum = upload_result["checksum"]

            # 2. Verify the actual uploaded size.
            if actual_size != file_size_bytes:
                raise RuntimeError(
                    "Uploaded file size does not match the declared size."
                )

        except Exception:
            # The multipart client already aborts incomplete uploads.
            # This also handles failures after a completed upload.
            if self.minio.object_exists(object_key):
                self._cleanup_storage_object(object_key)

            logger.exception(
                "Document upload failed for '%s'.",
                filename,
            )
            raise

        try:
            # 3. Persist document metadata in PostgreSQL.
            document = self.repository.create(
                filename=filename,
                mime_type=mime_type,
                file_extension=file_extension,
                file_size_bytes=actual_size,
                file_checksum=checksum,
                storage_bucket=self.minio.bucket,
                storage_object_key=object_key,
                metadata=metadata or {},
            )

            logger.info(
                "Document '%s' successfully registered with ID %s.",
                filename,
                document_id,
            )

            return document

        except Exception:
            # 4. Compensate for database failure.
            self._cleanup_storage_object(object_key)

            logger.exception(
                "Failed to persist metadata for document '%s'.",
                document_id,
            )
            raise
