
import hashlib
import io
import os
import uuid
from unittest.mock import MagicMock

import pytest

from backend.app.services.document_service import (
    DocumentService,
    DocumentValidationError,
)
from database.repositories.document_repository import DocumentRepository
from infra.storage.minio_client import MinIOClient


# ============================================================
# UNIT TESTS
# ============================================================

@pytest.fixture
def minio_mock():
    client = MagicMock()
    client.bucket = "documind-documents"
    return client


@pytest.fixture
def repository_mock():
    return MagicMock(spec=DocumentRepository)


@pytest.fixture
def service(minio_mock, repository_mock):
    return DocumentService(
        minio_client=minio_mock,
        document_repository=repository_mock,
        max_upload_size_bytes=10 * 1024 * 1024,
    )


def test_upload_document_success(
    service,
    minio_mock,
    repository_mock,
):
    """Verify successful upload and metadata persistence."""

    content = b"DocuMind AI integration test."
    expected_checksum = hashlib.sha256(content).hexdigest()

    minio_mock.upload_multipart_file.return_value = {
        "object_key": "documents/test.txt",
        "size": len(content),
        "checksum": expected_checksum,
    }

    repository_mock.create.return_value = {
        "id": uuid.uuid4(),
        "filename": "test.txt",
        "file_size_bytes": len(content),
        "file_checksum": expected_checksum,
        "status": "uploaded",
    }

    result = service.upload_document(
        file_obj=io.BytesIO(content),
        filename="test.txt",
        file_size_bytes=len(content),
        metadata={"source": "pytest"},
    )

    assert result["filename"] == "test.txt"
    assert result["status"] == "uploaded"

    minio_mock.upload_multipart_file.assert_called_once()
    repository_mock.create.assert_called_once()

    create_args = repository_mock.create.call_args.kwargs

    assert create_args["file_size_bytes"] == len(content)
    assert create_args["file_checksum"] == expected_checksum
    assert create_args["storage_bucket"] == minio_mock.bucket
    assert create_args["metadata"] == {"source": "pytest"}


def test_reject_empty_file(service, minio_mock):
    """Reject an empty file before accessing MinIO."""

    with pytest.raises(DocumentValidationError, match="empty"):
        service.upload_document(
            file_obj=io.BytesIO(b""),
            filename="empty.txt",
            file_size_bytes=0,
        )

    minio_mock.upload_multipart_file.assert_not_called()


def test_reject_unsupported_extension(service, minio_mock):
    """Reject unsupported file extensions."""

    content = b"some content"

    with pytest.raises(DocumentValidationError, match="Unsupported"):
        service.upload_document(
            file_obj=io.BytesIO(content),
            filename="malware.exe",
            file_size_bytes=len(content),
        )

    minio_mock.upload_multipart_file.assert_not_called()


def test_reject_file_exceeding_maximum_size(service, minio_mock):
    """Reject files exceeding the configured size limit."""

    content = b"some content"

    with pytest.raises(DocumentValidationError, match="maximum"):
        service.upload_document(
            file_obj=io.BytesIO(content),
            filename="large.txt",
            file_size_bytes=11 * 1024 * 1024,
        )

    minio_mock.upload_multipart_file.assert_not_called()


def test_cleanup_minio_if_database_insert_fails(
    service,
    minio_mock,
    repository_mock,
):
    """Delete the uploaded object if PostgreSQL persistence fails."""

    content = b"Database failure test."

    minio_mock.upload_multipart_file.return_value = {
        "object_key": "documents/test.txt",
        "size": len(content),
        "checksum": hashlib.sha256(content).hexdigest(),
    }

    minio_mock.object_exists.return_value = True

    repository_mock.create.side_effect = RuntimeError(
        "Database insertion failed."
    )

    with pytest.raises(RuntimeError, match="Database insertion failed"):
        service.upload_document(
            file_obj=io.BytesIO(content),
            filename="test.txt",
            file_size_bytes=len(content),
        )

    minio_mock.delete_file.assert_called_once()


def test_cleanup_if_uploaded_size_does_not_match(
    service,
    minio_mock,
    repository_mock,
):
    """Clean up an uploaded object if its size is unexpected."""

    content = b"Size mismatch test."

    minio_mock.upload_multipart_file.return_value = {
        "object_key": "documents/test.txt",
        "size": len(content) + 1,
        "checksum": hashlib.sha256(content).hexdigest(),
    }

    minio_mock.object_exists.return_value = True

    with pytest.raises(RuntimeError, match="size does not match"):
        service.upload_document(
            file_obj=io.BytesIO(content),
            filename="test.txt",
            file_size_bytes=len(content),
        )

    minio_mock.delete_file.assert_called_once()
    repository_mock.create.assert_not_called()


# ============================================================
# REAL INTEGRATION TEST
# ============================================================

@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1",
    reason="Set RUN_INTEGRATION_TESTS=1 to run real infrastructure tests.",
)
def test_document_service_real_upload():
    """
    Test DocumentService against actual MinIO and PostgreSQL.

    Requires both infrastructure services to be running.
    """

    minio = MinIOClient()
    repository = DocumentRepository()

    assert minio.check_connection() is True
    assert repository.db.check_connection() is True

    service = DocumentService(
        minio_client=minio,
        document_repository=repository,
    )

    content = b"Real DocuMind AI document service test."
    filename = f"integration-{uuid.uuid4()}.txt"

    document = None

    try:
        document = service.upload_document(
            file_obj=io.BytesIO(content),
            filename=filename,
            file_size_bytes=len(content),
            metadata={"source": "integration-test"},
        )

        # Verify PostgreSQL record
        assert document["filename"] == filename
        assert document["status"] == "uploaded"
        assert document["file_size_bytes"] == len(content)
        assert document["file_checksum"] == hashlib.sha256(
            content
        ).hexdigest()

        # Verify MinIO object
        assert minio.object_exists(
            document["storage_object_key"]
        ) is True

        # Verify database retrieval
        stored_document = repository.get_by_id(document["id"])

        assert stored_document is not None
        assert stored_document["id"] == document["id"]

    finally:
        if document is not None:
            try:
                minio.delete_file(document["storage_object_key"])
            finally:
                repository.delete_record(document["id"])
