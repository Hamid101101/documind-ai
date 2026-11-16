
import io
import uuid

import pytest

from infra.storage.minio_client import MinIOClient
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import io
import uuid


@pytest.fixture
def minio_client():
    return MinIOClient()


def test_minio_connection(minio_client):
    """Verify that the application can connect to MinIO."""
    assert minio_client.check_connection() is True



def test_upload_download_delete_file(minio_client):
    """Test the complete object storage lifecycle."""

    object_key = f"tests/{uuid.uuid4()}.txt"
    original_content = b"DocuMind AI - MinIO integration test."

    # 1. Upload
    file_obj = io.BytesIO(original_content)

    minio_client.upload_file(
        file_obj=file_obj,
        object_key=object_key,
        content_type="text/plain",
        metadata={"source": "pytest"},
    )

    try:
        # 2. Verify object exists
        assert minio_client.object_exists(object_key) is True

        # 3. Download into a temporary file
        with TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "downloaded.txt"

            minio_client.download_file(
                object_key=object_key,
                destination=str(destination),
            )

            assert destination.exists()
            assert destination.read_bytes() == original_content

    finally:
        # 4. Always clean up the MinIO test object
        minio_client.delete_file(object_key)

    # 5. Verify deletion
    assert minio_client.object_exists(object_key) is False


def test_multipart_upload(minio_client):
    """Test multipart upload and checksum calculation."""

    object_key = f"tests/multipart-{uuid.uuid4()}.bin"

    # Generate data larger than the minimum multipart part size.
    original_content = b"A" * (6 * 1024 * 1024 + 123)

    expected_checksum = hashlib.sha256(
        original_content
    ).hexdigest()

    result = minio_client.upload_multipart_file(
        file_obj=io.BytesIO(original_content),
        object_key=object_key,
        content_type="application/octet-stream",
    )

    try:
        assert result["object_key"] == object_key
        assert result["size"] == len(original_content)
        assert result["checksum"] == expected_checksum

        assert minio_client.object_exists(object_key) is True

        with TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "downloaded.bin"

            minio_client.download_file(
                object_key,
                destination,
            )

            assert destination.read_bytes() == original_content

    finally:
        if minio_client.object_exists(object_key):
            minio_client.delete_file(object_key)


def test_multipart_upload_rejects_empty_file(minio_client):
    """Ensure empty files are rejected."""

    object_key = f"tests/empty-{uuid.uuid4()}.txt"

    with pytest.raises(ValueError, match="empty file"):
        minio_client.upload_multipart_file(
            file_obj=io.BytesIO(b""),
            object_key=object_key,
        )

    assert minio_client.object_exists(object_key) is False


