
import uuid

import pytest

from database.repositories.document_repository import DocumentRepository


@pytest.fixture
def repository():
    return DocumentRepository()


@pytest.fixture
def test_document(repository):
    """Create a temporary test document and clean it up afterward."""

    document = repository.create(
        filename="test-document.txt",
        mime_type="text/plain",
        file_extension=".txt",
        file_size_bytes=100,
        file_checksum="test-checksum-" + uuid.uuid4().hex,
        storage_bucket="documind-documents",
        storage_object_key="pytest/test-document-" + uuid.uuid4().hex,
        metadata={"source": "pytest"},
    )

    yield document

    repository.delete_record(document["id"])


def test_create_document(test_document):
    assert test_document["filename"] == "test-document.txt"
    assert test_document["status"] == "uploaded"
    assert test_document["metadata"]["source"] == "pytest"


def test_get_document_by_id(repository, test_document):
    result = repository.get_by_id(test_document["id"])

    assert result is not None
    assert result["id"] == test_document["id"]


def test_get_document_by_object_key(repository, test_document):
    result = repository.get_by_object_key(
        test_document["storage_bucket"],
        test_document["storage_object_key"],
    )

    assert result is not None
    assert result["id"] == test_document["id"]


def test_list_documents(repository, test_document):
    result = repository.list_documents(page=1, page_size=10)

    assert result["total"] >= 1
    assert any(
        doc["id"] == test_document["id"]
        for doc in result["documents"]
    )


def test_update_document_status(repository, test_document):
    result = repository.update_status(
        test_document["id"],
        status="queued",
    )

    assert result is not None
    assert result["status"] == "queued"


def test_delete_document(repository):
    document = repository.create(
        filename="delete-test.txt",
        mime_type="text/plain",
        file_extension=".txt",
        file_size_bytes=100,
        file_checksum="test-checksum-" + uuid.uuid4().hex,
        storage_bucket="documind-documents",
        storage_object_key="pytest/delete-test-" + uuid.uuid4().hex,
    )

    assert repository.delete_record(document["id"]) is True
    assert repository.get_by_id(document["id"]) is None
