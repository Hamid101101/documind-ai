
from uuid import uuid4

import pytest
from pydantic import ValidationError

from retrieval.filters import RetrievalFilters


def test_filters_default_to_none():
    filters = RetrievalFilters()

    assert filters.document_ids is None
    assert filters.document_metadata is None
    assert filters.chunk_metadata is None


def test_filters_accept_document_ids():
    document_ids = [uuid4(), uuid4()]

    filters = RetrievalFilters(document_ids=document_ids)

    assert filters.document_ids == document_ids


def test_filters_accept_document_metadata():
    metadata = {"author": "Hamid", "category": "research"}

    filters = RetrievalFilters(document_metadata=metadata)

    assert filters.document_metadata == metadata


def test_filters_accept_chunk_metadata():
    metadata = {"section": "introduction", "language": "en"}

    filters = RetrievalFilters(chunk_metadata=metadata)

    assert filters.chunk_metadata == metadata


def test_filters_accept_all_filters_together():
    document_ids = [uuid4()]
    document_metadata = {"category": "research"}
    chunk_metadata = {"section": "introduction"}

    filters = RetrievalFilters(
        document_ids=document_ids,
        document_metadata=document_metadata,
        chunk_metadata=chunk_metadata,
    )

    assert filters.document_ids == document_ids
    assert filters.document_metadata == document_metadata
    assert filters.chunk_metadata == chunk_metadata


def test_filters_reject_empty_document_ids():
    with pytest.raises(ValidationError):
        RetrievalFilters(document_ids=[])


def test_filters_reject_duplicate_document_ids():
    document_id = uuid4()

    with pytest.raises(ValidationError):
        RetrievalFilters(document_ids=[document_id, document_id])


def test_filters_reject_unknown_fields():
    with pytest.raises(ValidationError):
        RetrievalFilters(unknown_filter="value")
