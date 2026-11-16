
from contextlib import contextmanager
from uuid import uuid4

import pytest

from ingestion.chunking.base import DocumentChunk


class FakeCursor:
    def __init__(self):
        self.executed_query = None
        self.executed_params = None
        self.batch_params = None
        self.rowcount = 0
        self.fetchone_result = {"total": 2}
        self.fetchall_result = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, query, params=None):
        self.executed_query = query
        self.executed_params = params

    def executemany(self, query, params):
        self.executed_query = query
        self.batch_params = list(params)
        self.rowcount = len(self.batch_params)

    def fetchone(self):
        return self.fetchone_result

    def fetchall(self):
        return self.fetchall_result


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def cursor(self):
        return self._cursor


class FakePostgreSQLClient:
    def __init__(self):
        self.cursor_instance = FakeCursor()
        self.connection_count = 0

    @contextmanager
    def connection(self):
        self.connection_count += 1
        yield FakeConnection(self.cursor_instance)


@pytest.fixture
def fake_db():
    return FakePostgreSQLClient()


@pytest.fixture
def chunk_repository(fake_db):
    from database.repositories.chunk_repository import ChunkRepository

    return ChunkRepository(db=fake_db)


@pytest.fixture
def document_id():
    return uuid4()


@pytest.fixture
def make_chunks():
    def _make_chunks(contents):
        return [
            DocumentChunk(
                chunk_index=index,
                content=content,
                page_start=1,
                page_end=1,
                token_count=len(content.split()),
                metadata={"source": "test"},
            )
            for index, content in enumerate(contents)
        ]

    return _make_chunks


@pytest.fixture
def make_embeddings():
    def _make_embeddings(count):
        return [[0.1] * 384 for _ in range(count)]

    return _make_embeddings
