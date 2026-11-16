
from math import isfinite
from typing import Any, Sequence
from uuid import UUID

from psycopg.types.json import Jsonb

from infra.postgres.postgres_client import PostgreSQLClient
from ingestion.chunking.base import DocumentChunk


class ChunkRepository:
    """Repository for document chunks and their pgvector embeddings."""

    EMBEDDING_DIMENSION = 384

    def __init__(self, db: PostgreSQLClient | None = None):
        self.db = db or PostgreSQLClient()

    @staticmethod
    def _serialize_embedding(embedding: Sequence[float]) -> str:
        """Convert a numeric embedding into a PostgreSQL vector literal."""

        if len(embedding) != ChunkRepository.EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding must contain exactly "
                f"{ChunkRepository.EMBEDDING_DIMENSION} dimensions."
            )

        values = []

        for value in embedding:
            if isinstance(value, bool):
                raise ValueError("Embedding values must be numeric.")

            numeric_value = float(value)

            if not isfinite(numeric_value):
                raise ValueError("Embedding contains non-finite values.")

            values.append(str(numeric_value))

        return f"[{','.join(values)}]"

    def create_many(
        self,
        document_id: UUID,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """
        Insert document chunks and their embeddings in one transaction.

        Returns the number of inserted chunks.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        if not chunks:
            return 0

        chunk_indexes = [chunk.chunk_index for chunk in chunks]

        if len(chunk_indexes) != len(set(chunk_indexes)):
            raise ValueError("Chunk indexes must be unique within a batch.")

        params = []

        # Validate and prepare the entire batch before opening a transaction.
        for chunk, embedding in zip(chunks, embeddings):
            if chunk.chunk_index < 0:
                raise ValueError("Chunk index cannot be negative.")

            if not chunk.content.strip():
                raise ValueError("Chunk content cannot be empty.")

            if chunk.token_count < 0:
                raise ValueError("Token count cannot be negative.")

            vector = self._serialize_embedding(embedding)

            params.append(
                (
                    document_id,
                    chunk.chunk_index,
                    chunk.content,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.token_count,
                    Jsonb(chunk.metadata or {}),
                    vector,
                )
            )

        query = """
            INSERT INTO document_chunks (
                document_id,
                chunk_index,
                content,
                page_start,
                page_end,
                token_count,
                metadata,
                embedding
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s::vector
            );
        """

        with self.db.connection() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(query, params)
                return cursor.rowcount

    def get_by_document_id(
        self,
        document_id: UUID,
    ) -> list[dict[str, Any]]:
        """Retrieve all chunks belonging to a document."""

        query = """
            SELECT
                id,
                document_id,
                chunk_index,
                content,
                page_start,
                page_end,
                token_count,
                metadata,
                embedding::text AS embedding,
                created_at
            FROM document_chunks
            WHERE document_id = %s
            ORDER BY chunk_index ASC;
        """

        with self.db.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (document_id,))
                return cursor.fetchall()

    def delete_by_document_id(self, document_id: UUID) -> int:
        """Delete all chunks belonging to a document."""

        query = """
            DELETE FROM document_chunks
            WHERE document_id = %s;
        """

        with self.db.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (document_id,))
                return cursor.rowcount

    def count_by_document_id(self, document_id: UUID) -> int:
        """Return the number of chunks belonging to a document."""

        query = """
            SELECT COUNT(*) AS total
            FROM document_chunks
            WHERE document_id = %s;
        """

        with self.db.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (document_id,))
                return cursor.fetchone()["total"]
