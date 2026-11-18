
from typing import Any
from uuid import UUID

from psycopg.types.json import Jsonb

from infra.postgres.postgres_client import PostgreSQLClient


class DocumentRepository:
    """Repository for document metadata stored in PostgreSQL."""

    def __init__(self, db: PostgreSQLClient | None = None):
        self.db = db or PostgreSQLClient()

    def create(
        self,
        *,
        filename: str,
        mime_type: str,
        file_extension: str | None,
        file_size_bytes: int,
        file_checksum: str | None,
        storage_bucket: str,
        storage_object_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Insert a document record and return it."""

        query = """
            INSERT INTO documents (
                filename,
                mime_type,
                file_extension,
                file_size_bytes,
                file_checksum,
                storage_bucket,
                storage_object_key,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *;
        """

        params = (
            filename,
            mime_type,
            file_extension,
            file_size_bytes,
            file_checksum,
            storage_bucket,
            storage_object_key,
            Jsonb(metadata or {}),
        )

        with self.db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()

    def get_by_id(self, document_id: UUID) -> dict[str, Any] | None:
        """Retrieve a document by UUID."""

        query = """
            SELECT *
            FROM documents
            WHERE id = %s;
        """

        with self.db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(query, (document_id,))
            return cursor.fetchone()

    def get_by_object_key(
        self,
        storage_bucket: str,
        storage_object_key: str,
    ) -> dict[str, Any] | None:
        """Retrieve a document by its MinIO storage reference."""

        query = """
            SELECT *
            FROM documents
            WHERE storage_bucket = %s
              AND storage_object_key = %s;
        """

        with self.db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(
                query,
                (storage_bucket, storage_object_key),
            )
            return cursor.fetchone()

    def list_documents(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        status: str | None = None,
    ) -> dict[str, Any]:
        """List documents with pagination and optional status filtering."""

        if page < 1:
            raise ValueError("Page must be at least 1.")

        if not 1 <= page_size <= 100:
            raise ValueError("Page size must be between 1 and 100.")

        offset = (page - 1) * page_size

        where_clause = ""
        params: list[Any] = []

        if status is not None:
            where_clause = "WHERE status = %s"
            params.append(status)

        count_query = f"""
            SELECT COUNT(*) AS total
            FROM documents
            {where_clause};
        """

        documents_query = f"""
            SELECT *
            FROM documents
            {where_clause}
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s;
        """

        with self.db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(count_query, params)
            total = cursor.fetchone()["total"]

            cursor.execute(
                documents_query,
                (*params, page_size, offset),
            )
            documents = cursor.fetchall()

        return {
            "documents": documents,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def update_status(
        self,
        document_id: UUID,
        status: str,
        error_message: str | None = None,
    ) -> dict[str, Any] | None:
        """Update document processing status."""

        query = """
            UPDATE documents
            SET
                status = %s,
                error_message = %s,
                updated_at = NOW()
            WHERE id = %s
            RETURNING *;
        """

        with self.db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(
                query,
                (status, error_message, document_id),
            )
            return cursor.fetchone()

    def delete_record(self, document_id: UUID) -> bool:
        """Delete a document record and its associated chunks."""

        query = """
            DELETE FROM documents
            WHERE id = %s
            RETURNING id;
        """

        with self.db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(query, (document_id,))
            return cursor.fetchone() is not None

    def mark_processing(self, document_id: UUID) -> dict[str, Any] | None:
        """Mark a document as processing."""

        query = """
            UPDATE documents
            SET
                status = 'processing',
                error_message = NULL,
                processing_started_at = NOW(),
                processing_completed_at = NULL,
                updated_at = NOW()
            WHERE id = %s
            RETURNING *;
        """

        with self.db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(query, (document_id,))
            return cursor.fetchone()

    def mark_completed(
        self,
        document_id: UUID,
        *,
        chunk_count: int,
        page_count: int | None,
        embedding_model: str,
    ) -> dict[str, Any] | None:
        """Mark a document as completed and save processing metadata."""

        query = """
            UPDATE documents
            SET
                status = 'completed',
                chunk_count = %s,
                page_count = %s,
                embedding_model = %s,
                error_message = NULL,
                processing_completed_at = NOW(),
                updated_at = NOW()
            WHERE id = %s
            RETURNING *;
        """

        with self.db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(
                query,
                (chunk_count, page_count, embedding_model, document_id),
            )
            return cursor.fetchone()


    def update_document(
        self,
        document_id: UUID,
        *,
        filename: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Update document information and return the updated record."""

        updates = {}
        params = []

        if filename is not None:
            updates["filename"] = "%s"
            params.append(filename)

        if metadata is not None:
            updates["metadata"] = "%s"
            params.append(Jsonb(metadata))

        if not updates:
            raise ValueError("At least one field must be provided.")

        set_clause = ", ".join(
            f"{column} = {placeholder}"
            for column, placeholder in updates.items()
        )

        query = f"""
            UPDATE documents
            SET {set_clause},
                updated_at = NOW()
            WHERE id = %s
            RETURNING *;
        """

        params.append(document_id)

        with self.db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()
