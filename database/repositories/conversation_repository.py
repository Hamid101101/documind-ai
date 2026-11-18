
from typing import Any
from uuid import UUID

from psycopg.types.json import Jsonb

from infra.postgres.postgres_client import PostgreSQLClient


class ConversationRepository:
    """Repository for conversation records stored in PostgreSQL."""

    def __init__(self, db: PostgreSQLClient | None = None):
        self.db = db or PostgreSQLClient()

    def create(
        self,
        *,
        title: str = "New conversation",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a conversation and return its record."""

        query = """
            INSERT INTO conversations (title, metadata)
            VALUES (%s, %s)
            RETURNING *;
        """

        params = (title, Jsonb(metadata or {}))

        with self.db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()

    def get_by_id(self, conversation_id: UUID) -> dict[str, Any] | None:
        """Retrieve a conversation by UUID."""

        query = """
            SELECT *
            FROM conversations
            WHERE id = %s;
        """

        with self.db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(query, (conversation_id,))
            return cursor.fetchone()

    def list_conversations(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        """List conversations with pagination."""

        if page < 1:
            raise ValueError("Page must be at least 1.")

        if not 1 <= page_size <= 100:
            raise ValueError("Page size must be between 1 and 100.")

        offset = (page - 1) * page_size

        count_query = """
            SELECT COUNT(*) AS total
            FROM conversations;
        """

        conversations_query = """
            SELECT *
            FROM conversations
            ORDER BY updated_at DESC, created_at DESC
            LIMIT %s OFFSET %s;
        """

        with self.db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(count_query)
            total = cursor.fetchone()["total"]

            cursor.execute(conversations_query, (page_size, offset))
            conversations = cursor.fetchall()

        return {
            "conversations": conversations,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def delete(self, conversation_id: UUID) -> bool:
        """Delete a conversation and its associated messages."""

        query = """
            DELETE FROM conversations
            WHERE id = %s
            RETURNING id;
        """

        with self.db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(query, (conversation_id,))
            return cursor.fetchone() is not None
