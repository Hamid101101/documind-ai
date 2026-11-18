
import json
from typing import Any, ClassVar
from uuid import UUID

from psycopg.types.json import Jsonb

from infra.postgres.postgres_client import PostgreSQLClient


class MessageRepository:
    """Repository for conversation messages stored in PostgreSQL."""

    VALID_ROLES: ClassVar[set[str]] = {"user", "assistant"}

    def __init__(self, db: PostgreSQLClient | None = None):
        self.db = db or PostgreSQLClient()

    def create(
        self,
        *,
        conversation_id: UUID,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a message and update its conversation timestamp."""

        if role not in self.VALID_ROLES:
            raise ValueError("Role must be either 'user' or 'assistant'.")

        if not content or not content.strip():
            raise ValueError("Message content cannot be empty.")

        query = """
            INSERT INTO messages (
                conversation_id,
                role,
                content,
                sources,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s)
            RETURNING *;
        """

        params = (
            conversation_id,
            role,
            content.strip(),
            Jsonb(sources or [], dumps=lambda value: json.dumps(value, default=str)),
            Jsonb(metadata or {}, dumps=lambda value: json.dumps(value, default=str)),
        )

        update_query = """
            UPDATE conversations
            SET updated_at = NOW()
            WHERE id = %s;
        """

        with self.db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(query, params)
            message = cursor.fetchone()

            cursor.execute(update_query, (conversation_id,))

            if cursor.rowcount == 0:
                raise ValueError("Conversation does not exist.")

            return message

    def get_by_id(self, message_id: UUID) -> dict[str, Any] | None:
        """Retrieve a message by UUID."""

        query = """
            SELECT *
            FROM messages
            WHERE id = %s;
        """

        with self.db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(query, (message_id,))
            return cursor.fetchone()

    def get_by_conversation_id(
        self,
        conversation_id: UUID,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve conversation messages in chronological order."""

        if limit is not None and limit <= 0:
            raise ValueError("Limit must be greater than zero.")

        query = """
            SELECT *
            FROM messages
            WHERE conversation_id = %s
            ORDER BY created_at ASC, id ASC
        """

        params: list[Any] = [conversation_id]

        if limit is not None:
            query += " LIMIT %s"
            params.append(limit)

        with self.db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(query, tuple(params))
            return cursor.fetchall()

    def count_by_conversation_id(self, conversation_id: UUID) -> int:
        """Return the number of messages in a conversation."""

        query = """
            SELECT COUNT(*) AS total
            FROM messages
            WHERE conversation_id = %s;
        """

        with self.db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(query, (conversation_id,))
            return cursor.fetchone()["total"]
