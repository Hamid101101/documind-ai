
from typing import Any
from uuid import UUID

from database.repositories.conversation_repository import ConversationRepository
from database.repositories.message_repository import MessageRepository


class ConversationService:
    """Service for managing conversations and their message history."""

    TITLE_MAX_LENGTH = 60

    def __init__(
        self,
        conversation_repository: ConversationRepository | None = None,
        message_repository: MessageRepository | None = None,
    ):
        self.conversation_repository = (
            conversation_repository
            if conversation_repository is not None
            else ConversationRepository()
        )
        self.message_repository = (
            message_repository
            if message_repository is not None
            else MessageRepository()
        )

    @classmethod
    def _generate_title(cls, question: str) -> str:
        """Generate a short conversation title from the first question."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        title = " ".join(question.split())

        if len(title) > cls.TITLE_MAX_LENGTH:
            title = title[: cls.TITLE_MAX_LENGTH - 3].rstrip() + "..."

        return title

    def create_conversation(
        self,
        first_question: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a conversation using its first question as the title."""

        title = self._generate_title(first_question)

        return self.conversation_repository.create(
            title=title,
            metadata=metadata,
        )

    def get_conversation(
        self,
        conversation_id: UUID,
        *,
        message_limit: int | None = None,
    ) -> dict[str, Any] | None:
        """Retrieve a conversation with its message history."""

        conversation = self.conversation_repository.get_by_id(conversation_id)

        if conversation is None:
            return None

        messages = self.message_repository.get_by_conversation_id(
            conversation_id,
            limit=message_limit,
        )

        return {
            **conversation,
            "messages": messages,
        }

    def list_conversations(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        """List conversations with pagination."""

        return self.conversation_repository.list_conversations(
            page=page,
            page_size=page_size,
        )

    def delete_conversation(self, conversation_id: UUID) -> bool:
        """Delete a conversation and its associated messages."""

        return self.conversation_repository.delete(conversation_id)

    def create_empty_conversation(
        self,
        title: str = "New conversation",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an empty conversation."""

        if not title or not title.strip():
            raise ValueError("Conversation title cannot be empty.")

        return self.conversation_repository.create(
            title=title.strip(),
            metadata=metadata,
        )
