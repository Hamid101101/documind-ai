
import asyncio
from typing import Any
from uuid import UUID

from backend.app.services.conversation_service import ConversationService
from database.repositories.message_repository import MessageRepository
from llm.generation import GenerationService
from retrieval.filters import RetrievalFilters
from retrieval.vector_search import VectorSearch


class ChatService:
    """Orchestrate conversation management, retrieval and generation."""

    HISTORY_LIMIT = 10

    def __init__(
        self,
        vector_search: VectorSearch | None = None,
        generation_service: GenerationService | None = None,
        conversation_service: ConversationService | None = None,
        message_repository: MessageRepository | None = None,
    ) -> None:
        self.vector_search = (
            vector_search if vector_search is not None else VectorSearch()
        )
        self.generation_service = (
            generation_service
            if generation_service is not None
            else GenerationService()
        )
        self.conversation_service = (
            conversation_service
            if conversation_service is not None
            else ConversationService()
        )
        self.message_repository = (
            message_repository
            if message_repository is not None
            else MessageRepository()
        )

    async def chat(
        self,
        question: str,
        conversation_id: UUID | None = None,
        filters: RetrievalFilters | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Process a question within a conversation."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        question = question.strip()

        if conversation_id is None:
            conversation = self.conversation_service.create_conversation(
                first_question=question
            )
            conversation_id = conversation["id"]
            conversation_history = []
        else:
            conversation = await asyncio.to_thread(
                self.conversation_service.get_conversation,
                conversation_id,
                message_limit=self.HISTORY_LIMIT,
            )

            if conversation is None:
                raise ValueError("Conversation not found.")

            conversation_history = [
                {
                    "role": message["role"],
                    "content": message["content"],
                }
                for message in conversation["messages"]
            ]

        self.message_repository.create(
            conversation_id=conversation_id,
            role="user",
            content=question,
        )

        context_result = await asyncio.to_thread(
            self.vector_search.hybrid_search_with_context,
            query=question,
            filters=filters,
            top_k=top_k,
            max_tokens=max_tokens,
        )

        result = await self.generation_service.generate(
            question=question,
            context_result=context_result,
            conversation_history=conversation_history,
        )

        self.message_repository.create(
            conversation_id=conversation_id,
            role="assistant",
            content=result["answer"],
            sources=result["sources"],
            metadata={
                "model": result["model"],
                "usage": result["usage"],
            },
        )

        return {
            **result,
            "conversation_id": conversation_id,
        }

    async def close(self) -> None:
        """Release the LLM client's resources."""
        await self.generation_service.close()
