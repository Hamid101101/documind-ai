
import asyncio
from typing import Any

from llm.generation import GenerationService
from retrieval.filters import RetrievalFilters
from retrieval.vector_search import VectorSearch


class ChatService:
    """Orchestrate document retrieval and LLM answer generation."""

    def __init__(
        self,
        vector_search: VectorSearch | None = None,
        generation_service: GenerationService | None = None,
    ) -> None:
        self.vector_search = (
            vector_search if vector_search is not None else VectorSearch()
        )
        self.generation_service = (
            generation_service
            if generation_service is not None
            else GenerationService()
        )

    async def chat(
        self,
        question: str,
        filters: RetrievalFilters | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Retrieve relevant document context and generate an answer."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        context_result = await asyncio.to_thread(
            self.vector_search.hybrid_search_with_context,
            query=question.strip(),
            filters=filters,
            top_k=top_k,
            max_tokens=max_tokens,
        )

        return await self.generation_service.generate(
            question=question.strip(),
            context_result=context_result,
        )

    async def close(self) -> None:
        """Release the LLM client's resources."""
        await self.generation_service.close()
