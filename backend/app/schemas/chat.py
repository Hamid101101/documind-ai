
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from retrieval.filters import RetrievalFilters


class ChatRequest(BaseModel):
    """Request schema for document-based question answering."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    filters: RetrievalFilters | None = None
    top_k: int | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)


class ChatUsage(BaseModel):
    """Token usage information returned by the LLM."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """Response schema for document-based question answering."""

    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    model: str | None = None
    usage: ChatUsage
