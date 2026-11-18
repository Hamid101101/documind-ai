
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ConversationCreate(BaseModel):
    """Request schema for creating a conversation."""

    title: str = Field(default="New conversation", min_length=1, max_length=255)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationResponse(BaseModel):
    """Response schema for conversation details."""

    id: UUID
    title: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ConversationListResponse(BaseModel):
    """Response schema for paginated conversation lists."""

    conversations: list[ConversationResponse]
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)


class MessageResponse(BaseModel):
    """Response schema for an individual conversation message."""

    id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ConversationDetailResponse(ConversationResponse):
    """Response schema for a conversation with its message history."""

    messages: list[MessageResponse] = Field(default_factory=list)
