
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    mime_type: str
    file_extension: str | None = None
    file_size_bytes: int
    file_checksum: str | None = None
    status: DocumentStatus
    page_count: int | None = None
    chunk_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    processing_started_at: datetime | None = None
    processing_completed_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)


class DocumentUpdate(BaseModel):
    """Request schema for updating document information."""

    model_config = ConfigDict(extra="forbid")

    filename: str | None = Field(default=None, min_length=1, max_length=255)
    metadata: dict[str, Any] | None = None
