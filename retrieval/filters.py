
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator


class RetrievalFilters(BaseModel):
    """Optional filters used to scope document chunk retrieval."""

    model_config = ConfigDict(extra="forbid")

    document_ids: list[UUID] | None = None
    document_metadata: dict[str, Any] | None = None
    chunk_metadata: dict[str, Any] | None = None

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(
        cls,
        value: list[UUID] | None,
    ) -> list[UUID] | None:
        if value is not None and not value:
            raise ValueError(
                "document_ids cannot be empty. Use None for global search."
            )

        if value is not None and len(value) != len(set(value)):
            raise ValueError("document_ids must not contain duplicates.")

        return value
