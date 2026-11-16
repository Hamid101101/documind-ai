
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    chunk_index: int
    content: str
    page_start: int | None
    page_end: int | None
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
