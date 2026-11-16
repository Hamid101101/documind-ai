
from abc import ABC, abstractmethod
from typing import Sequence

from ingestion.chunking.base import DocumentChunk


class BaseEmbedder(ABC):

    @abstractmethod
    def embed_chunks(
        self,
        chunks: Sequence[DocumentChunk],
    ) -> list[list[float]]:
        """Generate one embedding vector for each document chunk."""
        raise NotImplementedError
