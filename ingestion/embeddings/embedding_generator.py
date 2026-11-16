
from typing import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ingestion.chunking.base import DocumentChunk
from ingestion.embeddings.base import BaseEmbedder
from ingestion.embeddings.config import EmbeddingConfig


class EmbeddingGenerator(BaseEmbedder):
    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        model=None,
    ) -> None:
        self.config = config or EmbeddingConfig()
        self._model = model

    @property
    def model(self) -> SentenceTransformer:
        """Load the embedding model only when it is first needed."""
        if self._model is None:
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self._resolve_device(),
            )

        return self._model

    def _resolve_device(self) -> str:
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"

        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but no CUDA-compatible GPU is available."
            )

        return self.config.device

    def embed_chunks(
        self,
        chunks: Sequence[DocumentChunk],
    ) -> list[list[float]]:
        if not chunks:
            return []

        texts = [
            f"{self.config.document_prefix}{chunk.content}"
            for chunk in chunks
        ]

        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        expected_shape = (
            len(chunks),
            self.config.expected_dimension,
        )

        if embeddings.shape != expected_shape:
            raise ValueError(
                f"Expected embeddings with shape {expected_shape}, "
                f"received {embeddings.shape}."
            )

        if not np.isfinite(embeddings).all():
            raise ValueError("Generated embeddings contain invalid values.")

        return embeddings.tolist()
