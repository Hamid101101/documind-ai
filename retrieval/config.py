
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    max_top_k: int = 20
    similarity_threshold: float | None = None
    max_context_tokens: int = 4000
    rrf_k: int = 60

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.max_top_k <= 0:
            raise ValueError("max_top_k must be greater than zero.")

        if self.top_k > self.max_top_k:
            raise ValueError("top_k cannot exceed max_top_k.")

        if self.similarity_threshold is not None:
            if not -1.0 <= self.similarity_threshold <= 1.0:
                raise ValueError(
                    "similarity_threshold must be between -1 and 1."
                )

        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be greater than zero.")

        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")
    
