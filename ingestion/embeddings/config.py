
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = os.getenv(
        "EMBEDDING_MODEL",
        "intfloat/multilingual-e5-small",
    )
    batch_size: int = 32
    device: str = "auto"
    normalize_embeddings: bool = True
    expected_dimension: int = 768
    document_prefix: str = "passage: "
    query_prefix: str = "query: "

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("Model name cannot be empty.")

        if self.batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")

        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Device must be 'auto', 'cpu', or 'cuda'.")

        if self.expected_dimension <= 0:
            raise ValueError("Expected dimension must be positive.")
