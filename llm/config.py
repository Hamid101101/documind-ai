
import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM providers and generation."""

    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:3b"
    temperature: float = 0.1
    timeout: int = 120

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("LLM provider cannot be empty.")

        if not self.base_url.strip():
            raise ValueError("LLM base URL cannot be empty.")

        if not self.model.strip():
            raise ValueError("LLM model cannot be empty.")

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("LLM temperature must be between 0 and 2.")

        if self.timeout <= 0:
            raise ValueError("LLM timeout must be greater than zero.")

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load LLM configuration from environment variables."""

        load_dotenv()

        return cls(
            provider=os.getenv("LLM_PROVIDER", "ollama"),
            base_url=os.getenv(
                "LLM_BASE_URL",
                "http://localhost:11434",
            ).rstrip("/"),
            model=os.getenv("LLM_MODEL", "qwen2.5:3b"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            timeout=int(os.getenv("LLM_TIMEOUT", "120")),
        )
