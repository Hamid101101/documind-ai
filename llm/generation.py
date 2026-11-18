
from pathlib import Path
from typing import Any

from llm.client import OllamaClient
from llm.config import LLMConfig

PROMPTS_DIR = Path(__file__).parent / "prompts"


class GenerationService:
    """Orchestrate prompt construction and LLM answer generation."""

    def __init__(
        self,
        client: OllamaClient | None = None,
    ) -> None:
        self.client = client or OllamaClient(LLMConfig.from_env())

        self.system_prompt = self._load_prompt("system_prompt.txt")
        self.rag_prompt = self._load_prompt("rag_prompt.txt")

    @staticmethod
    def _load_prompt(filename: str) -> str:
        """Load a prompt template from the prompts directory."""

        prompt_path = PROMPTS_DIR / filename

        if not prompt_path.is_file():
            raise FileNotFoundError(
                f"Prompt template not found: {prompt_path}"
            )

        return prompt_path.read_text(encoding="utf-8").strip()

    def _build_prompt(
        self,
        question: str,
        context: str,
    ) -> str:
        """Inject the question and retrieved context into the RAG template."""

        return self.rag_prompt.format(
            context=context,
            question=question,
        )

    async def generate(
        self,
        question: str,
        context_result: dict[str, Any],
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Generate a grounded answer from retrieved document context."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not isinstance(context_result, dict):
            raise TypeError("Context result must be a dictionary.")

        context = context_result.get("context", "")
        sources = context_result.get("sources", [])

        if not isinstance(context, str):
            raise TypeError("Retrieved context must be a string.")

        if not isinstance(sources, list):
            raise TypeError("Retrieved sources must be a list.")

        if not context.strip():
            return {
                "answer": (
                    "I couldn't find relevant information in the "
                    "available documents to answer your question."
                ),
                "sources": [],
                "model": None,
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }

        prompt = self._build_prompt(
            question=question.strip(),
            context=context,
        )

        response = await self.client.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            conversation_history=conversation_history,
        )

        return {
            "answer": response["text"],
            "sources": sources,
            "model": response["model"],
            "usage": response["usage"],
        }

    async def close(self) -> None:
        """Release the underlying Ollama client resources."""
        await self.client.close()
