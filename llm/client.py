
from typing import Any

from ollama import AsyncClient, ResponseError

from llm.config import LLMConfig


class OllamaClientError(Exception):
    """Raised when communication with the Ollama service fails."""


class OllamaClient:
    """Asynchronous client for generating responses with Ollama."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = AsyncClient(
            host=config.base_url,
            timeout=config.timeout,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Generate a response using the configured Ollama model."""

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        messages = []

        if system_prompt is not None:
            if not system_prompt.strip():
                raise ValueError("System prompt cannot be empty.")

            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        for message in conversation_history or []:
            role = message.get("role")
            content = message.get("content")

            if role not in {"user", "assistant"}:
                raise ValueError("Conversation history contains an invalid role.")

            if not isinstance(content, str) or not content.strip():
                raise ValueError("Conversation history contains empty content.")

            messages.append({
                "role": role,
                "content": content,
            })

        messages.append({
            "role": "user",
            "content": prompt,
        })

        try:
            response = await self._client.chat(
                model=self.config.model,
                messages=messages,
                options={
                    "temperature": self.config.temperature,
                },
            )

        except (ResponseError, ConnectionError, TimeoutError) as exc:
            raise OllamaClientError(
                f"Ollama generation failed: {exc}"
            ) from exc

        prompt_tokens = response.prompt_eval_count or 0
        completion_tokens = response.eval_count or 0

        return {
            "text": response.message.content,
            "model": response.model,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()
