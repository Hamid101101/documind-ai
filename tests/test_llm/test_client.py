
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from llm.client import OllamaClient
from llm.config import LLMConfig


@pytest.fixture
def mock_ollama_client():
    response = SimpleNamespace(
        message=SimpleNamespace(content="Generated answer"),
        model="qwen2.5:3b",
        prompt_eval_count=20,
        eval_count=15,
    )

    with patch("llm.client.AsyncClient") as mock_class:
        mock_client = mock_class.return_value
        mock_client.chat = AsyncMock(return_value=response)
        mock_client.close = AsyncMock()

        yield mock_class, mock_client


@pytest.fixture
def config():
    return LLMConfig(
        provider="ollama",
        base_url="http://localhost:11434",
        model="qwen2.5:3b",
        temperature=0.1,
        timeout=120,
    )


def test_client_initialization(config, mock_ollama_client):
    mock_class, _ = mock_ollama_client

    OllamaClient(config)

    mock_class.assert_called_once_with(
        host=config.base_url,
        timeout=config.timeout,
    )


def test_generate_returns_expected_response(
    config,
    mock_ollama_client,
):
    _, mock_client = mock_ollama_client
    client = OllamaClient(config)

    result = asyncio.run(
        client.generate(
            prompt="What is machine learning?",
            system_prompt="You are a helpful assistant.",
        )
    )

    assert result == {
        "text": "Generated answer",
        "model": "qwen2.5:3b",
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 15,
            "total_tokens": 35,
        },
    }

    mock_client.chat.assert_awaited_once_with(
        model=config.model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "What is machine learning?",
            },
        ],
        options={"temperature": config.temperature},
    )


def test_generate_without_system_prompt(
    config,
    mock_ollama_client,
):
    _, mock_client = mock_ollama_client
    client = OllamaClient(config)

    asyncio.run(
        client.generate(prompt="Explain RAG.")
    )

    mock_client.chat.assert_awaited_once()

    messages = mock_client.chat.await_args.kwargs["messages"]

    assert messages == [
        {
            "role": "user",
            "content": "Explain RAG.",
        }
    ]


@pytest.mark.parametrize("prompt", ["", " ", "\n"])
def test_generate_rejects_empty_prompt(
    config,
    mock_ollama_client,
    prompt,
):
    _, mock_client = mock_ollama_client
    client = OllamaClient(config)

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        asyncio.run(client.generate(prompt=prompt))

    mock_client.chat.assert_not_awaited()


def test_generate_rejects_empty_system_prompt(
    config,
    mock_ollama_client,
):
    _, mock_client = mock_ollama_client
    client = OllamaClient(config)

    with pytest.raises(
        ValueError,
        match="System prompt cannot be empty",
    ):
        asyncio.run(
            client.generate(
                prompt="Explain RAG.",
                system_prompt=" ",
            )
        )

    mock_client.chat.assert_not_awaited()


def test_generate_handles_missing_usage_counts(
    config,
    mock_ollama_client,
):
    _, mock_client = mock_ollama_client
    mock_client.chat.return_value = SimpleNamespace(
        message=SimpleNamespace(content="Answer"),
        model="qwen2.5:3b",
        prompt_eval_count=None,
        eval_count=None,
    )

    client = OllamaClient(config)

    result = asyncio.run(
        client.generate(prompt="Question")
    )

    assert result["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def test_close_releases_client_resources(
    config,
    mock_ollama_client,
):
    _, mock_client = mock_ollama_client
    client = OllamaClient(config)

    asyncio.run(client.close())

    mock_client.close.assert_awaited_once()
