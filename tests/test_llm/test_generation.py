
import asyncio
from unittest.mock import AsyncMock

import pytest

from llm.generation import GenerationService


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.generate.return_value = {
        "text": "Machine learning is a subset of AI.",
        "model": "qwen2.5:3b",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        },
    }
    return client


@pytest.fixture
def generation_service(mock_client):
    return GenerationService(client=mock_client)


@pytest.fixture
def context_result():
    return {
        "context": "[Source 1: document.pdf, Page 1]\nMachine learning is a subset of AI.",
        "sources": [
            {
                "filename": "document.pdf",
                "page_start": 1,
                "page_end": 1,
            }
        ],
        "total_tokens": 20,
        "included_chunks": 1,
    }


def test_generate_returns_expected_response(
    generation_service,
    mock_client,
    context_result,
):
    result = asyncio.run(
        generation_service.generate(
            question="What is machine learning?",
            context_result=context_result,
        )
    )

    assert result == {
        "answer": "Machine learning is a subset of AI.",
        "sources": context_result["sources"],
        "model": "qwen2.5:3b",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        },
    }

    mock_client.generate.assert_awaited_once()


def test_generate_passes_question_and_context_to_prompt(
    generation_service,
    mock_client,
    context_result,
):
    asyncio.run(
        generation_service.generate(
            question="What is machine learning?",
            context_result=context_result,
        )
    )

    call_kwargs = mock_client.generate.await_args.kwargs

    assert "What is machine learning?" in call_kwargs["prompt"]
    assert context_result["context"] in call_kwargs["prompt"]
    assert call_kwargs["system_prompt"] == generation_service.system_prompt


def test_generate_strips_question_whitespace(
    generation_service,
    mock_client,
    context_result,
):
    asyncio.run(
        generation_service.generate(
            question="  What is machine learning?  ",
            context_result=context_result,
        )
    )

    prompt = mock_client.generate.await_args.kwargs["prompt"]

    assert "USER QUESTION:\nWhat is machine learning?" in prompt


def test_generate_returns_fallback_when_context_is_empty(
    generation_service,
    mock_client,
):
    result = asyncio.run(
        generation_service.generate(
            question="What is machine learning?",
            context_result={
                "context": "",
                "sources": [],
            },
        )
    )

    assert result["sources"] == []
    assert result["model"] is None
    assert result["usage"]["total_tokens"] == 0
    assert "couldn't find relevant information" in result["answer"]

    mock_client.generate.assert_not_awaited()


def test_generate_returns_fallback_when_context_is_whitespace(
    generation_service,
    mock_client,
):
    result = asyncio.run(
        generation_service.generate(
            question="What is machine learning?",
            context_result={
                "context": "   \n ",
                "sources": [],
            },
        )
    )

    assert result["model"] is None
    mock_client.generate.assert_not_awaited()


@pytest.mark.parametrize("question", ["", " ", "\n"])
def test_generate_rejects_empty_question(
    generation_service,
    mock_client,
    context_result,
    question,
):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        asyncio.run(
            generation_service.generate(
                question=question,
                context_result=context_result,
            )
        )

    mock_client.generate.assert_not_awaited()


def test_generate_rejects_invalid_context_result_type(
    generation_service,
):
    with pytest.raises(TypeError, match="Context result must be a dictionary"):
        asyncio.run(
            generation_service.generate(
                question="Question",
                context_result="invalid",
            )
        )


def test_generate_rejects_invalid_context_type(
    generation_service,
):
    with pytest.raises(TypeError, match="Retrieved context must be a string"):
        asyncio.run(
            generation_service.generate(
                question="Question",
                context_result={
                    "context": 123,
                    "sources": [],
                },
            )
        )


def test_generate_rejects_invalid_sources_type(
    generation_service,
):
    with pytest.raises(TypeError, match="Retrieved sources must be a list"):
        asyncio.run(
            generation_service.generate(
                question="Question",
                context_result={
                    "context": "Some context",
                    "sources": "invalid",
                },
            )
        )


def test_close_closes_ollama_client(
    generation_service,
    mock_client,
):
    asyncio.run(generation_service.close())

    mock_client.close.assert_awaited_once()


def test_prompt_templates_are_loaded(generation_service):
    assert generation_service.system_prompt
    assert generation_service.rag_prompt


def test_build_prompt_injects_context_and_question(
    generation_service,
):
    prompt = generation_service._build_prompt(
        question="What is RAG?",
        context="RAG combines retrieval and generation.",
    )

    assert "What is RAG?" in prompt
    assert "RAG combines retrieval and generation." in prompt
