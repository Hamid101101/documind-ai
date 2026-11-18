
import logging

from fastapi import APIRouter, HTTPException, Request, status

from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.services.chat_service import ChatService
from llm.client import OllamaClientError

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


def get_chat_service(request: Request) -> ChatService:
    """Retrieve the shared ChatService instance from application state."""

    return request.app.state.chat_service


@router.post(
    "/",
    response_model=ChatResponse,
    summary="Ask a question about uploaded documents",
)
async def chat(
    payload: ChatRequest,
    request: Request,
) -> ChatResponse:
    """Retrieve relevant document context and generate an answer."""

    service = get_chat_service(request)

    try:
        result = await service.chat(
            question=payload.question,
            conversation_id=payload.conversation_id,
            filters=payload.filters,
            top_k=payload.top_k,
            max_tokens=payload.max_tokens,
        )

        return ChatResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except OllamaClientError as exc:
        logger.exception("LLM generation failed.")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The language model service is currently unavailable.",
        ) from exc

    except Exception as exc:
        logger.exception("Chat request failed.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your question.",
        ) from exc
