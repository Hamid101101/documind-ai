
import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request, status
from starlette.concurrency import run_in_threadpool

from backend.app.schemas.conversation import (
    ConversationCreate,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationResponse,
)
from backend.app.services.conversation_service import ConversationService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"],
)


def get_conversation_service(request: Request) -> ConversationService:
    """Retrieve the shared ConversationService instance."""

    return request.app.state.chat_service.conversation_service


@router.post(
    "/",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a conversation",
)
async def create_conversation(
    payload: ConversationCreate,
    request: Request,
) -> ConversationResponse:
    """Create an empty conversation."""

    service = get_conversation_service(request)

    try:
        conversation = await run_in_threadpool(
            service.create_empty_conversation,
            title=payload.title,
            metadata=payload.metadata,
        )
        return ConversationResponse(**conversation)

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Conversation creation failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while creating the conversation.",
        ) from exc


@router.get(
    "/",
    response_model=ConversationListResponse,
    summary="List conversations",
)
async def list_conversations(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> ConversationListResponse:
    """List conversations with pagination."""

    service = get_conversation_service(request)

    try:
        result = await run_in_threadpool(
            service.list_conversations,
            page=page,
            page_size=page_size,
        )
        return ConversationListResponse(**result)

    except Exception as exc:
        logger.exception("Conversation listing failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving conversations.",
        ) from exc


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
    summary="Retrieve a conversation",
)
async def get_conversation(
    conversation_id: UUID,
    request: Request,
) -> ConversationDetailResponse:
    """Retrieve a conversation and its message history."""

    service = get_conversation_service(request)

    try:
        conversation = await run_in_threadpool(
            service.get_conversation,
            conversation_id,
        )

        if conversation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found.",
            )

        return ConversationDetailResponse(**conversation)

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Conversation retrieval failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving the conversation.",
        ) from exc


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a conversation",
)
async def delete_conversation(
    conversation_id: UUID,
    request: Request,
) -> None:
    """Delete a conversation and its associated messages."""

    service = get_conversation_service(request)

    try:
        deleted = await run_in_threadpool(
            service.delete_conversation,
            conversation_id,
        )

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found.",
            )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Conversation deletion failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while deleting the conversation.",
        ) from exc
