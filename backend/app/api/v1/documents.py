
import json
import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from backend.app.schemas.document import (
    DocumentListResponse,
    DocumentResponse,
    DocumentUpdate,
)
from backend.app.services.document_service import (
    DocumentService,
    DocumentValidationError,
)
from database.repositories.document_repository import DocumentRepository
from worker.tasks import process_document

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

document_service = DocumentService()
document_repository = DocumentRepository()


@router.post(
    "/",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document",
)
async def upload_document(
    file: UploadFile = File(...),
    metadata: str = Form("{}"),
) -> DocumentResponse:
    """Upload a document and dispatch asynchronous processing."""

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A filename is required.",
        )

    # Parse optional JSON metadata.
    try:
        parsed_metadata: Any = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Metadata must be valid JSON.",
        ) from exc

    if not isinstance(parsed_metadata, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Metadata must be a JSON object.",
        )

    try:
        # Determine actual upload size.
        file.file.seek(0, 2)
        file_size_bytes = file.file.tell()
        file.file.seek(0)

        # 1. Upload file to MinIO and create PostgreSQL record.
        document = await run_in_threadpool(
            document_service.upload_document,
            file.file,
            file.filename,
            file_size_bytes,
            parsed_metadata,
        )

        document_id = document["id"]

        # 2. Mark document as queued before dispatch.
        queued_document = await run_in_threadpool(
            document_repository.update_status,
            document_id,
            "queued",
        )

        if queued_document is None:
            raise RuntimeError(
                f"Could not mark document {document_id} as queued."
            )

        # 3. Dispatch asynchronous processing to Celery.
        try:
            task_result = await run_in_threadpool(
                process_document.delay,
                str(document_id),
            )

            logger.info(
                "Document %s queued successfully (task_id=%s).",
                document_id,
                task_result.id,
            )

        except Exception as exc:
            logger.exception(
                "Failed to dispatch document %s to Celery.",
                document_id,
            )

            # Do not leave the document stuck in queued.
            try:
                await run_in_threadpool(
                    document_repository.update_status,
                    document_id,
                    "failed",
                    f"Failed to dispatch Celery task: {str(exc)[:4900]}",
                )
            except Exception:
                logger.exception(
                    "Could not update failed status for document %s.",
                    document_id,
                )

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Document uploaded successfully, but processing "
                    "could not be scheduled. Please retry later."
                ),
            ) from exc

        # 4. Return the queued document.
        return DocumentResponse(**queued_document)

    except DocumentValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Document upload or scheduling failed: %s",
            file.filename,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while uploading the document.",
        ) from exc

    finally:
        await file.close()

@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List documents",
)
async def list_documents(
    page: int = 1,
    page_size: int = 20,
    status_filter: str | None = None,
) -> DocumentListResponse:
    """List documents with pagination and optional status filtering."""

    if page < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page must be greater than or equal to 1.",
        )

    if not 1 <= page_size <= 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page size must be between 1 and 100.",
        )

    try:
        result = await run_in_threadpool(
            document_service.list_documents,
            page=page,
            page_size=page_size,
            status=status_filter,
        )
        return DocumentListResponse(**result)

    except Exception as exc:
        logger.exception("Failed to list documents.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving documents.",
        ) from exc

@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get a document",
)
async def get_document(document_id: str) -> DocumentResponse:
    """Retrieve a document by its ID."""

    try:
        document = await run_in_threadpool(
            document_service.get_document,
            document_id,
        )

        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )

        return DocumentResponse(**document)

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID.",
        ) from exc

    except Exception as exc:
        logger.exception("Failed to retrieve document %s.", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving the document.",
        ) from exc

@router.patch(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Update a document",
)
async def update_document(
    document_id: str,
    payload: DocumentUpdate,
) -> DocumentResponse:
    """Update a document's filename or metadata."""

    try:
        document = await run_in_threadpool(
            document_service.update_document,
            document_id,
            filename=payload.filename,
            metadata=payload.metadata,
        )

        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )

        return DocumentResponse(**document)

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Failed to update document %s.", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while updating the document.",
        ) from exc

@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
)
async def delete_document(document_id: str) -> None:
    """Delete a document and its associated data."""

    try:
        deleted = await run_in_threadpool(
            document_service.delete_document,
            document_id,
        )

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID.",
        ) from exc

    except Exception as exc:
        logger.exception("Failed to delete document %s.", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while deleting the document.",
        ) from exc

@router.get(
    "/{document_id}/download",
    summary="Download a document",
)
async def download_document(document_id: UUID):
    """Stream the original document from MinIO."""

    try:
        result = await run_in_threadpool(
            document_service.open_document_stream,
            document_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )

        document, storage_response = result
        body = storage_response["Body"]

        def stream_file():
            try:
                while chunk := body.read(1024 * 1024):
                    yield chunk
            finally:
                body.close()

        return StreamingResponse(
            stream_file(),
            media_type=document["mime_type"],
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{document["filename"]}"'
                ),
                "Content-Length": str(storage_response["ContentLength"]),
            },
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Failed to download document %s.",
            document_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while downloading the document.",
        ) from exc