
import json
import logging
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from backend.app.schemas.document import DocumentResponse
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
