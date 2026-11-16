
from fastapi import APIRouter, HTTPException, status

from infra.postgres.postgres_client import PostgreSQLClient
from infra.storage.minio_client import MinIOClient

router = APIRouter(prefix="/health", tags=["Health"])

postgres_client = PostgreSQLClient()
minio_client = MinIOClient()


@router.get("/live")
def liveness_check():
    """Check whether the API process is running."""
    return {
        "status": "alive",
        "service": "documind-api",
    }


@router.get("/ready")
def readiness_check():
    """Check whether required infrastructure services are available."""

    postgres_status = postgres_client.check_connection()
    minio_status = minio_client.check_connection()

    services = {
        "postgres": "connected" if postgres_status else "unavailable",
        "minio": "connected" if minio_status else "unavailable",
    }

    is_ready = postgres_status and minio_status

    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "services": services,
            },
        )

    return {
        "status": "ready",
        "services": services,
    }
