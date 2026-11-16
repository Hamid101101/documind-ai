
from fastapi import FastAPI

from backend.app.api.v1.health import router as health_router
from backend.app.api.v1.documents import router as documents_router

app = FastAPI(
    title="DocuMind AI",
    description="AI-powered document knowledge assistant",
    version="0.1.0",
)

app.include_router(
    health_router,
    prefix="/api/v1",
)

app.include_router(
    documents_router,
    prefix="/api/v1",
)


@app.get("/")
def root():
    return {
        "name": "DocuMind AI",
        "message": "Document intelligence API",
        "docs": "/docs",
    }
