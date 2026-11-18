
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api.v1.chat import router as chat_router
from backend.app.api.v1.conversations import router as conversations_router
from backend.app.api.v1.documents import router as documents_router
from backend.app.api.v1.health import router as health_router
from backend.app.services.chat_service import ChatService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application-wide service initialization and cleanup."""

    chat_service = ChatService()
    app.state.chat_service = chat_service

    try:
        yield
    finally:
        await chat_service.close()


app = FastAPI(
    title="DocuMind AI",
    description="AI-powered document knowledge assistant",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(
    health_router,
    prefix="/api/v1",
)

app.include_router(
    documents_router,
    prefix="/api/v1",
)

app.include_router(
    chat_router,
    prefix="/api/v1",
)

app.include_router(
    conversations_router,
    prefix="/api/v1",
)


@app.get("/")
def root():
    return {
        "name": "DocuMind AI",
        "message": "Document intelligence API",
        "docs": "/docs",
    }
