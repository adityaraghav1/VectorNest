"""FastAPI application entry point for VectorNest."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from vectornest.api.errors import register_exception_handlers
from vectornest.api.routes import (
    collections_router,
    health_router,
    projection_router,
    rag_router,
    records_router,
    search_router,
    semantic_router,
)


def create_app() -> FastAPI:
    """Create and configure the VectorNest API application."""
    application = FastAPI(
        title="VectorNest",
        description=(
            "Educational vector database with brute-force, "
            "KD-tree and HNSW search."
        ),
        version="0.1.0",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5500",
            "http://localhost:5500",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(application)

    application.include_router(health_router)
    application.include_router(collections_router)
    application.include_router(records_router)
    application.include_router(search_router)
    application.include_router(semantic_router)
    application.include_router(projection_router)
    application.include_router(rag_router)

    return application


app = create_app()