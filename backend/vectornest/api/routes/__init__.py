"""API route modules for VectorNest."""

from vectornest.api.routes.collections import router as collections_router
from vectornest.api.routes.health import router as health_router
from vectornest.api.routes.records import router as records_router
from vectornest.api.routes.search import router as search_router

__all__ = [
    "collections_router",
    "health_router",
    "records_router",
    "search_router",
]