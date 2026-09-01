"""Request and response schemas for the VectorNest API."""

from vectornest.api.schemas.collections import (
    CollectionCreateRequest,
    CollectionResponse,
)
from vectornest.api.schemas.common import HealthResponse
from vectornest.api.schemas.records import (
    RecordCreateRequest,
    RecordResponse,
    RecordUpdateRequest,
)
from vectornest.api.schemas.search import (
    SearchHitResponse,
    SearchRequest,
    SearchResponse,
)

__all__ = [
    "CollectionCreateRequest",
    "CollectionResponse",
    "HealthResponse",
    "RecordCreateRequest",
    "RecordResponse",
    "RecordUpdateRequest",
    "SearchHitResponse",
    "SearchRequest",
    "SearchResponse",
]