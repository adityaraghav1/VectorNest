"""Vector search request and response schemas."""

from typing import Any

from pydantic import BaseModel, Field

from vectornest.core.types import DistanceMetric, IndexType
from vectornest.models.record import MetadataValue


class SearchRequest(BaseModel):
    """Request body for nearest-neighbour search."""

    query_vector: list[float] = Field(min_length=1)
    metric: DistanceMetric = DistanceMetric.COSINE
    index_type: IndexType = IndexType.BRUTE_FORCE
    k: int = Field(default=10, gt=0)
    metadata_filter: dict[str, MetadataValue] | None = None


class SearchHitResponse(BaseModel):
    """One ranked vector-search result."""

    id: str
    score: float
    vector: list[float]
    metadata: dict[str, Any]
    document: str | None


class SearchResponse(BaseModel):
    """Response containing ranked search results."""

    results: list[SearchHitResponse]