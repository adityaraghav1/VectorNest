"""Semantic ingestion and search API schemas."""

from typing import Any

from pydantic import BaseModel, Field

from vectornest.core.types import DistanceMetric, IndexType
from vectornest.models.record import MetadataValue


class DocumentIngestRequest(BaseModel):
    """Request body for document ingestion."""

    document: str = Field(min_length=1)
    document_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentIngestResponse(BaseModel):
    """Response returned after document ingestion."""

    document_id: str
    record_ids: list[str]
    chunks_created: int


class SemanticSearchRequest(BaseModel):
    """Request body for natural-language semantic search."""

    query: str = Field(min_length=1)
    metric: DistanceMetric = DistanceMetric.COSINE
    index_type: IndexType = IndexType.BRUTE_FORCE
    k: int = Field(default=10, gt=0)
    metadata_filter: dict[str, MetadataValue] | None = None