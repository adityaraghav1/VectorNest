"""RAG API schemas."""

from typing import Any

from pydantic import BaseModel, Field

from vectornest.core.types import DistanceMetric, IndexType
from vectornest.models.record import MetadataValue


class RAGRequest(BaseModel):
    """Request body for grounded RAG generation."""

    question: str = Field(min_length=1)

    metric: DistanceMetric = (
        DistanceMetric.COSINE
    )

    index_type: IndexType = (
        IndexType.BRUTE_FORCE
    )

    k: int = Field(
        default=4,
        gt=0,
    )

    metadata_filter: (
        dict[str, MetadataValue] | None
    ) = None


class RAGSourceResponse(BaseModel):
    """One source used in a generated RAG answer."""

    id: str
    score: float
    document: str
    metadata: dict[str, Any]


class RAGResponseSchema(BaseModel):
    """Generated RAG answer and sources."""

    answer: str
    sources: list[RAGSourceResponse]