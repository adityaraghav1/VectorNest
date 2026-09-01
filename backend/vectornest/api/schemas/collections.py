"""Collection request and response schemas."""

from pydantic import BaseModel, Field

from vectornest.core.types import DistanceMetric


class CollectionCreateRequest(BaseModel):
    """Request body for creating a collection."""

    name: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    distance_metric: DistanceMetric = DistanceMetric.COSINE


class CollectionResponse(BaseModel):
    """Serialized collection configuration."""

    name: str
    dimension: int
    distance_metric: DistanceMetric