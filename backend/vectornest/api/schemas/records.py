"""Vector record request and response schemas."""

from typing import Any

from pydantic import BaseModel, Field


class RecordCreateRequest(BaseModel):
    """Request body for inserting a vector record."""

    id: str = Field(min_length=1)
    vector: list[float] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    document: str | None = None


class RecordUpdateRequest(BaseModel):
    """Request body for replacing a vector record."""

    vector: list[float] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    document: str | None = None


class RecordResponse(BaseModel):
    """Serialized vector record returned by the API."""

    id: str
    vector: list[float]
    metadata: dict[str, Any]
    document: str | None