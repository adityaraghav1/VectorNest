"""Vector projection API schemas."""

from pydantic import BaseModel


class ProjectionPointResponse(BaseModel):
    id: str
    x: float
    y: float
    document: str | None = None
    is_query: bool = False


class ProjectionResponse(BaseModel):
    points: list[ProjectionPointResponse]