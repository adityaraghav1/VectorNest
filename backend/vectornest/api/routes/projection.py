"""Vector projection endpoint."""

from typing import Annotated

from fastapi import APIRouter, Query

from vectornest.api.dependencies import (
    EmbeddingProviderDependency,
    StorageDependency,
)
from vectornest.api.schemas.projection import (
    ProjectionPointResponse,
    ProjectionResponse,
)
from vectornest.services.projection import ProjectionService

router = APIRouter(
    prefix="/collections/{collection_name}",
    tags=["projection"],
)


@router.get(
    "/projection",
    response_model=ProjectionResponse,
)
def get_projection(
    collection_name: str,
    storage: StorageDependency,
    embedding_provider: EmbeddingProviderDependency,
    query: Annotated[
        str | None,
        Query(),
    ] = None,
) -> ProjectionResponse:
    records = storage.list_records(collection_name)

    query_vector = None

    if query is not None and query.strip():
        query_vector = embedding_provider.embed_text(
            query
        )

    service = ProjectionService()

    record_points, query_point = service.project_records(
        records,
        query_vector,
    )

    record_map = {
        record.id: record
        for record in records
    }

    points = [
        ProjectionPointResponse(
            id=record_id,
            x=x,
            y=y,
            document=record_map[record_id].document,
            is_query=False,
        )
        for record_id, x, y in record_points
    ]

    if query_point is not None:
        points.append(
            ProjectionPointResponse(
                id="semantic-query",
                x=query_point[0],
                y=query_point[1],
                document=query,
                is_query=True,
            )
        )

    return ProjectionResponse(points=points)