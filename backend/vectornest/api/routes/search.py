"""Vector search endpoint."""

from fastapi import APIRouter

from vectornest.api.dependencies import SearchServiceDependency
from vectornest.api.schemas.search import (
    SearchHitResponse,
    SearchRequest,
    SearchResponse,
)
from vectornest.query.filters import MetadataFilter

router = APIRouter(
    prefix="/collections/{collection_name}/search",
    tags=["search"],
)


@router.post(
    "",
    response_model=SearchResponse,
)
def search_collection(
    collection_name: str,
    payload: SearchRequest,
    search_service: SearchServiceDependency,
) -> SearchResponse:
    """Search a collection for nearest vector records."""

    metadata_filter = (
        MetadataFilter(payload.metadata_filter)
        if payload.metadata_filter is not None
        else None
    )

    results = search_service.search(
        collection_name,
        payload.query_vector,
        metric=payload.metric,
        k=payload.k,
        metadata_filter=metadata_filter,
        index_type=payload.index_type,
    )

    return SearchResponse(
        results=[
            SearchHitResponse(
                id=result.record.id,
                score=result.score,
                vector=result.record.vector.tolist(),
                metadata=result.record.metadata,
                document=result.record.document,
            )
            for result in results
        ]
    )