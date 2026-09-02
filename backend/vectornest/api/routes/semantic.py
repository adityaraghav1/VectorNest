"""Document ingestion and semantic search endpoints."""

from fastapi import APIRouter, status

from vectornest.api.dependencies import (
    DocumentIngestionServiceDependency,
    SemanticSearchServiceDependency,
)
from vectornest.api.schemas.search import (
    SearchHitResponse,
    SearchResponse,
)
from vectornest.api.schemas.semantic import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    SemanticSearchRequest,
)
from vectornest.query.filters import MetadataFilter

router = APIRouter(
    prefix="/collections/{collection_name}",
    tags=["semantic"],
)


@router.post(
    "/documents",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
)
def ingest_document(
    collection_name: str,
    payload: DocumentIngestRequest,
    ingestion_service: DocumentIngestionServiceDependency,
) -> DocumentIngestResponse:
    """Chunk, embed and store a document."""
    records = ingestion_service.ingest(
        collection_name,
        payload.document,
        document_id=payload.document_id,
        metadata=payload.metadata,
    )

    document_id = str(
        records[0].metadata["document_id"]
    )

    return DocumentIngestResponse(
        document_id=document_id,
        record_ids=[
            record.id
            for record in records
        ],
        chunks_created=len(records),
    )


@router.post(
    "/semantic-search",
    response_model=SearchResponse,
)
def semantic_search(
    collection_name: str,
    payload: SemanticSearchRequest,
    semantic_search_service: SemanticSearchServiceDependency,
) -> SearchResponse:
    """Search a collection using natural-language text."""
    metadata_filter = (
        MetadataFilter(payload.metadata_filter)
        if payload.metadata_filter is not None
        else None
    )

    results = semantic_search_service.search(
        collection_name,
        payload.query,
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