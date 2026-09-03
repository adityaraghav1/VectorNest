"""RAG question-answering endpoints."""

import json
from collections.abc import Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from vectornest.api.dependencies import (
    RAGServiceDependency,
)
from vectornest.api.schemas.rag import (
    RAGRequest,
    RAGResponseSchema,
    RAGSourceResponse,
)
from vectornest.query.filters import MetadataFilter
from vectornest.services.rag import RAGSource

router = APIRouter(
    prefix="/collections/{collection_name}",
    tags=["rag"],
)


@router.post(
    "/rag",
    response_model=RAGResponseSchema,
)
def ask_rag(
    collection_name: str,
    payload: RAGRequest,
    rag_service: RAGServiceDependency,
) -> RAGResponseSchema:
    metadata_filter = (
        MetadataFilter(
            payload.metadata_filter
        )
        if payload.metadata_filter
        is not None
        else None
    )

    result = rag_service.answer(
        collection_name,
        payload.question,
        metric=payload.metric,
        k=payload.k,
        metadata_filter=metadata_filter,
        index_type=payload.index_type,
    )

    return RAGResponseSchema(
        answer=result.answer,
        sources=[
            RAGSourceResponse(
                id=source.id,
                score=source.score,
                document=source.document,
                metadata=source.metadata,
            )
            for source in result.sources
        ],
    )


@router.post("/rag/stream")
def stream_rag(
    collection_name: str,
    payload: RAGRequest,
    rag_service: RAGServiceDependency,
) -> StreamingResponse:
    metadata_filter = (
        MetadataFilter(
            payload.metadata_filter
        )
        if payload.metadata_filter
        is not None
        else None
    )

    sources, answer_stream = (
        rag_service.stream_answer(
            collection_name,
            payload.question,
            metric=payload.metric,
            k=payload.k,
            metadata_filter=metadata_filter,
            index_type=payload.index_type,
        )
    )

    return StreamingResponse(
        _stream_response(
            sources,
            answer_stream,
        ),
        media_type="application/x-ndjson",
    )


def _stream_response(
    sources: list[RAGSource],
    answer_stream: Iterator[str],
) -> Iterator[str]:
    yield (
        json.dumps(
            {
                "type": "sources",
                "sources": [
                    {
                        "id": source.id,
                        "score": source.score,
                        "document": source.document,
                        "metadata": source.metadata,
                    }
                    for source in sources
                ],
            }
        )
        + "\n"
    )

    for chunk in answer_stream:
        yield (
            json.dumps(
                {
                    "type": "token",
                    "content": chunk,
                }
            )
            + "\n"
        )

    yield (
        json.dumps(
            {
                "type": "done",
            }
        )
        + "\n"
    )