"""Vector record management endpoints."""

from fastapi import APIRouter, Response, status

from vectornest.api.dependencies import StorageDependency
from vectornest.api.schemas.records import (
    RecordCreateRequest,
    RecordResponse,
    RecordUpdateRequest,
)
from vectornest.models.record import VectorRecord

router = APIRouter(
    prefix="/collections/{collection_name}/records",
    tags=["records"],
)


def _to_response(record: VectorRecord) -> RecordResponse:
    """Convert a domain vector record into an API response."""

    return RecordResponse(
        id=record.id,
        vector=record.vector.tolist(),
        metadata=record.metadata,
        document=record.document,
    )


@router.post(
    "",
    response_model=RecordResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_record(
    collection_name: str,
    payload: RecordCreateRequest,
    storage: StorageDependency,
) -> RecordResponse:
    """Insert a vector record into a collection."""

    record = VectorRecord(
        id=payload.id,
        vector=payload.vector,
        metadata=payload.metadata,
        document=payload.document,
    )

    storage.insert_record(
        collection_name,
        record,
    )

    return _to_response(record)


@router.get(
    "",
    response_model=list[RecordResponse],
)
def list_records(
    collection_name: str,
    storage: StorageDependency,
) -> list[RecordResponse]:
    """Return all records from a collection."""

    records = storage.list_records(
        collection_name
    )

    return [
        _to_response(record)
        for record in records
    ]


@router.get(
    "/{record_id}",
    response_model=RecordResponse,
)
def get_record(
    collection_name: str,
    record_id: str,
    storage: StorageDependency,
) -> RecordResponse:
    """Return one vector record."""

    record = storage.get_record(
        collection_name,
        record_id,
    )

    return _to_response(record)


@router.put(
    "/{record_id}",
    response_model=RecordResponse,
)
def update_record(
    collection_name: str,
    record_id: str,
    payload: RecordUpdateRequest,
    storage: StorageDependency,
) -> RecordResponse:
    """Replace an existing vector record."""

    record = VectorRecord(
        id=record_id,
        vector=payload.vector,
        metadata=payload.metadata,
        document=payload.document,
    )

    storage.update_record(
        collection_name,
        record,
    )

    return _to_response(record)


@router.delete(
    "/{record_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_record(
    collection_name: str,
    record_id: str,
    storage: StorageDependency,
) -> Response:
    """Delete one vector record."""

    storage.delete_record(
        collection_name,
        record_id,
    )

    return Response(
        status_code=status.HTTP_204_NO_CONTENT
    )