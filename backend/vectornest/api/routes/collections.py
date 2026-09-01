"""Collection management endpoints."""


from fastapi import APIRouter, Response, status

from vectornest.api.dependencies import StorageDependency
from vectornest.api.schemas.collections import (
    CollectionCreateRequest,
    CollectionResponse,
)
from vectornest.models.collection import CollectionConfig

router = APIRouter(
    prefix="/collections",
    tags=["collections"],
)


def _to_response(collection: CollectionConfig) -> CollectionResponse:
    """Convert a domain collection into an API response."""

    return CollectionResponse(
        name=collection.name,
        dimension=collection.dimension,
        distance_metric=collection.distance_metric,
    )


@router.post(
    "",
    response_model=CollectionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_collection(
    payload: CollectionCreateRequest,
    storage: StorageDependency,
) -> CollectionResponse:
    """Create a new vector collection."""

    collection = CollectionConfig(
        name=payload.name,
        dimension=payload.dimension,
        distance_metric=payload.distance_metric,
    )

    storage.create_collection(collection)

    return _to_response(collection)


@router.get(
    "/{name}",
    response_model=CollectionResponse,
)
def get_collection(
    name: str,
    storage: StorageDependency,
) -> CollectionResponse:
    """Return an existing collection."""

    collection = storage.get_collection(name)

    return _to_response(collection)


@router.delete(
    "/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_collection(
    name: str,
    storage: StorageDependency,
) -> Response:
    """Delete an existing collection."""

    storage.delete_collection(name)

    return Response(status_code=status.HTTP_204_NO_CONTENT)