"""Shared API dependencies."""

from typing import Annotated

from fastapi import Depends

from vectornest.services.search import SearchService
from vectornest.storage.engine import InMemoryStorage

_storage = InMemoryStorage()


def get_storage() -> InMemoryStorage:
    """Return the shared VectorNest storage instance."""

    return _storage


StorageDependency = Annotated[
    InMemoryStorage,
    Depends(get_storage),
]


def get_search_service(
    storage: StorageDependency,
) -> SearchService:
    """Create a search service backed by the shared storage."""

    return SearchService(storage)


SearchServiceDependency = Annotated[
    SearchService,
    Depends(get_search_service),
]