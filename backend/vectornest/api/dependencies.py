"""Shared API dependencies."""

from typing import Annotated

from fastapi import Depends

from vectornest.embeddings.base import EmbeddingProvider
from vectornest.services.ingestion import DocumentIngestionService
from vectornest.services.search import SearchService
from vectornest.services.semantic_search import SemanticSearchService
from vectornest.storage.engine import InMemoryStorage

_storage = InMemoryStorage()


def get_storage() -> InMemoryStorage:
    """Return the shared VectorNest storage instance."""

    return _storage


StorageDependency = Annotated[
    InMemoryStorage,
    Depends(get_storage),
]


def get_embedding_provider() -> EmbeddingProvider:
    """Return the configured embedding provider."""
    raise RuntimeError(
        "Embedding provider has not been configured."
    )


EmbeddingProviderDependency = Annotated[
    EmbeddingProvider,
    Depends(get_embedding_provider),
]


def get_ingestion_service(
    storage: StorageDependency,
    embedding_provider: EmbeddingProviderDependency,
) -> DocumentIngestionService:
    """Create the document ingestion service."""
    return DocumentIngestionService(
        storage=storage,
        embedding_provider=embedding_provider,
    )


DocumentIngestionServiceDependency = Annotated[
    DocumentIngestionService,
    Depends(get_ingestion_service),
]


def get_semantic_search_service(
    storage: StorageDependency,
    embedding_provider: EmbeddingProviderDependency,
) -> SemanticSearchService:
    """Create the semantic search service."""
    return SemanticSearchService(
        storage=storage,
        embedding_provider=embedding_provider,
    )


SemanticSearchServiceDependency = Annotated[
    SemanticSearchService,
    Depends(get_semantic_search_service),
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