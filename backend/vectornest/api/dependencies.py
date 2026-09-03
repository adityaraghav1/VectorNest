"""Shared API dependencies."""

from pathlib import Path
from typing import Annotated

from fastapi import Depends
from ollama import Client

from vectornest.embeddings.base import EmbeddingProvider
from vectornest.embeddings.ollama import OllamaEmbeddingProvider
from vectornest.services.ingestion import DocumentIngestionService
from vectornest.services.rag import RAGService
from vectornest.services.search import SearchService
from vectornest.services.semantic_search import SemanticSearchService
from vectornest.storage.persistent import PersistentStorage

_storage_path = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "vectornest"
)

_storage = PersistentStorage(
    root_path=_storage_path,
)


def get_storage() -> PersistentStorage:
    return _storage


StorageDependency = Annotated[
    PersistentStorage,
    Depends(get_storage),
]


_ollama_client = Client(
    host="http://127.0.0.1:11434"
)

_embedding_provider = OllamaEmbeddingProvider(
    client=_ollama_client,
    model="nomic-embed-text",
    dimension=768,
)


def get_embedding_provider() -> EmbeddingProvider:
    return _embedding_provider


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

def get_rag_service(
    semantic_search_service: SemanticSearchServiceDependency,
) -> RAGService:
    return RAGService(
        semantic_search_service=semantic_search_service,
        llm_client=_ollama_client,
        model="llama3.2",
    )


RAGServiceDependency = Annotated[
    RAGService,
    Depends(get_rag_service),
]