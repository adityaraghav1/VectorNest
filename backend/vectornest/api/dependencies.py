"""Shared FastAPI dependencies."""

from typing import Annotated

import ollama
from fastapi import Depends

from vectornest.core.config import Settings, load_settings
from vectornest.embeddings.ollama import OllamaEmbeddingProvider
from vectornest.services.ingestion import DocumentIngestionService
from vectornest.services.rag import RAGService
from vectornest.services.search import SearchService
from vectornest.services.semantic_search import SemanticSearchService
from vectornest.storage.persistent import PersistentStorage

_settings = load_settings()

_storage = PersistentStorage(
    root_path=_settings.data_dir,
)

_ollama_client = ollama.Client(
    host=_settings.ollama_host,
)

_embedding_provider = OllamaEmbeddingProvider(
    client=_ollama_client,
    model=_settings.embedding_model,
    dimension=_settings.embedding_dimension,
)


def get_settings() -> Settings:
    """Return application settings."""

    return _settings


def get_storage() -> PersistentStorage:
    """Return the shared persistent storage instance."""

    return _storage


def get_embedding_provider() -> OllamaEmbeddingProvider:
    """Return the configured embedding provider."""

    return _embedding_provider

def get_document_ingestion_service(
    storage: Annotated[
        PersistentStorage,
        Depends(get_storage),
    ],
    embedding_provider: Annotated[
        OllamaEmbeddingProvider,
        Depends(get_embedding_provider),
    ],
) -> DocumentIngestionService:
    """Create a document ingestion service."""

    return DocumentIngestionService(
        storage=storage,
        embedding_provider=embedding_provider,
    )


def get_search_service(
    storage: Annotated[
        PersistentStorage,
        Depends(get_storage),
    ],
) -> SearchService:
    """Create a search service."""

    return SearchService(storage)


def get_semantic_search_service(
    storage: Annotated[
        PersistentStorage,
        Depends(get_storage),
    ],
    embedding_provider: Annotated[
        OllamaEmbeddingProvider,
        Depends(get_embedding_provider),
    ],
) -> SemanticSearchService:
    """Create a semantic search service."""

    return SemanticSearchService(
        storage,
        embedding_provider,
    )


def get_rag_service(
    semantic_search_service: Annotated[
        SemanticSearchService,
        Depends(get_semantic_search_service),
    ],
) -> RAGService:
    """Create the configured RAG service."""

    return RAGService(
        semantic_search_service=semantic_search_service,
        llm_client=_ollama_client,
        model=_settings.llm_model,
    )


SettingsDependency = Annotated[
    Settings,
    Depends(get_settings),
]

StorageDependency = Annotated[
    PersistentStorage,
    Depends(get_storage),
]

EmbeddingProviderDependency = Annotated[
    OllamaEmbeddingProvider,
    Depends(get_embedding_provider),
]

SearchServiceDependency = Annotated[
    SearchService,
    Depends(get_search_service),
]

SemanticSearchServiceDependency = Annotated[
    SemanticSearchService,
    Depends(get_semantic_search_service),
]

RAGServiceDependency = Annotated[
    RAGService,
    Depends(get_rag_service),
]

DocumentIngestionServiceDependency = Annotated[
    DocumentIngestionService,
    Depends(get_document_ingestion_service),
]