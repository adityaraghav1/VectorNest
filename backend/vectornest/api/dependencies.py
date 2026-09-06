"""Shared FastAPI dependencies."""

from typing import Annotated

import ollama
from fastapi import Depends
from google import genai

from vectornest.core.config import Settings, load_settings
from vectornest.embeddings.base import EmbeddingProvider
from vectornest.embeddings.gemini import GeminiEmbeddingProvider
from vectornest.embeddings.ollama import OllamaEmbeddingProvider
from vectornest.generation.base import GenerationProvider
from vectornest.generation.gemini import GeminiGenerationProvider
from vectornest.generation.ollama import OllamaGenerationProvider
from vectornest.services.ingestion import DocumentIngestionService
from vectornest.services.rag import RAGService
from vectornest.services.search import SearchService
from vectornest.services.semantic_search import SemanticSearchService
from vectornest.storage.persistent import PersistentStorage

_settings = load_settings()

_storage = PersistentStorage(
    root_path=_settings.data_dir,
)


def _create_ai_providers() -> tuple[
    EmbeddingProvider,
    GenerationProvider,
]:
    """Create AI providers from the configured provider."""

    if _settings.ai_provider == "gemini":
        gemini_client = genai.Client(
            api_key=_settings.gemini_api_key,
        )

        embedding_provider = GeminiEmbeddingProvider(
            client=gemini_client,
            model=_settings.embedding_model,
            dimension=_settings.embedding_dimension,
        )

        generation_provider = GeminiGenerationProvider(
            client=gemini_client,
            model=_settings.llm_model,
        )

        return (
            embedding_provider,
            generation_provider,
        )

    ollama_client = ollama.Client(
        host=_settings.ollama_host,
    )

    embedding_provider = OllamaEmbeddingProvider(
        client=ollama_client,
        model=_settings.embedding_model,
        dimension=_settings.embedding_dimension,
    )

    generation_provider = OllamaGenerationProvider(
        client=ollama_client,
        model=_settings.llm_model,
    )

    return (
        embedding_provider,
        generation_provider,
    )


(
    _embedding_provider,
    _generation_provider,
) = _create_ai_providers()


def get_settings() -> Settings:
    """Return application settings."""

    return _settings


def get_storage() -> PersistentStorage:
    """Return the shared persistent storage instance."""

    return _storage


def get_embedding_provider() -> EmbeddingProvider:
    """Return the configured embedding provider."""

    return _embedding_provider


def get_generation_provider() -> GenerationProvider:
    """Return the configured generation provider."""

    return _generation_provider


def get_document_ingestion_service(
    storage: Annotated[
        PersistentStorage,
        Depends(get_storage),
    ],
    embedding_provider: Annotated[
        EmbeddingProvider,
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
        EmbeddingProvider,
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
    generation_provider: Annotated[
        GenerationProvider,
        Depends(get_generation_provider),
    ],
) -> RAGService:
    """Create the configured RAG service."""

    return RAGService(
        semantic_search_service=semantic_search_service,
        generation_provider=generation_provider,
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
    EmbeddingProvider,
    Depends(get_embedding_provider),
]

GenerationProviderDependency = Annotated[
    GenerationProvider,
    Depends(get_generation_provider),
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