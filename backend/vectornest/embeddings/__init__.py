"""Embedding support for VectorNest."""

from vectornest.embeddings.base import EmbeddingProvider
from vectornest.embeddings.ollama import OllamaEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "OllamaEmbeddingProvider",
]