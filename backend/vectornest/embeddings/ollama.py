"""Ollama embedding provider for VectorNest."""

from typing import Any

import numpy as np

from vectornest.core.exceptions import ValidationError
from vectornest.embeddings.base import EmbeddingProvider


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Generate embeddings using a local Ollama server."""

    def __init__(
        self,
        client: Any,
        model: str,
        dimension: int,
    ) -> None:
        if not model.strip():
            raise ValidationError(
                "Ollama model name cannot be empty."
            )

        if dimension <= 0:
            raise ValidationError(
                "Embedding dimension must be greater than zero."
            )

        self._client = client
        self._model = model.strip()
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    def embed_text(self, text: str) -> np.ndarray:
        """Generate one embedding using Ollama."""
        normalized_text = self.validate_text(text)

        response = self._client.embeddings(
            model=self._model,
            prompt=normalized_text,
        )

        if not isinstance(response, dict):
            raise ValidationError(
                "Ollama returned an invalid embedding response."
            )

        embedding = response.get("embedding")

        if embedding is None:
            raise ValidationError(
                "Ollama response does not contain an embedding."
            )

        return self.normalize_embedding(
            np.asarray(embedding),
            self.dimension,
        )