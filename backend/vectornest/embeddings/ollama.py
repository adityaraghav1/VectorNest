"""Ollama embedding provider for VectorNest."""

from typing import Any

import numpy as np

from vectornest.core.exceptions import ValidationError
from vectornest.embeddings.base import EmbeddingProvider


class OllamaEmbeddingProvider(EmbeddingProvider):
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
        return self._dimension

    def embed_text(self, text: str) -> np.ndarray:
        normalized_text = self.validate_text(text)

        response = self._client.embed(
            model=self._model,
            input=normalized_text,
        )

        embeddings = response.get("embeddings")

        if not embeddings:
            raise ValidationError(
                "Ollama response does not contain embeddings."
            )

        embedding = embeddings[0]

        return self.normalize_embedding(
            np.asarray(embedding),
            self.dimension,
        )