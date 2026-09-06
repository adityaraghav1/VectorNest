"""Gemini embedding provider for VectorNest."""

from typing import Any

import numpy as np
from google.genai import types

from vectornest.core.exceptions import (
    ExternalServiceError,
    ValidationError,
)
from vectornest.embeddings.base import EmbeddingProvider


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Generate text embeddings using the Gemini API."""

    def __init__(
        self,
        client: Any,
        model: str,
        dimension: int,
    ) -> None:
        if not model.strip():
            raise ValidationError(
                "Gemini embedding model name cannot be empty."
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

    def embed_text(
        self,
        text: str,
    ) -> np.ndarray:
        """Generate one embedding using Gemini."""

        normalized_text = self.validate_text(text)

        try:
            response = self._client.models.embed_content(
                model=self._model,
                contents=normalized_text,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.dimension,
                ),
            )
        except Exception as exc:
            raise ExternalServiceError(
                "Failed to generate embedding using Gemini."
            ) from exc

        embeddings = getattr(
            response,
            "embeddings",
            None,
        )

        if not embeddings:
            raise ExternalServiceError(
                "Gemini returned an invalid embedding response."
            )

        values = getattr(
            embeddings[0],
            "values",
            None,
        )

        if values is None:
            raise ExternalServiceError(
                "Gemini returned an invalid embedding response."
            )

        return self.normalize_embedding(
            np.asarray(values),
            self.dimension,
        )