"""Embedding provider abstractions for VectorNest."""

from abc import ABC, abstractmethod

import numpy as np

from vectornest.core.exceptions import ValidationError


class EmbeddingProvider(ABC):
    """Define the interface for text embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimension produced by this provider."""
        raise NotImplementedError

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Convert one text string into an embedding vector."""
        raise NotImplementedError

    def embed_batch(
        self,
        texts: list[str],
    ) -> list[np.ndarray]:
        """Embed multiple texts using the provider."""
        return [
            self.embed_text(text)
            for text in texts
        ]

    @staticmethod
    def validate_text(text: str) -> str:
        """Validate and normalize text before embedding."""
        if not isinstance(text, str):
            raise ValidationError(
                "Text to embed must be a string."
            )

        normalized = text.strip()

        if not normalized:
            raise ValidationError(
                "Text to embed cannot be empty."
            )

        return normalized

    @staticmethod
    def normalize_embedding(
        embedding: np.ndarray,
        expected_dimension: int,
    ) -> np.ndarray:
        """Validate and normalize an embedding vector."""
        array = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if array.ndim != 1:
            raise ValidationError(
                "Embedding must be one-dimensional."
            )

        if array.shape[0] != expected_dimension:
            raise ValidationError(
                "Embedding dimension does not match "
                "provider dimension."
            )

        if not np.all(np.isfinite(array)):
            raise ValidationError(
                "Embedding contains non-finite values."
            )

        return np.ascontiguousarray(array)