import numpy as np
import pytest

from vectornest.core.exceptions import ValidationError
from vectornest.embeddings.base import EmbeddingProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    """Simple deterministic provider used for unit tests."""

    @property
    def dimension(self) -> int:
        return 3

    def embed_text(self, text: str) -> np.ndarray:
        normalized = self.validate_text(text)

        embedding = np.array(
            [
                float(len(normalized)),
                1.0,
                2.0,
            ]
        )

        return self.normalize_embedding(
            embedding,
            self.dimension,
        )


def test_embedding_provider_embeds_text() -> None:
    provider = FakeEmbeddingProvider()

    embedding = provider.embed_text("hello")

    assert embedding.dtype == np.float32
    assert embedding.shape == (3,)
    assert embedding.flags["C_CONTIGUOUS"]


def test_embedding_provider_embeds_batch() -> None:
    provider = FakeEmbeddingProvider()

    embeddings = provider.embed_batch(
        [
            "hello",
            "world",
        ]
    )

    assert len(embeddings) == 2
    assert all(
        embedding.shape == (3,)
        for embedding in embeddings
    )


def test_embedding_provider_rejects_empty_text() -> None:
    provider = FakeEmbeddingProvider()

    with pytest.raises(
        ValidationError,
        match="cannot be empty",
    ):
        provider.embed_text("   ")


def test_embedding_provider_rejects_non_string_text() -> None:
    with pytest.raises(
        ValidationError,
        match="must be a string",
    ):
        EmbeddingProvider.validate_text(123)  # type: ignore[arg-type]


def test_normalize_embedding_converts_to_float32() -> None:
    embedding = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
    )

    normalized = EmbeddingProvider.normalize_embedding(
        embedding,
        expected_dimension=3,
    )

    assert normalized.dtype == np.float32


def test_normalize_embedding_rejects_wrong_dimension() -> None:
    embedding = np.array(
        [1.0, 2.0]
    )

    with pytest.raises(
        ValidationError,
        match="dimension",
    ):
        EmbeddingProvider.normalize_embedding(
            embedding,
            expected_dimension=3,
        )


def test_normalize_embedding_rejects_non_1d_array() -> None:
    embedding = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    with pytest.raises(
        ValidationError,
        match="one-dimensional",
    ):
        EmbeddingProvider.normalize_embedding(
            embedding,
            expected_dimension=4,
        )


def test_normalize_embedding_rejects_non_finite_values() -> None:
    embedding = np.array(
        [1.0, np.nan, 3.0]
    )

    with pytest.raises(
        ValidationError,
        match="non-finite",
    ):
        EmbeddingProvider.normalize_embedding(
            embedding,
            expected_dimension=3,
        )