import numpy as np
import pytest

from vectornest.core.exceptions import (
    ExternalServiceError,
    ValidationError,
)
from vectornest.embeddings.gemini import GeminiEmbeddingProvider


class FakeEmbedding:
    def __init__(
        self,
        values: list[float],
    ) -> None:
        self.values = values


class FakeEmbeddingResponse:
    def __init__(
        self,
        values: list[float],
    ) -> None:
        self.embeddings = [
            FakeEmbedding(values)
        ]


class FakeModels:
    def embed_content(
        self,
        **kwargs,
    ) -> FakeEmbeddingResponse:
        return FakeEmbeddingResponse(
            [1.0, 0.0, 0.0]
        )


class FakeGeminiClient:
    def __init__(self) -> None:
        self.models = FakeModels()


class FailingModels:
    def embed_content(
        self,
        **kwargs,
    ):
        raise ConnectionError(
            "Gemini is unavailable."
        )


class FailingGeminiClient:
    def __init__(self) -> None:
        self.models = FailingModels()


class InvalidModels:
    def embed_content(
        self,
        **kwargs,
    ):
        return object()


class InvalidGeminiClient:
    def __init__(self) -> None:
        self.models = InvalidModels()


def test_rejects_empty_model_name() -> None:
    with pytest.raises(
        ValidationError,
        match="Gemini embedding model name cannot be empty",
    ):
        GeminiEmbeddingProvider(
            client=FakeGeminiClient(),
            model="   ",
            dimension=3,
        )


def test_rejects_invalid_dimension() -> None:
    with pytest.raises(
        ValidationError,
        match="Embedding dimension must be greater than zero",
    ):
        GeminiEmbeddingProvider(
            client=FakeGeminiClient(),
            model="gemini-embedding-2",
            dimension=0,
        )


def test_embed_text_returns_normalized_vector() -> None:
    provider = GeminiEmbeddingProvider(
        client=FakeGeminiClient(),
        model="gemini-embedding-2",
        dimension=3,
    )

    embedding = provider.embed_text(
        "Vector databases are useful."
    )

    assert isinstance(embedding, np.ndarray)
    assert embedding.dtype == np.float32
    assert embedding.shape == (3,)
    assert embedding.flags.c_contiguous

    np.testing.assert_array_equal(
        embedding,
        np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        ),
    )


def test_embed_text_wraps_client_failure() -> None:
    provider = GeminiEmbeddingProvider(
        client=FailingGeminiClient(),
        model="gemini-embedding-2",
        dimension=3,
    )

    with pytest.raises(
        ExternalServiceError,
        match="Failed to generate embedding using Gemini",
    ):
        provider.embed_text(
            "Vector databases are useful."
        )


def test_embed_text_rejects_invalid_response() -> None:
    provider = GeminiEmbeddingProvider(
        client=InvalidGeminiClient(),
        model="gemini-embedding-2",
        dimension=3,
    )

    with pytest.raises(
        ExternalServiceError,
        match="Gemini returned an invalid embedding response",
    ):
        provider.embed_text(
            "Vector databases are useful."
        )