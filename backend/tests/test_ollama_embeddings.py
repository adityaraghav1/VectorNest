import numpy as np
import pytest

from vectornest.core.exceptions import ValidationError
from vectornest.embeddings.ollama import OllamaEmbeddingProvider


class FakeOllamaClient:
    def __init__(
        self,
        response: object,
    ) -> None:
        self.response = response
        self.calls: list[dict[str, str]] = []

    def embed(
        self,
        *,
        model: str,
        input: str,
    ) -> object:
        self.calls.append(
            {
                "model": model,
                "input": input,
            }
        )

        return self.response


def test_ollama_provider_embeds_text() -> None:
    client = FakeOllamaClient(
        {
            "embeddings": [
                [
                    0.1,
                    0.2,
                    0.3,
                ]
            ]
        }
    )

    provider = OllamaEmbeddingProvider(
        client=client,
        model="test-model",
        dimension=3,
    )

    embedding = provider.embed_text("hello world")

    assert embedding.shape == (3,)
    assert embedding.dtype == np.float32
    assert embedding.flags["C_CONTIGUOUS"]

    assert client.calls == [
        {
            "model": "test-model",
            "input": "hello world",
        }
    ]


def test_ollama_provider_normalizes_input_text() -> None:
    client = FakeOllamaClient(
        {
            "embeddings": [
                [
                    0.1,
                    0.2,
                    0.3,
                ]
            ]
        }
    )

    provider = OllamaEmbeddingProvider(
        client=client,
        model="test-model",
        dimension=3,
    )

    provider.embed_text("  hello world  ")

    assert client.calls[0]["input"] == "hello world"


def test_ollama_provider_rejects_empty_model_name() -> None:
    client = FakeOllamaClient({})

    with pytest.raises(
        ValidationError,
        match="model name",
    ):
        OllamaEmbeddingProvider(
            client=client,
            model="   ",
            dimension=3,
        )


def test_ollama_provider_rejects_invalid_dimension() -> None:
    client = FakeOllamaClient({})

    with pytest.raises(
        ValidationError,
        match="dimension",
    ):
        OllamaEmbeddingProvider(
            client=client,
            model="test-model",
            dimension=0,
        )


def test_ollama_provider_rejects_missing_embeddings() -> None:
    client = FakeOllamaClient(
        {
            "model": "test-model",
        }
    )

    provider = OllamaEmbeddingProvider(
        client=client,
        model="test-model",
        dimension=3,
    )

    with pytest.raises(
        ValidationError,
        match="does not contain embeddings",
    ):
        provider.embed_text("hello")


def test_ollama_provider_rejects_empty_embeddings() -> None:
    client = FakeOllamaClient(
        {
            "embeddings": [],
        }
    )

    provider = OllamaEmbeddingProvider(
        client=client,
        model="test-model",
        dimension=3,
    )

    with pytest.raises(
        ValidationError,
        match="does not contain embeddings",
    ):
        provider.embed_text("hello")


def test_ollama_provider_rejects_wrong_embedding_dimension() -> None:
    client = FakeOllamaClient(
        {
            "embeddings": [
                [
                    0.1,
                    0.2,
                ]
            ]
        }
    )

    provider = OllamaEmbeddingProvider(
        client=client,
        model="test-model",
        dimension=3,
    )

    with pytest.raises(
        ValidationError,
        match="dimension",
    ):
        provider.embed_text("hello")