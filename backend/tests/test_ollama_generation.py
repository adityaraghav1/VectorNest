from collections.abc import Iterator

import pytest

from vectornest.core.exceptions import (
    ExternalServiceError,
    ValidationError,
)
from vectornest.generation.ollama import OllamaGenerationProvider


class FakeOllamaClient:
    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
    ):
        if stream:
            return iter(
                [
                    {
                        "message": {
                            "content": "Hello "
                        }
                    },
                    {
                        "message": {
                            "content": "world."
                        }
                    },
                ]
            )

        return {
            "message": {
                "content": "Hello world."
            }
        }


class FailingOllamaClient:
    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
    ):
        raise ConnectionError(
            "Ollama is unavailable."
        )


class InvalidResponseClient:
    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
    ):
        return {}


def test_rejects_empty_model_name() -> None:
    with pytest.raises(
        ValidationError,
        match="Ollama generation model name cannot be empty",
    ):
        OllamaGenerationProvider(
            client=FakeOllamaClient(),
            model="   ",
        )


def test_generate_returns_complete_answer() -> None:
    provider = OllamaGenerationProvider(
        client=FakeOllamaClient(),
        model="llama3.2",
    )

    result = provider.generate(
        "Say hello."
    )

    assert result == "Hello world."


def test_stream_returns_chunks() -> None:
    provider = OllamaGenerationProvider(
        client=FakeOllamaClient(),
        model="llama3.2",
    )

    stream: Iterator[str] = provider.stream(
        "Say hello."
    )

    assert list(stream) == [
        "Hello ",
        "world.",
    ]


def test_generate_wraps_client_failure() -> None:
    provider = OllamaGenerationProvider(
        client=FailingOllamaClient(),
        model="llama3.2",
    )

    with pytest.raises(
        ExternalServiceError,
        match="Failed to generate answer using Ollama",
    ):
        provider.generate(
            "Say hello."
        )


def test_stream_wraps_client_failure() -> None:
    provider = OllamaGenerationProvider(
        client=FailingOllamaClient(),
        model="llama3.2",
    )

    with pytest.raises(
        ExternalServiceError,
        match="Failed to stream answer using Ollama",
    ):
        provider.stream(
            "Say hello."
        )


def test_generate_rejects_invalid_response() -> None:
    provider = OllamaGenerationProvider(
        client=InvalidResponseClient(),
        model="llama3.2",
    )

    with pytest.raises(
        ExternalServiceError,
        match=(
            "Ollama response does not contain "
            "a generated answer"
        ),
    ):
        provider.generate(
            "Say hello."
        )