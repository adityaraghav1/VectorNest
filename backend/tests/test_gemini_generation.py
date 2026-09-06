import pytest

from vectornest.core.exceptions import (
    ExternalServiceError,
    ValidationError,
)
from vectornest.generation.gemini import GeminiGenerationProvider


class FakeResponse:
    def __init__(
        self,
        text: str,
    ) -> None:
        self.text = text


class FakeModels:
    def generate_content(
        self,
        **kwargs,
    ) -> FakeResponse:
        return FakeResponse(
            "VectorNest is a vector database."
        )

    def generate_content_stream(
        self,
        **kwargs,
    ):
        return iter(
            [
                FakeResponse("VectorNest "),
                FakeResponse(
                    "is a vector database."
                ),
            ]
        )


class FakeGeminiClient:
    def __init__(self) -> None:
        self.models = FakeModels()


class FailingModels:
    def generate_content(
        self,
        **kwargs,
    ):
        raise ConnectionError(
            "Gemini is unavailable."
        )

    def generate_content_stream(
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
    def generate_content(
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
        match="Gemini generation model name cannot be empty",
    ):
        GeminiGenerationProvider(
            client=FakeGeminiClient(),
            model="   ",
        )


def test_generate_returns_answer() -> None:
    provider = GeminiGenerationProvider(
        client=FakeGeminiClient(),
        model="gemini-3.7-flash",
    )

    answer = provider.generate(
        "What is VectorNest?"
    )

    assert answer == (
        "VectorNest is a vector database."
    )


def test_stream_returns_chunks() -> None:
    provider = GeminiGenerationProvider(
        client=FakeGeminiClient(),
        model="gemini-3.7-flash",
    )

    chunks = list(
        provider.stream(
            "What is VectorNest?"
        )
    )

    assert chunks == [
        "VectorNest ",
        "is a vector database.",
    ]


def test_generate_wraps_client_failure() -> None:
    provider = GeminiGenerationProvider(
        client=FailingGeminiClient(),
        model="gemini-3.7-flash",
    )

    with pytest.raises(
        ExternalServiceError,
        match="Failed to generate answer using Gemini",
    ):
        provider.generate(
            "What is VectorNest?"
        )


def test_stream_wraps_client_failure() -> None:
    provider = GeminiGenerationProvider(
        client=FailingGeminiClient(),
        model="gemini-3.7-flash",
    )

    with pytest.raises(
        ExternalServiceError,
        match="Failed to stream answer using Gemini",
    ):
        provider.stream(
            "What is VectorNest?"
        )


def test_generate_rejects_invalid_response() -> None:
    provider = GeminiGenerationProvider(
        client=InvalidGeminiClient(),
        model="gemini-3.7-flash",
    )

    with pytest.raises(
        ExternalServiceError,
        match=(
            "Gemini response does not contain "
            "a generated answer"
        ),
    ):
        provider.generate(
            "What is VectorNest?"
        )