"""Gemini generation provider for VectorNest."""

from collections.abc import Iterator
from typing import Any

from vectornest.core.exceptions import (
    ExternalServiceError,
    ValidationError,
)
from vectornest.generation.base import GenerationProvider


class GeminiGenerationProvider(GenerationProvider):
    """Generate text using the Gemini API."""

    def __init__(
        self,
        client: Any,
        model: str,
    ) -> None:
        if not model.strip():
            raise ValidationError(
                "Gemini generation model name cannot be empty."
            )

        self._client = client
        self._model = model.strip()

    def generate(
        self,
        prompt: str,
    ) -> str:
        """Generate a complete Gemini response."""

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
            )
        except Exception as exc:
            raise ExternalServiceError(
                "Failed to generate answer using Gemini."
            ) from exc

        text = getattr(
            response,
            "text",
            None,
        )

        if not isinstance(text, str):
            raise ExternalServiceError(
                "Gemini response does not contain "
                "a generated answer."
            )

        answer = text.strip()

        if not answer:
            raise ExternalServiceError(
                "Gemini response does not contain "
                "a generated answer."
            )

        return answer

    def stream(
        self,
        prompt: str,
    ) -> Iterator[str]:
        """Stream generated Gemini response chunks."""

        try:
            response = self._client.models.generate_content_stream(
                model=self._model,
                contents=prompt,
            )
        except Exception as exc:
            raise ExternalServiceError(
                "Failed to stream answer using Gemini."
            ) from exc

        return self._iter_stream_content(response)

    @staticmethod
    def _iter_stream_content(
        response: Any,
    ) -> Iterator[str]:
        for chunk in response:
            text = getattr(
                chunk,
                "text",
                None,
            )

            if isinstance(text, str) and text:
                yield text