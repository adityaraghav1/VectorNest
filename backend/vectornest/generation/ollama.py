"""Ollama generation provider for VectorNest."""

from collections.abc import Iterator
from typing import Any

from vectornest.core.exceptions import (
    ExternalServiceError,
    ValidationError,
)
from vectornest.generation.base import GenerationProvider


class OllamaGenerationProvider(GenerationProvider):
    """Generate text using an Ollama model."""

    def __init__(
        self,
        client: Any,
        model: str,
    ) -> None:
        if not model.strip():
            raise ValidationError(
                "Ollama generation model name cannot be empty."
            )

        self._client = client
        self._model = model.strip()

    def generate(self, prompt: str) -> str:
        """Generate a complete response using Ollama."""

        try:
            response = self._client.chat(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
        except Exception as exc:
            raise ExternalServiceError(
                "Failed to generate answer using Ollama."
            ) from exc

        return self._extract_answer(response)

    def stream(self, prompt: str) -> Iterator[str]:
        """Stream generated response chunks using Ollama."""

        try:
            response = self._client.chat(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                stream=True,
            )
        except Exception as exc:
            raise ExternalServiceError(
                "Failed to stream answer using Ollama."
            ) from exc

        return self._iter_stream_content(response)

    @staticmethod
    def _iter_stream_content(
        response: Any,
    ) -> Iterator[str]:
        for chunk in response:
            content = OllamaGenerationProvider._extract_stream_chunk(
                chunk
            )

            if content:
                yield content

    @staticmethod
    def _extract_stream_chunk(
        chunk: Any,
    ) -> str:
        if isinstance(chunk, dict):
            message = chunk.get("message")

            if isinstance(message, dict):
                content = message.get("content")

                if isinstance(content, str):
                    return content

        message = getattr(
            chunk,
            "message",
            None,
        )

        content = getattr(
            message,
            "content",
            None,
        )

        if isinstance(content, str):
            return content

        return ""

    @staticmethod
    def _extract_answer(
        response: Any,
    ) -> str:
        if isinstance(response, dict):
            message = response.get("message")

            if isinstance(message, dict):
                content = message.get("content")

                if isinstance(content, str):
                    answer = content.strip()

                    if answer:
                        return answer

        message = getattr(
            response,
            "message",
            None,
        )

        content = getattr(
            message,
            "content",
            None,
        )

        if isinstance(content, str):
            answer = content.strip()

            if answer:
                return answer

        raise ExternalServiceError(
            "Ollama response does not contain a generated answer."
        )