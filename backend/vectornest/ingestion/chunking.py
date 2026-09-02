"""Text chunking utilities for document ingestion."""

from dataclasses import dataclass

from vectornest.core.exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class TextChunk:
    """Represent one chunk extracted from a document."""

    index: int
    text: str
    start_word: int
    end_word: int


class TextChunker:
    """Split text into overlapping word-based chunks."""

    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 40,
    ) -> None:
        if chunk_size <= 0:
            raise ValidationError(
                "Chunk size must be greater than zero."
            )

        if chunk_overlap < 0:
            raise ValidationError(
                "Chunk overlap cannot be negative."
            )

        if chunk_overlap >= chunk_size:
            raise ValidationError(
                "Chunk overlap must be smaller than chunk size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[TextChunk]:
        """Split text into ordered overlapping chunks."""
        if not isinstance(text, str):
            raise ValidationError(
                "Text must be a string."
            )

        normalized_text = text.strip()

        if not normalized_text:
            return []

        words = normalized_text.split()

        chunks: list[TextChunk] = []

        step = self.chunk_size - self.chunk_overlap
        start = 0
        chunk_index = 0

        while start < len(words):
            end = min(
                start + self.chunk_size,
                len(words),
            )

            chunk_text = " ".join(words[start:end])

            chunks.append(
                TextChunk(
                    index=chunk_index,
                    text=chunk_text,
                    start_word=start,
                    end_word=end,
                )
            )

            if end == len(words):
                break

            start += step
            chunk_index += 1

        return chunks