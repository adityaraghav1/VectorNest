import pytest

from vectornest.core.exceptions import ValidationError
from vectornest.ingestion.chunking import TextChunker


def test_chunker_returns_single_chunk_for_short_text() -> None:
    chunker = TextChunker(
        chunk_size=10,
        chunk_overlap=2,
    )

    chunks = chunker.chunk(
        "Vector databases store embeddings."
    )

    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].text == (
        "Vector databases store embeddings."
    )


def test_chunker_splits_text_into_multiple_chunks() -> None:
    chunker = TextChunker(
        chunk_size=4,
        chunk_overlap=1,
    )

    chunks = chunker.chunk(
        "one two three four five six seven"
    )

    assert len(chunks) == 2

    assert chunks[0].text == (
        "one two three four"
    )

    assert chunks[1].text == (
        "four five six seven"
    )


def test_chunker_preserves_overlap_between_chunks() -> None:
    chunker = TextChunker(
        chunk_size=5,
        chunk_overlap=2,
    )

    chunks = chunker.chunk(
        "one two three four five six seven eight"
    )

    assert chunks[0].text == (
        "one two three four five"
    )

    assert chunks[1].text == (
        "four five six seven eight"
    )


def test_chunker_tracks_word_positions() -> None:
    chunker = TextChunker(
        chunk_size=4,
        chunk_overlap=1,
    )

    chunks = chunker.chunk(
        "one two three four five six"
    )

    assert chunks[0].start_word == 0
    assert chunks[0].end_word == 4

    assert chunks[1].start_word == 3
    assert chunks[1].end_word == 6


def test_chunker_returns_empty_list_for_empty_text() -> None:
    chunker = TextChunker()

    assert chunker.chunk("") == []
    assert chunker.chunk("   ") == []


def test_chunker_rejects_invalid_chunk_size() -> None:
    with pytest.raises(
        ValidationError,
        match="Chunk size",
    ):
        TextChunker(chunk_size=0)


def test_chunker_rejects_negative_overlap() -> None:
    with pytest.raises(
        ValidationError,
        match="overlap",
    ):
        TextChunker(
            chunk_size=10,
            chunk_overlap=-1,
        )


def test_chunker_rejects_overlap_equal_to_chunk_size() -> None:
    with pytest.raises(
        ValidationError,
        match="smaller than chunk size",
    ):
        TextChunker(
            chunk_size=10,
            chunk_overlap=10,
        )


def test_chunker_rejects_overlap_larger_than_chunk_size() -> None:
    with pytest.raises(
        ValidationError,
        match="smaller than chunk size",
    ):
        TextChunker(
            chunk_size=10,
            chunk_overlap=11,
        )