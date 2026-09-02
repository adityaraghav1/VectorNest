import numpy as np
import pytest

from vectornest.core.exceptions import (
    DimensionMismatchError,
    ValidationError,
)
from vectornest.embeddings.base import EmbeddingProvider
from vectornest.ingestion.chunking import TextChunker
from vectornest.models.collection import CollectionConfig
from vectornest.services.ingestion import DocumentIngestionService
from vectornest.storage.engine import InMemoryStorage


class FakeEmbeddingProvider(EmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 3

    def embed_text(self, text: str) -> np.ndarray:
        self.validate_text(text)

        return self.normalize_embedding(
            np.array(
                [
                    float(len(text)),
                    1.0,
                    2.0,
                ]
            ),
            self.dimension,
        )


def create_storage() -> InMemoryStorage:
    storage = InMemoryStorage()

    storage.create_collection(
        CollectionConfig(
            name="documents",
            dimension=3,
        )
    )

    return storage


def test_ingestion_stores_document_chunks() -> None:
    storage = create_storage()

    service = DocumentIngestionService(
        storage=storage,
        embedding_provider=FakeEmbeddingProvider(),
        chunker=TextChunker(
            chunk_size=4,
            chunk_overlap=1,
        ),
    )

    records = service.ingest(
        "documents",
        "one two three four five six seven",
        document_id="guide",
    )

    assert len(records) == 2
    assert storage.count("documents") == 2

    assert records[0].id == "guide:chunk:0"
    assert records[1].id == "guide:chunk:1"


def test_ingestion_stores_chunk_text_as_document() -> None:
    storage = create_storage()

    service = DocumentIngestionService(
        storage=storage,
        embedding_provider=FakeEmbeddingProvider(),
        chunker=TextChunker(
            chunk_size=4,
            chunk_overlap=1,
        ),
    )

    records = service.ingest(
        "documents",
        "one two three four five six seven",
        document_id="guide",
    )

    assert records[0].document == (
        "one two three four"
    )

    assert records[1].document == (
        "four five six seven"
    )


def test_ingestion_adds_chunk_metadata() -> None:
    storage = create_storage()

    service = DocumentIngestionService(
        storage=storage,
        embedding_provider=FakeEmbeddingProvider(),
        chunker=TextChunker(
            chunk_size=4,
            chunk_overlap=1,
        ),
    )

    records = service.ingest(
        "documents",
        "one two three four five six",
        document_id="guide",
        metadata={
            "category": "tutorial",
        },
    )

    metadata = records[0].metadata

    assert metadata["category"] == "tutorial"
    assert metadata["document_id"] == "guide"
    assert metadata["chunk_index"] == 0
    assert metadata["start_word"] == 0
    assert metadata["end_word"] == 4


def test_ingestion_preserves_original_metadata() -> None:
    storage = create_storage()

    service = DocumentIngestionService(
        storage=storage,
        embedding_provider=FakeEmbeddingProvider(),
    )

    original_metadata = {
        "category": "python",
    }

    service.ingest(
        "documents",
        "Python programming language",
        document_id="python-guide",
        metadata=original_metadata,
    )

    assert original_metadata == {
        "category": "python",
    }


def test_ingestion_rejects_empty_document() -> None:
    storage = create_storage()

    service = DocumentIngestionService(
        storage=storage,
        embedding_provider=FakeEmbeddingProvider(),
    )

    with pytest.raises(
        ValidationError,
        match="non-empty",
    ):
        service.ingest(
            "documents",
            "   ",
        )


def test_ingestion_rejects_dimension_mismatch() -> None:
    storage = InMemoryStorage()

    storage.create_collection(
        CollectionConfig(
            name="documents",
            dimension=2,
        )
    )

    service = DocumentIngestionService(
        storage=storage,
        embedding_provider=FakeEmbeddingProvider(),
    )

    with pytest.raises(
        DimensionMismatchError,
        match="dimension",
    ):
        service.ingest(
            "documents",
            "Vector databases store embeddings.",
        )


def test_ingestion_generates_document_id_when_missing() -> None:
    storage = create_storage()

    service = DocumentIngestionService(
        storage=storage,
        embedding_provider=FakeEmbeddingProvider(),
    )

    records = service.ingest(
        "documents",
        "Vector databases store embeddings.",
    )

    document_id = records[0].metadata["document_id"]

    assert isinstance(document_id, str)
    assert document_id
    assert records[0].id == (
        f"{document_id}:chunk:0"
    )