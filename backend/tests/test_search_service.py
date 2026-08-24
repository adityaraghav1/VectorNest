import pytest

from vectornest.core.exceptions import (
    CollectionNotFoundError,
    ValidationError,
)
from vectornest.core.types import DistanceMetric
from vectornest.models.collection import CollectionConfig
from vectornest.models.record import VectorRecord
from vectornest.query.filters import MetadataFilter
from vectornest.services.search import SearchService
from vectornest.storage.engine import InMemoryStorage


def make_storage() -> InMemoryStorage:
    storage = InMemoryStorage()

    storage.create_collection(
        CollectionConfig(
            name="documents",
            dimension=2,
        )
    )

    return storage


def make_record(
    record_id: str,
    vector: list[float],
) -> VectorRecord:
    return VectorRecord(
        id=record_id,
        vector=vector,
        metadata={"source": "test"},
    )


def test_search_returns_ranked_results() -> None:
    storage = make_storage()

    storage.insert_record(
        "documents",
        make_record("best", [10.0, 0.0]),
    )
    storage.insert_record(
        "documents",
        make_record("middle", [1.0, 1.0]),
    )
    storage.insert_record(
        "documents",
        make_record("worst", [0.0, 1.0]),
    )

    service = SearchService(storage)

    results = service.search(
        "documents",
        [1.0, 0.0],
        metric=DistanceMetric.COSINE,
    )

    assert [result.record.id for result in results] == [
        "best",
        "middle",
        "worst",
    ]

def test_search_applies_metadata_filter() -> None:
    storage = make_storage()

    storage.insert_record(
        "documents",
        VectorRecord(
            id="resume",
            vector=[1.0, 0.0],
            metadata={
                "source": "resume.pdf",
            },
        ),
    )

    storage.insert_record(
        "documents",
        VectorRecord(
            id="notes",
            vector=[1.0, 0.0],
            metadata={
                "source": "notes.pdf",
            },
        ),
    )

    service = SearchService(storage)

    results = service.search(
        "documents",
        [1.0, 0.0],
        metric=DistanceMetric.COSINE,
        metadata_filter=MetadataFilter(
            {
                "source": "resume.pdf",
            }
        ),
    )

    assert [
        result.record.id
        for result in results
    ] == ["resume"]


def test_search_returns_empty_when_filter_matches_nothing() -> None:
    storage = make_storage()

    storage.insert_record(
        "documents",
        VectorRecord(
            id="resume",
            vector=[1.0, 0.0],
            metadata={
                "source": "resume.pdf",
            },
        ),
    )

    service = SearchService(storage)

    results = service.search(
        "documents",
        [1.0, 0.0],
        metric=DistanceMetric.COSINE,
        metadata_filter=MetadataFilter(
            {
                "source": "missing.pdf",
            }
        ),
    )

    assert results == []



def test_search_respects_k() -> None:
    storage = make_storage()

    storage.insert_record(
        "documents",
        make_record("first", [1.0, 0.0]),
    )
    storage.insert_record(
        "documents",
        make_record("second", [0.9, 0.1]),
    )
    storage.insert_record(
        "documents",
        make_record("third", [0.0, 1.0]),
    )

    service = SearchService(storage)

    results = service.search(
        "documents",
        [1.0, 0.0],
        metric=DistanceMetric.COSINE,
        k=2,
    )

    assert len(results) == 2


def test_search_rejects_invalid_k() -> None:
    storage = make_storage()
    service = SearchService(storage)

    with pytest.raises(ValidationError):
        service.search(
            "documents",
            [1.0, 0.0],
            metric=DistanceMetric.COSINE,
            k=0,
        )


def test_search_rejects_missing_collection() -> None:
    storage = InMemoryStorage()
    service = SearchService(storage)

    with pytest.raises(CollectionNotFoundError):
        service.search(
            "missing",
            [1.0, 0.0],
            metric=DistanceMetric.COSINE,
        )