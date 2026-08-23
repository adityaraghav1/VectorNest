import pytest

from vectornest.core.exceptions import (
    DimensionMismatchError,
    RecordNotFoundError,
    ValidationError,
)
from vectornest.core.types import DistanceMetric
from vectornest.indexes.brute_force import BruteForceIndex
from vectornest.models.record import VectorRecord


def make_record(
    record_id: str,
    vector: list[float],
) -> VectorRecord:
    return VectorRecord(
        id=record_id,
        vector=vector,
    )


def test_add_and_count() -> None:
    index = BruteForceIndex(
        dimension=3,
    )

    index.add(
        make_record(
            "r1",
            [1.0, 0.0, 0.0],
        )
    )

    assert index.count() == 1


def test_add_rejects_dimension_mismatch() -> None:
    index = BruteForceIndex(
        dimension=3,
    )

    with pytest.raises(DimensionMismatchError):
        index.add(
            make_record(
                "r1",
                [1.0, 2.0],
            )
        )


def test_get_record() -> None:
    index = BruteForceIndex(
        dimension=2,
    )

    record = make_record(
        "r1",
        [1.0, 0.0],
    )

    index.add(record)

    assert index.get("r1") is record


def test_get_missing_record() -> None:
    index = BruteForceIndex(
        dimension=2,
    )

    with pytest.raises(RecordNotFoundError):
        index.get("missing")


def test_remove_record() -> None:
    index = BruteForceIndex(
        dimension=2,
    )

    index.add(
        make_record(
            "r1",
            [1.0, 0.0],
        )
    )

    index.remove("r1")

    assert index.count() == 0


def test_remove_missing_record() -> None:
    index = BruteForceIndex(
        dimension=2,
    )

    with pytest.raises(RecordNotFoundError):
        index.remove("missing")


def test_search_returns_best_cosine_match() -> None:
    index = BruteForceIndex(
        dimension=2,
        metric=DistanceMetric.COSINE,
    )

    index.add_many(
        [
            make_record(
                "best",
                [10.0, 0.0],
            ),
            make_record(
                "middle",
                [1.0, 1.0],
            ),
            make_record(
                "worst",
                [0.0, 1.0],
            ),
        ]
    )

    results = index.search(
        [1.0, 0.0],
    )

    assert [result.record.id for result in results] == [
        "best",
        "middle",
        "worst",
    ]


def test_search_returns_smallest_euclidean_distance_first() -> None:
    index = BruteForceIndex(
        dimension=2,
        metric=DistanceMetric.EUCLIDEAN,
    )

    index.add_many(
        [
            make_record(
                "far",
                [10.0, 10.0],
            ),
            make_record(
                "closest",
                [1.0, 1.0],
            ),
            make_record(
                "middle",
                [3.0, 3.0],
            ),
        ]
    )

    results = index.search(
        [0.0, 0.0],
    )

    assert [result.record.id for result in results] == [
        "closest",
        "middle",
        "far",
    ]


def test_search_respects_limit() -> None:
    index = BruteForceIndex(
        dimension=2,
    )

    index.add_many(
        [
            make_record(
                "r1",
                [1.0, 0.0],
            ),
            make_record(
                "r2",
                [0.9, 0.1],
            ),
            make_record(
                "r3",
                [0.8, 0.2],
            ),
        ]
    )

    results = index.search(
        [1.0, 0.0],
        limit=2,
    )

    assert len(results) == 2


def test_search_rejects_invalid_limit() -> None:
    index = BruteForceIndex(
        dimension=2,
    )

    with pytest.raises(ValidationError):
        index.search(
            [1.0, 0.0],
            limit=0,
        )


def test_search_rejects_dimension_mismatch() -> None:
    index = BruteForceIndex(
        dimension=3,
    )

    with pytest.raises(DimensionMismatchError):
        index.search(
            [1.0, 2.0],
        )


def test_clear_removes_all_records() -> None:
    index = BruteForceIndex(
        dimension=2,
    )

    index.add_many(
        [
            make_record(
                "r1",
                [1.0, 0.0],
            ),
            make_record(
                "r2",
                [0.0, 1.0],
            ),
        ]
    )

    index.clear()

    assert index.count() == 0