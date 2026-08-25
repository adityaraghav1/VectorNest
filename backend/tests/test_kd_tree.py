import pytest

from vectornest.core.exceptions import (
    DimensionMismatchError,
    RecordNotFoundError,
    ValidationError,
)
from vectornest.indexes.kd_tree import KDTreeIndex
from vectornest.models.record import VectorRecord


def make_record(
    record_id: str,
    vector: list[float],
) -> VectorRecord:
    return VectorRecord(
        id=record_id,
        vector=vector,
        metadata={},
    )


def test_create_index() -> None:
    index = KDTreeIndex(dimension=2)

    assert index.dimension == 2
    assert index.count() == 0


def test_rejects_invalid_dimension() -> None:
    with pytest.raises(ValidationError):
        KDTreeIndex(dimension=0)


def test_add_and_get_record() -> None:
    index = KDTreeIndex(dimension=2)

    record = make_record(
        "r1",
        [1.0, 2.0],
    )

    index.add(record)

    assert index.get("r1") is record
    assert index.count() == 1


def test_add_rejects_dimension_mismatch() -> None:
    index = KDTreeIndex(dimension=2)

    with pytest.raises(DimensionMismatchError):
        index.add(
            make_record(
                "r1",
                [1.0, 2.0, 3.0],
            )
        )


def test_search_returns_nearest_record_first() -> None:
    index = KDTreeIndex(dimension=2)

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
                [4.0, 4.0],
            ),
        ]
    )

    results = index.search(
        [0.0, 0.0],
    )

    assert [
        result.record.id
        for result in results
    ] == [
        "closest",
        "middle",
        "far",
    ]


def test_search_respects_limit() -> None:
    index = KDTreeIndex(dimension=2)

    index.add_many(
        [
            make_record("r1", [1.0, 1.0]),
            make_record("r2", [2.0, 2.0]),
            make_record("r3", [3.0, 3.0]),
        ]
    )

    results = index.search(
        [0.0, 0.0],
        limit=2,
    )

    assert len(results) == 2

    assert [
        result.record.id
        for result in results
    ] == ["r1", "r2"]


def test_search_empty_index_returns_empty_list() -> None:
    index = KDTreeIndex(dimension=2)

    assert index.search(
        [1.0, 1.0]
    ) == []


def test_search_rejects_wrong_query_dimension() -> None:
    index = KDTreeIndex(dimension=2)

    with pytest.raises(DimensionMismatchError):
        index.search(
            [1.0, 2.0, 3.0]
        )


def test_search_rejects_invalid_limit() -> None:
    index = KDTreeIndex(dimension=2)

    with pytest.raises(ValidationError):
        index.search(
            [1.0, 2.0],
            limit=0,
        )


def test_remove_record() -> None:
    index = KDTreeIndex(dimension=2)

    index.add(
        make_record(
            "r1",
            [1.0, 2.0],
        )
    )

    index.remove("r1")

    assert index.count() == 0

    with pytest.raises(RecordNotFoundError):
        index.get("r1")


def test_remove_rebuilds_tree_before_next_search() -> None:
    index = KDTreeIndex(dimension=2)

    index.add_many(
        [
            make_record(
                "closest",
                [1.0, 1.0],
            ),
            make_record(
                "remaining",
                [5.0, 5.0],
            ),
        ]
    )

    first_results = index.search(
        [0.0, 0.0],
        limit=1,
    )

    assert first_results[0].record.id == "closest"

    index.remove("closest")

    second_results = index.search(
        [0.0, 0.0],
        limit=1,
    )

    assert second_results[0].record.id == "remaining"


def test_clear_index() -> None:
    index = KDTreeIndex(dimension=2)

    index.add(
        make_record(
            "r1",
            [1.0, 1.0],
        )
    )

    index.clear()

    assert index.count() == 0
    assert index.search([0.0, 0.0]) == []