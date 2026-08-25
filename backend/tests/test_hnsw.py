import pytest

from vectornest.core.exceptions import (
    DimensionMismatchError,
    RecordNotFoundError,
    ValidationError,
)
from vectornest.core.types import DistanceMetric
from vectornest.indexes.hnsw import HNSWIndex
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


def test_create_hnsw_index() -> None:
    index = HNSWIndex(
        dimension=2,
    )

    assert index.dimension == 2
    assert index.count() == 0


def test_rejects_invalid_dimension() -> None:
    with pytest.raises(ValidationError):
        HNSWIndex(
            dimension=0,
        )


def test_rejects_invalid_max_connections() -> None:
    with pytest.raises(ValidationError):
        HNSWIndex(
            dimension=2,
            max_connections=0,
        )


def test_add_and_get_record() -> None:
    index = HNSWIndex(
        dimension=2,
    )

    record = make_record(
        "r1",
        [1.0, 0.0],
    )

    index.add(record)

    assert index.get("r1") is record
    assert index.count() == 1


def test_add_rejects_dimension_mismatch() -> None:
    index = HNSWIndex(
        dimension=2,
    )

    with pytest.raises(DimensionMismatchError):
        index.add(
            make_record(
                "r1",
                [1.0, 2.0, 3.0],
            )
        )


def test_multiple_records_are_connected() -> None:
    index = HNSWIndex(
        dimension=2,
        max_connections=2,
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
                [0.0, 1.0],
            ),
        ]
    )

    assert index.count() == 3

    assert any(
        node.neighbours
        for node in index._nodes.values()
    )


def test_search_finds_close_cosine_record() -> None:
    index = HNSWIndex(
        dimension=2,
        metric=DistanceMetric.COSINE,
        max_connections=3,
    )

    index.add_many(
        [
            make_record(
                "best",
                [1.0, 0.0],
            ),
            make_record(
                "middle",
                [0.7, 0.7],
            ),
            make_record(
                "worst",
                [0.0, 1.0],
            ),
        ]
    )

    results = index.search(
        [1.0, 0.0],
        limit=1,
    )

    assert results[0].record.id == "best"

def test_rejects_invalid_ef_search() -> None:
    with pytest.raises(ValidationError):
        HNSWIndex(
            dimension=2,
            ef_search=0,
        )


def test_search_limit_can_exceed_default_ef() -> None:
    index = HNSWIndex(
        dimension=2,
        metric=DistanceMetric.EUCLIDEAN,
        max_connections=4,
        ef_search=2,
    )

    index.add_many(
        [
            make_record("r1", [1.0, 1.0]),
            make_record("r2", [2.0, 2.0]),
            make_record("r3", [3.0, 3.0]),
            make_record("r4", [4.0, 4.0]),
        ]
    )

    results = index.search(
        [0.0, 0.0],
        limit=3,
    )

    assert len(results) == 3

def test_random_levels_are_reproducible() -> None:
    first = HNSWIndex(
        dimension=2,
        random_seed=42,
    )

    second = HNSWIndex(
        dimension=2,
        random_seed=42,
    )

    records = [
        make_record("r1", [1.0, 0.0]),
        make_record("r2", [2.0, 0.0]),
        make_record("r3", [3.0, 0.0]),
        make_record("r4", [4.0, 0.0]),
    ]

    first.add_many(records)
    second.add_many(records)

    first_levels = {
        node_id: node.level
        for node_id, node in first._nodes.items()
    }

    second_levels = {
        node_id: node.level
        for node_id, node in second._nodes.items()
    }

    assert first_levels == second_levels


def test_hierarchy_tracks_highest_level() -> None:
    index = HNSWIndex(
        dimension=2,
        level_probability=0.5,
        random_seed=7,
    )

    index.add_many(
        [
            make_record("r1", [1.0, 0.0]),
            make_record("r2", [2.0, 0.0]),
            make_record("r3", [3.0, 0.0]),
            make_record("r4", [4.0, 0.0]),
            make_record("r5", [5.0, 0.0]),
        ]
    )

    highest = max(
        node.level
        for node in index._nodes.values()
    )

    assert index._max_level == highest

    assert (
        index._nodes[index._entry_point].level
        == highest
    )


def test_node_exists_on_every_level_up_to_its_maximum() -> None:
    index = HNSWIndex(
        dimension=2,
        random_seed=11,
    )

    index.add_many(
        [
            make_record("r1", [1.0, 0.0]),
            make_record("r2", [2.0, 0.0]),
            make_record("r3", [3.0, 0.0]),
        ]
    )

    for node in index._nodes.values():
        assert set(node.neighbours) == set(
            range(node.level + 1)
        )


def test_removing_highest_node_recalculates_entry_point() -> None:
    index = HNSWIndex(
        dimension=2,
        random_seed=21,
    )

    index.add_many(
        [
            make_record("r1", [1.0, 0.0]),
            make_record("r2", [2.0, 0.0]),
            make_record("r3", [3.0, 0.0]),
            make_record("r4", [4.0, 0.0]),
        ]
    )

    old_entry = index._entry_point

    assert old_entry is not None

    index.remove(old_entry)

    if index.count() > 0:
        highest = max(
            node.level
            for node in index._nodes.values()
        )

        assert index._max_level == highest
        assert (
            index._nodes[index._entry_point].level
            == highest
        )


def test_clear_resets_hierarchy() -> None:
    index = HNSWIndex(
        dimension=2,
        random_seed=42,
    )

    index.add(
        make_record(
            "r1",
            [1.0, 0.0],
        )
    )

    index.clear()

    assert index._entry_point is None
    assert index._max_level == -1


def test_candidate_search_returns_results_in_metric_order() -> None:
    index = HNSWIndex(
        dimension=2,
        metric=DistanceMetric.EUCLIDEAN,
        max_connections=4,
        ef_search=4,
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
                [4.0, 4.0],
            ),
        ]
    )

    results = index.search(
        [0.0, 0.0],
        limit=3,
    )

    assert [
        result.record.id
        for result in results
    ] == [
        "closest",
        "middle",
        "far",
    ]


def test_candidate_search_orders_cosine_similarity_correctly() -> None:
    index = HNSWIndex(
        dimension=2,
        metric=DistanceMetric.COSINE,
        max_connections=4,
        ef_search=4,
    )

    index.add_many(
        [
            make_record(
                "best",
                [1.0, 0.0],
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
        limit=3,
    )

    assert [
        result.record.id
        for result in results
    ] == [
        "best",
        "middle",
        "worst",
    ]


def test_search_supports_euclidean_distance() -> None:
    index = HNSWIndex(
        dimension=2,
        metric=DistanceMetric.EUCLIDEAN,
        max_connections=3,
    )

    index.add_many(
        [
            make_record(
                "closest",
                [1.0, 1.0],
            ),
            make_record(
                "middle",
                [4.0, 4.0],
            ),
            make_record(
                "far",
                [10.0, 10.0],
            ),
        ]
    )

    results = index.search(
        [0.0, 0.0],
        limit=1,
    )

    assert results[0].record.id == "closest"


def test_search_empty_index_returns_empty_list() -> None:
    index = HNSWIndex(
        dimension=2,
    )

    assert index.search(
        [1.0, 0.0]
    ) == []


def test_search_rejects_invalid_limit() -> None:
    index = HNSWIndex(
        dimension=2,
    )

    with pytest.raises(ValidationError):
        index.search(
            [1.0, 0.0],
            limit=0,
        )


def test_remove_record() -> None:
    index = HNSWIndex(
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

    with pytest.raises(RecordNotFoundError):
        index.get("r1")


def test_remove_updates_entry_point() -> None:
    index = HNSWIndex(
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

    index.remove("r1")

    assert index._entry_point == "r2"


def test_clear_index() -> None:
    index = HNSWIndex(
        dimension=2,
    )

    index.add(
        make_record(
            "r1",
            [1.0, 0.0],
        )
    )

    index.clear()

    assert index.count() == 0
    assert index.search([1.0, 0.0]) == []