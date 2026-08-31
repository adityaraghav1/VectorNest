import pytest

from vectornest.benchmarking.service import BenchmarkService
from vectornest.core.exceptions import ValidationError
from vectornest.core.types import DistanceMetric, IndexType
from vectornest.models.record import VectorRecord


def make_records() -> list[VectorRecord]:
    return [
        VectorRecord(
            id="r1",
            vector=[1.0, 1.0],
        ),
        VectorRecord(
            id="r2",
            vector=[2.0, 2.0],
        ),
        VectorRecord(
            id="r3",
            vector=[3.0, 3.0],
        ),
        VectorRecord(
            id="r4",
            vector=[8.0, 8.0],
        ),
        VectorRecord(
            id="r5",
            vector=[10.0, 10.0],
        ),
    ]


def test_brute_force_has_perfect_recall() -> None:
    service = BenchmarkService()

    result = service.benchmark(
        records=make_records(),
        queries=[
            [0.0, 0.0],
            [9.0, 9.0],
        ],
        dimension=2,
        metric=DistanceMetric.EUCLIDEAN,
        index_type=IndexType.BRUTE_FORCE,
        k=2,
    )

    assert result.recall_at_k == pytest.approx(
        1.0
    )

    assert result.record_count == 5
    assert result.query_count == 2
    assert result.k == 2


def test_kd_tree_has_perfect_recall_on_simple_data() -> None:
    service = BenchmarkService()

    result = service.benchmark(
        records=make_records(),
        queries=[
            [0.0, 0.0],
            [9.0, 9.0],
        ],
        dimension=2,
        metric=DistanceMetric.EUCLIDEAN,
        index_type=IndexType.KD_TREE,
        k=2,
    )

    assert result.recall_at_k == pytest.approx(
        1.0
    )


def test_hnsw_returns_valid_recall() -> None:
    service = BenchmarkService()

    result = service.benchmark(
        records=make_records(),
        queries=[
            [0.0, 0.0],
            [9.0, 9.0],
        ],
        dimension=2,
        metric=DistanceMetric.EUCLIDEAN,
        index_type=IndexType.HNSW,
        k=2,
    )

    assert 0.0 <= result.recall_at_k <= 1.0


def test_benchmark_reports_non_negative_timings() -> None:
    service = BenchmarkService()

    result = service.benchmark(
        records=make_records(),
        queries=[
            [0.0, 0.0],
        ],
        dimension=2,
        metric=DistanceMetric.EUCLIDEAN,
        index_type=IndexType.BRUTE_FORCE,
        k=2,
    )

    assert result.build_time_ms >= 0.0
    assert result.average_query_time_ms >= 0.0


def test_benchmark_rejects_empty_queries() -> None:
    service = BenchmarkService()

    with pytest.raises(ValidationError):
        service.benchmark(
            records=make_records(),
            queries=[],
            dimension=2,
            metric=DistanceMetric.EUCLIDEAN,
            index_type=IndexType.BRUTE_FORCE,
            k=2,
        )


def test_benchmark_rejects_invalid_k() -> None:
    service = BenchmarkService()

    with pytest.raises(ValidationError):
        service.benchmark(
            records=make_records(),
            queries=[
                [0.0, 0.0],
            ],
            dimension=2,
            metric=DistanceMetric.EUCLIDEAN,
            index_type=IndexType.BRUTE_FORCE,
            k=0,
        )


def test_benchmark_rejects_invalid_dimension() -> None:
    service = BenchmarkService()

    with pytest.raises(ValidationError):
        service.benchmark(
            records=make_records(),
            queries=[
                [0.0, 0.0],
            ],
            dimension=0,
            metric=DistanceMetric.EUCLIDEAN,
            index_type=IndexType.BRUTE_FORCE,
            k=2,
        )