import math

import pytest

from vectornest.core.exceptions import DimensionMismatchError, ZeroVectorError
from vectornest.core.types import DistanceMetric
from vectornest.metrics.functions import (
    calculate_metric,
    cosine_similarity,
    dot_product,
    euclidean_distance,
    is_similarity_metric,
    manhattan_distance,
)


def test_cosine_similarity_for_orthogonal_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_for_same_direction_ignores_magnitude() -> None:
    assert cosine_similarity([1.0, 2.0], [10.0, 20.0]) == pytest.approx(1.0)


def test_cosine_similarity_rejects_zero_vector() -> None:
    with pytest.raises(ZeroVectorError):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])


def test_euclidean_distance() -> None:
    assert euclidean_distance([0.0, 0.0], [3.0, 4.0]) == pytest.approx(5.0)


def test_dot_product() -> None:
    assert dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(32.0)


def test_manhattan_distance() -> None:
    assert manhattan_distance([1.0, -2.0], [4.0, 2.0]) == pytest.approx(7.0)


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (DistanceMetric.COSINE, 0.0),
        (DistanceMetric.EUCLIDEAN, math.sqrt(2)),
        (DistanceMetric.DOT_PRODUCT, 0.0),
        (DistanceMetric.MANHATTAN, 2.0),
    ],
)
def test_metric_dispatch(metric: DistanceMetric, expected: float) -> None:
    assert calculate_metric([1.0, 0.0], [0.0, 1.0], metric) == pytest.approx(expected)


def test_metric_direction() -> None:
    assert is_similarity_metric(DistanceMetric.COSINE)
    assert is_similarity_metric(DistanceMetric.DOT_PRODUCT)
    assert not is_similarity_metric(DistanceMetric.EUCLIDEAN)
    assert not is_similarity_metric(DistanceMetric.MANHATTAN)


def test_metrics_reject_dimension_mismatch() -> None:
    with pytest.raises(DimensionMismatchError):
        euclidean_distance([1.0, 2.0], [1.0])
