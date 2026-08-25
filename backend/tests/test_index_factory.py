import pytest

from vectornest.core.exceptions import ValidationError
from vectornest.core.types import DistanceMetric, IndexType
from vectornest.indexes.brute_force import BruteForceIndex
from vectornest.indexes.factory import create_index
from vectornest.indexes.kd_tree import KDTreeIndex


def test_factory_creates_brute_force_index() -> None:
    index = create_index(
        dimension=3,
        metric=DistanceMetric.COSINE,
        index_type=IndexType.BRUTE_FORCE,
    )

    assert isinstance(index, BruteForceIndex)
    assert index.dimension == 3
    assert index.metric is DistanceMetric.COSINE


def test_factory_creates_kd_tree_index() -> None:
    index = create_index(
        dimension=3,
        metric=DistanceMetric.EUCLIDEAN,
        index_type=IndexType.KD_TREE,
    )

    assert isinstance(index, KDTreeIndex)
    assert index.dimension == 3


def test_kd_tree_rejects_cosine_metric() -> None:
    with pytest.raises(
        ValidationError,
        match="only Euclidean",
    ):
        create_index(
            dimension=3,
            metric=DistanceMetric.COSINE,
            index_type=IndexType.KD_TREE,
        )


def test_kd_tree_rejects_dot_product_metric() -> None:
    with pytest.raises(ValidationError):
        create_index(
            dimension=3,
            metric=DistanceMetric.DOT_PRODUCT,
            index_type=IndexType.KD_TREE,
        )