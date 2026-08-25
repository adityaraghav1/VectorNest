"""Factory for constructing VectorNest indexes."""

from vectornest.core.exceptions import ValidationError
from vectornest.core.types import DistanceMetric, IndexType
from vectornest.indexes.base import VectorIndex
from vectornest.indexes.brute_force import BruteForceIndex
from vectornest.indexes.hnsw import HNSWIndex
from vectornest.indexes.kd_tree import KDTreeIndex


def create_index(
    dimension: int,
    metric: DistanceMetric,
    index_type: IndexType,
) -> VectorIndex:
    """Create an index compatible with the requested configuration."""

    if index_type is IndexType.BRUTE_FORCE:
        return BruteForceIndex(
            dimension=dimension,
            metric=metric,
        )

    if index_type is IndexType.KD_TREE:
        if metric is not DistanceMetric.EUCLIDEAN:
            raise ValidationError(
                "KD-tree currently supports only Euclidean distance."
            )

        return KDTreeIndex(
            dimension=dimension,
        )

    if index_type is IndexType.HNSW:
        return HNSWIndex(
            dimension=dimension,
            metric=metric,
        )

    raise ValidationError(
        f"Unsupported index type: {index_type!r}."
    )