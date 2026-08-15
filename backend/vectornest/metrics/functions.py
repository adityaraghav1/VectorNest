"""Numerically safe vector similarity and distance implementations."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from vectornest.core.exceptions import DimensionMismatchError, ValidationError, ZeroVectorError
from vectornest.core.types import DistanceMetric

VectorInput = NDArray[np.float32] | Sequence[float]


def cosine_similarity(left: VectorInput, right: VectorInput) -> float:
    """Return cosine similarity in the interval [-1, 1].

    Cosine measures the angle between two vectors rather than their magnitude.
    It is commonly used for text embeddings, where direction represents meaning.
    """
    left_vector, right_vector = _validate_pair(left, right)
    left_norm = float(np.linalg.norm(left_vector))
    right_norm = float(np.linalg.norm(right_vector))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ZeroVectorError("Cosine similarity is undefined for a zero-magnitude vector.")

    value = float(np.dot(left_vector, right_vector) / (left_norm * right_norm))
    # Floating-point round-off can produce values like 1.0000001.
    return float(np.clip(value, -1.0, 1.0))


def euclidean_distance(left: VectorInput, right: VectorInput) -> float:
    """Return straight-line (L2) distance between two vectors."""
    left_vector, right_vector = _validate_pair(left, right)
    return float(np.linalg.norm(left_vector - right_vector))


def dot_product(left: VectorInput, right: VectorInput) -> float:
    """Return the inner product of two vectors.

    For unit-normalized vectors, this is equivalent to cosine similarity, while
    preserving the magnitude information for vectors that are not normalized.
    """
    left_vector, right_vector = _validate_pair(left, right)
    return float(np.dot(left_vector, right_vector))


def manhattan_distance(left: VectorInput, right: VectorInput) -> float:
    """Return L1 distance: the sum of absolute coordinate differences."""
    left_vector, right_vector = _validate_pair(left, right)
    return float(np.abs(left_vector - right_vector).sum(dtype=np.float64))


def calculate_metric(left: VectorInput, right: VectorInput, metric: DistanceMetric) -> float:
    """Calculate ``metric`` for two compatible vectors.

    Keeping dispatch here gives Brute Force, KD-Tree, and HNSW one consistent
    metric contract rather than duplicating conditional logic in every index.
    """
    match metric:
        case DistanceMetric.COSINE:
            return cosine_similarity(left, right)
        case DistanceMetric.EUCLIDEAN:
            return euclidean_distance(left, right)
        case DistanceMetric.DOT_PRODUCT:
            return dot_product(left, right)
        case DistanceMetric.MANHATTAN:
            return manhattan_distance(left, right)
        case _:
            raise ValidationError(f"Unsupported distance metric: {metric!r}.")


def is_similarity_metric(metric: DistanceMetric) -> bool:
    """Return whether larger metric values represent better matches."""
    if metric in (DistanceMetric.COSINE, DistanceMetric.DOT_PRODUCT):
        return True
    if metric in (DistanceMetric.EUCLIDEAN, DistanceMetric.MANHATTAN):
        return False
    raise ValidationError(f"Unsupported distance metric: {metric!r}.")


def _validate_pair(left: VectorInput, right: VectorInput) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Convert a vector pair to float32 and validate metric preconditions."""
    left_vector = _normalize_input(left, "left")
    right_vector = _normalize_input(right, "right")
    if left_vector.size != right_vector.size:
        raise DimensionMismatchError(
            f"Cannot compare vectors with dimensions {left_vector.size} and {right_vector.size}."
        )
    return left_vector, right_vector


def _normalize_input(vector: VectorInput, parameter_name: str) -> NDArray[np.float32]:
    """Return a finite, one-dimensional contiguous float32 array."""
    try:
        normalized = np.asarray(vector, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{parameter_name} vector must contain numeric values.") from error
    if normalized.ndim != 1 or normalized.size == 0:
        raise ValidationError(f"{parameter_name} vector must be non-empty and one-dimensional.")
    if not np.isfinite(normalized).all():
        raise ValidationError(f"{parameter_name} vector must contain only finite values.")
    return np.ascontiguousarray(normalized)
