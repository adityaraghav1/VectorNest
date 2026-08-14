"""Shared domain types used throughout the engine."""

from enum import StrEnum


class DistanceMetric(StrEnum):
    """Metrics supported by VectorNest search indexes."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"
