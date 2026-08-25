"""Shared domain types used throughout the engine."""

from enum import StrEnum


class DistanceMetric(StrEnum):
    """Metrics supported by VectorNest search indexes."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


class IndexType(StrEnum):
    """Vector index implementations available in VectorNest."""

    BRUTE_FORCE = "brute_force"
    KD_TREE = "kd_tree"