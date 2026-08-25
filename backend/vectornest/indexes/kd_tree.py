"""KD-tree vector index implementation."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vectornest.core.exceptions import (
    DimensionMismatchError,
    RecordNotFoundError,
    ValidationError,
)
from vectornest.core.types import DistanceMetric
from vectornest.indexes.base import VectorIndex
from vectornest.indexes.results import SearchResult
from vectornest.metrics.functions import VectorInput, euclidean_distance
from vectornest.models.record import VectorRecord


@dataclass(slots=True)
class _KDNode:
    """Represent one node inside the KD-tree."""

    record: VectorRecord
    axis: int
    left: "_KDNode | None" = None
    right: "_KDNode | None" = None


class KDTreeIndex(VectorIndex):
    """Index vectors using a balanced KD-tree."""

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValidationError(
                "Index dimension must be greater than zero."
            )

        self.dimension = dimension
        self.metric = DistanceMetric.EUCLIDEAN
        self._records: dict[str, VectorRecord] = {}
        self._root: _KDNode | None = None
        self._dirty = False

    def add(self, record: VectorRecord) -> None:
        """Add or replace a record in the index."""
        record.ensure_dimension(self.dimension)

        self._records[record.id] = record
        self._dirty = True

    def add_many(
        self,
        records: Iterable[VectorRecord],
    ) -> None:
        """Add multiple records to the index."""
        for record in records:
            self.add(record)

    def remove(self, record_id: str) -> None:
        """Remove a record from the index."""
        if record_id not in self._records:
            raise RecordNotFoundError(
                f"Record '{record_id}' does not exist in the index."
            )

        del self._records[record_id]
        self._dirty = True

    def get(self, record_id: str) -> VectorRecord:
        """Return a record by ID."""
        try:
            return self._records[record_id]
        except KeyError as error:
            raise RecordNotFoundError(
                f"Record '{record_id}' does not exist in the index."
            ) from error

    def search(
        self,
        query: VectorInput,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Return nearest records using Euclidean distance."""
        query_vector = self._validate_query(query)

        if limit <= 0:
            raise ValidationError(
                "Search limit must be greater than zero."
            )

        self._ensure_tree()

        if self._root is None:
            return []

        candidates: list[SearchResult] = []

        self._search_node(
            self._root,
            query_vector,
            limit,
            candidates,
        )

        candidates.sort(
            key=lambda result: result.score
        )

        return candidates[:limit]

    def count(self) -> int:
        """Return the number of indexed records."""
        return len(self._records)

    def clear(self) -> None:
        """Remove all records from the index."""
        self._records.clear()
        self._root = None
        self._dirty = False

    def _ensure_tree(self) -> None:
        """Rebuild the tree when records have changed."""
        if not self._dirty:
            return

        self._root = self._build_tree(
            list(self._records.values()),
            depth=0,
        )

        self._dirty = False

    def _build_tree(
        self,
        records: list[VectorRecord],
        depth: int,
    ) -> _KDNode | None:
        """Recursively build a balanced KD-tree."""
        if not records:
            return None

        axis = depth % self.dimension

        records.sort(
            key=lambda record: float(
                record.vector[axis]
            )
        )

        median = len(records) // 2

        return _KDNode(
            record=records[median],
            axis=axis,
            left=self._build_tree(
                records[:median],
                depth + 1,
            ),
            right=self._build_tree(
                records[median + 1 :],
                depth + 1,
            ),
        )

    def _search_node(
        self,
        node: _KDNode | None,
        query: NDArray[np.float32],
        limit: int,
        candidates: list[SearchResult],
    ) -> None:
        """Recursively search the KD-tree."""
        if node is None:
            return

        score = euclidean_distance(
            query,
            node.record.vector,
        )

        candidates.append(
            SearchResult(
                record=node.record,
                score=score,
            )
        )

        candidates.sort(
            key=lambda result: result.score
        )

        if len(candidates) > limit:
            candidates.pop()

        axis = node.axis

        difference = float(
            query[axis] - node.record.vector[axis]
        )

        if difference <= 0:
            near_branch = node.left
            far_branch = node.right
        else:
            near_branch = node.right
            far_branch = node.left

        self._search_node(
            near_branch,
            query,
            limit,
            candidates,
        )

        worst_distance = (
            candidates[-1].score
            if len(candidates) == limit
            else float("inf")
        )

        if abs(difference) <= worst_distance:
            self._search_node(
                far_branch,
                query,
                limit,
                candidates,
            )

    def _validate_query(
        self,
        query: VectorInput,
    ) -> NDArray[np.float32]:
        """Validate and normalize a query vector."""
        try:
            query_vector = np.asarray(
                query,
                dtype=np.float32,
            )
        except (TypeError, ValueError) as error:
            raise ValidationError(
                "Query vector must contain numeric values."
            ) from error

        if (
            query_vector.ndim != 1
            or query_vector.size == 0
        ):
            raise ValidationError(
                "Query vector must be non-empty "
                "and one-dimensional."
            )

        if not np.isfinite(query_vector).all():
            raise ValidationError(
                "Query vector must contain only finite values."
            )

        query_vector = np.ascontiguousarray(
            query_vector
        )

        if query_vector.size != self.dimension:
            raise DimensionMismatchError(
                f"Query vector has dimension "
                f"{query_vector.size}, "
                f"but index dimension is "
                f"{self.dimension}."
            )

        return query_vector