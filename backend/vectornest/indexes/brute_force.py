"""Brute-force vector index implementation."""

from collections.abc import Iterable

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
from vectornest.metrics.functions import (
    VectorInput,
    calculate_metric,
    is_similarity_metric,
)
from vectornest.models.record import VectorRecord


class BruteForceIndex(VectorIndex):
    """Store vectors and search them by comparing every record."""

    def __init__(
        self,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
    ) -> None:
        if dimension <= 0:
            raise ValidationError("Index dimension must be greater than zero.")

        self.dimension = dimension
        self.metric = metric
        self._records: dict[str, VectorRecord] = {}

    def add(self, record: VectorRecord) -> None:
        """Add a vector record to the index."""
        record.ensure_dimension(self.dimension)
        self._records[record.id] = record

    def add_many(self, records: Iterable[VectorRecord]) -> None:
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
        """Return the best matching records for a query vector."""
        query_vector = self._validate_query(query)

        if limit <= 0:
            raise ValidationError("Search limit must be greater than zero.")

        results = [
            SearchResult(
                record=record,
                score=calculate_metric(
                    query_vector,
                    record.vector,
                    self.metric,
                ),
            )
            for record in self._records.values()
        ]

        results.sort(
            key=lambda result: result.score,
            reverse=is_similarity_metric(self.metric),
        )

        return results[:limit]

    def count(self) -> int:
        """Return the number of indexed records."""
        return len(self._records)

    def clear(self) -> None:
        """Remove every record from the index."""
        self._records.clear()

    def _validate_query(
        self,
        query: VectorInput,
    ) -> NDArray[np.float32]:
        """Validate and normalize a search query."""
        try:
            query_vector = np.asarray(query, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValidationError(
                "Query vector must contain numeric values."
            ) from error

        if query_vector.ndim != 1 or query_vector.size == 0:
            raise ValidationError(
                "Query vector must be non-empty and one-dimensional."
            )

        if not np.isfinite(query_vector).all():
            raise ValidationError(
                "Query vector must contain only finite values."
            )

        query_vector = np.ascontiguousarray(query_vector)

        if query_vector.size != self.dimension:
            raise DimensionMismatchError(
                f"Query vector has dimension {query_vector.size}, "
                f"but index dimension is {self.dimension}."
            )

        return query_vector