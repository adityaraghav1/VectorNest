"""Abstract contract shared by VectorNest indexes."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

from vectornest.indexes.results import SearchResult
from vectornest.metrics.functions import VectorInput
from vectornest.models.record import VectorRecord


class VectorIndex(ABC):
    """Define the common interface implemented by every vector index."""

    @abstractmethod
    def add(self, record: VectorRecord) -> None:
        """Add one record to the index."""

    @abstractmethod
    def add_many(self, records: Iterable[VectorRecord]) -> None:
        """Add multiple records to the index."""

    @abstractmethod
    def remove(self, record_id: str) -> None:
        """Remove one record from the index."""

    @abstractmethod
    def get(self, record_id: str) -> VectorRecord:
        """Return one record by ID."""

    @abstractmethod
    def search(
        self,
        query: VectorInput,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Return the best matches for a query vector."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of indexed records."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all records from the index."""