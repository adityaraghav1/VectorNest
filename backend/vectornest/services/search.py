"""Search orchestration for VectorNest collections."""

from vectornest.core.exceptions import ValidationError
from vectornest.core.types import DistanceMetric
from vectornest.indexes.brute_force import BruteForceIndex
from vectornest.indexes.results import SearchResult
from vectornest.metrics.functions import VectorInput
from vectornest.storage.engine import InMemoryStorage


class SearchService:
    """Coordinate stored collections with vector indexes."""

    def __init__(self, storage: InMemoryStorage) -> None:
        self.storage = storage

    def search(
        self,
        collection_name: str,
        query: VectorInput,
        metric: DistanceMetric,
        k: int | None = None,
    ) -> list[SearchResult]:
        """Search a collection and return ranked matches."""
        collection = self.storage.get_collection(collection_name)

        if k is not None and k <= 0:
            raise ValidationError("k must be greater than zero.")

        index = BruteForceIndex(
            dimension=collection.dimension,
            metric=metric,
        )

        index.add_many(
            self.storage.list_records(collection_name)
        )

        if k is None:
            return index.search(query)

        return index.search(query, limit=k)