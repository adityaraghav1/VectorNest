"""Semantic search service for VectorNest."""

from vectornest.core.exceptions import DimensionMismatchError
from vectornest.core.types import DistanceMetric, IndexType
from vectornest.embeddings.base import EmbeddingProvider
from vectornest.query.filters import MetadataFilter
from vectornest.services.search import SearchResult, SearchService
from vectornest.storage.base import StorageBackend


class SemanticSearchService:
    """Search VectorNest collections using natural-language queries."""

    def __init__(
        self,
        storage: StorageBackend,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._storage = storage
        self._embedding_provider = embedding_provider
        self._search_service = SearchService(storage)

    def search(
        self,
        collection_name: str,
        query_text: str,
        *,
        metric: DistanceMetric,
        k: int = 10,
        metadata_filter: MetadataFilter | None = None,
        index_type: IndexType = IndexType.BRUTE_FORCE,
    ) -> list[SearchResult]:
        """Embed a text query and search the collection."""
        collection = self._storage.get_collection(
            collection_name
        )

        if (
            collection.dimension
            != self._embedding_provider.dimension
        ):
            raise DimensionMismatchError(
                "Embedding provider dimension does not match "
                "collection dimension."
            )

        query_vector = self._embedding_provider.embed_text(
            query_text
        )

        return self._search_service.search(
            collection_name,
            query_vector,
            metric=metric,
            k=k,
            metadata_filter=metadata_filter,
            index_type=index_type,
        )