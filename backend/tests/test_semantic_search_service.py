import numpy as np
import pytest

from vectornest.core.exceptions import DimensionMismatchError
from vectornest.core.types import DistanceMetric, IndexType
from vectornest.embeddings.base import EmbeddingProvider
from vectornest.models.collection import CollectionConfig
from vectornest.models.record import VectorRecord
from vectornest.query.filters import MetadataFilter
from vectornest.services.semantic_search import SemanticSearchService
from vectornest.storage.engine import InMemoryStorage


class FakeEmbeddingProvider(EmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 2

    def embed_text(self, text: str) -> np.ndarray:
        normalized = self.validate_text(text)

        if "python" in normalized.lower():
            embedding = np.array(
                [1.0, 0.0]
            )
        else:
            embedding = np.array(
                [0.0, 1.0]
            )

        return self.normalize_embedding(
            embedding,
            self.dimension,
        )


def create_storage() -> InMemoryStorage:
    storage = InMemoryStorage()

    storage.create_collection(
        CollectionConfig(
            name="documents",
            dimension=2,
        )
    )

    storage.insert_record(
        "documents",
        VectorRecord(
            id="python",
            vector=np.array(
                [1.0, 0.0]
            ),
            metadata={
                "topic": "programming",
            },
            document="Python programming guide.",
        ),
    )

    storage.insert_record(
        "documents",
        VectorRecord(
            id="travel",
            vector=np.array(
                [0.0, 1.0]
            ),
            metadata={
                "topic": "travel",
            },
            document="Travel destination guide.",
        ),
    )

    return storage


def test_semantic_search_embeds_query_text() -> None:
    storage = create_storage()

    service = SemanticSearchService(
        storage=storage,
        embedding_provider=FakeEmbeddingProvider(),
    )

    results = service.search(
        "documents",
        "Tell me about Python programming",
        metric=DistanceMetric.COSINE,
        k=1,
    )

    assert len(results) == 1
    assert results[0].record.id == "python"


def test_semantic_search_returns_ranked_results() -> None:
    storage = create_storage()

    service = SemanticSearchService(
        storage=storage,
        embedding_provider=FakeEmbeddingProvider(),
    )

    results = service.search(
        "documents",
        "Where should I travel?",
        metric=DistanceMetric.COSINE,
        k=2,
    )

    assert len(results) == 2
    assert results[0].record.id == "travel"


def test_semantic_search_supports_metadata_filter() -> None:
    storage = create_storage()

    service = SemanticSearchService(
        storage=storage,
        embedding_provider=FakeEmbeddingProvider(),
    )

    results = service.search(
        "documents",
        "Python programming",
        metric=DistanceMetric.COSINE,
        k=5,
        metadata_filter=MetadataFilter(
            {
                "topic": "travel",
            }
        ),
    )

    assert len(results) == 1
    assert results[0].record.id == "travel"


def test_semantic_search_supports_index_selection() -> None:
    storage = create_storage()

    service = SemanticSearchService(
        storage=storage,
        embedding_provider=FakeEmbeddingProvider(),
    )

    results = service.search(
        "documents",
        "Python programming",
        metric=DistanceMetric.EUCLIDEAN,
        k=1,
        index_type=IndexType.KD_TREE,
    )

    assert len(results) == 1
    assert results[0].record.id == "python"


def test_semantic_search_rejects_dimension_mismatch() -> None:
    storage = InMemoryStorage()

    storage.create_collection(
        CollectionConfig(
            name="documents",
            dimension=3,
        )
    )

    service = SemanticSearchService(
        storage=storage,
        embedding_provider=FakeEmbeddingProvider(),
    )

    with pytest.raises(
        DimensionMismatchError,
        match="dimension",
    ):
        service.search(
            "documents",
            "Python",
            metric=DistanceMetric.COSINE,
        )