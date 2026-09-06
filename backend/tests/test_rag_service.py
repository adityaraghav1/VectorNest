from collections.abc import Iterator

import numpy as np
import pytest

from vectornest.core.exceptions import ExternalServiceError
from vectornest.core.types import DistanceMetric
from vectornest.embeddings.base import EmbeddingProvider
from vectornest.generation.base import GenerationProvider
from vectornest.models.collection import CollectionConfig
from vectornest.models.record import VectorRecord
from vectornest.services.rag import RAGService
from vectornest.services.semantic_search import SemanticSearchService
from vectornest.storage.engine import InMemoryStorage


class FakeEmbeddingProvider(EmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 3

    def embed_text(
        self,
        text: str,
    ) -> np.ndarray:
        self.validate_text(text)

        return np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        )


class FakeGenerationProvider(GenerationProvider):
    def generate(
        self,
        prompt: str,
    ) -> str:
        return (
            "Python is commonly used "
            "for machine learning."
        )

    def stream(
        self,
        prompt: str,
    ) -> Iterator[str]:
        return iter(
            [
                "Python ",
                (
                    "is commonly used "
                    "for machine learning."
                ),
            ]
        )


class FailingGenerationProvider(GenerationProvider):
    def generate(
        self,
        prompt: str,
    ) -> str:
        raise ExternalServiceError(
            "Generation provider failed."
        )

    def stream(
        self,
        prompt: str,
    ) -> Iterator[str]:
        raise ExternalServiceError(
            "Generation provider stream failed."
        )


def make_storage(
    *,
    metric: DistanceMetric,
    record_id: str,
    vector: np.ndarray,
    document: str,
) -> InMemoryStorage:
    storage = InMemoryStorage()

    storage.create_collection(
        CollectionConfig(
            name="knowledge",
            dimension=3,
            distance_metric=metric,
        )
    )

    storage.insert_record(
        "knowledge",
        VectorRecord(
            id=record_id,
            vector=vector,
            document=document,
        ),
    )

    return storage


def make_rag_service(
    storage: InMemoryStorage,
    *,
    generation_provider: GenerationProvider | None = None,
    minimum_score: float = 0.45,
) -> RAGService:
    embedding_provider = FakeEmbeddingProvider()

    semantic_service = SemanticSearchService(
        storage,
        embedding_provider,
    )

    return RAGService(
        semantic_search_service=semantic_service,
        generation_provider=(
            generation_provider
            if generation_provider is not None
            else FakeGenerationProvider()
        ),
        minimum_score=minimum_score,
    )


def test_rag_returns_answer_and_sources() -> None:
    storage = InMemoryStorage()

    storage.create_collection(
        CollectionConfig(
            name="knowledge",
            dimension=3,
            distance_metric=DistanceMetric.COSINE,
        )
    )

    storage.insert_record(
        "knowledge",
        VectorRecord(
            id="python:chunk:0",
            vector=np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            document=(
                "Python is widely used "
                "for machine learning."
            ),
        ),
    )

    storage.insert_record(
        "knowledge",
        VectorRecord(
            id="unrelated:chunk:0",
            vector=np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float32,
            ),
            document=(
                "Vector databases can persist records "
                "between application restarts."
            ),
        ),
    )

    rag_service = make_rag_service(storage)

    result = rag_service.answer(
        "knowledge",
        "How is Python used?",
        metric=DistanceMetric.COSINE,
        k=2,
    )

    assert result.answer == (
        "Python is commonly used "
        "for machine learning."
    )

    assert len(result.sources) == 1

    assert result.sources[0].id == "python:chunk:0"

    assert all(
        source.id != "unrelated:chunk:0"
        for source in result.sources
    )


def test_rag_filters_low_cosine_similarity() -> None:
    storage = make_storage(
        metric=DistanceMetric.COSINE,
        record_id="low-score:chunk:0",
        vector=np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float32,
        ),
        document=(
            "This document is not similar "
            "to the query vector."
        ),
    )

    rag_service = make_rag_service(storage)

    result = rag_service.answer(
        "knowledge",
        "Tell me about Python.",
        metric=DistanceMetric.COSINE,
        k=1,
    )

    assert result.sources == []

    assert result.answer == (
        "I could not find relevant context "
        "in this collection."
    )


def test_rag_does_not_apply_cosine_threshold_to_euclidean() -> None:
    storage = make_storage(
        metric=DistanceMetric.EUCLIDEAN,
        record_id="distance:chunk:0",
        vector=np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        document=(
            "Python is widely used "
            "for machine learning."
        ),
    )

    rag_service = make_rag_service(storage)

    result = rag_service.answer(
        "knowledge",
        "How is Python used?",
        metric=DistanceMetric.EUCLIDEAN,
        k=1,
    )

    assert len(result.sources) == 1

    assert result.sources[0].id == "distance:chunk:0"

    assert result.sources[0].score == 0.0


def test_rag_propagates_generation_provider_failure() -> None:
    storage = make_storage(
        metric=DistanceMetric.COSINE,
        record_id="python:chunk:0",
        vector=np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        document=(
            "Python is widely used "
            "for machine learning."
        ),
    )

    rag_service = make_rag_service(
        storage,
        generation_provider=FailingGenerationProvider(),
    )

    with pytest.raises(
        ExternalServiceError,
        match="Generation provider failed.",
    ):
        rag_service.answer(
            "knowledge",
            "How is Python used?",
            metric=DistanceMetric.COSINE,
            k=1,
        )


def test_rag_propagates_stream_provider_failure() -> None:
    storage = make_storage(
        metric=DistanceMetric.COSINE,
        record_id="python:chunk:0",
        vector=np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        document=(
            "Python is widely used "
            "for machine learning."
        ),
    )

    rag_service = make_rag_service(
        storage,
        generation_provider=FailingGenerationProvider(),
    )

    with pytest.raises(
        ExternalServiceError,
        match="Generation provider stream failed.",
    ):
        rag_service.stream_answer(
            "knowledge",
            "How is Python used?",
            metric=DistanceMetric.COSINE,
            k=1,
        )