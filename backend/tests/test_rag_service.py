import numpy as np

from vectornest.core.types import DistanceMetric
from vectornest.embeddings.base import EmbeddingProvider
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


class FakeLLMClient:
    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
    ) -> dict:
        return {
            "message": {
                "content": (
                    "Python is commonly used "
                    "for machine learning."
                )
            }
        }


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

    provider = FakeEmbeddingProvider()

    semantic_service = SemanticSearchService(
        storage,
        provider,
    )

    rag_service = RAGService(
        semantic_search_service=semantic_service,
        llm_client=FakeLLMClient(),
        model="fake-model",
        minimum_score=0.45,
    )

    result = rag_service.answer(
        "knowledge",
        "How is Python used?",
        metric=DistanceMetric.COSINE,
        k=2,
    )

    assert (
        result.answer
        == (
            "Python is commonly used "
            "for machine learning."
        )
    )

    assert len(result.sources) == 1

    assert (
        result.sources[0].id
        == "python:chunk:0"
    )

    assert all(
        source.id != "unrelated:chunk:0"
        for source in result.sources
    )