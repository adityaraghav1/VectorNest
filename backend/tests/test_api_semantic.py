import numpy as np
from fastapi.testclient import TestClient

from vectornest.api.app import create_app
from vectornest.api.dependencies import (
    get_embedding_provider,
    get_storage,
)
from vectornest.embeddings.base import EmbeddingProvider
from vectornest.storage.engine import InMemoryStorage


class FakeSemanticEmbeddingProvider(EmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 2

    def embed_text(self, text: str) -> np.ndarray:
        normalized = self.validate_text(text).lower()

        if (
            "python" in normalized
            or "programming" in normalized
            or "machine learning" in normalized
        ):
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


def create_client() -> TestClient:
    storage = InMemoryStorage()
    embedding_provider = FakeSemanticEmbeddingProvider()

    application = create_app()

    application.dependency_overrides[
        get_storage
    ] = lambda: storage

    application.dependency_overrides[
        get_embedding_provider
    ] = lambda: embedding_provider

    return TestClient(application)


def test_document_ingestion_and_semantic_search() -> None:
    client = create_client()

    response = client.post(
        "/collections",
        json={
            "name": "knowledge",
            "dimension": 2,
            "distance_metric": "cosine",
        },
    )

    assert response.status_code == 201

    response = client.post(
        "/collections/knowledge/documents",
        json={
            "document_id": "python-guide",
            "document": (
                "Python is a programming language commonly "
                "used for machine learning and artificial "
                "intelligence."
            ),
            "metadata": {
                "category": "technology",
            },
        },
    )

    assert response.status_code == 201

    body = response.json()

    assert body["document_id"] == "python-guide"
    assert body["chunks_created"] == 1
    assert body["record_ids"] == [
        "python-guide:chunk:0"
    ]

    response = client.post(
        "/collections/knowledge/semantic-search",
        json={
            "query": (
                "Which programming language is useful "
                "for machine learning?"
            ),
            "metric": "cosine",
            "index_type": "brute_force",
            "k": 1,
        },
    )

    assert response.status_code == 200

    results = response.json()["results"]

    assert len(results) == 1
    assert results[0]["id"] == "python-guide:chunk:0"
    assert results[0]["metadata"]["document_id"] == (
        "python-guide"
    )


def test_semantic_search_supports_metadata_filter() -> None:
    client = create_client()

    client.post(
        "/collections",
        json={
            "name": "knowledge",
            "dimension": 2,
        },
    )

    client.post(
        "/collections/knowledge/documents",
        json={
            "document_id": "python-guide",
            "document": "Python programming and machine learning.",
            "metadata": {
                "category": "technology",
            },
        },
    )

    client.post(
        "/collections/knowledge/documents",
        json={
            "document_id": "travel-guide",
            "document": "Beautiful mountains and travel destinations.",
            "metadata": {
                "category": "travel",
            },
        },
    )

    response = client.post(
        "/collections/knowledge/semantic-search",
        json={
            "query": "Python programming",
            "k": 5,
            "metadata_filter": {
                "category": "travel",
            },
        },
    )

    assert response.status_code == 200

    results = response.json()["results"]

    assert len(results) == 1
    assert results[0]["metadata"]["category"] == "travel"


def test_semantic_routes_appear_in_openapi() -> None:
    client = create_client()

    response = client.get("/openapi.json")

    assert response.status_code == 200

    paths = response.json()["paths"]

    assert (
        "/collections/{collection_name}/documents"
        in paths
    )

    assert (
        "/collections/{collection_name}/semantic-search"
        in paths
    )