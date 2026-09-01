import pytest
from fastapi.testclient import TestClient

from vectornest.api.app import create_app
from vectornest.api.dependencies import get_storage
from vectornest.models.collection import CollectionConfig
from vectornest.models.record import VectorRecord
from vectornest.storage.engine import InMemoryStorage


@pytest.fixture
def storage() -> InMemoryStorage:
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
            id="best",
            vector=[1.0, 0.0],
            metadata={"category": "ai"},
            document="Best match.",
        ),
    )

    storage.insert_record(
        "documents",
        VectorRecord(
            id="middle",
            vector=[0.8, 0.2],
            metadata={"category": "tech"},
            document="Middle match.",
        ),
    )

    storage.insert_record(
        "documents",
        VectorRecord(
            id="worst",
            vector=[0.0, 1.0],
            metadata={"category": "other"},
            document="Worst match.",
        ),
    )

    return storage


@pytest.fixture
def client(storage: InMemoryStorage) -> TestClient:
    application = create_app()

    application.dependency_overrides[
        get_storage
    ] = lambda: storage

    return TestClient(application)


def test_search_with_brute_force(
    client: TestClient,
) -> None:
    response = client.post(
        "/collections/documents/search",
        json={
            "query_vector": [1.0, 0.0],
            "metric": "cosine",
            "index_type": "brute_force",
            "k": 2,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["results"]) == 2
    assert data["results"][0]["id"] == "best"
    assert data["results"][0]["score"] == pytest.approx(
        1.0
    )


def test_search_with_metadata_filter(
    client: TestClient,
) -> None:
    response = client.post(
        "/collections/documents/search",
        json={
            "query_vector": [1.0, 0.0],
            "metric": "cosine",
            "index_type": "brute_force",
            "k": 3,
            "metadata_filter": {
                "category": "tech",
            },
        },
    )

    assert response.status_code == 200

    results = response.json()["results"]

    assert len(results) == 1
    assert results[0]["id"] == "middle"
    assert results[0]["metadata"]["category"] == "tech"


def test_search_without_matching_metadata_returns_empty_results(
    client: TestClient,
) -> None:
    response = client.post(
        "/collections/documents/search",
        json={
            "query_vector": [1.0, 0.0],
            "metadata_filter": {
                "category": "missing",
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["results"] == []

def test_search_with_kd_tree(
    client: TestClient,
) -> None:
    response = client.post(
        "/collections/documents/search",
        json={
            "query_vector": [1.0, 0.0],
            "metric": "euclidean",
            "index_type": "kd_tree",
            "k": 1,
        },
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["id"] == "best"


def test_search_with_hnsw(
    client: TestClient,
) -> None:
    response = client.post(
        "/collections/documents/search",
        json={
            "query_vector": [1.0, 0.0],
            "metric": "cosine",
            "index_type": "hnsw",
            "k": 1,
        },
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["id"] == "best"


def test_search_uses_default_options(
    client: TestClient,
) -> None:
    response = client.post(
        "/collections/documents/search",
        json={
            "query_vector": [1.0, 0.0],
        },
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["id"] == "best"


def test_search_rejects_invalid_k(
    client: TestClient,
) -> None:
    response = client.post(
        "/collections/documents/search",
        json={
            "query_vector": [1.0, 0.0],
            "k": 0,
        },
    )

    assert response.status_code == 422


def test_search_rejects_empty_query_vector(
    client: TestClient,
) -> None:
    response = client.post(
        "/collections/documents/search",
        json={
            "query_vector": [],
        },
    )

    assert response.status_code == 422