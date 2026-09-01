import pytest
from fastapi.testclient import TestClient

from vectornest.api.app import create_app
from vectornest.api.dependencies import get_storage
from vectornest.models.collection import CollectionConfig
from vectornest.models.record import VectorRecord
from vectornest.storage.engine import InMemoryStorage


@pytest.fixture
def storage() -> InMemoryStorage:
    return InMemoryStorage()


@pytest.fixture
def client(storage: InMemoryStorage) -> TestClient:
    application = create_app()
    application.dependency_overrides[get_storage] = lambda: storage

    return TestClient(application)


def test_duplicate_collection_returns_409(
    client: TestClient,
) -> None:
    payload = {
        "name": "documents",
        "dimension": 3,
        "distance_metric": "cosine",
    }

    first_response = client.post(
        "/collections",
        json=payload,
    )
    second_response = client.post(
        "/collections",
        json=payload,
    )

    assert first_response.status_code == 201
    assert second_response.status_code == 409
    assert second_response.json() == {
        "detail": "Collection 'documents' already exists."
    }


def test_missing_collection_returns_404(
    client: TestClient,
) -> None:
    response = client.get(
        "/collections/missing"
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Collection 'missing' does not exist."
    }


def test_duplicate_record_returns_409(
    client: TestClient,
    storage: InMemoryStorage,
) -> None:
    storage.create_collection(
        CollectionConfig(
            name="documents",
            dimension=2,
        )
    )

    payload = {
        "id": "r1",
        "vector": [1.0, 0.0],
    }

    first_response = client.post(
        "/collections/documents/records",
        json=payload,
    )
    second_response = client.post(
        "/collections/documents/records",
        json=payload,
    )

    assert first_response.status_code == 201
    assert second_response.status_code == 409
    assert second_response.json() == {
        "detail": "Record 'r1' already exists."
    }


def test_missing_record_returns_404(
    client: TestClient,
    storage: InMemoryStorage,
) -> None:
    storage.create_collection(
        CollectionConfig(
            name="documents",
            dimension=2,
        )
    )

    response = client.get(
        "/collections/documents/records/missing"
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": (
            "Record 'missing' does not exist "
            "in collection 'documents'."
        )
    }


def test_dimension_mismatch_returns_422(
    client: TestClient,
    storage: InMemoryStorage,
) -> None:
    storage.create_collection(
        CollectionConfig(
            name="documents",
            dimension=3,
        )
    )

    response = client.post(
        "/collections/documents/records",
        json={
            "id": "r1",
            "vector": [1.0, 2.0],
        },
    )

    assert response.status_code == 422
    assert "dimension" in response.json()["detail"].lower()


def test_invalid_search_configuration_returns_422(
    client: TestClient,
    storage: InMemoryStorage,
) -> None:
    storage.create_collection(
        CollectionConfig(
            name="documents",
            dimension=2,
        )
    )

    storage.insert_record(
        "documents",
        VectorRecord(
            id="r1",
            vector=[1.0, 0.0],
        ),
    )

    response = client.post(
        "/collections/documents/search",
        json={
            "query_vector": [1.0, 0.0],
            "metric": "cosine",
            "index_type": "kd_tree",
            "k": 1,
        },
    )

    assert response.status_code == 422