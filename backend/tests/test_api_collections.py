import pytest
from fastapi.testclient import TestClient

from vectornest.api.app import create_app
from vectornest.api.dependencies import get_storage
from vectornest.storage.engine import InMemoryStorage


@pytest.fixture
def storage() -> InMemoryStorage:
    return InMemoryStorage()


@pytest.fixture
def client(storage: InMemoryStorage) -> TestClient:
    application = create_app()

    application.dependency_overrides[get_storage] = lambda: storage

    return TestClient(application)


def test_create_collection(client: TestClient) -> None:
    response = client.post(
        "/collections",
        json={
            "name": "documents",
            "dimension": 3,
            "distance_metric": "cosine",
        },
    )

    assert response.status_code == 201

    assert response.json() == {
        "name": "documents",
        "dimension": 3,
        "distance_metric": "cosine",
    }


def test_get_collection(client: TestClient) -> None:
    client.post(
        "/collections",
        json={
            "name": "documents",
            "dimension": 3,
            "distance_metric": "euclidean",
        },
    )

    response = client.get(
        "/collections/documents"
    )

    assert response.status_code == 200

    assert response.json() == {
        "name": "documents",
        "dimension": 3,
        "distance_metric": "euclidean",
    }


def test_delete_collection(client: TestClient) -> None:
    client.post(
        "/collections",
        json={
            "name": "documents",
            "dimension": 3,
        },
    )

    response = client.delete(
        "/collections/documents"
    )

    assert response.status_code == 204
    assert response.content == b""


def test_invalid_dimension_is_rejected(
    client: TestClient,
) -> None:
    response = client.post(
        "/collections",
        json={
            "name": "documents",
            "dimension": 0,
        },
    )

    assert response.status_code == 422