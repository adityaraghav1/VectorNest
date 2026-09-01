import pytest
from fastapi.testclient import TestClient

from vectornest.api.app import create_app
from vectornest.api.dependencies import get_storage
from vectornest.models.collection import CollectionConfig
from vectornest.storage.engine import InMemoryStorage


@pytest.fixture
def storage() -> InMemoryStorage:
    storage = InMemoryStorage()

    storage.create_collection(
        CollectionConfig(
            name="documents",
            dimension=3,
        )
    )

    return storage


@pytest.fixture
def client(storage: InMemoryStorage) -> TestClient:
    application = create_app()

    application.dependency_overrides[
        get_storage
    ] = lambda: storage

    return TestClient(application)


def test_create_record(client: TestClient) -> None:
    response = client.post(
        "/collections/documents/records",
        json={
            "id": "r1",
            "vector": [1.0, 2.0, 3.0],
            "metadata": {
                "category": "ai",
            },
            "document": "Vector databases store embeddings.",
        },
    )

    assert response.status_code == 201

    assert response.json() == {
        "id": "r1",
        "vector": [1.0, 2.0, 3.0],
        "metadata": {
            "category": "ai",
        },
        "document": "Vector databases store embeddings.",
    }


def test_get_record(client: TestClient) -> None:
    client.post(
        "/collections/documents/records",
        json={
            "id": "r1",
            "vector": [1.0, 2.0, 3.0],
        },
    )

    response = client.get(
        "/collections/documents/records/r1"
    )

    assert response.status_code == 200
    assert response.json()["id"] == "r1"
    assert response.json()["vector"] == [
        1.0,
        2.0,
        3.0,
    ]


def test_list_records(client: TestClient) -> None:
    client.post(
        "/collections/documents/records",
        json={
            "id": "r1",
            "vector": [1.0, 2.0, 3.0],
        },
    )

    client.post(
        "/collections/documents/records",
        json={
            "id": "r2",
            "vector": [4.0, 5.0, 6.0],
        },
    )

    response = client.get(
        "/collections/documents/records"
    )

    assert response.status_code == 200

    ids = {
        record["id"]
        for record in response.json()
    }

    assert ids == {"r1", "r2"}


def test_update_record(client: TestClient) -> None:
    client.post(
        "/collections/documents/records",
        json={
            "id": "r1",
            "vector": [1.0, 2.0, 3.0],
        },
    )

    response = client.put(
        "/collections/documents/records/r1",
        json={
            "vector": [7.0, 8.0, 9.0],
            "metadata": {
                "updated": True,
            },
            "document": "Updated document.",
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "id": "r1",
        "vector": [7.0, 8.0, 9.0],
        "metadata": {
            "updated": True,
        },
        "document": "Updated document.",
    }


def test_delete_record(client: TestClient) -> None:
    client.post(
        "/collections/documents/records",
        json={
            "id": "r1",
            "vector": [1.0, 2.0, 3.0],
        },
    )

    response = client.delete(
        "/collections/documents/records/r1"
    )

    assert response.status_code == 204
    assert response.content == b""