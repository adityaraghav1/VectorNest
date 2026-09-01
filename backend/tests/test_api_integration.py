from fastapi.testclient import TestClient

from vectornest.api.app import create_app
from vectornest.api.dependencies import get_storage
from vectornest.storage.engine import InMemoryStorage


def test_complete_api_workflow() -> None:
    storage = InMemoryStorage()

    application = create_app()
    application.dependency_overrides[get_storage] = lambda: storage

    client = TestClient(application)

    # 1. Create a collection.
    response = client.post(
        "/collections",
        json={
            "name": "documents",
            "dimension": 2,
            "distance_metric": "cosine",
        },
    )

    assert response.status_code == 201
    assert response.json()["name"] == "documents"

    # 2. Insert the first record.
    response = client.post(
        "/collections/documents/records",
        json={
            "id": "python",
            "vector": [1.0, 0.0],
            "metadata": {
                "topic": "programming",
            },
            "document": "Python programming language.",
        },
    )

    assert response.status_code == 201

    # 3. Insert the second record.
    response = client.post(
        "/collections/documents/records",
        json={
            "id": "travel",
            "vector": [0.0, 1.0],
            "metadata": {
                "topic": "travel",
            },
            "document": "Travel destinations.",
        },
    )

    assert response.status_code == 201

    # 4. List records.
    response = client.get(
        "/collections/documents/records"
    )

    assert response.status_code == 200
    assert len(response.json()) == 2

    # 5. Retrieve one record.
    response = client.get(
        "/collections/documents/records/python"
    )

    assert response.status_code == 200
    assert response.json()["document"] == (
        "Python programming language."
    )

    # 6. Update the record.
    response = client.put(
        "/collections/documents/records/python",
        json={
            "vector": [0.9, 0.1],
            "metadata": {
                "topic": "programming",
                "language": "python",
            },
            "document": "Updated Python document.",
        },
    )

    assert response.status_code == 200
    assert response.json()["metadata"]["language"] == "python"

    # 7. Search the collection.
    response = client.post(
        "/collections/documents/search",
        json={
            "query_vector": [1.0, 0.0],
            "metric": "cosine",
            "index_type": "brute_force",
            "k": 1,
        },
    )

    assert response.status_code == 200

    search_results = response.json()["results"]

    assert len(search_results) == 1
    assert search_results[0]["id"] == "python"

    # 8. Search with metadata filtering.
    response = client.post(
        "/collections/documents/search",
        json={
            "query_vector": [1.0, 0.0],
            "metric": "cosine",
            "index_type": "brute_force",
            "k": 5,
            "metadata_filter": {
                "topic": "travel",
            },
        },
    )

    assert response.status_code == 200

    filtered_results = response.json()["results"]

    assert len(filtered_results) == 1
    assert filtered_results[0]["id"] == "travel"

    # 9. Delete one record.
    response = client.delete(
        "/collections/documents/records/travel"
    )

    assert response.status_code == 204

    # 10. Verify the record was deleted.
    response = client.get(
        "/collections/documents/records"
    )

    assert response.status_code == 200
    assert len(response.json()) == 1

    # 11. Delete the collection.
    response = client.delete(
        "/collections/documents"
    )

    assert response.status_code == 204

    # 12. Verify the collection no longer exists.
    response = client.get(
        "/collections/documents"
    )

    assert response.status_code == 404


def test_openapi_contains_main_vectornest_routes() -> None:
    application = create_app()
    client = TestClient(application)

    response = client.get("/openapi.json")

    assert response.status_code == 200

    paths = response.json()["paths"]

    assert "/health" in paths
    assert "/collections" in paths
    assert "/collections/{name}" in paths
    assert "/collections/{collection_name}/records" in paths
    assert (
        "/collections/{collection_name}/records/{record_id}"
        in paths
    )
    assert "/collections/{collection_name}/search" in paths