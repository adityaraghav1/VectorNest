import json
from collections.abc import Iterator

from fastapi.testclient import TestClient

from vectornest.api.app import create_app
from vectornest.api.dependencies import get_rag_service
from vectornest.services.rag import (
    RAGResponse,
    RAGSource,
)


class FakeRAGService:
    def answer(
        self,
        collection_name: str,
        question: str,
        **kwargs: object,
    ) -> RAGResponse:
        return RAGResponse(
            answer=(
                "Python is commonly used "
                "for machine learning."
            ),
            sources=[
                RAGSource(
                    id="python-guide:chunk:0",
                    document=(
                        "Python is a programming language "
                        "used for machine learning."
                    ),
                    score=0.95,
                    metadata={
                        "category": "technology",
                    },
                )
            ],
        )

    def stream_answer(
        self,
        collection_name: str,
        question: str,
        **kwargs: object,
    ) -> tuple[
        list[RAGSource],
        Iterator[str],
    ]:
        sources = [
            RAGSource(
                id="python-guide:chunk:0",
                document=(
                    "Python is a programming language "
                    "used for machine learning."
                ),
                score=0.95,
                metadata={
                    "category": "technology",
                },
            )
        ]

        answer_stream = iter(
            [
                "Python is ",
                "commonly used ",
                "for machine learning.",
            ]
        )

        return sources, answer_stream


class FakeEmptyRAGService:
    def answer(
        self,
        collection_name: str,
        question: str,
        **kwargs: object,
    ) -> RAGResponse:
        return RAGResponse(
            answer=(
                "I could not find relevant context "
                "in this collection."
            ),
            sources=[],
        )

    def stream_answer(
        self,
        collection_name: str,
        question: str,
        **kwargs: object,
    ) -> tuple[
        list[RAGSource],
        Iterator[str],
    ]:
        return (
            [],
            iter(
                [
                    (
                        "I could not find relevant context "
                        "in this collection."
                    )
                ]
            ),
        )


def create_client(
    rag_service: object | None = None,
) -> TestClient:
    application = create_app()

    service = (
        rag_service
        if rag_service is not None
        else FakeRAGService()
    )

    application.dependency_overrides[
        get_rag_service
    ] = lambda: service

    return TestClient(application)


def test_rag_returns_answer_and_sources() -> None:
    client = create_client()

    response = client.post(
        "/collections/knowledge/rag",
        json={
            "question": (
                "How is Python used?"
            ),
            "metric": "cosine",
            "index_type": "brute_force",
            "k": 4,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["answer"] == (
        "Python is commonly used "
        "for machine learning."
    )

    assert len(body["sources"]) == 1

    source = body["sources"][0]

    assert source["id"] == (
        "python-guide:chunk:0"
    )

    assert source["score"] == 0.95

    assert source["metadata"] == {
        "category": "technology",
    }


def test_rag_stream_returns_ndjson_events() -> None:
    client = create_client()

    response = client.post(
        "/collections/knowledge/rag/stream",
        json={
            "question": (
                "How is Python used?"
            ),
            "metric": "cosine",
            "index_type": "brute_force",
            "k": 4,
        },
    )

    assert response.status_code == 200

    assert (
        "application/x-ndjson"
        in response.headers["content-type"]
    )

    events = [
        json.loads(line)
        for line in response.text.splitlines()
        if line.strip()
    ]

    assert events[0]["type"] == "sources"

    assert len(
        events[0]["sources"]
    ) == 1

    assert (
        events[0]["sources"][0]["id"]
        == "python-guide:chunk:0"
    )

    assert events[1] == {
        "type": "token",
        "content": "Python is ",
    }

    assert events[2] == {
        "type": "token",
        "content": "commonly used ",
    }

    assert events[3] == {
        "type": "token",
        "content": "for machine learning.",
    }

    assert events[4] == {
        "type": "done",
    }


def test_rag_returns_no_context_response() -> None:
    client = create_client(
        FakeEmptyRAGService()
    )

    response = client.post(
        "/collections/knowledge/rag",
        json={
            "question": (
                "What is quantum computing?"
            ),
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["sources"] == []

    assert body["answer"] == (
        "I could not find relevant context "
        "in this collection."
    )


def test_rag_stream_handles_no_context() -> None:
    client = create_client(
        FakeEmptyRAGService()
    )

    response = client.post(
        "/collections/knowledge/rag/stream",
        json={
            "question": (
                "What is quantum computing?"
            ),
        },
    )

    assert response.status_code == 200

    events = [
        json.loads(line)
        for line in response.text.splitlines()
        if line.strip()
    ]

    assert events[0] == {
        "type": "sources",
        "sources": [],
    }

    assert events[1] == {
        "type": "token",
        "content": (
            "I could not find relevant context "
            "in this collection."
        ),
    }

    assert events[2] == {
        "type": "done",
    }


def test_rag_routes_appear_in_openapi() -> None:
    client = create_client()

    response = client.get(
        "/openapi.json"
    )

    assert response.status_code == 200

    paths = response.json()["paths"]

    assert (
        "/collections/{collection_name}/rag"
        in paths
    )

    assert (
        "/collections/{collection_name}/rag/stream"
        in paths
    )