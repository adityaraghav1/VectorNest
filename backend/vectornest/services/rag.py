"""Retrieval-augmented generation service for VectorNest."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from vectornest.core.exceptions import ValidationError
from vectornest.core.types import DistanceMetric, IndexType
from vectornest.generation.base import GenerationProvider
from vectornest.query.filters import MetadataFilter
from vectornest.services.semantic_search import SemanticSearchService


@dataclass(frozen=True, slots=True)
class RAGSource:
    """One retrieved source used to generate an answer."""

    id: str
    document: str
    score: float
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RAGResponse:
    """Generated RAG answer and the retrieved supporting sources."""

    answer: str
    sources: list[RAGSource]


class RAGService:
    """Retrieve relevant chunks and generate grounded answers."""

    def __init__(
        self,
        semantic_search_service: SemanticSearchService,
        generation_provider: GenerationProvider,
        minimum_score: float = 0.45,
    ) -> None:
        self._semantic_search_service = semantic_search_service
        self._generation_provider = generation_provider
        self._minimum_score = minimum_score

    def answer(
        self,
        collection_name: str,
        question: str,
        *,
        metric: DistanceMetric = DistanceMetric.COSINE,
        k: int = 4,
        metadata_filter: MetadataFilter | None = None,
        index_type: IndexType = IndexType.BRUTE_FORCE,
    ) -> RAGResponse:
        """Generate a grounded answer from retrieved context."""

        normalized_question = self._validate_question(question)

        sources = self._retrieve_sources(
            collection_name,
            normalized_question,
            metric=metric,
            k=k,
            metadata_filter=metadata_filter,
            index_type=index_type,
        )

        if not sources:
            return RAGResponse(
                answer=(
                    "I could not find relevant context "
                    "in this collection."
                ),
                sources=[],
            )

        prompt = self._create_rag_prompt(
            normalized_question,
            sources,
        )

        answer = self._generation_provider.generate(prompt)

        return RAGResponse(
            answer=answer,
            sources=sources,
        )

    def stream_answer(
        self,
        collection_name: str,
        question: str,
        *,
        metric: DistanceMetric = DistanceMetric.COSINE,
        k: int = 4,
        metadata_filter: MetadataFilter | None = None,
        index_type: IndexType = IndexType.BRUTE_FORCE,
    ) -> tuple[
        list[RAGSource],
        Iterator[str],
    ]:
        """Stream generated answer chunks from the LLM."""

        normalized_question = self._validate_question(question)

        sources = self._retrieve_sources(
            collection_name,
            normalized_question,
            metric=metric,
            k=k,
            metadata_filter=metadata_filter,
            index_type=index_type,
        )

        if not sources:
            return (
                [],
                iter(
                    [
                        (
                            "I could not find relevant "
                            "context in this collection."
                        )
                    ]
                ),
            )

        prompt = self._create_rag_prompt(
            normalized_question,
            sources,
        )

        return (
            sources,
            self._generation_provider.stream(prompt),
        )

    def _retrieve_sources(
        self,
        collection_name: str,
        question: str,
        *,
        metric: DistanceMetric,
        k: int,
        metadata_filter: MetadataFilter | None,
        index_type: IndexType,
    ) -> list[RAGSource]:
        search_results = self._semantic_search_service.search(
            collection_name,
            question,
            metric=metric,
            k=k,
            metadata_filter=metadata_filter,
            index_type=index_type,
        )

        sources: list[RAGSource] = []

        for result in search_results:
            if not result.record.document:
                continue

            if (
                metric == DistanceMetric.COSINE
                and result.score < self._minimum_score
            ):
                continue

            sources.append(
                RAGSource(
                    id=result.record.id,
                    document=result.record.document,
                    score=result.score,
                    metadata=result.record.metadata,
                )
            )

        return sources

    def _create_rag_prompt(
        self,
        question: str,
        sources: list[RAGSource],
    ) -> str:
        context = self._build_context(sources)

        return self._build_prompt(
            question=question,
            context=context,
        )

    @staticmethod
    def _validate_question(
        question: str,
    ) -> str:
        if not isinstance(question, str):
            raise ValidationError(
                "RAG question must be a string."
            )

        normalized_question = question.strip()

        if not normalized_question:
            raise ValidationError(
                "RAG question cannot be empty."
            )

        return normalized_question

    @staticmethod
    def _build_context(
        sources: list[RAGSource],
    ) -> str:
        context_parts: list[str] = []

        for index, source in enumerate(
            sources,
            start=1,
        ):
            context_parts.append(
                f"[Source {index} | "
                f"{source.id}]\n"
                f"{source.document}"
            )

        return "\n\n".join(context_parts)

    @staticmethod
    def _build_prompt(
        question: str,
        context: str,
    ) -> str:
        return (
            "You are the RAG assistant for VectorNest.\n\n"
            "Answer the user's question using only the "
            "provided context.\n"
            "If the answer is not supported by the context, "
            "say that the available documents do not contain "
            "enough information.\n"
            "Do not invent facts.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )