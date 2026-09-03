"""Retrieval-augmented generation service for VectorNest."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from vectornest.core.exceptions import ValidationError
from vectornest.core.types import DistanceMetric, IndexType
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
        llm_client: Any,
        model: str,
        minimum_score: float = 0.45,
    ) -> None:
        if not model.strip():
            raise ValidationError(
                "RAG model name cannot be empty."
            )

        self._semantic_search_service = (
            semantic_search_service
        )

        self._llm_client = llm_client
        self._model = model.strip()
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
        normalized_question = (
            self._validate_question(
                question
            )
        )

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

        response = self._llm_client.chat(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        answer = self._extract_answer(
            response
        )

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
        """Stream generated answer tokens from the LLM."""

        normalized_question = (
            self._validate_question(
                question
            )
        )

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

        response = self._llm_client.chat(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=True,
        )

        return (
            sources,
            self._iter_stream_content(
                response
            ),
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
        search_results = (
            self._semantic_search_service.search(
                collection_name,
                question,
                metric=metric,
                k=k,
                metadata_filter=metadata_filter,
                index_type=index_type,
            )
        )

        return [
            RAGSource(
                id=result.record.id,
                document=(
                    result.record.document
                    or ""
                ),
                score=result.score,
                metadata=(
                    result.record.metadata
                ),
            )
            for result in search_results
            if (
                result.record.document
                and result.score
                >= self._minimum_score
            )
        ]

    def _create_rag_prompt(
        self,
        question: str,
        sources: list[RAGSource],
    ) -> str:
        context = self._build_context(
            sources
        )

        return self._build_prompt(
            question=question,
            context=context,
        )

    @staticmethod
    def _validate_question(
        question: str,
    ) -> str:
        if not isinstance(
            question,
            str,
        ):
            raise ValidationError(
                "RAG question must be a string."
            )

        normalized_question = (
            question.strip()
        )

        if not normalized_question:
            raise ValidationError(
                "RAG question cannot be empty."
            )

        return normalized_question

    @staticmethod
    def _iter_stream_content(
        response: Any,
    ) -> Iterator[str]:
        """Yield text chunks from an Ollama stream."""

        for chunk in response:
            content = (
                RAGService._extract_stream_chunk(
                    chunk
                )
            )

            if content:
                yield content

    @staticmethod
    def _extract_stream_chunk(
        chunk: Any,
    ) -> str:
        if isinstance(
            chunk,
            dict,
        ):
            message = chunk.get(
                "message"
            )

            if isinstance(
                message,
                dict,
            ):
                content = message.get(
                    "content"
                )

                if isinstance(
                    content,
                    str,
                ):
                    return content

        message = getattr(
            chunk,
            "message",
            None,
        )

        content = getattr(
            message,
            "content",
            None,
        )

        if isinstance(
            content,
            str,
        ):
            return content

        return ""

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

        return "\n\n".join(
            context_parts
        )

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

    @staticmethod
    def _extract_answer(
        response: Any,
    ) -> str:
        if isinstance(
            response,
            dict,
        ):
            message = response.get(
                "message"
            )

            if isinstance(
                message,
                dict,
            ):
                content = message.get(
                    "content"
                )

                if isinstance(
                    content,
                    str,
                ):
                    answer = (
                        content.strip()
                    )

                    if answer:
                        return answer

        message = getattr(
            response,
            "message",
            None,
        )

        content = getattr(
            message,
            "content",
            None,
        )

        if isinstance(
            content,
            str,
        ):
            answer = content.strip()

            if answer:
                return answer

        raise ValidationError(
            "Ollama response does not contain "
            "a generated answer."
        )