"""Document ingestion service for VectorNest."""

from typing import Any
from uuid import uuid4

from vectornest.core.exceptions import DimensionMismatchError, ValidationError
from vectornest.embeddings.base import EmbeddingProvider
from vectornest.ingestion.chunking import TextChunker
from vectornest.models.record import VectorRecord
from vectornest.storage.base import StorageBackend


class DocumentIngestionService:
    """Chunk, embed and store documents in VectorNest."""

    def __init__(
        self,
        storage: StorageBackend,
        embedding_provider: EmbeddingProvider,
        chunker: TextChunker | None = None,
    ) -> None:
        self._storage = storage
        self._embedding_provider = embedding_provider
        self._chunker = chunker or TextChunker()

    def ingest(
        self,
        collection_name: str,
        document: str,
        *,
        document_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[VectorRecord]:
        """Chunk, embed and store one document."""
        if not isinstance(document, str) or not document.strip():
            raise ValidationError(
                "Document must be a non-empty string."
            )

        collection = self._storage.get_collection(
            collection_name
        )

        if (
            collection.dimension
            != self._embedding_provider.dimension
        ):
            raise DimensionMismatchError(
                "Embedding provider dimension does not match "
                "collection dimension."
            )

        chunks = self._chunker.chunk(document)

        if not chunks:
            return []

        base_document_id = (
            document_id.strip()
            if document_id is not None
            and document_id.strip()
            else str(uuid4())
        )

        base_metadata = dict(metadata or {})

        records: list[VectorRecord] = []

        for chunk in chunks:
            embedding = self._embedding_provider.embed_text(
                chunk.text
            )

            chunk_metadata = dict(base_metadata)
            chunk_metadata.update(
                {
                    "document_id": base_document_id,
                    "chunk_index": chunk.index,
                    "start_word": chunk.start_word,
                    "end_word": chunk.end_word,
                }
            )

            record = VectorRecord(
                id=(
                    f"{base_document_id}"
                    f":chunk:{chunk.index}"
                ),
                vector=embedding,
                metadata=chunk_metadata,
                document=chunk.text,
            )

            self._storage.insert_record(
                collection_name,
                record,
            )

            records.append(record)

        return records