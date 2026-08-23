"""Search result models returned by vector indexes."""

from dataclasses import dataclass

from vectornest.models.record import VectorRecord


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Represent one vector search match."""

    record: VectorRecord
    score: float