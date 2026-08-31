"""Benchmark result models for VectorNest indexes."""

from dataclasses import dataclass

from vectornest.core.types import IndexType


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Summarize one index benchmark run."""

    index_type: IndexType
    record_count: int
    query_count: int
    k: int
    build_time_ms: float
    average_query_time_ms: float
    recall_at_k: float