"""Benchmark VectorNest index implementations."""

from collections.abc import Iterable
from time import perf_counter

from vectornest.benchmarking.results import BenchmarkResult
from vectornest.core.exceptions import ValidationError
from vectornest.core.types import DistanceMetric, IndexType
from vectornest.indexes.brute_force import BruteForceIndex
from vectornest.indexes.factory import create_index
from vectornest.indexes.results import SearchResult
from vectornest.metrics.functions import VectorInput
from vectornest.models.record import VectorRecord


class BenchmarkService:
    """Measure index latency and retrieval quality."""

    def benchmark(
        self,
        records: Iterable[VectorRecord],
        queries: Iterable[VectorInput],
        dimension: int,
        metric: DistanceMetric,
        index_type: IndexType,
        k: int = 10,
    ) -> BenchmarkResult:
        """Benchmark one index against brute-force ground truth."""

        if dimension <= 0:
            raise ValidationError(
                "Benchmark dimension must be greater than zero."
            )

        if k <= 0:
            raise ValidationError(
                "Benchmark k must be greater than zero."
            )

        record_list = list(records)
        query_list = list(queries)

        if not query_list:
            raise ValidationError(
                "Benchmark requires at least one query."
            )

        reference = BruteForceIndex(
            dimension=dimension,
            metric=metric,
        )

        reference.add_many(record_list)

        build_start = perf_counter()

        candidate_index = create_index(
            dimension=dimension,
            metric=metric,
            index_type=index_type,
        )

        candidate_index.add_many(record_list)

        build_time_ms = (
            perf_counter() - build_start
        ) * 1000.0

        total_query_time = 0.0
        total_recall = 0.0

        for query in query_list:
            ground_truth = reference.search(
                query,
                limit=k,
            )

            query_start = perf_counter()

            candidate_results = candidate_index.search(
                query,
                limit=k,
            )

            total_query_time += (
                perf_counter() - query_start
            ) * 1000.0

            total_recall += self._recall_at_k(
                ground_truth,
                candidate_results,
            )

        return BenchmarkResult(
            index_type=index_type,
            record_count=len(record_list),
            query_count=len(query_list),
            k=k,
            build_time_ms=build_time_ms,
            average_query_time_ms=(
                total_query_time
                / len(query_list)
            ),
            recall_at_k=(
                total_recall
                / len(query_list)
            ),
        )

    @staticmethod
    def _recall_at_k(
        ground_truth: list[SearchResult],
        candidate_results: list[SearchResult],
    ) -> float:
        """Return overlap with exact brute-force neighbours."""

        if not ground_truth:
            return 1.0

        expected_ids = {
            result.record.id
            for result in ground_truth
        }

        candidate_ids = {
            result.record.id
            for result in candidate_results
        }

        matches = len(
            expected_ids & candidate_ids
        )

        return matches / len(expected_ids)